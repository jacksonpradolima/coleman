"""Real minimal/all-extra wheels and copied flows outside the checkout."""

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tests.installed_support import build_install_offline_wheel, seed_locked_dependencies

ROOT = Path(__file__).parents[2]
EXTRAS = tuple(sorted(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]))


def command(arguments, *, cwd, environment):
    result = subprocess.run(
        arguments, cwd=cwd, env=environment, capture_output=True, text=True, timeout=240, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    owned = tmp_path_factory.mktemp("operational-installed")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(UV_LINK_MODE="copy", UV_NO_PROGRESS="1")
    virtual = owned / "minimal"
    supplied = environment.get("COLEMAN_TEST_OPERATIONAL_WHEEL")
    if supplied:
        wheel = Path(supplied).resolve(strict=True)
        assert wheel.suffix == ".whl"
        command([uv, "venv", str(virtual), "--python", sys.executable], cwd=ROOT, environment=environment)
        seed_locked_dependencies(uv, virtual, environment)
        command(
            [
                uv,
                "pip",
                "install",
                "--offline",
                "--no-index",
                "--no-deps",
                "--no-build",
                "--python",
                str(virtual / "bin/python"),
                str(wheel),
            ],
            cwd=owned,
            environment=environment,
        )
    else:
        wheel = build_install_offline_wheel(uv, owned / "dist", virtual, environment)
    outside = owned / "outside"
    outside.mkdir()
    assert not outside.is_relative_to(ROOT)
    for filename in ("ci.py", "measure.py"):
        shutil.copyfile(ROOT / "examples/operational" / filename, outside / filename)
    shutil.copyfile(ROOT / "tests/operational/cold_import.py", outside / "cold_import.py")
    return owned, uv, wheel, virtual, outside, environment


def invoke(python, filename, arguments, outside, environment):
    return command([str(python), "-I", "-B", str(outside / filename), *arguments], cwd=outside, environment=environment)


def test_actual_minimal_wheel_noop_ci_and_cold_boundary_without_optional_sdk(installed):
    _, _, _, virtual, outside, environment = installed
    receipt = json.loads(invoke(virtual / "bin/python", "ci.py", ["--mode", "noop"], outside, environment))
    assert receipt["workload"] == {"schema_version": 1, "experts": 2, "panels": 2, "trajectories": 1}
    assert len(receipt["snapshots"]) == 2
    weights = receipt["final_states"][0]["payload"]["weights"]
    assert [w["value"] for w in weights] == pytest.approx([0.5, 0.5], abs=1e-15)
    result = json.loads(invoke(virtual / "bin/python", "cold_import.py", [], outside, environment))
    assert result["new_threads"] == []
    assert result["optional_imports"] == []
    assert "opentelemetry-sdk" not in result["installed"]


def test_actual_all_extra_locked_wheel_optional_import_invariance_and_local_sdk(installed):
    owned, uv, wheel, _, outside, inherited = installed
    virtual = owned / "all-extras"
    environment = dict(inherited)
    # This intentionally acquires the exact remaining locked optional wheels;
    # CI's selected runtime profile is not a false all-extra installation claim.
    environment.pop("UV_OFFLINE", None)
    environment.pop("UV_NO_INDEX", None)
    command([uv, "venv", str(virtual), "--python", sys.executable], cwd=ROOT, environment=environment)
    command(
        [
            uv,
            "sync",
            "--frozen",
            "--all-extras",
            "--no-default-groups",
            "--no-dev",
            "--no-install-project",
            "--no-build",
            "--inexact",
            "--python",
            str(virtual / "bin/python"),
        ],
        cwd=ROOT,
        environment={**environment, "UV_PROJECT_ENVIRONMENT": str(virtual)},
    )
    python = virtual / "bin/python"
    command(
        [
            uv,
            "pip",
            "install",
            "--offline",
            "--no-index",
            "--no-deps",
            "--no-build",
            "--python",
            str(python),
            str(wheel) + "[" + ",".join(EXTRAS) + "]",
        ],
        cwd=outside,
        environment=environment,
    )
    command([uv, "pip", "check", "--python", str(python)], cwd=outside, environment=environment)
    result = json.loads(invoke(python, "cold_import.py", [], outside, environment))
    assert result["optional_imports"] == []
    assert {
        "mcp",
        "opentelemetry-sdk",
        "clickhouse-connect",
        "duckdb",
        "polars",
        "pyarrow",
        "marimo",
        "zensical",
    } <= set(result["installed"])
    receipt = json.loads(invoke(python, "ci.py", ["--mode", "memory"], outside, environment))
    assert "coleman.lifecycle.concentration" in receipt["sdk_metric_names"]
    report_path = invoke(
        python, "measure.py", ["--quick", "--output", str(owned / "measured")], outside, environment
    ).strip()
    report = json.loads(Path(report_path).read_text())
    assert report["complete"] is True
    assert len(report["measurements"]) == 3
    assert len({row["mathematical_output"] for row in report["measurements"]}) == 1
    instrumentation = report["instrumentation_runtime"]
    assert instrumentation["selected_extras"] == ["telemetry"]
    versions = {d["name"]: d["version"] for d in instrumentation["distributions"]}
    assert versions["opentelemetry-sdk"] == versions["opentelemetry-api"] == "1.45.2"
    assert not {"numpy", "pylint", "mcp", "clickhouse-connect"} & versions.keys()
