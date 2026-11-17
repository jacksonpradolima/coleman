"""Real minimal/all-extra wheels and copied flows outside the checkout."""

import hashlib
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


def command(arguments, *, cwd, environment, timeout=240):
    result = subprocess.run(
        arguments, cwd=cwd, env=environment, capture_output=True, text=True, timeout=timeout, check=False
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
    for filename in ("ci.py", "measure.py", "bootstrap.py", "tasks.py", "task_records.py", "task_inputs.py"):
        shutil.copyfile(ROOT / "examples/operational" / filename, outside / filename)
    shutil.copyfile(ROOT / "tests/operational/cold_import.py", outside / "cold_import.py")
    return owned, uv, wheel, virtual, outside, environment


@pytest.fixture(scope="module")
def installed_cli(installed):
    """Keep the real CLI closure separate from the genuine minimal API image."""
    owned, uv, wheel, _, outside, environment = installed
    virtual = owned / "cli"
    command([uv, "venv", str(virtual), "--python", sys.executable], cwd=outside, environment=environment)
    seed_locked_dependencies(uv, virtual, environment, extras=("cli",))
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
            str(wheel) + "[cli]",
        ],
        cwd=outside,
        environment=environment,
    )
    command([uv, "pip", "check", "--python", str(virtual / "bin/python")], cwd=outside, environment=environment)
    return virtual


def invoke(python, filename, arguments, outside, environment, timeout=240):
    return command(
        [str(python), "-I", "-B", str(outside / filename), *arguments],
        cwd=outside,
        environment=environment,
        timeout=timeout,
    )


def test_actual_minimal_wheel_noop_ci_and_cold_boundary_without_optional_sdk(installed):
    _, _, _, virtual, outside, environment = installed
    script = (
        "import importlib.metadata as metadata,json,runpy,sys; "
        "namespace=runpy.run_path(sys.argv[1]); "
        "assert 'click' not in sys.modules; "
        "assert not any(name=='click' or name.startswith('opentelemetry') "
        "for name in (item.metadata['Name'].lower() for item in metadata.distributions())); "
        "receipt=namespace['run'](namespace['Workload'](schema_version=1),'noop'); "
        "assert 'click' not in sys.modules; print(receipt.to_json())"
    )
    receipt = json.loads(
        command(
            [str(virtual / "bin/python"), "-I", "-B", "-c", script, str(outside / "ci.py")],
            cwd=outside,
            environment=environment,
        )
    )
    assert receipt["workload"] == {"schema_version": 1, "experts": 2, "panels": 2, "trajectories": 1}
    assert len(receipt["snapshots"]) == 2
    weights = receipt["final_states"][0]["payload"]["weights"]
    assert [w["value"] for w in weights] == pytest.approx([0.5, 0.5], abs=1e-15)
    result = json.loads(invoke(virtual / "bin/python", "cold_import.py", [], outside, environment))
    assert result["new_threads"] == []
    assert result["optional_imports"] == []
    assert "opentelemetry-sdk" not in result["installed"]
    assert "click" not in result["installed"]


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
        "click",
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
    driver = report["driver_runtime"]
    assert driver["selected_extras"] == ["cli"]
    assert {d["name"]: d["version"] for d in driver["distributions"]}["click"] == "8.1.8"
    versions = {d["name"]: d["version"] for d in instrumentation["distributions"]}
    assert versions["opentelemetry-sdk"] == versions["opentelemetry-api"] == "1.45.2"
    assert not {"numpy", "pylint", "mcp", "clickhouse-connect"} & versions.keys()


def test_actual_empty_private_dependency_cache_and_same_offline_warm_cache(installed, installed_cli):
    owned, uv, wheel, virtual, outside, environment = installed
    report = json.loads(
        invoke(
            installed_cli / "bin/python",
            "bootstrap.py",
            [
                "--project",
                str(ROOT),
                "--wheel",
                str(wheel),
                "--uv",
                uv,
                "--output",
                str(owned / "cache-observations"),
            ],
            outside,
            environment,
        )
    )
    assert report["complete"] is True
    cold, warm = report["cells"]
    assert cold["cache_before_files"] == cold["cache_before_bytes"] == 0
    assert cold["cache_after_files"] == warm["cache_before_files"] > 0
    assert cold["cache_after_bytes"] == warm["cache_before_bytes"] > 0
    assert cold["runtime"] == warm["runtime"]
    assert report["driver_runtime"]["selected_extras"] == ["cli"]
    assert {d["name"]: d["version"] for d in report["driver_runtime"]["distributions"]}["click"] == "8.1.8"
    for mode in ("empty-online", "warm-offline"):
        python = owned / "cache-observations" / mode / "bin/python"
        result = json.loads(invoke(python, "cold_import.py", [], outside, environment))
        assert result["optional_imports"] == result["new_threads"] == []


@pytest.fixture(scope="module")
def installed_task_provider(installed):
    owned, uv, wheel, _, outside, environment = installed
    provider_wheels = []
    for name in ("provider", "support"):
        target = owned / (name + "-wheels")
        command(
            [
                uv,
                "build",
                "--offline",
                "--no-build-isolation",
                "--wheel",
                "--python",
                sys.executable,
                "--out-dir",
                str(target),
            ],
            cwd=ROOT / "examples/typed_executor" / name,
            environment=environment,
            timeout=180,
        )
        built = tuple(target.glob("*.whl"))
        assert len(built) == 1
        provider_wheels.extend(built)
    virtual = owned / "task-consumer"
    command([uv, "venv", str(virtual), "--python", sys.executable], cwd=outside, environment=environment)
    seed_locked_dependencies(uv, virtual, environment, extras=("cli",))
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
            str(wheel) + "[cli]",
            *(str(path) for path in provider_wheels),
        ],
        cwd=outside,
        environment=environment,
    )
    command([uv, "pip", "check", "--python", str(virtual / "bin/python")], cwd=outside, environment=environment)
    shutil.copyfile(ROOT / "examples/typed_executor/ci.py", outside / "typed-ci.py")
    shutil.copyfile(
        ROOT / "examples/typed_executor/provider/src/coleman_typed_hedge/recipe.json", outside / "recipe.json"
    )
    invoke(
        virtual / "bin/python",
        "typed-ci.py",
        ["approve", str(outside / "recipe.json"), str(outside / "profile.json")],
        outside,
        environment,
        timeout=180,
    )
    return owned, virtual, outside, environment


def test_actual_queued_tasks_and_naturally_larger_artifacts_keep_complete_state(installed_task_provider):
    owned, virtual, outside, environment = installed_task_provider
    report = json.loads(
        invoke(
            virtual / "bin/python",
            "tasks.py",
            ["--profile", str(outside / "profile.json"), "--output", str(owned / "task-load")],
            outside,
            environment,
            timeout=180,
        )
    )
    assert report["planned_updates"] == 132
    assert report["driver_runtime"]["selected_extras"] == ["cli"]
    assert {d["name"]: d["version"] for d in report["driver_runtime"]["distributions"]}["click"] == "8.1.8"
    assert report["helper_sources"] == [
        [name, hashlib.sha256((outside / name).read_bytes()).hexdigest()]
        for name in ("task_inputs.py", "task_records.py")
    ]
    assert report["outcome"] in {"completed", "limited"}
    assert len(report["cells"]) == 4
    assert sum(cell["planned_updates"] for cell in report["cells"]) == 132
    jobs = [row for cell in report["cells"] for row in cell["jobs"]]
    assert jobs  # The actual acceptance flow must have exercised real tasks.
    assert len(jobs) <= 8
    assert report["acknowledged_update_lower_bound"] == sum(row["accepted_updates"] for row in jobs) <= 132
    assert all(not row["worker_active"] for row in jobs)
    assert all(row["job"]["state"] in {"completed", "cancelled", "failed"} for row in jobs)
    complete = all(cell["complete"] for cell in report["cells"])
    assert report["complete"] is complete
    artifact = owned / "task-load" / ("complete.json" if complete else "limited.json")
    assert json.loads(artifact.read_text()) == report
    if not complete:
        assert not (owned / "task-load/complete.json").exists()
        assert report["outcome"] == "limited"
    for cell in report["cells"]:
        assert (owned / "task-load" / f"cell-{cell['batch']}-{cell['size']}.json").is_file()
        if cell["disposition"] == "not_started":
            assert cell["jobs"] == []
            assert cell["observation_status"] == "unavailable"
            assert cell["input_bytes"] is cell["elapsed_seconds"] is cell["host_cpu_seconds"] is None
        else:
            assert len(cell["jobs"]) == cell["batch"]
            assert cell["worker_peak_rss_mib"] is None
            assert cell["worker_resource_status"] == "unavailable"
    assert any(cell["batch"] == 3 and (cell["max_queued_views_per_sweep"] or 0) > 0 for cell in report["cells"])
    for row in jobs:
        assert row["planned_updates"] >= row["accepted_updates"]
        if row["job"]["state"] == "completed":
            assert row["complete_state_status"] == "verified"
            assert row["accepted_updates"] == row["planned_updates"]
        else:
            assert row["result_bytes"] is row["mathematical_digest"] is None
            assert row["complete_state_status"] == "not_applicable"
    if report["cells"][2]["input_bytes"] is not None:
        assert report["cells"][2]["input_bytes"] > report["cells"][0]["input_bytes"]
    small, larger = report["cells"][0]["jobs"], report["cells"][2]["jobs"]
    if small and larger and small[0]["job"]["state"] == larger[0]["job"]["state"] == "completed":
        assert larger[0]["artifact_bytes"] > small[0]["artifact_bytes"]
    assert {item["name"] for item in report["implementation"]["distributions"]} >= {
        "coleman",
        "coleman-typed-hedge-reference",
        "coleman-typed-hedge-support",
    }


@pytest.mark.parametrize("filename", ("ci.py", "measure.py", "bootstrap.py", "tasks.py"))
def test_actual_installed_cli_help_and_malformed_flags_before_work(installed, installed_cli, filename):
    """Use actual Click from the noneditable optional image outside checkout."""
    _, _, _, _, outside, environment = installed
    python = str(installed_cli / "bin/python")
    help_output = invoke(python, filename, ["--help"], outside, environment)
    assert "Usage:" in help_output and "--help" in help_output
    result = subprocess.run(
        [python, "-I", "-B", str(outside / filename), "--unknown-operational-flag"],
        cwd=outside,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 2
    assert "No such option" in result.stderr
    assert "Traceback" not in result.stderr


def test_actual_installed_cli_noop_and_typed_child_use_only_selected_tooling(installed, installed_cli):
    """Run the genuine learner through Click without the telemetry SDK."""
    _, _, _, _, outside, environment = installed
    python = installed_cli / "bin/python"
    boundary = json.loads(invoke(python, "cold_import.py", [], outside, environment))
    assert "click" in boundary["installed"]
    assert boundary["optional_imports"] == boundary["new_threads"] == []
    receipt = json.loads(invoke(python, "ci.py", ["--mode", "noop"], outside, environment))
    assert len(receipt["snapshots"]) == 2
    assert receipt["final_states"][0]["payload"]["update_count"] == 2
    child = json.loads(invoke(python, "measure.py", ["--child", "noop", "2", "2", "1"], outside, environment))
    assert child["accepted_updates"] == 2 and child["mode"] == "noop"
    assert child["snapshot_bytes"] > 0
