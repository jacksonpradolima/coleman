"""Noneditable base-wheel math, prediction, constrained solve and real recovery."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

import pytest

from tests.installed_support import build_install_offline_wheel, seed_locked_dependencies
from tests.learning.test_imports import BLOCKED, COLD_SCRIPT

ROOT = Path(__file__).parents[2]


def invoke(arguments, environment, *, cwd):
    result = subprocess.run(
        arguments, cwd=cwd, env=environment, capture_output=True, text=True, timeout=240, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def installed_continuous(tmp_path_factory):
    owned = tmp_path_factory.mktemp("installed-continuous-prediction")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_NO_PROGRESS="1", UV_LINK_MODE="copy")
    virtual = owned / "base"
    supplied = environment.get("COLEMAN_TEST_CONTINUOUS_WHEEL")
    if supplied:
        wheel = Path(supplied).resolve(strict=True)
        assert wheel.suffix == ".whl"
        invoke([uv, "venv", str(virtual), "--python", sys.executable], environment, cwd=owned)
        seed_locked_dependencies(uv, virtual, environment, extras=())
        invoke(
            [
                uv,
                "pip",
                "install",
                "--offline",
                "--no-index",
                "--no-deps",
                "--python",
                str(virtual / "bin/python"),
                str(wheel),
            ],
            environment,
            cwd=owned,
        )
    else:
        wheel = build_install_offline_wheel(uv, owned / "dist", virtual, environment)
    python = virtual / "bin/python"
    invoke([uv, "pip", "check", "--python", str(python)], environment, cwd=owned)
    outside = owned / "outside"
    outside.mkdir()
    assert not outside.is_relative_to(ROOT)
    copied = {}
    for relative in (
        "examples/continuous/model.py",
        "examples/continuous/flow.py",
        "examples/continuous/allocation.py",
        "tests/continuous_integration/installed_driver.py",
        "tests/continuous_integration/recovery_driver.py",
    ):
        source = ROOT / relative
        shutil.copyfile(source, outside / source.name)
        copied[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    sdk = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("coleman/") and not name.endswith("/"):
                content = archive.read(name)
                assert content == (ROOT / name).read_bytes(), name
                sdk[name] = hashlib.sha256(content).hexdigest()
    evidence = {
        "wheel": str(wheel),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "sdk": sdk,
        "copied_public_sources": copied,
        "python": str(python),
        "outside": str(outside),
        "supplied_prebuilt": bool(supplied),
    }
    (owned / "installation.json").write_text(json.dumps(evidence, sort_keys=True) + "\n")
    return {"owned": owned, "outside": outside, "python": python, "environment": environment, "evidence": evidence}


def test_installed_actual_prediction_updates_pending_resume_and_separate_constrained_solve(installed_continuous):
    installed = installed_continuous
    result = json.loads(
        invoke(
            [str(installed["python"]), "-I", "-B", str(installed["outside"] / "installed_driver.py")],
            installed["environment"],
            cwd=installed["outside"],
        )
    )
    assert (result["real_updates_before_resume"], result["real_updates_after_resume"]) == (1, 2)
    assert result["complete_continuation_equal"]
    assert result["coupled_optimum"] == [0.5, 1.25]
    assert result["composed"]["forecast"]["samples"] == 4
    assert result["composed"]["inference"] == "insufficient_evidence"
    (installed["owned"] / "installed-prediction-solve-result.json").write_text(
        json.dumps(result, sort_keys=True) + "\n"
    )


def test_pure_forecasting_and_optimization_cold_imports_need_no_infrastructure(tmp_path):
    sites = sorted({sysconfig.get_paths()[name] for name in ("purelib", "platlib")})
    modules = [
        "coleman.prediction",
        "coleman.prediction.contracts",
        "coleman.prediction.ridge",
        "coleman.optimization.continuous",
        "coleman.optimization.continuous_contracts",
    ]
    result = json.loads(
        invoke(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                COLD_SCRIPT,
                json.dumps([str(ROOT), *sites]),
                json.dumps(modules),
                str(ROOT),
                json.dumps(BLOCKED + ("coleman.replay",)),
            ],
            dict(os.environ),
            cwd=tmp_path,
        )
    )
    assert result["forbidden"] == []


def test_installed_pure_boundary_and_base_distribution_inventory(installed_continuous):
    installed = installed_continuous
    python, environment, outside = installed["python"], installed["environment"], installed["outside"]
    sites = json.loads(
        invoke(
            [
                str(python),
                "-I",
                "-B",
                "-c",
                "import sysconfig,json;"
                "print(json.dumps(sorted({sysconfig.get_paths()[key] for key in ('purelib','platlib')})))",
            ],
            environment,
            cwd=outside,
        )
    )
    modules = ["coleman.prediction.ridge", "coleman.optimization.continuous"]
    result = json.loads(
        invoke(
            [
                str(python),
                "-I",
                "-S",
                "-B",
                "-c",
                COLD_SCRIPT,
                json.dumps(sites),
                json.dumps(modules),
                str(python.parent.parent),
                json.dumps(BLOCKED + ("coleman.replay",)),
            ],
            environment,
            cwd=outside,
        )
    )
    assert result["forbidden"] == []
    inventory = json.loads(
        invoke(
            [
                str(python),
                "-I",
                "-B",
                "-c",
                "import importlib.metadata,json;"
                "print(json.dumps(sorted(d.metadata['Name'].lower() for d in importlib.metadata.distributions())))",
            ],
            environment,
            cwd=outside,
        )
    )
    assert not {"numpy", "scipy", "mcp", "duckdb", "polars", "pyarrow"} & set(inventory)


def recovery(installed, operation, root, *flags):
    return json.loads(
        invoke(
            [
                str(installed["python"]),
                "-I",
                "-B",
                str(installed["outside"] / "recovery_driver.py"),
                operation,
                str(root),
                *flags,
            ],
            installed["environment"],
            cwd=installed["outside"],
        )
    )


def test_installed_one_ack_restores_actual32_predictors_and32_hedge_learners(installed_continuous):
    installed = installed_continuous
    case = installed["outside"] / "actual64"
    case.mkdir()
    root = case / "store"
    flags = ("--children", "32", "--with-hedge")
    initial = recovery(installed, "initialize", root, *flags)
    assert (
        initial["children"],
        initial["components"],
        initial["samples"],
        initial["pending"],
        initial["hedge_updates"],
    ) == (
        32,
        64,
        32,
        32,
        32,
    )
    final = recovery(installed, "transition", root, *flags)
    assert (final["generation"], final["samples"], final["pending"], final["hedge_updates"]) == (1, 64, 0, 64)
    assert final["status"] == "new"
    assert all(state["host_state"]["ridge"]["payload"]["sample_count"] == 2 for state in final["states"])
    assert all(state["payload"]["update_count"] == 2 for state in final["hedge_states"])
    repeated = recovery(installed, "transition", root, *flags)
    assert repeated == {**final, "status": "retained"}
    (installed["owned"] / "actual64-installed-recovery.json").write_text(json.dumps(final, sort_keys=True) + "\n")


def test_installed_interrupted_sqlite_commit_never_acknowledges_partial_predictor_state(installed_continuous):
    installed = installed_continuous
    case = installed["outside"] / "interrupted"
    case.mkdir()
    root = case / "store"
    initial = recovery(installed, "initialize", root)
    failed = subprocess.run(
        [
            str(installed["python"]),
            "-I",
            "-B",
            str(installed["outside"] / "recovery_driver.py"),
            "transition",
            str(root),
            "--fault",
            "before-ack",
        ],
        cwd=installed["outside"],
        env=installed["environment"],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert failed.returncode == 87, failed.stdout + failed.stderr
    assert (case / "fault.jsonl").read_text().strip() == "after_head_update"
    assert recovery(installed, "inspect", root) == initial
    final = recovery(installed, "transition", root)
    assert (final["generation"], final["samples"], final["pending"]) == (1, 1, 0)
    repeated = recovery(installed, "transition", root)
    assert repeated == {**final, "status": "retained"}
