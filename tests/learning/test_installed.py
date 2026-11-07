"""Actual base-wheel learning and typed-host exchange outside the checkout.

The optional CI profile only checks retained native imports. A single supplied
noneditable wheel can qualify both profiles; no build runs during collection.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.installed_support import build_install_offline_wheel, seed_locked_dependencies
from tests.learning.test_imports import BLOCKED, COLD_SCRIPT

ROOT = Path(__file__).parents[2]


def _command(arguments, environment, *, cwd=ROOT):
    result = subprocess.run(
        arguments, cwd=cwd, env=environment, capture_output=True, text=True, timeout=240, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _install_existing_wheel(uv, wheel, virtual, environment, *, extras=()):
    _command([uv, "venv", str(virtual), "--python", sys.executable], environment)
    seed_locked_dependencies(uv, virtual, environment, extras=extras)
    requirement = str(wheel) + ("[" + ",".join(extras) + "]" if extras else "")
    python = virtual / "bin/python"
    _command(
        [uv, "pip", "install", "--offline", "--no-index", "--no-deps", "--python", str(python), requirement],
        environment,
    )
    _command([uv, "pip", "check", "--python", str(python)], environment)
    return python


@pytest.fixture(scope="module")
def installed_learning(tmp_path_factory):
    owned = tmp_path_factory.mktemp("installed-expert-mixtures")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_NO_PROGRESS="1", UV_LINK_MODE="copy")
    virtual = owned / "base"
    supplied = environment.get("COLEMAN_TEST_LEARNING_WHEEL")
    if supplied:
        wheel = Path(supplied).resolve(strict=True)
        assert wheel.is_file() and wheel.suffix == ".whl"
        python = _install_existing_wheel(uv, wheel, virtual, environment)
    else:
        wheel = build_install_offline_wheel(uv, owned / "dist", virtual, environment)
        python = virtual / "bin/python"
    outside = owned / "outside"
    outside.mkdir()
    assert not outside.is_relative_to(ROOT)
    shutil.copyfile(ROOT / "tests/learning/installed_driver.py", outside / "driver.py")
    shutil.copyfile(ROOT / "examples/learning/ci.py", outside / "ci.py")
    for family in ("learning", "optimization"):
        shutil.copytree(ROOT / "tests/fixtures" / family / "v1", outside / "fixtures" / family / "v1")
    installation = {
        "wheel": str(wheel),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "python": str(python),
        "outside": str(outside),
        "supplied_prebuilt": bool(supplied),
    }
    (owned / "installation.json").write_text(json.dumps(installation, sort_keys=True))
    return {"uv": uv, "wheel": wheel, "python": python, "outside": outside, "environment": environment, "owned": owned}


def _invoke(installed, mode):
    outside = installed["outside"]
    result = subprocess.run(
        [str(installed["python"]), "-I", "-B", str(outside / "driver.py"), mode],
        cwd=outside,
        env=installed["environment"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_installed_base_ships_actual_three_reference_methods_and_shared_conformance(installed_learning):
    result = _invoke(installed_learning, "reference")
    assert {item["profile"] for item in result["methods"]} == {
        "hedge-constant",
        "hedge-anytime-cumulative",
        "fixed-share-excluding-origin",
    }
    assert all(item["raw_updates"] == 3 and item["shared_conformance_cases"] == 3 for item in result["methods"])
    assert result["independent_host_cases"] == 3
    assert result["scope"] == "constructed-protocol-mechanics"


def test_installed_positive_lambda_round_cadence_retains_pending_complete_state_and_exact_period_usage(
    installed_learning,
):
    result = _invoke(installed_learning, "controlled")
    assert result["profile"] == "controlled-hedge-l1-movement"
    assert (result["U"], result["R"], result["restored_pending_panels"]) == (4, 2, 1)
    assert result["positive_lambda_moving_and_hold"] is True
    assert result["physical_executions"] == 0


def test_installed_controlled_profiles_pass_actual_shared_round_protocol(installed_learning):
    result = _invoke(installed_learning, "controlled-conformance")
    assert {item["strength"] for item in result["modes"]} == {None, 0.0, 0.05}
    assert all(item["shared_conformance_cases"] == 3 for item in result["modes"])
    assert all(item["anchor_and_bounds"] and item["cadence_panels"] == 2 for item in result["modes"])


def test_installed_interval_profiles_resume_real_causal_replay_after_held_panel(installed_learning):
    result = _invoke(installed_learning, "interval-replay")
    assert {item["strength"] for item in result["modes"]} == {None, 0.05}
    for mode in result["modes"]:
        assert (mode["D"], mode["U"], mode["R"], mode["restored_pending_panels"]) == (2, 1, 1, 1)
        assert mode["complete_continuation_equal"] is True
        assert mode["anchor_and_bounds"] and mode["cadence"] == "interval"
        assert mode["physical_executions"] == 0


def test_installed_actual_reference_exchanges_typed_json_with_stdlib_only_host(installed_learning):
    result = _invoke(installed_learning, "exchange")
    assert result["schema_version"] == 1 and result["host_imported_coleman"] is False
    assert result["unknown_receipt_preserved"] is True and result["unavailable_feature_preserved"] is True
    assert result["microsecond_precision"] == 123456 and result["actual_updates"] == 1
    assert result["host_runtime"]["implementation"] == "cpython"
    assert result["host_runtime"]["version"]


def test_installed_learning_cold_imports_use_only_the_selected_base_distribution(installed_learning):
    python, outside, environment = (
        installed_learning["python"],
        installed_learning["outside"],
        installed_learning["environment"],
    )
    script = (
        'import json,sysconfig;print(json.dumps(sorted({sysconfig.get_paths()[key] for key in ("purelib","platlib")})))'
    )
    sites = json.loads(_command([str(python), "-I", "-B", "-c", script], environment, cwd=outside))
    modules = [
        "coleman.learning." + name
        for name in (
            "contracts",
            "_math",
            "reference",
            "control_contracts",
            "_controlled_output",
            "_controlled_state",
            "controlled",
            "direct",
            "schedule",
        )
    ]
    blocked = BLOCKED + ("coleman.replay",)
    result = _command(
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
            json.dumps(blocked),
        ],
        environment,
        cwd=outside,
    )
    observed = json.loads(result)
    assert observed["forbidden"] == [] and set(observed["modules"]) == set(modules)
    inventory = _command(
        [
            str(python),
            "-I",
            "-B",
            "-c",
            "import importlib.metadata,json;"
            "print(json.dumps(sorted(d.metadata['Name'].lower().replace('_','-') "
            "for d in importlib.metadata.distributions())))",
        ],
        environment,
        cwd=outside,
    )
    assert not {"numpy", "polars", "pyarrow", "duckdb", "scipy", "mcp"} & set(json.loads(inventory))


def test_same_installed_wheel_retains_legacy_native_imports_under_declared_ci_extra(installed_learning):
    installed = dict(installed_learning)
    installed["python"] = _install_existing_wheel(
        installed["uv"], installed["wheel"], installed["owned"] / "ci", installed["environment"], extras=("ci",)
    )
    result = _invoke(installed, "legacy")
    assert result["profile"] == "ci"
    assert result["legacy"] == ["Bandit", "DynamicBandit", "EXP3Policy", "UCB1Policy"]
    assert result["new_method"] == "Hedge"
