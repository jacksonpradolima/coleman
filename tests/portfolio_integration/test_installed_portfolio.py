"""Noneditable SDK and real synthetic provider wheels outside the checkout."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.installed_support import build_install_stdlib_wheel, seed_locked_dependencies
from tests.portfolio_integration.installed_driver import _state_attempt

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("phase", ["open", "commit"])
@pytest.mark.parametrize("code", [5, 6, 261, 262])
def test_numeric_contention_is_bounded_at_both_host_operations_without_retry(phase, code):
    import sqlite3

    calls = []
    error = sqlite3.OperationalError("opaque message")
    error.sqlite_errorcode = code

    def open_store():
        calls.append("open")
        if phase == "open":
            raise error
        return "opened-store"

    def commit(store):
        assert store == "opened-store"
        calls.append("commit")
        raise error

    assert _state_attempt(open_store, commit) == "busy"
    assert calls == (["open"] if phase == "open" else ["open", "commit"])


@pytest.mark.parametrize("phase", ["open", "commit"])
@pytest.mark.parametrize("code", [None, 1, 11, "5", -251])
def test_other_sql_errors_propagate_even_when_the_message_mentions_a_lock(phase, code):
    import sqlite3

    error = sqlite3.OperationalError("database is locked")
    if code is not None:
        error.sqlite_errorcode = code

    def operation():
        raise error

    with pytest.raises(sqlite3.OperationalError) as caught:
        _state_attempt(operation if phase == "open" else lambda: None, lambda _: operation())
    assert caught.value is error


@pytest.mark.parametrize("phase", ["open", "commit"])
def test_existing_lock_timeout_is_busy_and_success_keeps_the_real_receipt_status(phase):
    def timeout():
        raise TimeoutError("original fixed lock wait")

    assert _state_attempt(timeout if phase == "open" else lambda: None, lambda _: timeout()) == "busy"
    assert _state_attempt(lambda: "opened-store", lambda value: (value, "new")) == ("opened-store", "new")


def command(arguments, environment, cwd, *, timeout=180):
    result = subprocess.run(
        arguments, cwd=cwd, env=environment, capture_output=True, text=True, timeout=timeout, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    owned = tmp_path_factory.mktemp("installed-synthetic-portfolio")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(
        UV_OFFLINE="1", UV_NO_INDEX="1", UV_NO_PROGRESS="1", UV_LINK_MODE="copy", UV_PYTHON_DOWNLOADS="never"
    )
    virtual = owned / "base"
    core = build_install_stdlib_wheel(uv, owned / "core-dist", virtual, environment)
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
            str(owned / "provider-dist"),
        ],
        environment,
        ROOT / "examples/synthetic_portfolio/provider",
    )
    providers = tuple((owned / "provider-dist").glob("*.whl"))
    assert len(providers) == 1
    wheels = (core, providers[0])
    command(
        [
            uv,
            "pip",
            "install",
            "--offline",
            "--no-index",
            "--no-deps",
            "--python",
            str(virtual / "bin/python"),
            str(providers[0]),
        ],
        environment,
        owned,
    )
    command([uv, "pip", "check", "--python", str(virtual / "bin/python")], environment, owned)
    outside = owned / "outside"
    outside.mkdir()
    assert not outside.is_relative_to(ROOT)
    shutil.copyfile(ROOT / "tests/portfolio_integration/installed_driver.py", outside / "driver.py")
    return {
        "owned": owned,
        "virtual": virtual,
        "environment": environment,
        "outside": outside,
        "uv": uv,
        "wheels": wheels,
    }


def invoke(installed, mode, *, virtual=None):
    selected = virtual or installed["virtual"]
    output = command(
        [
            str(selected / "bin/python"),
            "-I",
            "-B",
            str(installed["outside"] / "driver.py"),
            mode,
            str(installed["outside"] / mode),
        ],
        installed["environment"],
        installed["outside"],
    )
    return json.loads(output)


def expected_state(installed):
    """Retain exact-source direct evidence without depending on pytest order."""
    if "direct" not in installed:
        installed["direct"] = invoke(installed, "direct")
    return installed["direct"]["state"]


def test_installed_domain_has_real_exact_arithmetic_and_no_transport_dependency(installed):
    result = invoke(installed, "boundaries")
    assert result == {"actual_net_value": "10000/101", "transport_imports": False}


def test_installed_full_components_continue_original_pending_and_state(installed):
    result = invoke(installed, "direct")
    assert result["complete_continuation"] and result["periods"] == 16 and result["predictor_labels"] == 15
    installed["direct"] = result


def test_installed_actual_causal_engine_matches_direct_with_ready_phase_restart(installed):
    result = invoke(installed, "replay")
    assert result["actual_causal_engine"] and not result["future_tape_in_consumer"]
    assert result["state"] == expected_state(installed)


def test_installed_real_store_interrupts_without_false_ack_and_resumes_identically(installed):
    result = invoke(installed, "durable")
    assert result["acknowledged_before_interruption"] == 8
    assert result["retained_frontier"] == 16
    assert result["concurrent_mathematical_bodies"] == 1
    assert result["concurrent_writer_statuses"].count("new") == 1
    assert result["state"] == expected_state(installed)


def test_installed_real_shared_task_uses_identical_components_and_original_request(installed):
    result = invoke(installed, "task")
    assert result["periods"] == 16 and result["state"] == expected_state(installed)


def test_installed_actual_cli_executes_same_frozen_task(installed):
    result = invoke(installed, "cli")
    assert result["transport"] == "actual-cli" and result["state"] == expected_state(installed)


def test_installed_actual_optional_mcp_executes_same_frozen_task(installed):
    virtual = installed["owned"] / "optional"
    command(
        [sys.executable, "-I", "-B", "-m", "venv", "--without-pip", str(virtual)],
        installed["environment"],
        installed["outside"],
    )
    seed_locked_dependencies(installed["uv"], virtual, installed["environment"], extras=("mcp",))
    command(
        [
            installed["uv"],
            "pip",
            "install",
            "--offline",
            "--no-index",
            "--no-deps",
            "--python",
            str(virtual / "bin/python"),
            *(str(path) for path in installed["wheels"]),
        ],
        installed["environment"],
        installed["outside"],
    )
    result = invoke(installed, "mcp", virtual=virtual)
    assert result["transport"] == "actual-stdio" and result["state"] == expected_state(installed)
