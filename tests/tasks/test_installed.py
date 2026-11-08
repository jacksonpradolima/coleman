"""Real base/external wheels, durable process faults, CLI/MCP and native parity."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PROCESS_OBSERVATION_SECONDS = 180
SETUP_OBSERVATION_SECONDS = 600
COPIED_DRIVERS = ("installed_driver.py", "_installed_support.py", "_installed_faults.py", "_installed_interfaces.py")


def _bootstrap(context, mode):
    """Charge all three setup phases to one finite600-second actual budget."""
    remaining = context["setup_remaining"]
    if remaining <= 0:
        raise TimeoutError("Installed fixture exhausted its aggregate600-second setup allowance")
    started = time.monotonic()
    arguments = [sys.executable, "-m", "tests.tasks._installed_build", str(context["owned"]), mode]
    with subprocess.Popen(
        arguments,
        cwd=ROOT,
        env=context["environment"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as child:
        try:
            output, errors = child.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
            raise
    elapsed = time.monotonic() - started
    context["setup_remaining"] -= elapsed
    observation = {
        "mode": mode,
        "seconds": elapsed,
        "returncode": child.returncode,
        "remaining_seconds": context["setup_remaining"],
    }
    context["setup_steps"].append(observation)
    Path(context["owned"]).joinpath(f"bootstrap-{mode}.log").write_text(output + errors)
    assert child.returncode == 0, output + errors


def _command(arguments, environment, *, cwd, body=None, timeout=PROCESS_OBSERVATION_SECONDS):
    result = subprocess.run(
        arguments, cwd=cwd, env=environment, input=body, capture_output=True, text=True, timeout=timeout, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(name="installed_tasks", scope="module")
def _installed_tasks(tmp_path_factory):
    """Let ROOT execute the actual offline build once, under one finite fence."""
    owned = tmp_path_factory.mktemp("installed-typed-executors")
    if os.name != "posix":
        pytest.skip("The documented immutable typed-executor host requires POSIX local ownership")
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(
        UV_OFFLINE="1", UV_NO_INDEX="1", UV_NO_PROGRESS="1", UV_LINK_MODE="copy", UV_PYTHON_DOWNLOADS="never"
    )
    fixture = {
        "owned": str(owned),
        "environment": environment,
        "setup_remaining": float(SETUP_OBSERVATION_SECONDS),
        "setup_steps": [],
    }
    _bootstrap(fixture, "base")
    built = json.loads((owned / "built.json").read_text())
    outside = owned / "outside"
    outside.mkdir()
    assert not outside.is_relative_to(ROOT)
    copied = {}
    for name in COPIED_DRIVERS:
        source = ROOT / "tests/tasks" / name
        destination = outside / ("driver.py" if name == "installed_driver.py" else name)
        shutil.copyfile(source, destination)
        copied[destination.name] = sha256(destination.read_bytes()).hexdigest()
    recipe = ROOT / "examples/typed_executor/provider/src/coleman_typed_hedge/recipe.json"
    shutil.copyfile(recipe, outside / "recipe.json")
    shutil.copyfile(ROOT / "examples/typed_executor/ci.py", outside / "ci.py")
    guide = ROOT / "docs/typed-execution.md"
    blocks = re.findall(r"```python\n(.*?)\n```", guide.read_text(), re.DOTALL)
    assert len(blocks) == 4, "Qualify every complete literal Python guide block in one process"
    (outside / "guide.py").write_text("\n\n".join(blocks) + "\n")
    shutil.copyfile(recipe, outside / "operator-recipe.json")
    copied["recipe.json"] = sha256((outside / "recipe.json").read_bytes()).hexdigest()
    copied["ci.py"] = sha256((outside / "ci.py").read_bytes()).hexdigest()
    copied["guide.py"] = sha256((outside / "guide.py").read_bytes()).hexdigest()
    copied["operator-recipe.json"] = sha256((outside / "operator-recipe.json").read_bytes()).hexdigest()
    fixture.update(**built, outside=str(outside), copied=copied)
    observed = _invoke(fixture, "installation", built["virtual"], *built["wheels"], setup=True)
    approval = _invoke(fixture, "approve", outside / "recipe.json", outside / "profile.json", setup=True)
    fixture["profile"] = str(outside / "profile.json")
    (owned / "installation.json").write_text(
        json.dumps(
            {
                "actual": observed,
                "approval": approval,
                "copied": copied,
                "guide_source_sha256": sha256(guide.read_bytes()).hexdigest(),
                "setup_steps": fixture["setup_steps"],
                "limits": {"setup_seconds": 600, "process_seconds": 180, "job_seconds": 60},
            },
            sort_keys=True,
        )
    )
    return fixture


def _invoke(installed, mode, *arguments, python=None, setup=False):
    outside = Path(installed["outside"])
    executable = installed["python"] if python is None else python
    timeout = PROCESS_OBSERVATION_SECONDS
    if setup:
        timeout = min(timeout, installed["setup_remaining"])
        if timeout <= 0:
            raise TimeoutError("Installed fixture exhausted its aggregate setup allowance")
    start = time.monotonic()
    result = _command(
        [str(executable), "-I", "-B", str(outside / "driver.py"), mode, *(str(value) for value in arguments)],
        installed["environment"],
        cwd=outside,
        timeout=timeout,
    )
    if setup:
        seconds = time.monotonic() - start
        installed["setup_remaining"] -= seconds
        installed["setup_steps"].append(
            {"mode": mode, "seconds": seconds, "remaining_seconds": installed["setup_remaining"]}
        )
    return json.loads(result.stdout)


def test_installed_base_actual_direct_and_durable_complete_hedge_state(installed_tasks):
    """Qualify the exact public example and every literal Python guide block."""
    root = Path(installed_tasks["outside"]) / "base-run"
    result = _invoke(installed_tasks, "execute", root, installed_tasks["profile"], "installed-base")
    assert result["state_equal"] is True
    assert result["updates"] == 3
    assert result["generation"] == 2
    assert result["attempts"] == 1
    assert result["invocations"] == 1
    outside = Path(installed_tasks["outside"])
    literal = _command(
        [installed_tasks["python"], "-I", "-B", str(outside / "guide.py")], installed_tasks["environment"], cwd=outside
    )
    assert literal.stdout.strip() == "constructed_mechanism_only"
    guide = _invoke(installed_tasks, "inspect-guide", outside / "task-workspace", outside / "host-bootstrap.json")
    assert guide["state_equal"] is True
    assert guide["updates"] == 3
    assert guide["generation"] == 2


def test_installed_detached_parent_exit_and_current_profile_authority(installed_tasks):
    """Require supplied approval and actual queued work surviving parent exit."""
    outside = Path(installed_tasks["outside"])
    root = outside / "detached"
    prepared = _invoke(installed_tasks, "prepare", root, installed_tasks["profile"], "parent-exits")
    refused = _invoke(installed_tasks, "provider-free", root, installed_tasks["profile"], prepared["job_id"])
    assert refused["provider_loaded"] is False
    assert refused["missing_refused"] is True
    assert refused["wrong_refused"] is True
    evidence = outside / "parent-exit.json"
    _command(
        [
            installed_tasks["python"],
            "-I",
            "-B",
            str(outside / "driver.py"),
            "detached-parent",
            str(root),
            installed_tasks["profile"],
            prepared["request"],
            str(evidence),
        ],
        installed_tasks["environment"],
        cwd=outside,
    )
    launch = json.loads(evidence.read_text())
    assert launch["state_before_parent_exit"] == "queued"
    assert launch["actual_owned_gate"] is True
    result = _invoke(installed_tasks, "inspect", root, installed_tasks["profile"], prepared["job_id"])
    assert result["state_equal"] is True
    assert result["attempts"] == 1


def test_installed_real_inner_ack_death_reconciles_without_repeating_panels(installed_tasks):
    """Kill the real owner after its inner ACK and preserve accepted panels."""
    result = _invoke(
        installed_tasks, "recover", Path(installed_tasks["outside"]) / "ack-gap", installed_tasks["profile"]
    )
    assert result["interruption"]["exit_code"] == -9
    assert result["interruption"]["before"]["job"]["frontier"] is None
    assert result["complete"]["state_equal"] is True
    assert result["complete"]["updates"] == 3
    assert result["complete"]["generation"] == 2
    assert result["complete"]["attempts"] == 2


def test_installed_real_cancelled_owner_resumes_the_same_complete_state(installed_tasks):
    """Cancel an owned partial trajectory and resume its complete learner state."""
    result = _invoke(
        installed_tasks, "cancel-resume", Path(installed_tasks["outside"]) / "cancel", installed_tasks["profile"]
    )
    assert result["cancelled"]["state"] == "cancelled"
    assert result["cancelled"]["frontier"]["position"] == 2
    assert result["complete"]["state_equal"] is True
    assert result["complete"]["attempts"] == 2
    assert result["complete"]["invocations"] == 2


def test_installed_same_version_dependency_change_refuses_before_factory_or_restore(installed_tasks):
    """Refuse changed effective source bytes before any provider import."""
    result = _invoke(
        installed_tasks, "same-version", Path(installed_tasks["outside"]) / "changed-image", installed_tasks["profile"]
    )
    assert result["refusal"]["before_factory"] is True
    assert result["refusal"]["provider_loaded"] is False
    assert result["refusal"]["attempts_unchanged"] is True
    assert result["complete"]["state_equal"] is True
    assert result["complete"]["attempts"] == 2


def test_installed_actual_cli_stdio_and_native_share_one_engine_without_tool_leakage(installed_tasks):
    """Exercise task transports without CI, then native TCPCI in the same store."""
    optional = Path(installed_tasks["owned"]) / "optional"
    _bootstrap(installed_tasks, "optional-mcp")
    python = optional / "bin/python"
    outside = Path(installed_tasks["outside"])
    approval = outside / "optional-profile.json"
    _invoke(installed_tasks, "approve", outside / "recipe.json", approval, python=python, setup=True)
    root = outside / "mixed"
    generic = _invoke(installed_tasks, "generic-interfaces", root, approval, python=python)
    assert generic["ci_installed"] is False
    assert len(generic["task_tools"]) == 9
    assert generic["cli"]["state_equal"] is True
    assert generic["mcp"]["state_equal"] is True
    _bootstrap(installed_tasks, "optional-ci")
    native = _invoke(installed_tasks, "native-interfaces", root, approval, python=python)
    assert native["native_rows"] == 3
    assert native["native_attempts"] == 1
    assert native["same_store_jobs"] == 3
    assert [len(items) for items in native["legacy_tools"]] == [7, 21]
    (Path(installed_tasks["owned"]) / "optional-installation.json").write_text(
        json.dumps(
            {
                "setup_steps": installed_tasks["setup_steps"],
                "remaining_seconds": installed_tasks["setup_remaining"],
                "same_wheels": installed_tasks["wheels"],
            },
            sort_keys=True,
        )
    )
