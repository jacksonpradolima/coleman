"""Host arguments and bounded stdin stay separate from task request authority."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import closing
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from coleman.application import task_cli
from coleman.application.workflow_store import WorkflowStore
from coleman.tasks.identity import HostBootstrapProfile
from coleman.tasks.requests import TaskJobCommand
from coleman.transports import task_mcp

FIXTURES = Path(__file__).parents[1] / "fixtures" / "tasks" / "v1"


def test_stdin_status_uses_real_metadata_service_and_one_bounded_read(tmp_path, monkeypatch, capsys):
    """Bound the complete stdin request before reaching the real owned metadata."""
    reads = []
    command = TaskJobCommand(
        schema_version=1, contract="coleman-tasks-v1", artifact_type="job_command", job_id="missing-owned-job"
    )

    class BoundedInput(BytesIO):
        """Record acquisition size while returning the original request bytes."""

        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=BoundedInput(command.to_json().encode())))
    monkeypatch.setattr(
        sys,
        "argv",
        ["coleman-task", "--workspace", str(tmp_path), "--owner", "reader", "status", "--request", "-"],
    )
    task_cli.main()
    reply = json.loads(capsys.readouterr().out)
    assert reads == [task_cli.MAXIMUM_REQUEST_BYTES + 1]
    assert reply["artifact_type"] == "failure"
    assert reply["error"] == "not_found"
    store = WorkflowStore(tmp_path)
    with closing(store.connect()) as connection:
        assert not store.records(connection)


def test_explicit_host_profile_registers_declarations_without_loading_provider(tmp_path, monkeypatch):
    """Constructed metadata establishes equality only, never installed approval."""
    selected = tmp_path / "operator.json"
    selected.write_bytes((FIXTURES / "bootstrap.json").read_bytes())
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    service = task_cli.open_task_service(tmp_path / "workspace", bootstrap=selected, owner="operator")
    expected = HostBootstrapProfile.from_json(selected.read_bytes())
    assert service.discover().bootstrap == expected.binding
    assert service.owner == "operator"
    assert service.bootstrap_profile == expected
    assert not (service.store.root / "jobs").exists()


def test_excessive_operator_profile_refuses_before_workspace_creation(tmp_path, monkeypatch):
    """Oversized bootstrap input never initializes its requested workspace."""
    selected = tmp_path / "operator.json"
    selected.write_bytes(b" " * (16 * 1024 + 1))
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    with pytest.raises(ValueError):
        task_cli.open_task_service(workspace, bootstrap=selected, owner="operator")
    assert not workspace.exists()


def test_mcp_main_passes_only_trusted_host_setup_to_fixed_stdio_transport(tmp_path, monkeypatch):
    """Isolate argv/transport selection; installed tests exercise the actual SDK."""
    calls = []
    service = object()

    def open_service(workspace, *, bootstrap, owner):
        calls.append(("host", workspace, bootstrap, owner))
        return service

    def create_server(selected):
        assert selected is service
        return SimpleNamespace(run=lambda **options: calls.append(("run", options)))

    profile = tmp_path / "reviewed-operator.json"
    monkeypatch.setattr(task_mcp, "open_task_service", open_service)
    monkeypatch.setattr(task_mcp, "create_task_server", create_server)
    monkeypatch.setattr(
        sys,
        "argv",
        ["coleman-task-mcp", "--workspace", str(tmp_path), "--bootstrap", str(profile), "--owner", "reader"],
    )
    task_mcp.main()
    assert calls == [("host", tmp_path, profile, "reader"), ("run", {"transport": "stdio"})]


@pytest.mark.parametrize("module", ("coleman.application.task_cli", "coleman.transports.task_mcp"))
def test_actual_module_help_needs_no_optional_backend_or_workspace(tmp_path, module):
    """Exercise real module entry points with optional imports actively refused."""
    script = """
import importlib.abc, runpy, sys
class NoOptionalBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('mcp', 'numpy', 'polars', 'pyarrow'):
            raise AssertionError('Help initialized an optional backend: ' + fullname)
sys.meta_path.insert(0, NoOptionalBackend())
module = sys.argv[1]
sys.argv = [module, '--help']
runpy.run_module(module, run_name='__main__')
"""
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, module],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--workspace" in result.stdout and "--bootstrap" in result.stdout
    assert not (tmp_path / "workspace.sqlite3").exists()
    assert not (tmp_path / "jobs").exists()
