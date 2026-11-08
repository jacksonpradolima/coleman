"""The inherited CLI/MCP directory bounds startup filesystem authority."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from coleman.application import task_cli
from coleman.application.task_commands import TaskCommands
from coleman.application.tasks import TaskService
from coleman.tasks.identity import HostBootstrapProfile
from coleman.tasks.requests import TaskJobCommand

FIXTURES = Path(__file__).parents[1] / "fixtures" / "tasks" / "v1"
ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("field", ("workspace", "bootstrap"))
@pytest.mark.parametrize("route", ("absolute", "parent", "sibling_prefix", "symlink"))
def test_startup_escape_refuses_before_file_read_or_workspace_creation(tmp_path, monkeypatch, field, route):
    """Resolve real paths before either approval I/O or service construction."""
    directory = tmp_path / "allowed"
    directory.mkdir()
    outside = tmp_path / ("allowed-other" if route == "sibling_prefix" else "outside")
    outside.mkdir()
    sentinel = outside / "approval.json"
    sentinel.write_bytes(b"authored outside bytes\n")
    approval = directory / "approval.json"
    approval.write_bytes((FIXTURES / "bootstrap.json").read_bytes())
    workspace = directory / "new-workspace"
    selected = outside / ("workspace" if field == "workspace" else "approval.json")
    if route == "parent":
        selected = Path("..") / outside.name / selected.name
    elif route == "symlink":
        alias = directory / "alias"
        alias.symlink_to(outside, target_is_directory=True)
        selected = alias / selected.name
    if field == "workspace":
        workspace = selected
    else:
        approval = selected
    monkeypatch.chdir(directory)
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    opened = []
    services = []

    def forbidden_open(path, *args, **kwargs):
        opened.append(path)
        raise AssertionError("Escaping startup must refuse before opening any file")

    def forbidden_service(*args, **kwargs):
        services.append(args)
        raise AssertionError("Escaping startup must refuse before creating metadata")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", forbidden_open)
        guard.setattr(task_cli, "TaskService", forbidden_service)
        with pytest.raises(ValueError, match="inherited current directory"):
            task_cli.open_task_service(workspace, bootstrap=approval, owner="operator")
    assert opened == []
    assert services == []
    assert not workspace.exists()
    assert sentinel.read_bytes() == b"authored outside bytes\n"
    assert not (directory / "new-workspace").exists()


def test_contained_canonical_startup_opens_real_provider_free_metadata(tmp_path, monkeypatch):
    """Normalize admitted relative paths without importing a provider."""
    (tmp_path / "config").mkdir()
    approval = tmp_path / "approval.json"
    approval.write_bytes((FIXTURES / "bootstrap.json").read_bytes())
    expected = HostBootstrapProfile.from_json(approval.read_bytes())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    service = task_cli.open_task_service(
        Path("config/../workspace"), bootstrap=Path("config/../approval.json"), owner="operator"
    )
    assert service.store.root == tmp_path / "workspace"
    assert service.owner == "operator"
    assert service.discover().bootstrap == expected.binding
    with closing(service.store.connect()) as connection:
        assert service.store.records(connection) == ()


def test_programmatic_operator_retains_explicit_root_outside_cli_namespace(tmp_path, monkeypatch):
    """The host namespace is an entry-point boundary, not a library root change."""
    directory = tmp_path / "host"
    directory.mkdir()
    monkeypatch.chdir(directory)
    root = tmp_path / "programmatic-workspace"
    service = TaskService(root, owner="operator")
    assert service.store.root == root
    assert service.owner == "operator"
    with closing(service.store.connect()) as connection:
        assert service.store.records(connection) == ()


@pytest.mark.parametrize("field", ("workspace", "bootstrap", "owner", "directory"))
def test_request_json_cannot_replace_the_configured_host_namespace(tmp_path, monkeypatch, field):
    """The shared CLI/MCP command edge rejects path authority in request JSON."""
    root = tmp_path / "workspace"
    service = TaskService(root, owner="operator")
    commands = TaskCommands(service)
    command = TaskJobCommand(
        schema_version=1, contract="coleman-tasks-v1", artifact_type="job_command", job_id="unknown-job"
    )
    raw = command.model_dump(mode="json")
    raw[field] = str(tmp_path / "outside" / "approval.json")
    wire = json.dumps(raw)
    opened = []

    def forbidden_open(path, *args, **kwargs):
        opened.append(path)
        raise AssertionError("Request JSON cannot choose filesystem authority")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", forbidden_open)
        reply = json.loads(commands.invoke("status", wire))
    assert reply["error"] == "invalid_input"
    assert opened == []
    assert service.store.root == root
    assert service.owner == "operator"
    assert service.bootstrap_profile is None
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("module", ("coleman.application.task_cli", "coleman.transports.task_mcp"))
@pytest.mark.parametrize("field", ("workspace", "bootstrap"))
def test_actual_entrypoint_refuses_escape_before_optional_imports(tmp_path, module, field):
    """Real startup rejects an outsider before MCP/provider code or metadata."""
    directory = tmp_path / "host"
    directory.mkdir()
    workspace = directory / "workspace"
    approval = directory / "approval.json"
    approval.write_bytes((FIXTURES / "bootstrap.json").read_bytes())
    outside = tmp_path / "outside"
    if field == "workspace":
        workspace = outside / "workspace"
    else:
        approval = outside / "approval.json"
    argv = [module, "--workspace", str(workspace), "--bootstrap", str(approval)]
    if module == "coleman.application.task_cli":
        argv.append("discover")
    script = f"""
import runpy
import sys
sys.path.insert(0, {str(ROOT)!r})
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mcp' or fullname.startswith(('mcp.', 'coleman_typed_hedge')):
            raise AssertionError('Startup escaped its path guard: ' + fullname)
sys.meta_path.insert(0, Guard())
sys.argv = {argv!r}
runpy.run_module({module!r}, run_name='__main__')
"""
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "Task host startup paths must remain within the inherited current directory" in result.stderr
    assert "AssertionError" not in result.stderr
    assert not workspace.exists()
    assert not outside.exists()
