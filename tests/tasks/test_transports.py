"""Actual generic command/CLI/MCP metadata parity and cold import boundaries."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import cast

import pytest

from coleman.application.task_commands import MAXIMUM_REQUEST_BYTES, OPERATIONS, TaskCommands, TaskOperation
from coleman.application.tasks import TaskService
from coleman.tasks.requests import TaskJobCommand
from coleman.transports import task_mcp
from coleman.workflows.contracts import WorkflowError


def _command(identity="missing"):
    return TaskJobCommand(
        schema_version=1, contract="coleman-tasks-v1", artifact_type="job_command", job_id=identity
    ).to_json()


def test_actual_neutral_commands_match_library_and_do_not_ignore_request_fields(tmp_path):
    """Keep metadata parity and refuse fields that a route does not consume."""
    service = TaskService(tmp_path)
    commands = TaskCommands(service)
    assert commands.invoke("discover") == service.discover().to_json()
    assert json.loads(commands.invoke("reconnect"))["jobs"] == []
    assert json.loads(commands.invoke("cleanup"))["expired_jobs"] == []
    for operation in ("discover", "cleanup", "reconnect"):
        assert json.loads(commands.invoke(operation, "{}"))["error"] == "invalid_input"
    for operation in ("status", "cancel", "resume", "report"):
        assert json.loads(commands.invoke(operation, _command()))["error"] == "not_found"
    assert json.loads(commands.invoke(cast(TaskOperation, "unapproved-operation")))["error"] == "invalid_input"


@pytest.mark.parametrize(
    "operation", [operation for operation in OPERATIONS if operation not in ("discover", "cleanup", "reconnect")]
)
@pytest.mark.parametrize("body", (None, '{"schema_version":2}', " " * (MAXIMUM_REQUEST_BYTES + 1)))
def test_interactive_commands_bound_or_reject_malformed_requests_before_provider(tmp_path, operation, body):
    """Reject absent, future or oversized input before execution dispatch."""
    response = json.loads(TaskCommands(TaskService(tmp_path)).invoke(operation, body))
    assert response["artifact_type"] == "failure"
    assert response["error"] == "invalid_input"
    assert "Traceback" not in json.dumps(response)


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (WorkflowError("private-host-detail"), "conflict_or_limit"),
        (ValueError("private-host-detail"), "invalid_input"),
        (OSError("private-host-detail"), "storage_unavailable"),
        (sqlite3.OperationalError("private-host-detail"), "storage_unavailable"),
    ],
)
def test_expected_transport_failures_never_expose_host_exception_details(tmp_path, monkeypatch, failure, code):
    """Translate expected boundary failures without publishing private details."""
    service = TaskService(tmp_path)

    def fail():
        raise failure

    monkeypatch.setattr(service, "discover", fail)
    body = TaskCommands(service).invoke("discover")
    assert json.loads(body)["error"] == code
    assert "private-host-detail" not in body


def test_real_mcp_sdk_uses_same_neutral_commands_and_only_opt_in_task_tools(tmp_path):
    """Negotiate actual tools and compare their replies with the library edge."""
    sdk = pytest.importorskip("mcp.client")
    service = TaskService(tmp_path)
    commands = TaskCommands(service)

    async def exercise():
        async with sdk.Client(task_mcp.create_task_server(service)) as client:
            tools = (await client.list_tools()).tools
            assert {tool.name for tool in tools} == {"task_" + operation for operation in OPERATIONS}
            for operation in ("discover", "reconnect", "cleanup"):
                reply = await client.call_tool("task_" + operation, {})
                assert not reply.is_error
                assert reply.content[0].text == commands.invoke(operation)
            for operation in ("status", "cancel", "resume", "report"):
                reply = await client.call_tool("task_" + operation, {"command_json": _command()})
                assert not reply.is_error
                assert reply.content[0].text == commands.invoke(operation, _command())
            for operation in ("plan", "execute"):
                reply = await client.call_tool("task_" + operation, {"request_json": '{"schema_version":2}'})
                assert not reply.is_error
                assert json.loads(reply.content[0].text)["error"] == "invalid_input"
            # Principal/setup fields are absent from every SDK-generated tool schema.
            for tool in tools:
                fields = tool.input_schema.get("properties", {})
                assert not {"owner", "workspace", "bootstrap", "module", "path", "url"}.intersection(fields)

    asyncio.run(exercise())


@pytest.mark.parametrize("missing", ("mcp", "nested_sdk_dependency"))
def test_optional_sdk_loading_explains_only_missing_top_level_sdk(tmp_path, monkeypatch, missing):
    """Explain a missing optional SDK while preserving nested import failures."""

    def absent(_name):
        raise ModuleNotFoundError("missing", name=missing)

    monkeypatch.setattr(task_mcp, "import_module", absent)
    service = TaskService(tmp_path)
    if missing == "mcp":
        with pytest.raises(ImportError, match=r"\[mcp\]"):
            task_mcp.create_task_server(service)
    else:
        with pytest.raises(ModuleNotFoundError) as caught:
            task_mcp.create_task_server(service)
        assert caught.value.name == missing


def test_source_cli_metadata_round_trip_matches_same_library_service(tmp_path):
    """Exercise the real source console separately from installed qualification."""
    result = subprocess.run(
        [sys.executable, "-B", "-m", "coleman.application.task_cli", "--workspace", str(tmp_path), "discover"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == TaskService(tmp_path).discover().to_json()
    # This source-console check is separate from noneditable installed qualification.


def test_generic_service_metadata_cold_import_requires_no_ci_or_mcp_stack(tmp_path):
    """Block optional scientific/transport imports in an actual cold child."""
    paths = [str(Path(__file__).parents[2]), sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    code = """
import importlib.abc,sys
sys.path[:0]=PATHS
blocked=('polars','numpy','pyarrow','duckdb','mcp','coleman.cli','coleman.transports',
         'coleman.application.native_executor','coleman.application.workflow_native','coleman.runner')
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==p or fullname.startswith(p+'.') for p in blocked):
            raise AssertionError('Unexpected domain/transport dependency: '+fullname)
sys.meta_path.insert(0,Boundary())
from pathlib import Path
from coleman.application.tasks import TaskService
from coleman.tasks.requests import TaskPlanningRequest
from coleman.tasks.results import TaskCompletion
service=TaskService(Path(WORKSPACE))
assert service.discover().bootstrap is None
assert not any(any(n==p or n.startswith(p+'.') for p in blocked) for n in sys.modules)
print('neutral metadata without CI/MCP')
""".replace("PATHS", repr(paths)).replace("WORKSPACE", repr(str(tmp_path)))
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "neutral metadata without CI/MCP"
