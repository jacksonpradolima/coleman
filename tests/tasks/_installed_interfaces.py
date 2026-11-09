"""Actual installed CLI/stdio MCP and native TCPCI integration, without models."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from contextlib import closing
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from coleman.application.tasks import TaskService
from coleman.application.workflow_store import JobRecord
from coleman.tasks.contracts import TaskJobRecord

if TYPE_CHECKING:
    from tests.tasks import _installed_support as shared
else:
    shared = import_module("task_installed_support")

TASK_TOOLS = {
    "task_" + name
    for name in ("discover", "plan", "execute", "status", "cancel", "resume", "report", "reconnect", "cleanup")
}
APPLICATION_TOOLS = {"discover", "submit", "status", "cancel", "result", "export_result", "import_result"}
WORKFLOW_TOOLS = {
    "workflow_discover",
    "plan_experiment",
    "execute_plan",
    "experiment_status",
    "cancel_experiment",
    "resume_experiment",
    "inspect_experiment",
    "evaluate_experiment",
    "experiment_report",
    "export_experiment",
    "import_experiment_evidence",
    "read_experiment_evidence",
    "list_experiment_artifacts",
    "read_experiment_artifact",
}


def _command(arguments: list[str], *, body: str | None = None):
    """Run a real installed command with the unchanged180-second process fence."""
    result = shared.run_checked(arguments, body=body)
    value = json.loads(result.stdout)
    assert value.get("artifact_type") != "failure", value
    return value


def _task_cli(root: Path, approval: Path, operation: str, body: str):
    return _command(
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "coleman.application.task_cli",
            "--workspace",
            str(root),
            "--bootstrap",
            str(approval),
            operation,
            "--request",
            "-",
        ],
        body=body,
    )


def _tool_payload(reply):
    """Decode the actual SDK's string-tool text, never a fabricated response."""
    assert not reply.is_error
    assert len(reply.content) == 1
    value = json.loads(reply.content[0].text)
    assert value.get("artifact_type") != "failure", value
    return value


def generic_interfaces(root: Path, approval: Path):
    """Run actual CLI and stdio task jobs before installing any CI capability."""
    for name in ("numpy", "polars", "pyarrow"):
        assert importlib.util.find_spec(name) is None
    current = shared.profile(approval)
    supplied = shared.sample(current)
    request = supplied.planning_request().to_json()
    preview = _task_cli(root, approval, "plan", request)
    execution = _task_cli(root, approval, "execute", shared.command(preview["plan_id"], "installed-cli").to_json())
    cli_identity = execution["job"]["job_id"]
    cli = shared.inspect_complete(root, approval, cli_identity)
    mcp = asyncio.run(_generic_stdio(root, approval, request))
    assert mcp["plan"] == preview["digest"]
    actual = shared.inspect_complete(root, approval, mcp["job_id"])
    assert actual["direct_state"] == cli["direct_state"]
    records = TaskService(root, bootstrap_profile=current).reconnect()
    assert len(records) == 2
    result = {"cli": cli, "mcp": actual, "task_tools": mcp["tools"], "ci_installed": False}
    shared.save(root.parent / "generic-interfaces-proof.json", result)
    return result


async def _generic_stdio(root: Path, approval: Path, request: str):
    """Disconnect/reconnect real stdio clients while fixed workers remain durable."""
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-I", "-B", "-m", "coleman.transports.task_mcp", "--workspace", str(root), "--bootstrap", str(approval)],
        cwd=str(Path.cwd()),
    )
    async with Client(parameters) as client:
        await _check_task_tools(client)
        preview = _tool_payload(await client.call_tool("task_plan", {"request_json": request}))
        execution = _tool_payload(
            await client.call_tool(
                "task_execute", {"request_json": shared.command(preview["plan_id"], "installed-stdio").to_json()}
            )
        )
        identity = execution["job"]["job_id"]
    # Closing the requesting stdio server is deliberately not cancellation.
    command_json = json.dumps(
        {"contract": "coleman-tasks-v1", "schema_version": 1, "artifact_type": "job_command", "job_id": identity}
    )
    async with Client(parameters) as client:
        deadline = time.monotonic() + shared.JOB_OBSERVATION_SECONDS
        while time.monotonic() < deadline:
            status = _tool_payload(await client.call_tool("task_status", {"command_json": command_json}))
            if status["job"]["state"] == "completed" and not status["worker_active"]:
                break
            assert status["job"]["state"] in {"queued", "running", "completed"}, status
            await asyncio.sleep(0.025)
        else:
            raise TimeoutError("Installed stdio task did not complete within its declared allowance")
        completion = _tool_payload(await client.call_tool("task_report", {"command_json": command_json}))
        assert completion["plan_fingerprint"] == preview["digest"]
    return {"job_id": identity, "plan": preview["digest"], "tools": sorted(TASK_TOOLS)}


async def _check_task_tools(client):
    """Check the actual tool catalogue and reject transport authority leakage."""
    catalogue = (await client.list_tools()).tools
    assert {tool.name for tool in catalogue} == TASK_TOOLS
    for tool in catalogue:
        fields = tool.input_schema.get("properties", {})
        assert not {"owner", "workspace", "bootstrap", "module", "url", "path"}.intersection(fields)


def _wait_native(service, view):
    """Observe an actual native owner under its existing60-second task fence."""
    deadline = time.monotonic() + shared.JOB_OBSERVATION_SECONDS
    while time.monotonic() < deadline:
        view = service.status(view.job_id)
        if view.state == "completed" and not view.worker_active:
            return view
        assert view.state in {"queued", "running", "completed"}, view
        time.sleep(0.025)
    raise TimeoutError("Installed actual TCPCI did not complete within60seconds")


def _run_native(service, root: Path):
    """Run the actual admitted three-build native workflow through its service."""
    from coleman.application.workflow_registry import owned_fixture
    from coleman.workflows.contracts import PlanningRequest, WorkflowLimits
    from coleman.workflows.lifecycle import ExecutePlan

    manifest = service.register(owned_fixture(root))
    preview = service.plan(
        PlanningRequest(
            schema_version=1,
            dataset_id=manifest.dataset_id,
            policies=("Random",),
            seeds=(42,),
            limits=WorkflowLimits(max_attempts=1, attempt_seconds=60.0),
        )
    )
    view = service.execute(ExecutePlan(schema_version=1, plan_id=preview.plan_id, request_id="installed-real-tcpci"))
    view = _wait_native(service, view)
    report = service.report(view.job_id)
    assert len(report.trials) == 1
    assert tuple(row.source_build_id for row in report.trials[0].rows) == (1, 2, 3)
    assert view.attempts == 1
    assert view.invocations == 1
    return view, report


def native_and_catalogues(root: Path, approval: Path):
    """Use actual TCPCI in the SAME store/engine and unchanged legacy tools."""
    from coleman.application.workflows import WorkflowService

    service = WorkflowService(root, bootstrap_profile=shared.profile(approval))
    view, report = _run_native(service, root)
    with closing(service.store.connect()) as connection:
        records = service.store.records(connection)
    assert sum(isinstance(item, JobRecord) for item in records) == 1
    assert sum(isinstance(item, TaskJobRecord) for item in records) == 2
    catalogues = asyncio.run(_legacy_catalogues(root))
    result = {
        "native_job": view.job_id,
        "native_rows": 3,
        "native_attempts": view.attempts,
        "same_store_jobs": len(records),
        "legacy_tools": catalogues,
        "native_report": report.to_json(),
    }
    shared.save(root.parent / "native-and-catalogues-proof.json", result)
    return result


async def _legacy_catalogues(root: Path):
    """Negotiate both real legacy stdio catalogues; generic tools do not leak."""
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    observed = []
    for workspace in (None, root):
        arguments = ["-I", "-B", "-m", "coleman.transports.mcp"]
        expected = APPLICATION_TOOLS
        if workspace is not None:
            arguments.extend(["--workspace", str(workspace)])
            expected = expected | WORKFLOW_TOOLS
        parameters = StdioServerParameters(command=sys.executable, args=arguments, cwd=str(Path.cwd()))
        async with Client(parameters) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            assert names == expected
            observed.append(sorted(names))
    return observed
