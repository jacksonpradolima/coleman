"""Real optional SDK structured tools, typed inputs and transport-neutral error boundaries."""

import asyncio
import json
import sqlite3
import time

import pytest

from coleman.application import ApplicationService, Caller, default_registry
from coleman.application.workflow_exports import EvidenceBundle
from coleman.transports.workflow_tools import _reply
from coleman.workflows.contracts import PlanningRequest
from tests.workflows.test_execution import setup_workflow


def test_sdk_in_process_tuning_and_portable_read_only_evidence(tmp_path):
    pytest.importorskip("mcp")
    from mcp.client import Client

    from coleman.transports.mcp import create_server

    async def exercise():
        workflows, _ = setup_workflow(tmp_path / "workspace")
        with ApplicationService(default_registry(), root=tmp_path / "ephemeral") as ephemeral:
            async with Client(create_server(ephemeral, caller=Caller(subject="sdk"), workflows=workflows)) as client:
                assert client.protocol_version
                tools = (await client.list_tools()).tools
                assert len(tools) == 21
                assert next(tool.output_schema for tool in tools if tool.name == "plan_experiment") is not None
                discovery = (await client.call_tool("workflow_discover", {})).structured_content
                assert discovery["data"]["datasets"][0]["duration_unit"] == "s"
                for incomplete in ({"schema_version": 1}, {"schema_version": 1, "dataset_id": None}):
                    missing_dataset = await client.call_tool("plan_experiment", {"request": incomplete})
                    assert missing_dataset.structured_content is not None
                    assert missing_dataset.structured_content["error"] == "invalid_input"
                    assert missing_dataset.structured_content["missing_fields"] == ["dataset_id"]
                    assert missing_dataset.structured_content["data"] is None
                response = await client.call_tool(
                    "plan_experiment", {"request": {"schema_version": 1, "dataset_id": "owned-ci-three-build-v1"}}
                )
                plan = response.structured_content["data"]
                assert (plan["configurations"], plan["trials"], plan["expected_build_rows"]) == (4, 8, 24)
                response = await client.call_tool(
                    "execute_plan",
                    {"request": {"schema_version": 1, "request_id": "actual-sdk", "plan_id": plan["plan_id"]}},
                )
                job = response.structured_content["data"]
                command = {"request": {"schema_version": 1, "job_id": job["job_id"]}}
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    job = (await client.call_tool("experiment_status", command)).structured_content["data"]
                    if job["state"] == "completed":
                        break
                    assert job["state"] in ("queued", "running"), job
                    await asyncio.sleep(0.05)
                assert job["state"] == "completed"
                inspected = (await client.call_tool("inspect_experiment", command)).structured_content["data"]
                assert len(inspected["trials"]) == 4
                assert len(inspected["coordinates"]) == 4
                assert len(inspected["attempts"]) == 4
                await client.call_tool("resume_experiment", command)
                assert (await client.call_tool("cancel_experiment", command)).structured_content["data"][
                    "state"
                ] == "completed"
                assert (await client.call_tool("evaluate_experiment", command)).structured_content[
                    "error"
                ] == "conflict_or_limit"
                exported = (await client.call_tool("export_experiment", command)).structured_content
                assert exported["error"] is None
                bundle = EvidenceBundle.model_validate(exported["data"])
                imported = (
                    await client.call_tool("import_experiment_evidence", {"bundle_json": bundle.to_json()})
                ).structured_content["data"]
                read = await client.call_tool(
                    "read_experiment_evidence",
                    {"request": {"schema_version": 1, "evidence_id": imported["evidence_id"]}},
                )
                assert EvidenceBundle.model_validate(read.structured_content["data"]) == bundle
                missing = await client.call_tool(
                    "experiment_status", {"request": {"schema_version": 1, "job_id": "missing"}}
                )
                assert missing.structured_content["error"] == "not_found"
                unknown = await client.call_tool(
                    "plan_experiment", {"request": {"schema_version": 2, "dataset_id": "owned-ci-three-build-v1"}}
                )
                assert unknown.is_error
                malicious = await client.call_tool(
                    "plan_experiment",
                    {"request": {"schema_version": 1, "dataset_id": "owned-ci-three-build-v1", "path": "/unapproved"}},
                )
                assert malicious.is_error

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "error,code",
    [
        (KeyError("private"), "not_found"),
        (ValueError("private"), "conflict_or_limit"),
        (OSError("private"), "storage_unavailable"),
        (sqlite3.OperationalError("private"), "storage_unavailable"),
    ],
)
def test_bounded_public_errors_do_not_expose_host_exceptions(error, code):
    def fail():
        raise error

    reply = _reply(fail)
    assert reply.error == code
    assert "private" not in reply.to_json()


def test_structured_missing_fields():
    reply = _reply(lambda: PlanningRequest.model_validate({"schema_version": 1}))
    assert reply.error == "invalid_input"
    assert reply.missing_fields == ("dataset_id",)
    assert json.loads(reply.to_json())["data"] is None
