"""Reproduce installed-library/stdio parity over the pinned, pre-populated Karaf source.

Run with the noneditable wheel's Python outside the checkout, providing an
approved cache and private output directory. This acceptance oracle starts real
native jobs, retains raw evidence and writes proof.json; it never fetches sources.
Assertions verify the demonstration, not empirical policy superiority.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

import coleman
from coleman.application.workflow_presets import register_karaf
from coleman.application.workflows import WorkflowService
from coleman.corpus.materialization import SourceCache
from coleman.results.domain import fingerprint
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.workflows.contracts import PlanningRequest, WorkflowLimits
from coleman.workflows.lifecycle import ExecutePlan, HostPolicy, WorkflowReport

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache", type=Path, required=True, help="Permitted pre-populated public Karaf source cache")
parser.add_argument("--output", type=Path, required=True, help="Private evidence/workspace directory outside Git")
args = parser.parse_args()
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=True)
assert Path(coleman.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), "Install a noneditable wheel first"
policy = HostPolicy(
    schema_version=1, caps=WorkflowLimits(max_configurations=64, max_trials=128, max_output_bytes=67108864)
)
service = WorkflowService(root / ("workspace-" + InstalledRuntime.capture().package_fingerprint[7:19]), policy=policy)
registered = register_karaf(service, SourceCache(args.cache.resolve()), root / "source-etl")
assert (registered.receipt.observed_cases, registered.receipt.replay_cases, registered.receipt.excluded_not_run) == (
    6876,
    6567,
    309,
)
request = PlanningRequest(
    schema_version=1,
    dataset_id=registered.manifest.dataset_id,
    limits=WorkflowLimits(max_trials=8, max_output_bytes=67108864),
)
plan = service.plan(request)
assert (plan.configurations, plan.trials, plan.expected_build_rows) == (4, 8, 64)
print("ORIGINAL SOURCE REGISTERED", registered.manifest.sha256, flush=True)


def wait(job):
    """Wait for real native completion, rejecting a failed or interrupted state.

    Parameters
    ----------
    job : JobView
        Host-issued durable handle returned by the actual application service.

    Returns
    -------
    JobView
        Complete snapshot after physical ownership releases. Polls for at most
        300 seconds; this acceptance timeout is separate from worker resource caps.
    """
    for _ in range(6000):
        view = service.status(job.job_id)
        if view.state == "completed" and not view.worker_active:
            return view
        assert view.state in ("queued", "running", "completed"), view.to_json()
        time.sleep(0.05)
    raise AssertionError("Original-source native workflow did not complete")


def mathematics(report):
    """Project complete logical rows while excluding physical process samples.

    Parameters
    ----------
    report : WorkflowReport
        Verified completed native evidence, including original full rankings.

    Returns
    -------
    dict
        Trial-keyed original build IDs, dimensionless qualities, complete ranking,
        modeled source seconds and selected/unselected counts. Canonical JSON
        converts tuples to arrays before computing the existing domain fingerprint.
    """
    return {
        trial.trial_id: [
            (
                row.source_build_id,
                row.napfd,
                row.apfdc,
                row.ranking,
                row.modeled_full_seconds,
                row.tests_ran,
                row.tests_not_ran,
            )
            for row in trial.rows
        ]
        for trial in report.trials
    }


job = wait(
    service.execute(ExecutePlan(schema_version=1, request_id="installed-library-original", plan_id=plan.plan_id))
)
reference = service.report(job.job_id)
assert len(reference.trials) == 8 and sum(len(trial.rows) for trial in reference.trials) == 64
assert all(sum(len(row.ranking) for row in trial.rows) == 6567 for trial in reference.trials)
print("INSTALLED LIBRARY VERIFIED", job.job_id, flush=True)
trace = []


async def sdk():
    """Reconnect actual stdio clients and verify counts, provenance and full artifact bytes.

    Returns
    -------
    dict
        Verified SDK/protocol, mathematical digest and bounded-response accounting.
        The private output directory retains complete downloaded evidence; no model,
        source acquisition or scientific source admission is performed.
    """
    native_threads = service.saved_plan(plan.plan_id).native_threads
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "coleman.transports.mcp", "--workspace", str(service.store.root)],
        cwd=str(root),
        env={"POLARS_MAX_THREADS": str(native_threads)},
    )
    async with Client(params) as client:
        protocol = client.protocol_version
        assert len((await client.list_tools()).tools) == 21
        planned = await client.call_tool("plan_experiment", {"request": request.model_dump(mode="json")})
        assert planned.structured_content["error"] is None, planned
        assert planned.structured_content["data"]["plan_id"] == plan.plan_id
        response = await client.call_tool(
            "execute_plan",
            {"request": {"schema_version": 1, "request_id": "installed-mcp-original", "plan_id": plan.plan_id}},
        )
        assert response.structured_content["error"] is None, response
        job_id = response.structured_content["data"]["job_id"]
        trace.append({"operation": "execute_plan", "job_id": job_id})
    async with Client(params) as client:
        for _ in range(6000):
            response = await client.call_tool("experiment_status", {"request": {"schema_version": 1, "job_id": job_id}})
            data = response.structured_content["data"]
            if data["state"] == "completed":
                break
            assert data["state"] in ("queued", "running"), data
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("Reconnected original-source MCP job did not complete")
        overview = (
            await client.call_tool("experiment_report", {"request": {"schema_version": 1, "job_id": job_id}})
        ).structured_content
        assert overview["error"] is None, overview
        assert (overview["data"]["trial_count"], overview["data"]["native_build_rows"]) == (8, 64)
        trace.append({"operation": "experiment_report", "data": overview["data"]})
        for offset in range(8):
            page = (
                await client.call_tool(
                    "inspect_experiment",
                    {"request": {"schema_version": 1, "job_id": job_id, "offset": offset, "limit": 1}},
                )
            ).structured_content
            assert page["error"] is None, page
            assert len(page["data"]["trials"]) == 1
            assert len(page["data"]["trials"][0]["rows"]) == 8
            assert len(json.dumps(page).encode()) < 524288
        descriptors = []
        offset = 0
        while offset is not None:
            response = (
                await client.call_tool(
                    "list_experiment_artifacts",
                    {"request": {"schema_version": 1, "job_id": job_id, "offset": offset, "limit": 32}},
                )
            ).structured_content
            assert response["error"] is None, response
            descriptors.extend(response["data"]["artifacts"])
            offset = response["data"]["next_offset"]
        assert len(descriptors) == 67
        assert any(item["kind"] == "source_receipt" for item in descriptors)
        retained = {}
        for descriptor in [item for item in descriptors if item["kind"] in ("plan", "report", "source_receipt")]:
            offset, parts = 0, []
            while offset is not None:
                response = (
                    await client.call_tool(
                        "read_experiment_artifact",
                        {
                            "request": {
                                "schema_version": 1,
                                "job_id": job_id,
                                "artifact_id": descriptor["artifact_id"],
                                "offset": offset,
                                "maximum_bytes": 65536,
                            }
                        },
                    )
                ).structured_content
                assert response["error"] is None, response
                data = response["data"]
                part = base64.b64decode(data["content_base64"])
                assert data["chunk_sha256"] == "sha256:" + hashlib.sha256(part).hexdigest()
                parts.append(part)
                offset = data["next_offset"]
            content = b"".join(parts)
            assert len(content) == descriptor["size_bytes"]
            assert "sha256:" + hashlib.sha256(content).hexdigest() == descriptor["sha256"]
            retained[descriptor["kind"]] = content
            (root / ("downloaded-" + descriptor["name"])).write_bytes(content)
        actual = WorkflowReport.from_json(retained["report"])
        assert mathematics(actual) == mathematics(reference)
        assert retained["source_receipt"] == registered.receipt.to_json().encode()
        return {
            "protocol": protocol,
            "mcp_sdk": version("mcp"),
            "native_threads": native_threads,
            "mcp_job_id": job_id,
            "math_digest": fingerprint(json.loads(json.dumps(mathematics(actual)))),
            "artifact_count": len(descriptors),
            "report_bytes": len(retained["report"]),
            "largest_inspection_bytes": len(json.dumps(page).encode()),
        }


proof = asyncio.run(sdk())
proof.update(
    {
        "schema_version": 1,
        "installed_package": str(Path(coleman.__file__).resolve()),
        "source": registered.model_dump(mode="json"),
        "plan_id": plan.plan_id,
        "configurations": 4,
        "trials": 8,
        "native_build_rows": 64,
        "library_job_id": job.job_id,
        "interpretation": "descriptive_only",
        "generalization": "insufficient_evidence",
        "trace": trace,
    }
)
(root / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
print(
    "ORIGINAL-SOURCE INSTALLED MCP/LIBRARY PARITY VERIFIED",
    json.dumps({k: proof[k] for k in ("protocol", "mcp_sdk", "math_digest", "artifact_count", "report_bytes")}),
    flush=True,
)
