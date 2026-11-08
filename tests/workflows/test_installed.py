"""Noneditable CI-capability wheel and actual optional stdio workflows outside the checkout."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def test_installed_wheel_without_model_and_optional_sdk(tmp_path):
    pytest.importorskip("mcp")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {**os.environ, "UV_OFFLINE": "1", "UV_NO_PROGRESS": "1", "UV_LINK_MODE": "copy"}

    def command(arguments, cwd=tmp_path, extra_environment=None):
        result = subprocess.run(
            arguments,
            cwd=cwd,
            env={**environment, **(extra_environment or {})},
            text=True,
            capture_output=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        return result

    distributions = tmp_path / "dist"
    command(
        [uv, "build", "--wheel", "--no-build-isolation", "--python", sys.executable, "--out-dir", str(distributions)],
        ROOT,
    )
    wheel = next(distributions.glob("*.whl"))
    virtual = tmp_path / "installed"
    command([uv, "venv", str(virtual), "--python", sys.executable])
    python = virtual / "bin/python"

    def dependencies(*extras):
        command(
            [
                uv,
                "sync",
                "--frozen",
                "--offline",
                "--no-default-groups",
                "--no-dev",
                "--no-install-project",
                "--inexact",
                "--python",
                str(python),
                *extras,
            ],
            ROOT,
            {"UV_PROJECT_ENVIRONMENT": str(virtual)},
        )

    dependencies("--extra", "ci")
    command([uv, "pip", "install", "--no-deps", "--python", str(python), str(wheel) + "[ci]"])
    program = tmp_path / "qualified.py"
    program.write_text("""
import asyncio,importlib.util,json,sqlite3,sys,time
from contextlib import closing
from pathlib import Path
import coleman
from coleman.application.workflow_store import _sqlite_command
from coleman.application.workflows import WorkflowService
from coleman.application.workflow_registry import owned_fixture
from coleman.workflows.contracts import PlanningRequest
from coleman.workflows.lifecycle import ExecutePlan
assert Path(coleman.__file__).resolve().is_relative_to(Path.cwd()/"installed")
assert not Path.cwd().is_relative_to(Path(coleman.__file__).resolve().parent)
service=WorkflowService(Path("workspace"))
service.register(owned_fixture(service.store.root))
plan=service.plan(PlanningRequest(schema_version=1,dataset_id="owned-ci-three-build-v1"))
def wait(job,busy_observed=None):
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        try:
            view=service.status(job.job_id)
        except sqlite3.OperationalError as error:
            code=getattr(error,"sqlite_errorcode",None)
            if not isinstance(code,int) or code&0xff not in (sqlite3.SQLITE_BUSY,sqlite3.SQLITE_LOCKED):
                raise
            if busy_observed is not None:busy_observed.set()
        else:
            if view.state=="completed" and not view.worker_active:return view
            assert view.state in ("queued","running","completed"),view
        remaining=deadline-time.monotonic()
        if remaining>0:time.sleep(min(.05,remaining))
    raise AssertionError("Installed native job did not complete")
if sys.argv[1]=="base":
    assert importlib.util.find_spec("mcp") is None
    job=wait(service.execute(ExecutePlan(schema_version=1,request_id="installed-library",plan_id=plan.plan_id)))
    report=service.report(job.job_id)
    assert len(report.trials)==8 and sum(len(trial.rows) for trial in report.trials)==24
    # A completed native job isolates observation contention from execution.
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    acquired,release,busy_observed=Event(),Event(),Event()
    def hold_writer():
        # A normal WAL writer permits read-only status. This connection-local
        # exclusive fault genuinely blocks readers without changing any budget.
        deadline=time.monotonic()+10
        with closing(service.store.connect(deadline=deadline)) as connection:
            assert connection.execute("PRAGMA locking_mode=EXCLUSIVE").fetchone()[0]=="exclusive"
            _sqlite_command(connection,lambda:connection.execute("BEGIN IMMEDIATE"),deadline=deadline)
            try:
                acquired.set()
                assert release.wait(15),"Coordinated writer was not released"
            finally:
                connection.rollback()
    with ThreadPoolExecutor(max_workers=2) as pool:
        holder=pool.submit(hold_writer)
        try:
            if not acquired.wait(10):
                holder.result(timeout=1)
                raise AssertionError("Real writer did not acquire BEGIN IMMEDIATE")
            observer=pool.submit(wait,job,busy_observed)
            if not busy_observed.wait(5):
                if observer.done():observer.result()
                raise AssertionError("Status did not encounter actual numeric contention")
        finally:
            release.set()
        holder.result(timeout=5)
        observed=observer.result(timeout=5)
    assert observed.job_id==job.job_id and observed.result_digest==job.result_digest
    assert (observed.attempts,observed.invocations,observed.completed_trials)==(
        job.attempts,job.invocations,job.completed_trials)
    assert service.report(observed.job_id).to_json()==report.to_json()
    values={trial.trial_id:[(row.source_build_id,row.napfd,row.apfdc,row.ranking)
        for row in trial.rows] for trial in report.trials}
    Path("mathematics.json").write_text(json.dumps(values))
else:
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters
    async def sdk():
        params=StdioServerParameters(command=sys.executable,
            args=["-m","coleman.transports.mcp","--workspace",str(service.store.root)],cwd=str(Path.cwd()))
        async with Client(params) as client:
            assert len((await client.list_tools()).tools)==21
            print("Negotiated protocol",client.protocol_version)
            response=await client.call_tool("plan_experiment",{"request":{
                "schema_version":1,"dataset_id":"owned-ci-three-build-v1"}})
            assert response.structured_content["data"]["plan_id"]==plan.plan_id
            response=await client.call_tool("execute_plan",{"request":{
                "schema_version":1,"plan_id":plan.plan_id,"request_id":"installed-mcp"}})
            job_id=response.structured_content["data"]["job_id"]
        async with Client(params) as client:
            for _ in range(600):
                response=await client.call_tool("experiment_status",{"request":{"schema_version":1,"job_id":job_id}})
                view=response.structured_content["data"]
                if view["state"]=="completed":break
                assert view["state"] in ("queued","running"),view
                await asyncio.sleep(.05)
            response=await client.call_tool("experiment_report",{"request":{"schema_version":1,"job_id":job_id}})
            assert response.structured_content["data"]["trial_count"]==8
            files=await client.call_tool("list_experiment_artifacts",{"request":{
                "schema_version":1,"job_id":job_id,"limit":2}})
            artifact=next(item for item in files.structured_content["data"]["artifacts"] if item["kind"]=="report")
            import base64,hashlib
            offset,parts=0,[]
            while offset is not None:
                chunk=await client.call_tool("read_experiment_artifact",{"request":{
                    "schema_version":1,"job_id":job_id,"artifact_id":artifact["artifact_id"],"offset":offset}})
                data=chunk.structured_content["data"]
                parts.append(base64.b64decode(data["content_base64"]))
                offset=data["next_offset"]
            content=b"".join(parts)
            assert "sha256:"+hashlib.sha256(content).hexdigest()==artifact["sha256"]
            trials=json.loads(content)["trials"]
            actual={trial["trial_id"]:[(row["source_build_id"],row["napfd"],row["apfdc"],row["ranking"])
                for row in trial["rows"]] for trial in trials}
            assert json.loads(json.dumps(actual))==json.loads(Path("mathematics.json").read_text())
            response=await client.call_tool("export_experiment",{"request":{"schema_version":1,"job_id":job_id}})
            assert response.structured_content["error"] is None
    asyncio.run(sdk())
print("INSTALLED WORKFLOW VERIFIED",sys.argv[1])
""")
    assert "INSTALLED WORKFLOW VERIFIED base" in command([str(python), str(program), "base"]).stdout
    dependencies("--extra", "ci", "--extra", "mcp")
    assert "INSTALLED WORKFLOW VERIFIED mcp" in command([str(python), str(program), "mcp"]).stdout
