"""Real native workflow execution, portable evidence, state handoff and durable recovery."""

import asyncio
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from coleman.application.baseline_lock import baseline_lock
from coleman.application.workflow_exports import EvidenceBundle
from coleman.application.workflow_registry import owned_fixture
from coleman.application.workflow_store import JobRecord
from coleman.application.workflows import WorkflowService
from coleman.corpus.baseline_contracts import FAILURE_MEANING, BaselineBuild, BaselineTest
from coleman.corpus.baseline_reference import baseline_reference
from coleman.pilot.contracts import SECONDS
from coleman.results.domain import Measurement
from coleman.workflows.contracts import PlanningRequest, WorkflowLimits
from coleman.workflows.lifecycle import ExecutePlan, HostPolicy, TrialResult, TrialSummary


def setup_workflow(root, **choices):
    service = WorkflowService(root)
    service.register(owned_fixture(root))
    preview = service.plan(PlanningRequest(schema_version=1, dataset_id="owned-ci-three-build-v1", **choices))
    return service, preview


def wait(service, job_id, expected="completed"):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        view = service.status(job_id)
        if view.state not in ("queued", "running") and not view.worker_active:
            assert view.state == expected, [path.read_text() for path in service.store.root.rglob("failure-*.json")]
            return view
        time.sleep(0.025)
    pytest.fail("Durable workflow did not reach its expected terminal boundary")


def mathematics(report):
    return {
        trial.trial_id: tuple((row.source_build_id, row.napfd, row.apfdc, row.ranking) for row in trial.rows)
        for trial in report.trials
    }


def archived_handoff(service, preview, root, monkeypatch):
    """Create genuine archived prefix evidence; the native engine and later clocks remain real."""
    from datetime import UTC, datetime

    import coleman.application.workflow_evaluation as evaluation

    plan = service.saved_plan(preview.plan_id)
    assert plan.dataset.temporal_plan is not None
    archived = plan.dataset.temporal_plan.test_end.timestamp() + 1

    class ArchivedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(archived, tz)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "datetime", ArchivedClock)
        handoff = evaluation.prepare_evaluation(service.store.root, root, plan)
    assert handoff.design.plan.test_end < datetime.now(UTC)
    return handoff


def assert_independent_fixture_metrics(report):
    """Use the independent rational/binary32 reference, never the production metric class."""
    for result in report.trials:
        for row in result.rows:
            tests = tuple(
                BaselineTest(
                    test_id=name,
                    duration=Measurement(
                        name="source_duration",
                        value=duration,
                        unit="s",
                        meaning=SECONDS.meaning,
                        status="observed",
                        direction="minimize",
                        minimum=0.0,
                    ),
                    failure=Measurement(
                        name="source_failure",
                        value=float(row.source_build_id % 2 if name == "integration" else 0),
                        unit="1",
                        meaning=FAILURE_MEANING,
                        status="observed",
                        direction="none",
                        minimum=0.0,
                    ),
                )
                for name, duration in (("unit", 1.0), ("integration", 2.0), ("system", 4.0))
            )
            source = BaselineBuild(
                step=row.step,
                member_id=f"build-{row.source_build_id}",
                source_metadata=None,
                decided_at=None,
                tests=tests,
                original_order=None,
            )
            reference = baseline_reference(source, row.ranking, 3.5)
            assert row.napfd == pytest.approx(1.0 if reference.napfd is None else reference.napfd)
            assert row.apfdc == pytest.approx(1.0 if reference.apfdc is None else reference.apfdc)
            assert row.tests_ran == len(reference.selected)


def test_actual_eight_trials_export_import_and_two_coordinators(tmp_path):
    service, plan = setup_workflow(tmp_path)
    other = WorkflowService(tmp_path)
    request = ExecutePlan(schema_version=1, request_id="same", plan_id=plan.plan_id)
    with ThreadPoolExecutor(2) as pool:
        views = list(pool.map(lambda host: host.execute(request), (service, other)))
    assert views[0].job_id == views[1].job_id
    service.close()
    other.reconnect()
    view = wait(other, views[0].job_id)
    assert view.completed_trials == view.total_trials == view.attempts == 8
    report = other.report(view.job_id)
    assert len(report.trials) == 8
    assert sum(len(result.rows) for result in report.trials) == 24
    assert all(row.tests_ran + row.tests_not_ran == 3 for result in report.trials for row in result.rows)
    assert report.interpretation == "descriptive_only"
    assert_independent_fixture_metrics(report)
    assert report.selected_configuration is not None
    assert other.execute(request).job_id == view.job_id
    assert other.resume(view.job_id).result_digest == view.result_digest
    pages = other.inspect(view.job_id, limit=3)
    assert len(pages.trials) == 3
    assert pages.next_offset == 3
    bundle = other.export(view.job_id)
    assert EvidenceBundle.from_json(bundle.to_json()) == bundle
    imported = other.import_evidence(bundle.to_json())
    assert other.import_evidence(bundle.to_json()) == imported
    assert mathematics(other.imported_evidence(imported.evidence_id).report) == mathematics(report)
    with pytest.raises(KeyError):
        other.resume(imported.evidence_id)
    foreign = WorkflowService(tmp_path, owner="another-principal")
    with pytest.raises(KeyError):
        foreign.report(view.job_id)
    with pytest.raises(KeyError):
        foreign.imported_evidence(imported.evidence_id)
    # The same mathematical plan is independently ownable; its ID is not a secret.
    assert foreign.plan(service.saved_plan(plan.plan_id).request) == plan
    alternate = other.plan(PlanningRequest(schema_version=1, dataset_id="owned-ci-three-build-v1", seeds=(0,)))
    conflicting_request = ExecutePlan(schema_version=1, request_id="same", plan_id=alternate.plan_id)
    with pytest.raises(ValueError, match="conflicts"):
        other.execute(conflicting_request)
    bad = json.loads(bundle.to_json())
    bad["report"]["trials"][0]["rows"][0]["napfd"] = 0.12345
    changed_evidence_json = json.dumps(bad)
    with pytest.raises(ValueError, match="native bytes"):
        other.import_evidence(changed_evidence_json)
    first = report.trials[0].files[0]
    path = tmp_path / "jobs" / view.job_id / first.name
    path.write_bytes(path.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="bytes changed"):
        other.report(view.job_id)


def test_actual_protected_prefix_audit_and_learning_continuation(tmp_path):
    service, preview = setup_workflow(tmp_path, evaluation="protected", limits=WorkflowLimits(max_attempts=2))
    job = service.execute(ExecutePlan(schema_version=1, request_id="protected", plan_id=preview.plan_id))
    wait(service, job.job_id, "awaiting_evaluation")
    prefix = service.inspect(job.job_id, limit=8)
    assert sum(len(trial.rows) for trial in prefix.trials) == 16
    assert all(tuple(row.step for row in trial.rows) == (1, 2) for trial in prefix.trials)
    assert not list((tmp_path / "audit").rglob("evaluation-access/*/*.json"))
    assert not (tmp_path / "jobs" / job.job_id / "complete.json").exists()
    service.evaluate(job.job_id)
    wait(service, job.job_id)
    report = service.report(job.job_id)
    assert report.interpretation == "insufficient_evidence"
    assert_independent_fixture_metrics(report)
    assert report.fallback_configuration is not None
    assert report.comparison is not None
    assert report.comparison.results[0].reason == "too_few_independent_units"
    assert report.comparison.results[0].observed_pairs == 1
    assert len(report.trials) == 4
    assert sum(len(trial.rows) for trial in report.trials) == 12
    planned = service.saved_plan(preview.plan_id)
    completed_ids = {receipt.trial_id for receipt in report.trials}
    compared = {trial.configuration_id for trial in planned.trials if trial.trial_id in completed_ids}
    unrelated = next(trial.configuration_id for trial in planned.trials if trial.configuration_id not in compared)
    portable = service.export(job.job_id).model_dump(mode="json")
    portable["report"]["selected_configuration"] = unrelated
    with pytest.raises(ValueError, match="frozen candidate family"):
        EvidenceBundle.model_validate(portable)
    prefix_rows = {trial.trial_id: trial.rows for trial in prefix.trials}
    assert all(TrialSummary.from_result(trial).rows[:2] == prefix_rows[trial.trial_id] for trial in report.trials)
    access = report.comparison.access
    assert access.prior_test_accesses == 0
    assert EvidenceBundle.from_json(service.export(job.job_id).to_json()).report == report
    handoff = json.loads((tmp_path / "jobs" / job.job_id / "evaluation-handoff.json").read_text())
    assert handoff["checkpoints"]
    assert all(
        "learning_state" in json.loads(candidate.configuration_json)
        for candidate in report.comparison.frozen.candidates
    )
    service.evaluate(job.job_id)
    assert service.report(job.job_id).comparison.access == access
    # A new experiment on the same retained source must not reset holdout exposure.
    second = service.execute(ExecutePlan(schema_version=1, request_id="protected-again", plan_id=preview.plan_id))
    wait(service, second.job_id, "awaiting_evaluation")
    service.evaluate(second.job_id)
    wait(service, second.job_id)
    assert service.report(second.job_id).comparison.access.prior_test_accesses == 1


_CRASH_AT_REAL_CHECKPOINT = """
import os,sys
from pathlib import Path
from coleman.application.workflow_worker import run_job
from coleman.application.workflow_store import WorkflowStore
root,job_id=Path(sys.argv[1]),sys.argv[2]
def checkpoint(trial,step):
    store=WorkflowStore(root)
    with store.transaction() as connection:
        job=next(j for j in store.jobs(connection) if j.job_id==job_id)
    if job.completed_trials==1 and step==1:
        os._exit(87)
run_job(root,job_id,progress_hook=checkpoint)
"""


_PAUSE_AT_REAL_CHECKPOINT = """
import sys,time
from pathlib import Path
from coleman.application.workflow_worker import run_job
root,job_id=Path(sys.argv[1]),sys.argv[2]
def checkpoint(trial,step):
    (root/'checkpoint-ready').write_text(trial)
    time.sleep(30)
run_job(root,job_id,progress_hook=checkpoint)
"""


def test_running_cancellation_preserves_checkpoint_and_concurrent_resume_identity(tmp_path, monkeypatch):
    service, preview = setup_workflow(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(service, "_start", lambda: None)
        job = service.execute(ExecutePlan(schema_version=1, request_id="cancel-running", plan_id=preview.plan_id))
    with baseline_lock(tmp_path / "locks" / f"{job.job_id}.lock", handoff=True) as descriptor:
        with service.store.transaction() as connection:
            record = service._job(connection, job.job_id)
            service.store.save_job(
                connection, JobRecord.model_validate({**record.model_dump(), "state": "running", "invocations": 1})
            )
        process = subprocess.Popen(
            [sys.executable, "-c", _PAUSE_AT_REAL_CHECKPOINT, str(tmp_path), job.job_id],
            pass_fds=(descriptor,),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "checkpoint-ready").exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.025)
        assert (tmp_path / "checkpoint-ready").exists()
        before = service.status(job.job_id)
        assert before.worker_active
        assert before.checkpoint_step == before.attempts == 1
        cancelled = service.cancel(job.job_id)
        assert cancelled.state == "cancelled"
        assert cancelled.result_digest is None
        _, error = process.communicate(timeout=10)
        assert process.returncode == 75, error.decode()
        assert service.status(job.job_id).state == "cancelled"
        assert not (tmp_path / "jobs" / job.job_id / "complete.json").exists()
        with ThreadPoolExecutor(2) as pool:
            views = list(pool.map(lambda host: host.resume(job.job_id), (service, WorkflowService(tmp_path))))
        assert {view.job_id for view in views} == {job.job_id}
        final = wait(service, job.job_id)
        assert final.attempts == 9
        assert final.invocations == 2
        report = service.report(job.job_id)
        assert len(report.trials) == 8
        assert sum(len(trial.rows) for trial in report.trials) == 24
        assert_independent_fixture_metrics(report)
        assert service.cancel(job.job_id).result_digest == final.result_digest
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_real_checkpoint_crash_then_actual_stdio_reconnection(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    service, preview = setup_workflow(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(service, "_start", lambda: None)
        job = service.execute(ExecutePlan(schema_version=1, request_id="recover", plan_id=preview.plan_id))
    with baseline_lock(tmp_path / "locks" / f"{job.job_id}.lock", handoff=True) as descriptor:
        with service.store.transaction() as connection:
            record = service._job(connection, job.job_id)
            service.store.save_job(
                connection, JobRecord.model_validate({**record.model_dump(), "state": "running", "invocations": 1})
            )
        process = subprocess.Popen(
            [sys.executable, "-c", _CRASH_AT_REAL_CHECKPOINT, str(tmp_path), job.job_id],
            pass_fds=(descriptor,),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    _, error = process.communicate(timeout=30)
    assert process.returncode == 87, error.decode()
    interrupted = service.status(job.job_id)
    assert interrupted.state == "interrupted"
    assert interrupted.completed_trials == 1
    assert interrupted.checkpoint_step == 1
    assert interrupted.attempts == 2
    assert interrupted.result_digest is None
    assert not (tmp_path / "jobs" / job.job_id / "complete.json").exists()
    summary = service.inspect(job.job_id).trials[0]
    receipt_path = tmp_path / "jobs" / job.job_id / "evidence" / summary.trial_id[7:] / "descriptive" / "complete.json"
    receipt = TrialResult.from_json(receipt_path.read_bytes())
    assert receipt.fingerprint == summary.receipt_digest
    before = {pin.name: (tmp_path / "jobs" / job.job_id / pin.name).read_bytes() for pin in receipt.files}
    service.close()

    async def reconnect():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "coleman.transports.mcp", "--workspace", str(tmp_path)],
            cwd=str(tmp_path),
        )
        async with Client(parameters) as client:
            tools = await client.list_tools()
            assert len(tools.tools) == 21
            schema = next(tool.input_schema for tool in tools.tools if tool.name == "plan_experiment")
            assert "PlanningRequest" in str(schema)
            response = await client.call_tool(
                "experiment_status", {"request": {"schema_version": 1, "job_id": job.job_id}}
            )
            data = response.structured_content
            assert data["data"]["state"] == "interrupted"
            await client.call_tool("resume_experiment", {"request": {"schema_version": 1, "job_id": job.job_id}})
        # Actual client/stdio server close while detached native execution continues.
        async with Client(parameters) as client:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                response = await client.call_tool(
                    "experiment_status", {"request": {"schema_version": 1, "job_id": job.job_id}}
                )
                if response.structured_content["data"]["state"] == "completed":
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("Reconnected MCP job did not complete")
            response = await client.call_tool(
                "experiment_report", {"request": {"schema_version": 1, "job_id": job.job_id}}
            )
            assert response.structured_content["data"]["interpretation"] == "descriptive_only"

    asyncio.run(reconnect())
    reopened = WorkflowService(tmp_path)
    final = wait(reopened, job.job_id)
    assert final.attempts == 9
    report = reopened.report(job.job_id)
    assert len(report.trials) == 8
    assert sum(len(trial.rows) for trial in report.trials) == 24
    assert next(trial for trial in report.trials if trial.trial_id == receipt.trial_id) == receipt
    assert all((tmp_path / "jobs" / job.job_id / name).read_bytes() == payload for name, payload in before.items())


def test_retention_keys_and_unknown_workspace_version(tmp_path, monkeypatch):
    policy = HostPolicy(schema_version=1, retention_seconds=0.001)
    service = WorkflowService(tmp_path, policy=policy)
    service.register(owned_fixture(tmp_path))
    plan = service.plan(
        PlanningRequest(schema_version=1, dataset_id="owned-ci-three-build-v1", seeds=(0,), policies=("Random",))
    )
    with monkeypatch.context() as patch:
        patch.setattr(service, "_start", lambda: None)
        request = ExecutePlan(schema_version=1, request_id="expired-key", plan_id=plan.plan_id)
        job = service.execute(request)
        service.cancel(job.job_id)
        time.sleep(0.01)
        assert service.cleanup() == (job.job_id,)
        assert service.execute(request).state == "expired"
    with pytest.raises(ValueError, match="Expired"):
        service.resume(job.job_id)
    with service.store.transaction() as connection:
        connection.execute("PRAGMA user_version=2")
    with pytest.raises(ValueError, match="schema"):
        WorkflowService(tmp_path)
