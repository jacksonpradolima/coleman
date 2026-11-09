"""New native execution requires its selected closure; legacy evidence stays readable."""

import copy
import json
from contextlib import closing

import pytest

from coleman.application.baseline_lock import baseline_lock
from coleman.application.workflow_registry import owned_fixture
from coleman.application.workflow_store import JobRecord
from coleman.application.workflow_worker import run_job
from coleman.application.workflows import WorkflowService
from coleman.artifacts import ArtifactWriter
from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.checkpoint.state import CheckpointPayload
from coleman.policy import base as policy_base
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.selected_runtime import SelectedInstalledRuntime, capture_native_runtime
from coleman.workflows.contracts import ExperimentPlan, PlanningRequest
from coleman.workflows.lifecycle import ExecutePlan


def service_and_plan(root, *, protected=False):
    """Use actual native source registration/planning with one seed and bounded coordinates."""
    service = WorkflowService(root)
    manifest = service.register(owned_fixture(root))
    preview = service.plan(
        PlanningRequest(
            schema_version=1,
            dataset_id=manifest.dataset_id,
            policies=("Random", "UCB") if protected else ("Random",),
            seeds=(42,),
            ucb_exploration=(1.0,) if protected else (0.5, 1.0, 2.0),
            evaluation="protected" if protected else "descriptive",
        )
    )
    return service, preview


def changed_runtime(monkeypatch):
    """Change only the recorded installed boundary, leaving admission and workers real."""
    current = capture_native_runtime()
    different = SelectedInstalledRuntime.model_validate({**current.model_dump(), "machine": "synthetic-other-machine"})
    monkeypatch.setattr(SelectedInstalledRuntime, "capture", lambda *, extras: different)


def update_job(service, job_id, **updates):
    """Represent a stopped/claimed host state through the real validated metadata transaction."""
    with service.store.transaction() as connection:
        record = service._job(connection, job_id)
        value = JobRecord.model_validate({**record.model_dump(), **updates})
        service.store.save_job(connection, value)
    return value


def test_historical_runtime_plan_is_readable_but_cannot_admit_native_execution(tmp_path):
    service, preview = service_and_plan(tmp_path)
    current = service.saved_plan(preview.plan_id)
    historical = ExperimentPlan.model_validate({**current.model_dump(), "runtime": InstalledRuntime.capture()})
    target = tmp_path / "plans" / f"{historical.fingerprint[7:]}.json"
    ArtifactWriter(tmp_path).write_immutable(target, historical.to_json())
    with service.store.transaction() as connection:
        connection.execute(
            "INSERT INTO plans VALUES (?,?,?)", (historical.fingerprint[7:], service.owner, historical.to_json())
        )
    assert service.saved_plan(historical.fingerprint[7:]) == historical
    request = ExecutePlan(schema_version=1, request_id="historical-read-only", plan_id=historical.fingerprint[7:])
    with pytest.raises(ValueError, match="pinned v4"):
        service.execute(request)
    with closing(service.store.connect()) as connection:
        assert service.store.jobs(connection) == ()
    assert target.read_text() == historical.to_json()


@pytest.mark.parametrize("operation", ["resume", "evaluate"])
def test_changed_runtime_cannot_requeue_or_freeze_evaluation(tmp_path, monkeypatch, operation):
    service, preview = service_and_plan(tmp_path, protected=operation == "evaluate")
    monkeypatch.setattr(service, "_start", lambda: None)
    job = service.execute(ExecutePlan(schema_version=1, request_id="runtime-guard", plan_id=preview.plan_id))
    stopped = update_job(service, job.job_id, state="awaiting_evaluation" if operation == "evaluate" else "interrupted")
    changed_runtime(monkeypatch)
    with pytest.raises(ValueError, match="incompatible"):
        getattr(service, operation)(job.job_id)
    with closing(service.store.connect()) as connection:
        retained = service._job(connection, job.job_id)
    assert retained == stopped
    assert not (tmp_path / "jobs" / job.job_id / "evaluation-handoff.json").exists()


def test_worker_runtime_recheck_precedes_checkpoint_loading_and_rng_restoration(tmp_path, monkeypatch):
    # Declare a positive 1 MiB sample for this runtime-order unit test.
    monkeypatch.setattr("coleman.application.workflow_worker._worker_peak_rss_mib", lambda: 1.0)
    service, preview = service_and_plan(tmp_path)
    monkeypatch.setattr(service, "_start", lambda: None)
    job = service.execute(ExecutePlan(schema_version=1, request_id="worker-runtime", plan_id=preview.plan_id))
    update_job(service, job.job_id, state="running", invocations=1)
    checkpoint_root = tmp_path / "jobs" / job.job_id / "trusted-checkpoint"
    LocalCheckpointStore(str(checkpoint_root)).save(CheckpointPayload(run_id="owned", experiment=1, step=1))
    before = copy.deepcopy(policy_base._rng.bit_generator.state)
    loads = []

    def forbidden_load(*args):
        loads.append(args)
        raise AssertionError("Incompatible runtime must reject before pickle loading")

    monkeypatch.setattr(LocalCheckpointStore, "load", forbidden_load)
    changed_runtime(monkeypatch)
    with baseline_lock(tmp_path / "locks" / f"{job.job_id}.lock"):
        run_job(tmp_path, job.job_id)
    assert loads == []
    assert policy_base._rng.bit_generator.state == before
    with closing(service.store.connect()) as connection:
        rejected = service._job(connection, job.job_id)
    assert rejected.state == "failed"
    assert rejected.error == "native_workflow_ValueError"
    assert rejected.attempts == 0
    failure = json.loads((tmp_path / "jobs" / job.job_id / "failure-0001.json").read_bytes())
    assert failure["exception"] == "ValueError"
    assert "incompatible" in failure["detail"]
    assert not (tmp_path / "jobs" / job.job_id / "complete.json").exists()
