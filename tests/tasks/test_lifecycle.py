"""Real common metadata, ownership and all-family resource admission regressions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier

import pytest

from coleman.application import execution_lifecycle as lifecycle
from coleman.application.baseline_lock import baseline_lock
from coleman.application.execution_views import changed
from coleman.application.tasks import TaskService
from coleman.application.workflow_store import JobRecord, WorkflowStore
from coleman.application.workflows import WorkflowService
from coleman.tasks.contracts import TaskJobRecord
from coleman.tasks.requests import TaskExecuteRequest
from coleman.workflows.contracts import WorkflowError
from coleman.workflows.lifecycle import ExecutePlan, HostPolicy
from tests.tasks.test_storage_records import _native, _plans, _task


def _retain(store, plan, record):
    lifecycle.save_plan(store, plan, record.owner)
    return lifecycle.admit_record(store, plan, record)


def test_simultaneous_identical_admission_has_one_actual_record_and_slot(tmp_path):
    """Deduplicate competing real SQLite admissions into one retained job."""
    _, plan = _plans()
    store = WorkflowStore(tmp_path)
    lifecycle.save_plan(store, plan, "owner")
    barrier = Barrier(2)

    def admit(identity):
        proposed = _task(plan, job_id=identity)
        barrier.wait(timeout=5)
        return lifecycle.admit_record(store, plan, proposed)

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = tuple(pool.map(admit, ("first", "second")))
    assert records[0] == records[1]
    with closing(store.connect()) as connection:
        assert store.records(connection) == (records[0],)
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
    assert lifecycle.retained_request(store, "owner", "task-request") == records[0]


def test_cross_family_and_changed_payload_keys_conflict_before_new_admission(tmp_path):
    """Keep a caller key bound to its original payload and execution family."""
    native, task = _plans()
    store = WorkflowStore(tmp_path)
    original = _retain(store, native, _native(native, request_id="same"))
    lifecycle.save_plan(store, task, original.owner)
    for candidate in (
        _task(task, request_id="same"),
        original.model_copy(update={"job_id": "another", "request_digest": "sha256:" + "0" * 64}),
    ):
        selected_plan = task if isinstance(candidate, TaskJobRecord) else native
        with pytest.raises(WorkflowError, match="Idempotency"):
            lifecycle.admit_record(store, selected_plan, candidate)
    with closing(store.connect()) as connection:
        assert store.records(connection) == (original,)


@pytest.mark.parametrize("first_family", ("native", "task"))
@pytest.mark.parametrize("capacity", ("queue", "job", "output"))
def test_admission_pools_include_other_family_and_other_owner(tmp_path, first_family, capacity):
    """Count both families and all owners in each shared capacity ceiling."""
    native, task = _plans()
    options = (
        {"max_queued": 0}
        if capacity == "queue"
        else {"max_jobs": 1}
        if capacity == "job"
        else {"max_retained_bytes": native.request.limits.max_output_bytes + task.limits.max_output_bytes - 1}
    )
    store = WorkflowStore(tmp_path, HostPolicy.model_validate({"schema_version": 1, **options}))
    first_plan, second_plan = (native, task) if first_family == "native" else (task, native)
    first = _native(native, owner="first") if first_family == "native" else _task(task, owner="first")
    second = _task(task, owner="second") if first_family == "native" else _native(native, owner="second")
    _retain(store, first_plan, first)
    lifecycle.save_plan(store, second_plan, second.owner)
    with pytest.raises(WorkflowError, match="capacity|slots|reservation"):
        lifecycle.admit_record(store, second_plan, second)
    with closing(store.connect()) as connection:
        assert len(store.records(connection)) == 1


def test_cancelled_actual_kernel_owner_retains_queue_slot_until_release(tmp_path):
    """Do not release a still kernel-owned worker's slot on cancellation."""
    native, task = _plans()
    store = WorkflowStore(tmp_path, HostPolicy(schema_version=1, max_queued=0))
    first = _retain(store, native, _native(native))
    lifecycle.cancel_record(store, first.job_id, first.owner)
    lifecycle.save_plan(store, task, "owner")
    with baseline_lock(tmp_path / "locks" / f"{first.job_id}.lock"), pytest.raises(WorkflowError, match="capacity"):
        lifecycle.admit_record(store, task, _task(task))
    second = lifecycle.admit_record(store, task, _task(task))
    assert second.state == "queued"


def test_expired_actual_bytes_keep_output_reservation_until_successful_cleanup(tmp_path):
    """Retain expired output capacity until actual deletion succeeds."""
    native, task = _plans()
    policy = HostPolicy(
        schema_version=1,
        max_retained_bytes=native.request.limits.max_output_bytes + task.limits.max_output_bytes - 1,
    )
    store = WorkflowStore(tmp_path, policy)
    first = _retain(store, native, _native(native))
    with store.transaction() as connection:
        store.save_record(connection, changed(first, state="expired"))
    directory = store.root / "jobs" / first.job_id
    directory.mkdir(parents=True)
    (directory / "retained.bin").write_bytes(b"retained")
    lifecycle.save_plan(store, task, "owner")
    with pytest.raises(WorkflowError, match="reservation"):
        lifecycle.admit_record(store, task, _task(task))
    assert lifecycle.cleanup_records(store, "owner", JobRecord) == (first.job_id,)
    assert not directory.exists()
    assert lifecycle.admit_record(store, task, _task(task)).state == "queued"
    with closing(store.connect()) as connection:
        assert len(store.records(connection)) == 2  # Expiry never recycles the old key/slot.


def test_one_worker_cannot_exceed_shared_active_memory_capacity(tmp_path):
    """Refuse a single task that cannot fit the common memory pool."""
    _, task = _plans()
    store = WorkflowStore(tmp_path, HostPolicy(schema_version=1, max_active_memory_bytes=67108864))
    lifecycle.save_plan(store, task, "owner")
    with pytest.raises(WorkflowError, match="aggregate active memory"):
        lifecycle.admit_record(store, task, _task(task))
    with closing(store.connect()) as connection:
        records = store.records(connection)
        assert isinstance(records, tuple) and not records


def test_native_facade_projects_only_native_but_rejects_generic_handle(tmp_path):
    """Share metadata while keeping caller and native/task facade boundaries."""
    native, task = _plans()
    native_service = WorkflowService(tmp_path, owner="owner")
    first = _retain(native_service.store, native, _native(native))
    second = _retain(native_service.store, task, _task(task))
    assert native_service.status(first.job_id).job_id == first.job_id
    with pytest.raises(WorkflowError, match="native workflow job"):
        native_service.status(second.job_id)
    task_service = TaskService(tmp_path, owner="owner")
    assert task_service.status(second.job_id).job == second
    with pytest.raises(WorkflowError, match="task job"):
        task_service.status(first.job_id)
    with pytest.raises(KeyError):
        TaskService(tmp_path, owner="another").status(second.job_id)
    # No scheduling call here: these fixtures qualify metadata, not actual providers.


def test_both_facades_reject_retained_key_from_the_other_family_before_bootstrap(tmp_path):
    """Reject cross-facade idempotency conflicts without loading a provider."""
    native, task = _plans()
    store = WorkflowStore(tmp_path)
    _retain(store, native, _native(native, request_id="native-key"))
    _retain(store, task, _task(task, request_id="task-key"))
    with pytest.raises(WorkflowError, match="Idempotency"):
        WorkflowService(tmp_path, owner="owner").execute(
            ExecutePlan(schema_version=1, request_id="task-key", plan_id=native.fingerprint[7:])
        )
    with pytest.raises(WorkflowError, match="Idempotency"):
        TaskService(tmp_path, owner="owner").execute(
            TaskExecuteRequest(
                schema_version=1,
                contract="coleman-tasks-v1",
                artifact_type="execute_request",
                request_id="native-key",
                plan_id=task.fingerprint[7:],
            )
        )


def test_outer_dead_worker_reconciles_without_provider_and_cancel_is_idempotent(tmp_path):
    """Record lost actual ownership without loading code or restarting work."""
    _, plan = _plans()
    service = TaskService(tmp_path, owner="owner")
    job = _retain(service.store, plan, _task(plan))
    with service.store.transaction() as connection:
        service.store.save_record(connection, changed(job, state="running", invocations=1))
    result = service.status(job.job_id)
    assert result.job.state == "interrupted"
    assert result.job.error == "worker_ownership_lost"
    assert not result.worker_active
    assert result.eta_seconds is None and result.eta_status == "unavailable"
    assert service.cancel(job.job_id).job == result.job  # Already terminal.
    with service.store.transaction() as connection:
        service.store.save_record(connection, changed(result.job, state="queued", error=None))
    assert service.cancel(job.job_id).job.state == "cancelled"
    assert service.cancel(job.job_id).job.state == "cancelled"


def test_requeue_keeps_original_progress_and_refuses_live_cancelled_owner(tmp_path):
    """Preserve accepted outer frontier through an explicitly requeued job."""
    _, plan = _plans()
    store = WorkflowStore(tmp_path)
    job = _retain(store, plan, _task(plan))
    coordinate = plan.coordinates[0]
    interrupted = changed(
        job,
        state="interrupted",
        current_coordinate=coordinate.fingerprint,
        frontier=coordinate.target,
        invocations=1,
        attempts=1,
    )
    assert isinstance(interrupted, TaskJobRecord)
    with store.transaction() as connection:
        store.save_record(connection, interrupted)
    queued = lifecycle.requeue_record(store, interrupted, plan)
    assert isinstance(queued, TaskJobRecord)
    assert queued.current_coordinate == interrupted.current_coordinate
    assert queued.frontier == interrupted.frontier
    assert (queued.attempts, queued.invocations) == (1, 1)
    assert lifecycle.requeue_record(store, queued, plan) == queued
    cancelled = changed(queued, state="cancelled")
    with store.transaction() as connection:
        store.save_record(connection, cancelled)
    with (
        baseline_lock(tmp_path / "locks" / f"{job.job_id}.lock"),
        pytest.raises(WorkflowError, match="active cancelled"),
    ):
        lifecycle.requeue_record(store, cancelled, plan)
