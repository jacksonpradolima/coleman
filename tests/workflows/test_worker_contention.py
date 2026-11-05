"""Bounded worker metadata contention preserves the sole actual native invocation."""

import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

from coleman.application.baseline_lock import baseline_lock
from coleman.application.workflow_store import JobRecord, WorkflowStore
from coleman.application.workflow_worker import run_job
from coleman.workflows.contracts import WorkflowError, WorkflowLimits
from coleman.workflows.lifecycle import ExecutePlan
from tests.workflows.test_execution import setup_workflow


def test_real_checkpoint_waits_for_writer_without_consuming_another_invocation(tmp_path):
    code = (
        "import sys; from pathlib import Path; "
        "from tests.workflows.test_worker_contention import _checkpoint_contention_case; "
        "_checkpoint_contention_case(Path(sys.argv[1]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr


def _checkpoint_contention_case(root):
    """Keep the real worker's resource accounting isolated from the accumulated pytest process."""
    service, preview = setup_workflow(
        root, policies=("Random",), seeds=(42,), limits=WorkflowLimits(max_attempts=1, attempt_seconds=30.0)
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(service, "_start", lambda: None)
        job = service.execute(ExecutePlan(schema_version=1, request_id="checkpoint-busy", plan_id=preview.plan_id))
    acquired, release = Event(), Event()

    def writer():
        with service.store.transaction():
            acquired.set()
            assert release.wait(timeout=1.5) is False

    with baseline_lock(root / "locks" / f"{job.job_id}.lock"), ThreadPoolExecutor(max_workers=1) as pool:
        with service.store.transaction() as connection:
            record = next(record for record in service.store.jobs(connection) if record.job_id == job.job_id)
            service.store.save_job(
                connection, JobRecord.model_validate({**record.model_dump(), "state": "running", "invocations": 1})
            )
        writers = []

        def checkpoint_hook(trial_id, step):
            if step == 1:
                writers.append(pool.submit(writer))
                assert acquired.wait(timeout=5)

        run_job(root, job.job_id, progress_hook=checkpoint_hook)
        for future in writers:
            future.result(timeout=10)
    state = service.status(job.job_id)
    assert state.state == "completed"
    assert state.invocations == state.attempts == 1
    report = service.report(job.job_id)
    assert len(report.trials) == 1
    assert tuple(row.step for row in report.trials[0].rows) == (1, 2, 3)
    assert not list((root / "jobs" / job.job_id).glob("failure-*.json"))


def test_reopening_existing_workspace_does_not_require_the_writers_lock(tmp_path):
    store = WorkflowStore(tmp_path)
    with store.transaction():
        reopened = WorkflowStore(tmp_path)
        assert reopened.policy == store.policy


@pytest.mark.parametrize("allowance", [0.15, 1.2])
def test_metadata_retry_stops_at_attempt_deadline_without_entering_the_body(tmp_path, allowance):
    store = WorkflowStore(tmp_path)
    entered = []
    with store.transaction():
        began = time.monotonic()
        deadline = began + allowance
        with pytest.raises(WorkflowError, match="wall-time"), store.transaction(deadline=deadline):
            entered += [True]
        assert time.monotonic() - began < allowance + 0.5
    assert entered == []


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_metadata_deadlines_reject_before_opening(tmp_path, deadline):
    store = WorkflowStore(tmp_path)
    with pytest.raises(ValueError, match="finite"):
        store.connect(deadline=deadline)


def test_commit_retry_does_not_repeat_transaction_body_or_native_reservations(tmp_path, monkeypatch):
    store = WorkflowStore(tmp_path)
    original_connect = store.connect

    class BusyCommit(sqlite3.Connection):
        commit_calls = 0

        def commit(self):
            self.commit_calls += 1
            if self.commit_calls == 1:
                error = sqlite3.OperationalError("Opaque transient commit contention")
                monkeypatch.setattr(error, "sqlite_errorcode", sqlite3.SQLITE_BUSY, raising=False)
                raise error
            return super().commit()

    connection = sqlite3.connect(store.path, isolation_level=None, factory=BusyCommit)
    monkeypatch.setattr(store, "connect", lambda **kwargs: connection)
    entered = []
    with store.transaction(deadline=time.monotonic() + 5) as transaction:
        entered.append(True)
        transaction.execute("INSERT INTO settings VALUES ('once','retained')")
    assert entered == [True]
    assert connection.commit_calls == 2
    with closing(original_connect()) as retained:
        assert retained.execute("SELECT count(*) FROM settings WHERE name='once'").fetchone()[0] == 1
