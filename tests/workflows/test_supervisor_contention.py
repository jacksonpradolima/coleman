"""Real SQLite write contention must not strand acknowledged native workflow queues."""

import sqlite3
import time
from contextlib import closing

import pytest

import coleman.application.workflow_supervisor as supervision
from coleman.application.baseline_lock import BaselineBusyError, baseline_lock
from coleman.workflows.lifecycle import ExecutePlan
from tests.workflows.test_execution import setup_workflow, wait


@pytest.mark.parametrize("phase", ["startup", "polling"])
def test_actual_busy_writer_preserves_coordinator_and_completes_queue(tmp_path, monkeypatch, phase):
    service, preview = setup_workflow(tmp_path, policies=("Random",), seeds=(42,))
    first_request = ExecutePlan(schema_version=1, request_id="busy-first", plan_id=preview.plan_id)
    # Admit first, then establish a real kernel handoff barrier. The coordinator,
    # SQLite waits and eventual native workers are real, detached processes.
    with monkeypatch.context() as patch:
        patch.setattr(service, "_start", lambda: None)
        first = service.execute(first_request)
    supervisor_lock = tmp_path / "supervisor.lock"
    if phase == "startup":
        with service.store.transaction():
            with baseline_lock(supervisor_lock, handoff=True) as descriptor:
                supervision._spawn(tmp_path, "supervise", descriptor)
            time.sleep(3.0)  # Allows child imports plus the actual one-second SQLite timeout.
            with pytest.raises(BaselineBusyError), baseline_lock(supervisor_lock):
                pass
        last = first
    else:
        # A live inherited worker lock keeps both jobs queued without timing a
        # policy's speed or changing the real native runner or database records.
        with baseline_lock(tmp_path / "locks" / f"{first.job_id}.lock"):
            last = service.execute(ExecutePlan(schema_version=1, request_id="busy-second", plan_id=preview.plan_id))
            assert last.state == "queued"
            with service.store.transaction():
                time.sleep(1.5)
                with pytest.raises(BaselineBusyError), baseline_lock(supervisor_lock):
                    pass
    # Status polling never starts/reconnects a coordinator. Releasing the writer
    # and handoff barrier alone must let this same owner run the original jobs.
    for job in (first, last):
        completed = wait(service, job.job_id)
        assert completed.invocations == 1
        assert completed.attempts == 1
        report = service.report(job.job_id)
        assert len(report.trials) == 1
        assert len(report.trials[0].rows) == 3
    retry = service.execute(first_request)
    assert retry.job_id == first.job_id
    assert retry.result_digest == service.status(first.job_id).result_digest


def test_noncontention_sqlite_errors_are_not_retried(tmp_path, monkeypatch):
    def invalid_schema(store, descriptor):
        with closing(store.connect()) as connection:
            connection.execute("SELECT * FROM nonexistent_coordination_table")

    monkeypatch.setattr(supervision, "_schedule", invalid_schema)
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        supervision._supervise(tmp_path, -1)


@pytest.mark.parametrize(
    "code", [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_BUSY | 256, sqlite3.SQLITE_LOCKED | 256]
)
def test_extended_contention_codes_retry_without_message_matching(tmp_path, monkeypatch, code):
    attempts, pauses = [], []
    error = sqlite3.OperationalError("An opaque message that does not name the contention type")
    monkeypatch.setattr(error, "sqlite_errorcode", code, raising=False)

    def contended_once(store, descriptor):
        attempts.append(descriptor)
        if len(attempts) == 1:
            raise error
        return False

    monkeypatch.setattr(supervision, "_schedule", contended_once)
    monkeypatch.setattr(supervision.time, "sleep", pauses.append)
    supervision._supervise(tmp_path, -1)
    assert attempts == [-1, -1]
    assert pauses == [0.05]
