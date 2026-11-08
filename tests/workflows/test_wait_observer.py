"""Real SQLite contention respects the workflow test observer's existing bounds."""

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from coleman.application.workflows import WorkflowService
from coleman.workflows.lifecycle import ExecutePlan
from tests.workflows.test_execution import setup_workflow, wait


class _StatusProbe:
    """Delegate actual status calls and release a writer after real contention."""

    def __init__(self, service, release):
        self.service = service
        self.store = service.store
        self.release = release
        self.codes = []

    def status(self, job_id):
        """Retain real views/errors without replacing a principal operation."""
        try:
            return self.service.status(job_id)
        except sqlite3.OperationalError as error:
            code = getattr(error, "sqlite_errorcode", None)
            if type(code) is int and (code & 0xFF) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                self.codes.append(code)
                self.release.set()
            raise


def test_wait_retries_actual_sqlite_contention_without_repeating_native_work(tmp_path):
    service, preview = setup_workflow(tmp_path, policies=("Random",), seeds=(42,))
    job = service.execute(ExecutePlan(schema_version=1, request_id="observed-busy", plan_id=preview.plan_id))
    completed = wait(service, job.job_id)
    assert completed.invocations == 1
    assert completed.attempts == 1
    report = service.report(job.job_id).to_json()
    acquired, release = Event(), Event()
    probe = _StatusProbe(service, release)

    def writer():
        # The real connection is created, used and closed in this thread. Its
        # existing bounded transaction waits for a real failed status attempt.
        with service.store.transaction(deadline=time.monotonic() + 45):
            acquired.set()
            assert release.wait(timeout=45)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(writer)
        try:
            assert acquired.wait(timeout=45)
            observed = wait(probe, job.job_id)
        finally:
            release.set()
            future.result(timeout=45)
    assert probe.codes
    assert all((code & 0xFF) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) for code in probe.codes)
    # JobView elapsed_seconds is a fresh observation, not retained job state.
    assert observed.model_dump(exclude={"elapsed_seconds"}) == completed.model_dump(exclude={"elapsed_seconds"})
    assert service.report(job.job_id).to_json() == report


def test_wait_propagates_actual_noncontention_sqlite_error(tmp_path):
    service = WorkflowService(tmp_path)
    with service.store.transaction() as connection:
        connection.execute("DROP TABLE jobs")
    with pytest.raises(sqlite3.OperationalError) as raised:
        wait(service, "absent-job")
    assert raised.value.sqlite_errorcode & 0xFF == sqlite3.SQLITE_ERROR
