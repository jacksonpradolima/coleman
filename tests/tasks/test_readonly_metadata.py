"""Read operational metadata under a real WAL writer and reconcile lost owners.

Authored metadata fixtures do not represent provider work or published results.
SQLite and kernel locks are real; SQL traces synchronize races without replacing
statements, transaction bodies or ownership probes.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, nullcontext
from threading import Event

import pytest

from coleman.application.baseline_lock import baseline_lock
from coleman.application.execution_bootstrap import stored_policy, stored_profile
from coleman.application.tasks import TaskService
from coleman.application.workflow_store import WorkflowStore
from coleman.application.workflows import WorkflowService
from coleman.core.identity import Reference
from coleman.tasks.contracts import TaskJobRecord
from coleman.tasks.identity import HostBootstrapProfile
from tests.tasks.test_storage_records import FIXTURES, _native, _plans, _put_plan, _task

WAIT_SECONDS = 10  # New observation fences only; SQLite keeps its original 1 s.
RESULT = "sha256:" + "d" * 64  # Authored metadata marker, never a result/ACK proof.


def _changed(value, **updates):
    """Revalidate complete authored metadata without adding runtime fields."""
    return type(value).model_validate({**value.model_dump(), **updates})


def _profile():
    return HostBootstrapProfile.from_json((FIXTURES / "bootstrap.json").read_bytes())


def _rows(connection):
    """Record exact operational rows while the owning transaction is held."""
    return {
        name: tuple(tuple(row) for row in connection.execute(f"SELECT * FROM {name} ORDER BY rowid"))
        for name in ("settings", "jobs")
    }


@contextmanager
def _writer(store):
    """Acquire the actual independent WAL writer without changing busy_timeout."""
    with closing(store.connect()) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 1000
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        finally:
            connection.rollback()


def _seed_case(root, family, state):
    """Install valid selected metadata without invoking a learner or scheduler."""
    profile = _profile()
    initial = TaskService(root, owner="owner", bootstrap_profile=profile)
    native, task = _plans()
    plan = native if family == "native" else task
    job = _native(plan) if family == "native" else _task(plan)
    if state == "completed":
        count = (
            {"completed_trials": job.total_trials}
            if family == "native"
            else {"completed_coordinates": job.total_coordinates}
        )
        job = _changed(job, state=state, result_digest=RESULT, **count)
    elif state == "running":
        job = _changed(job, state=state, invocations=1)
    else:
        assert state == "queued"
    with initial.store.transaction() as connection:
        _put_plan(connection, plan)
        initial.store.save_record(connection, job, create=True)
    service = WorkflowService(root, owner="owner") if family == "native" else TaskService(root, owner="owner")
    return service, job


def _job(view):
    """Normalize only facade DTO shape, leaving both principal status methods real."""
    return view.job if hasattr(view, "job") else view


@pytest.mark.parametrize("approval", ("matching", "provider_free"))
def test_registered_task_host_reopens_under_an_independent_wal_writer(tmp_path, approval):
    profile = _profile()
    original = TaskService(tmp_path, owner="owner", bootstrap_profile=profile)
    with _writer(original.store) as writer:
        before = _rows(writer)
        selected = profile if approval == "matching" else None
        reopened = TaskService(tmp_path, owner="owner", bootstrap_profile=selected)
        assert reopened.policy.to_json() == original.policy.to_json()
        assert reopened.bootstrap_profile == selected
        assert _rows(writer) == before


def test_provider_free_unregistered_host_is_read_only_and_does_not_install_authority(tmp_path):
    original = WorkflowStore(tmp_path)
    with _writer(original) as writer:
        before = _rows(writer)
        service = TaskService(tmp_path)
        assert service.bootstrap_profile is None
        assert stored_profile(writer) is None
        assert stored_policy(writer) is None
        assert _rows(writer) == before


@pytest.mark.parametrize("conflict", ("profile", "policy"))
def test_registered_host_conflict_refuses_before_competing_for_the_writer(tmp_path, conflict):
    profile = _profile()
    original = TaskService(tmp_path, bootstrap_profile=profile)
    other_profile = _changed(profile, reference=Reference(id="different-operator-profile", version="1"))
    other_caps = _changed(original.policy.caps, max_attempts=1 if original.policy.caps.max_attempts != 1 else 2)
    other_policy = _changed(original.policy, caps=other_caps)
    selected = other_profile if conflict == "profile" else profile
    policy = other_policy if conflict == "policy" else None
    with _writer(original.store) as writer:
        before = _rows(writer)
        with pytest.raises(ValueError, match="cannot be silently replaced"):
            TaskService(tmp_path, bootstrap_profile=selected, task_policy=policy)
        assert _rows(writer) == before


@pytest.mark.parametrize("missing", ("task_bootstrap_v1", "task_policy_v1"))
def test_corrupt_registration_pair_is_not_repaired_by_a_read_only_reopen(tmp_path, missing):
    profile = _profile()
    original = TaskService(tmp_path, bootstrap_profile=profile)
    with original.store.transaction() as connection:
        connection.execute("DELETE FROM settings WHERE name=?", (missing,))
    with _writer(original.store) as writer:
        before = _rows(writer)
        with pytest.raises(ValueError, match="Incomplete generic host equality metadata"):
            TaskService(tmp_path, bootstrap_profile=profile)
        assert _rows(writer) == before


def test_first_registration_still_requires_real_write_ownership_and_keeps_the_one_second_limit(tmp_path):
    store = WorkflowStore(tmp_path)
    profile = _profile()
    with _writer(store) as writer:
        before = _rows(writer)
        with pytest.raises(sqlite3.OperationalError) as failure:
            TaskService(tmp_path, bootstrap_profile=profile)
        assert failure.value.sqlite_errorcode & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
        assert _rows(writer) == before
    registered = TaskService(tmp_path, bootstrap_profile=profile)
    assert registered.bootstrap_profile == profile
    with closing(registered.store.connect()) as connection:
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 1000
        assert stored_profile(connection) == profile
        assert stored_policy(connection) == registered.policy


@pytest.mark.parametrize("family", ("native", "task"))
@pytest.mark.parametrize("state", ("queued", "completed", "running"))
def test_provider_free_status_does_not_need_the_writer_for_a_read_only_snapshot(tmp_path, family, state):
    service, job = _seed_case(tmp_path, family, state)
    ownership = baseline_lock(tmp_path / "locks" / f"{job.job_id}.lock") if state == "running" else nullcontext()
    with ownership, _writer(service.store) as writer:
        before = _rows(writer)
        view = service.status(job.job_id)
        actual = _job(view)
        assert actual.state == state
        assert actual.job_id == job.job_id
        assert view.worker_active is (state == "running")
        assert actual.result_digest == job.result_digest
        assert _rows(writer) == before


@pytest.mark.parametrize("family", ("native", "task"))
def test_lost_running_owner_still_requires_a_serialized_write_then_interrupts(tmp_path, family):
    service, job = _seed_case(tmp_path, family, "running")
    with _writer(service.store) as writer:
        before = _rows(writer)
        with pytest.raises(sqlite3.OperationalError) as failure:
            service.status(job.job_id)
        assert failure.value.sqlite_errorcode & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
        assert _rows(writer) == before
    view = service.status(job.job_id)
    assert _job(view).state == "interrupted"
    assert _job(view).error == "worker_ownership_lost"
    assert view.worker_active is False


def _observe_write_acquisition(store, monkeypatch):
    """Pause only a real SQL trace at the stale-read/write-acquisition seam.

    The bounded observer pause precedes SQLite executing BEGIN IMMEDIATE. The
    main test commits its real competing writer before releasing the observer;
    the unchanged one-second busy wait is not spent on test-thread scheduling.
    No statement, write body, lock probe or returned model is replaced.
    """
    entered, release, armed = Event(), Event(), Event()
    original_connect = store.connect

    def observed_connect(**kwargs):
        connection = original_connect(**kwargs)

        def trace(statement):
            if statement.strip().upper() == "BEGIN IMMEDIATE" and armed.is_set():
                entered.set()
                release.wait(WAIT_SECONDS)

        connection.set_trace_callback(trace)
        return connection

    monkeypatch.setattr(store, "connect", observed_connect)
    return entered, release, armed


@pytest.mark.parametrize("family", ("native", "task"))
@pytest.mark.parametrize("winner", ("completed", "cancelled"))
def test_running_owner_reconciliation_rereads_the_winning_commit_after_write_acquisition(
    tmp_path, monkeypatch, family, winner
):
    service, job = _seed_case(tmp_path, family, "running")
    enter_writer, release_writer, armed = _observe_write_acquisition(service.store, monkeypatch)
    updates = {"state": winner, "changed": 1.0}
    if winner == "completed":
        updates["result_digest"] = RESULT
        if isinstance(job, TaskJobRecord):
            updates["completed_coordinates"] = job.total_coordinates
        else:
            updates["completed_trials"] = job.total_trials
    winning = _changed(job, **updates)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with _writer(service.store) as writer:
            armed.set()
            outcome = executor.submit(service.status, job.job_id)
            assert enter_writer.wait(WAIT_SECONDS), "Lost-running reconciliation must acquire a writer"
            service.store.save_record(writer, winning)
            writer.commit()
            release_writer.set()
        view = outcome.result(timeout=WAIT_SECONDS)
    actual = _job(view)
    assert actual.state == winner
    assert actual.error is None
    assert actual.result_digest == winning.result_digest
    assert view.worker_active is False
    with closing(service.store.connect()) as connection:
        retained = service.store.records(connection)
    assert len(retained) == 1
    assert retained[0].to_json() == winning.to_json()


@pytest.mark.parametrize("family", ("native", "task"))
def test_reconciliation_rechecks_new_kernel_ownership_after_its_inactive_snapshot(tmp_path, monkeypatch, family):
    service, job = _seed_case(tmp_path, family, "running")
    enter_writer, release_writer, armed = _observe_write_acquisition(service.store, monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as executor, _writer(service.store) as writer:
        armed.set()
        outcome = executor.submit(service.status, job.job_id)
        assert enter_writer.wait(WAIT_SECONDS), "The inactive snapshot must reach the genuine writer path"
        # A new real owner now holds the retained inode before acquisition.
        with baseline_lock(tmp_path / "locks" / f"{job.job_id}.lock"):
            writer.commit()
            release_writer.set()
            view = outcome.result(timeout=WAIT_SECONDS)
    assert _job(view).state == "running"
    assert _job(view).error is None
    assert view.worker_active is True
    with closing(service.store.connect()) as connection:
        retained = service.store.records(connection)
    assert len(retained) == 1
    assert retained[0].to_json() == job.to_json()
