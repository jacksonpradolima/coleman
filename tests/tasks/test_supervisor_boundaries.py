"""Real mixed queue/lock boundaries with narrowly recorded process launches.

Only subprocess creation/waiting is replaced in launch-unit cases. Plans,
SQLite transactions, generation/profile checks and kernel ownership remain
real. A recorded launch is not a child ACK, accepted learner state or result.
"""

from __future__ import annotations

import os
import sys
from contextlib import closing
from pathlib import Path

import pytest

from coleman.application import execution_bootstrap as bootstrap
from coleman.application import execution_lifecycle as lifecycle
from coleman.application import workflow_supervisor as supervisor
from coleman.application.baseline_lock import BaselineBusyError, baseline_lock
from coleman.application.execution_views import changed
from coleman.application.tasks import TaskService
from coleman.application.workflow_store import WorkflowStore
from coleman.application.workflows import WorkflowService
from coleman.core.identity import Reference
from coleman.tasks.identity import HostBootstrapProfile
from coleman.workflows.lifecycle import HostPolicy
from tests.tasks.test_storage_records import _native, _plans, _task

FIXTURES = Path(__file__).parents[1] / "fixtures" / "tasks" / "v1"


def _profile():
    """Select the explicit declaration fixture; this is not installed approval."""
    return HostBootstrapProfile.from_json((FIXTURES / "bootstrap.json").read_bytes())


def _retain(store, plan, record):
    """Use actual saved metadata and shared admission, without executing code."""
    lifecycle.save_plan(store, plan, record.owner)
    return lifecycle.admit_record(store, plan, record)


def _records(store):
    with closing(store.connect()) as connection:
        return store.records(connection)


@pytest.mark.parametrize("with_profile", [False, True])
def test_fixed_spawn_carries_only_supported_internal_argv_and_actual_owned_descriptor(
    tmp_path, monkeypatch, with_profile
):
    """The host carries explicit approval; stored or public content selects no executable."""
    observed = []

    def process(arguments, **options):
        observed.append((arguments, options))
        os.fstat(options["pass_fds"][0])

    monkeypatch.setattr(supervisor.subprocess, "Popen", process)
    current = _profile() if with_profile else None
    with baseline_lock(tmp_path / "locks" / "owned.lock", handoff=True) as descriptor:
        supervisor._spawn(tmp_path, "work", descriptor, "owned", bootstrap_profile=current)
        arguments, options = observed[0]
        prefix = [sys.executable, "-I", "-B"] if with_profile else [sys.executable]
        expected = prefix + [
            "-m",
            "coleman.application.workflow_supervisor",
            "work",
            str(tmp_path),
            str(descriptor),
            "owned",
        ]
        if current is not None:
            expected += ["--bootstrap-profile", current.to_json()]
        assert arguments == expected
        assert options["pass_fds"] == (descriptor,)
        assert options["cwd"] == tmp_path
        assert options["start_new_session"] is True
        with pytest.raises(BaselineBusyError), baseline_lock(tmp_path / "locks" / "owned.lock"):
            pass
    assert len(observed) == 1


@pytest.mark.parametrize("selection", ["missing", "wrong"])
def test_queued_task_without_current_matching_profile_refuses_before_generation_or_launch(
    tmp_path, monkeypatch, selection
):
    """A matching profile retained in SQL never replaces current operator approval."""
    current = _profile()
    service = TaskService(tmp_path, owner="owner", bootstrap_profile=current)
    _, task = _plans()
    original = _retain(service.store, task, _task(task))
    supplied = None
    if selection == "wrong":
        supplied = HostBootstrapProfile.model_validate(
            current.model_copy(update={"reference": Reference(id="wrong", version="1")})
        )
    calls = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(ValueError, match="matching explicit current"):
        supervisor.ensure_supervisor(tmp_path, bootstrap_profile=supplied)
    assert _records(service.store) == (original,)
    assert not calls
    with closing(service.store.connect()) as connection:
        assert supervisor._progress(connection).requested == 0
        assert bootstrap.stored_profile(connection) == current


def test_matching_current_profile_is_acknowledged_only_after_real_queue_transaction(tmp_path, monkeypatch):
    """A launch alone grants no observation; the actual scheduler commits that metadata."""
    current = _profile()
    service = TaskService(tmp_path, owner="owner", bootstrap_profile=current)
    _, task = _plans()
    original = _retain(service.store, task, _task(task))
    generation = supervisor._request_observation(service.store, bootstrap_profile=current)
    assert generation == 1
    assert supervisor._observed(service.store, generation, bootstrap_profile=current) is False
    calls = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    with baseline_lock(tmp_path / "supervisor.lock", handoff=True) as descriptor:
        assert supervisor._schedule(service.store, descriptor, bootstrap_profile=current) is True
    retained = _records(service.store)[0]
    assert retained.job_id == original.job_id
    assert retained.state == "running"
    assert retained.invocations == 1
    assert retained.attempts == 0
    assert supervisor._observed(service.store, generation, bootstrap_profile=current) is True
    assert supervisor._observed(service.store, generation + 1, bootstrap_profile=current) is False
    wrong = HostBootstrapProfile.model_validate(
        current.model_copy(update={"reference": Reference(id="different", version="1")})
    )
    assert supervisor._observed(service.store, generation, bootstrap_profile=wrong) is False
    assert len(calls) == 1
    # No child was launched by this unit: the real missing lock is reconciled,
    # rather than treating metadata "running" as proof of worker ownership.
    assert service.status(original.job_id).job.state == "interrupted"


@pytest.mark.parametrize("family", ["native", "task"])
def test_scheduler_exhausted_invocations_fail_without_launch_or_attempt(tmp_path, monkeypatch, family):
    """Native/task invocation caps differ but both refuse before process creation."""
    native, task = _plans()
    current = _profile() if family == "task" else None
    if current is not None:
        TaskService(tmp_path, owner="owner", bootstrap_profile=current)
    store = WorkflowStore(tmp_path)
    plan = task if family == "task" else native
    record = _task(task) if family == "task" else _native(native)
    maximum = task.limits.max_invocations if family == "task" else native.request.limits.max_attempts
    _retain(store, plan, record)
    with store.transaction() as connection:
        store.save_record(connection, changed(record, invocations=maximum))
    calls = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    with baseline_lock(tmp_path / "supervisor.lock", handoff=True) as descriptor:
        supervisor._schedule(store, descriptor, bootstrap_profile=current)
    retained = _records(store)[0]
    assert retained.state == "failed"
    assert retained.error == "invocation_limit"
    assert retained.invocations == maximum
    assert retained.attempts == 0
    assert not calls


def test_scheduler_preserves_live_cancelled_memory_reservation_and_queued_job(tmp_path, monkeypatch):
    """A cancelled kernel owner still consumes memory until its descriptor closes."""
    plan, _ = _plans()
    policy = HostPolicy(schema_version=1, max_running=2, max_active_memory_bytes=plan.request.limits.max_memory_bytes)
    store = WorkflowStore(tmp_path, policy)
    first = _retain(store, plan, _native(plan, job_id="active", request_id="first"))
    second = _retain(store, plan, _native(plan, job_id="waiting", request_id="second"))
    with store.transaction() as connection:
        store.save_record(connection, changed(first, state="cancelled"))
    calls = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    with baseline_lock(tmp_path / "locks" / f"{first.job_id}.lock"):
        with baseline_lock(tmp_path / "supervisor.lock", handoff=True) as descriptor:
            assert supervisor._schedule(store, descriptor) is True
        assert _records(store)[1] == second
        assert not calls
    with baseline_lock(tmp_path / "supervisor.lock", handoff=True) as descriptor:
        assert supervisor._schedule(store, descriptor) is True
    assert _records(store)[1].state == "running"
    assert len(calls) == 1


def test_busy_job_inode_does_not_create_a_second_worker_or_discard_queue(tmp_path, monkeypatch):
    """Actual kernel contention leaves the retained queued job unchanged."""
    plan, _ = _plans()
    store = WorkflowStore(tmp_path)
    original = _retain(store, plan, _native(plan))
    calls = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    with (
        baseline_lock(tmp_path / "locks" / f"{original.job_id}.lock"),
        baseline_lock(tmp_path / "supervisor.lock", handoff=True) as descriptor,
    ):
        assert supervisor._schedule(store, descriptor) is True
    assert _records(store) == (original,)
    assert not calls


def test_lost_running_ownership_reconciles_interrupted_and_releases_coordinator(tmp_path):
    """A recorded PID/state grants no ownership; no replacement worker is started."""
    plan, _ = _plans()
    store = WorkflowStore(tmp_path)
    record = _retain(store, plan, _native(plan))
    with store.transaction() as connection:
        store.save_record(connection, changed(record, state="running", invocations=1))
    # The scheduler intentionally closes its own retiring descriptor inside SQL.
    with baseline_lock(tmp_path / "supervisor.lock", handoff=True) as descriptor:
        retiring = os.dup(descriptor)
        assert supervisor._schedule(store, retiring) is False
        with pytest.raises(OSError):
            os.fstat(retiring)
    retained = _records(store)[0]
    assert retained.state == "interrupted"
    assert retained.error == "worker_ownership_lost"
    assert retained.invocations == 1


def test_fixed_entry_checks_actual_inode_and_unknown_role_before_waiting(tmp_path, monkeypatch):
    """A valid descriptor cannot authorize a substituted file or arbitrary role."""
    lock = tmp_path / "supervisor.lock"
    with baseline_lock(lock, handoff=True) as descriptor:
        supervisor._verify_inherited_lock(lock, descriptor)
        foreign = tmp_path / "foreign.lock"
        with baseline_lock(foreign, handoff=True) as other, pytest.raises(ValueError, match="actual retained lock"):
            supervisor._verify_inherited_lock(lock, other)
        calls = []
        monkeypatch.setattr(supervisor, "_supervise", lambda *args, **kwargs: calls.append((args, kwargs)))
        monkeypatch.setattr(
            sys,
            "argv",
            ["fixed", "supervise", str(tmp_path), str(descriptor), "--bootstrap-profile", _profile().to_json()],
        )
        supervisor.main()
        assert calls == [((tmp_path, descriptor), {"bootstrap_profile": _profile()})]
    with baseline_lock(tmp_path / "locks" / "owned.lock", handoff=True) as descriptor:
        monkeypatch.setattr(sys, "argv", ["fixed", "unknown", str(tmp_path), str(descriptor), "owned"])
        with pytest.raises(ValueError, match="Unknown fixed"):
            supervisor.main()


def test_native_facade_with_explicit_profile_and_empty_queue_does_not_launch_or_load_provider(tmp_path, monkeypatch):
    """The optional generic host context preserves the empty native startup path."""
    current = _profile()
    TaskService(tmp_path, bootstrap_profile=current)
    service = WorkflowService(tmp_path, bootstrap_profile=current)
    calls = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    snapshots = service.reconnect()
    assert isinstance(snapshots, tuple)
    assert not snapshots
    assert not calls
    with closing(service.store.connect()) as connection:
        assert supervisor._progress(connection).requested == 0
