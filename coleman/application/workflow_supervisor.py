"""Fixed detached local coordinator; kernel locks own workers across host restarts."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from coleman.application.baseline_lock import BaselineBusyError, baseline_lock
from coleman.application.execution_bootstrap import acknowledge_profile, profile_observed, validate_queued_tasks
from coleman.application.execution_views import changed, limits
from coleman.application.workflow_ownership import worker_active
from coleman.application.workflow_store import WorkflowStore, _sqlite_contended
from coleman.tasks.contracts import TaskJobRecord, TaskLimits
from coleman.tasks.identity import HostBootstrapProfile
from coleman.workflows.contracts import WorkflowContract, WorkflowError

_ACK_WAIT_SECONDS = 10.0
_MAX_GENERATION = 2**63 - 1


class _SupervisionProgress(WorkflowContract):
    """Bounded coordination counters; these never represent scientific observations.

    Parameters
    ----------
    requested, observed : int
        Monotonic local generations. Observation occurs inside the scheduler's
        metadata transaction; a busy kernel lock alone is not acknowledgement.
    """

    requested: Annotated[int, Field(strict=True, ge=0, le=_MAX_GENERATION)] = 0
    observed: Annotated[int, Field(strict=True, ge=0, le=_MAX_GENERATION)] = 0

    @model_validator(mode="after")
    def _ordered(self):
        """Reject corrupt/future progress instead of inventing acknowledgement."""
        if self.observed > self.requested:
            raise WorkflowError("Coordinator observation exceeds its requested generation")
        return self


def _progress(connection) -> _SupervisionProgress:
    """Read one versioned fixed-size metadata record; absence supports an older workspace."""
    row = connection.execute("SELECT payload FROM settings WHERE name='supervisor_progress_v1'").fetchone()
    return _SupervisionProgress(schema_version=1) if row is None else _SupervisionProgress.from_json(row[0])


def _save_progress(connection, progress: _SupervisionProgress) -> None:
    """Persist bounded coordination metadata in the caller's existing transaction."""
    connection.execute(
        "INSERT INTO settings(name,payload) VALUES('supervisor_progress_v1',?) "
        "ON CONFLICT(name) DO UPDATE SET payload=excluded.payload",
        (progress.to_json(),),
    )


def _request_observation(store: WorkflowStore, *, bootstrap_profile: HostBootstrapProfile | None = None) -> int | None:
    """Request a new queue observation without resetting or recycling a generation."""
    with store.transaction() as connection:
        if not any(job.state == "queued" for job in store.records(connection)):
            return None
        validate_queued_tasks(store, connection, bootstrap_profile)
        progress = _progress(connection)
        if progress.requested == _MAX_GENERATION:
            raise WorkflowError("Coordinator generation capacity exhausted; use a new owned workspace")
        updated = _SupervisionProgress(schema_version=1, requested=progress.requested + 1, observed=progress.observed)
        _save_progress(connection, updated)
        return updated.requested


def _observed(store: WorkflowStore, generation: int, *, bootstrap_profile: HostBootstrapProfile | None = None) -> bool:
    """Check transactional acknowledgement, rather than equating a busy inode with a live reader."""
    with closing(store.connect()) as connection:
        return _progress(connection).observed >= generation and (
            bootstrap_profile is None or profile_observed(connection, bootstrap_profile, generation)
        )


def _spawn(
    root: Path,
    role: str,
    descriptor: int,
    job_id: str | None = None,
    *,
    bootstrap_profile: HostBootstrapProfile | None = None,
) -> None:
    """Spawn only a fixed installed module, explicitly inheriting one ownership descriptor."""
    arguments = [sys.executable, "-m", "coleman.application.workflow_supervisor", role, str(root), str(descriptor)]
    if job_id is not None:
        arguments.append(job_id)
    if bootstrap_profile is not None:
        arguments[1:1] = ["-I", "-B"]
        arguments.extend(["--bootstrap-profile", HostBootstrapProfile.model_validate(bootstrap_profile).to_json()])
    subprocess.Popen(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=root,
        pass_fds=(descriptor,),
        start_new_session=True,
    )


def ensure_supervisor(root: Path, *, bootstrap_profile: HostBootstrapProfile | None = None) -> None:
    """Ensure one workspace coordinator without taking execution ownership from live workers.

    Parameters
    ----------
    root : pathlib.Path
        Trusted persistent POSIX workspace. A fixed detached Python process inherits
        its lock; no transport lifetime, PID guess or arbitrary command is involved.
    bootstrap_profile : HostBootstrapProfile or None, optional
        Current operator-supplied generic approval, carried through fixed internal
        argv. None preserves the native startup route and cannot schedule tasks.

    Raises
    ------
    OSError, ValueError
        Unsupported/unsafe storage or spawn failure. Existing queued records remain
        recoverable by reopening the workspace and retrying the same request.
        Missing acknowledgement raises a safe retryable WorkflowError; it never
        silently returns an unobserved queue as healthy.

    Notes
    -----
    Waits up to ten seconds for an actual queue observation. A retiring child can
    close its descriptor while an old parent still retains another copy; this
    bounded handshake retries that interval. Work is durably admitted first, so
    an acknowledgement timeout requires retrying the same request, not a new job.
    """
    store = WorkflowStore(root)
    generation = (
        _request_observation(store)
        if bootstrap_profile is None
        else _request_observation(store, bootstrap_profile=bootstrap_profile)
    )
    if generation is None:
        return
    deadline = time.monotonic() + _ACK_WAIT_SECONDS
    while time.monotonic() < deadline:
        observed = (
            _observed(store, generation)
            if bootstrap_profile is None
            else _observed(store, generation, bootstrap_profile=bootstrap_profile)
        )
        if observed:
            return
        try:
            with baseline_lock(root / "supervisor.lock", handoff=True) as descriptor:
                if bootstrap_profile is None:
                    _spawn(root, "supervise", descriptor)
                else:
                    _spawn(root, "supervise", descriptor, bootstrap_profile=bootstrap_profile)
        except BaselineBusyError:
            # An actual scheduler must acknowledge this generation, or the lock
            # must become available. A retiring parent's duplicate can stay busy.
            pass
        time.sleep(0.05)
    raise WorkflowError("Coordinator did not acknowledge queued work; retry the same operation")


def _schedule(
    store: WorkflowStore, supervisor_descriptor: int, *, bootstrap_profile: HostBootstrapProfile | None = None
) -> bool:
    """Reconcile dead ownership and claim bounded queued work atomically under native locks."""
    with store.transaction() as connection:
        jobs = store.records(connection)
        validate_queued_tasks(store, connection, bootstrap_profile)
        active = [job for job in jobs if worker_active(store.root, job.job_id)]
        _reconcile_ownership(store, connection, jobs, active)
        memory = sum(limits(store.read_plan(connection, job.plan_id, job.owner)).max_memory_bytes for job in active)
        for job in jobs:
            if job.state != "queued" or len(active) >= store.policy.max_running:
                continue
            plan = store.read_plan(connection, job.plan_id, job.owner)
            caps = limits(plan)
            if memory + caps.max_memory_bytes > store.policy.max_active_memory_bytes:
                continue
            try:
                with baseline_lock(store.root / "locks" / f"{job.job_id}.lock", handoff=True) as worker_descriptor:
                    maximum = caps.max_invocations if isinstance(caps, TaskLimits) else caps.max_attempts
                    if job.invocations >= maximum:
                        store.save_record(
                            connection,
                            changed(job, state="failed", error="invocation_limit"),
                        )
                        continue
                    updated = changed(job, state="running", invocations=job.invocations + 1, error=None)
                    store.save_record(connection, updated)
                    # A queued child snapshot synchronizes with this commit before
                    # deciding to run; inherited ownership is already continuous.
                    if bootstrap_profile is None:
                        _spawn(store.root, "work", worker_descriptor, job.job_id)
                    else:
                        _spawn(store.root, "work", worker_descriptor, job.job_id, bootstrap_profile=bootstrap_profile)
                    active.append(updated)
                    memory += caps.max_memory_bytes
            except BaselineBusyError:
                continue
        pending = bool(active) or any(job.state == "queued" for job in jobs)
        progress = _progress(connection)
        if bootstrap_profile is not None:
            acknowledge_profile(connection, bootstrap_profile, progress.requested)
        _save_progress(
            connection,
            _SupervisionProgress(schema_version=1, requested=progress.requested, observed=progress.requested),
        )
        if not pending:
            # Release ownership while admission is still serialized. A later
            # admission cannot observe a retiring owner and strand queued work.
            os.close(supervisor_descriptor)
        return pending


def _reconcile_ownership(store, connection, jobs, active) -> None:
    """Record lost kernel ownership inside the caller's transaction without restarting native work."""
    active_ids = {job.job_id for job in active}
    for job in jobs:
        if job.state == "running" and job.job_id not in active_ids:
            store.save_record(
                connection,
                changed(job, state="interrupted", error="worker_ownership_lost"),
            )


def _supervise(root: Path, descriptor: int, *, bootstrap_profile: HostBootstrapProfile | None = None) -> None:
    """Keep inherited ownership while retrying transient SQLite write contention.

    Startup and later polling retain the same coordinator lock across numeric
    SQLITE_BUSY/SQLITE_LOCKED errors, including their extended codes. A trusted
    registry publication can exceed the one-second connection timeout without
    stranding acknowledged queued jobs. Retry polling uses the existing 50 ms
    pause; no job, invocation or acknowledgement is invented by a failed poll.
    Other SQLite, storage and contract errors propagate instead of being hidden.
    """
    store = None
    while True:
        try:
            if store is None:
                store = WorkflowStore(root)
            pending = (
                _schedule(store, descriptor)
                if bootstrap_profile is None
                else _schedule(store, descriptor, bootstrap_profile=bootstrap_profile)
            )
            if not pending:
                return
        except sqlite3.OperationalError as error:
            if not _sqlite_contended(error):
                raise
        time.sleep(0.05)


def main() -> None:
    """Private fixed subprocess entry point; public clients cannot select these arguments."""
    role, directory, descriptor, *identity = sys.argv[1:]
    bootstrap_profile = None
    if len(identity) >= 2 and identity[-2] == "--bootstrap-profile":
        bootstrap_profile = HostBootstrapProfile.from_json(identity.pop())
        identity.pop()
    root = Path(directory)
    lock = root / ("supervisor.lock" if role == "supervise" else f"locks/{identity[0]}.lock")
    _verify_inherited_lock(lock, int(descriptor))
    if role == "supervise" and not identity:
        if bootstrap_profile is None:
            _supervise(root, int(descriptor))
        else:
            _supervise(root, int(descriptor), bootstrap_profile=bootstrap_profile)
    elif role == "work" and len(identity) == 1:
        _work(root, identity[0], bootstrap_profile)
    else:
        raise ValueError("Unknown fixed workflow subprocess role")


def _work(root: Path, job_id: str, bootstrap_profile: HostBootstrapProfile | None) -> None:
    """Select a known wrapper while keeping one actual coordinate driver."""
    store = WorkflowStore(root)
    with closing(store.connect()) as connection:
        task = next(job for job in store.records(connection) if job.job_id == job_id)
    if isinstance(task, TaskJobRecord):
        from coleman.application.execution_driver import run_job as run_task

        run_task(root, job_id, bootstrap_profile=bootstrap_profile)
    else:
        from coleman.application.workflow_worker import run_job

        run_job(root, job_id)


def _verify_inherited_lock(path: Path, descriptor: int) -> None:
    """Verify the actual retained inode and exclusive inherited ownership, without a PID assumption."""
    import fcntl

    from coleman.corpus.materialization import validate_source_storage

    validate_source_storage(path)
    inherited, current = os.fstat(descriptor), path.stat()
    if (inherited.st_dev, inherited.st_ino) != (current.st_dev, current.st_ino):
        raise ValueError("Fixed workflow worker did not inherit its actual retained lock")
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == "__main__":
    main()
