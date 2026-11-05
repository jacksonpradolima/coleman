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
from coleman.application.workflow_ownership import worker_active
from coleman.application.workflow_store import JobRecord, WorkflowStore, _sqlite_contended
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


def _request_observation(store: WorkflowStore) -> int | None:
    """Request a new queue observation without resetting or recycling a generation."""
    with store.transaction() as connection:
        if not any(job.state == "queued" for job in store.jobs(connection)):
            return None
        progress = _progress(connection)
        if progress.requested == _MAX_GENERATION:
            raise WorkflowError("Coordinator generation capacity exhausted; use a new owned workspace")
        updated = _SupervisionProgress(schema_version=1, requested=progress.requested + 1, observed=progress.observed)
        _save_progress(connection, updated)
        return updated.requested


def _observed(store: WorkflowStore, generation: int) -> bool:
    """Check transactional acknowledgement, rather than equating a busy inode with a live reader."""
    with closing(store.connect()) as connection:
        return _progress(connection).observed >= generation


def _spawn(root: Path, role: str, descriptor: int, job_id: str | None = None) -> None:
    """Spawn only a fixed installed module, explicitly inheriting one ownership descriptor."""
    arguments = [sys.executable, "-m", "coleman.application.workflow_supervisor", role, str(root), str(descriptor)]
    if job_id is not None:
        arguments.append(job_id)
    subprocess.Popen(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=root,
        pass_fds=(descriptor,),
        start_new_session=True,
    )


def ensure_supervisor(root: Path) -> None:
    """Ensure one workspace coordinator without taking execution ownership from live workers.

    Parameters
    ----------
    root : pathlib.Path
        Trusted persistent POSIX workspace. A fixed detached Python process inherits
        its lock; no transport lifetime, PID guess or arbitrary command is involved.

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
    generation = _request_observation(store)
    if generation is None:
        return
    deadline = time.monotonic() + _ACK_WAIT_SECONDS
    while time.monotonic() < deadline:
        if _observed(store, generation):
            return
        try:
            with baseline_lock(root / "supervisor.lock", handoff=True) as descriptor:
                _spawn(root, "supervise", descriptor)
        except BaselineBusyError:
            # An actual scheduler must acknowledge this generation, or the lock
            # must become available. A retiring parent's duplicate can stay busy.
            pass
        time.sleep(0.05)
    raise WorkflowError("Coordinator did not acknowledge queued work; retry the same operation")


def _schedule(store: WorkflowStore, supervisor_descriptor: int) -> bool:
    """Reconcile dead ownership and claim bounded queued work atomically under native locks."""
    with store.transaction() as connection:
        jobs = store.jobs(connection)
        active = [job for job in jobs if worker_active(store.root, job.job_id)]
        _reconcile_ownership(store, connection, jobs, active)
        memory = sum(store.plan(connection, job.plan_id, job.owner).request.limits.max_memory_bytes for job in active)
        for job in jobs:
            if job.state != "queued" or len(active) >= store.policy.max_running:
                continue
            plan = store.plan(connection, job.plan_id, job.owner)
            if memory + plan.request.limits.max_memory_bytes > store.policy.max_active_memory_bytes:
                continue
            try:
                with baseline_lock(store.root / "locks" / f"{job.job_id}.lock", handoff=True) as worker_descriptor:
                    if job.invocations >= plan.request.limits.max_attempts:
                        store.save_job(
                            connection,
                            JobRecord.model_validate(
                                {
                                    **job.model_dump(),
                                    "state": "failed",
                                    "changed": time.time(),
                                    "error": "invocation_limit",
                                }
                            ),
                        )
                        continue
                    updated = JobRecord.model_validate(
                        {
                            **job.model_dump(),
                            "state": "running",
                            "changed": time.time(),
                            "invocations": job.invocations + 1,
                            "error": None,
                        }
                    )
                    store.save_job(connection, updated)
                    # A queued child snapshot synchronizes with this commit before
                    # deciding to run; inherited ownership is already continuous.
                    _spawn(store.root, "work", worker_descriptor, job.job_id)
                    active.append(updated)
                    memory += plan.request.limits.max_memory_bytes
            except BaselineBusyError:
                continue
        pending = bool(active) or any(job.state == "queued" for job in jobs)
        progress = _progress(connection)
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
            store.save_job(
                connection,
                JobRecord.model_validate(
                    {
                        **job.model_dump(),
                        "state": "interrupted",
                        "changed": time.time(),
                        "error": "worker_ownership_lost",
                    }
                ),
            )


def _supervise(root: Path, descriptor: int) -> None:
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
            if not _schedule(store, descriptor):
                return
        except sqlite3.OperationalError as error:
            if not _sqlite_contended(error):
                raise
        time.sleep(0.05)


def main() -> None:
    """Private fixed subprocess entry point; public clients cannot select these arguments."""
    role, directory, descriptor, *identity = sys.argv[1:]
    root = Path(directory)
    lock = root / ("supervisor.lock" if role == "supervise" else f"locks/{identity[0]}.lock")
    _verify_inherited_lock(lock, int(descriptor))
    if role == "supervise" and not identity:
        _supervise(root, int(descriptor))
    elif role == "work" and len(identity) == 1:
        from coleman.application.workflow_worker import run_job

        run_job(root, identity[0])
    else:
        raise ValueError("Unknown fixed workflow subprocess role")


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
