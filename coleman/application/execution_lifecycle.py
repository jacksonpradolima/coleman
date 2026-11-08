"""Shared provider-free admission and lifecycle rules for both execution families."""

from __future__ import annotations

import shutil
import sqlite3
import time
from contextlib import closing

from coleman.application import execution_views as views
from coleman.application._workflow_records import require_record_plan
from coleman.application.workflow_ownership import worker_active
from coleman.application.workflow_store import WorkflowStore
from coleman.artifacts import ArtifactWriter
from coleman.tasks.contracts import TaskLimits
from coleman.workflows.contracts import WorkflowError
from coleman.workflows.lifecycle import TERMINAL

QUEUE_CAPACITY_ERROR = "Workspace running/queue capacity is exhausted"


def owned_record(store: WorkflowStore, connection: sqlite3.Connection, job_id: str, owner: str) -> views.Job:
    """Read an owned known job without decoding any provider-defined value.

    Parameters
    ----------
    store, connection : WorkflowStore, sqlite3.Connection
        Existing common store and caller-owned consistent metadata connection.
    job_id, owner : str
        Validated host-issued handle and independently supplied local principal.

    Returns
    -------
    JobRecord or TaskJobRecord
        Complete known metadata; the facade separately checks the desired arm.

    Raises
    ------
    KeyError, ValueError
        Missing/nonowned handle or corrupt/unknown metadata. No provider loads.
    """
    job = next((item for item in store.records(connection) if (item.job_id, item.owner) == (job_id, owner)), None)
    if job is None:
        raise KeyError("No owned execution job")
    return job


def check_queue_capacity(store: WorkflowStore, jobs: tuple[views.Job, ...]) -> None:
    """Reserve finite all-owner queue slots, including cancelled live owners.

    Parameters
    ----------
    store : WorkflowStore
        Existing host-owned common policy and actual kernel ownership root.
    jobs : tuple of JobRecord or TaskJobRecord
        Complete current metadata population from the admission transaction.

    Raises
    ------
    WorkflowError
        Running/queued or still kernel-owned jobs exhaust the common capacity.
        Status labels do not release a live worker's slot.
    """
    count = sum(item.state in ("queued", "running") or worker_active(store.root, item.job_id) for item in jobs)
    if count >= store.policy.max_running + store.policy.max_queued:
        raise WorkflowError(QUEUE_CAPACITY_ERROR)


def check_admission(
    store: WorkflowStore, connection: sqlite3.Connection, jobs: tuple[views.Job, ...], plan: views.Plan
) -> None:
    """Check common tombstone, queue, output and worker-memory reservations.

    Parameters
    ----------
    store, connection : WorkflowStore, sqlite3.Connection
        Existing store and serialized admission transaction.
    jobs : tuple of JobRecord or TaskJobRecord
        All owners and both record families, including retained tombstones.
    plan : ExperimentPlan or TaskPlan
        Already concretely validated plan; only pure known metadata is used.

    Raises
    ------
    WorkflowError, ValueError
        Exhausted common capacity or corrupt retained plan. Expired output keeps
        its reservation until its actual owned directory has been cleaned.
    """
    if len(jobs) >= store.policy.max_jobs:
        raise WorkflowError("Job/tombstone slots are full; create another owned workspace")
    check_queue_capacity(store, jobs)
    if views.limits(plan).max_memory_bytes > store.policy.max_active_memory_bytes:
        raise WorkflowError("One worker exceeds aggregate active memory capacity")
    reserved = sum(
        views.limits(store.read_plan(connection, item.plan_id, item.owner)).max_output_bytes
        for item in jobs
        if item.state != "expired" or (store.root / "jobs" / item.job_id).exists()
    )
    if reserved + views.limits(plan).max_output_bytes > store.policy.max_retained_bytes:
        raise WorkflowError("Aggregate retained output reservation exhausted")


def save_plan(store: WorkflowStore, plan: views.Plan, owner: str) -> None:
    """Save immutable known plan bytes in the existing finite plan catalogue.

    Parameters
    ----------
    store : WorkflowStore
        Existing host-owned metadata and artifact owner.
    plan : ExperimentPlan or TaskPlan
        Complete prevalidated plan. Concrete provider callbacks already ran
        outside this function; serialization here uses only known contracts.
    owner : str
        Independently supplied local principal.

    Raises
    ------
    WorkflowError, ValueError, OSError, sqlite3.Error
        Exhausted aggregate slots, changed immutable bytes or storage failure.
        Published orphan plans consume finite slots; no key is recycled.
    """
    identity, payload = plan.fingerprint.removeprefix("sha256:"), plan.to_json()
    with store.transaction() as connection:
        existing = connection.execute("SELECT payload FROM plans WHERE id=? AND owner=?", (identity, owner)).fetchone()
        if existing is not None:
            if store.read_plan(connection, identity, owner).fingerprint != plan.fingerprint:
                raise WorkflowError("Saved plan identity conflicts with its immutable content")
            return
        if connection.execute("SELECT count(*) FROM plans").fetchone()[0] >= store.policy.max_plans:
            raise WorkflowError("Saved plan slots are full; create another owned workspace")
        target = store.root / "plans" / f"{identity}.json"
        if not target.exists() and len(list(target.parent.glob("*"))) >= store.policy.max_plans:
            raise WorkflowError("Saved plan artifacts/orphans exhaust the finite workspace slots")
        ArtifactWriter(store.root).write_immutable(target, payload)
        connection.execute("INSERT INTO plans VALUES (?,?,?)", (identity, owner, payload))


def admit_record(store: WorkflowStore, plan: views.Plan, proposed: views.Job) -> views.Job:
    """Serialize one known-family admission after concrete adapter validation.

    Parameters
    ----------
    store : WorkflowStore
        Existing shared metadata, reservations and immutable plan catalogue.
    plan : ExperimentPlan or TaskPlan
        Exact plan whose adapter checked concrete codecs/runtime outside SQL.
    proposed : JobRecord or TaskJobRecord
        Fully constructed initial queued record, with no provider callbacks.

    Returns
    -------
    JobRecord or TaskJobRecord
        Original logical record for an identical retry, otherwise one newly
        admitted record. Concurrent identical requests consume a single slot.

    Raises
    ------
    WorkflowError, ValueError, KeyError
        Cross-family/payload key conflict, changed/missing plan or exhausted
        shared capacity. No source, bootstrap, learner or restore runs in SQL.
    """
    require_record_plan(proposed, plan)
    if proposed.state != "queued" or proposed.invocations or proposed.attempts:
        raise WorkflowError("A new admission must be queued with no physical work")
    with store.transaction() as connection:
        jobs = store.records(connection)
        existing = next(
            (job for job in jobs if (job.owner, job.request_id) == (proposed.owner, proposed.request_id)), None
        )
        if existing is not None:
            if type(existing) is not type(proposed) or existing.request_digest != proposed.request_digest:
                raise WorkflowError("Idempotency key conflicts with its retained payload or execution family")
            return existing
        retained = store.read_plan(connection, proposed.plan_id, proposed.owner)
        if type(retained) is not type(plan) or retained.fingerprint != plan.fingerprint:
            raise WorkflowError("Admission differs from its retained immutable plan")
        check_admission(store, connection, jobs, plan)
        store.save_record(connection, proposed, create=True)
        return proposed


def reconcile_record(store: WorkflowStore, job_id: str, owner: str) -> tuple[views.Job, bool]:
    """Reconcile lost kernel ownership once without importing or restoring code.

    Parameters
    ----------
    store : WorkflowStore
        Shared durable metadata and actual execution-lock root.
    job_id, owner : str
        Validated host handle and independently supplied local principal.

    Returns
    -------
    tuple
        Known job and actual lock-ownership observation. A dead running worker
        becomes interrupted; no queued or interrupted work starts here.

    Raises
    ------
    KeyError, ValueError, OSError, sqlite3.Error
        Missing/nonowned or corrupt metadata, unsafe lock storage or SQLite
        failure. An actual interruption transition retains the original
        one-second fail-fast write wait; other errors propagate unchanged.

    Notes
    -----
    Read the job in a consistent read transaction, then probe its actual kernel
    lock. These are observations, not an atomic cross-resource ownership claim.
    Nonrunning jobs and observed live owners need no writer reservation. Only
    a possibly orphaned running job enters the original write transaction,
    after closing the read connection. It rechecks both current metadata and
    the kernel lock before changing running to interrupted. A negative probe
    never authorizes execution; scheduling retains its separate owned claim.
    """
    with closing(store.connect()) as connection:
        connection.execute("BEGIN")
        job = owned_record(store, connection, job_id, owner)
        active = worker_active(store.root, job.job_id)
    if job.state != "running" or active:
        return job, active
    with store.transaction() as connection:
        job = owned_record(store, connection, job_id, owner)
        active = worker_active(store.root, job.job_id)
        if job.state == "running" and not active:
            job = views.changed(job, state="interrupted", error="worker_ownership_lost")
            store.save_record(connection, job)
    return job, active


def cancel_record(store: WorkflowStore, job_id: str, owner: str) -> None:
    """Serialize idempotent cancellation against complete-last publication.

    Parameters
    ----------
    store : WorkflowStore
        Shared metadata owner; opens one short write transaction.
    job_id, owner : str
        Validated owned handle and trusted principal.

    Notes
    -----
    A completion committed first remains completed. Otherwise nonterminal state
    becomes cancelled and the live worker exits at its existing guard sample.
    This operation neither imports a provider nor claims the process is dead.
    """
    with store.transaction() as connection:
        job = owned_record(store, connection, job_id, owner)
        if job.state not in TERMINAL:
            store.save_record(connection, views.changed(job, state="cancelled"))


def requeue_record(store: WorkflowStore, expected: views.Job, plan: views.Plan) -> views.Job:
    """Requeue explicitly validated continuation without another runtime engine.

    Parameters
    ----------
    store : WorkflowStore
        Existing common store and queue capacities.
    expected : JobRecord or TaskJobRecord
        Exact record inspected before adapter compatibility checks outside SQL.
    plan : ExperimentPlan or TaskPlan
        Exact compatible retained plan; no provider callback runs in this body.

    Returns
    -------
    JobRecord or TaskJobRecord
        Same logical job, queued or already live/completed/awaiting evaluation.

    Raises
    ------
    WorkflowError, ValueError
        Expired/live cancelled state, changed stage/plan, exhausted invocation
        allowance or common queue capacity. Missing state never resets here.
    """
    with store.transaction() as connection:
        job = owned_record(store, connection, expected.job_id, expected.owner)
        require_record_plan(job, plan)
        if type(job) is not type(expected) or (job.plan_id, job.stage) != (expected.plan_id, expected.stage):
            raise WorkflowError("Continuation changed its retained family, plan or stage")
        if job.state in ("completed", "queued", "running", "awaiting_evaluation"):
            return job
        if job.state == "expired" or worker_active(store.root, job.job_id):
            raise WorkflowError("Expired work or an active cancelled worker cannot resume")
        caps = views.limits(plan)
        maximum = caps.max_invocations if isinstance(caps, TaskLimits) else caps.max_attempts
        if job.invocations >= maximum:
            raise WorkflowError("Worker invocation allowance exhausted")
        check_queue_capacity(store, store.records(connection))
        updated = views.changed(job, state="queued", error=None)
        store.save_record(connection, updated)
        return updated


def cleanup_records(store: WorkflowStore, owner: str, family: type[views.Job]) -> tuple[str, ...]:
    """Expire one facade's terminal bytes while retaining common tombstones.

    Parameters
    ----------
    store : WorkflowStore
        Existing common metadata and artifact owner.
    owner : str
        Independently supplied local principal.
    family : type
        Built-in known JobRecord or TaskJobRecord selected by the facade.

    Returns
    -------
    tuple of str
        Owned expired handles. Live workers and nonterminal jobs are retained.

    Notes
    -----
    Metadata expiry precedes deletion, so failed cleanup retains actual bytes
    and their reservation. Retrying cleanup removes them after ownership ends.
    Plans, input snapshots and nonrecyclable request keys are never deleted.
    """
    expired = []
    with store.transaction() as connection:
        for job in store.records(connection):
            if not isinstance(job, family) or job.owner != owner or worker_active(store.root, job.job_id):
                continue
            if job.state == "expired":
                expired.append(job.job_id)
            elif job.state in TERMINAL and time.time() - job.changed >= store.policy.retention_seconds:
                store.save_record(connection, views.changed(job, state="expired"))
                expired.append(job.job_id)
    for identity in expired:
        directory = store.root / "jobs" / identity
        if directory.exists():
            shutil.rmtree(directory)
    return tuple(expired)


def retained_request(store: WorkflowStore, owner: str, request_id: str) -> views.Job | None:
    """Look up an all-family caller key before expensive compatibility checks.

    Parameters
    ----------
    store : WorkflowStore
        Shared finite metadata owner.
    owner, request_id : str
        Trusted principal and validated retained idempotency key.

    Returns
    -------
    JobRecord or TaskJobRecord or None
        Existing logical record including expiry, or no prior admission. The
        serialized admission must recheck this key to close competing submits.
    """
    with closing(store.connect()) as connection:
        return next(
            (job for job in store.records(connection) if (job.owner, job.request_id) == (owner, request_id)), None
        )
