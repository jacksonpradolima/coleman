"""Known operational projections for the single driver, without provider callbacks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from coleman.application.workflow_store import JobRecord
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint, fingerprint
from coleman.tasks._base import TaskContract
from coleman.tasks.contracts import TaskFrontier, TaskId, TaskJobRecord, TaskLimits, TaskPlan
from coleman.workflows.contracts import ExperimentPlan, WorkflowError, WorkflowLimits

type Job = JobRecord | TaskJobRecord
type Plan = ExperimentPlan | TaskPlan


class CoordinateAttempt(TaskContract):
    """Retain a generic physical reservation, never a learner commit or fee.

    Attributes
    ----------
    coordinate_fingerprint, plan_fingerprint : str
        Exact declared coordinate and full immutable plan identities.
    attempt_id : str
        Job/coordinate/ordinal identity, independent of mathematical updates.
    stage : Reference
        Exact plan-owned stage, without conversion to a native stage label.
    worker_invocation, ordinal : int
        Actual worker start and coordinate reservation, each one through eight.
    started_epoch_seconds : float
        Finite actual reservation epoch seconds, distinct from source clocks.

    Notes
    -----
    A crash after reservation can precede any computation. Reading proves only
    the retained declaration; actual owned-state ACK uses a separate verifier.
    """

    artifact_type: Literal["physical_attempt"]
    coordinate_fingerprint: Fingerprint
    plan_fingerprint: Fingerprint
    attempt_id: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    stage: Reference
    worker_invocation: Annotated[int, Field(strict=True, ge=1, le=8)]
    ordinal: Annotated[int, Field(strict=True, ge=1, le=8)]
    started_epoch_seconds: Annotated[Binary64, Field(ge=0)]
    _maximum_bytes = 16 * 1024


def limits(plan: Plan) -> WorkflowLimits | TaskLimits:
    """Return existing declared caps without converting their work units.

    Parameters
    ----------
    plan : ExperimentPlan or TaskPlan
        Revalidated retained plan from the known metadata union.

    Returns
    -------
    WorkflowLimits or TaskLimits
        Original immutable limits. Native build work and generic named work
        retain distinct meanings; this projection performs no I/O.
    """
    return plan.limits if isinstance(plan, TaskPlan) else plan.request.limits


def completed(job: Job) -> int:
    """Read completed logical coordinates, never attempts or observations.

    Parameters
    ----------
    job : JobRecord or TaskJobRecord
        Revalidated retained operational record.

    Returns
    -------
    int
        Nonnegative original stage count, without inference or side effects.
    """
    return job.completed_coordinates if isinstance(job, TaskJobRecord) else job.completed_trials


def current_coordinate(job: Job) -> str | None:
    """Read the active exact coordinate identity from its retained family.

    Parameters
    ----------
    job : JobRecord or TaskJobRecord
        Revalidated retained operational record.

    Returns
    -------
    str or None
        Exact content identity or explicit no-active-coordinate state, with no
        provider lookup or metadata mutation.
    """
    return job.current_coordinate if isinstance(job, TaskJobRecord) else job.current_trial


def position(job: Job, coordinate_fingerprint: str) -> int:
    """Read acknowledged work for this coordinate; zero denotes no known work.

    Parameters
    ----------
    job : JobRecord or TaskJobRecord
        Revalidated operational metadata, not proof of current store acceptance.
    coordinate_fingerprint : str
        Exact selected trajectory identity.

    Returns
    -------
    int
        Native committed build ordinal or generic named-profile work count.
        Zero for another coordinate or absent frontier; no store is opened.
    """
    if current_coordinate(job) != coordinate_fingerprint:
        return 0
    if isinstance(job, TaskJobRecord):
        return 0 if job.frontier is None else job.frontier.position
    return job.checkpoint_step


def changed(job: Job, **changes) -> Job:
    """Revalidate one known metadata arm; no codec/provider callback executes.

    Parameters
    ----------
    job : JobRecord or TaskJobRecord
        Complete retained record, with unchanged family and indexed identity.
    **changes : object
        Internal built-in field changes; unknown fields remain invalid.

    Returns
    -------
    JobRecord or TaskJobRecord
        Revalidated value with actual time.time epoch seconds; writes nothing.

    Raises
    ------
    ValueError
        Invalid lifecycle/content. The host separately enforces transitions,
        plan joins and serialized cancellation before saving this value.
    """
    return type(job).model_validate({**job.model_dump(), **changes, "changed": time.time()})


def reserved(job: Job, coordinate_fingerprint: str) -> Job:
    """Count one physical reservation while retaining only its own frontier.

    Parameters
    ----------
    job : JobRecord or TaskJobRecord
        Current running record inside the common reservation transaction.
    coordinate_fingerprint : str
        Exact admitted trajectory; changing coordinates clears its work origin.

    Returns
    -------
    JobRecord or TaskJobRecord
        Revalidated attempt count/current-coordinate value. No computation,
        learner update, artifact write or acknowledgement is performed here.
    """
    if isinstance(job, TaskJobRecord):
        frontier = job.frontier if job.current_coordinate == coordinate_fingerprint else None
        return changed(job, attempts=job.attempts + 1, current_coordinate=coordinate_fingerprint, frontier=frontier)
    step = job.checkpoint_step if job.current_trial == coordinate_fingerprint else 0
    return changed(job, attempts=job.attempts + 1, current_trial=coordinate_fingerprint, checkpoint_step=step)


def progressed(job: Job, coordinate_fingerprint: str, frontier: TaskFrontier) -> Job:
    """Project verified progress without moving a retained coordinate backward.

    Parameters
    ----------
    job : JobRecord or TaskJobRecord
        Current running metadata after actual concrete progress verification.
    coordinate_fingerprint : str
        Exact currently reserved trajectory.
    frontier : TaskFrontier
        Revalidated named logical position, independently verified against the
        actual owned state or weaker original native checkpoint evidence.

    Returns
    -------
    JobRecord or TaskJobRecord
        New pure metadata value; no store, callback or transaction is opened.

    Raises
    ------
    WorkflowError, ValueError
        Different coordinate, backward progress or invalid original field bounds.
        A publication generation is never converted into a logical work count.
    """
    if current_coordinate(job) != coordinate_fingerprint:
        raise WorkflowError("Committed progress differs from the active coordinate")
    if frontier.position < position(job, coordinate_fingerprint):
        raise WorkflowError("Committed progress cannot rewind acknowledged logical work")
    if isinstance(job, TaskJobRecord):
        return changed(job, frontier=frontier)
    return changed(job, checkpoint_step=frontier.position)


def coordinate_completed(job: Job, count: int) -> Job:
    """Keep monotone stage counts and the original native checkpoint field.

    Parameters
    ----------
    job : JobRecord or TaskJobRecord
        Current metadata after complete coordinate evidence was verified.
    count : int
        Completed coordinates in this stage, not attempts or independent units.

    Returns
    -------
    JobRecord or TaskJobRecord
        Pure revalidated value. Native counters remain monotone across tuning
        and evaluation; the original native checkpoint_step is retained. Reusing
        an already completed generic prefix retains another coordinate's active
        frontier; it must not erase independent evidence of accepted state.
    """
    if isinstance(job, TaskJobRecord):
        if count <= job.completed_coordinates:
            return job
        return changed(job, completed_coordinates=max(completed(job), count), current_coordinate=None, frontier=None)
    return changed(job, completed_trials=max(completed(job), count), current_trial=None)


def require_pending_coordinate(job: Job, index: int) -> None:
    """Refuse missing evidence in an independently retained completed prefix.

    Parameters
    ----------
    job : JobRecord or TaskJobRecord
        Complete record at the actual worker start. Generic counts refer to the
        frozen ordered TaskPlan catalogue, independent of attempts and state
        publication generations. Native stage counters keep their old semantics.
    index : int
        Zero-based coordinate position from the one common ordered work loop.

    Raises
    ------
    WorkflowError
        A generic coordinate was already completed but its selected owner now
        reports absent complete evidence. Refuse before a fresh reservation or
        coordinate advancement; no implicit fresh run replaces acceptance.

    Notes
    -----
    The caller invokes this only after read_completed reports absence. This
    pure check loads no files or providers and leaves native behavior unchanged.
    """
    if isinstance(job, TaskJobRecord) and index < job.completed_coordinates:
        raise WorkflowError("Previously completed task coordinate evidence is missing; fresh fallback is forbidden")


def attempt_directory(root: Path, coordinate_fingerprint: str) -> Path:
    """Use the unchanged native directory convention for exact content IDs.

    Parameters
    ----------
    root : pathlib.Path
        Trusted actual job directory, never selected by portable evidence.
    coordinate_fingerprint : str
        Validated content SHA-256 with its explicit sha256 prefix.

    Returns
    -------
    pathlib.Path
        Reserved relative attempts directory; this helper creates nothing.
    """
    return root / "attempts" / coordinate_fingerprint.removeprefix("sha256:")


def attempt_payload(job: Job, plan: Plan, coordinate_fingerprint: str, ordinal: int) -> tuple[str, str]:
    """Encode a known physical reservation before entering a metadata transaction.

    Parameters
    ----------
    job, plan : JobRecord or TaskJobRecord, ExperimentPlan or TaskPlan
        Exact already joined retained metadata; no provider determines routing.
    coordinate_fingerprint : str
        Current declared trajectory content identity.
    ordinal : int
        Bounded next coordinate reservation, starting at one.

    Returns
    -------
    tuple of str
        Actual attempt ID and canonical complete declaration. Native IDs and
        PhysicalAttempt bytes retain their original profile and fields.

    Raises
    ------
    ValueError
        Invalid complete reservation. Performs no file write or source access.
    """
    namespace = "task.physical.v1" if isinstance(job, TaskJobRecord) else "workflow.physical.v1"
    identity = fingerprint([namespace, job.job_id, coordinate_fingerprint, ordinal])[7:]
    if isinstance(job, TaskJobRecord):
        event = CoordinateAttempt(
            schema_version=1,
            contract="coleman-tasks-v1",
            artifact_type="physical_attempt",
            coordinate_fingerprint=coordinate_fingerprint,
            plan_fingerprint=plan.fingerprint,
            attempt_id=identity,
            stage=job.stage,
            worker_invocation=job.invocations,
            ordinal=ordinal,
            started_epoch_seconds=time.time(),
        )
    else:
        from coleman.workflows.lifecycle import PhysicalAttempt

        event = PhysicalAttempt(
            schema_version=1,
            trial_id=coordinate_fingerprint,
            attempt_id=identity,
            plan_fingerprint=plan.fingerprint,
            stage=job.stage,
            worker_invocation=job.invocations,
            ordinal=ordinal,
            started_epoch_seconds=time.time(),
        )
    return identity, event.to_json()


def retained_attempts(root: Path, plan: Plan, *, job_id: str) -> int:
    """Validate all bounded actual reservation files before reconciling metadata.

    Parameters
    ----------
    root : pathlib.Path
        Trusted job directory, outside source imports and portable evidence.
    plan : ExperimentPlan or TaskPlan
        Exact original coordinates and per-coordinate reservation ceiling.
    job_id : str
        Actual known owned job identity, supplied independently of event bytes
        and directory names. Generic attempt IDs bind this exact identity.

    Returns
    -------
    int
        Complete retained physical reservation count, including interrupted work.

    Raises
    ------
    ValueError, OSError
        Missing/corrupt membership, bad versions, altered bindings or limits.
        No owner state, native checkpoint or provider factory is loaded.
    """
    if not isinstance(plan, TaskPlan):
        from coleman.application.workflow_artifacts import read_attempts

        return len(read_attempts(root, plan))
    identity = TypeAdapter(TaskId).validate_python(job_id)
    count = 0
    for coordinate in plan.coordinates:
        paths = sorted(attempt_directory(root, coordinate.fingerprint).glob("*.json"))
        if len(paths) > plan.limits.max_attempts:
            raise WorkflowError("Retained physical attempts exceed the saved cap")
        for ordinal, path in enumerate(paths, 1):
            with path.open("rb") as stream:
                event = CoordinateAttempt.from_json(stream.read(16 * 1024 + 1))
            if (
                event.coordinate_fingerprint,
                event.plan_fingerprint,
                event.stage,
                event.ordinal,
                path.name,
                event.attempt_id,
                event.worker_invocation <= plan.limits.max_invocations,
            ) != (
                coordinate.fingerprint,
                plan.fingerprint,
                plan.stage,
                ordinal,
                f"{ordinal:04d}.json",
                fingerprint(["task.physical.v1", identity, coordinate.fingerprint, ordinal])[7:],
                True,
            ):
                raise WorkflowError("Retained physical attempt binding mismatch")
            count += 1
    return count
