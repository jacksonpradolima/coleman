"""Read existing typed host views without mutating their operational contracts."""

from __future__ import annotations

import math
from datetime import datetime

from pydantic import TypeAdapter

from coleman.core.actions import MixtureAction
from coleman.core.decisions import ExecutedDecision
from coleman.core.temporal import Timestamp
from coleman.replay.contracts import ReplayState
from coleman.replay.retained_contracts import RetainedReplayState
from coleman.tasks.contracts import TaskJobRecord, TaskPlan
from coleman.telemetry.lifecycle import LifecycleSnapshot, MetricPoint, Outcome, point
from coleman.telemetry.resources import ResourceSnapshot


def _clock(value: datetime, previous: tuple[datetime, ...]) -> datetime:
    """Validate the explicit observation clock before computing any age."""
    now = TypeAdapter(Timestamp).validate_python(value)
    if any(now < before for before in previous):
        raise ValueError("Operational observation cannot precede an emitted/applied source fact")
    return now


def action_snapshot(executed: ExecutedDecision, *, at: datetime, outcome: Outcome) -> LifecycleSnapshot:
    """Inspect one actual action without confusing coefficients with risk.

    Parameters
    ----------
    executed : ExecutedDecision
        Original proposal and actual mapped action, revalidated before use.
    at : datetime
        Explicit zoned observation instant, at or after any decision clock.
    outcome : str
        Caller-declared operational outcome. No feasibility or fallback is
        inferred from coefficients or a sampling declaration.

    Returns
    -------
    LifecycleSnapshot
        Concentration for mixture coefficients, otherwise not applicable;
        explicit constraint/fallback indicators and implementation annotation.

    Raises
    ------
    ValueError
        Invalid source, outcome or clock. No learning, execution or I/O occurs.
    """
    current = ExecutedDecision.model_validate(executed)
    timing = current.proposal.request.timing
    now = _clock(at, () if timing is None else (timing.decision_at,))
    concentration = None
    if isinstance(current.action, MixtureAction):
        concentration = math.fsum(item.value * item.value for item in current.action.weights)
    return LifecycleSnapshot(
        schema_version=1,
        route="direct",
        outcome=outcome,
        observed_at=now,
        source_fingerprint=current.fingerprint,
        algorithm=current.proposal.implementation,
        state_codec=None,
        points=(
            point("concentration", concentration, missing="not_applicable"),
            point("constraint", int(outcome == "infeasible")),
            point("fallback", int(outcome == "fallback")),
        ),
    )


def replay_snapshot(state: ReplayState | RetainedReplayState, *, at: datetime) -> LifecycleSnapshot:
    """Observe known full/retained replay generations without applying facts.

    Parameters
    ----------
    state : ReplayState or RetainedReplayState
        Existing full-history v1 or explicitly sealed v2 metadata, including
        original producer joins. A v2 sealed prefix is already quiescent.
    at : datetime
        Explicit zoned operational clock, no earlier than emitted decisions or
        applied updates/drains. Future admitted revisions do not become updates.

    Returns
    -------
    LifecycleSnapshot
        Exact admitted D/U/pending counts and ages in seconds. Missing feedback
        counts decisions without an admitted revision, not failed decisions.
        Empty pending/update ages are not applicable. An actual update whose
        clock was retired by a v2 seal has unavailable age, never invented zero
        or the possibly later last-drain clock. Lifetime D/U include the seal.

    Raises
    ------
    ValueError
        Invalid/unknown source or reversed observation clock. No host calls,
        scheduling, state restoration, deduplication or storage occurs.
    """
    if type(state) not in {ReplayState, RetainedReplayState}:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("This mapper supports only the known full v1 and retained v2 replay types")
    current = type(state).model_validate(state)
    seal = current.seal if isinstance(current, RetainedReplayState) else None
    now = _clock(
        at,
        tuple(d.accepted_at for d in current.decisions)
        + tuple(u.application.apply_at for u in current.updates)
        + tuple(d.apply_at for d in current.drains)
        + (() if seal is None else (seal.last_drain.apply_at,)),
    )
    applied = {(u.application.feedback.lineage, u.application.executed.fingerprint) for u in current.updates}
    admitted = {(f.lineage, f.batch.decision_fingerprint) for f in current.feedback}
    pending = [d for d in current.decisions if (d.lineage, d.executed.fingerprint) not in applied]
    pending_age = (now - min(d.accepted_at for d in pending)).total_seconds() if pending else None
    update_age = (
        (now - max(u.application.apply_at for u in current.updates)).total_seconds() if current.updates else None
    )
    return LifecycleSnapshot(
        schema_version=1,
        route="replay",
        outcome="pending" if pending else "accepted",
        observed_at=now,
        source_fingerprint=current.fingerprint,
        algorithm=current.decisions[-1].executed.proposal.implementation if current.decisions else None,
        state_codec=current.state_codec,
        points=(
            point("decisions", current.decision_count),
            point("updates", current.update_count),
            point("pending", len(pending)),
            point("pending_age", pending_age, missing="not_applicable"),
            point("update_age", update_age, missing="unavailable" if current.update_count else "not_applicable"),
            point("missing_feedback", sum((d.lineage, d.executed.fingerprint) not in admitted for d in pending)),
        ),
    )


def task_snapshot(job: TaskJobRecord, plan: TaskPlan, *, at: datetime) -> LifecycleSnapshot:
    """Read a real durable job without promoting partial work to completion.

    Parameters
    ----------
    job : TaskJobRecord
        Existing retained job, whose exact plan binding and caps are checked.
    plan : TaskPlan
        Original immutable plan. Its limits are reused, never widened here.
    at : datetime
        Zoned operational clock at or after job.changed epoch seconds.

    Returns
    -------
    LifecycleSnapshot
        Progress ratio and actual attempts/invocations. Interrupted/cancelled
        work remains pending; only the original job can establish completion.

    Raises
    ------
    ValueError
        Bad source, cap, binding or clock. No provider/worker/store is started.
    """
    current, declared = TaskJobRecord.model_validate(job), TaskPlan.model_validate(plan)
    current.validate_for(declared)
    now = _clock(at, ())
    if now.timestamp() < current.changed:
        raise ValueError("Operational clock cannot precede retained job progress")
    outcome: Outcome
    if current.state == "failed":
        outcome = "failed"
    elif current.state == "completed":
        outcome = "accepted"
    else:
        outcome = "pending"
    return LifecycleSnapshot(
        schema_version=1,
        route="task",
        outcome=outcome,
        observed_at=now,
        source_fingerprint=current.fingerprint,
        algorithm=None,
        state_codec=None,
        points=(
            point("task_progress", current.completed_coordinates / current.total_coordinates),
            point("attempts", current.attempts),
            point("invocations", current.invocations),
        ),
    )


def resource_points(sample: ResourceSnapshot) -> tuple[MetricPoint, ...]:
    """Map the existing process tracker with its original attribution limits.

    Parameters
    ----------
    sample : ResourceSnapshot
        Actual current-executable measurements. This function reads no clock,
        filesystem, child-process resource usage or external power meter.

    Returns
    -------
    tuple of MetricPoint
        Five explicit observations in MiB, seconds and percent. Null remains
        unavailable; CPU utilization may exceed100 for multiple threads.

    Raises
    ------
    ValueError
        Nonfinite/negative source reading. Cumulative samples are not summed.
    """
    return (
        point("rss", sample.current_rss_mib),
        point("peak_rss", sample.peak_rss_mib),
        point("cpu_time", sample.cpu_time_seconds),
        point("wall_time", sample.wall_time_seconds),
        point("cpu_utilization", sample.cpu_utilization_percent),
    )
