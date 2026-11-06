"""Pure bounded replay preflight, visibility and pending-state inspection."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import TypeAdapter

from coleman.core.base import CoreImmutable
from coleman.core.capabilities import validate_feedback
from coleman.core.identity import Reference
from coleman.core.primitives import MAX_MEMBERS, Fingerprint, Identifier
from coleman.core.temporal import Timestamp
from coleman.observations.contracts import ObservationHistory, ObservationSnapshot
from coleman.replay.contracts import (
    ReplayApplication,
    ReplayDecisionRecord,
    ReplayFeedback,
    ReplayOpportunity,
    ReplayState,
)
from coleman.replay.revisions import feedback_chains, feedback_key, feedback_value_reason, validate_opportunity

PendingReason = Literal[
    "not_drained",
    "unknown_receipt",
    "not_visible",
    "awaiting_target_or_maturity",
    "partial_components",
    "missing_required_value",
    "no_observed_value",
    "absent",
]


class PendingRound(CoreImmutable):
    """Inspect one admitted but unapplied logical round, without changing state.

    Attributes
    ----------
    lineage : Reference
        Producing decision's lineage, including an inherited parent when relevant.
    executed_fingerprint : str
        Exact executed action binding, never just a proposal or display name.
    feedback_id : str
        Stable logical feedback identity; revisions are not independent rounds.
    revision_id : str or None
        Latest source-visible revision, or null when none is yet visible.
    reason : str
        Explicit blocking state. It never imputes a numeric value.

    Notes
    -----
    This is audit inventory, not an input to a learner or a causal-prefix digest.
    Future admissions may change this inspection without changing old decisions.
    """

    lineage: Reference
    executed_fingerprint: Fingerprint
    feedback_id: Identifier
    revision_id: Identifier | None
    reason: PendingReason


def _instant(value: Timestamp) -> datetime:
    """Validate an explicit UTC instant without collecting an ambient clock."""
    return TypeAdapter(Timestamp).validate_python(value)


def _frontiers(state: ReplayState, eligible_through: Timestamp, apply_at: Timestamp) -> tuple[datetime, datetime]:
    """Keep processing time and information eligibility independently monotonic."""
    cutoff, processing = _instant(eligible_through), _instant(apply_at)
    if cutoff > processing:
        raise ValueError("Replay eligibility cannot follow its processing instant")
    if state.eligible_through is not None and cutoff < state.eligible_through:
        raise ValueError("Replay information frontier cannot move backwards")
    if state.apply_at is not None and processing < state.apply_at:
        raise ValueError("Replay processing frontier cannot move backwards")
    return cutoff, processing


def _producer(state: ReplayState, feedback: ReplayFeedback) -> ReplayDecisionRecord:
    """Find the exact registered execution in its original producing lineage."""
    for record in state.decisions:
        if record.lineage == feedback.lineage and record.executed.fingerprint == feedback.batch.decision_fingerprint:
            return record
    raise ValueError("Replay feedback has no registered producing decision and lineage")


def _visible(feedback: ReplayFeedback, cutoff: datetime) -> bool:
    """Test source visibility before maturity or numeric readiness is considered."""
    timing = feedback.batch.timing
    return (
        timing is not None and timing.received_at is not None and max(timing.available_at, timing.received_at) <= cutoff
    )


def _latest_visible(chain: tuple[ReplayFeedback, ...], cutoff: datetime) -> ReplayFeedback | None:
    """Follow explicit ancestry, never lexical revisions or ready-row fallback."""
    return next((feedback for feedback in reversed(chain) if _visible(feedback, cutoff)), None)


def _eligible_at(record: ReplayDecisionRecord, feedback: ReplayFeedback) -> datetime | None:
    """Combine target, producer and feedback-local gates without backdating."""
    timing = feedback.batch.timing
    decision_timing = record.opportunity.request.timing
    if timing is None or timing.usable_at is None or decision_timing is None:
        return None
    return max(timing.usable_at, decision_timing.decision_at, decision_timing.target_end)


def _reason(
    state: ReplayState, chain: tuple[ReplayFeedback, ...], cutoff: datetime | None
) -> tuple[ReplayFeedback | None, PendingReason | None]:
    """Select the newest visible revision before testing any update requirement."""
    if cutoff is None:
        return None, "not_drained"
    selected = _latest_visible(chain, cutoff)
    if selected is None:
        unknown = any(row.batch.timing is None or row.batch.timing.received_at is None for row in chain)
        return None, "unknown_receipt" if unknown else "not_visible"
    eligible = _eligible_at(_producer(state, selected), selected)
    if eligible is None or eligible > cutoff:
        return selected, "awaiting_target_or_maturity"
    return selected, feedback_value_reason(selected, state.policy.capability)


def _pending(state: ReplayState) -> tuple[PendingRound, ...]:
    """Describe retained unapplied rounds; do not include them in causal state."""
    applied = {feedback_key(record.application.feedback) for record in state.updates}
    result = []
    for chain in feedback_chains(state.feedback):
        if feedback_key(chain[0]) in applied:
            continue
        selected, reason = _reason(state, chain, state.eligible_through)
        if reason is None:
            # A new same-instant producer may have been registered after the
            # last closed drain. It is ready for the next explicit drain.
            reason = "not_drained"
        result.append(
            PendingRound(
                lineage=chain[0].lineage,
                executed_fingerprint=chain[0].batch.decision_fingerprint,
                feedback_id=chain[0].batch.feedback_id,
                revision_id=None if selected is None else selected.batch.revision_id,
                reason=reason,
            )
        )
    return tuple(result)


def _applications(state: ReplayState, cutoff: datetime, processing: datetime) -> tuple[ReplayApplication, ...]:
    """Prevalidate every selected update and capacity before calling a host."""
    applied = {feedback_key(record.application.feedback) for record in state.updates}
    ready = []
    for chain in feedback_chains(state.feedback):
        if feedback_key(chain[0]) in applied:
            continue
        selected, reason = _reason(state, chain, cutoff)
        if selected is None or reason is not None:
            continue
        producer = _producer(state, selected)
        eligible = _eligible_at(producer, selected)
        if eligible is None:
            raise ValueError("A ready replay update must have known eligibility")
        actual_delay = state.decision_count - producer.decision_index - 1
        if state.policy.delivery == "synchronous" and actual_delay > 0:
            raise ValueError("Synchronous replay cannot consume additional logical feedback delay")
        if selected.batch.delay_rounds != actual_delay:
            raise ValueError("Declared feedback delay_rounds differs from intervening successful decisions")
        validate_feedback(producer.executed, selected.batch, state.policy.capability, apply_at=processing)
        ready.append((eligible, producer.decision_index, selected))
    ready.sort(key=lambda row: (row[0], row[1], feedback_key(row[2])))
    _capacity(state, len(ready) + 1)
    return tuple(
        ReplayApplication(
            schema_version=1,
            lineage=state.policy.lineage,
            executed=_producer(state, selected).executed,
            feedback=selected,
            eligible_at=eligible,
            apply_at=processing,
            decision_count_before=state.decision_count,
            update_count_before=state.update_count + index,
        )
        for index, (eligible, _, selected) in enumerate(ready)
    )


def _capacity(state: ReplayState, additional: int) -> None:
    """Check the joint retained-record budget before any host mutation."""
    retained = len(state.decisions) + len(state.feedback) + len(state.updates) + len(state.drains)
    if retained + additional > MAX_MEMBERS:
        raise ValueError("Replay inventory exceeds 4096 retained records")


def _history_update(state: ReplayState, history: ObservationHistory) -> ObservationHistory:
    """Reject deletion, changed source facts and backfilled old selected views."""
    checked = ObservationHistory.model_validate(history)
    if Reference(id=checked.source_id, version=checked.source_version) != state.policy.source:
        raise ValueError("Replay source history identity/version differs from its policy")
    if state.history is not None:
        old = {
            (row.metadata.observation_id, row.metadata.revision_id): row.to_json() for row in state.history.revisions
        }
        new = {(row.metadata.observation_id, row.metadata.revision_id): row.to_json() for row in checked.revisions}
        if any(new.get(key) != value for key, value in old.items()):
            raise ValueError("Replay source history must retain every prior revision byte-for-byte")
        if state.eligible_through is not None and (
            checked.as_of(state.eligible_through).to_json() != state.history.as_of(state.eligible_through).to_json()
        ):
            raise ValueError("Source-history backfill crosses a closed replay information frontier")
    for record in state.decisions:
        timing = record.opportunity.request.timing
        if timing is None or checked.as_of(timing.information_cutoff).to_json() != record.snapshot.to_json():
            raise ValueError("Source-history backfill would rewrite a previously emitted eligible snapshot")
    return checked


def _opportunity(state: ReplayState, opportunity: ReplayOpportunity) -> ObservationSnapshot:
    """Validate a new opportunity's clocks, active universe and declared inputs."""
    request = opportunity.request
    timing, history = request.timing, state.history
    if timing is None or history is None:
        raise ValueError("Causal replay requires explicit decision timing and an admitted source history")
    if request.task != state.policy.stream:
        raise ValueError("Replay request task must match its exact declared stream")
    if request.round != state.decision_count or state.decision_count >= MAX_MEMBERS:
        raise ValueError("Replay request.round must count successful decisions within the finite bound")
    _frontiers(state, timing.information_cutoff, timing.decision_at)
    validate_opportunity(state.policy, state.decisions[0].opportunity if state.decisions else None, opportunity)
    snapshot = history.as_of(timing.information_cutoff)
    if timing.eligible_input_fingerprint != snapshot.fingerprint:
        raise ValueError("Decision eligible-input fingerprint does not match actual source history.as_of")
    return snapshot
