"""Pure finite counters, pending panels and complete controlled-state validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from coleman.learning._control_geometry import _geometry, _origin, _period_history
from coleman.learning._control_history import _closed_intervals, _last_outcome, _unresolved_failure
from coleman.learning._control_values import _encoded
from coleman.learning.contracts import read_loss_panel
from coleman.objectives.controls import IntervalCadence, RoundCadence

if TYPE_CHECKING:
    from coleman.learning.control_contracts import AcceptedPanel, ControlledPayload


def _counters(payload: ControlledPayload) -> None:
    """Keep accepted, consumed and successful-release clocks independently exact."""
    if payload.update_count != payload.consumed_panels + len(payload.pending):
        raise ValueError("Accepted panels must be consumed or retained exactly once")
    if payload.release_count > payload.consumed_panels:
        raise ValueError("Nonempty release counters disagree with consumed observations")
    if (payload.release_count == 0) != (payload.consumed_panels == 0):
        raise ValueError("Nonempty release counters disagree with consumed observations")
    limits = payload.config.limits
    if max(payload.update_count, payload.release_count) > limits.maximum_updates:
        raise ValueError("Controlled update/release count exceeds the selected limit")
    if len(payload.pending) > limits.maximum_pending:
        raise ValueError("Controlled pending panels exceed the selected limit")
    if max(len(payload.intervals), len(payload.periods)) > limits.maximum_intervals:
        raise ValueError("Controlled intervals exceed the selected limit")


def _cadence(payload: ControlledPayload) -> None:
    """Retain whole logical blocks and preserve original acceptance order."""
    cadence = payload.config.controls.cadence
    if isinstance(cadence, RoundCadence):
        if payload.intervals:
            raise ValueError("Round cadence cannot retain host release intervals")
        if payload.consumed_panels != payload.release_count * cadence.panels or len(payload.pending) > cadence.panels:
            raise ValueError("Round cadence cannot flush or merge a partial/failed block")
        expected = tuple(range(payload.consumed_panels + 1, payload.update_count + 1))
        if tuple(item.sequence for item in payload.pending) != expected:
            raise ValueError("Round-cadence pending panels must retain their contiguous order")
    elif not isinstance(cadence, IntervalCadence):
        raise ValueError("Unknown controlled cadence")


def _pending(payload: ControlledPayload) -> None:
    """Validate retained full observations and unique original panel identities."""
    sequences = tuple(item.sequence for item in payload.pending)
    if sequences != tuple(sorted(set(sequences))) or any(item > payload.update_count for item in sequences):
        raise ValueError("Pending panels require unique original accepted sequence numbers")
    if len({item.reference.model_dump_json() for item in payload.pending}) != len(payload.pending):
        raise ValueError("Pending panels require unique original identities")
    members = tuple(item.member for item in payload.config.controls.initial_prior.weights)
    for item in payload.pending:
        read_loss_panel(
            item.panel, members=members, loss_spec=payload.config.loss_spec, transform=payload.config.transform
        )
        _pending_clock(payload, item)


def _pending_clock(payload: ControlledPayload, item: AcceptedPanel) -> None:
    """Retain known clocks without assigning an old block to a new period."""
    timed = isinstance(payload.config.controls.cadence, IntervalCadence)
    if timed and (item.eligible_at is None or item.apply_at is None):
        raise ValueError("Interval pending panels require explicit eligibility and application clocks")
    if item.apply_at is not None and (payload.last_apply_at is None or item.apply_at > payload.last_apply_at):
        raise ValueError("Retained applications cannot follow the last processing clock")
    if payload.config.period_clock is not None and (
        item.apply_at is None or not any(period.start <= item.apply_at < period.end for period in payload.periods)
    ):
        raise ValueError("Period-controlled panels require an originally admitted application period")


def validate_payload(payload: ControlledPayload) -> None:
    """Validate complete controlled continuation, including retained input joins.

    Parameters
    ----------
    payload : ControlledPayload
        Finite fixed-law payload. Pending observations retain exact members,
        loss units and transform; structural support is explicit.

    Returns
    -------
    None
        Original counter, history, origin and geometry joins are valid. A
        well-formed mathematically infeasible exact set remains valid state.

    Raises
    ------
    ValueError, TypeError
        Malformed joins, forged receipts, selected capacities or bytes.

    Notes
    -----
    This validates mathematical state, not authenticated event history.
    Complete outer AlgorithmState bytes must additionally fit the selected cap.
    """
    _counters(payload)
    _origin(payload, payload.origin)
    if payload.support != tuple(item.value > 0 for item in payload.origin.weights):
        raise ValueError("Structural support must exactly name every positive emitted coefficient")
    _cadence(payload)
    _period_history(payload)
    _pending(payload)
    _geometry(payload)
    _closed_intervals(payload)
    _last_outcome(payload)
    _unresolved_failure(payload)
    if len(_encoded(payload.model_dump(mode="json"))) > payload.config.limits.maximum_bytes:
        raise ValueError("Controlled payload exceeds the selected complete-byte capacity")
