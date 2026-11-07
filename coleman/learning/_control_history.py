"""Validate original controlled-release receipts against their fixed law.

A saved receipt carries its own pre-attempt origin and movement context.
Later releases and period resets must not substitute the current ledger into
those historical inputs. These pure checks reconstruct geometry and finite
frontiers; they neither re-solve past optima nor authenticate observation history.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from coleman.core.identity import Reference
from coleman.learning._control_geometry import _origin
from coleman.learning._control_values import _encoded, _weight_bytes
from coleman.objectives.controls import (
    CoefficientOrigin,
    ExactExpertInterval,
    ProjectionOutput,
    exact_mixture_intervals,
)

if TYPE_CHECKING:
    from coleman.learning.control_contracts import ControlledPayload, ControlOutcome
    from coleman.objectives.controls import MovementAllowance


def _outcome_law(payload: ControlledPayload, outcome: ControlOutcome) -> None:
    """Join retained receipts to fixed law, canonical members and original counters."""
    if outcome.law != payload.config.binding:
        raise ValueError("A retained outcome must bind the complete fixed controlled law")
    if tuple(item.member for item in outcome.weights) != tuple(item.member for item in payload.origin.weights):
        raise ValueError("A retained outcome must keep the exact configured members")
    _origin(payload, outcome.origin)
    if outcome.emitted_origin is not None:
        _origin(payload, outcome.emitted_origin)
    counters = (outcome.update_count, outcome.release_count, outcome.consumed_panels)
    current = (payload.update_count, payload.release_count, payload.consumed_panels)
    if any(old > new for old, new in zip(counters, current, strict=True)):
        raise ValueError("A retained outcome cannot pass a current counter frontier")
    if outcome.release_count > outcome.consumed_panels or outcome.consumed_panels > outcome.update_count:
        raise ValueError("A retained outcome has incompatible accepted/consumed/release counters")
    _outcome_counts(outcome)
    _outcome_output(payload, outcome)
    _outcome_context(payload, outcome)


def _outcome_context(payload: ControlledPayload, outcome: ControlOutcome) -> None:
    """Reconstruct historical exact geometry from its complete original context."""
    if outcome.disposition in {"held", "empty"}:
        return
    _context_profile(payload, outcome)
    geometry = exact_mixture_intervals(
        payload.config.controls.bounds,
        outcome.origin,
        movement=payload.config.controls.movement,
        usage=outcome.movement_usage,
        exact_path=outcome.exact_path_before,
    )
    if outcome.solver_inputs is not None and outcome.solver_inputs != geometry.inputs_fingerprint:
        raise ValueError("A retained solver input fingerprint must reproduce its original exact geometry")
    _geometry_disposition(outcome, geometry.status)
    _released_bounds(outcome, geometry.intervals)


def _geometry_disposition(outcome: ControlOutcome, status: Literal["feasible", "infeasible"]) -> None:
    """Keep original exact geometry and pre/post-solver failure receipts coherent."""
    if outcome.disposition == "infeasible" and status != "infeasible":
        raise ValueError("A retained outcome contradicts its original exact geometry disposition")
    if (
        status == "infeasible"
        and outcome.disposition != "infeasible"
        and (outcome.disposition == "released" or outcome.solver_inputs is not None)
    ):
        raise ValueError("A retained outcome contradicts its original exact geometry disposition")


def _released_bounds(outcome: ControlOutcome, intervals: tuple[ExactExpertInterval, ...]) -> None:
    """Check historical emitted values against original rational effective boxes."""
    if outcome.disposition == "released" and any(
        not box.lower <= Fraction(weight.value) <= box.upper
        for weight, box in zip(outcome.weights, intervals, strict=True)
    ):
        raise ValueError("A committed release violates its original exact effective box")


def _context_profile(payload: ControlledPayload, outcome: ControlOutcome) -> None:
    """Keep per-release, net-period and accumulated-period context distinct."""
    movement = payload.config.controls.movement
    if movement is None or movement.scope != "per_interval":
        if outcome.movement_usage is not None or outcome.exact_path_before is not None:
            raise ValueError("A nonperiodic release cannot import historical period usage")
        return
    if movement.measure == "net_displacement":
        if outcome.movement_usage is None or outcome.exact_path_before is not None:
            raise ValueError("A net-period attempt requires only its complete original movement usage")
        _context_period(payload, outcome.movement_usage.interval, outcome.movement_usage.origin)
        return
    _accumulated_context(payload, outcome, movement)


def _accumulated_context(payload: ControlledPayload, outcome: ControlOutcome, movement: MovementAllowance) -> None:
    """Join an original accumulated-period context to its admitted fixed law."""
    path = outcome.exact_path_before
    if path is None or outcome.movement_usage is not None:
        raise ValueError("An accumulated-period attempt requires only its complete original exact path")
    if path.movement.to_json() != movement.to_json():
        raise ValueError("A retained exact path must bind the complete fixed movement law")
    _context_period(payload, path.interval, path.period_origin)


def _context_period(payload: ControlledPayload, interval: Reference, origin: CoefficientOrigin) -> None:
    """Check historical period membership and its original static-valid origin."""
    if not any(period.interval == interval for period in payload.periods):
        raise ValueError("A historical movement context must name an originally admitted period")
    _origin(payload, origin)


def _outcome_counts(outcome: ControlOutcome) -> None:
    """Keep successful blocks and retained unconsumed membership arithmetically possible."""
    if (outcome.release_count == 0) != (outcome.consumed_panels == 0):
        raise ValueError("A retained outcome has incompatible nonempty release counters")
    available = (
        outcome.consumed_panels if outcome.disposition == "released" else outcome.update_count - outcome.consumed_panels
    )
    if len(outcome.panels) > available:
        raise ValueError("A retained outcome names more panels than its counter frontier permits")


def _outcome_output(payload: ControlledPayload, outcome: ControlOutcome) -> None:
    """Keep the selected output discriminator and fixed penalty strength exact."""
    if outcome.output is None:
        return
    fixed = payload.config.controls.output
    if isinstance(fixed, ProjectionOutput):
        if not isinstance(outcome.output, ProjectionOutput):
            raise ValueError("A retained outcome cannot change the fixed output profile")
    else:
        expected = fixed.model_dump(mode="json") | {"origin": outcome.origin.model_dump(mode="json")}
        if _encoded(outcome.output.model_dump(mode="json")) != _encoded(expected):
            raise ValueError("A retained outcome cannot change the fixed penalty law")


def _closed_intervals(payload: ControlledPayload) -> None:
    """Retain unique nonoverlapping cadence points and monotone original results."""
    seen = set()
    end = None
    previous = (0, 0, 0)
    for closed in payload.intervals:
        interval = closed.interval
        if payload.last_apply_at is None or interval.release_at > payload.last_apply_at:
            raise ValueError("Closed release cannot follow an unknown or earlier processing frontier")
        if interval.interval in seen or (end is not None and interval.start < end):
            raise ValueError("Closed release intervals cannot repeat, overlap or rewind")
        _outcome_law(payload, closed.outcome)
        counters = (closed.outcome.update_count, closed.outcome.release_count, closed.outcome.consumed_panels)
        if any(old > new for old, new in zip(previous, counters, strict=True)):
            raise ValueError("Closed interval counters cannot rewind")
        if closed.outcome.disposition == "held":
            raise ValueError("A closed interval cannot retain a held-only disposition")
        seen.add(interval.interval)
        end = interval.end
        previous = counters


def _last_outcome(payload: ControlledPayload) -> None:
    """Keep latest receipt aligned while allowing internal candidate staging."""
    outcome = payload.last_outcome
    if outcome is None:
        return
    _outcome_law(payload, outcome)
    counters = (outcome.update_count, outcome.release_count, outcome.consumed_panels)
    current = (payload.update_count, payload.release_count, payload.consumed_panels)
    if counters != current or _weight_bytes(outcome.weights) != _weight_bytes(payload.origin.weights):
        raise ValueError("The latest outcome must retain current counters and emitted coefficients")
    retained = outcome.emitted_origin if outcome.emitted_origin is not None else outcome.origin
    if retained.to_json() != payload.origin.to_json():
        raise ValueError("The latest outcome must retain the actual current origin")


def _failure_required(payload: ControlledPayload) -> bool:
    """Refuse dropping a still-current negative receipt retained elsewhere."""
    latest = payload.last_outcome
    if latest is not None and latest.disposition in {"infeasible", "numerical_failure"}:
        return True
    return any(
        item.outcome.disposition in {"infeasible", "numerical_failure"}
        and (item.outcome.release_count, item.outcome.consumed_panels)
        == (payload.release_count, payload.consumed_panels)
        for item in payload.intervals
    )


def _unresolved_failure(payload: ControlledPayload) -> None:
    """Retain a failed attempt independently from the latest held/empty receipt."""
    failed = payload.unresolved_failure
    if failed is None:
        if _failure_required(payload):
            raise ValueError("A negative latest outcome requires the explicit unresolved failure frontier")
        return
    if failed.disposition not in {"infeasible", "numerical_failure"}:
        raise ValueError("The unresolved failure frontier must retain a negative release outcome")
    _outcome_law(payload, failed)
    if (failed.release_count, failed.consumed_panels) != (payload.release_count, payload.consumed_panels):
        raise ValueError("An unresolved failure cannot survive a later successful release")
    if failed.origin.to_json() != payload.origin.to_json() or (
        _weight_bytes(failed.weights) != _weight_bytes(payload.origin.weights)
    ):
        raise ValueError("The unresolved failure must retain the actual current origin and weights")
    pending = {item.reference.model_dump_json() for item in payload.pending}
    if not {item.model_dump_json() for item in failed.panels}.issubset(pending):
        raise ValueError("Every unresolved failed panel must remain in the complete pending inventory")
    latest = payload.last_outcome
    negative = latest is not None and latest.disposition in {"infeasible", "numerical_failure"}
    if latest is not None and negative and failed.to_json() != latest.to_json():
        raise ValueError("The explicit unresolved failure differs from the latest negative outcome")
