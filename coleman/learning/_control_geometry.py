"""Pure current-origin and exact movement-frontier validation."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from coleman.objectives.arithmetic import checked_fraction
from coleman.objectives.controls import CoefficientOrigin, MovementLimit, MovementUsage, exact_mixture_intervals

if TYPE_CHECKING:
    from coleman.learning.control_contracts import ControlledPayload


def _origin(payload: ControlledPayload, origin: CoefficientOrigin) -> None:
    """Join current or period-start coefficients to exact original static boxes."""
    bounds = payload.config.controls.bounds.bounds
    if tuple(item.member for item in origin.weights) != tuple(item.member for item in bounds):
        raise ValueError("An emitted origin must retain every exact configured member")
    if any(not bound.lower <= item.value <= bound.upper for item, bound in zip(origin.weights, bounds, strict=True)):
        raise ValueError("An emitted origin must satisfy every original static box")


def _period_history(payload: ControlledPayload) -> None:
    """Join monotone explicit periods under the fixed per-interval movement law."""
    if bool(payload.periods) != (payload.period_origin is not None):
        raise ValueError("Movement period history and its origin must be present together")
    if payload.config.period_clock is None and (payload.periods or payload.exact_path is not None):
        raise ValueError("A law without per-interval movement cannot retain period state")
    seen = set()
    end = None
    for period in payload.periods:
        if period.interval in seen or period.calendar != payload.config.period_clock:
            raise ValueError("Movement periods require unique identities under the declared clock")
        if end is not None and period.start < end:
            raise ValueError("Movement periods cannot overlap or rewind")
        seen.add(period.interval)
        end = period.end
    if payload.period_origin is not None:
        _origin(payload, payload.period_origin)
        current = payload.periods[-1]
        if payload.last_apply_at is None or not current.start <= payload.last_apply_at < current.end:
            raise ValueError("An admitted movement period requires its original processing frontier")


def _geometry(payload: ControlledPayload) -> None:
    """Check original geometry joins while retaining a well-formed empty exact set."""
    movement = payload.config.controls.movement
    periodic = movement is not None and movement.scope == "per_interval"
    accumulated = movement is not None and periodic and movement.measure == "accumulated_absolute"
    if (bool(payload.periods) and accumulated) != (payload.exact_path is not None):
        raise ValueError("The accumulated movement period requires its complete exact ledger")
    if periodic and not payload.periods:
        if payload.update_count or payload.last_apply_at is not None:
            raise ValueError("Per-interval movement requires an admitted current period")
        return
    _path_binding(payload)
    exact_mixture_intervals(
        payload.config.controls.bounds,
        payload.origin,
        movement=movement,
        usage=_net_usage(payload),
        exact_path=payload.exact_path,
    )


def _path_binding(payload: ControlledPayload) -> None:
    """Require complete fixed law, current period and exact origin ledger joins."""
    path = payload.exact_path
    if path is None:
        return
    if path.interval != payload.periods[-1].interval:
        raise ValueError("Exact usage belongs to another movement period")
    movement, start = payload.config.controls.movement, payload.period_origin
    if movement is None or start is None:
        raise ValueError("Exact usage requires the declared movement law and period-start origin")
    if path.movement.to_json() != movement.to_json():
        raise ValueError("Exact usage must retain the complete fixed movement law")
    if path.period_origin.to_json() != start.to_json():
        raise ValueError("Exact usage must retain the original period-start origin")
    if path.through_origin != payload.origin.binding:
        raise ValueError("Exact usage must bind the current actual emitted origin")


def _net_usage(payload: ControlledPayload) -> MovementUsage | None:
    """Derive the net display while geometry retains original exact origins."""
    movement = payload.config.controls.movement
    if movement is None or movement.scope != "per_interval" or movement.measure != "net_displacement":
        return None
    if payload.period_origin is None:
        raise ValueError("Per-interval net movement requires its period-start origin")
    used = tuple(
        MovementLimit(member=old.member, maximum=checked_fraction(abs(Fraction(new.value) - Fraction(old.value))))
        for old, new in zip(payload.period_origin.weights, payload.origin.weights, strict=True)
    )
    return MovementUsage(interval=payload.periods[-1].interval, origin=payload.period_origin, used=used)
