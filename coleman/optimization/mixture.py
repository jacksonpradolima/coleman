"""Joint bounded-simplex proximity and L1 output with qualified emission.

The pre-round minimizer is obtained from exact integer-grid breakpoints. The
emitted binary64 vector is rounded once and checked separately; it is never
renormalized. This module imports no learner, replay, storage or transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, NamedTuple

from coleman.core.actions import Weight
from coleman.core.numeric import SIMPLEX_TOLERANCE
from coleman.core.primitives import MAX_MEMBERS
from coleman.objectives.arithmetic import checked_fraction
from coleman.objectives.controls import (
    CoefficientOrigin,
    ExactMixtureIntervals,
    ExactPathUsage,
    ExpertBounds,
    L1MovementPenalty,
    MixtureIntersection,
    MovementAllowance,
    MovementUsage,
    ProjectionOutput,
    exact_mixture_intervals,
)
from coleman.objectives.preferences import coefficient_weights

SOLVER_PROFILE = "binary64-grid-breakpoint-kkt-v1"
EMISSION_PROFILE = "nearest-binary64-original-box-kkt-v1"
MAX_EVENTS = 4 * MAX_MEMBERS
MAX_INTEGER_BITS = 4096
MAX_COORDINATE_ERROR = Fraction(1, 1 << 53)
_QUANTUM = 1 << 1074
SolveStatus = Literal["optimal", "infeasible", "numerical_failure"]


class MixtureCertificate(NamedTuple):
    """Retain exact pre-round KKT and qualified emitted-output diagnostics.

    Attributes
    ----------
    exact_weights : tuple of Fraction
        Unique exact pre-round minimizer in canonical member order.
    dual : Fraction
        Deterministically selected KKT multiplier; it need not be unique.
    proximity, full_l1, weighted_l1, objective : Fraction
        Emitted half-squared proximity, full coefficient movement, lambda times
        movement and their sum. These are not financial fees or holdings.
    exact_objective : Fraction
        Objective of the exact pre-round minimizer.
    simplex_residual : Fraction
        Absolute emitted sum-one residual, at most 2**-49.
    maximum_rounding_error, stationarity_residual : Fraction
        Largest emitted-coordinate distance from its exact scalar dual
        minimizer, at most 2**-53. The latter is a proximal stationarity metric,
        not a claim that a rounded active face has zero gradient residual.
    objective_error, objective_error_bound : Fraction
        Absolute emitted-versus-exact objective difference and its derived
        lambda-scaled rounding envelope.
    emission_profile : str
        One nearest rounding, original boxes, simplex and KKT qualification.

    Notes
    -----
    All diagnostics are bounded ephemeral rational values, not a JSON codec or
    implicitly representable float measurements. No infinity or underflowed
    nonzero diagnostic is stored. Strict convexity establishes the unique
    exact primal optimizer; binary64 emission has the stated residual envelope.
    """

    exact_weights: tuple[Fraction, ...]
    dual: Fraction
    proximity: Fraction
    full_l1: Fraction
    weighted_l1: Fraction
    objective: Fraction
    exact_objective: Fraction
    simplex_residual: Fraction
    maximum_rounding_error: Fraction
    stationarity_residual: Fraction
    objective_error: Fraction
    objective_error_bound: Fraction
    emission_profile: str = EMISSION_PROFILE


class MixtureSolveResult(NamedTuple):
    """Return a typed output disposition without consuming learner state.

    Attributes
    ----------
    status : {'optimal', 'infeasible', 'numerical_failure'}
        Qualified emission, empty exact feasible set, or failed numerical work.
    reason : str
        Deterministic exact-feasibility or numerical qualification category.
    proposal : tuple of Weight
        Validated original proposal, retaining order and signed input zeros.
    weights : tuple of Weight or None
        Canonical emitted vector only on optimal status; never a repaired input.
    geometry : ExactMixtureIntervals
        Complete exact-input identity and original rational constraints.
    output : ProjectionOutput or L1MovementPenalty
        Complete selected profile, including the current penalty-origin binding.
    certificate : MixtureCertificate or None
        In-process exact/rounded evidence only on qualified optimal output.
    profile : str
        Versioned exact breakpoint implementation convention.

    Notes
    -----
    Invalid declarations raise before solving. A valid empty problem returns
    infeasible, and an unrepresentable rounded optimizer returns numerical
    failure. The owner decides whether to commit a successful nonempty release;
    a successful hold still consumes that opportunity in its owning protocol.
    """

    status: SolveStatus
    reason: str
    proposal: tuple[Weight, ...]
    weights: tuple[Weight, ...] | None
    geometry: ExactMixtureIntervals[MixtureIntersection]
    output: ProjectionOutput | L1MovementPenalty
    certificate: MixtureCertificate | None
    profile: str = SOLVER_PROFILE


class _NumericalError(ValueError):
    """Carry a bounded deterministic refusal without changing caller state."""


@dataclass(frozen=True, slots=True)
class _Coordinate:
    """Keep validated coefficient-grid inputs together during the exact solve."""

    proposal: int
    origin: int
    lower: int
    upper: int


@dataclass(frozen=True, slots=True)
class _Problem:
    """Bind canonical proposal, current origin, exact geometry and strength."""

    proposal: tuple[Weight, ...]
    origin: CoefficientOrigin
    geometry: ExactMixtureIntervals[MixtureIntersection]
    strength: Fraction


def _bounded(value: int | Fraction) -> None:
    """Guard the derived rational work envelope without accepting user rationals."""
    if isinstance(value, Fraction):
        bits = max(value.numerator.bit_length(), value.denominator.bit_length())
    else:
        bits = value.bit_length()
    if bits > MAX_INTEGER_BITS:
        raise _NumericalError("work_limit")


def _grid(value: Fraction) -> int:
    """Convert a derived binary64-grid endpoint without any rounding."""
    scaled = value * _QUANTUM
    _bounded(scaled)
    if scaled.denominator != 1:
        raise _NumericalError("work_limit")
    return scaled.numerator


def _events(coordinates: tuple[_Coordinate, ...], strength: int) -> dict[int, int]:
    """Build at most four exact slope events per coordinate before sorting."""
    if 4 * len(coordinates) > MAX_EVENTS:
        raise _NumericalError("work_limit")
    events: dict[int, int] = {}
    for coordinate in coordinates:
        positive_lower = max(coordinate.lower, coordinate.origin)
        negative_upper = min(coordinate.upper, coordinate.origin)
        if coordinate.upper > positive_lower:
            _add_segment(
                events,
                coordinate.proposal - strength - coordinate.upper,
                coordinate.proposal - strength - positive_lower,
            )
        if negative_upper > coordinate.lower:
            _add_segment(
                events,
                coordinate.proposal + strength - negative_upper,
                coordinate.proposal + strength - coordinate.lower,
            )
    return events


def _add_segment(events: dict[int, int], start: int, end: int) -> None:
    """Group tied breakpoints of one decreasing affine scalar segment."""
    _bounded(start)
    _bounded(end)
    events[start] = events.get(start, 0) - 1
    events[end] = events.get(end, 0) + 1


def _dual(coordinates: tuple[_Coordinate, ...], strength: int) -> Fraction:
    """Sweep grouped affine pieces, solving a crossing algebraically once."""
    events = _events(coordinates, strength)
    total = sum(coordinate.upper for coordinate in coordinates)
    if not events:
        return Fraction()
    ordered = sorted(events)
    previous = ordered[0]
    slope = 0
    for point in ordered:
        if total == _QUANTUM:
            return Fraction(previous)
        following = total + slope * (point - previous)
        _bounded(following)
        if slope < 0 and following <= _QUANTUM < total:
            result = previous + Fraction(total - _QUANTUM, -slope)
            _bounded(result)
            return result
        total = following
        slope += events[point]
        previous = point
    if total != _QUANTUM:
        raise _NumericalError("unresolved_kkt")
    return Fraction(previous)


def _scalar(coordinate: _Coordinate, dual: Fraction, strength: int) -> Fraction:
    """Evaluate the exact shifted soft-threshold map and original box."""
    difference = coordinate.proposal - dual - coordinate.origin
    soft = max(difference - strength, Fraction()) - max(-difference - strength, Fraction())
    return min(Fraction(coordinate.upper), max(Fraction(coordinate.lower), coordinate.origin + soft))


def _optimizer(problem: _Problem) -> tuple[tuple[Fraction, ...], Fraction]:
    """Obtain the exact primal point from bounded integer-grid KKT work."""
    coordinates = tuple(
        _Coordinate(_grid(Fraction(weight.value)), _grid(Fraction(old.value)), _grid(box.lower), _grid(box.upper))
        for weight, old, box in zip(problem.proposal, problem.origin.weights, problem.geometry.intervals, strict=True)
    )
    penalty = _grid(problem.strength)
    dual = _dual(coordinates, penalty)
    solution = tuple(_scalar(coordinate, dual, penalty) / _QUANTUM for coordinate in coordinates)
    if sum(solution, Fraction()) != 1:
        raise _NumericalError("unresolved_kkt")
    for value in solution:
        _bounded(value)
    return solution, dual / _QUANTUM


def _objective(weights: tuple[Fraction, ...], problem: _Problem) -> tuple[Fraction, Fraction, Fraction]:
    """Evaluate separate proximity, full movement and their weighted objective."""
    proximity = (
        sum(
            ((value - Fraction(weight.value)) ** 2 for value, weight in zip(weights, problem.proposal, strict=True)),
            Fraction(),
        )
        / 2
    )
    movement = sum(
        (abs(value - Fraction(old.value)) for value, old in zip(weights, problem.origin.weights, strict=True)),
        Fraction(),
    )
    objective = proximity + problem.strength * movement
    for value in (proximity, movement, objective):
        _bounded(value)
    return proximity, movement, objective


def _certificate(
    exact: tuple[Fraction, ...],
    emitted: tuple[Weight, ...],
    dual: Fraction,
    problem: _Problem,
) -> MixtureCertificate:
    """Qualify the one-rounding error and lambda-scaled objective envelope."""
    values = tuple(Fraction(weight.value) for weight in emitted)
    errors = tuple(abs(value - optimum) for value, optimum in zip(values, exact, strict=True))
    maximum_error = max(errors)
    residual = abs(sum(values, Fraction()) - 1)
    if maximum_error > MAX_COORDINATE_ERROR or residual > SIMPLEX_TOLERANCE:
        raise _NumericalError("rounding_envelope")
    proximity, movement, objective = _objective(values, problem)
    _, _, exact_objective = _objective(exact, problem)
    envelope = sum(
        (
            abs(optimum - Fraction(weight.value)) * error + error * error / 2 + problem.strength * error
            for optimum, weight, error in zip(exact, problem.proposal, errors, strict=True)
        ),
        Fraction(),
    )
    error = abs(objective - exact_objective)
    _bounded(envelope)
    if error > envelope:
        raise _NumericalError("rounding_envelope")
    return MixtureCertificate(
        exact,
        dual,
        proximity,
        movement,
        problem.strength * movement,
        objective,
        exact_objective,
        residual,
        maximum_error,
        maximum_error,
        error,
        envelope,
    )


def _emission(exact: tuple[Fraction, ...], geometry: ExactMixtureIntervals[MixtureIntersection]) -> tuple[Weight, ...]:
    """Round once and reject any violation of the original rational boxes."""
    try:
        emitted = tuple(
            Weight(member=box.member, value=checked_fraction(value))
            for value, box in zip(exact, geometry.intervals, strict=True)
        )
    except ValueError as exc:
        raise _NumericalError("nonrepresentable_output") from exc
    if any(
        not box.lower <= Fraction(weight.value) <= box.upper
        for weight, box in zip(emitted, geometry.intervals, strict=True)
    ):
        raise _NumericalError("rounded_box")
    return emitted


def _selected_output(
    output: ProjectionOutput | L1MovementPenalty, origin: CoefficientOrigin
) -> tuple[ProjectionOutput | L1MovementPenalty, Fraction]:
    """Validate exact penalty-origin identity independently of hard movement."""
    if isinstance(output, L1MovementPenalty):
        selected = L1MovementPenalty.model_validate(output)
        if selected.origin.to_json() != origin.to_json():
            raise ValueError("L1 output requires the exact current emitted origin and version")
        return selected, Fraction(selected.strength)
    return ProjectionOutput.model_validate(output), Fraction()


def _solve_prepared(
    original: tuple[Weight, ...], selected: ProjectionOutput | L1MovementPenalty, problem: _Problem
) -> MixtureSolveResult:
    """Separate mathematical disposition from input validation and identity."""
    if problem.geometry.status == "infeasible":
        return MixtureSolveResult(
            "infeasible", problem.geometry.reason, original, None, problem.geometry, selected, None
        )
    try:
        exact, dual = _optimizer(problem)
        emitted = _emission(exact, problem.geometry)
        certificate = _certificate(exact, emitted, dual, problem)
    except _NumericalError as exc:
        return MixtureSolveResult("numerical_failure", str(exc), original, None, problem.geometry, selected, None)
    return MixtureSolveResult("optimal", "optimal", original, emitted, problem.geometry, selected, certificate)


# Independent optional hard/soft controls remain explicit keyword-only inputs.
def solve_mixture_output(  # pylint: disable=too-many-arguments
    *,
    proposal: tuple[Weight, ...],
    bounds: ExpertBounds,
    origin: CoefficientOrigin,
    output: ProjectionOutput | L1MovementPenalty,
    movement: MovementAllowance | None = None,
    usage: MovementUsage | None = None,
    exact_path: ExactPathUsage | None = None,
) -> MixtureSolveResult:
    """Solve one joint bounded-simplex proximity-plus-L1 output problem.

    Parameters
    ----------
    proposal : tuple of Weight
        Complete finite proposed coefficient simplex, retained without repair.
    bounds : ExpertBounds
        Complete static coefficient boxes.
    origin : CoefficientOrigin
        Actual previous emitted vector and exact version.
    output : ProjectionOutput or L1MovementPenalty
        Unpenalized projection or one joint full-L1 objective. Lambda zero uses
        exactly the same geometry, solve, rounding and qualification path.
    movement : MovementAllowance or None, optional
        Explicit hard movement law, independently absent from the soft penalty.
    usage : MovementUsage or None, optional
        Legacy period attestation under its unchanged interpretation.
    exact_path : ExactPathUsage or None, optional
        Exact emitted-path ledger, mutually exclusive with legacy usage.

    Returns
    -------
    MixtureSolveResult
        Qualified optimal emission, typed exact infeasibility or typed numerical
        failure. No input, learner or movement ledger is mutated.

    Raises
    ------
    ValueError
        Malformed/nonfinite input, wrong members/versions/units or origin join.

    Notes
    -----
    Each coordinate contributes at most two slope-minus-one intervals. Grouped
    integer breakpoints give O(n log n) sorting and O(n) sweep/evaluation work,
    with at most 4n events for n<=4096. Binary64 input-grid numerators need at
    most 2099 bits; sum/root work adds at most ceil(log2(n)) bits. Squared and
    lambda-weighted diagnostics remain below 3200 bits; every stored rational
    is guarded at 4096 bits. No iteration heuristic, clipping/renormalization,
    sequential prox/projection, float dual cancellation or arbitrary Fraction
    input is used. The exact primal is unique by strict convexity; emission is
    qualified under the separate named residual/rounding envelope.
    """
    # pylint: disable-next=unidiomatic-typecheck
    if type(proposal) is not tuple or not 1 <= len(proposal) <= MAX_MEMBERS:
        raise ValueError("Mixture output requires a tuple of one through 4096 weights")
    original = tuple(Weight.model_validate(weight) for weight in proposal)
    ordered = coefficient_weights(original, "binary64-simplex-v1")
    current = CoefficientOrigin.model_validate(origin)
    selected, strength = _selected_output(output, current)
    geometry = exact_mixture_intervals(bounds, current, movement=movement, usage=usage, exact_path=exact_path)
    if tuple(weight.member for weight in ordered) != tuple(box.member for box in geometry.intervals):
        raise ValueError("Mixture proposal requires every exact geometry expert and version")
    return _solve_prepared(original, selected, _Problem(ordered, current, geometry, strength))
