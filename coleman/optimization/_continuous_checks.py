"""Recompute bounded emission certificates without active-face enumeration."""

from fractions import Fraction
from typing import TYPE_CHECKING, NamedTuple

from coleman.core.primitives import canonical_json
from coleman.optimization._continuous_geometry import Geometry, dot, linear_geometry, primal_residual
from coleman.optimization._continuous_math import (
    ArithmeticLimitError,
    bounded,
    emit,
    gradient,
    objective_value,
    positive_definite,
    solve_linear,
)

if TYPE_CHECKING:
    from coleman.optimization.continuous_contracts import ContinuousProblem, ContinuousSolution


class ResidualValues(NamedTuple):
    """Keep ephemeral original-set residuals without an upward codec import."""

    primal: float
    stationarity: float
    complementarity: float


class CertificateValues(NamedTuple):
    """Keep scalar certificate computation separate from the wire DTO."""

    value: float
    gradient: tuple[float, ...]
    duals: tuple[float, ...]
    equality_dual: float | None
    residuals: ResidualValues


def residuals(
    problem: ContinuousProblem,
    point: tuple[float, ...],
    duals: tuple[float, ...],
    equality_dual: float | None,
) -> ResidualValues:
    """Recompute original primal and KKT residuals on emitted binary64 inputs.

    Parameters
    ----------
    problem : ContinuousProblem
        Original fixed set, objective, tolerance and bit-work declarations.
    point, duals : tuple of float
        Emitted coordinates and multipliers in compiled original order.
    equality_dual : float or None
        Sum-one multiplier exactly when that equality is present.

    Returns
    -------
    ResidualValues
        Absolute residual diagnostics rounded only after exact qualification.

    Raises
    ------
    ArithmeticLimitError
        The exact residual exceeds its threshold, cap or output range.
    """
    geometry = linear_geometry(problem.feasible_set)
    primal = tuple(Fraction(x) for x in point)
    multipliers = tuple(Fraction(x) for x in duals)
    stationarity = list(gradient(problem.objective, primal, problem.limits.maximum_integer_bits))
    _dual_gradient(stationarity, geometry, multipliers, equality_dual)
    complementarity = max(
        (
            abs(mu * (dot(row.normal, primal) - row.bound))
            for mu, row in zip(multipliers, geometry.inequalities, strict=True)
        ),
        default=Fraction(),
    )
    raw = (primal_residual(primal, geometry), max(abs(x) for x in stationarity), complementarity)
    thresholds = (problem.tolerances.primal, problem.tolerances.stationarity, problem.tolerances.complementarity)
    if any(error > Fraction(limit) for error, limit in zip(raw, thresholds, strict=True)):
        raise ArithmeticLimitError("emitted_kkt_tolerance")
    reported = emit(tuple(bounded(item, problem.limits.maximum_integer_bits) for item in raw))
    return ResidualValues(primal=reported[0], stationarity=reported[1], complementarity=reported[2])


def _dual_gradient(
    stationarity: list[Fraction], geometry: Geometry, multipliers: tuple[Fraction, ...], equality_dual: float | None
) -> None:
    """Accumulate original normal multipliers without rounding partial sums."""
    for multiplier, row in zip(multipliers, geometry.inequalities, strict=True):
        for index, normal in enumerate(row.normal):
            stationarity[index] += multiplier * normal
    if geometry.equality is not None and equality_dual is not None:
        for index, normal in enumerate(geometry.equality.normal):
            stationarity[index] += Fraction(equality_dual) * normal


def certificate(
    problem: ContinuousProblem, point: tuple[float, ...], duals: tuple[float, ...], equality_dual: float | None
) -> CertificateValues:
    """Build evidence for a fresh emitted point under the original objective.

    Parameters
    ----------
    problem : ContinuousProblem
        Complete scalar law and original geometry/qualification envelope.
    point, duals : tuple of float
        Actual emitted coordinates and original ordered inequality multipliers.
    equality_dual : float or None
        Declared-profile equality multiplier, or null when absent.

    Returns
    -------
    CertificateValues
        Ephemeral typed diagnostics; no wire object or owned state is mutated.
    """
    exact = tuple(Fraction(x) for x in point)
    return CertificateValues(
        value=emit((objective_value(problem.objective, exact, problem.limits.maximum_integer_bits),))[0],
        gradient=emit(gradient(problem.objective, exact, problem.limits.maximum_integer_bits)),
        duals=duals,
        equality_dual=equality_dual,
        residuals=residuals(problem, point, duals, equality_dual),
    )


def validate_solution(result: ContinuousSolution) -> None:
    """Recheck bounded certificate/content joins, never solve the active faces.

    Parameters
    ----------
    result : ContinuousSolution
        Complete known-version result at its guarded DTO validation boundary.

    Raises
    ------
    ValueError
        Closed disposition, exact input or emitted KKT content is inconsistent.
    """
    dimension = len(result.problem.feasible_set.variables)
    if result.faces_examined > result.problem.limits.maximum_faces:
        raise ValueError("Solution exceeds its declared face-work limit")
    if result.proposal is not None and len(result.proposal) != dimension:
        raise ValueError("Unconstrained proposal has the wrong coordinate dimension")
    if result.status != "optimal":
        if result.executed is not None or result.certificate is not None:
            raise ValueError("Nonoptimal disposition cannot claim an accepted allocation")
        return
    if result.executed is None or result.certificate is None or len(result.executed) != dimension:
        raise ValueError("Optimal disposition requires every executed coordinate and certificate")
    _optimal_certificate(result)


def _optimal_certificate(result: ContinuousSolution) -> None:
    """Validate exact input convexity and rederive all emitted diagnostics."""
    problem, proof = result.problem, result.certificate
    if proof is None or result.executed is None:
        raise ValueError("Missing optimal evidence")
    geometry = linear_geometry(problem.feasible_set)
    matrix = tuple(tuple(Fraction(x) for x in row) for row in problem.objective.matrix)
    try:
        if not positive_definite(matrix, problem.limits.maximum_integer_bits):
            raise ValueError("Optimal certificate requires this profile's positive-definite objective")
        _check_proposal(result, matrix)
        if len(proof.duals) != len(geometry.inequalities) or any(mu < 0 for mu in proof.duals):
            raise ValueError("Certificate requires a nonnegative multiplier for every original inequality")
        if (geometry.equality is None) != (proof.equality_dual is None):
            raise ValueError("Certificate equality multiplier differs from the declared profile")
        expected = certificate(problem, result.executed, proof.duals, proof.equality_dual)
    except ArithmeticLimitError as error:
        raise ValueError("Certificate exceeds the declared arithmetic envelope") from error
    expected_wire = dict(expected._asdict())
    expected_wire["schema_version"] = 1
    expected_wire["gradient"] = list(expected.gradient)
    expected_wire["duals"] = list(expected.duals)
    expected_wire["residuals"] = {"schema_version": 1, **expected.residuals._asdict()}
    if proof.to_json() != canonical_json(expected_wire):
        raise ValueError("Certificate differs from independently recomputed emitted diagnostics")


def _check_proposal(result: ContinuousSolution, matrix: tuple[tuple[Fraction, ...], ...]) -> None:
    """Check the retained unconstrained point without active-face enumeration."""
    exact = solve_linear(
        matrix, tuple(-Fraction(x) for x in result.problem.objective.linear), result.problem.limits.maximum_integer_bits
    )
    expected = None
    if exact is not None:
        try:
            expected = emit(exact)
        except ArithmeticLimitError:
            expected = None
    if result.proposal != expected:
        raise ValueError("Retained proposal differs from the original unconstrained quadratic minimizer")
