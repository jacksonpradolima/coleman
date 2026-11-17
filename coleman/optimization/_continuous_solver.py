"""Deterministic exact active-face search for a small positive-definite QP."""

from fractions import Fraction
from itertools import combinations
from typing import NamedTuple

from coleman.optimization._continuous_checks import certificate
from coleman.optimization._continuous_geometry import Geometry, linear_geometry, primal_residual
from coleman.optimization._continuous_math import ArithmeticLimitError, bounded, emit, positive_definite, solve_linear
from coleman.optimization.continuous_contracts import (
    ContinuousCertificate,
    ContinuousProblem,
    ContinuousSolution,
    KktResiduals,
)


class ExactCandidate(NamedTuple):
    """Keep a finite exact KKT primal and its original ordered duals."""

    point: tuple[Fraction, ...]
    duals: tuple[Fraction, ...]
    equality_dual: Fraction | None


def _face(problem: ContinuousProblem, geometry: Geometry, active: tuple[int, ...]) -> ExactCandidate | None:
    """Solve one exact active face; dependent faces are explicitly skipped."""
    objective = problem.objective
    size = len(objective.linear)
    equations = (() if geometry.equality is None else (geometry.equality,)) + tuple(
        geometry.inequalities[index] for index in active
    )
    matrix = tuple(
        tuple(Fraction(x) for x in row) + tuple(item.normal[index] for item in equations)
        for index, row in enumerate(objective.matrix)
    ) + tuple(item.normal + (Fraction(),) * len(equations) for item in equations)
    right = tuple(-Fraction(x) for x in objective.linear) + tuple(item.bound for item in equations)
    solution = solve_linear(matrix, right, problem.limits.maximum_integer_bits)
    if solution is None:
        return None
    point = solution[:size]
    offset = size + int(geometry.equality is not None)
    if any(mu < 0 for mu in solution[offset:]) or primal_residual(point, geometry):
        return None
    duals = [Fraction()] * len(geometry.inequalities)
    for index, multiplier in zip(active, solution[offset:], strict=True):
        duals[index] = multiplier
    equality = None if geometry.equality is None else solution[size]
    return ExactCandidate(point, tuple(duals), equality)


def _search(problem: ContinuousProblem, geometry: Geometry) -> tuple[ExactCandidate | None, int, bool]:
    """Enumerate independent-size faces, distinguishing exhaustion from empty."""
    count = 0
    dimension = len(problem.objective.linear) - int(geometry.equality is not None)
    for size in range(dimension + 1):
        for active in combinations(range(len(geometry.inequalities)), size):
            if count >= problem.limits.maximum_faces:
                return None, count, False
            count += 1
            try:
                candidate = _face(problem, geometry, active)
            except ArithmeticLimitError as error:
                raise ArithmeticLimitError(str(error), faces_examined=count) from error
            if candidate is not None:
                return candidate, count, True
    return None, count, True


def _outcome(
    problem: ContinuousProblem, status: str, reason: str, count: int, proposal: tuple[float, ...] | None
) -> ContinuousSolution:
    """Construct a closed refusal with no accepted point or fake certificate."""
    return ContinuousSolution.model_validate(
        {
            "schema_version": 1,
            "problem": problem,
            "status": status,
            "reason": reason,
            "proposal": proposal,
            "executed": None,
            "certificate": None,
            "faces_examined": count,
        }
    )


def _qualified(
    problem: ContinuousProblem, candidate: ExactCandidate, count: int, proposal: tuple[float, ...] | None
) -> ContinuousSolution:
    """Round once and qualify the original KKT system without repairs."""
    point, duals = emit(candidate.point), emit(candidate.duals)
    equality = None if candidate.equality_dual is None else emit((candidate.equality_dual,))[0]
    proof = certificate(problem, point, duals, equality)
    return ContinuousSolution(
        schema_version=1,
        problem=problem,
        status="optimal",
        reason="exact_face_then_checked_binary64",
        proposal=proposal,
        executed=point,
        certificate=ContinuousCertificate(
            schema_version=1,
            value=proof.value,
            gradient=proof.gradient,
            duals=proof.duals,
            equality_dual=proof.equality_dual,
            residuals=KktResiduals(schema_version=1, **proof.residuals._asdict()),
        ),
        faces_examined=count,
    )


def solve(problem: ContinuousProblem) -> ContinuousSolution:
    """Run the finite positive-definite profile on an already validated problem.

    Parameters
    ----------
    problem : ContinuousProblem
        Original bounded n<=4 SPD objective, geometry and fixed numerical caps.

    Returns
    -------
    ContinuousSolution
        Qualified nearest emission or a closed typed disposition without output.
        Resource exhaustion never constitutes an infeasibility proof.
    """
    count, proposal = 0, None
    try:
        maximum = problem.limits.maximum_integer_bits
        matrix = tuple(tuple(bounded(Fraction(x), maximum) for x in row) for row in problem.objective.matrix)
        if not positive_definite(matrix, maximum):
            return _outcome(problem, "unsupported", "objective_not_positive_definite", 0, None)
        unconstrained = solve_linear(matrix, tuple(-Fraction(x) for x in problem.objective.linear), maximum)
        if unconstrained is not None:
            proposal = _optional_proposal(unconstrained)
        candidate, count, complete = _search(problem, linear_geometry(problem.feasible_set))
        if candidate is not None:
            return _qualified(problem, candidate, count, proposal)
        if not complete:
            return _outcome(problem, "limited", "maximum_faces_exhausted", count, proposal)
        return _outcome(problem, "infeasible", "empty_exact_feasible_set", count, proposal)
    except ArithmeticLimitError as error:
        return _outcome(problem, "numerical_failure", str(error), max(count, error.faces_examined), proposal)


def _optional_proposal(values: tuple[Fraction, ...]) -> tuple[float, ...] | None:
    """Keep an unrepresentable unconstrained proposal absent without repair."""
    try:
        return emit(values)
    except ArithmeticLimitError:
        return None
