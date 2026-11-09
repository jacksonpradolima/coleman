"""Small coupled convex allocation, Euclidean projection and online steps.

The exact pre-round minimizer is obtained by a bounded rational active-face
search. Only its one-round binary64 emission is accepted, after original-set
and KKT rechecks. This module imports no NumPy, learner, store or transport.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import model_validator

from coleman.core.identity import Reference
from coleman.core.numeric import Binary64, ScalarSpec, binary64
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionLineage
from coleman.optimization._continuous_geometry import linear_geometry, primal_residual
from coleman.optimization._continuous_math import ArithmeticLimitError, bounded, emit, gradient, positive_definite
from coleman.optimization._continuous_solver import solve
from coleman.optimization.continuous_contracts import (
    ContinuousContract,
    ContinuousProblem,
    ContinuousSolution,
    Point,
    ProjectionRequest,
    QuadraticObjective,
)

SOLVER_PROFILE = "small-spd-rational-active-face-binary64-v1"
__all__ = ["ArithmeticLimitError", "ProjectedStep", "project_continuous", "projected_update", "solve_continuous"]


def solve_continuous(problem: ContinuousProblem) -> ContinuousSolution:
    """Minimize one bounded, coupled strictly convex quadratic.

    Parameters
    ----------
    problem : ContinuousProblem
        Complete original dimensionless law, compact simplex/signed boxes,
        explicit cash/net/gross/leverage limits, comparator and numerical caps.

    Returns
    -------
    ContinuousSolution
        optimal with a checked emitted KKT certificate; infeasible only after
        the entire exact face search; unsupported for non-SPD Q; limited on
        face exhaustion; numerical_failure on bit/finite-output failure.

    Raises
    ------
    ValueError, TypeError
        Malformed declarations, dimensions, fields or versions, before work.

    Notes
    -----
    A positive-definite quadratic on a nonempty compact polytope has a unique
    minimizer. Polyhedral KKT conditions are necessary and sufficient, including
    sets without a strict interior. At most n independent active normals are
    needed, or n-1 alongside the simplex equality. Search uses exact rational
    interpretations of stored binary64 inputs. Output is rounded once, without
    clipping, renormalization or diagonal jitter. The accepted point satisfies
    declared absolute feasibility/KKT residuals; no conditioning-independent
    objective-gap guarantee is inferred from those residuals. Iterations are
    deterministic mathematical work, not observations or scientific evidence.
    """
    return solve(ContinuousProblem.model_validate(problem))


def _projection_problem(request: ProjectionRequest) -> ContinuousProblem:
    """Bind the Euclidean minimizer with its constant-offset-free score."""
    size = len(request.proposal)
    objective = QuadraticObjective(
        schema_version=1,
        reference=Reference(id="continuous-euclidean-projection-score", version="1"),
        lineage=DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None),
        matrix=tuple(tuple(float(i == j) for j in range(size)) for i in range(size)),
        linear=tuple(-value for value in request.proposal),
        constant=0.0,
        score=ScalarSpec(
            dimension=Reference(id="continuous-euclidean-projection-score", version="1"),
            role="objective",
            semantics=UnitSemantics(unit="1", meaning="half-squared-norm-minus-original-proposal-dot-point"),
        ),
    )
    return ContinuousProblem(
        schema_version=1,
        feasible_set=request.feasible_set,
        objective=objective,
        comparator=request.comparator,
        tolerances=request.tolerances,
        limits=request.limits,
    )


def project_continuous(request: ProjectionRequest) -> ContinuousSolution:
    """Project an original proposal onto its complete declared convex set.

    Parameters
    ----------
    request : ProjectionRequest
        Finite dimensionless proposal, original set, comparator and limits.

    Returns
    -------
    ContinuousSolution
        Complete projection problem/result, retaining the original proposal's
        bytes. Its score omits the constant half-squared proposal norm, and is
        explicitly not a reported squared distance or realized transition cost.

    Raises
    ------
    ValueError, TypeError
        Malformed shapes, limits, quantities or versions before solving.
    """
    checked = ProjectionRequest.model_validate(request)
    result = solve_continuous(_projection_problem(checked))
    return ContinuousSolution.model_validate(result.model_dump() | {"proposal": checked.proposal})


class ProjectedStep(ContinuousContract):
    """Retain one per-loss projected update, independently of any runtime.

    Attributes
    ----------
    problem : ContinuousProblem
        Original per-round quadratic loss, set, comparator and transition model.
    current, gradient : tuple
        Original feasible allocation and analytic gradient at that allocation.
    step_size : float
        Finite nonnegative dimensionless rate, including the identity endpoint0.
    projection : ContinuousSolution
        Proposed x-step*gradient and separately checked executed allocation.
        Nonoptimal projections supply no fallback or silently accepted action.

    Notes
    -----
    This stateless mathematical call supplies no labels, temporal eligibility,
    event deduplication, RNG or checkpoint owner. A caller's online observation
    count is distinct from projection faces. External-regret comparisons on the
    same per-round loss functions do not bound comparator-specific wealth or
    transition/accounting paths. No regret theorem is implemented by this DTO.
    """

    problem: ContinuousProblem
    current: Point
    gradient: Point
    step_size: Binary64
    projection: ContinuousSolution

    @model_validator(mode="after")
    def _joins(self) -> Self:
        """Rederive the original gradient, proposal and projection law."""
        exact, proposal = _step_values(self.problem, self.current, self.step_size)
        if self.gradient != emit(exact) or self.projection.proposal != proposal:
            raise ValueError("Projected step differs from its original loss/point/rate")
        request = _step_request(self.problem, proposal)
        if self.projection.problem.to_json() != _projection_problem(request).to_json():
            raise ValueError("Projected step changes its complete original projection context")
        return self


def _step_request(problem: ContinuousProblem, proposal: tuple[float, ...]) -> ProjectionRequest:
    """Carry original geometry, comparator and envelopes into projection."""
    return ProjectionRequest(
        schema_version=1,
        feasible_set=problem.feasible_set,
        proposal=proposal,
        comparator=problem.comparator,
        tolerances=problem.tolerances,
        limits=problem.limits,
    )


def _step_values(
    problem: ContinuousProblem, current: tuple[float, ...], step_size: float
) -> tuple[tuple[Fraction, ...], tuple[float, ...]]:
    """Validate supported convex loss and form one exact pre-round proposal."""
    if len(current) != len(problem.objective.linear) or step_size < 0:
        raise ValueError("Projected step requires matching dimension and a nonnegative rate")
    exact = tuple(Fraction(value) for value in current)
    if primal_residual(exact, linear_geometry(problem.feasible_set)) > Fraction(problem.tolerances.primal):
        raise ValueError("Projected step requires a feasible original point")
    matrix = tuple(tuple(Fraction(x) for x in row) for row in problem.objective.matrix)
    if not positive_definite(matrix, problem.limits.maximum_integer_bits):
        raise ValueError("Projected step supports the declared strictly convex quadratic loss profile")
    derivative = gradient(problem.objective, exact, problem.limits.maximum_integer_bits)
    proposal = emit(
        tuple(
            bounded(x - Fraction(step_size) * g, problem.limits.maximum_integer_bits)
            for x, g in zip(exact, derivative, strict=True)
        )
    )
    return derivative, proposal


def projected_update(problem: ContinuousProblem, current: tuple[float, ...], step_size: float) -> ProjectedStep:
    """Apply one projected analytic-gradient step for a supplied per-round loss.

    Parameters
    ----------
    problem : ContinuousProblem
        This round's complete SPD quadratic loss on a fixed compact set, with
        dimensionless objective/coefficients and explicit comparison context.
    current : tuple of float
        Original allocation feasible within the named primal tolerance.
    step_size : float
        Finite nonnegative dimensionless rate;0 still rechecks the same set.

    Returns
    -------
    ProjectedStep
        Original loss, point, gradient and proposal plus actual projection or
        typed nonoptimal result. No state or observation counter is advanced.

    Raises
    ------
    ValueError, TypeError
        Invalid dimensions, infeasible starting point, non-SPD loss or rate.
    ArithmeticLimitError
        The bounded gradient/proposal cannot be emitted in finite binary64.
        No partial update or silently jittered proposal is returned.
    """
    checked = ContinuousProblem.model_validate(problem)
    point = _step_request(checked, current).proposal
    rate = binary64(step_size)
    exact, proposal = _step_values(checked, point, rate)
    return ProjectedStep(
        schema_version=1,
        problem=checked,
        current=point,
        gradient=emit(exact),
        step_size=rate,
        projection=project_continuous(_step_request(checked, proposal)),
    )
