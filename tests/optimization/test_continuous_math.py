"""Coupled optima, exact signed constraints and projected online arithmetic."""

from fractions import Fraction

import pytest

from coleman.optimization.continuous import project_continuous, projected_update, solve_continuous
from coleman.optimization.continuous_contracts import (
    ContinuousProblem,
    ContinuousSet,
    ProjectionRequest,
    SolveLimits,
    SolverTolerances,
)
from tests.optimization._continuous_cases import geometry, interval, problem


def quadratic(point, matrix, linear):
    """Evaluate the independent exact scalar definition, without solver helpers."""
    x = tuple(Fraction(value) for value in point)
    return sum(x[i] * Fraction(matrix[i][j]) * x[j] / 2 for i in range(len(x)) for j in range(len(x))) + sum(
        Fraction(c) * value for c, value in zip(linear, x, strict=True)
    )


def test_coupled_simplex_optimum_and_original_kkt_have_an_exact_hand_derivation():
    declared = problem()
    result = solve_continuous(declared)
    assert result.status == "optimal"
    assert result.proposal == (1.0, 0.5)
    assert result.executed == (0.75, 0.25)
    proof = result.certificate
    assert proof is not None
    assert proof.value == -25 / 16
    assert proof.gradient == (-0.75, -0.75)
    assert proof.equality_dual == 0.75
    assert proof.residuals.primal == 0
    assert proof.residuals.stationarity == 0
    assert proof.residuals.complementarity == 0
    clipped = (Fraction(1), Fraction(1, 2))
    assert sum(clipped) != 1
    repaired = (Fraction(2, 3), Fraction(1, 3))
    assert quadratic(repaired, declared.objective.matrix, declared.objective.linear) + Fraction(25, 16) == Fraction(
        1, 144
    )


def test_active_box_coupling_moves_the_unclipped_coordinate_instead_of_clipping():
    feasible = geometry("signed-box-v1", ((-1.0, 0.5), (-1.0, 2.0)))
    declared = problem(feasible)
    result = solve_continuous(declared)
    assert result.status == "optimal"
    assert result.executed == (0.5, 0.75)
    assert result.proposal == (1.0, 0.5)
    assert result.certificate is not None
    assert result.certificate.gradient == (-0.75, 0.0)
    assert result.certificate.duals[0] == 0.75
    clipped = (0.5, 0.5)
    assert quadratic(clipped, declared.objective.matrix, declared.objective.linear) == Fraction(-3, 2)
    assert result.certificate.value == -25 / 16


def test_signed_gross_and_cash_inclusive_leverage_are_distinct_actual_constraints():
    feasible = geometry("signed-box-v1", ((-2.0, 2.0), (-2.0, 2.0)), gross=1.0, leverage=2.0)
    declared = problem(feasible, matrix=((1.0, 0.0), (0.0, 1.0)), linear=(1.0, -1.0))
    result = solve_continuous(declared)
    assert result.status == "optimal"
    assert result.executed == (-0.5, 0.5)
    assert result.executed is not None
    assert sum(abs(x) for x in result.executed) == 1
    assert 1 - sum(result.executed) == 1
    assert sum(abs(x) for x in result.executed) + abs(1 - sum(result.executed)) == 2
    stricter = ContinuousSet.model_validate(feasible.model_dump() | {"leverage_limit": 1.0})
    held = solve_continuous(problem(stricter, matrix=((1.0, 0.0), (0.0, 1.0)), linear=(1.0, -1.0)))
    assert held.status == "optimal"
    assert held.executed == (0.0, 1.0)


def test_zero_gross_is_valid_and_keeps_explicit_unit_cash():
    feasible = geometry("signed-box-v1", ((-1.0, 1.0), (-1.0, 1.0)), gross=0.0, leverage=1.0)
    result = solve_continuous(problem(feasible))
    assert result.status == "optimal"
    assert result.executed == (0.0, 0.0)


@pytest.mark.parametrize("change", [{"net": interval(0.0, 0.0)}, {"cash": interval(1.0, 1.0)}, {"gross_limit": 0.5}])
def test_jointly_empty_simplex_geometry_returns_infeasible_not_a_repaired_vector(change):
    feasible = ContinuousSet.model_validate(geometry().model_dump() | change)
    result = solve_continuous(problem(feasible))
    assert result.status == "infeasible"
    assert result.executed is None
    assert result.certificate is None


@pytest.mark.parametrize("matrix", [((0.0, 0.0), (0.0, 0.0)), ((1.0, 1.0), (1.0, 1.0)), ((1.0, 2.0), (2.0, 1.0))])
def test_singular_or_nonconvex_objective_is_unsupported_without_jitter(matrix):
    result = solve_continuous(problem(matrix=matrix))
    assert result.status == "unsupported"
    assert result.faces_examined == 0
    assert result.executed is None


def test_a_face_limit_is_not_an_infeasibility_certificate():
    feasible = geometry(boxes=((0.0, 0.5), (0.0, 1.0)))
    declared = problem(feasible)
    limited = ContinuousProblem.model_validate(
        declared.model_dump() | {"limits": SolveLimits(schema_version=1, maximum_faces=1)}
    )
    refused = solve_continuous(limited)
    assert refused.status == "limited"
    assert refused.faces_examined == 1
    accepted = solve_continuous(declared)
    assert accepted.status == "optimal"
    assert accepted.executed == (0.5, 0.5)


@pytest.mark.parametrize("scale", [0.125, 1.0, 8.0])
def test_positive_objective_scaling_keeps_the_original_optimizer(scale):
    declared = problem(matrix=((2 * scale, scale), (scale, 2 * scale)), linear=(-2.5 * scale, -2 * scale))
    result = solve_continuous(declared)
    assert result.status == "optimal"
    assert result.executed == (0.75, 0.25)
    assert result.certificate is not None
    assert result.certificate.value == -25 * scale / 16


def test_consistent_coordinate_permutation_preserves_the_physical_vector():
    original = solve_continuous(problem())
    assert original.executed is not None
    permuted = solve_continuous(problem(linear=(-2.0, -2.5)))
    assert permuted.status == "optimal"
    assert permuted.executed == tuple(reversed(original.executed))


def test_euclidean_projection_is_an_actual_coupled_solve_and_keeps_its_input():
    declared = problem()
    request = ProjectionRequest(
        schema_version=1, feasible_set=declared.feasible_set, proposal=(2.0, 0.0), comparator=declared.comparator
    )
    result = project_continuous(request)
    assert result.status == "optimal"
    assert result.proposal == (2.0, 0.0)
    assert result.executed == (1.0, 0.0)
    assert request.proposal == (2.0, 0.0)


def test_projected_gradient_uses_one_original_loss_and_preserves_proposed_vs_executed():
    declared = problem()
    step = projected_update(declared, (0.5, 0.5), 1.0)
    assert step.gradient == (-1.0, -0.5)
    assert step.projection.proposal == (1.5, 1.0)
    assert step.projection.executed == (0.75, 0.25)
    assert step.problem.fingerprint == declared.fingerprint
    assert step.projection.problem.objective.fingerprint != declared.objective.fingerprint
    zero = projected_update(declared, (0.5, 0.5), 0.0)
    assert zero.projection.proposal == (0.5, 0.5)
    assert zero.projection.executed == (0.5, 0.5)


def test_analytic_gradient_matches_independent_symmetric_difference():
    declared = problem()
    step = projected_update(declared, (0.5, 0.5), 0.0)
    point = (Fraction(1, 2),) * 2
    epsilon = Fraction(1, 1024)
    for index in range(2):
        plus, minus = list(point), list(point)
        plus[index] += epsilon
        minus[index] -= epsilon
        difference = (
            quadratic(plus, declared.objective.matrix, declared.objective.linear)
            - quadratic(minus, declared.objective.matrix, declared.objective.linear)
        ) / (2 * epsilon)
        assert step.gradient[index] == difference


def test_exact_zero_tolerances_can_refuse_an_unrepresentable_emission_without_clipping():
    feasible = geometry(boxes=((0.0, 1.0),) * 3)
    declared = problem(feasible, matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), linear=(0.0, 0.0, 0.0))
    strict = ContinuousProblem.model_validate(
        declared.model_dump()
        | {"tolerances": SolverTolerances(schema_version=1, primal=0.0, stationarity=0.0, complementarity=0.0)}
    )
    result = solve_continuous(strict)
    assert result.status == "numerical_failure"
    assert result.executed is None
    assert result.certificate is None


@pytest.mark.parametrize("count", [1, 4])
def test_declared_dimension_endpoints_use_the_same_real_simplex_solver(count):
    """One and four coordinates have exact representable equal-weight minima."""
    feasible = geometry(boxes=((0.0, 1.0),) * count)
    matrix = tuple(tuple(float(i == j) for j in range(count)) for i in range(count))
    result = solve_continuous(problem(feasible, matrix=matrix, linear=(0.0,) * count))
    assert result.status == "optimal"
    assert result.executed == (1.0 / count,) * count


def test_cash_and_net_limits_intersect_without_renaming_one_as_the_other():
    """Cash is exactly the retained complement of original noncash net."""
    base = geometry("signed-box-v1", ((-2.0, 2.0),) * 2)
    feasible = ContinuousSet.model_validate(
        base.model_dump() | {"net": interval(0.0, 0.5), "cash": interval(0.75, 1.0)}
    )
    result = solve_continuous(problem(feasible, matrix=((1.0, 0.0), (0.0, 1.0)), linear=(-2.0, -2.0)))
    assert result.status == "optimal"
    assert result.executed == (0.125, 0.125)


def test_a_fixed_signed_allocation_is_valid_without_a_strict_interior():
    """Polyhedral KKT handles singleton feasible sets without Slater fiction."""
    feasible = geometry("signed-box-v1", ((-0.25, -0.25), (0.75, 0.75)), gross=1.0, leverage=1.5)
    result = solve_continuous(problem(feasible))
    assert result.status == "optimal"
    assert result.executed == (-0.25, 0.75)


def test_additive_score_offset_changes_only_reported_objective():
    """A constant is not an action transition or a realized paid charge."""
    declared = problem()
    objective = type(declared.objective).model_validate(declared.objective.model_dump() | {"constant": 17.0})
    shifted = ContinuousProblem.model_validate(declared.model_dump() | {"objective": objective})
    original = solve_continuous(declared)
    result = solve_continuous(shifted)
    assert result.executed == original.executed
    assert result.certificate is not None
    assert original.certificate is not None
    assert result.certificate.value == original.certificate.value + 17.0


def test_independent_finite_grid_has_an_explicit_nonzero_discretization_gap():
    """A grid corroborates a bound and is not called the continuous optimum."""
    declared = problem()
    rows = ((Fraction(k, 7), Fraction(7 - k, 7)) for k in range(8))
    grid_minimum = min(quadratic(point, declared.objective.matrix, declared.objective.linear) for point in rows)
    exact_minimum = Fraction(-25, 16)
    assert grid_minimum - exact_minimum == Fraction(1, 784)
    assert grid_minimum - exact_minimum <= Fraction(1, 196)
    actual = solve_continuous(declared)
    assert actual.certificate is not None
    assert actual.certificate.value == exact_minimum
