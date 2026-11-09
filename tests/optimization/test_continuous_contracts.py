"""Strict continuous wire, original-law joins and substantive refusal cases."""

import json
from math import copysign

import pytest

from coleman.core.numeric import ScalarSpec
from coleman.core.units import UnitSemantics
from coleman.optimization.continuous import ArithmeticLimitError, ProjectedStep, projected_update, solve_continuous
from coleman.optimization.continuous_contracts import (
    CoefficientInterval,
    ContinuousProblem,
    ContinuousSet,
    ContinuousSolution,
    ProjectionRequest,
    QuadraticObjective,
    SolveLimits,
)
from tests.optimization._continuous_cases import geometry, problem


@pytest.mark.parametrize("reader", ["from_json", "model_validate_json"])
def test_complete_problem_and_output_wire_round_trip_recheck_the_same_certificate(reader):
    declared = problem()
    outcome = solve_continuous(declared)
    step = projected_update(declared, (0.5, 0.5), 1.0)
    for value in (declared, outcome, step):
        restored = getattr(type(value), reader)(value.to_json())
        assert restored.to_json() == value.to_json()
        assert restored.fingerprint == value.fingerprint


@pytest.mark.parametrize("version", [False, 0, 2, "1"])
@pytest.mark.parametrize("kind", ["problem", "solution", "step"])
def test_future_unknown_and_coerced_versions_reject_for_every_public_output(version, kind):
    declared = problem()
    values = {
        "problem": declared,
        "solution": solve_continuous(declared),
        "step": projected_update(declared, (0.5, 0.5), 1.0),
    }
    selected = values[kind]
    raw = selected.model_dump(mode="json") | {"schema_version": version}
    payload = json.dumps(raw)
    with pytest.raises(ValueError):
        type(selected).from_json(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_faces", 0),
        ("maximum_faces", 65537),
        ("maximum_integer_bits", 63),
        ("maximum_integer_bits", 32769),
        ("maximum_faces", True),
    ],
)
def test_actual_solver_work_envelope_is_strict_and_finite(field, value):
    raw = {"schema_version": 1, field: value}
    with pytest.raises(ValueError):
        SolveLimits.model_validate(raw)


@pytest.mark.parametrize(
    "change",
    [
        {"future": 1},
        {"gross_limit": -1.0},
        {"leverage_limit": float("inf")},
        {"gross_limit": True},
        {"profile": "financial-short-v1"},
    ],
)
def test_unknown_profiles_nonfinite_and_negative_resource_declarations_refuse(change):
    raw = geometry().model_dump() | change
    with pytest.raises(ValueError):
        ContinuousSet.model_validate(raw)


@pytest.mark.parametrize("matrix", [((1.0, 2.0), (0.0, 1.0)), ((1.0,),), ((float("nan"), 0.0), (0.0, 1.0))])
def test_malformed_quadratic_cannot_reach_the_solver(matrix):
    raw = problem().objective.model_dump() | {"matrix": matrix}
    with pytest.raises(ValueError):
        QuadraticObjective.model_validate(raw)


@pytest.mark.parametrize(
    "semantics", [UnitSemantics(unit="s", meaning="duration"), UnitSemantics(unit="USD", meaning="amount")]
)
def test_objective_does_not_silently_accept_dimensional_cost_as_coefficient_score(semantics):
    original = problem().objective
    spec = ScalarSpec(dimension=original.score.dimension, role="objective", semantics=semantics)
    raw = original.model_dump() | {"score": spec}
    with pytest.raises(ValueError):
        QuadraticObjective.model_validate(raw)


@pytest.mark.parametrize(
    "change", ["point", "objective", "primal", "dual", "equality", "gradient", "value", "proposal"]
)
def test_complete_result_reader_rejects_altered_claims_even_when_original_fingerprint_is_retained(change):
    outcome = solve_continuous(problem())
    raw = outcome.model_dump(mode="json")
    if change == "point":
        raw["executed"][0] = 0.5
    elif change == "objective":
        raw["problem"]["objective"]["linear"][0] = -3.0
    elif change == "primal":
        raw["certificate"]["residuals"]["primal"] = 1e-20
    elif change == "dual":
        raw["certificate"]["duals"][0] = -1.0
    elif change == "equality":
        raw["certificate"]["equality_dual"] = None
    elif change == "gradient":
        raw["certificate"]["gradient"][0] = 0.0
    elif change == "proposal":
        raw["proposal"][0] = 0.0
    else:
        raw["certificate"]["value"] = 0.0
    payload = json.dumps(raw)
    with pytest.raises(ValueError):
        ContinuousSolution.from_json(payload)


def test_zero_and_negative_coefficients_are_distinct_from_invalid_simplex_boxes():
    raw = geometry().model_dump()
    raw["variables"][0]["lower"] = -0.25
    with pytest.raises(ValueError):
        ContinuousSet.model_validate(raw)
    negative = geometry("signed-box-v1", ((-1.0, 1.0), (-1.0, 1.0)))
    assert negative.variables[0].lower == -1.0
    signed = CoefficientInterval(schema_version=1, lower=-0.0, upper=1.0)
    restored = CoefficientInterval.from_json(signed.to_json())
    assert copysign(1.0, restored.lower) == -1.0


@pytest.mark.parametrize("rate", [-1.0, float("nan"), float("inf"), True])
def test_projected_step_refuses_invalid_rate_before_producing_a_new_action(rate):
    declared = problem()
    before = declared.to_json()
    with pytest.raises(ValueError):
        projected_update(declared, (0.5, 0.5), rate)
    assert declared.to_json() == before


@pytest.mark.parametrize("point", [(0.5,), (-0.1, 1.1), (float("nan"), 0.5)])
def test_online_step_refuses_wrong_or_infeasible_original_allocation(point):
    declared = problem()
    with pytest.raises(ValueError):
        projected_update(declared, point, 1.0)


def test_rational_bit_exhaustion_is_an_explicit_atomic_numerical_refusal():
    declared = problem(matrix=((1e-100, 0.0), (0.0, 1.0)))
    restricted = ContinuousProblem.model_validate(
        declared.model_dump() | {"limits": SolveLimits(schema_version=1, maximum_integer_bits=64)}
    )
    before = restricted.to_json()
    result = solve_continuous(restricted)
    assert result.status == "numerical_failure"
    assert result.reason == "derived_rational_bit_limit"
    assert result.executed is None
    assert restricted.to_json() == before


def test_gradient_emission_failure_never_returns_a_partial_online_step():
    declared = problem(linear=(-1e308, -1e308))
    with pytest.raises(ArithmeticLimitError):
        projected_update(declared, (0.5, 0.5), 1e308)


def test_readers_keep_the_hard_depth_byte_unknown_field_and_duplicate_key_guards():
    payload = '{"schema_version":1,"deep":' + "[" * 66 + "0" + "]" * 66 + "}"
    with pytest.raises(ValueError):
        ContinuousProblem.from_json(payload)
    unknown = problem().model_dump(mode="json") | {"unknown": None}
    unknown_json = json.dumps(unknown)
    with pytest.raises(ValueError):
        ContinuousProblem.from_json(unknown_json)
    duplicate = '{"schema_version":1,"schema_version":1}'
    with pytest.raises(ValueError):
        ContinuousProblem.from_json(duplicate)
    oversized = bytearray(16 * 1024**2 + 1)
    with pytest.raises(ValueError, match="before copying"):
        ContinuousProblem.model_validate_json(oversized)


def test_different_losses_with_equal_local_gradients_remain_distinct_complete_contexts():
    """A matching step does not authenticate the original full loss function."""
    step = projected_update(problem(), (0.5, 0.5), 1.0)
    other = projected_update(step.projection.problem, (0.5, 0.5), 1.0)
    assert other.gradient == step.gradient
    assert other.projection.executed == step.projection.executed
    assert other.problem.fingerprint != step.problem.fingerprint
    assert other.fingerprint != step.fingerprint
    assert ProjectedStep.from_json(other.to_json()).to_json() == other.to_json()


def test_dimension_and_coordinate_duplicates_reject_before_projection_work():
    raw = geometry().model_dump()
    raw["variables"] = (raw["variables"][0],) * 2
    with pytest.raises(ValueError):
        ContinuousSet.model_validate(raw)
    declared = problem()
    request = {
        "schema_version": 1,
        "feasible_set": declared.feasible_set,
        "proposal": (0.0,),
        "comparator": declared.comparator,
    }
    with pytest.raises(ValueError):
        ProjectionRequest.model_validate(request)


def test_active_face_arithmetic_failure_retains_actual_attempted_work_without_a_partial_vector():
    """The original exact bound is too fine for the selected rational cap."""
    feasible = geometry("signed-box-v1", ((0.0, 1e-100), (0.0, 1.0)))
    declared = problem(feasible)
    limited = ContinuousProblem.model_validate(
        declared.model_dump() | {"limits": SolveLimits(schema_version=1, maximum_integer_bits=64)}
    )
    result = solve_continuous(limited)
    assert result.status == "numerical_failure"
    assert result.faces_examined > 0
    assert result.faces_examined <= limited.limits.maximum_faces
    assert result.executed is None


def test_dimensionless_coefficient_semantics_cannot_be_relabelled_as_currency():
    """Negative mathematical exposure does not turn into qualified short money."""
    raw = geometry().model_dump() | {"coefficient_semantics": UnitSemantics(unit="USD", meaning="money")}
    with pytest.raises(ValueError):
        ContinuousSet.model_validate(raw)


@pytest.mark.parametrize(
    "change", ["missing_output", "missing_certificate", "false_refusal", "short_duals", "wrong_dimension", "over_work"]
)
def test_a_closed_reader_cannot_confuse_absent_or_partial_output_with_an_accepted_allocation(change):
    """Public complete-result bytes retain atomic status and full coordinates."""
    raw = solve_continuous(problem()).model_dump(mode="json")
    if change == "missing_output":
        raw["executed"] = None
    elif change == "missing_certificate":
        raw["certificate"] = None
    elif change == "false_refusal":
        raw["status"] = "infeasible"
    elif change == "short_duals":
        raw["certificate"]["duals"] = []
    elif change == "wrong_dimension":
        raw["proposal"] = [0.75]
    else:
        raw["faces_examined"] = 4097
    payload = json.dumps(raw)
    with pytest.raises(ValueError):
        ContinuousSolution.from_json(payload)


def test_consistent_but_unqualified_kkt_diagnostics_cannot_claim_optimal_output():
    """Reporting the actual nonzero residual does not waive its frozen limit."""
    raw = solve_continuous(problem()).model_dump(mode="json")
    raw["executed"] = [0.5, 0.5]
    raw["certificate"]["value"] = -1.5
    raw["certificate"]["gradient"] = [-1.0, -0.5]
    raw["certificate"]["residuals"]["stationarity"] = 0.25
    payload = json.dumps(raw)
    with pytest.raises(ValueError, match="arithmetic envelope"):
        ContinuousSolution.from_json(payload)
