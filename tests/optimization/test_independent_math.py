"""Compare constrained outputs with authored exact mathematics, not self-oracles."""

import json
import math
from fractions import Fraction
from itertools import permutations
from pathlib import Path
from typing import Literal

import pytest

from coleman.core.actions import Weight
from coleman.core.identity import Reference
from coleman.objectives.bindings import DefinitionLineage
from coleman.objectives.controls import (
    CoefficientOrigin,
    ExactPathUsage,
    ExpertBound,
    ExpertBounds,
    L1MovementPenalty,
    MovementAllowance,
    MovementLimit,
    MovementUsage,
    ProjectionOutput,
    advance_exact_path_usage,
    check_mixture_intersection,
    exact_mixture_intervals,
    initial_exact_path_usage,
)
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS
from coleman.optimization.mixture import solve_mixture_output
from tests.optimization._independent_oracles import (
    effective_intervals,
    exchange_certificate,
    face_minimum,
    full_movement,
    objective,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "optimization" / "v1"
ZERO = Fraction()
ONE = Fraction(1)
THIRDS = (Fraction(1, 3),) * 3
PROPOSAL = (Fraction(1, 2), Fraction(3, 10), Fraction(1, 5))


def _fractions(values):
    return tuple(Fraction(value) for value in values)


@pytest.mark.parametrize("case_index", [0, 1])
def test_authored_rational_goldens_and_feasible_direction_certificate(case_index):
    document = json.loads((FIXTURES / "hand-minima.json").read_text())
    assert document["schema_version"] == 1
    assert document["arithmetic"] == "symbolic-rational-fixtures-v1"
    case = document["cases"][case_index]
    proposal, old = _fractions(case["proposal"]), _fractions(case["old"])
    lower, upper = _fractions(case["lower"]), _fractions(case["upper"])
    strength, expected = Fraction(case["strength"]), _fractions(case["minimum"])
    found = face_minimum(proposal, old, lower, upper, strength)
    assert found is not None
    assert found.weights == expected
    assert found.dual_lower == found.dual_upper == Fraction(case["dual"])
    assert found.objective == Fraction(case["objective"])
    assert full_movement(expected, old) == Fraction(case["full_L1"])
    assert exchange_certificate(expected, proposal, old, lower, upper, strength)
    moved = (expected[0] - Fraction(1, 300), expected[1] + Fraction(1, 300), expected[2])
    assert not exchange_certificate(moved, proposal, old, lower, upper, strength)


def test_barycentric_grid_corroborates_only_the_grid_minimum():
    strength = Fraction(1, 20)
    grid = ((Fraction(i, 30), Fraction(j, 30), Fraction(30 - i - j, 30)) for i in range(31) for j in range(31 - i))
    ranked = [(objective(p, PROPOSAL, THIRDS, strength), p) for p in grid]
    best = min(value for value, _ in ranked)
    assert best == Fraction(1, 75)
    assert [p for value, p in ranked if value == best] == [(Fraction(13, 30), Fraction(1, 3), Fraction(7, 30))]


def test_sequential_unconstrained_prox_then_projection_is_not_joint_minimum():
    # First apply the unconstrained shifted L1 prox, then its ordinary simplex
    # projection. This independent wrong implementation has the wrong optimum.
    prox = (Fraction(9, 20), Fraction(1, 3), Fraction(1, 4))
    shift = (sum(prox, ZERO) - 1) / 3
    sequential = tuple(value - shift for value in prox)
    assert sequential == (Fraction(79, 180), Fraction(29, 90), Fraction(43, 180))
    assert objective(sequential, PROPOSAL, THIRDS, Fraction(1, 20)) == Fraction(29, 2160)
    assert Fraction(29, 2160) - Fraction(1, 75) == Fraction(1, 10800)
    assert not exchange_certificate(sequential, PROPOSAL, THIRDS, (ZERO,) * 3, (ONE,) * 3, Fraction(1, 20))


def test_clip_and_renormalize_violates_the_actual_static_cap():
    clipped = (Fraction(3, 5), Fraction(1, 10))
    repaired = tuple(value / sum(clipped) for value in clipped)
    assert repaired == (Fraction(6, 7), Fraction(1, 7))
    assert repaired[0] > Fraction(3, 5)
    correct = face_minimum(
        (Fraction(9, 10), Fraction(1, 10)), (Fraction(1, 2),) * 2, (ZERO,) * 2, (Fraction(3, 5), ONE), ZERO
    )
    assert correct is not None
    assert correct.weights == (Fraction(3, 5), Fraction(2, 5))


def test_hold_threshold_unique_primal_and_nonunique_multiplier():
    for strength in (Fraction(3, 20), Fraction(1, 5), Fraction(100)):
        found = face_minimum(PROPOSAL, THIRDS, (ZERO,) * 3, (ONE,) * 3, strength)
        assert found is not None
        assert found.weights == THIRDS
    plateau = face_minimum(PROPOSAL, THIRDS, (ZERO,) * 3, (ONE,) * 3, Fraction(1, 5))
    assert plateau is not None
    assert (plateau.dual_lower, plateau.dual_upper) == (Fraction(-1, 30), Fraction(1, 15))


@pytest.mark.parametrize(
    ("old", "lower", "upper"),
    [
        ((ONE,), (ZERO,), (ONE,)),
        ((Fraction(1, 2),) * 2, (Fraction(1, 2),) * 2, (Fraction(1, 2),) * 2),
        ((ONE, ZERO, ZERO), (ZERO, Fraction(1, 2), Fraction(1, 2)), (ZERO, Fraction(1, 2), Fraction(1, 2))),
    ],
)
def test_fixed_coordinates_singletons_and_infeasible_old_do_not_justify_holding(old, lower, upper):
    proposal = tuple(reversed(old))
    found = face_minimum(proposal, old, lower, upper, Fraction(100))
    assert found is not None
    expected = lower if sum(lower, ZERO) == 1 else old
    assert found.weights == expected
    assert exchange_certificate(found.weights, proposal, old, lower, upper, Fraction(100))


def test_oracle_caps_and_empty_sets_are_not_fabricated_optima():
    assert face_minimum(THIRDS, THIRDS, (Fraction(1, 2),) * 3, (ONE,) * 3, ZERO) is None
    assert face_minimum(THIRDS, THIRDS, (ZERO,) * 3, (Fraction(1, 4),) * 3, ZERO) is None
    five = (Fraction(1, 5),) * 5
    five_lower, five_upper = (ZERO,) * 5, (ONE,) * 5
    with pytest.raises(ValueError, match="one through four"):
        face_minimum(five, five, five_lower, five_upper, ZERO)
    with pytest.raises(ValueError, match="nonnegative"):
        face_minimum(THIRDS, THIRDS, (ZERO,) * 3, (ONE,) * 3, -ONE)


def test_symbolic_thirds_are_not_exact_stored_binary64_thirds():
    stored = Fraction.from_float(1 / 3)
    assert 3 * stored == 1 - Fraction(1, 2**54)
    lower, upper = effective_intervals((ZERO,) * 3, (ONE,) * 3, (stored,) * 3, allowances=(ZERO,) * 3)
    assert lower == upper == (stored,) * 3
    assert face_minimum(PROPOSAL, (stored,) * 3, lower, upper, ZERO) is None
    symbolic = face_minimum(PROPOSAL, THIRDS, THIRDS, THIRDS, ZERO)
    assert symbolic is not None
    assert symbolic.weights == THIRDS


def test_raw_period_net_and_accumulated_geometry_use_different_centers():
    old, period = (Fraction(3, 4), Fraction(1, 4)), (Fraction(1, 2),) * 2
    allowance = (Fraction(1, 2),) * 2
    net = effective_intervals((ZERO,) * 2, (ONE,) * 2, old, allowances=allowance, period_origin=period)
    accumulated = effective_intervals(
        (ZERO,) * 2, (ONE,) * 2, old, allowances=allowance, period_origin=period, used=(Fraction(3, 8),) * 2
    )
    assert net == ((ZERO, ZERO), (ONE, ONE))
    assert accumulated == ((Fraction(5, 8), Fraction(1, 8)), (Fraction(7, 8), Fraction(3, 8)))


def test_fixed_geometry_full_l1_is_nonincreasing_in_strength_and_permutation_equivariant():
    lower = (Fraction(1, 10), ZERO, ZERO)
    upper = (Fraction(2, 5), ONE, ONE)
    strengths = (ZERO, Fraction(1, 100), Fraction(1, 20), Fraction(1, 10), ONE)
    movements = []
    for strength in strengths:
        found = face_minimum(PROPOSAL, THIRDS, lower, upper, strength)
        assert found is not None
        movements.append(full_movement(found.weights, THIRDS))
        for order in permutations(range(3)):
            permuted = face_minimum(
                tuple(PROPOSAL[i] for i in order),
                tuple(THIRDS[i] for i in order),
                tuple(lower[i] for i in order),
                tuple(upper[i] for i in order),
                strength,
            )
            assert permuted is not None
            assert permuted.weights == tuple(found.weights[i] for i in order)
    assert movements == sorted(movements, reverse=True)


def _path_fixture():
    document = json.loads((FIXTURES / "exact-path.json").read_text())
    assert document["schema_version"] == 1
    delta = Fraction(document["delta"])
    rows = tuple(tuple(Fraction(atom) * delta for atom in row) for row in document["emissions_in_delta_units"])
    return document, delta, rows


def test_authored_five_vector_accounting_exposes_nearest_rounding_overspend():
    document, delta, rows = _path_fixture()
    assert all(sum(row) == 1 for row in rows)
    assert all(Fraction.from_float(float(value)) == value for row in rows for value in row)
    exact = [ZERO, ZERO, ZERO]
    rounded = [0.0, 0.0, 0.0]
    for index, (before, after) in enumerate(zip(rows, rows[1:], strict=False), 1):
        for i, (a, b) in enumerate(zip(before, after, strict=True)):
            exact[i] += abs(b - a)
            rounded[i] += float(abs(b - a))
        if index == 3:
            assert exact[0] == Fraction(document["true_first_usage_after_release_3"]) == 1 - 3 * delta
            assert (
                Fraction.from_float(rounded[0])
                == Fraction(document["nearest_first_usage_after_release_3"])
                == 1 - 4 * delta
            )
            assert 1 - exact[0] == 3 * delta
        if index == 4:
            assert exact[0] == Fraction(document["true_first_usage_after_release_4"]) == 1 + delta
            assert rounded[0] == float(document["nearest_first_usage_after_release_4"]) == 1.0
    assert exact[1] <= 2 and exact[2] <= 2


def test_exact_variation_partition_reversal_stutter_and_subnormal_quantum():
    quantum = Fraction.from_float(math.ulp(0.0))
    assert quantum == Fraction(1, 2**1074)
    path = (ZERO, quantum, 2 * quantum, 3 * quantum)
    spent = sum((abs(b - a) for a, b in zip(path, path[1:], strict=False)), ZERO)
    assert spent == abs(path[-1] - path[0]) == 3 * quantum
    stutter = (*path[:2], path[1], *path[2:])
    assert sum((abs(b - a) for a, b in zip(stutter, stutter[1:], strict=False)), ZERO) == spent
    reversal = (*path, 2 * quantum, 3 * quantum)
    assert sum((abs(b - a) for a, b in zip(reversal, reversal[1:], strict=False)), ZERO) == spent + 2 * quantum


def _ref(name):
    return Reference(id=name, version="v1")


def _metadata(name):
    return {
        "schema_version": 1,
        "reference": _ref(name),
        "lineage": DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None),
    }


def _weights(values):
    return tuple(Weight(member=_ref(f"expert-{i}"), value=float(value)) for i, value in enumerate(values))


def _origin(values, name="old"):
    return CoefficientOrigin(
        **_metadata(name), weights=_weights(values), profile="binary64-simplex-v1", semantics=COEFFICIENT_SEMANTICS
    )


def _bounds(lower, upper):
    return ExpertBounds(
        **_metadata("bounds"),
        bounds=tuple(
            ExpertBound(member=_ref(f"expert-{i}"), lower=float(lo), upper=float(hi))
            for i, (lo, hi) in enumerate(zip(lower, upper, strict=True))
        ),
        semantics=COEFFICIENT_SEMANTICS,
    )


def _penalty(old, strength):
    return L1MovementPenalty(
        **_metadata("penalty"), origin=old, strength=float(strength), semantics=COEFFICIENT_SEMANTICS
    )


def _movement(
    allowances,
    scope: Literal["per_release", "per_interval"] = "per_release",
    measure: Literal["net_displacement", "accumulated_absolute"] = "net_displacement",
):
    return MovementAllowance(
        **_metadata("movement"),
        limits=tuple(MovementLimit(member=_ref(f"expert-{i}"), maximum=float(a)) for i, a in enumerate(allowances)),
        scope=scope,
        measure=measure,
        semantics=COEFFICIENT_SEMANTICS,
    )


def _stored(values):
    return tuple(Fraction.from_float(float(value)) for value in values)


def _require_oracle_agreement(result, proposal, old, lower, upper, strength):
    expected = face_minimum(proposal, old, lower, upper, strength)
    if expected is None:
        assert result.status == "infeasible"
        assert result.weights is result.certificate is None
        return
    assert result.status == "optimal", result.reason
    certificate = result.certificate
    assert certificate is not None and result.weights is not None
    assert certificate.exact_weights == expected.weights
    assert certificate.exact_objective == expected.objective
    assert expected.dual_lower is None or certificate.dual >= expected.dual_lower
    assert expected.dual_upper is None or certificate.dual <= expected.dual_upper
    assert exchange_certificate(certificate.exact_weights, proposal, old, lower, upper, strength)
    emitted = tuple(Fraction.from_float(weight.value) for weight in result.weights)
    assert all(lo <= value <= hi for value, lo, hi in zip(emitted, lower, upper, strict=True))
    assert certificate.objective == objective(emitted, proposal, old, strength)
    assert certificate.full_l1 == full_movement(emitted, old)
    assert certificate.objective_error == abs(certificate.objective - expected.objective)
    assert certificate.objective_error <= certificate.objective_error_bound


@pytest.mark.parametrize("cap", [1.0, 0.4])
@pytest.mark.parametrize("strength", [0.0, math.ulp(0.0), 0.05, 0.15, 0.2, 1e308])
def test_real_solver_matches_independent_faces_for_binary64_analogs_of_hand_goldens(cap, strength):
    proposal, old = (0.5, 0.3, 0.2), (1 / 3,) * 3
    lower, upper = (0.0,) * 3, (cap, 1.0, 1.0)
    current, bounds = _origin(old), _bounds(lower, upper)
    result = solve_mixture_output(
        proposal=_weights(proposal), bounds=bounds, origin=current, output=_penalty(current, strength)
    )
    _require_oracle_agreement(
        result, _stored(proposal), _stored(old), _stored(lower), _stored(upper), Fraction(strength)
    )
    if strength == 0.05:
        expected = (13 / 30, 1 / 3, 7 / 30) if cap == 1 else (0.4, 1 / 3, 4 / 15)
        assert result.weights is not None
        assert tuple(weight.value for weight in result.weights) == pytest.approx(expected, abs=2e-16)


@pytest.mark.parametrize("measure", ["per_release", "net_displacement", "accumulated_absolute"])
@pytest.mark.parametrize("strength", [0.0, 0.125, 1.0])
def test_real_solver_movement_intersections_are_derived_from_raw_declarations(measure, strength):
    proposal, old = (0.75, 0.125, 0.125), (0.5, 0.25, 0.25)
    lower, upper, allowances = (0.0,) * 3, (1.0,) * 3, (0.125, 0.25, 0.25)
    period = (0.375, 0.375, 0.25)
    current, bounds = _origin(old), _bounds(lower, upper)
    if measure == "per_release":
        movement, usage = _movement(allowances), None
        expected = effective_intervals(_stored(lower), _stored(upper), _stored(old), allowances=_stored(allowances))
    else:
        movement = _movement(allowances, "per_interval", measure)
        spent = (0.125, 0.125, 0.0)
        usage = MovementUsage(
            interval=_ref("interval"),
            origin=_origin(period, "period"),
            used=tuple(MovementLimit(member=_ref(f"expert-{i}"), maximum=u) for i, u in enumerate(spent)),
        )
        expected = effective_intervals(
            _stored(lower),
            _stored(upper),
            _stored(old),
            allowances=_stored(allowances),
            period_origin=_stored(period),
            used=_stored(spent) if measure == "accumulated_absolute" else None,
        )
    result = solve_mixture_output(
        proposal=_weights(proposal),
        bounds=bounds,
        origin=current,
        output=_penalty(current, strength),
        movement=movement,
        usage=usage,
    )
    assert tuple((box.lower, box.upper) for box in result.geometry.intervals) == tuple(zip(*expected, strict=True))
    _require_oracle_agreement(result, _stored(proposal), _stored(old), *expected, Fraction(strength))


@pytest.mark.parametrize(
    ("proposal", "old", "lower", "upper"),
    [
        ((1.0,), (1.0,), (0.0,), (1.0,)),
        ((0.75, 0.25), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
        ((0.0, 0.5, 0.5), (1.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.0, 0.5, 0.5)),
        ((0.5, 0.25, 0.125, 0.125), (0.25,) * 4, (0.0, 0.0, 0.25, 0.0), (0.375, 1.0, 0.25, 1.0)),
        ((0.5, 0.5), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0)),
    ],
)
def test_real_solver_fixed_faces_structural_zeros_ties_and_empty_set(proposal, old, lower, upper):
    current, bounds = _origin(old), _bounds(lower, upper)
    result = solve_mixture_output(
        proposal=_weights(proposal), bounds=bounds, origin=current, output=_penalty(current, 0.125)
    )
    _require_oracle_agreement(result, _stored(proposal), _stored(old), _stored(lower), _stored(upper), Fraction(1, 8))


def test_lambda_zero_uses_identical_real_geometry_rounding_and_failure_dispositions():
    cases = [
        ((0.5, 0.3, 0.2), (1 / 3,) * 3, None),
        ((math.ulp(0.0), 1.0), (0.5, 0.5), None),
        ((0.5, 0.3, 0.2), (1 / 3,) * 3, (0.0,) * 3),
    ]
    statuses = []
    for proposal, old, allowances in cases:
        current = _origin(old)
        bounds = _bounds((0.0,) * len(old), (1.0,) * len(old))
        movement = None if allowances is None else _movement(allowances)
        plain = solve_mixture_output(
            proposal=_weights(proposal), bounds=bounds, origin=current, output=ProjectionOutput(), movement=movement
        )
        penalized = solve_mixture_output(
            proposal=_weights(proposal), bounds=bounds, origin=current, output=_penalty(current, 0.0), movement=movement
        )
        assert (plain.status, plain.reason, plain.weights, plain.geometry, plain.certificate) == (
            penalized.status,
            penalized.reason,
            penalized.weights,
            penalized.geometry,
            penalized.certificate,
        )
        statuses.append((plain.status, plain.reason))
    assert statuses == [
        ("optimal", "optimal"),
        ("numerical_failure", "nonrepresentable_output"),
        ("infeasible", "simplex_sum"),
    ]


def test_exact_empty_thirds_preserves_legacy_numerical_admission_and_original_bytes():
    current = _origin((1 / 3,) * 3)
    bounds, movement = _bounds((0.0,) * 3, (1.0,) * 3), _movement((0.0,) * 3)
    before = (current.to_json(), bounds.to_json(), movement.to_json())
    legacy = check_mixture_intersection(bounds, current, movement=movement)
    exact = exact_mixture_intervals(bounds, current, movement=movement)
    assert legacy.status == "feasible" and legacy.exact_feasible is False
    assert exact.status == "infeasible" and exact.reason == "simplex_sum"
    assert exact.legacy_intersection == legacy
    assert before == (current.to_json(), bounds.to_json(), movement.to_json())


def test_exact_dyadic_hold_boundary_neighbors_and_real_permutations():
    proposal, old = (0.75, 0.125, 0.125), (0.5, 0.25, 0.25)
    threshold = 0.1875
    for strength in (math.nextafter(threshold, 0.0), threshold, math.nextafter(threshold, math.inf)):
        for order in permutations(range(3)):
            p, before = tuple(proposal[i] for i in order), tuple(old[i] for i in order)
            current, bounds = _origin(before), _bounds((0.0,) * 3, (1.0,) * 3)
            result = solve_mixture_output(
                proposal=_weights(p), bounds=bounds, origin=current, output=_penalty(current, strength)
            )
            _require_oracle_agreement(result, _stored(p), _stored(before), (ZERO,) * 3, (ONE,) * 3, Fraction(strength))
            assert result.certificate is not None
            if strength >= threshold:
                assert result.certificate.exact_weights == _stored(before)
            else:
                assert result.certificate.exact_weights != _stored(before)


def test_real_exact_path_uses_true_five_vector_usage_and_preserves_legacy_reader():
    _, delta, rows = _path_fixture()
    origins = tuple(_origin(row, f"emission-{i}") for i, row in enumerate(rows))
    movement = _movement((1.0, 2.0, 2.0), "per_interval", "accumulated_absolute")
    ledger = initial_exact_path_usage(interval=_ref("period"), movement=movement, origin=origins[0])
    spent = [ZERO] * 3
    for i in range(1, 4):
        ledger = advance_exact_path_usage(
            usage=ledger, origin=origins[i - 1], emitted=origins[i], release=_ref(f"release-{i}")
        )
        spent = [value + abs(b - a) for value, a, b in zip(spent, rows[i - 1], rows[i], strict=True)]
        assert tuple(Fraction(int(item.atoms_decimal), 2**1074) for item in ledger.used) == tuple(spent)
        assert ExactPathUsage.from_json(ledger.to_json()) == ledger
    before = ledger.to_json()
    bounds = _bounds((0.0,) * 3, (1.0,) * 3)
    exact = exact_mixture_intervals(bounds, origins[3], movement=movement, exact_path=ledger)
    expected = effective_intervals(
        (ZERO,) * 3,
        (ONE,) * 3,
        rows[3],
        allowances=(ONE, Fraction(2), Fraction(2)),
        period_origin=rows[0],
        used=tuple(spent),
    )
    assert tuple((box.lower, box.upper) for box in exact.intervals) == tuple(zip(*expected, strict=True))
    assert exact.intervals[0].upper == 4 * delta
    rounded_usage = MovementUsage(
        interval=_ref("period"),
        origin=origins[0],
        used=tuple(MovementLimit(member=_ref(f"expert-{i}"), maximum=float(value)) for i, value in enumerate(spent)),
    )
    legacy = check_mixture_intersection(bounds, origins[3], movement=movement, usage=rounded_usage)
    assert Fraction(legacy.effective[0].upper) == 5 * delta
    release_four = _ref("release-4")
    with pytest.raises(ValueError, match="exceeds"):
        advance_exact_path_usage(usage=ledger, origin=origins[3], emitted=origins[4], release=release_four)
    assert ledger.to_json() == before
    valid = _origin((4 * delta, rows[3][1], 4 * delta), "exact-spend")
    exhausted = advance_exact_path_usage(usage=ledger, origin=origins[3], emitted=valid, release=release_four)
    assert Fraction(int(exhausted.used[0].atoms_decimal), 2**1074) == 1
    assert ledger.through_release == _ref("release-3")


def test_real_fixed_geometry_total_exact_movement_is_nonincreasing_in_strength():
    proposal, old = (0.5, 0.3, 0.2), (1 / 3,) * 3
    current, bounds = _origin(old), _bounds((0.0,) * 3, (0.4, 1.0, 1.0))
    movements = []
    for strength in (0.0, 0.01, 0.05, 0.1, 1.0, 1e308):
        result = solve_mixture_output(
            proposal=_weights(proposal), bounds=bounds, origin=current, output=_penalty(current, strength)
        )
        assert result.status == "optimal" and result.certificate is not None
        movements.append(full_movement(result.certificate.exact_weights, _stored(old)))
    assert movements == sorted(movements, reverse=True)


@pytest.mark.parametrize("strength", [-1.0, math.nan, math.inf, -math.inf])
def test_real_solver_revalidates_unchecked_invalid_strength_before_any_output(strength):
    current, bounds = _origin((0.5, 0.5)), _bounds((0.0, 0.0), (1.0, 1.0))
    valid = _penalty(current, 0.125)
    unchecked = valid.model_copy(update={"strength": strength})
    proposal = _weights((0.75, 0.25))
    before = current.to_json()
    with pytest.raises(ValueError, match="finite|greater than or equal"):
        solve_mixture_output(proposal=proposal, bounds=bounds, origin=current, output=unchecked)
    assert current.to_json() == before


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_real_solver_revalidates_unchecked_nonfinite_proposal(invalid):
    current, bounds = _origin((0.5, 0.5)), _bounds((0.0, 0.0), (1.0, 1.0))
    bad = Weight.model_construct(member=_ref("expert-0"), value=invalid)
    proposal = (bad, Weight(member=_ref("expert-1"), value=0.5))
    output = ProjectionOutput()
    with pytest.raises(ValueError, match="finite"):
        solve_mixture_output(proposal=proposal, bounds=bounds, origin=current, output=output)


def test_authored_exact_path_v1_wire_round_trip_and_deterministic_content_identity():
    payload = (FIXTURES / "exact-path-initial-wire.json").read_text().rstrip("\n")
    ledger = ExactPathUsage.from_json(payload)
    assert ledger.to_json() == payload
    assert ExactPathUsage.from_json(ledger.to_json()) == ledger
    assert ledger.fingerprint == "sha256:d390763a8cdf8ec10dc41e02163d987c44a001090eab0ff325c535da29afdb71"
    constructed = initial_exact_path_usage(
        interval=ledger.interval, movement=ledger.movement, origin=ledger.period_origin
    )
    assert constructed.to_json() == payload


@pytest.mark.parametrize("version", [None, 0, 2, "1", True])
def test_exact_path_wire_rejects_missing_old_unknown_and_coerced_versions(version):
    document = json.loads((FIXTURES / "exact-path-initial-wire.json").read_text())
    if version is None:
        document.pop("schema_version")
    else:
        document["schema_version"] = version
    payload = json.dumps(document)
    with pytest.raises(ValueError, match="schema_version"):
        ExactPathUsage.from_json(payload)


@pytest.mark.parametrize("atoms", ["00", "+0", "-0", " 0", "0.0", "1e0", "١", "1" * 633])
def test_exact_path_wire_rejects_noncanonical_or_overlong_integer_atoms(atoms):
    document = json.loads((FIXTURES / "exact-path-initial-wire.json").read_text())
    document["used"][0]["atoms_decimal"] = atoms
    payload = json.dumps(document)
    with pytest.raises(ValueError, match="canonical|digits|632|leading zeroes"):
        ExactPathUsage.from_json(payload)


@pytest.mark.parametrize("alteration", ["unknown_profile", "physical_units", "unknown_member", "future_nested_version"])
def test_exact_path_wire_rejects_profile_units_members_and_nested_version(alteration):
    document = json.loads((FIXTURES / "exact-path-initial-wire.json").read_text())
    if alteration == "unknown_profile":
        document["profile"] = "unqualified-nearest-rounded-accounting-v99"
    elif alteration == "physical_units":
        document["semantics"]["unit"] = "s"
    elif alteration == "unknown_member":
        document["used"][0]["member"]["id"] = "different-expert"
    else:
        document["movement"]["schema_version"] = 2
    payload = json.dumps(document)
    with pytest.raises(ValueError, match="profile|fraction|expert|schema_version"):
        ExactPathUsage.from_json(payload)
