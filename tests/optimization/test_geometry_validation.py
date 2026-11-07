"""Qualify exact geometry, emitted rounding and authoritative path accounting."""

import json
import math
from fractions import Fraction
from typing import Literal

import pytest

from coleman.core.actions import Weight
from coleman.core.identity import Reference
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionLineage
from coleman.objectives.controls import (
    CoefficientOrigin,
    ExactCoefficientVariation,
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


def ref(name, version="1"):
    return Reference(id=name, version=version)


def metadata(name):
    return {
        "schema_version": 1,
        "reference": ref(name),
        "lineage": DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None),
    }


def weights(values):
    return tuple(Weight(member=ref(f"expert-{i:04}"), value=value) for i, value in enumerate(values))


def origin(values, name="emitted-0"):
    return CoefficientOrigin(
        **metadata(name), weights=weights(values), profile="binary64-simplex-v1", semantics=COEFFICIENT_SEMANTICS
    )


def bounds(count, lowers=None, uppers=None):
    lows = (0.0,) * count if lowers is None else lowers
    highs = (1.0,) * count if uppers is None else uppers
    return ExpertBounds(
        **metadata("bounds"),
        semantics=COEFFICIENT_SEMANTICS,
        bounds=tuple(
            ExpertBound(member=ref(f"expert-{i:04}"), lower=low, upper=high)
            for i, (low, high) in enumerate(zip(lows, highs, strict=True))
        ),
    )


def movement(
    values,
    scope: Literal["per_interval", "per_release"] = "per_interval",
    measure: Literal["accumulated_absolute", "net_displacement"] = "accumulated_absolute",
):
    return MovementAllowance(
        **metadata("movement"),
        semantics=COEFFICIENT_SEMANTICS,
        scope=scope,
        measure=measure,
        limits=tuple(MovementLimit(member=ref(f"expert-{i:04}"), maximum=value) for i, value in enumerate(values)),
    )


def penalty(current, strength):
    return L1MovementPenalty(**metadata("penalty"), semantics=COEFFICIENT_SEMANTICS, origin=current, strength=strength)


def test_legacy_residual_admission_and_exact_empty_simplex_remain_distinct():
    current = origin((1 / 3,) * 3)
    declared = bounds(3)
    hard = movement((0.0,) * 3, scope="per_release")
    old = check_mixture_intersection(declared, current, movement=hard)
    exact = exact_mixture_intervals(declared, current, movement=hard)
    result = solve_mixture_output(
        proposal=current.weights, bounds=declared, origin=current, output=ProjectionOutput(), movement=hard
    )
    assert old.status == "feasible" and old.exact_feasible is False
    assert exact.status == "infeasible" and exact.reason == "simplex_sum"
    assert exact.legacy_intersection is not None
    assert exact.legacy_intersection.to_json() == old.to_json()
    assert result.status == "infeasible" and result.weights is None and result.proposal == current.weights


def test_exact_intervals_retain_unrounded_original_movement_endpoints():
    current = origin((0.5, 0.5))
    hard = movement((2**-55,) * 2, scope="per_release")
    exact = exact_mixture_intervals(bounds(2), current, movement=hard)
    assert exact.intervals[0].lower == Fraction(1, 2) - Fraction(1, 2**55)
    assert exact.intervals[0].upper == Fraction(1, 2) + Fraction(1, 2**55)
    assert exact.legacy_intersection is not None
    assert Fraction(exact.legacy_intersection.effective[0].upper) != exact.intervals[0].upper


def test_projection_and_zero_penalty_share_the_complete_numerical_path():
    current = origin((0.5, 0.5))
    declared = bounds(2, uppers=(0.6, 1.0))
    proposal = weights((0.9, 0.1))
    projected = solve_mixture_output(proposal=proposal, bounds=declared, origin=current, output=ProjectionOutput())
    penalized = solve_mixture_output(proposal=proposal, bounds=declared, origin=current, output=penalty(current, 0.0))
    assert projected.status == penalized.status == "optimal"
    assert projected.weights == penalized.weights == weights((0.6, 0.4))
    assert projected.certificate == penalized.certificate
    assert declared.bounds[0].upper < 0.6 / (0.6 + 0.1)


def test_exact_dyadic_hold_plateau_has_one_output_and_a_deterministic_multiplier():
    current = origin((0.5, 0.25, 0.25))
    result = solve_mixture_output(
        proposal=weights((0.75, 0.125, 0.125)),
        bounds=bounds(3),
        origin=current,
        output=penalty(current, 0.25),
    )
    assert result.status == "optimal" and result.weights == current.weights
    assert result.certificate is not None
    assert result.certificate.exact_weights == (Fraction(1, 2), Fraction(1, 4), Fraction(1, 4))
    assert result.certificate.full_l1 == 0
    assert Fraction(0) <= result.certificate.dual <= Fraction(1, 8)


def test_positive_subnormal_optimizer_underflow_is_a_typed_failure():
    current = origin((0.5, 0.5))
    proposed = weights((math.ulp(0.0), 1.0))
    result = solve_mixture_output(proposal=proposed, bounds=bounds(2), origin=current, output=ProjectionOutput())
    assert result.status == "numerical_failure" and result.reason == "nonrepresentable_output"
    assert result.weights is None and result.certificate is None and result.proposal == proposed
    assert result.geometry.status == "feasible"


def test_exact_endpoint_feasibility_does_not_excuse_rounding_outside_the_box():
    current = origin((0.5 - 2**-54, 0.5))
    hard = movement((2**-55,) * 2, scope="per_release")
    result = solve_mixture_output(
        proposal=weights((0.5, 0.5)), bounds=bounds(2), origin=current, output=ProjectionOutput(), movement=hard
    )
    assert result.geometry.status == "feasible"
    assert result.status == "numerical_failure" and result.reason == "rounded_box"


def test_exact_diagnostics_do_not_overflow_or_pretend_to_be_finite_float_values():
    current = origin((1.0, 0.0))
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    result = solve_mixture_output(
        proposal=weights((0.0, 1.0)),
        bounds=bounds(2, lowers=(0.0, 1.0), uppers=(0.0, 1.0)),
        origin=current,
        output=penalty(current, maximum),
    )
    assert result.status == "optimal" and result.weights == weights((0.0, 1.0))
    assert result.certificate is not None
    assert result.certificate.objective == 2 * Fraction(maximum)
    assert result.certificate.objective > Fraction(maximum)


def test_penalty_origin_joins_include_version_and_original_signed_zero_bytes():
    current = origin((-0.0, 1.0))
    different = origin((0.0, 1.0))
    selected = penalty(different, 0.5)
    proposed = weights((-0.0, 1.0))
    declared = bounds(2)
    with pytest.raises(ValueError, match="exact current emitted origin"):
        solve_mixture_output(proposal=proposed, bounds=declared, origin=current, output=selected)
    a = exact_mixture_intervals(declared, current)
    b = exact_mixture_intervals(declared, different)
    assert a.intervals == b.intervals and a.inputs_fingerprint != b.inputs_fingerprint
    result = solve_mixture_output(proposal=proposed, bounds=declared, origin=current, output=ProjectionOutput())
    assert math.copysign(1.0, result.proposal[0].value) == -1.0


@pytest.mark.parametrize("bad", ["", "00", "01", "+1", "-0", " 1", "1 ", "1e2", "1.0", "١", "9" * 633, True, 1])
def test_exact_counts_reject_noncanonical_or_overlong_input_before_integer_parsing(bad):
    member = ref("expert-0000")
    with pytest.raises(ValueError, match="Exact movement atoms"):
        ExactCoefficientVariation(member=member, atoms_decimal=bad)


def test_exact_path_geometry_and_committed_actual_deltas_use_the_same_ledger():
    delta = 2**-55
    values = (
        (0.0, 0.5, 0.5),
        (0.5 - 2 * delta, 0.5, 2 * delta),
        (0.0, 1 - 8 * delta, 8 * delta),
        (delta, 1 - 8 * delta, 7 * delta),
    )
    origins = tuple(origin(row, f"emitted-{i}") for i, row in enumerate(values))
    hard = movement((1.0, 2.0, 2.0))
    usage = initial_exact_path_usage(interval=ref("period"), movement=hard, origin=origins[0])
    for i in range(1, len(origins)):
        usage = advance_exact_path_usage(usage=usage, origin=origins[i - 1], emitted=origins[i], release=ref(f"r{i}"))
    true_used = Fraction(int(usage.used[0].atoms_decimal), 2**1074)
    assert true_used == 1 - 3 * Fraction(delta)
    assert Fraction(float(true_used)) == 1 - 4 * Fraction(delta)
    exact = exact_mixture_intervals(bounds(3), origins[-1], movement=hard, exact_path=usage)
    assert exact.intervals[0].upper == 4 * Fraction(delta)
    legacy = MovementUsage(
        interval=usage.interval,
        origin=origins[0],
        used=tuple(
            MovementLimit(member=item.member, maximum=float(Fraction(int(item.atoms_decimal), 2**1074)))
            for item in usage.used
        ),
    )
    old = check_mixture_intersection(bounds(3), origins[-1], movement=hard, usage=legacy)
    assert old.effective[0].upper == 5 * delta
    bad = origin((5 * delta, 1 - 8 * delta, 3 * delta), "overspent")
    before = usage.to_json()
    release = ref("r4")
    with pytest.raises(ValueError, match="exceeds its declared allowance"):
        advance_exact_path_usage(usage=usage, origin=origins[-1], emitted=bad, release=release)
    assert usage.to_json() == before
    good = origin((4 * delta, 1 - 8 * delta, 4 * delta), "exhausted")
    exhausted = advance_exact_path_usage(usage=usage, origin=origins[-1], emitted=good, release=release)
    assert Fraction(int(exhausted.used[0].atoms_decimal), 2**1074) == 1
    assert ExactPathUsage.from_json(exhausted.to_json()).to_json() == exhausted.to_json()
    following = exact_mixture_intervals(bounds(3), good, movement=hard, exact_path=exhausted)
    assert following.intervals[0].lower == following.intervals[0].upper == Fraction(4 * delta)


def test_holds_charge_zero_but_advance_release_and_do_not_erase_signed_origin():
    current = origin((-0.0, 1.0))
    hard = movement((2.0, 2.0))
    initial = initial_exact_path_usage(interval=ref("period"), movement=hard, origin=current)
    first_release = ref("release")
    held = advance_exact_path_usage(usage=initial, origin=current, emitted=current, release=first_release)
    assert held.used == initial.used and held.through_origin == current.binding
    assert held.through_release == first_release and held.fingerprint != initial.fingerprint
    with pytest.raises(ValueError, match="same release twice"):
        advance_exact_path_usage(usage=held, origin=current, emitted=current, release=first_release)


def test_exact_path_can_exceed_one_and_reconstructs_subnormal_grid_deltas():
    initial_origin = origin((0.0, 1.0))
    other = origin((1.0, 0.0), "other")
    hard = movement((2.0, 2.0))
    initial = initial_exact_path_usage(interval=ref("period"), movement=hard, origin=initial_origin)
    first = advance_exact_path_usage(usage=initial, origin=initial_origin, emitted=other, release=ref("r1"))
    second = advance_exact_path_usage(usage=first, origin=other, emitted=initial_origin, release=ref("r2"))
    assert all(int(item.atoms_decimal) == 2 * 2**1074 for item in second.used)
    tiny = origin((math.ulp(0.0), 1.0), "tiny")
    one_atom = advance_exact_path_usage(usage=initial, origin=initial_origin, emitted=tiny, release=ref("tiny"))
    assert one_atom.used[0].atoms_decimal == "1"


def test_wrong_exact_path_units_profile_origin_and_legacy_combination_reject():
    current = origin((0.5, 0.5))
    hard = movement((1.0, 1.0))
    usage = initial_exact_path_usage(interval=ref("period"), movement=hard, origin=current)
    wrong_units = usage.model_dump(mode="python")
    wrong_units["semantics"] = UnitSemantics(unit="%", meaning="mixture-coefficient-fraction-v1")
    with pytest.raises(ValueError, match="canonical coefficient-fraction"):
        ExactPathUsage.model_validate(wrong_units)
    wrong_origin = origin((0.5, 0.5), "new-version")
    declared = bounds(2)
    with pytest.raises(ValueError, match="current origin"):
        exact_mixture_intervals(declared, wrong_origin, movement=hard, exact_path=usage)
    legacy = MovementUsage(
        interval=ref("period"),
        origin=current,
        used=tuple(MovementLimit(member=w.member, maximum=0.0) for w in current.weights),
    )
    with pytest.raises(ValueError, match="excludes legacy usage"):
        exact_mixture_intervals(declared, current, movement=hard, usage=legacy, exact_path=usage)


@pytest.mark.parametrize("reader", [ExactPathUsage.from_json, ExactPathUsage.model_validate_json])
def test_exact_path_both_json_readers_bound_bytes_depth_versions_and_duplicate_keys(reader):
    current = origin((0.5, 0.5))
    hard = movement((1.0, 1.0))
    usage = initial_exact_path_usage(interval=ref("period"), movement=hard, origin=current)
    assert reader(usage.to_json()).to_json() == usage.to_json()
    future = usage.model_dump(mode="json")
    future["schema_version"] = 2
    future_json = json.dumps(future)
    with pytest.raises(ValueError, match="schema_version"):
        reader(future_json)
    oversized = " " * (16 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="maximum_bytes"):
        reader(oversized)
    nested = "[" * 65 + "0" + "]" * 65
    with pytest.raises(ValueError, match="depth"):
        reader(nested)
    duplicate = usage.to_json().replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1)
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        reader(duplicate)


def test_full_4096_coordinate_domain_and_extreme_finite_strength_are_admitted():
    count = 4096
    current = origin((1 / count,) * count)
    result = solve_mixture_output(
        proposal=current.weights,
        bounds=bounds(count),
        origin=current,
        output=penalty(current, float.fromhex("0x1.fffffffffffffp+1023")),
    )
    assert result.status == "optimal" and result.weights == current.weights
    assert result.certificate is not None
    assert result.certificate.simplex_residual == result.certificate.full_l1 == 0


def test_full_4096_distinct_breakpoint_domain_matches_symmetric_scalar_arithmetic():
    count = 4096
    current = origin((1 / count,) * count)
    strength = 2**-40
    proposed = weights(tuple(1 / count + (2 * i - (count - 1)) * 2**-26 for i in range(count)))
    expected = tuple(weight.value + (strength if i < count // 2 else -strength) for i, weight in enumerate(proposed))
    result = solve_mixture_output(
        proposal=proposed, bounds=bounds(count), origin=current, output=penalty(current, strength)
    )
    assert result.status == "optimal" and result.weights == weights(expected)
    assert result.certificate is not None and result.certificate.dual == 0
    assert result.certificate.maximum_rounding_error == result.certificate.simplex_residual == 0
