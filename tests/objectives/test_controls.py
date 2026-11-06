"""Exercise explicit coefficient controls independently from a learner/runtime."""

import json
import math
from itertools import permutations

import pytest

from coleman.core.actions import Weight
from coleman.core.identity import Reference
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionLineage
from coleman.objectives.controls import (
    CoefficientOrigin,
    ExpertBound,
    ExpertBounds,
    IntervalCadence,
    L1MovementPenalty,
    MixtureControls,
    MixtureIntersection,
    MovementAllowance,
    MovementLimit,
    MovementUsage,
    ProjectionOutput,
    ReleaseInterval,
    RoundCadence,
    check_mixture_intersection,
)
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS, DeclaredTargetShare, InitialPrior


def _ref(name, version="v1"):
    return Reference(id=name, version=version)


def _metadata(name="control"):
    return {
        "schema_version": 1,
        "reference": _ref(name),
        "lineage": DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None),
    }


def _weights(values=(0.5, 0.5)):
    return tuple(Weight(member=_ref(f"expert-{index}"), value=value) for index, value in enumerate(values))


def _origin(values=(0.5, 0.5), name="emitted-1"):
    return CoefficientOrigin(
        **_metadata(name), weights=_weights(values), profile="binary64-simplex-v1", semantics=COEFFICIENT_SEMANTICS
    )


def _bounds(lowers=(0.0, 0.0), uppers=(1.0, 1.0)):
    return ExpertBounds(
        **_metadata("bounds"),
        bounds=tuple(
            ExpertBound(member=_ref(f"expert-{i}"), lower=lo, upper=hi)
            for i, (lo, hi) in enumerate(zip(lowers, uppers, strict=True))
        ),
        semantics=COEFFICIENT_SEMANTICS,
    )


def _limits(values=(0.25, 0.25)):
    return tuple(MovementLimit(member=_ref(f"expert-{index}"), maximum=value) for index, value in enumerate(values))


def _movement(scope="per_release", measure="net_displacement", values=(0.25, 0.25)):
    return MovementAllowance(
        **_metadata("movement"), limits=_limits(values), scope=scope, measure=measure, semantics=COEFFICIENT_SEMANTICS
    )


def _usage(values=(0.0, 0.0), origin=None):
    return MovementUsage(
        interval=_ref("period-1"), origin=_origin() if origin is None else origin, used=_limits(values)
    )


def _penalty(strength=0.1, origin=None):
    return L1MovementPenalty(
        **_metadata("l1"),
        strength=strength,
        origin=_origin() if origin is None else origin,
        semantics=COEFFICIENT_SEMANTICS,
    )


def _controls(output=None):
    prior = InitialPrior(
        **_metadata("prior"), weights=_weights(), profile="binary64-simplex-v1", semantics=COEFFICIENT_SEMANTICS
    )
    return MixtureControls(
        **_metadata("controls"),
        initial_prior=prior,
        target=None,
        bounds=_bounds(),
        movement=None,
        cadence=RoundCadence(panels=1),
        output=ProjectionOutput() if output is None else output,
    )


def test_well_formed_empty_simplex_is_typed_infeasible_not_schema_error():
    for bounds in (_bounds((0.75, 0.75)), _bounds(uppers=(0.25, 0.25))):
        result = check_mixture_intersection(bounds, _origin())
        assert result.status == "infeasible" and result.reason == "simplex_sum"
        assert MixtureIntersection.from_json(result.to_json()) == result
    with pytest.raises(ValueError, match="lower bound"):
        ExpertBound(member=_ref("a"), lower=0.75, upper=0.5)


def test_static_and_movement_intersection_is_pure_and_does_not_project():
    bounds = _bounds((0.0, 0.0), (0.6, 1.0))
    origin = _origin()
    before = origin.to_json()
    result = check_mixture_intersection(bounds, origin, movement=_movement())
    assert result.status == "feasible"
    assert tuple((item.lower, item.upper) for item in result.effective) == ((0.25, 0.6), (0.25, 0.75))
    assert origin.to_json() == before
    assert "weights" not in result.model_dump()
    assert check_mixture_intersection(bounds, origin, movement=_movement()) == result


def test_coordinate_conflict_differs_from_bad_individual_box():
    result = check_mixture_intersection(_bounds((0.8, 0.0)), _origin(), movement=_movement(values=(0.1, 0.1)))
    assert result.status == "infeasible" and result.reason == "empty_coordinate"
    assert result.effective[0].lower > result.effective[0].upper


def test_interval_net_displacement_uses_period_origin_not_last_emitted_vector():
    current = _origin((0.75, 0.25), "emitted-2")
    usage = _usage((0.25, 0.25))
    movement = _movement("per_interval", values=(0.25, 0.25))
    result = check_mixture_intersection(_bounds(), current, movement=movement, usage=usage)
    assert tuple((item.lower, item.upper) for item in result.effective) == ((0.25, 0.75), (0.25, 0.75))
    assert result.usage is not None
    assert result.usage.interval == _ref("period-1")
    assert result.origin == current.binding and result.usage.origin.binding == _origin().binding
    with pytest.raises(ValueError, match="usage disagrees"):
        check_mixture_intersection(_bounds(), current, movement=movement, usage=_usage())


def test_interval_accumulated_usage_cannot_reset_or_use_net_displacement_semantics():
    current = _origin((0.75, 0.25), "emitted-2")
    movement = _movement("per_interval", "accumulated_absolute", (0.5, 0.5))
    usage = _usage((0.375, 0.375))
    result = check_mixture_intersection(_bounds(), current, movement=movement, usage=usage)
    assert tuple((item.lower, item.upper) for item in result.effective) == ((0.625, 0.875), (0.125, 0.375))
    assert (
        check_mixture_intersection(_bounds(), current, movement=movement, usage=usage).fingerprint == result.fingerprint
    )
    with pytest.raises(ValueError, match="below known"):
        check_mixture_intersection(_bounds(), current, movement=movement, usage=_usage())
    exhausted = check_mixture_intersection(_bounds(), _origin(), movement=movement, usage=_usage((0.75, 0.75)))
    assert exhausted.status == "infeasible" and exhausted.reason == "period_budget"


def test_missing_or_wrong_period_and_version_do_not_silently_reset_allowances():
    with pytest.raises(ValueError, match="explicit period"):
        check_mixture_intersection(_bounds(), _origin(), movement=_movement("per_interval"))
    with pytest.raises(ValueError, match="Per-release"):
        check_mixture_intersection(_bounds(), _origin(), movement=_movement(), usage=_usage())
    with pytest.raises(ValueError, match="requires a declared"):
        check_mixture_intersection(_bounds(), _origin(), usage=_usage())
    bad_origin = _origin().model_dump()
    bad_origin["weights"][0]["member"]["version"] = "v2"
    with pytest.raises(ValueError, match="expert IDs and versions"):
        check_mixture_intersection(_bounds(), CoefficientOrigin.model_validate(bad_origin))
    with pytest.raises(ValueError, match="every exact"):
        MovementUsage(interval=_ref("period"), origin=_origin(), used=_limits((0.0,)))


def test_coefficient_order_is_a_metamorphic_set_declaration():
    origins = []
    results = []
    for weights in permutations(_weights()):
        origin = CoefficientOrigin.model_validate({**_origin().model_dump(), "weights": weights})
        for bounds in permutations(_bounds().bounds):
            definition = ExpertBounds.model_validate({**_bounds().model_dump(), "bounds": bounds})
            origins.append(origin.fingerprint)
            results.append(check_mixture_intersection(definition, origin).fingerprint)
    assert len(set(origins)) == len(set(results)) == 1


@pytest.mark.parametrize("strength", [0.0, 0.1, 2.0, 1e308])
def test_l1_strength_is_not_share_alpha_and_contract_does_not_run_a_solver(strength):
    penalty = _penalty(strength)
    assert penalty.strength == strength
    assert L1MovementPenalty.from_json(penalty.to_json()) == penalty
    assert penalty.origin.weights == _weights()
    assert penalty.objective_convention == "half_squared_plus_full_l1_v1"
    assert "result" not in penalty.model_dump()


@pytest.mark.parametrize("strength", [-0.1, True, float("inf"), float("nan"), "0.1"])
def test_l1_bad_strength_rejects_without_interpreting_missing_as_zero(strength):
    with pytest.raises(ValueError):
        _penalty(strength)
    payload = _penalty().model_dump()
    del payload["strength"]
    with pytest.raises(ValueError):
        L1MovementPenalty.model_validate(payload)


def test_l1_meaning_profile_origin_and_legacy_projection_remain_explicit():
    payload = _penalty().model_dump()
    payload["semantics"] = UnitSemantics(unit="percent", meaning="mixture-coefficient-fraction-v1")
    with pytest.raises(ValueError, match="coefficient-fraction"):
        L1MovementPenalty.model_validate(payload)
    payload = _penalty().model_dump()
    payload["objective_convention"] = "half-l1-turnover"
    with pytest.raises(ValueError):
        L1MovementPenalty.model_validate(payload)
    plain, penalized = _controls(), _controls(_penalty())
    assert MixtureControls.from_json(plain.to_json()).output.kind == "projection"
    assert MixtureControls.from_json(penalized.to_json()).output.kind == "l1_movement"
    assert plain.binding != penalized.binding
    missing = plain.model_dump()
    del missing["output"]
    with pytest.raises(ValueError):
        MixtureControls.model_validate(missing)
    assert _penalty(origin=_origin(name="different-origin")).binding != _penalty().binding


def test_target_can_be_outside_hard_feasible_set_without_malformed_schema():
    controls = _controls()
    target = DeclaredTargetShare(
        **_metadata("target"),
        weights=_weights((1.0, 0.0)),
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
        alpha=0.5,
    )
    changed = MixtureControls.model_validate(
        {**controls.model_dump(), "target": target, "bounds": _bounds(uppers=(0.6, 1.0))}
    )
    assert changed.target is not None
    assert changed.target.weights[0].value > changed.bounds.bounds[0].upper
    bad = target.model_dump()
    bad["weights"][0]["member"]["version"] = "v2"
    with pytest.raises(ValueError, match="expert IDs and versions"):
        MixtureControls.model_validate({**controls.model_dump(), "target": bad})


def test_cadence_profiles_and_zoned_release_declare_no_observations_or_clock():
    round_gate = RoundCadence(panels=2)
    interval_gate = IntervalCadence(calendar=_ref("public-calendar"))
    assert round_gate.kind != interval_gate.kind
    interval = ReleaseInterval.model_validate(
        {
            "interval": _ref("period-1"),
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-01-02T00:00:00Z",
            "cutoff": "2026-01-01T23:00:00Z",
            "release_at": "2026-01-02T00:00:00Z",
        }
    )
    assert interval.release_at == interval.end
    assert interval_gate.model_dump() == {"kind": "interval_clock", "calendar": _ref("public-calendar").model_dump()}
    bad = interval.model_dump()
    bad["release_at"] = "2026-01-01T22:00:00Z"
    with pytest.raises(ValueError):
        ReleaseInterval.model_validate(bad)
    with pytest.raises(ValueError):
        RoundCadence(panels=True)
    with pytest.raises(ValueError):
        RoundCadence(panels=0)


@pytest.mark.parametrize("version", [0, 2, True, "1"])
def test_future_control_schemas_are_rejected_before_migration(version):
    payload = _controls().model_dump()
    payload["schema_version"] = version
    with pytest.raises(ValueError):
        MixtureControls.model_validate(payload)


def test_unchecked_fake_completed_intersection_is_rejected_on_serialization():
    result = check_mixture_intersection(_bounds((0.75, 0.75)), _origin())
    fake = result.model_copy(update={"status": "feasible", "reason": "nonempty"})
    with pytest.raises(ValueError, match="effective boxes"):
        fake.to_json()
    with pytest.raises(ValueError, match="full definitions"):
        result.model_copy(update={"reason": "nonempty"}).to_json()


def test_required_control_combination_binds_all_definitions_and_explicit_nulls():
    controls = _controls(_penalty())
    target = DeclaredTargetShare(
        **_metadata("target"),
        weights=_weights((0.25, 0.75)),
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
        alpha=0.1,
    )
    complete = MixtureControls.model_validate(
        {
            **controls.model_dump(),
            "target": target,
            "movement": _movement("per_interval"),
            "cadence": IntervalCadence(calendar=_ref("public-calendar")),
        }
    )
    assert MixtureControls.from_json(complete.to_json()) == complete
    assert complete.output.kind == "l1_movement" and complete.cadence.kind == "interval_clock"
    assert complete.binding != controls.binding
    missing = controls.model_dump()
    del missing["target"]
    with pytest.raises(ValueError):
        MixtureControls.model_validate(missing)


def test_numerical_core_simplex_hold_preserves_float_thirds_without_exact_real_claim():
    thirds = _origin((1 / 3, 1 / 3, 1 / 3))
    bounds = _bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    movement = _movement(values=(0.0, 0.0, 0.0))
    result = check_mixture_intersection(bounds, thirds, movement=movement)
    assert result.status == "feasible" and not result.exact_feasible
    assert result.numerical_profile == "binary64-simplex-v1"
    assert tuple(item.lower for item in result.effective) == tuple(weight.value for weight in thirds.weights)
    assert all(item.lower == item.upper for item in result.effective)
    assert MixtureIntersection.from_json(result.to_json()) == result
    with pytest.raises(ValueError, match="full definitions"):
        result.model_copy(update={"exact_feasible": True}).to_json()


def test_sum_tolerance_never_relaxes_empty_coordinate_intervals_or_large_residuals():
    fixed = (0.5, 0.5 - 2**-47)
    result = check_mixture_intersection(_bounds(fixed, fixed), _origin())
    assert result.status == "infeasible" and not result.exact_feasible
    assert result.reason == "simplex_sum"
    close = (0.5, 0.5 - 2**-50)
    admitted = check_mixture_intersection(_bounds(close, close), _origin())
    assert admitted.status == "feasible" and not admitted.exact_feasible
    conflict = check_mixture_intersection(
        _bounds((0.5 + 2**-52, 0.0)), _origin(), movement=_movement(values=(0.0, 0.0))
    )
    assert conflict.status == "infeasible" and conflict.reason == "empty_coordinate"


@pytest.mark.parametrize("used", [2**-60, math.ulp(0.0)])
def test_tiny_original_period_overspend_is_typed_infeasible_after_endpoint_rounding(used):
    movement = _movement("per_interval", "accumulated_absolute", (0.0, 0.0))
    result = check_mixture_intersection(_bounds(), _origin(), movement=movement, usage=_usage((used, used)))
    assert result.status == "infeasible" and result.reason == "period_budget"
    assert not result.exact_feasible
    assert all(bound.lower == bound.upper == 0.5 for bound in result.effective)
    assert MixtureIntersection.from_json(result.to_json()) == result


def test_original_empty_coordinate_is_not_hidden_by_rounding_display_endpoints():
    current = _origin((math.nextafter(0.5, 0.0), 0.5))
    allowance = math.nextafter(2**-54, 0.0)
    result = check_mixture_intersection(_bounds((0.5, 0.0)), current, movement=_movement(values=(allowance, 1.0)))
    assert result.reason == "empty_coordinate" and result.status == "infeasible"
    assert result.effective[0].lower == result.effective[0].upper == 0.5
    assert not result.exact_feasible


@pytest.mark.parametrize("route", ["python", "json", "reader"])
def test_intersection_readers_recompute_bounds_and_original_budget_facts(route):
    result = check_mixture_intersection(_bounds(uppers=(0.6, 1.0)), _origin(), movement=_movement())
    payload = result.model_dump(mode="json")
    payload["effective"][0]["upper"] = 0.7
    read = {
        "python": MixtureIntersection.model_validate,
        "json": MixtureIntersection.model_validate_json,
        "reader": MixtureIntersection.from_json,
    }[route]
    with pytest.raises(ValueError, match="full definitions"):
        read(payload if route == "python" else json.dumps(payload))
    payload = result.model_dump(mode="json")
    payload["bounds"]["definition_fingerprint"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="full definitions"):
        MixtureIntersection.model_validate(payload)
