"""Independent conformance drift, missingness, arithmetic and metamorphic tests."""

import json
from datetime import timedelta
from fractions import Fraction
from pathlib import Path

import pytest

from coleman.evaluation.prequential.conformance import (
    MetricTolerance,
    ReferenceConformanceReport,
    check_reference_conformance,
)
from coleman.evaluation.prequential.contracts import ConformanceLimits, ConformanceTrace
from tests.evaluation.prequential.conformance_support import (
    NOW,
    binding,
    changed,
    key,
    metric,
    observation,
    plan,
    receipt,
    trace,
)

FIXTURES = Path(__file__).parent / "fixtures" / "conformance-v1"


def test_real_nonidentity_finite_intervention_conforms_with_complete_recomputed_checks():
    actual = trace()
    result = check_reference_conformance(plan(), actual, actual)
    assert result.status == "pass"
    assert len(result.checks) == 34
    assert all(row.status == "pass" for row in result.checks)
    assert result.plan_fingerprint == plan().fingerprint
    assert result.reference_trace.steps[0].receipt.executed.mapping == "projected"
    assert result.validate_for(plan()) is None


@pytest.mark.parametrize("drift", ("proposed_action", "executed_action", "intervention", "cost", "time"))
def test_action_cost_time_or_intervention_drift_cannot_hide_behind_equal_final_score(drift):
    original = receipt()
    execution, component = original, observation(original)
    if drift == "proposed_action":
        execution = receipt(proposed="fast")
        component = observation(execution)
    elif drift == "executed_action":
        execution = receipt(selected="full", upper=8.0)
        component = observation(execution)
    elif drift == "intervention":
        execution = changed(original, intervention=binding("different-intervention"))
        component = observation(execution)
    elif drift == "cost":
        component = observation(original, value=3.0)
    else:
        assert component.applied_at is not None
        component = changed(component, applied_at=component.applied_at + timedelta(microseconds=1))
    actual = trace(execution, components=(component,))
    result = check_reference_conformance(plan(), trace(), actual)
    assert result.status == "fail"
    assert any(row.status == "fail" for row in result.checks)
    if drift != "cost":
        assert actual.steps[0].components[0].value.value == trace().steps[0].components[0].value.value


@pytest.mark.parametrize("side", ("reference", "actual"))
def test_missing_or_extra_opportunity_is_a_complete_catalogue_failure(side):
    empty = ConformanceTrace(schema_version=1, recipe=trace().recipe, steps=())
    left, right = (empty, trace()) if side == "reference" else (trace(), empty)
    missing = check_reference_conformance(plan(), left, right)
    assert missing.status == "fail"
    assert "missing_opportunity" in {row.reason for row in missing.checks}
    extra = trace(receipt(key("undeclared")))
    left, right = (extra, trace()) if side == "reference" else (trace(), extra)
    undeclared = check_reference_conformance(plan(), left, right)
    assert undeclared.status == "fail"
    assert "undeclared_opportunity" in {row.reason for row in undeclared.checks}


@pytest.mark.parametrize("missing", ("component", "receipt", "unknown", "unavailable"))
def test_missing_component_or_availability_remains_unknown_instead_of_equal_null_pass(missing):
    if missing == "component":
        actual = trace(components=())
    elif missing == "receipt":
        actual = trace(components=(observation(received=False),))
    else:
        actual = trace(components=(observation(value=None, status=missing),))
    reference = actual if missing != "component" else trace()
    result = check_reference_conformance(plan(), reference, actual)
    assert result.status == "unknown"
    assert any(row.status == "unknown" for row in result.checks)


def test_not_applicable_is_distinct_from_unknown_and_observed_zero():
    not_applicable = trace(components=(observation(value=None, status="not_applicable"),))
    result = check_reference_conformance(plan(), not_applicable, not_applicable)
    assert result.status == "pass"
    different = check_reference_conformance(plan(), not_applicable, trace(components=(observation(value=0.0),)))
    assert different.status == "fail"
    assert "different_status" in {row.reason for row in different.checks}


def test_declared_rational_tolerances_have_exact_boundary_and_no_float_overflow():
    declared = plan(tolerances=(MetricTolerance(metric_name="net_duration", absolute=0.125, relative=0.25),))
    reference = trace(components=(observation(value=-2.0),))
    boundary = trace(components=(observation(value=-1.375),))
    outside = trace(components=(observation(value=-1.25),))
    assert Fraction(1, 8) + Fraction(1, 4) * 2 == Fraction(5, 8)
    assert check_reference_conformance(declared, reference, boundary).status == "pass"
    assert check_reference_conformance(declared, reference, outside).status == "fail"
    huge = plan(tolerances=(MetricTolerance(metric_name="net_duration", relative=1e308),))
    positive = trace(components=(observation(value=20.0),))
    negative = trace(components=(observation(value=-20.0),))
    assert check_reference_conformance(huge, positive, negative).status == "pass"


def test_clock_tolerance_uses_inclusive_integer_microseconds_without_changing_source_clocks():
    reference = observation()
    boundary = changed(reference, applied_at=NOW + timedelta(seconds=14, microseconds=2))
    outside = changed(reference, applied_at=NOW + timedelta(seconds=14, microseconds=3))
    declared = plan(clock_tolerance_us=2)
    assert check_reference_conformance(declared, trace(), trace(components=(boundary,))).status == "pass"
    assert check_reference_conformance(declared, trace(), trace(components=(outside,))).status == "fail"
    assert boundary.timing.available_at == reference.timing.available_at


@pytest.mark.parametrize("field", ("adapter", "configuration", "data", "origin", "metric", "extra_metric"))
def test_reference_plan_is_not_a_self_reported_adapter_or_source_pass(field):
    execution = receipt()
    component = observation(execution)
    declared = plan()
    if field == "adapter":
        execution = changed(execution, adapter=binding("other-adapter"))
        component = observation(execution)
    elif field == "configuration":
        declared = plan(base_configuration_fingerprint="sha256:" + "3" * 64)
    elif field == "data":
        declared = plan(data_fingerprint="sha256:" + "3" * 64)
    elif field == "origin":
        component = observation(execution, origin="modeled_counterfactual")
    elif field == "metric":
        component = observation(execution, quantity=metric(unit="ms"))
    if field == "extra_metric":
        components = (component, observation(execution, quantity=metric("unexpected")))
    else:
        components = (component,)
    result = check_reference_conformance(declared, trace(), trace(execution, components=components))
    assert result.status == "fail"


@pytest.mark.parametrize("change", ("status", "checks", "plan"))
def test_persisted_report_recomputes_full_evidence_instead_of_trusting_a_claim(change):
    passing = check_reference_conformance(plan(), trace(), trace())
    if change == "status":
        invalid = passing.model_copy(update={"status": "fail"})
    elif change == "checks":
        invalid = passing.model_copy(update={"checks": passing.checks[:-1]})
    else:
        invalid = passing.model_copy(update={"plan": plan(data_fingerprint="sha256:" + "4" * 64)})
    with pytest.raises(ValueError, match="recomputed"):
        ReferenceConformanceReport.model_validate(invalid)


def test_validate_for_rejects_detached_posthoc_tolerance_plan():
    result = check_reference_conformance(plan(), trace(), trace())
    detached = plan(clock_tolerance_us=1)
    with pytest.raises(ValueError, match="another declared plan"):
        result.validate_for(detached)


def test_selected_limits_fail_for_actual_steps_checks_or_encoded_complete_envelope():
    limited_checks = plan(limits=ConformanceLimits(max_checks=1))
    checks_reference, checks_actual = trace(), trace()
    with pytest.raises(ValueError, match="checks exceed"):
        check_reference_conformance(limited_checks, checks_reference, checks_actual)
    second = trace(receipt(key("build-2")))
    combined = ConformanceTrace(schema_version=1, recipe=trace().recipe, steps=trace().steps + second.steps)
    limited_steps = plan(limits=ConformanceLimits(max_steps=1))
    with pytest.raises(ValueError, match="step/component"):
        check_reference_conformance(limited_steps, combined, combined)
    limited_bytes = plan(limits=ConformanceLimits(max_bytes=8192))
    bytes_reference, bytes_actual = trace(), trace()
    with pytest.raises(ValueError, match="byte budget|encoded-byte|byte capacity"):
        check_reference_conformance(limited_bytes, bytes_reference, bytes_actual)


def test_permuting_complete_steps_preserves_identity_but_appending_future_steps_changes_full_bundle():
    first, second = trace().steps[0], trace(receipt(key("build-2"))).steps[0]
    declared = plan(opportunities=(first.receipt.key, second.receipt.key))
    forward = ConformanceTrace(schema_version=1, recipe=trace().recipe, steps=(first, second))
    reverse = ConformanceTrace(schema_version=1, recipe=trace().recipe, steps=(second, first))
    assert forward.to_json() == reverse.to_json()
    assert check_reference_conformance(declared, forward, reverse).status == "pass"
    assert forward.steps[0].receipt.fingerprint in {first.receipt.fingerprint, second.receipt.fingerprint}
    assert forward.fingerprint != trace().fingerprint
    assert first.to_json() == trace().steps[0].to_json()


def test_independent_hand_arithmetic_keeps_gross_surrogate_and_modeled_net_cost_separate():
    hand = json.loads((FIXTURES / "hand.json").read_text())
    durations = tuple(Fraction(item) for item in hand["durations_seconds"])
    coefficients = tuple(Fraction(item) for item in hand["candidate_coefficients"])
    baseline = tuple(Fraction(item) for item in hand["baseline_coefficients"])
    gross = sum(x * p for x, p in zip(durations, coefficients, strict=True))
    fixed = sum(x * p for x, p in zip(durations, baseline, strict=True))
    movement = sum(abs(p - old) for p, old in zip(coefficients, baseline, strict=True))
    charge = movement * hand["modeled_seconds_per_full_coefficient_l1"]
    assert gross == hand["expected_candidate_gross_seconds"] == 3
    assert fixed == hand["expected_baseline_gross_seconds"] == 4
    assert gross / hand["frozen_common_scale_seconds"] == Fraction(hand["expected_candidate_surrogate"])
    assert movement == Fraction(hand["expected_full_coefficient_l1"])
    assert charge == hand["expected_modeled_charge_seconds"] == 2
    assert gross + charge == hand["expected_candidate_net_seconds"] == 5
    assert gross < fixed < gross + charge
    assert ((Fraction(0) + 1) / 2) ** 2 == Fraction(hand["square_at_mean"])
    assert (Fraction(0) ** 2 + 1) / 2 == Fraction(hand["mean_of_squares"])
    assert Fraction(hand["square_at_mean"]) < Fraction(hand["mean_of_squares"])


@pytest.mark.parametrize("side", ("reference", "actual"))
def test_reference_recipe_is_checked_against_retained_trace_not_only_declared_in_plan(side):
    original = trace()
    wrong = changed(original, recipe=binding("unmatched-recipe"))
    reference, actual = (wrong, original) if side == "reference" else (original, wrong)
    result = check_reference_conformance(plan(), reference, actual)
    assert result.status == "fail"
    assert any(row.name == side + "_recipe" and row.status == "fail" for row in result.checks)


@pytest.mark.parametrize("invalid", ({"absolute": -0.1}, {"relative": True}, {"absolute": float("inf")}))
def test_tolerance_contract_rejects_negative_nonfinite_and_coerced_values(invalid):
    fields = {"metric_name": "net_duration", **invalid}
    with pytest.raises(ValueError):
        MetricTolerance.model_validate(fields)


def test_metric_tolerance_monotonicity_over_independent_dyadic_differences():
    reference = trace(components=(observation(value=1.0),))
    deltas = (Fraction(0), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2))
    for delta in deltas:
        actual = trace(components=(observation(value=float(1 + delta)),))
        for allowance in (Fraction(0), Fraction(1, 4), Fraction(1, 2)):
            declared = plan(tolerances=(MetricTolerance(metric_name="net_duration", absolute=float(allowance)),))
            result = check_reference_conformance(declared, reference, actual)
            assert (result.status == "pass") == (delta <= allowance)
