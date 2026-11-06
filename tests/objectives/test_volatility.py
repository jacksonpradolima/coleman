"""Independent arithmetic, causal scale selection and guarded receipt evidence."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError

from coleman.core.feedback import OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionLineage, QuantityMeaning
from coleman.objectives.volatility import (
    ClippingStats,
    FitSample,
    FitSnapshot,
    FrozenScale,
    RollingScaleSpec,
    ScaleSelection,
    SignedVolatilitySpec,
    VolatilityApplication,
    _rounded_sqrt,
    apply_signed_volatility,
    clipping_statistics,
    fit_rolling_scale,
    select_scale,
    volatility_outcome,
)

NOW = datetime(2026, 1, 10, tzinfo=UTC)


def ref(name, version="v1"):
    return Reference(id=name, version=version)


def digest(name):
    return "sha256:" + hashlib.sha256(name.encode()).hexdigest()


def lineage(**changes: object) -> DefinitionLineage:
    return DefinitionLineage.model_validate(
        {
            "source": ref("public-synthetic-definition"),
            "source_fingerprint": digest("definition"),
            "declared_at": NOW - timedelta(days=10),
            "supersedes": None,
            **changes,
        }
    )


def meaning(unit="resource_credit", horizon=None):
    return QuantityMeaning(
        spec=ScalarSpec(
            dimension=ref("net-outcome"),
            semantics=UnitSemantics(unit=unit, meaning="signed-net-outcome-v1"),
            role="objective",
        ),
        interpretation="raw_outcome",
        horizon=horizon,
    )


def scale_spec(quantity=None, **changes: object) -> RollingScaleSpec:
    quantity = quantity or meaning()
    fields = {
        "schema_version": 1,
        "reference": ref("rolling-sample-sd"),
        "lineage": lineage(),
        "quantity": quantity,
        "scale_spec": ScalarSpec(
            dimension=ref("past-sample-sd"),
            semantics=UnitSemantics(unit=quantity.spec.semantics.unit, meaning="sample-sd-ddof-one-v1"),
            role="diagnostic",
            minimum=0.0,
        ),
        "window": 3,
        "scope": "shared",
        "maximum_age_seconds": None,
    }
    return RollingScaleSpec.model_validate(fields | changes)


def historical(values=(-1.0, 0.0, 1.0), *, definition=None, member=None):
    definition = definition or scale_spec()
    samples = tuple(
        FitSample(
            observation=ref(f"observation-{index}"),
            value=Scalar(
                spec=definition.quantity.spec,
                value=value,
                status="observed" if value is not None else "unknown",
            ),
            timing=FeedbackTiming(
                event_at=NOW - timedelta(days=len(values) - index),
                available_at=NOW - timedelta(days=len(values) - index, hours=-1),
                mature_at=NOW - timedelta(days=len(values) - index, hours=-2),
                received_at=NOW - timedelta(days=len(values) - index, hours=-3),
            ),
        )
        for index, value in enumerate(values)
    )
    return FitSnapshot(
        schema_version=1,
        source=ref("public-synthetic-history"),
        quantity=definition.quantity,
        member=member if definition.scope == "per_member" else None,
        training_cutoff=NOW - timedelta(hours=2),
        samples=samples,
    )


def frozen(values=(-1.0, 0.0, 1.0), *, definition=None, member=None):
    definition = definition or scale_spec()
    return fit_rolling_scale(
        definition,
        historical(values, definition=definition, member=member),
        frozen_at=NOW - timedelta(hours=1),
    )


def loss_spec(definition=None, **changes: object) -> SignedVolatilitySpec:
    definition = definition or scale_spec()
    output = QuantityMeaning(
        spec=ScalarSpec(
            dimension=ref("bounded-signed-loss"),
            semantics=UnitSemantics(unit="1", meaning="signed-volatility-loss-v1"),
            role="loss",
            minimum=0.0,
            maximum=1.0,
        ),
        interpretation="normalized_loss",
        horizon=definition.quantity.horizon,
    )
    return SignedVolatilitySpec.model_validate(
        {
            "schema_version": 1,
            "reference": ref("signed-volatility-loss"),
            "lineage": lineage(),
            "input": definition.quantity,
            "output": output,
            "scale_definition": definition.binding,
            "c": 2.0,
        }
        | changes
    )


def decision_timing(**changes: object) -> DecisionTiming:
    return DecisionTiming.model_validate(
        {
            "decision_at": NOW,
            "information_cutoff": NOW,
            "target_start": NOW,
            "target_end": NOW + timedelta(days=1),
            "eligible_input_fingerprint": digest("selected-input"),
            "mode": "future",
        }
        | changes
    )


def selected(*, values=(-1.0, 0.0, 1.0), definition=None, member=None, **changes):
    definition = definition or scale_spec()
    # This fixture deliberately mutates ingress fields, including invalid types.
    parameters: dict[str, Any] = {
        "decision_fingerprint": digest("producing-decision"),
        "timing": decision_timing(),
        "outcome": ref("current-outcome"),
        "member": member,
        **changes,
    }
    return select_scale(
        loss_spec(definition),
        frozen(values, definition=definition, member=member),
        **parameters,
    )


def application(value, *, selection=None, status="observed", name=None):
    selection = selection or selected()
    if name is not None:
        selection = ScaleSelection.model_validate(
            selection.model_copy(update={"outcome": ref(name), "decision_fingerprint": digest(name)})
        )
    return apply_signed_volatility(
        selection,
        Scalar.model_validate({"spec": selection.definition.input.spec, "value": value, "status": status}),
    )


@pytest.mark.parametrize(
    ("value", "loss", "clipping"),
    [
        (-3.0, 1.0, "upper"),
        (-2.0, 1.0, "none"),
        (-1.0, 0.75, "none"),
        (0.0, 0.5, "none"),
        (-0.0, 0.5, "none"),
        (1.0, 0.25, "none"),
        (2.0, 0.0, "none"),
        (3.0, 0.0, "lower"),
    ],
)
def test_independent_signed_sample_scale_fixture(value, loss, clipping):
    result = application(value)
    assert result.selection.scale.scale.value == 1.0
    assert result.z == value
    assert result.preclip_loss == float((1 - Fraction.from_float(value) / 2) / 2)
    assert result.output.value == loss
    assert result.clipping == clipping
    assert result.reason == "computed"
    assert result.raw.value is not None
    assert result.raw.value.hex() == value.hex()
    assert VolatilityApplication.from_json(result.to_json()).to_json() == result.to_json()


def test_centering_is_for_scale_not_for_new_outcome():
    selection = selected(values=(9.0, 10.0, 11.0))
    assert selection.scale.scale.value == 1.0
    assert application(1.0, selection=selection).output.value == 0.25


def test_window_selection_is_event_ordered_and_input_permutation_invariant():
    definition = scale_spec()
    history = historical((500.0, -500.0, -1.0, 0.0, 1.0), definition=definition)
    fitted = fit_rolling_scale(definition, history, frozen_at=NOW)
    reversed_history = FitSnapshot.model_validate(history.model_copy(update={"samples": history.samples[::-1]}))
    again = fit_rolling_scale(definition, reversed_history, frozen_at=NOW)
    assert tuple(item.id for item in fitted.selected) == ("observation-2", "observation-3", "observation-4")
    assert fitted.scale.value == 1.0
    assert fitted.to_json() == again.to_json()
    assert fitted.fingerprint == again.fingerprint


def test_missing_rows_are_excluded_and_full_observed_support_is_required():
    complete = frozen((-1.0, None, 0.0, None, 1.0))
    assert complete.status == "available"
    assert complete.scale.value == 1.0
    partial = frozen((-1.0, None, 1.0))
    assert partial.status == "insufficient_support"
    assert partial.scale.status == "unavailable"
    assert partial.scale.value is None
    result = application(0.0, selection=selected(values=(-1.0, None, 1.0)))
    assert result.reason == "insufficient_support"
    assert result.output.value is None


@pytest.mark.parametrize("status", ["unknown", "unavailable", "not_applicable"])
def test_distinct_null_states_remain_null_without_scale_or_loss_imputation(status):
    result = application(None, status=status, selection=selected(values=(0.0, 0.0, 0.0)))
    assert result.output.status == status
    assert result.reason == "input_" + status
    assert result.z is None and result.preclip_loss is None
    assert result.clipping == "not_evaluated"


def test_observed_zero_scale_is_distinct_from_unknown_scale_and_zero_outcome():
    selection = selected(values=(0.0, 0.0, 0.0))
    assert selection.scale.status == "zero_scale"
    assert selection.scale.scale.status == "observed"
    assert selection.scale.scale.value == 0.0
    result = application(0.0, selection=selection)
    assert result.output.status == "unavailable"
    assert result.reason == "zero_scale"
    assert result.raw.status == "observed" and result.raw.value == 0.0


@pytest.mark.parametrize("field", ["received_at", "mature_at", "available_at"])
def test_future_or_unknown_visibility_cannot_enter_fit(field):
    history = historical()
    sample = history.samples[0]
    value = None if field == "received_at" else NOW + timedelta(days=1)
    fields = sample.timing.model_dump() | {field: value}
    if field == "available_at":
        fields["received_at"] = NOW + timedelta(days=2)
    timing = FeedbackTiming(**fields)
    altered = sample.model_copy(update={"timing": timing})
    with pytest.raises(ValueError, match="not received"):
        FitSnapshot.model_validate(history.model_copy(update={"samples": (altered, *history.samples[1:])}))


def test_cutoff_ties_are_eligible_and_wrong_revision_duplicates_reject():
    history = historical()
    boundary = history.training_cutoff
    timing = FeedbackTiming(event_at=boundary, available_at=boundary, mature_at=boundary, received_at=boundary)
    rows = (history.samples[0].model_copy(update={"timing": timing}), *history.samples[1:])
    assert FitSnapshot.model_validate(history.model_copy(update={"samples": rows})).samples
    duplicate = history.samples[0].model_copy(update={"observation": ref(history.samples[0].observation.id, "v2")})
    with pytest.raises(ValueError, match="one revision"):
        FitSnapshot.model_validate(history.model_copy(update={"samples": (*history.samples, duplicate)}))


def test_frozen_scale_reader_recomputes_all_facts_and_rejects_backdated_freeze():
    result = frozen()
    restored = FrozenScale.from_json(result.to_json())
    assert restored.fingerprint == result.fingerprint
    mutations = [
        {"selected": ()},
        {"scale": result.scale.model_copy(update={"value": 2.0})},
        {"status": "zero_scale"},
        {"frozen_at": NOW - timedelta(days=5)},
        {"definition": result.definition.model_copy(update={"lineage": lineage(declared_at=NOW)})},
    ]
    for changes in mutations:
        with pytest.raises(ValueError):
            FrozenScale.model_validate(result.model_copy(update=changes))


def test_delayed_outcome_uses_original_selection_and_cannot_use_later_fit():
    selection = selected()
    original = application(-1.0, selection=selection)
    restored = ScaleSelection.from_json(selection.to_json())
    assert application(-1.0, selection=restored).fingerprint == original.fingerprint
    later = frozen().model_copy(update={"frozen_at": NOW + timedelta(days=4)})
    with pytest.raises(ValueError, match="producing decision"):
        select_scale(
            selection.definition,
            later,
            decision_fingerprint=selection.decision_fingerprint,
            timing=selection.timing,
            outcome=selection.outcome,
        )
    assert "application" not in type(selection).model_fields
    assert application(-1.0, selection=selection).to_json() == original.to_json()


def test_ongoing_targets_do_not_allow_the_producing_outcome_into_its_own_fit():
    timing = decision_timing(mode="ongoing", target_start=NOW - timedelta(days=30))
    with pytest.raises(ValueError, match="Producing outcome"):
        selected(timing=timing, outcome=ref("observation-2", "corrected-v2"))
    assert selected(timing=timing).timing.mode == "ongoing"


def test_per_member_scope_and_exact_versions_are_required():
    definition = scale_spec(scope="per_member")
    member = ref("expert-a")
    good = selected(definition=definition, member=member)
    assert good.member == member
    with pytest.raises(ValueError, match="another expert"):
        ScaleSelection.model_validate(good.model_copy(update={"member": ref("expert-a", "v2")}))
    with pytest.raises(ValueError, match="scope"):
        frozen(definition=definition)


def test_staleness_is_measured_at_original_cutoff_with_explicit_seconds():
    definition = scale_spec(maximum_age_seconds=86400.0)
    assert application(0.0, selection=selected(definition=definition)).reason == "computed"
    stale_definition = scale_spec(maximum_age_seconds=86399.0)
    stale = application(0.0, selection=selected(definition=stale_definition))
    assert stale.reason == "stale_scale"
    assert stale.selection.scale.scale.value == 1.0
    assert stale.output.status == "unavailable"


@pytest.mark.parametrize("bad", [0, -1, True, float("inf"), float("nan"), "2"])
def test_invalid_transform_strength_rejects(bad):
    with pytest.raises(ValueError):
        loss_spec(c=bad)


@pytest.mark.parametrize("bad", [1, 0, 4097, True, 3.0])
def test_invalid_window_rejects(bad):
    with pytest.raises(ValueError):
        scale_spec(window=bad)


def test_incompatible_physical_units_meaning_horizon_and_bindings_reject():
    definition = scale_spec()
    cases = [
        {"scale_spec": definition.scale_spec.model_copy(update={"semantics": UnitSemantics(unit="s", meaning="sd")})},
        {"scale_spec": definition.scale_spec.model_copy(update={"role": "prediction"})},
        {"scale_spec": definition.scale_spec.model_copy(update={"minimum": -1.0})},
        {"scale_spec": definition.scale_spec.model_copy(update={"dimension": definition.quantity.spec.dimension})},
        {"quantity": definition.quantity.model_copy(update={"interpretation": "priority_score"})},
    ]
    for changes in cases:
        with pytest.raises(ValueError):
            RollingScaleSpec.model_validate(definition.model_copy(update=changes))
    selection = selected()
    with pytest.raises(ValueError, match="binding differs"):
        ScaleSelection.model_validate(selection.model_copy(update={"definition": loss_spec(scale_spec(window=2))}))
    wrong_quantity = meaning(unit="another_credit")
    with pytest.raises(ValueError, match="quantity/units/horizon"):
        ScaleSelection.model_validate(selection.model_copy(update={"definition": loss_spec(input=wrong_quantity)}))
    with pytest.raises(ValueError, match="same horizon"):
        loss_spec(output=selection.definition.output.model_copy(update={"horizon": ref("other-horizon")}))
    with pytest.raises(ValueError, match="declared after"):
        ScaleSelection.model_validate(
            selection.model_copy(update={"definition": loss_spec(lineage=lineage(declared_at=NOW + timedelta(days=1)))})
        )


def test_valid_score_or_loss_schema_cannot_substitute_for_raw_outcome_interpretation():
    raw = meaning()
    score = QuantityMeaning(
        spec=raw.spec.model_copy(update={"role": "prediction"}),
        interpretation="priority_score",
        horizon=None,
    )
    with pytest.raises(ValueError, match="raw or monetary"):
        scale_spec(quantity=score)
    with pytest.raises(ValueError, match="raw or monetary"):
        loss_spec(input=loss_spec().output)


def test_fitter_rejects_a_different_horizon_even_when_numeric_history_is_identical():
    expected = scale_spec()
    different = scale_spec(quantity=meaning(horizon=ref("different-horizon")))
    history = historical(definition=different)
    with pytest.raises(ValueError, match="quantity/horizon"):
        fit_rolling_scale(expected, history, frozen_at=NOW)


def test_receipt_facts_cannot_be_rewritten_including_signed_zero():
    result = application(0.0)
    for changes in [
        {"z": -0.0},
        {"preclip_loss": 0.4},
        {"clipping": "upper"},
        {"reason": "zero_scale"},
        {"output": result.output.model_copy(update={"value": 0.4})},
    ]:
        with pytest.raises(ValueError, match="differs"):
            VolatilityApplication.model_validate(result.model_copy(update=changes))


def test_small_standard_deviation_survives_internal_variance_below_binary64():
    value = 1e-200
    assert float(Fraction.from_float(value) ** 2) == 0.0
    result = frozen((-value, 0.0, value))
    assert result.scale.value == value
    assert result.scale.status == "observed"


def test_extreme_standard_deviation_and_operation_boundaries_reject():
    with pytest.raises(ValueError, match="overflows"):
        frozen((-sys.float_info.max, sys.float_info.max), definition=scale_spec(window=2))
    smallest = math.ulp(0.0)
    with pytest.raises(ValueError, match="underflows"):
        frozen((0.0,) * 8 + (smallest,), definition=scale_spec(window=9))
    with pytest.raises(ValueError, match="underflows"):
        application(smallest, selection=selected(values=(-2.0, 0.0, 2.0)))
    with pytest.raises(ValueError, match="overflow"):
        application(sys.float_info.max, selection=selected(values=(-1e-200, 0.0, 1e-200)))
    selection = selected()
    tiny_c = loss_spec(c=smallest)
    with pytest.raises(ValueError, match="overflow"):
        application(sys.float_info.max, selection=selection.model_copy(update={"definition": tiny_c}))


@pytest.mark.parametrize("base", [0.0, math.ulp(0.0), 1.0, math.nextafter(1.0, math.inf), 2.0])
def test_root_rounding_uses_independent_adjacent_midpoint_oracle(base):
    upper = math.nextafter(base, math.inf)
    low_ratio, high_ratio = Fraction.from_float(base), Fraction.from_float(upper)
    midpoint = (low_ratio + high_ratio) / 2
    # Exact square makes the true root the rational midpoint, with an even tie.
    bits_even = int.from_bytes(__import__("struct").pack(">d", base), "big") % 2 == 0
    expected = base if bits_even else upper
    if expected == 0:
        with pytest.raises(ValueError, match="underflows"):
            _rounded_sqrt(midpoint**2)
    else:
        assert _rounded_sqrt(midpoint**2) == expected
    assert _rounded_sqrt(high_ratio**2) == upper


def test_root_maximum_endpoint_and_overflow_midpoint():
    maximum = Fraction.from_float(sys.float_info.max)
    previous = Fraction.from_float(math.nextafter(sys.float_info.max, 0.0))
    boundary = maximum + (maximum - previous) / 2
    assert _rounded_sqrt((maximum + (boundary - maximum) / 2) ** 2) == sys.float_info.max
    with pytest.raises(ValueError, match="overflows"):
        _rounded_sqrt(boundary**2)


def test_random_sample_sd_matches_independent_high_precision_decimal_reference():
    rng = random.Random(44571)
    for _ in range(25):
        values = tuple(rng.uniform(-100, 100) for _ in range(7))
        result = frozen(values, definition=scale_spec(window=7))
        with localcontext() as context:
            context.prec = 100
            observed = tuple(Decimal.from_float(value) for value in values)
            mean = sum(observed) / Decimal(len(observed))
            variance = sum((value - mean) ** 2 for value in observed) / Decimal(len(observed) - 1)
            expected = float(variance.sqrt())
        assert result.scale.value == expected


def test_positive_unit_rescaling_preserves_surrogate_without_touching_raw_values():
    for factor in [0.5, 2.0, 1024.0]:
        scale = selected(values=tuple(value * factor for value in (-1.0, 0.0, 1.0)))
        assert application(-0.5 * factor, selection=scale).output.value == 0.625


def test_summary_denominators_keep_nulls_noncomputable_and_duplicate_retries_separate():
    rows = [
        application(-3.0, name="upper"),
        application(3.0, name="lower"),
        application(2.0, name="threshold"),
        application(0.0, name="zero"),
        *(application(None, status=status, name=status) for status in ["unknown", "unavailable", "not_applicable"]),
        application(0.0, name="zero-scale", selection=selected(values=(0.0, 0.0, 0.0))),
    ]
    stats = clipping_statistics((*rows, rows[0]), split=ref("development"))
    assert (stats.observed, stats.computed, stats.noncomputable) == (5, 4, 1)
    assert (stats.unknown, stats.unavailable, stats.not_applicable) == (1, 1, 1)
    assert (stats.lower, stats.upper, stats.lower_rate, stats.upper_rate) == (1, 1, 0.25, 0.25)
    assert len(stats.applications) == 8
    assert clipping_statistics(rows[::-1], split=ref("development")).fingerprint == stats.fingerprint
    assert ClippingStats.from_json(stats.to_json()).to_json() == stats.to_json()


def test_no_computed_loss_has_null_rates_rather_than_zero_over_zero():
    stats = clipping_statistics((application(None, status="unknown"),), split=ref("protected"))
    assert stats.computed == 0
    assert stats.lower_rate is None and stats.upper_rate is None


def test_summary_conflicts_mixed_groups_empty_and_unbounded_inputs_reject():
    original = application(0.0)
    with pytest.raises(ValueError, match="Conflicting"):
        clipping_statistics((original, application(-0.0)), split=ref("development"))
    with pytest.raises(ValueError, match="one exact transform"):
        clipping_statistics(
            (original, application(0.0, name="other", selection=selected(member=ref("expert")))),
            split=ref("development"),
        )
    with pytest.raises(ValueError, match="at least one"):
        clipping_statistics((), split=ref("development"))
    with pytest.raises(ValueError, match="4096"):
        clipping_statistics((original for _ in range(4097)), split=ref("development"))


@pytest.mark.parametrize(
    "changes",
    [{"observed": 2}, {"computed": 0}, {"lower": 2}, {"lower_rate": 1.0}, {"applications": ()}],
)
def test_summary_rejects_inconsistent_serialized_counts(changes):
    stats = clipping_statistics((application(0.0),), split=ref("development"))
    with pytest.raises(ValueError):
        ClippingStats.model_validate(stats.model_copy(update=changes))


def test_versioned_readers_reject_future_unknown_duplicate_keys_and_relaxed_strictness():
    receipt = application(0.0)
    objects = [scale_spec(), historical(), frozen(), loss_spec(), selected(), receipt]
    objects.append(clipping_statistics((receipt,), split=ref("development")))
    for record in objects:
        payload = json.loads(record.to_json())
        for changed in [payload | {"schema_version": 2}, payload | {"unexpected": True}]:
            with pytest.raises(ValueError):
                type(record).from_json(json.dumps(changed))
        with pytest.raises(ValueError):
            type(record).model_validate_json(record.to_json(), strict=False)
        duplicated = record.to_json().replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1)
        with pytest.raises(ValueError):
            type(record).from_json(duplicated)


def test_immutable_continuation_is_new_snapshot_and_preserves_old_output():
    old = frozen()
    old_json = old.to_json()
    restored = FrozenScale.from_json(old_json)
    newer = fit_rolling_scale(
        restored.definition,
        historical((-1.0, 0.0, 1.0, 20.0), definition=restored.definition),
        frozen_at=NOW,
    )
    assert newer.scale.value != old.scale.value
    assert newer.fingerprint != old.fingerprint
    assert old.to_json() == old_json == restored.to_json()


def test_full_bounded_window_and_too_many_rows():
    definition = scale_spec(window=4096)
    assert frozen((0.0,) * 4096, definition=definition).status == "zero_scale"
    with pytest.raises(ValidationError):
        historical((0.0,) * 4097, definition=definition)


@pytest.mark.parametrize("origin", ["realized", "modeled_counterfactual", "synthetic"])
@pytest.mark.parametrize("status", ["observed", "unknown", "unavailable", "not_applicable"])
def test_volatility_outcome_preserves_origin_and_explicit_input_null_states(origin, status):
    value = -1.0 if status == "observed" else None
    result = application(value, status=status)
    raw = OutcomeValue(spec=result.raw.spec, value=value, status=status, origin=origin, transform=None)
    original_json = raw.model_dump_json()
    output = volatility_outcome(result, raw)
    assert output.origin == origin
    assert output.status == status
    assert output.value == (0.75 if status == "observed" else None)
    assert output.transform == result.selection.definition.reference
    assert raw.model_dump_json() == original_json


@pytest.mark.parametrize("values", [(0.0, 0.0, 0.0), (-1.0, None, 1.0)])
def test_noncomputable_scale_remains_null_without_a_synthetic_feedback_value(values):
    result = application(0.0, selection=selected(values=values))
    raw = OutcomeValue(spec=result.raw.spec, value=0.0, status="observed", origin="synthetic", transform=None)
    output = volatility_outcome(result, raw)
    assert output.origin == "synthetic"
    assert output.status == "unavailable" and output.value is None


@pytest.mark.parametrize("value", [-0.0, 1.0])
def test_outcome_helper_rejects_different_raw_scalar_content(value):
    result = application(0.0)
    raw = OutcomeValue(spec=result.raw.spec, value=value, status="observed", origin="realized", transform=None)
    with pytest.raises(ValueError, match="exact.*raw"):
        volatility_outcome(result, raw)
