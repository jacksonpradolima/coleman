"""Independent unit/affine arithmetic and content-bound semantic contracts."""

import json
import math
import sys
from decimal import Decimal
from fractions import Fraction
from typing import Any, cast

import pytest

from coleman.core.feedback import Origin, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.units import UnitSemantics
from coleman.objectives.arithmetic import checked_fraction
from coleman.objectives.bindings import DefinitionLineage, QuantityMeaning, require_quantity
from coleman.objectives.transforms import AffineLoss, IdentityLoss, LossApplication, apply_loss, loss_outcome
from coleman.objectives.units import ConversionReceipt, ConversionSpec, conversion_factor, convert_quantity


def ref(name):
    return Reference(id=name, version="1")


def lineage():
    return DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)


def meaning(name="net", unit="credit", role="objective", interpretation="raw_outcome", **bounds):
    return QuantityMeaning(
        spec=ScalarSpec(
            dimension=ref(name), semantics=UnitSemantics(unit=unit, meaning=name + "-v1"), role=role, **bounds
        ),
        interpretation=interpretation,
        horizon=None,
    )


def normalized(name="loss"):
    return meaning(name, "1", "loss", "normalized_loss", minimum=0.0, maximum=1.0)


def observed_loss(application):
    assert application.status == "applied"
    assert application.output is not None and application.output.status == "observed"
    assert application.output.value is not None
    return application.output.value


def affine(**changes):
    return AffineLoss.model_validate(
        {
            "schema_version": 1,
            "reference": ref("net-loss"),
            "lineage": lineage(),
            "input": meaning(),
            "output": normalized(),
            "lower": -5.0,
            "upper": 5.0,
            "direction": "maximize",
            "outside": "clip",
            "bound_basis": "modeling_assumption",
            **changes,
        }
    )


def raw(quantity, value=0.0, status="observed"):
    return Scalar(spec=quantity.spec, value=value, status=status)


@pytest.mark.parametrize("version", [0, 2, True, "1"])
def test_new_versions_reject_without_changing_old_core(version):
    payload = affine().model_dump(mode="json")
    payload["schema_version"] = version
    with pytest.raises(ValueError):
        AffineLoss.from_json(json.dumps(payload))


def test_unknown_and_duplicate_fields_and_nonzero_json_underflow_reject():
    encoded = affine().to_json()
    with pytest.raises(ValueError):
        AffineLoss.from_json(encoded[:-1] + ',"surprise":1}')
    with pytest.raises(ValueError, match="Duplicate"):
        AffineLoss.model_validate_json(encoded[:-1] + ',"schema_version":1}')
    payload = affine().model_dump(mode="json")
    payload["lower"] = "TOKEN"
    with pytest.raises(ValueError):
        AffineLoss.model_validate_json(json.dumps(payload).replace('"TOKEN"', "-1e-9999"))


@pytest.mark.parametrize("value", [True, "0", Decimal("1e-9999"), float("inf"), float("nan")])
def test_numerical_ingress_is_explicit(value):
    with pytest.raises(ValueError):
        affine(lower=value)


def test_interpretations_do_not_follow_dimensionless_unit_symbols():
    loss = normalized()
    probability = QuantityMeaning(
        spec=loss.spec.model_copy(update={"role": "prediction"}), interpretation="probability_estimate", horizon=None
    )
    with pytest.raises(ValueError, match="normalized loss"):
        IdentityLoss(schema_version=1, reference=ref("identity"), lineage=lineage(), quantity=probability)
    with pytest.raises(ValueError):
        QuantityMeaning(spec=loss.spec, interpretation="priority_score", horizon=None)
    with pytest.raises(ValueError, match="risk"):
        meaning(interpretation="risk_measure")
    with pytest.raises(ValueError):
        meaning(interpretation="probability_estimate", role="prediction")
    risk = QuantityMeaning(spec=meaning().spec, interpretation="risk_measure", horizon=None, risk_definition=ref("var"))
    assert risk.risk_definition == ref("var")
    with pytest.raises(ValueError):
        risk.model_copy(update={"interpretation": "raw_outcome"}).model_validate(
            risk.model_copy(update={"interpretation": "raw_outcome"})
        )


def test_exact_quantity_schema_includes_signed_zero_bounds():
    quantity = meaning(minimum=-0.0)
    changed = quantity.spec.model_copy(update={"minimum": 0.0})
    with pytest.raises(ValueError, match="exact"):
        require_quantity(Scalar(spec=changed, value=0.0, status="observed"), quantity)


@pytest.mark.parametrize("value,expected", [(-7.0, 1.0), (-5.0, 1.0), (-2.5, 0.75), (0.0, 0.5), (5.0, 0.0), (7.0, 0.0)])
def test_independent_affine_signed_zero_threshold_and_clipping(value, expected):
    definition = affine()
    result = apply_loss(definition, raw(definition.input, value))
    assert observed_loss(result) == expected
    assert result.raw.value == value
    expected_clip = "upper" if value < -5 else "lower" if value > 5 else "none"
    assert result.clipping == expected_clip
    assert LossApplication.from_json(result.to_json()).to_json() == result.to_json()


@pytest.mark.parametrize("status", ["unknown", "unavailable", "not_applicable"])
def test_missing_loss_and_conversion_remain_distinct(status):
    definition = affine()
    result = apply_loss(definition, raw(definition.input, None, status))
    assert result.status == "missing" and result.preclip_loss is None
    assert result.output is not None
    assert result.output.value is None and result.output.status == status
    conversion = ConversionSpec(
        schema_version=1, reference=ref("identity-unit"), lineage=lineage(), input=meaning(), output=meaning()
    )
    receipt = convert_quantity(conversion, raw(conversion.input, None, status))
    assert receipt.converted.status == status and receipt.converted.value is None


def test_reject_and_forged_receipts_do_not_claim_a_completed_loss():
    definition = affine(outside="reject")
    result = apply_loss(definition, raw(definition.input, -7.0))
    assert result.status == "rejected" and result.output is None
    assert result.raw.value == -7.0 and result.preclip_loss == 1.2
    with pytest.raises(ValueError):
        LossApplication.model_validate(result.model_copy(update={"status": "applied"}))
    applied = apply_loss(affine(), raw(meaning(), -2.5))
    with pytest.raises(ValueError):
        LossApplication.model_validate(applied.model_copy(update={"preclip_loss": 0.5}))


def test_identity_preserves_signed_zero_and_outcome_origin():
    definition = IdentityLoss(schema_version=1, reference=ref("identity"), lineage=lineage(), quantity=normalized())
    result = apply_loss(definition, raw(definition.quantity, -0.0))
    assert math.copysign(1, observed_loss(result)) == -1
    with pytest.raises(ValueError):
        LossApplication.model_validate(result.model_copy(update={"preclip_loss": 0.0}))
    origins: tuple[Origin, ...] = ("realized", "modeled_counterfactual", "synthetic")
    for origin in origins:
        original = OutcomeValue(**result.raw.model_dump(), origin=origin, transform=ref("raw-identity"))
        derived = loss_outcome(result, original)
        assert derived.origin == origin and derived.transform == definition.reference
    different = OutcomeValue(
        spec=normalized().spec, value=0.5, status="observed", origin="synthetic", transform=ref("x")
    )
    with pytest.raises(ValueError):
        loss_outcome(result, different)


def test_affine_definitions_reject_bad_directions_horizons_and_source_claims():
    for changes in [{"upper": -5.0}, {"lower": 6.0}, {"direction": "best"}, {"bound_basis": "fitted"}]:
        with pytest.raises(ValueError):
            affine(**changes)
    with pytest.raises(ValueError, match="source"):
        affine(bound_basis="external_bound")
    with pytest.raises(ValueError, match="horizon"):
        affine(output=normalized().model_copy(update={"horizon": ref("next-build")}))
    with pytest.raises(ValueError, match="full normalized"):
        affine(output=meaning("loss", "1", "loss", "normalized_loss", minimum=0.1, maximum=0.9))


def test_affine_scale_metamorphism_and_loss_values_have_independent_arithmetic():
    for lower, upper in [(-5.0, 5.0), (-4.0, 8.0), (0.0, 12.0)]:
        for value in [lower, (upper + lower) / 2, upper]:
            for scale in [0.5, 2.0, 8.0]:
                first = affine(lower=lower, upper=upper, direction="minimize")
                second = affine(lower=lower * scale, upper=upper * scale, direction="minimize")
                expected = float((Fraction(value) - Fraction(lower)) / (Fraction(upper) - Fraction(lower)))
                assert observed_loss(apply_loss(first, raw(first.input, value))) == expected
                assert observed_loss(apply_loss(second, raw(second.input, value * scale))) == expected


def test_conversion_catalog_and_receipts_preserve_units_and_raw_bytes():
    first = meaning("elapsed", "ms", "resource", "resource", minimum=-1000.0, maximum=5000.0)
    last = meaning("elapsed", "s", "resource", "resource", minimum=-1.0, maximum=5.0)
    definition = ConversionSpec(
        schema_version=1, reference=ref("elapsed-conversion"), lineage=lineage(), input=first, output=last
    )
    result = convert_quantity(definition, raw(first, -500.0))
    assert result.converted.value == -0.5 and result.raw.value == -500.0
    assert ConversionReceipt.from_json(result.to_json()).to_json() == result.to_json()
    assert conversion_factor("min", "us") == 60_000_000
    assert conversion_factor("MiB", "byte") == 1024**2
    assert conversion_factor("opaque-credit", "opaque-credit") == 1
    negative_zero = convert_quantity(definition, raw(first, -0.0))
    assert negative_zero.converted.value is not None
    assert float(negative_zero.converted.value).hex() == "-0x0.0p+0"
    with pytest.raises(ValueError):
        ConversionReceipt.model_validate(result.model_copy(update={"converted": raw(last, 0.5)}))
    for source, target in [("USD", "EUR"), ("s", "byte"), ("days", "s")]:
        with pytest.raises(ValueError):
            conversion_factor(source, target)
    for source, target in [("", ""), (True, True), ("s", " ")]:
        with pytest.raises(ValueError):
            conversion_factor(cast(Any, source), cast(Any, target))
    with pytest.raises(ValueError, match="bounds"):
        ConversionSpec(
            schema_version=1,
            reference=ref("bad"),
            lineage=lineage(),
            input=first,
            output=last.model_copy(update={"spec": last.spec.model_copy(update={"minimum": None})}),
        )
    with pytest.raises(ValueError, match="meaning"):
        ConversionSpec(
            schema_version=1,
            reference=ref("bad"),
            lineage=lineage(),
            input=first,
            output=meaning("different", "s", "resource", "resource"),
        )


def test_extreme_arithmetic_rejects_unrepresentable_outputs_and_keeps_valid_spans():
    for rational in [Fraction(1, 10**1000), Fraction(10**1000)]:
        with pytest.raises(ValueError):
            checked_fraction(rational)
    for value in [True, 0.0, "1"]:
        with pytest.raises(TypeError):
            checked_fraction(cast(Any, value))
    assert checked_fraction(Fraction()) == 0
    definition = affine(lower=-sys.float_info.max, upper=sys.float_info.max)
    assert observed_loss(apply_loss(definition, raw(definition.input, 0.0))) == 0.5
    tiny = affine(lower=0.0, upper=math.ulp(0.0), direction="minimize")
    with pytest.raises(ValueError, match="overflow"):
        apply_loss(tiny, raw(tiny.input, sys.float_info.max))
    with pytest.raises(TypeError):
        apply_loss(cast(Any, "affine"), raw(meaning()))


def test_definition_content_digest_changes_with_parameters_not_bare_reference():
    first, second = affine(), affine(lower=-10.0)
    assert first.reference == second.reference and first.binding != second.binding
    assert AffineLoss.from_json(first.to_json()).binding == first.binding
    with pytest.raises(ValueError):
        first.model_copy(update={"lower": float("nan")}).to_json()
