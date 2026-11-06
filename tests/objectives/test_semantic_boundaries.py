"""Independent causal, quantity and semantic compatibility boundary evidence."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from coleman.core.actions import MixtureSpace, Weight
from coleman.core.decisions import DecisionRequest
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.timing import DecisionTiming
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionLineage, QuantityMeaning
from coleman.objectives.context import ObjectiveDefinition, ObjectiveTerm, SemanticContext, SemanticRequest
from coleman.objectives.controls import L1MovementPenalty, MixtureControls, ProjectionOutput
from coleman.objectives.preferences import (
    ObjectiveWeight,
    ObjectiveWeights,
    ReferencePoint,
    ReferencePoints,
    StaticQuadraticAnchor,
)
from coleman.objectives.transforms import AffineLoss, LossApplication, apply_loss
from coleman.objectives.units import ConversionReceipt, ConversionSpec, convert_quantity
from coleman.objectives.volatility import (
    FitSnapshot,
    RollingScaleSpec,
    SignedVolatilitySpec,
    VolatilityApplication,
    fit_rolling_scale,
    select_scale,
)

FIXTURES = Path(__file__).parents[1] / "fixtures/objectives"


def _ref(name: str, version: str = "1") -> Reference:
    return Reference(id=name, version=version)


def _lineage(declared_at: datetime | None = None) -> DefinitionLineage:
    return DefinitionLineage(source=None, source_fingerprint=None, declared_at=declared_at, supersedes=None)


def _quantity(name: str, unit: str, *, minimum=None, maximum=None) -> QuantityMeaning:
    return QuantityMeaning(
        spec=ScalarSpec(
            dimension=_ref(name),
            semantics=UnitSemantics(unit=unit, meaning="elapsed-job-time-v1"),
            role="resource",
            minimum=minimum,
            maximum=maximum,
        ),
        interpretation="resource",
        horizon=_ref("one-build"),
    )


def _template() -> VolatilityApplication:
    return VolatilityApplication.from_json((FIXTURES / "volatility-application-v1.json").read_bytes())


def _controls() -> MixtureControls:
    return MixtureControls.from_json((FIXTURES / "mixture-controls-l1-v1.json").read_bytes())


def _space(controls: MixtureControls | None = None) -> MixtureSpace:
    control = controls or _controls()
    return MixtureSpace(
        feasible_set=_ref("explicit-experts"),
        members=tuple(weight.member for weight in control.initial_prior.weights),
        profile="binary64-simplex-v1",
    )


def _context(transform: AffineLoss | SignedVolatilitySpec, **changes: object) -> SemanticContext:
    objective = ObjectiveDefinition(
        schema_version=1,
        reference=_ref("one-objective"),
        lineage=_lineage(),
        terms=(ObjectiveTerm(quantity=transform.input, direction="maximize", transform=transform.binding),),
    )
    return SemanticContext.model_validate(
        {
            "schema_version": 1,
            "reference": _ref("semantic-context"),
            "lineage": _lineage(),
            "objective": objective,
            "transforms": (transform,),
            **changes,
        }
    )


def _core(context: SemanticContext, timing: DecisionTiming | None) -> DecisionRequest:
    return DecisionRequest(
        schema_version=1,
        task=_ref("synthetic-task"),
        decision_id="next-observation",
        round=4,
        space=_space(),
        objective=context.objective.reference,
        timing=timing,
    )


def _plain(**changes: object) -> tuple[DecisionRequest, SemanticContext]:
    prototype = _template()
    transform = AffineLoss(
        schema_version=1,
        reference=_ref("signed-affine-loss"),
        lineage=_lineage(),
        input=prototype.selection.definition.input,
        output=prototype.selection.definition.output,
        lower=-5.0,
        upper=5.0,
        direction="maximize",
        outside="clip",
        bound_basis="modeling_assumption",
    )
    context = _context(transform, **changes)
    return _core(context, prototype.selection.timing), context


def _indexed_scales() -> tuple[DecisionRequest, SemanticContext]:
    prototype = _template().selection
    definition = RollingScaleSpec.model_validate(prototype.scale.definition.model_copy(update={"scope": "per_member"}))
    transform = SignedVolatilitySpec.model_validate(
        prototype.definition.model_copy(update={"scale_definition": definition.binding})
    )
    context = _context(transform)
    core = _core(context, prototype.timing)
    assert isinstance(core.space, MixtureSpace)
    selections = []
    for index, member in enumerate(core.space.members):
        # Distinct expert histories are allowed by the declared per-member protocol.
        factor = float(index + 1)
        rows = []
        for sample in prototype.scale.snapshot.samples:
            assert sample.value.value is not None
            rows.append(
                sample.model_copy(
                    update={
                        "value": sample.value.model_copy(update={"value": sample.value.value * factor}),
                    }
                )
            )
        history = FitSnapshot.model_validate(
            prototype.scale.snapshot.model_copy(update={"member": member, "samples": tuple(rows)})
        )
        scale = fit_rolling_scale(definition, history, frozen_at=prototype.scale.frozen_at)
        selections.append(
            select_scale(
                transform,
                scale,
                decision_fingerprint=core.fingerprint,
                timing=prototype.timing,
                outcome=_ref("outcome-" + member.id),
                member=member,
            )
        )
    context = SemanticContext.model_validate(context.model_copy(update={"scale_selections": tuple(selections)}))
    return core, context


def test_actual_per_member_scale_coverage_allows_distinct_frozen_series_and_round_trips():
    core, context = _indexed_scales()
    value = SemanticRequest(schema_version=1, request=core, context=context)
    assert {selection.scale.scale.value for selection in value.context.scale_selections} == {1.0, 2.0}
    assert SemanticRequest.from_json(value.to_json()).to_json() == value.to_json()
    assert isinstance(value.request.space, MixtureSpace)
    for member in value.request.space.members:
        assert sum(selection.member == member for selection in value.context.scale_selections) == 1


@pytest.mark.parametrize("mismatch", ["none", "target", "inputs"])
def test_signed_scale_request_rejects_original_timing_mismatch_even_with_correct_digest(mismatch):
    core, context = _indexed_scales()
    assert core.timing is not None
    replacement = None
    if mismatch == "target":
        replacement = core.timing.model_copy(update={"target_end": core.timing.target_end + timedelta(hours=1)})
    elif mismatch == "inputs":
        replacement = core.timing.model_copy(update={"eligible_input_fingerprint": "sha256:" + "f" * 64})
    changed = DecisionRequest.model_validate(core.model_copy(update={"timing": replacement}))
    selections = tuple(
        selection.model_copy(update={"decision_fingerprint": changed.fingerprint})
        for selection in context.scale_selections
    )
    with pytest.raises(ValueError, match="original timing"):
        SemanticRequest(
            schema_version=1, request=changed, context=context.model_copy(update={"scale_selections": selections})
        )


def test_indexed_scale_coverage_uses_exact_member_versions_and_rejects_partial_retries():
    core, context = _indexed_scales()
    with pytest.raises(ValueError, match="cover"):
        SemanticRequest(
            schema_version=1,
            request=core,
            context=context.model_copy(update={"scale_selections": context.scale_selections[:1]}),
        )
    assert isinstance(core.space, MixtureSpace)
    changed_members = tuple(member.model_copy(update={"version": "changed"}) for member in core.space.members)
    changed = core.model_copy(update={"space": core.space.model_copy(update={"members": changed_members})})
    with pytest.raises(ValueError, match="outside"):
        SemanticRequest(schema_version=1, request=changed, context=context)


def test_projection_control_profile_retains_timed_configuration_validation_without_l1():
    controls = MixtureControls.model_validate(_controls().model_copy(update={"output": ProjectionOutput()}))
    core, context = _plain(controls=controls)
    value = SemanticRequest(schema_version=1, request=core, context=context)
    assert value.context.controls is not None
    assert value.context.controls.output.kind == "projection"


def _anchor(controls: MixtureControls) -> StaticQuadraticAnchor:
    return StaticQuadraticAnchor(
        schema_version=1,
        reference=_ref("static-preference-anchor"),
        lineage=_lineage(),
        weights=controls.initial_prior.weights,
        profile="binary64-simplex-v1",
        semantics=controls.initial_prior.semantics,
        strength=0.25,
        output=ScalarSpec(
            dimension=_ref("subjective-anchor-value"),
            role="objective",
            semantics=UnitSemantics(unit="1", meaning="subjective-value-v1"),
            minimum=0.0,
        ),
    )


def test_static_anchor_and_initial_prior_membership_are_validated_separately():
    controls = _controls()
    anchor = _anchor(controls)
    core, context = _plain(anchor=anchor, initial_prior=controls.initial_prior)
    assert SemanticRequest(schema_version=1, request=core, context=context)
    alien = (Weight(member=_ref("unmodeled-expert"), value=1.0),)
    for field, value, message in [
        ("anchor", anchor.model_copy(update={"weights": alien}), "Static anchor"),
        ("initial_prior", controls.initial_prior.model_copy(update={"weights": alien}), "Initial prior"),
    ]:
        with pytest.raises(ValueError, match=message):
            SemanticRequest(schema_version=1, request=core, context=context.model_copy(update={field: value}))


@pytest.mark.parametrize(
    "owner",
    [
        "preferences",
        "reference_points",
        "anchor",
        "initial_prior",
        "controls",
        "controls.initial_prior",
        "controls.target",
        "controls.bounds",
        "controls.movement",
        "controls.output",
        "controls.output.origin",
    ],
)
def test_every_nested_known_declaration_obeys_the_producing_cutoff(owner):
    controls = _controls()
    core, context = _plain(controls=controls)
    assert core.timing is not None
    cutoff = core.timing.information_cutoff
    operation = context.transforms[0]
    assert isinstance(operation, AffineLoss)
    loss = operation.output
    raw = context.objective.terms[0].quantity
    definitions = {
        "preferences": ObjectiveWeights(
            schema_version=1,
            reference=_ref("declared-tradeoff"),
            lineage=_lineage(),
            terms=(ObjectiveWeight(quantity=loss, weight=1.0),),
            output=ScalarSpec(
                dimension=_ref("subjective-value"),
                role="objective",
                semantics=UnitSemantics(unit="1", meaning="subjective-value-v1"),
                minimum=0.0,
                maximum=1.0,
            ),
        ),
        "reference_points": ReferencePoints(
            schema_version=1,
            reference=_ref("raw-target"),
            lineage=_lineage(),
            points=(ReferencePoint(quantity=raw, value=Scalar(spec=raw.spec, value=0.0, status="observed")),),
        ),
        "anchor": _anchor(controls),
        "initial_prior": controls.initial_prior,
    }
    for declared_at, accepted in [(None, True), (cutoff, True), (cutoff + timedelta(microseconds=1), False)]:
        source = _lineage(declared_at)
        if not owner.startswith("controls"):
            changed = definitions[owner].model_copy(update={"lineage": source})
            # The standalone prior case must not fail earlier as a conflicting duplicate.
            updates = {owner: changed} | ({"controls": None} if owner == "initial_prior" else {})
            candidate = context.model_copy(update=updates)
        else:
            changed = controls
            if owner == "controls":
                changed = controls.model_copy(update={"lineage": source})
            elif owner == "controls.output.origin":
                assert isinstance(controls.output, L1MovementPenalty)
                output = controls.output.model_copy(
                    update={"origin": controls.output.origin.model_copy(update={"lineage": source})}
                )
                changed = controls.model_copy(update={"output": output})
            else:
                field = owner.split(".")[1]
                child = getattr(controls, field)
                assert child is not None
                changed = controls.model_copy(update={field: child.model_copy(update={"lineage": source})})
            candidate = context.model_copy(update={"controls": changed})
        if accepted:
            assert SemanticRequest(schema_version=1, request=core, context=candidate)
        else:
            with pytest.raises(ValueError, match="declaration"):
                SemanticRequest(schema_version=1, request=core, context=candidate)


def test_context_set_permutation_preserves_identity():
    core, context = _plain()
    first = context.transforms[0]
    assert isinstance(first, AffineLoss)
    second_input = first.input.model_copy(
        update={"spec": first.input.spec.model_copy(update={"dimension": _ref("other-net")})}
    )
    second_output = first.output.model_copy(
        update={"spec": first.output.spec.model_copy(update={"dimension": _ref("other-loss")})}
    )
    second = AffineLoss.model_validate(
        first.model_copy(update={"reference": _ref("other-transform"), "input": second_input, "output": second_output})
    )
    terms = (
        *context.objective.terms,
        ObjectiveTerm(quantity=second.input, direction="maximize", transform=second.binding),
    )
    objective = ObjectiveDefinition.model_validate(context.objective.model_copy(update={"terms": terms}))
    contexts = [
        SemanticContext.model_validate(
            context.model_copy(
                update={
                    "objective": objective.model_copy(update={"terms": sequence}),
                    "transforms": transforms,
                }
            )
        )
        for sequence, transforms in [(terms, (first, second)), (terms[::-1], (second, first))]
    ]
    envelopes = [SemanticRequest(schema_version=1, request=core, context=item) for item in contexts]
    assert envelopes[0].fingerprint == envelopes[1].fingerprint


@pytest.mark.parametrize(
    ("source", "target", "factor"),
    [
        ("KiB", "byte", 1024),
        ("MiB", "KiB", 1024),
        ("s", "ms", 1000),
        ("min", "s", 60),
    ],
)
def test_positive_physical_conversion_round_trip_keeps_bounds_sign_and_dimension_semantics(source, target, factor):
    first = _quantity("source-time", source, minimum=-2.0, maximum=4.0)
    last = _quantity("target-time", target, minimum=-2.0 * factor, maximum=4.0 * factor)
    forward = ConversionSpec(
        schema_version=1, reference=_ref("forward-units"), lineage=_lineage(), input=first, output=last
    )
    reverse = ConversionSpec(
        schema_version=1, reference=_ref("reverse-units"), lineage=_lineage(), input=last, output=first
    )
    for value in [-2.0, -0.0, 0.0, 0.5, 4.0]:
        raw = Scalar(spec=first.spec, value=value, status="observed")
        scaled = convert_quantity(forward, raw)
        restored = convert_quantity(reverse, scaled.converted)
        assert scaled.converted.value == value * factor
        assert restored.converted.model_dump_json() == raw.model_dump_json()
        assert ConversionReceipt.from_json(scaled.to_json()).to_json() == scaled.to_json()


@pytest.mark.parametrize(
    ("source", "target", "value"),
    [
        ("MiB", "byte", sys.float_info.max),
        ("byte", "MiB", math.ulp(0.0)),
    ],
)
def test_conversion_operation_overflow_or_nonzero_underflow_preserves_raw_input(source, target, value):
    first, last = _quantity("source", source), _quantity("target", target)
    definition = ConversionSpec(
        schema_version=1, reference=_ref("extreme-units"), lineage=_lineage(), input=first, output=last
    )
    raw = Scalar(spec=first.spec, value=value, status="observed")
    before = raw.model_dump_json()
    with pytest.raises(ValueError, match="overflow|underflow"):
        convert_quantity(definition, raw)
    assert raw.model_dump_json() == before


def test_conversion_definition_rejects_bounds_that_overflow_even_without_an_observed_value():
    first = _quantity("source", "MiB", maximum=sys.float_info.max)
    with pytest.raises(ValueError, match="overflow"):
        ConversionSpec(
            schema_version=1,
            reference=_ref("invalid-bounds"),
            lineage=_lineage(),
            input=first,
            output=_quantity("target", "byte"),
        )


def test_guarded_receipt_readers_reject_signed_zero_and_derived_value_forgery():
    first, last = _quantity("seconds", "s"), _quantity("milliseconds", "ms")
    definition = ConversionSpec(schema_version=1, reference=_ref("units"), lineage=_lineage(), input=first, output=last)
    receipt = convert_quantity(definition, Scalar(spec=first.spec, value=0.0, status="observed"))
    encoded = json.loads(receipt.to_json())
    encoded["converted"]["value"] = -0.0
    with pytest.raises(ValueError, match="does not reproduce"):
        ConversionReceipt.from_json(json.dumps(encoded))
    _, context = _plain()
    operation = context.transforms[0]
    assert isinstance(operation, AffineLoss)
    loss = apply_loss(operation, Scalar(spec=operation.input.spec, value=0.0, status="observed"))
    encoded = json.loads(loss.to_json())
    encoded["output"]["value"] = 0.25
    with pytest.raises(ValueError, match="does not reproduce"):
        LossApplication.from_json(json.dumps(encoded))


def test_guarded_semantic_reader_rejects_late_nested_declarations():
    core, context = _plain()
    request = SemanticRequest(schema_version=1, request=core, context=context)
    assert core.timing is not None
    encoded = json.loads(request.to_json())
    encoded["context"]["lineage"]["declared_at"] = (core.timing.information_cutoff + timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="declaration"):
        SemanticRequest.from_json(json.dumps(encoded))
