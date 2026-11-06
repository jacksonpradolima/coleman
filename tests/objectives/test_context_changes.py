"""Semantic retries, compatible v1 requests and future-only change declarations."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from coleman.core.actions import ChoiceSpace, MixtureSpace
from coleman.core.decisions import DecisionRequest
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionLineage, QuantityMeaning
from coleman.objectives.changes import SemanticChange, validate_change
from coleman.objectives.context import (
    ObjectiveDefinition,
    ObjectiveTerm,
    SemanticContext,
    SemanticRequest,
    deduplicate_semantic_requests,
)
from coleman.objectives.controls import L1MovementPenalty, MixtureControls
from coleman.objectives.feasibility import ScalarConstraint
from coleman.objectives.preferences import ObjectiveWeight, ObjectiveWeights, ReferencePoint, ReferencePoints
from coleman.objectives.transforms import AffineLoss, IdentityLoss
from coleman.objectives.volatility import ScaleSelection, VolatilityApplication, fit_rolling_scale

FIXTURES = Path(__file__).parents[1] / "fixtures/objectives"


def ref(name):
    return Reference(id=name, version="1")


def lineage():
    return DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)


def definition(lower=-5.0):
    raw = QuantityMeaning(
        spec=ScalarSpec(
            dimension=ref("net"), semantics=UnitSemantics(unit="credit", meaning="net-credit"), role="objective"
        ),
        interpretation="raw_outcome",
        horizon=None,
    )
    loss = QuantityMeaning(
        spec=ScalarSpec(
            dimension=ref("loss"),
            semantics=UnitSemantics(unit="1", meaning="loss"),
            role="loss",
            minimum=0.0,
            maximum=1.0,
        ),
        interpretation="normalized_loss",
        horizon=None,
    )
    return AffineLoss(
        schema_version=1,
        reference=ref("transform"),
        lineage=lineage(),
        input=raw,
        output=loss,
        lower=lower,
        upper=5.0,
        direction="maximize",
        outside="clip",
        bound_basis="modeling_assumption",
    )


def context(transform=None, **changes):
    transform = transform or definition()
    term = ObjectiveTerm(quantity=transform.input, direction="maximize", transform=transform.binding)
    objective = ObjectiveDefinition(schema_version=1, reference=ref("objective"), lineage=lineage(), terms=(term,))
    return SemanticContext.model_validate(
        {
            "schema_version": 1,
            "reference": ref("context"),
            "lineage": lineage(),
            "objective": objective,
            "transforms": (transform,),
            **changes,
        }
    )


def request(ctx=None):
    ctx = ctx or context()
    inner = DecisionRequest(
        schema_version=1,
        task=ref("allocation-task"),
        decision_id="one",
        round=0,
        space=ChoiceSpace(feasible_set=ref("two-alternatives"), members=(ref("a"), ref("b"))),
        objective=ctx.objective.reference,
    )
    return SemanticRequest(schema_version=1, request=inner, context=ctx)


def change(previous=None, replacement=None, **updates):
    return SemanticChange.model_validate(
        {
            "schema_version": 1,
            "stream": ref("allocation-task"),
            "change_id": "new-objective-context",
            "previous": previous or context().binding,
            "replacement": replacement or context(definition(-10.0)).binding,
            "effective_round": 2,
            "effective_at": None,
            "disposition": "restart",
            "migration": None,
            "lineage": lineage(),
            **updates,
        }
    )


def test_optional_context_preserves_existing_request_and_no_preference():
    envelope = request()
    original = envelope.request.to_json()
    assert envelope.context.preferences is None
    assert envelope.logical_identity == envelope.request.logical_identity
    assert SemanticRequest.from_json(envelope.to_json()).to_json() == envelope.to_json()
    assert DecisionRequest.from_json(original).to_json() == original
    assert envelope.request.to_json() == original


def test_context_binds_parameters_not_only_the_same_human_reference():
    first, second = request(), request(context(definition(-10.0)))
    assert first.request.to_json() == second.request.to_json()
    assert first.logical_identity == second.logical_identity
    assert first.fingerprint != second.fingerprint
    assert len(deduplicate_semantic_requests((first, first))) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        deduplicate_semantic_requests((first, second))
    for invalid in [[], (first,) * 4097]:
        with pytest.raises(ValueError):
            deduplicate_semantic_requests(cast(Any, invalid))


def test_context_rejects_missing_conflicting_and_direction_substituted_definitions():
    original = context()
    for changed in [(), (definition(-10.0),), (definition(), definition())]:
        with pytest.raises(ValueError):
            SemanticContext.model_validate(original.model_copy(update={"transforms": changed}))
    term = original.objective.terms[0].model_copy(update={"direction": "minimize"})
    objective = ObjectiveDefinition.model_validate(original.objective.model_copy(update={"terms": (term,)}))
    with pytest.raises(ValueError, match="direction"):
        context(objective=objective)
    term = original.objective.terms[0]
    changed_quantity = term.quantity.model_copy(update={"horizon": ref("later")})
    changed = ObjectiveDefinition.model_validate(
        original.objective.model_copy(update={"terms": (term.model_copy(update={"quantity": changed_quantity}),)})
    )
    with pytest.raises(ValueError, match="exact input"):
        context(objective=changed)
    with pytest.raises(ValueError):
        ObjectiveDefinition.model_validate(original.objective.model_copy(update={"terms": (term, term)}))
    invalid = request().request.model_copy(update={"objective": ref("other")})
    with pytest.raises(ValueError, match="objective"):
        SemanticRequest(schema_version=1, request=invalid, context=original)


def test_explicit_tradeoff_and_reference_point_keep_exact_loss_raw_semantics():
    transform = definition()
    weighted = ObjectiveWeights(
        schema_version=1,
        reference=ref("weights"),
        lineage=lineage(),
        terms=(ObjectiveWeight(quantity=transform.output, weight=1.0),),
        output=ScalarSpec(
            dimension=ref("subjective"),
            semantics=UnitSemantics(unit="1", meaning="subjective"),
            role="objective",
            minimum=0.0,
            maximum=1.0,
        ),
    )
    points = ReferencePoints(
        schema_version=1,
        reference=ref("reference-points"),
        lineage=lineage(),
        points=(
            ReferencePoint(
                quantity=transform.input, value=Scalar(spec=transform.input.spec, value=0.0, status="observed")
            ),
        ),
    )
    assert context(preferences=weighted, reference_points=points).preferences.binding == weighted.binding
    foreign_loss = transform.output.model_copy(update={"horizon": ref("different")})
    bad_weights = ObjectiveWeights.model_validate(
        weighted.model_copy(update={"terms": (ObjectiveWeight(quantity=foreign_loss, weight=1.0),)})
    )
    with pytest.raises(ValueError, match="exactly"):
        context(preferences=bad_weights)
    wrong = transform.input.model_copy(update={"horizon": ref("different")})
    bad_points = ReferencePoints.model_validate(
        points.model_copy(
            update={
                "points": (ReferencePoint(quantity=wrong, value=Scalar(spec=wrong.spec, value=0.0, status="observed")),)
            }
        )
    )
    with pytest.raises(ValueError, match="Reference points"):
        context(reference_points=bad_points)


def test_identity_loss_context_has_no_inferred_maximization():
    identity = IdentityLoss(
        schema_version=1, reference=ref("identity"), lineage=lineage(), quantity=definition().output
    )
    objective = ObjectiveDefinition(
        schema_version=1,
        reference=ref("objective"),
        lineage=lineage(),
        terms=(ObjectiveTerm(quantity=identity.quantity, direction="minimize", transform=identity.binding),),
    )
    ctx = SemanticContext(
        schema_version=1, reference=ref("context"), lineage=lineage(), objective=objective, transforms=(identity,)
    )
    assert ctx.objective.terms[0].direction == "minimize"


def test_future_change_never_rewrites_old_content_or_executes_a_migration():
    first = context()
    before = first.to_json()
    declared = change(previous=first.binding)
    assert validate_change(declared, stream=declared.stream, current=first.binding, frontier=1) == declared
    assert first.to_json() == before
    assert SemanticChange.from_json(declared.to_json()).to_json() == declared.to_json()
    named = change(disposition="external_migration", migration=ref("host-migration"))
    assert named.migration == ref("host-migration")
    for invalid in [change(effective_round=1), change(stream=ref("other"))]:
        with pytest.raises(ValueError):
            validate_change(invalid, stream=declared.stream, current=first.binding, frontier=1)
    with pytest.raises(ValueError, match="binding"):
        validate_change(declared, stream=declared.stream, current=declared.replacement, frontier=1)
    for invalid in [True, -1, "1"]:
        with pytest.raises(ValueError):
            validate_change(declared, stream=declared.stream, current=first.binding, frontier=cast(Any, invalid))
    for updates in [
        {"replacement": first.binding},
        {"migration": ref("unexpected")},
        {"disposition": "external_migration"},
    ]:
        with pytest.raises(ValueError):
            change(**updates)


def test_timed_change_requires_consistent_supplied_future_instants():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    declared = change(effective_at=now + timedelta(days=1))
    common: dict[str, Any] = {"stream": declared.stream, "current": declared.previous, "frontier": 1}
    assert validate_change(declared, **common, frontier_at=now) == declared
    for instant in [None, now + timedelta(days=1), now + timedelta(days=2), datetime(2026, 1, 1)]:
        with pytest.raises(ValueError):
            validate_change(declared, **common, frontier_at=instant)
    with pytest.raises(ValueError):
        validate_change(change(), **common, frontier_at=now)


def signed_context(member=None):
    application = VolatilityApplication.from_json((FIXTURES / "volatility-application-v1.json").read_bytes())
    selection = ScaleSelection.model_validate(application.selection.model_copy(update={"member": member}))
    transform = selection.definition
    objective = ObjectiveDefinition(
        schema_version=1,
        reference=ref("signed-objective"),
        lineage=lineage(),
        terms=(ObjectiveTerm(quantity=transform.input, direction="maximize", transform=transform.binding),),
    )
    core = DecisionRequest.model_validate(
        request().request.model_copy(update={"objective": objective.reference, "timing": selection.timing})
    )
    selection = ScaleSelection.model_validate(selection.model_copy(update={"decision_fingerprint": core.fingerprint}))
    ctx = SemanticContext(
        schema_version=1,
        reference=ref("signed-context"),
        lineage=lineage(),
        objective=objective,
        transforms=(transform,),
        scale_selections=(selection,),
    )
    return core, ctx


def test_signed_request_freezes_scale_to_producing_request_and_full_definition():
    core, ctx = signed_context()
    value = SemanticRequest(schema_version=1, request=core, context=ctx)
    assert SemanticRequest.from_json(value.to_json()).to_json() == value.to_json()
    with pytest.raises(ValueError, match="preselected"):
        SemanticRequest(schema_version=1, request=core, context=ctx.model_copy(update={"scale_selections": ()}))
    selection = ctx.scale_selections[0]
    mismatched = ScaleSelection.model_validate(
        selection.model_copy(update={"decision_fingerprint": "sha256:" + "0" * 64})
    )
    for changes in [{"scale_selections": (mismatched,)}, {"scale_selections": (selection, selection)}]:
        with pytest.raises(ValueError):
            SemanticRequest(schema_version=1, request=core, context=ctx.model_copy(update=changes))
    with pytest.raises(ValueError, match="supplied loss"):
        context(scale_selections=(selection,))
    changed_core = DecisionRequest.model_validate(core.model_copy(update={"decision_id": "different"}))
    with pytest.raises(ValueError, match="producing request"):
        SemanticRequest(schema_version=1, request=changed_core, context=ctx)


def test_scale_member_coverage_rejects_foreign_partial_duplicate_and_mixed_choices():
    core, ctx = signed_context(ref("ghost"))
    with pytest.raises(ValueError, match="outside"):
        SemanticRequest(schema_version=1, request=core, context=ctx)
    core, ctx = signed_context(ref("a"))
    with pytest.raises(ValueError, match="cover"):
        SemanticRequest(schema_version=1, request=core, context=ctx)
    first = ctx.scale_selections[0]
    second = ScaleSelection.model_validate(first.model_copy(update={"member": ref("b")}))
    complete = SemanticContext.model_validate(ctx.model_copy(update={"scale_selections": (second, first)}))
    assert SemanticRequest(schema_version=1, request=core, context=complete).context.scale_selections
    alternate = ScaleSelection.model_validate(first.model_copy(update={"outcome": ref("another-outcome")}))
    with pytest.raises(ValueError, match="cover"):
        SemanticRequest(
            schema_version=1,
            request=core,
            context=ctx.model_copy(update={"scale_selections": (first, alternate, second)}),
        )
    global_choice = ScaleSelection.model_validate(first.model_copy(update={"member": None}))
    with pytest.raises(ValueError, match="global"):
        SemanticRequest(
            schema_version=1, request=core, context=ctx.model_copy(update={"scale_selections": (first, global_choice)})
        )
    history = second.scale.snapshot.model_copy(update={"source": ref("another-shared-history")})
    different_scale = fit_rolling_scale(second.scale.definition, history, frozen_at=second.scale.frozen_at)
    second = ScaleSelection.model_validate(second.model_copy(update={"scale": different_scale}))
    with pytest.raises(ValueError, match="same frozen scale"):
        SemanticRequest(
            schema_version=1, request=core, context=ctx.model_copy(update={"scale_selections": (first, second)})
        )


@pytest.mark.parametrize("owner", ["context", "objective", "transform", "constraint"])
def test_known_future_configuration_declarations_cannot_enter_timed_requests(owner):
    core, signed = signed_context()
    ctx = context()
    core = DecisionRequest.model_validate(core.model_copy(update={"objective": ctx.objective.reference}))
    assert core.timing is not None
    instant = core.timing.information_cutoff
    future = lineage().model_copy(update={"declared_at": instant + timedelta(seconds=1)})
    if owner == "context":
        ctx = ctx.model_copy(update={"lineage": future})
    elif owner == "objective":
        ctx = ctx.model_copy(update={"objective": ctx.objective.model_copy(update={"lineage": future})})
    elif owner == "constraint":
        constraint = ScalarConstraint(
            schema_version=1,
            reference=ref("cap"),
            lineage=future,
            quantity=ctx.objective.terms[0].quantity,
            lower=None,
            upper=3.0,
        )
        ctx = ctx.model_copy(update={"constraints": (constraint,)})
    else:
        transform = definition().model_copy(update={"lineage": future})
        ctx = context(transform)
    with pytest.raises(ValueError, match="declaration"):
        SemanticRequest(schema_version=1, request=core, context=ctx)
    accepted = context().model_copy(update={"lineage": lineage().model_copy(update={"declared_at": instant})})
    assert SemanticRequest(schema_version=1, request=core, context=accepted)
    assert SemanticRequest(schema_version=1, request=core, context=context())
    assert signed.scale_selections  # The independent timed fixture retains its scale.


def test_controls_bind_exact_members_and_reject_future_origin_declarations():
    controls = MixtureControls.from_json((FIXTURES / "mixture-controls-l1-v1.json").read_bytes())
    core, _ = signed_context()
    ctx = context(controls=controls, initial_prior=controls.initial_prior)
    with pytest.raises(ValueError, match="mixture action"):
        SemanticRequest(
            schema_version=1, request=core.model_copy(update={"objective": ctx.objective.reference}), context=ctx
        )
    space = MixtureSpace(
        feasible_set=ref("experts"),
        members=tuple(weight.member for weight in controls.initial_prior.weights),
        profile="binary64-simplex-v1",
    )
    core = DecisionRequest.model_validate(
        core.model_copy(update={"objective": ctx.objective.reference, "space": space})
    )
    assert SemanticRequest(schema_version=1, request=core, context=ctx)
    assert isinstance(controls.output, L1MovementPenalty)
    assert core.timing is not None
    instant = core.timing.information_cutoff
    future = lineage().model_copy(update={"declared_at": instant + timedelta(days=1)})
    origin = controls.output.origin.model_copy(update={"lineage": future})
    output = controls.output.model_copy(update={"origin": origin})
    conflicting = controls.model_copy(update={"output": output})
    with pytest.raises(ValueError, match="declaration"):
        SemanticRequest(schema_version=1, request=core, context=context(controls=conflicting))
    with pytest.raises(ValueError, match="conflicting initial"):
        context(
            controls=controls, initial_prior=controls.initial_prior.model_copy(update={"reference": ref("other-prior")})
        )
    foreign_space = MixtureSpace(feasible_set=ref("experts"), members=(ref("alien"),), profile="binary64-simplex-v1")
    with pytest.raises(ValueError, match="Initial prior"):
        SemanticRequest(schema_version=1, request=core.model_copy(update={"space": foreign_space}), context=ctx)


def test_known_change_declaration_cannot_follow_its_own_effective_instant():
    instant = datetime(2026, 1, 1, tzinfo=UTC)
    future = lineage().model_copy(update={"declared_at": instant + timedelta(seconds=1)})
    with pytest.raises(ValueError, match="declaration"):
        change(effective_at=instant, lineage=future)
    assert change(effective_at=instant, lineage=lineage())
    assert change(effective_at=instant, lineage=lineage().model_copy(update={"declared_at": instant}))
