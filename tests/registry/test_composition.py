"""Small public recipe origins retain actual actions and complete causal joins."""

import hashlib
import json
from datetime import timedelta
from itertools import permutations

import pytest
from pydantic import ValidationError

from coleman.core.actions import space_members
from coleman.core.decisions import ExecutedDecision
from coleman.core.feedback import FeedbackBatch, FeedbackRow, FullPanel, OutcomeValue
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.timing import DecisionTiming
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionBinding
from coleman.registry.composition import (
    RecipeExecution,
    RecipeFeedback,
    RegistryCheckpoint,
    RegistryOwner,
    require_recipe_eligibility,
)
from coleman.registry.definitions import Catalogue, MemberDefinition, RecipeBinding
from coleman.registry.envelopes import seal_request
from coleman.registry.lifecycle import LifecycleFact, RegistryState
from coleman.registry.limits import RegistryLimits
from coleman.registry.ownership import OwnershipDeclaration
from tests.registry.test_envelopes import capability, loss_spec, registered_request
from tests.registry.test_lifecycle import FIXTURES, START, ref


def checkpoint(*, retained=True, clock="untimed", **changes):
    values = json.loads((FIXTURES / "composition-checkpoint.json").read_text(encoding="utf-8"))
    if clock == "timed":
        registry = values["registries"][0]
        registry["policy"]["clock"] = "timed"
        registry["facts"][0]["known_at"] = START
        values["executions"] = []
    elif not retained:
        values["executions"] = []
    result = RegistryCheckpoint.model_validate(values | changes)
    if retained and clock == "timed":
        result = result.record(origin(result))
    return result


def origin(current, round_number=0, decision_id=None, *, registry=None, recipe=None, cutoff=None):
    selected = current.registries[0] if registry is None else registry
    binding = current.ownership.recipes[0] if recipe is None else recipe
    if selected.policy.clock == "timed" and cutoff is None:
        cutoff = START + timedelta(minutes=round_number)
    snapshot = selected.snapshot(round_number, cutoff)
    literal = json.loads((FIXTURES / "composition-checkpoint.json").read_text(encoding="utf-8"))
    values = literal["executions"][0]["executed"]
    request = values["proposal"]["request"]
    request.update(round=round_number, decision_id=decision_id or f"opportunity-{round_number}")
    if cutoff is not None:
        request["timing"] = DecisionTiming(
            decision_at=START + timedelta(days=1),
            information_cutoff=cutoff,
            target_start=START + timedelta(days=1),
            target_end=START + timedelta(days=2),
            eligible_input_fingerprint="sha256:" + "e" * 64,
            mode="future",
        )
    return RecipeExecution(
        schema_version=1, snapshot=snapshot, recipe=binding, executed=ExecutedDecision.model_validate(values)
    )


def extended_registry(current, *, introduce=False, admitted=10, kind="expert", name="unrelated-recipe"):
    registry = current.registries[0]
    leaf = MemberDefinition.model_validate(
        registry.catalogue.definitions[0].model_dump()
        | {
            "kind": kind,
            "member": ref(name),
        }
    )
    grown = RegistryState.model_validate(registry.model_dump() | {"catalogue": registry.catalogue.register(leaf)})
    if introduce:
        grown = grown.append(
            LifecycleFact(
                schema_version=1,
                event_id="prospective-introduction",
                sequence=len(grown.facts),
                member=leaf.member,
                definition_fingerprint=leaf.fingerprint,
                operation="introduce",
                effective_from_round=10,
                admitted_before_round=admitted,
                known_at=START if grown.policy.clock == "timed" else None,
                reason="public-prospective-recipe",
                state_epoch=ref(name + ":e0"),
                predecessor_epoch=None,
                disposition=None,
            )
        )
    return grown


def feedback(producer, values=(0.25, 0.75)):
    rows = tuple(
        FeedbackRow(
            member=member,
            values=(
                OutcomeValue(
                    spec=loss_spec(),
                    value=value,
                    status="unavailable" if value is None else "observed",
                    origin="synthetic",
                    transform=ref("identity-loss"),
                ),
            ),
        )
        for member, value in zip(producer.executed.action.space.members, values, strict=True)
    )
    batch = FeedbackBatch(
        schema_version=1,
        decision_fingerprint=producer.executed.fingerprint,
        feedback_id="recipe-feedback",
        revision_id="initial",
        supersedes=None,
        horizon=None,
        feedback=FullPanel(rows=rows),
    )
    return RecipeFeedback(schema_version=1, producer_fingerprint=producer.fingerprint, batch=batch)


def changed_inner(producer, *, request=None, **proposal):
    values = producer.executed.model_dump()
    if request:
        values["proposal"]["request"].update(request)
    values["proposal"].update(proposal)
    return ExecutedDecision.model_validate(values)


def test_literal_checkpoint_and_producer_keep_independently_authored_core_action_bytes():
    literal = (FIXTURES / "composition-checkpoint.json").read_text(encoding="utf-8").rstrip("\n")
    oracle = json.loads((FIXTURES / "composition-oracle.json").read_text(encoding="utf-8"))
    current = RegistryCheckpoint.from_json(literal)
    assert current.to_json() == literal
    assert (
        current.fingerprint
        == oracle["checkpoint_fingerprint"]
        == ("sha256:" + hashlib.sha256(literal.encode("utf-8")).hexdigest())
    )
    producer = current.executions[0]
    original_inner = json.loads((FIXTURES / "execution.json").read_text(encoding="utf-8"))["executed"]
    assert producer.executed.to_json() == ExecutedDecision.model_validate(original_inner).to_json()
    assert [member.id for member in producer.snapshot.active] == oracle["active_expert_ids"] == ["recipe-A"]
    assert (
        [member.id for member in space_members(producer.executed.action.space)]
        == oracle["inner_action_ids"]
        == ["A", "B"]
    )
    assert producer.executed.fingerprint == oracle["inner_execution_fingerprint"]
    assert producer.fingerprint == oracle["producer_fingerprint"]
    assert producer.logical_identity == producer.executed.logical_identity
    assert RecipeExecution.from_json(producer.to_json()) == producer
    assert current.for_execution(producer.executed.fingerprint) == producer
    with pytest.raises(ValueError, match="Unknown original"):
        current.for_execution("sha256:" + "f" * 64)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("task", ref("wrong-task"), "expert-population task"),
        (
            "expert",
            DefinitionBinding(reference=ref("unknown-expert"), definition_fingerprint="sha256:" + "f" * 64),
            "registered expert version",
        ),
        (
            "expert",
            DefinitionBinding(reference=ref("recipe-A", "v2"), definition_fingerprint="sha256:" + "f" * 64),
            "registered expert version",
        ),
        (
            "expert",
            DefinitionBinding(reference=ref("recipe-A"), definition_fingerprint="sha256:" + "f" * 64),
            "active expert definition and epoch",
        ),
        ("state_epoch", ref("replacement-epoch"), "active expert definition and epoch"),
        ("objective", ref("wrong-objective"), "declared objective"),
    ],
)
def test_producer_rejects_exact_recipe_task_version_definition_epoch_and_objective_mismatch(field, value, message):
    producer = checkpoint().executions[0]
    recipe = RecipeBinding.model_validate(producer.recipe.model_dump() | {field: value})
    candidate_values = producer.model_dump() | {"recipe": recipe}
    with pytest.raises(ValueError, match=message):
        RecipeExecution.model_validate(candidate_values)


@pytest.mark.parametrize(
    "request_changes,proposal,message",
    [
        ({"task": ref("wrong-child")}, {}, "producing task or round"),
        ({"round": 1}, {}, "producing task or round"),
        ({}, {"implementation": ref("other-implementation")}, "implementation reference"),
    ],
)
def test_producer_rejects_inner_task_round_and_implementation_without_changing_action_bytes(
    request_changes, proposal, message
):
    producer = checkpoint().executions[0]
    inner = changed_inner(producer, request=request_changes, **proposal)
    assert inner.action.to_json() == producer.executed.action.to_json()
    candidate_values = producer.model_dump() | {"executed": inner}
    with pytest.raises(ValueError, match=message):
        RecipeExecution.model_validate(candidate_values)


def test_producer_rejects_wrong_clock_sleeping_expert_and_action_population():
    current = checkpoint(retained=False)
    untimed = origin(current)
    timed = checkpoint(retained=False, clock="timed")
    candidate_values = untimed.model_dump() | {"snapshot": timed.registries[0].snapshot(0, START)}
    with pytest.raises(ValueError, match="clock profile or cutoff"):
        RecipeExecution.model_validate(candidate_values)
    state = current.registries[0]
    fact = LifecycleFact.model_validate(
        state.facts[0].model_dump()
        | {
            "event_id": "sleep",
            "sequence": 1,
            "operation": "inactive",
            "effective_from_round": 1,
            "admitted_before_round": 1,
            "state_epoch": None,
        }
    )
    sleeping = state.append(fact)
    with pytest.raises(ValueError, match="active expert definition and epoch"):
        origin(current, 1, registry=sleeping)
    values = state.model_dump()
    action = MemberDefinition.model_validate(state.catalogue.definitions[0].model_dump() | {"kind": "action"})
    values["catalogue"] = Catalogue(schema_version=1, reference=state.catalogue.reference, definitions=(action,))
    values["policy"]["kind"] = "action"
    values["facts"][0]["definition_fingerprint"] = action.fingerprint
    action_state = RegistryState.model_validate(values)
    candidate_values = untimed.model_dump() | {"snapshot": action_state.snapshot(0)}
    with pytest.raises(ValueError, match="expert-population task"):
        RecipeExecution.model_validate(candidate_values)


def test_recipe_feedback_joins_outer_and_inner_producer_and_actual_active_action_panel():
    current = checkpoint()
    producer = current.executions[0]
    observed = feedback(producer)
    observed.validate_for(producer, capability())
    assert RecipeFeedback.from_json(observed.to_json()) == observed
    later = origin(current, 1)
    selected_consumer = capability()
    with pytest.raises(ValueError, match="original outer producer"):
        observed.validate_for(later, selected_consumer)
    wrong_inner = RecipeFeedback.model_validate(
        observed.model_dump()
        | {
            "batch": observed.batch.model_dump() | {"decision_fingerprint": later.executed.fingerprint},
        }
    )
    selected_consumer = capability()
    with pytest.raises(ValueError, match="executed decision content"):
        wrong_inner.validate_for(producer, selected_consumer)
    unavailable = feedback(producer, (0.25, None))
    assert isinstance(unavailable.batch.feedback, FullPanel)
    assert unavailable.batch.feedback.rows[1].values[0].value is None
    selected_consumer = capability()
    with pytest.raises(ValueError):
        unavailable.validate_for(producer, selected_consumer)
    unavailable.validate_for(producer, capability(requires_observed=False))
    assert isinstance(observed.batch.feedback, FullPanel)
    incomplete = RecipeFeedback.model_validate(
        observed.model_dump()
        | {
            "batch": observed.batch.model_dump() | {"feedback": FullPanel(rows=(observed.batch.feedback.rows[0],))},
        }
    )
    selected_consumer = capability()
    with pytest.raises(ValueError, match="Full panel"):
        incomplete.validate_for(producer, selected_consumer)


def test_recipe_feedback_after_explicit_restart_still_belongs_to_the_old_epoch():
    current = checkpoint()
    old = current.executions[0]
    registry = current.registries[0]
    retired = registry.append(
        LifecycleFact.model_validate(
            registry.facts[0].model_dump()
            | {
                "event_id": "retire-old-epoch",
                "sequence": 1,
                "operation": "retire",
                "effective_from_round": 1,
                "admitted_before_round": 1,
                "state_epoch": None,
            }
        )
    )
    restarted = retired.append(
        LifecycleFact.model_validate(
            registry.facts[0].model_dump()
            | {
                "event_id": "restart-new-epoch",
                "sequence": 2,
                "operation": "reenter",
                "effective_from_round": 2,
                "admitted_before_round": 2,
                "state_epoch": ref("recipe-A:e1"),
                "predecessor_epoch": old.recipe.state_epoch,
                "disposition": "restart",
            }
        )
    )
    recipe = RecipeBinding.model_validate(old.recipe.model_dump() | {"state_epoch": ref("recipe-A:e1")})
    replacement = origin(current, 2, registry=restarted, recipe=recipe)
    delayed = feedback(old)
    delayed.validate_for(old, capability())
    selected_consumer = capability()
    with pytest.raises(ValueError, match="original outer producer"):
        delayed.validate_for(replacement, selected_consumer)
    assert old.recipe.state_epoch == ref("recipe-A:e0")
    assert replacement.recipe.state_epoch == ref("recipe-A:e1")


def test_record_exact_retry_conflict_stale_prefix_round_and_capacity_leave_prior_state_unchanged():
    current = checkpoint()
    original = current.to_json()
    producer = current.executions[0]
    assert current.record(RecipeExecution.from_json(producer.to_json())) == current
    candidate_origin = origin(current, 1, "opportunity-0")
    with pytest.raises(ValueError, match="Conflicting original"):
        current.record(candidate_origin)
    candidate_origin = origin(current, 0, "another-same-round")
    with pytest.raises(ValueError, match="opportunity frontier"):
        current.record(candidate_origin)
    advanced = current.record(origin(current, 2))
    candidate_origin = origin(advanced, 1)
    with pytest.raises(ValueError, match="opportunity frontier"):
        advanced.record(candidate_origin)
    grown = current.replace_registry(extended_registry(current, introduce=True))
    candidate_origin = origin(current, 1)
    with pytest.raises(ValueError, match="actual current predecision"):
        grown.record(candidate_origin)
    limited = RegistryCheckpoint.model_validate(current.model_dump() | {"limits": RegistryLimits(maximum_producers=1)})
    assert limited.record(producer) == limited
    candidate_origin = origin(limited, 1)
    with pytest.raises(ValueError, match="producer capacity"):
        limited.record(candidate_origin)
    assert current.to_json() == original


def test_historical_publications_resolve_exactly_after_unrelated_growth_and_keep_original_retry():
    current = checkpoint()
    original = current.executions[0]
    grown = current.replace_registry(extended_registry(current))
    assert len(grown.catalogues) == 2
    assert grown.record(original) == grown
    assert grown.for_execution(original.executed.fingerprint) == original
    assert original.snapshot.catalogue.definition_fingerprint != grown.registries[0].catalogue.fingerprint
    later = grown.record(origin(grown, 1))
    assert later.executions[0].recipe == later.executions[1].recipe
    assert RegistryCheckpoint.from_json(later.to_json()) == later
    candidate_values = grown.model_dump() | {"catalogues": (grown.registries[0].catalogue,)}
    with pytest.raises(ValueError, match="exact retained catalogue publication"):
        RegistryCheckpoint.model_validate(candidate_values)
    wrong = Catalogue.model_validate(grown.catalogues[0].model_dump() | {"reference": ref("other-namespace")})
    candidate_values = grown.model_dump() | {"catalogues": (*grown.catalogues, wrong)}
    with pytest.raises(ValueError, match="unknown catalogue namespace/version"):
        RegistryCheckpoint.model_validate(candidate_values)
    changed_snapshot = original.snapshot.model_validate(
        original.snapshot.model_dump()
        | {
            "catalogue": {
                "reference": original.snapshot.catalogue.reference,
                "definition_fingerprint": "sha256:" + "f" * 64,
            },
        }
    )
    wrong_origin = RecipeExecution.model_validate(original.model_dump() | {"snapshot": changed_snapshot})
    candidate_values = current.model_dump() | {"executions": (wrong_origin,)}
    with pytest.raises(ValueError, match="exact retained catalogue publication"):
        RegistryCheckpoint.model_validate(candidate_values)
    unselected = grown.registries[0].catalogue.resolve("expert", ref("unrelated-recipe"))
    wrong_publication = Catalogue(
        schema_version=1, reference=current.registries[0].catalogue.reference, definitions=(unselected,)
    )
    snapshot = original.snapshot.model_validate(
        original.snapshot.model_dump()
        | {
            "catalogue": DefinitionBinding(
                reference=wrong_publication.reference, definition_fingerprint=wrong_publication.fingerprint
            ),
        }
    )
    wrong_origin = RecipeExecution.model_validate(original.model_dump() | {"snapshot": snapshot})
    candidate_values = grown.model_dump() | {
        "catalogues": (*grown.catalogues, wrong_publication),
        "executions": (wrong_origin,),
    }
    with pytest.raises(ValueError, match="Unknown catalogue member"):
        RegistryCheckpoint.model_validate(candidate_values)


def test_checkpoint_reader_rejects_retrospective_suffix_prefix_rewind_clock_rewind_and_same_round():
    current = checkpoint()
    invalid = extended_registry(current, introduce=True, admitted=0)
    candidate_values = current.model_dump() | {"registries": (invalid,)}
    with pytest.raises(ValueError, match="already-produced recipe round"):
        RegistryCheckpoint.model_validate(candidate_values)
    future = extended_registry(current, introduce=True)
    first = origin(current, 0, registry=future)
    second = origin(current, 1)
    candidate_values = current.model_dump() | {"registries": (future,), "executions": (first, second)}
    with pytest.raises(ValueError, match="history prefixes cannot rewind"):
        RegistryCheckpoint.model_validate(candidate_values)
    timed = checkpoint(retained=False, clock="timed")
    for second_round, second_cutoff, message in (
        (1, START, "knowledge cutoffs cannot rewind"),
        (0, START + timedelta(minutes=2), "distinct increasing rounds"),
    ):
        first = origin(timed, 0, "timed-first", cutoff=START + timedelta(minutes=1))
        second = origin(timed, second_round, "timed-second", cutoff=second_cutoff)
        candidate_values = timed.model_dump() | {"executions": (first, second)}
        with pytest.raises(ValueError, match=message):
            RegistryCheckpoint.model_validate(candidate_values)


@pytest.mark.parametrize("field", ["target", "loss_transform", "controls", "owners"])
def test_checkpoint_freezes_complete_recipe_binding_for_one_child_epoch(field):
    current = checkpoint()
    recipe = current.ownership.recipes[0]
    if field == "owners":
        value = (recipe.owners[0].model_validate(recipe.owners[0].model_dump() | {"state_codec": ref("stale-codec")}),)
    else:
        value = DefinitionBinding(reference=ref("changed-" + field), definition_fingerprint=fingerprint([field, "v2"]))
    changed = RecipeBinding.model_validate(recipe.model_dump() | {field: value})
    producer = origin(current, 1, recipe=changed)
    with pytest.raises(ValueError, match="frozen complete recipe binding"):
        current.record(producer)


def test_registry_replacement_preserves_old_leaves_facts_policy_limits_and_catalogue_namespace():
    current = checkpoint(retained=False)
    state = current.registries[0]
    for values, message in (
        ({"facts": ()}, "accepted lifecycle facts"),
        (
            {"policy": state.policy.model_dump() | {"population": ref("other-population")}},
            "fixed policy or selected limits",
        ),
        ({"limits": RegistryLimits(maximum_producers=12)}, "fixed policy or selected limits"),
        ({"catalogue": state.catalogue.model_dump() | {"reference": ref("other-catalogue")}}, "catalogue namespace"),
    ):
        replacement = RegistryState.model_validate(state.model_dump() | values)
        with pytest.raises(ValueError, match=message):
            current.replace_registry(replacement)
    grown = current.replace_registry(extended_registry(current, kind="action", name="recipe-A"))
    registry = grown.registries[0]
    assert registry.catalogue.resolve("action", ref("recipe-A")).member == ref("recipe-A")
    trimmed = RegistryState.model_validate(registry.model_dump() | {"catalogue": state.catalogue})
    with pytest.raises(ValueError, match="Unknown catalogue"):
        grown.replace_registry(trimmed)


def test_owned_recipe_checkpoint_explicitly_rejects_mixing_the_standalone_request_seal_profile():
    current = checkpoint(retained=False)
    state = current.registries[0]
    sealed = seal_request(state, registered_request(state))
    assert sealed.producers
    assert current.profile == "owned-recipes-v1"
    candidate_values = current.model_dump() | {"registries": (sealed,)}
    with pytest.raises(ValueError, match="cannot mix standalone registered-request seals"):
        RegistryCheckpoint.model_validate(candidate_values)
    with pytest.raises(ValueError, match="cannot mix standalone registered-request seals"):
        current.replace_registry(sealed)
    candidate_values = current.model_dump() | {"profile": "future-profile-v2"}
    with pytest.raises(ValueError):
        RegistryCheckpoint.model_validate(candidate_values)


def test_registry_owner_restore_is_monotone_for_origins_publications_and_fixed_ownership():
    current = checkpoint()
    owner = RegistryOwner(current)
    owner.record(origin(current, 1))
    advanced = owner.export_state()
    assert len(advanced.executions) == 2
    owner.restore_state(advanced)
    with pytest.raises(ValueError, match="retained original producers"):
        owner.restore_state(current)
    assert owner.export_state() == advanced
    empty = checkpoint(retained=False)
    grown = empty.replace_registry(extended_registry(empty))
    owner = RegistryOwner(grown)
    lost = RegistryCheckpoint.model_validate(grown.model_dump() | {"catalogues": (grown.registries[0].catalogue,)})
    with pytest.raises(ValueError, match="retained catalogue publications"):
        owner.restore_state(lost)
    recipe = empty.ownership.recipes[0]
    changed = RecipeBinding.model_validate(
        recipe.model_dump()
        | {
            "controls": DefinitionBinding(
                reference=ref("new-controls"),
                definition_fingerprint=fingerprint(["new-controls"]),
            )
        }
    )
    ownership = OwnershipDeclaration.model_validate(empty.ownership.model_dump() | {"recipes": (changed,)})
    alternate = RegistryCheckpoint.model_validate(empty.model_dump() | {"ownership": ownership})
    restoring_owner = RegistryOwner(empty)
    with pytest.raises(ValueError, match="fixed ownership or selected limits"):
        restoring_owner.restore_state(alternate)


def test_checkpoint_admits_raw_counts_and_coupled_work_before_nested_invalid_producers():
    current = checkpoint(retained=False)
    for values, message in (
        ({"executions": ({}, {}), "limits": RegistryLimits(maximum_producers=1)}, "finite registry/producer counts"),
        ({"catalogues": ({}, {}), "limits": RegistryLimits(maximum_publications=1)}, "publication capacity"),
        (
            {"executions": ({}, {}), "limits": RegistryLimits(maximum_checkpoint_work=1)},
            "history validation work capacity",
        ),
    ):
        candidate_values = current.model_dump() | values
        with pytest.raises(ValidationError, match=message) as caught:
            RegistryCheckpoint.model_validate(candidate_values)
        assert len(caught.value.errors()) == 1
        assert caught.value.errors()[0]["loc"] == ()


def test_normalized_publication_work_has_constructor_writer_reader_parity_at_the_tight_cap():
    current = checkpoint(retained=False)
    publication = current.registries[0].catalogue
    for publications in ((), (publication,)):
        minimal = RegistryCheckpoint.model_validate(
            current.model_dump()
            | {
                "catalogues": publications,
                "limits": RegistryLimits(maximum_checkpoint_work=1),
            }
        )
        assert minimal.catalogues == (publication,)
        assert RegistryCheckpoint.from_json(minimal.to_json()) == minimal
        assert RegistryCheckpoint.model_validate_json(minimal.to_json()) == minimal
    candidate_values = current.model_dump() | {
        "executions": (origin(current),),
        "catalogues": (),
        "limits": RegistryLimits(maximum_checkpoint_work=1),
    }
    with pytest.raises(ValueError, match="normalized coupled validation work"):
        RegistryCheckpoint.model_validate(candidate_values)
    candidate_values = current.model_dump() | {"catalogues": (publication, publication)}
    with pytest.raises(ValueError, match="duplicate explicit catalogue publications"):
        RegistryCheckpoint.model_validate(candidate_values)


def test_checkpoint_resource_publication_and_work_caps_are_selected_separately_without_eviction():
    current = checkpoint(retained=False)
    claim = current.ownership.resources[0]
    weights = claim.model_validate(claim.model_dump() | {"resource": ref("extra-weights"), "role": "weights"})
    ownership = OwnershipDeclaration.model_validate(
        current.ownership.model_dump()
        | {
            "resources": (*current.ownership.resources, weights),
        }
    )
    candidate_values = current.model_dump() | {
        "ownership": ownership,
        "limits": RegistryLimits(maximum_resources=1),
    }
    with pytest.raises(ValueError, match="ownership resource capacity"):
        RegistryCheckpoint.model_validate(candidate_values)
    ownership_values = current.ownership.model_dump() | {"resources": ({}, {})}
    candidate_values = current.model_dump() | {
        "ownership": ownership_values,
        "limits": RegistryLimits(maximum_resources=1),
    }
    with pytest.raises(ValidationError, match="ownership resource capacity") as caught:
        RegistryCheckpoint.model_validate(candidate_values)
    assert caught.value.errors()[0]["loc"] == ()
    admitted = RegistryCheckpoint.model_validate(
        current.model_dump()
        | {
            "ownership": ownership,
            "limits": RegistryLimits(maximum_resources=2),
        }
    )
    assert len(admitted.ownership.resources) == 2
    limited = RegistryCheckpoint.model_validate(
        current.model_dump() | {"limits": RegistryLimits(maximum_publications=1)}
    )
    before = limited.to_json()
    candidate_registry = extended_registry(limited)
    with pytest.raises(ValueError, match="publication capacity"):
        limited.replace_registry(candidate_registry)
    assert limited.to_json() == before
    grown = extended_registry(current)
    one_publication = RegistryCheckpoint.model_validate(
        current.model_dump()
        | {
            "registries": (grown,),
            "catalogues": (),
            "limits": RegistryLimits(maximum_publications=1, maximum_checkpoint_work=2),
        }
    )
    assert len(one_publication.catalogues) == 1
    assert len(one_publication.catalogues[0].definitions) == 2
    one_origin = RegistryCheckpoint.model_validate(
        checkpoint().model_dump()
        | {
            "limits": RegistryLimits(maximum_prefix_work=1, maximum_checkpoint_work=2),
        }
    )
    assert RegistryCheckpoint.from_json(one_origin.to_json()) == one_origin
    assert one_origin.limits.maximum_prefix_work == 1


def test_owner_restore_rejects_replacing_a_retained_origin_with_other_valid_actual_action_bytes():
    current = checkpoint()
    producer = current.executions[0]
    values = producer.executed.model_dump()
    values["action"]["chosen"] = ref("B")
    values["proposal"]["action"]["chosen"] = ref("B")
    actual = producer.executed.action.model_validate(values["action"])
    values["sampling"]["action_fingerprint"] = actual.fingerprint
    values["proposal"]["sampling"]["action_fingerprint"] = actual.fingerprint
    alternative = RecipeExecution.model_validate(
        producer.model_dump()
        | {
            "executed": ExecutedDecision.model_validate(values),
        }
    )
    assert alternative.logical_identity == producer.logical_identity
    assert alternative.executed.fingerprint != producer.executed.fingerprint
    candidate = RegistryCheckpoint.model_validate(current.model_dump() | {"executions": (alternative,)})
    owner = RegistryOwner(current)
    with pytest.raises(ValueError, match="retained original producers"):
        owner.restore_state(candidate)
    assert owner.export_state() == current


def test_owner_restore_preserves_exact_retained_origin_bytes_when_valid_scalar_signed_zero_changes():
    current = checkpoint()
    producer = current.executions[0]
    spec = ScalarSpec(
        dimension=ref("context"),
        semantics=UnitSemantics(unit="1", meaning="public-context"),
        role="feature",
        minimum=None,
        maximum=None,
    )
    candidates = []
    for value in (0.0, -0.0):
        feature = Scalar(spec=spec, value=value, status="observed")
        inner = changed_inner(producer, request={"inputs": (feature,)})
        changed = RecipeExecution.model_validate(producer.model_dump() | {"executed": inner})
        candidates.append(RegistryCheckpoint.model_validate(current.model_dump() | {"executions": (changed,)}))
    positive, negative = candidates
    old, replacement = positive.executions[0], negative.executions[0]
    assert old.logical_identity == replacement.logical_identity
    assert old.executed.action.to_json() == replacement.executed.action.to_json()
    assert old.executed.sampling == replacement.executed.sampling
    assert old.executed.proposal.sampling == replacement.executed.proposal.sampling
    assert old.to_json() != replacement.to_json()
    assert old.fingerprint != replacement.fingerprint
    assert old.executed.fingerprint != replacement.executed.fingerprint
    assert RegistryCheckpoint.from_json(positive.to_json()).to_json() == positive.to_json()
    assert RegistryCheckpoint.from_json(negative.to_json()).to_json() == negative.to_json()
    owner = RegistryOwner(positive)
    before = owner.export_state().to_json()
    with pytest.raises(ValueError, match="retained original producers"):
        owner.restore_state(negative)
    assert owner.export_state().to_json() == before


def test_public_recipe_eligibility_preflight_allows_active_before_host_preparation():
    current = checkpoint(retained=False)
    producer = origin(current)
    calls = []
    require_recipe_eligibility(producer.snapshot, producer.recipe, producer.executed.proposal.request)
    calls.extend(("prepare", "emit"))
    assert calls == ["prepare", "emit"]
    assert producer.snapshot.active == (ref("recipe-A"),)
    assert space_members(producer.executed.action.space) == (ref("A"), ref("B"))


@pytest.mark.parametrize("operation", ["inactive", "unavailable", "retire"])
def test_public_recipe_eligibility_preflight_rejects_sleeping_and_retired_before_any_host_call(operation):
    current = checkpoint(retained=False)
    active = origin(current)
    registry = current.registries[0]
    event = LifecycleFact.model_validate(
        registry.facts[0].model_dump()
        | {
            "event_id": "predecision-" + operation,
            "sequence": 1,
            "operation": operation,
            "effective_from_round": 1,
            "admitted_before_round": 1,
            "state_epoch": None,
        }
    )
    sleeping = registry.append(event).snapshot(1)
    request = active.executed.proposal.request.model_validate(
        active.executed.proposal.request.model_dump()
        | {
            "round": 1,
            "decision_id": "predecision-1",
        }
    )
    before = sleeping.to_json()
    calls = []

    def prepare_and_emit():
        require_recipe_eligibility(sleeping, active.recipe, request)
        calls.extend(("prepare", "emit"))

    with pytest.raises(ValueError, match="active expert definition and epoch"):
        prepare_and_emit()
    assert not calls
    assert not sleeping.active
    assert sleeping.to_json() == before


def test_current_catalogue_namespaces_reject_conflicting_complete_publications_and_accept_distinct_namespaces():
    current = checkpoint(retained=False)
    registry = current.registries[0]
    leaf = MemberDefinition.model_validate(registry.catalogue.definitions[0].model_dump() | {"member": ref("recipe-B")})
    catalogue = Catalogue(schema_version=1, reference=registry.catalogue.reference, definitions=(leaf,))
    state = RegistryState.model_validate(
        registry.model_dump()
        | {
            "catalogue": catalogue,
            "policy": registry.policy.model_dump() | {"task": ref("other-task"), "population": ref("other-population")},
            "facts": (
                LifecycleFact.model_validate(
                    registry.facts[0].model_dump()
                    | {
                        "member": leaf.member,
                        "definition_fingerprint": leaf.fingerprint,
                        "state_epoch": ref("recipe-B:e0"),
                    }
                ),
            ),
        }
    )
    original = current.ownership.recipes[0]
    owner = original.owners[0].model_validate(original.owners[0].model_dump() | {"owner": ref("child-B-state")})
    recipe = RecipeBinding.model_validate(
        original.model_dump()
        | {
            "child": ref("child-B"),
            "task": state.policy.task,
            "expert": leaf.binding,
            "state_epoch": ref("recipe-B:e0"),
            "owners": (owner,),
        }
    )
    claim = current.ownership.resources[0].model_validate(
        current.ownership.resources[0].model_dump()
        | {
            "resource": ref("child-B-owner"),
            "owner": owner,
            "child": recipe.child,
        }
    )
    ownership = OwnershipDeclaration.model_validate(
        current.ownership.model_dump()
        | {
            "recipes": (*current.ownership.recipes, recipe),
            "resources": (*current.ownership.resources, claim),
        }
    )
    candidate_values = current.model_dump() | {
        "registries": (registry, state),
        "ownership": ownership,
        "catalogues": (),
    }
    with pytest.raises(ValueError, match="conflicting complete publications"):
        RegistryCheckpoint.model_validate(candidate_values)
    separate = RegistryState.model_validate(
        state.model_dump()
        | {
            "catalogue": Catalogue(schema_version=1, reference=ref("other-catalogue"), definitions=(leaf,)),
        }
    )
    result = RegistryCheckpoint.model_validate(
        current.model_dump()
        | {
            "registries": (registry, separate),
            "ownership": ownership,
            "catalogues": (),
        }
    )
    permuted = RegistryCheckpoint.model_validate(result.model_dump() | {"registries": (separate, registry)})
    assert permuted.to_json() == result.to_json()


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_composition_readers_reject_missing_future_coerced_versions_and_unknown_fields(version):
    current = checkpoint()
    for contract, sample in (
        (RegistryCheckpoint, current),
        (RecipeExecution, current.executions[0]),
        (RecipeFeedback, feedback(current.executions[0])),
    ):
        values = sample.model_dump() | {"schema_version": version}
        with pytest.raises(ValueError):
            contract.model_validate(values)
        values.pop("schema_version")
        with pytest.raises(ValueError):
            contract.model_validate(values)
        candidate_values = sample.model_dump() | {"untyped_metadata": {"hidden": "value"}}
        with pytest.raises(ValueError):
            contract.model_validate(candidate_values)


def test_publication_and_execution_declaration_permutations_have_one_canonical_identity():
    current = checkpoint()
    current = current.replace_registry(extended_registry(current))
    current = current.record(origin(current, 1))
    for publications in permutations(current.catalogues):
        for executions in permutations(current.executions):
            changed = RegistryCheckpoint.model_validate(
                current.model_dump()
                | {
                    "catalogues": publications,
                    "executions": executions,
                }
            )
            assert changed.to_json() == current.to_json()
