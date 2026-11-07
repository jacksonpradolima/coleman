"""Public registry refusal boundaries reached with otherwise valid small inputs."""

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from coleman.core.actions import (
    ChoiceAction,
    ChoiceSpace,
    RankingAction,
    RankingSpace,
    VectorSpace,
)
from coleman.core.decisions import Decision, ExecutedDecision
from coleman.core.sampling import SamplingRecord
from coleman.core.timing import DecisionTiming
from coleman.registry.composition import RecipeExecution, RegistryCheckpoint, RegistryOwner
from coleman.registry.definitions import Catalogue, MemberDefinition, RecipeBinding
from coleman.registry.envelopes import RegisteredExecution, RegistryRequest, seal_request
from coleman.registry.lifecycle import LifecycleFact, RegistryState
from coleman.registry.limits import RegistryLimits
from coleman.registry.ownership import OwnershipDeclaration
from tests.registry.test_composition import changed_inner, checkpoint, extended_registry, origin
from tests.registry.test_envelopes import execution, registered_request
from tests.registry.test_lifecycle import START, fact, history, ref


def _ownership(current, recipe):
    return OwnershipDeclaration.model_validate(current.ownership.model_dump() | {"recipes": (recipe,)})


def _foreign_origin(current):
    previous = origin(current)
    state = current.registries[0]
    foreign = ref("other-task")
    registry = RegistryState.model_validate(
        state.model_dump()
        | {
            "policy": state.policy.model_dump() | {"task": foreign},
        }
    )
    recipe = RecipeBinding.model_validate(previous.recipe.model_dump() | {"task": foreign})
    producer = RecipeExecution(
        schema_version=1,
        snapshot=registry.snapshot(0),
        recipe=recipe,
        executed=changed_inner(previous, request={"task": foreign}),
    )
    return registry, producer


def _unchanged_execution(request, action):
    sampling = SamplingRecord(
        mechanism="deterministic",
        mechanism_ref=ref("public-rejection-host"),
        status="not_required",
        action_fingerprint=action.fingerprint,
        probability=None,
    )
    proposed = Decision(
        schema_version=1,
        request=request,
        implementation=ref("public-rejection-host"),
        action=action,
        sampling=sampling,
        state_owner="host",
    )
    return ExecutedDecision(
        schema_version=1,
        proposal=proposed,
        action=action,
        mapping="unchanged",
        sampling_disposition="preserved",
        sampling=sampling,
    )


def test_checkpoint_task_inventory_rejects_duplicate_missing_and_wrong_version_matches():
    current = checkpoint(retained=False)
    registry = current.registries[0]
    recipe = current.ownership.recipes[0]
    before = current.to_json()
    candidate_values = current.model_dump() | {"registries": (registry, registry)}
    with pytest.raises(ValueError, match="unique logical identities"):
        RegistryCheckpoint.model_validate(candidate_values)
    wrong_task = RecipeBinding.model_validate(recipe.model_dump() | {"task": ref("other-task")})
    candidate_values = current.model_dump() | {"ownership": _ownership(current, wrong_task)}
    with pytest.raises(ValueError, match="exactly one registry per declared recipe task"):
        RegistryCheckpoint.model_validate(candidate_values)
    wrong_version = RecipeBinding.model_validate(recipe.model_dump() | {"task": ref(recipe.task.id, "v2")})
    candidate_values = current.model_dump() | {"ownership": _ownership(current, wrong_version)}
    with pytest.raises(ValueError, match="recipe and expert-population task differ"):
        RegistryCheckpoint.model_validate(candidate_values)
    assert current.to_json() == before


def test_checkpoint_task_inventory_rejects_two_valid_children_declaring_one_task():
    current = checkpoint(retained=False)
    recipe = current.ownership.recipes[0]
    owner = recipe.owners[0].model_validate(recipe.owners[0].model_dump() | {"owner": ref("second-owner")})
    second = RecipeBinding.model_validate(recipe.model_dump() | {"child": ref("second-child"), "owners": (owner,)})
    claim = current.ownership.resources[0].model_validate(
        current.ownership.resources[0].model_dump()
        | {
            "resource": ref("second-owner-root"),
            "owner": owner,
            "child": second.child,
        }
    )
    declaration = OwnershipDeclaration.model_validate(
        current.ownership.model_dump()
        | {
            "recipes": (recipe, second),
            "resources": (*current.ownership.resources, claim),
        }
    )
    assert len(declaration.recipes) == 2
    candidate_values = current.model_dump() | {"ownership": declaration}
    with pytest.raises(ValueError, match="exactly one registry per declared recipe task"):
        RegistryCheckpoint.model_validate(candidate_values)


def test_checkpoint_rejects_known_expert_with_conflicting_binding_and_unknown_producer_task():
    current = checkpoint(retained=False)
    recipe = current.ownership.recipes[0]
    changed = RecipeBinding.model_validate(
        recipe.model_dump()
        | {
            "expert": recipe.expert.model_dump() | {"definition_fingerprint": "sha256:" + "f" * 64},
        }
    )
    candidate_values = current.model_dump() | {"ownership": _ownership(current, changed)}
    with pytest.raises(ValueError, match="exact immutable expert definition"):
        RegistryCheckpoint.model_validate(candidate_values)
    foreign_registry, foreign = _foreign_origin(current)
    assert RecipeExecution.from_json(foreign.to_json()) == foreign
    candidate_values = current.model_dump() | {"executions": (foreign,)}
    with pytest.raises(ValueError, match="frozen complete recipe binding"):
        RegistryCheckpoint.model_validate(candidate_values)
    before = current.to_json()
    with pytest.raises(ValueError, match="Unknown checkpoint registry task"):
        current.record(foreign)
    with pytest.raises(ValueError, match="Unknown checkpoint registry task"):
        current.replace_registry(foreign_registry)
    assert current.to_json() == before


def test_checkpoint_reader_rejects_duplicate_origins_and_conflicting_current_recipe_namespace():
    current = checkpoint()
    first = current.executions[0]
    candidate_values = current.model_dump() | {"executions": (first, first)}
    with pytest.raises(ValueError, match="duplicate or conflicting logical producers"):
        RegistryCheckpoint.model_validate(candidate_values)
    other = RecipeBinding.model_validate(
        first.recipe.model_dump()
        | {
            "controls": {
                "reference": ref("other-controls"),
                "definition_fingerprint": "sha256:" + "f" * 64,
            }
        }
    )
    producer = origin(current, 1, recipe=other)
    candidate_values = current.model_dump() | {"executions": (producer,)}
    with pytest.raises(ValueError, match="frozen complete recipe binding"):
        RegistryCheckpoint.model_validate(candidate_values)


def test_original_snapshot_requires_current_policy_namespace_history_length_and_exact_prefix():
    current = checkpoint()
    registry = current.registries[0]
    original = current.executions[0]
    wrong_policy = RegistryState.model_validate(
        registry.model_dump()
        | {
            "policy": registry.policy.model_dump() | {"population": ref("other-population")},
        }
    )
    longer = extended_registry(current, introduce=True)
    changed_fact = LifecycleFact.model_validate(registry.facts[0].model_dump() | {"reason": "changed-old-reason"})
    rewritten = RegistryState.model_validate(registry.model_dump() | {"facts": (changed_fact,)})
    for state, message in (
        (wrong_policy, "another registry policy or history"),
        (longer, "another registry policy or history"),
        (rewritten, "exact accepted lifecycle prefix"),
    ):
        candidate = origin(current, registry=state)
        candidate_values = current.model_dump() | {"executions": (candidate,)}
        with pytest.raises(ValueError, match=message):
            RegistryCheckpoint.model_validate(candidate_values)
    snapshot = original.snapshot.model_validate(
        original.snapshot.model_dump()
        | {
            "catalogue": original.snapshot.catalogue.model_dump() | {"reference": ref("other-catalogue")},
        }
    )
    candidate = RecipeExecution.model_validate(original.model_dump() | {"snapshot": snapshot})
    candidate_values = current.model_dump() | {"executions": (candidate,)}
    with pytest.raises(ValueError, match="another catalogue namespace/version"):
        RegistryCheckpoint.model_validate(candidate_values)


def test_conflicting_historical_publication_and_registry_leaf_rewrites_preserve_prior_bytes():
    current = checkpoint(retained=False)
    registry = current.registries[0]
    bad_leaf = MemberDefinition.model_validate(
        registry.catalogue.definitions[0].model_dump()
        | {
            "configuration_fingerprint": "sha256:" + "f" * 64,
        }
    )
    bad_publication = Catalogue(schema_version=1, reference=registry.catalogue.reference, definitions=(bad_leaf,))
    candidate_values = current.model_dump() | {"catalogues": (bad_publication,)}
    with pytest.raises(ValueError, match="immutable retained leaves"):
        RegistryCheckpoint.model_validate(candidate_values)
    grown = current.replace_registry(extended_registry(current, kind="action", name="unrelated-action"))
    state = grown.registries[0]
    unrelated = state.catalogue.resolve("action", ref("unrelated-action"))
    changed = MemberDefinition.model_validate(
        unrelated.model_dump() | {"configuration_fingerprint": "sha256:" + "f" * 64}
    )
    leaves = tuple(changed if leaf == unrelated else leaf for leaf in state.catalogue.definitions)
    replacement = RegistryState.model_validate(
        state.model_dump()
        | {
            "catalogue": Catalogue(
                schema_version=1,
                reference=state.catalogue.reference,
                definitions=leaves,
            )
        }
    )
    before = grown.to_json()
    with pytest.raises(ValueError, match="immutable historical definition"):
        grown.replace_registry(replacement)
    assert grown.to_json() == before


def test_registry_owner_replace_refuses_fact_rewrite_policy_limits_namespace_and_unknown_task_atomically():
    current = checkpoint(retained=False)
    state = current.registries[0]
    changed_fact = LifecycleFact.model_validate(state.facts[0].model_dump() | {"reason": "rewritten"})
    changes = (
        ({"facts": (changed_fact,)}, "accepted lifecycle facts"),
        ({"policy": state.policy.model_dump() | {"population": ref("another-population")}}, "fixed policy"),
        ({"limits": RegistryLimits(maximum_producers=10)}, "selected limits"),
        ({"catalogue": state.catalogue.model_dump() | {"reference": ref("another-catalogue")}}, "catalogue namespace"),
        ({"policy": state.policy.model_dump() | {"task": ref("unknown-task")}}, "Unknown checkpoint registry task"),
    )
    owner = RegistryOwner(current)
    before = owner.export_state().to_json()
    for fields, message in changes:
        replacement = RegistryState.model_validate(state.model_dump() | fields)
        with pytest.raises(ValueError, match=message):
            owner.replace_registry(replacement)
        assert owner.export_state().to_json() == before


def test_attribute_input_applies_selected_resource_and_producer_caps_after_valid_field_extraction():
    current = checkpoint(retained=False)
    assert RegistryCheckpoint.model_validate(SimpleNamespace(**current.model_dump()), from_attributes=True) == current
    claim = current.ownership.resources[0]
    extra = claim.model_validate(claim.model_dump() | {"resource": ref("extra-weights"), "role": "weights"})
    declaration = OwnershipDeclaration.model_validate(
        current.ownership.model_dump()
        | {
            "resources": (*current.ownership.resources, extra),
        }
    )
    for fields, message in (
        ({"ownership": declaration, "limits": RegistryLimits(maximum_resources=1)}, "ownership resource capacity"),
        (
            {"executions": (origin(current), origin(current, 1)), "limits": RegistryLimits(maximum_producers=1)},
            "producer capacity",
        ),
    ):
        supplied = SimpleNamespace(**(current.model_dump() | fields))
        with pytest.raises(ValueError, match=message):
            RegistryCheckpoint.model_validate(supplied, from_attributes=True)


@pytest.mark.parametrize(
    "leaf_count,message",
    [
        (2, "publication-leaf validation work"),
        (3, "publication normalization work"),
    ],
)
def test_automatic_current_publication_work_is_bounded_before_original_comparisons(leaf_count, message):
    current = checkpoint(retained=False)
    registry = extended_registry(current, name="extra-1")
    if leaf_count == 3:
        leaf = MemberDefinition.model_validate(
            registry.catalogue.definitions[0].model_dump() | {"member": ref("extra-2")}
        )
        registry = RegistryState.model_validate(
            registry.model_dump() | {"catalogue": registry.catalogue.register(leaf)}
        )
    assert len(registry.catalogue.definitions) == leaf_count
    candidate_values = current.model_dump() | {
        "registries": (registry,),
        "catalogues": (),
        "limits": RegistryLimits(maximum_checkpoint_work=1),
    }
    with pytest.raises(ValueError, match=message):
        RegistryCheckpoint.model_validate(candidate_values)


@pytest.mark.parametrize("root", [None, [], 1, "not-a-checkpoint"])
def test_checkpoint_readers_reject_non_object_roots_without_coercion(root):
    for reader, payload in (
        (RegistryCheckpoint.model_validate, root),
        (RegistryCheckpoint.from_json, json.dumps(root)),
    ):
        with pytest.raises(ValueError):
            reader(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("registries", {}),
        ("executions", "bad-shape"),
        ("catalogues", {}),
        ("registries", ({"facts": None},)),
        ("catalogues", ({"definitions": None},)),
        ("ownership", None),
    ],
)
def test_checkpoint_reader_rejects_malformed_collection_and_nested_contract_shapes(field, value):
    current = checkpoint(retained=False)
    candidate_values = current.model_dump() | {field: value}
    with pytest.raises(ValueError):
        RegistryCheckpoint.model_validate(candidate_values)


def test_catalogue_selected_definition_capacity_refuses_new_leaf_but_keeps_exact_retry():
    catalogue = history(count=0).catalogue
    limits = RegistryLimits(maximum_definitions=len(catalogue.definitions))
    first = catalogue.definitions[0]
    new = MemberDefinition.model_validate(first.model_dump() | {"member": ref("new-versioned-member")})
    before = catalogue.to_json()
    assert catalogue.register(first, limits=limits) == catalogue
    with pytest.raises(ValueError, match="definition capacity exhausted"):
        catalogue.register(new, limits=limits)
    assert catalogue.to_json() == before


def test_registry_json_bytearray_ingress_reuses_strict_core_readers_and_byte_admission():
    catalogue = history(count=0).catalogue
    payload = bytearray(catalogue.to_json().encode("utf-8"))
    assert Catalogue.model_validate_json(payload) == catalogue
    assert Catalogue.model_validate_json(payload, by_alias=False, by_name=True) == catalogue
    with pytest.raises(ValueError):
        Catalogue.model_validate_json(payload, strict=False)
    with pytest.raises(ValueError):
        Catalogue.model_validate_json(payload, extra="allow")
    invalid_payload = bytearray(16 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="hard ingress cap"):
        Catalogue.model_validate_json(invalid_payload)
    unknown = catalogue.to_json().replace('"schema_version":1', '"unrecognized":null,"schema_version":1', 1)
    invalid_payload = bytearray(unknown.encode("utf-8"))
    with pytest.raises(ValueError):
        Catalogue.model_validate_json(invalid_payload)


@pytest.mark.parametrize(
    "operation,fields",
    [
        ("introduce", {"state_epoch": None}),
        ("introduce", {"disposition": "retain"}),
        ("reenter", {"state_epoch": ref("new-epoch"), "predecessor_epoch": None, "disposition": "restart"}),
        ("inactive", {"state_epoch": ref("old-epoch")}),
        ("retire", {"predecessor_epoch": ref("old-epoch")}),
        ("resume", {"disposition": "retain"}),
    ],
)
def test_lifecycle_fact_rejects_missing_or_ignored_explicit_epoch_fields(operation, fields):
    initial = history(count=1).facts[0]
    candidate_values = initial.model_dump() | {"operation": operation} | fields
    with pytest.raises(ValueError, match="requires|require|retain epochs"):
        LifecycleFact.model_validate(candidate_values)


def test_ordinary_lifecycle_cannot_resume_active_repeat_sleep_or_change_version_without_restart():
    state = history(count=1)
    before = state.to_json()
    candidate_fact = fact(state, operation="resume", round_number=1)
    with pytest.raises(ValueError, match="Invalid lifecycle transition"):
        state.append(candidate_fact)
    sleeping = state.append(fact(state, operation="inactive", round_number=1))
    candidate_fact = fact(sleeping, operation="inactive", round_number=2)
    with pytest.raises(ValueError, match="Invalid lifecycle transition"):
        sleeping.append(candidate_fact)
    second = MemberDefinition.model_validate(
        state.catalogue.resolve("expert", ref("A")).model_dump()
        | {
            "member": ref("A", "v2"),
        }
    )
    extended = RegistryState.model_validate(state.model_dump() | {"catalogue": state.catalogue.register(second)})
    candidate_fact = fact(
        extended,
        operation="retire",
        round_number=1,
        member=second.member,
        definition_fingerprint=second.fingerprint,
    )
    with pytest.raises(ValueError, match="cannot inherit another definition/version"):
        extended.append(candidate_fact)
    retired = state.append(fact(state, operation="retire", round_number=1))
    candidate_fact = fact(
        retired,
        operation="reenter",
        round_number=2,
        state_epoch=ref("A:e1"),
        predecessor_epoch=ref("other-epoch"),
        disposition="restart",
    )
    with pytest.raises(ValueError, match="exact retired predecessor epoch"):
        retired.append(candidate_fact)
    assert state.to_json() == before


def test_registry_reader_checks_selected_fact_producer_counts_and_unique_seals():
    state = history(count=2)
    candidate_values = state.model_dump() | {"limits": RegistryLimits(maximum_facts=1)}
    with pytest.raises(ValueError, match="definition/fact count"):
        RegistryState.model_validate(candidate_values)
    first = seal_request(state, registered_request(state, decision_id="first"))
    second = seal_request(state, registered_request(state, decision_id="second"))
    seals = (*first.producers, *second.producers)
    combined = RegistryState.model_validate(state.model_dump() | {"producers": seals})
    assert len(combined.producers) == 2
    candidate_values = state.model_dump() | {"producers": seals, "limits": RegistryLimits(maximum_producers=1)}
    with pytest.raises(ValueError, match="producer count exceeds selected capacity"):
        RegistryState.model_validate(candidate_values)
    candidate_values = state.model_dump() | {"producers": (*first.producers, *first.producers)}
    with pytest.raises(ValueError, match="Duplicate or conflicting retained logical opportunity"):
        RegistryState.model_validate(candidate_values)
    wrong_clock = first.producers[0].model_validate(first.producers[0].model_dump() | {"cutoff": START})
    candidate_values = state.model_dump() | {"producers": (wrong_clock,)}
    with pytest.raises(ValueError, match="fixed population clock policy"):
        RegistryState.model_validate(candidate_values)


def test_same_round_seals_accept_distinct_requests_for_one_exact_snapshot():
    state = history(count=1)
    first_request = registered_request(state, decision_id="first")
    second_request = registered_request(state, decision_id="second")
    assert first_request.snapshot.to_json() == second_request.snapshot.to_json()
    first = seal_request(state, first_request).producers[0]
    second = seal_request(state, second_request).producers[0]
    assert first.logical_identity != second.logical_identity
    assert first.request_fingerprint != second.request_fingerprint
    assert first.envelope_fingerprint != second.envelope_fingerprint
    assert first.snapshot_fingerprint == second.snapshot_fingerprint
    combined = RegistryState.model_validate(state.model_dump() | {"producers": (first, second)})
    assert len(combined.producers) == 2
    assert RegistryState.from_json(combined.to_json()).to_json() == combined.to_json()


def test_same_round_seals_reject_conflicting_snapshot_fingerprints_at_reader():
    state = history(count=1)
    first = seal_request(state, registered_request(state, decision_id="first")).producers[0]
    second = seal_request(state, registered_request(state, decision_id="second")).producers[0]
    altered = type(second).model_validate(second.model_dump() | {"snapshot_fingerprint": "sha256:" + "f" * 64})
    assert altered.snapshot_fingerprint != second.snapshot_fingerprint
    assert altered.model_dump(exclude={"snapshot_fingerprint"}) == second.model_dump(exclude={"snapshot_fingerprint"})
    assert (first.round, first.fact_count, first.cutoff, first.history_fingerprint) == (
        second.round,
        second.fact_count,
        second.cutoff,
        second.history_fingerprint,
    )
    candidate = state.model_dump() | {"producers": (first, altered)}
    with pytest.raises(ValueError, match="Same-round producing seals"):
        RegistryState.model_validate(candidate)


def test_same_round_seals_cannot_capture_distinct_valid_prefixes_or_cutoffs():
    initial = history(count=1)
    first = seal_request(initial, registered_request(initial, decision_id="first"))
    extended = initial.append(fact(initial, name="B", round_number=10))
    second = seal_request(extended, registered_request(extended, decision_id="second"))
    candidate_values = extended.model_dump() | {"producers": (*first.producers, *second.producers)}
    with pytest.raises(ValueError, match="Same-round producing seals"):
        RegistryState.model_validate(candidate_values)
    timed = history(count=1, clock="timed")
    first = seal_request(timed, registered_request(timed, decision_id="first"))
    later = START + timedelta(minutes=1)
    snapshot = timed.snapshot(0, later)
    original = registered_request(timed, decision_id="second")
    assert original.request.timing is not None
    timing = DecisionTiming.model_validate(
        original.request.timing.model_dump()
        | {
            "decision_at": later,
            "information_cutoff": later,
            "target_start": later,
            "target_end": later,
        }
    )
    request = original.request.model_validate(original.request.model_dump() | {"timing": timing})
    second = seal_request(timed, RegistryRequest(schema_version=1, request=request, snapshot=snapshot))
    candidate_values = timed.model_dump() | {"producers": (*first.producers, *second.producers)}
    with pytest.raises(ValueError, match="Same-round producing seals"):
        RegistryState.model_validate(candidate_values)


def test_sealing_rejects_backward_new_opportunity_and_foreign_complete_snapshot_before_mutation():
    state = history(count=1)
    earlier = registered_request(state, 1, "earlier")
    sealed = seal_request(state, registered_request(state, 2, "later"))
    before = sealed.to_json()
    with pytest.raises(ValueError, match="move backward across sealed rounds"):
        seal_request(sealed, earlier)
    leaf = MemberDefinition.model_validate(state.catalogue.definitions[0].model_dump() | {"member": ref("unrelated")})
    changed = RegistryState.model_validate(state.model_dump() | {"catalogue": state.catalogue.register(leaf)})
    candidate_request = registered_request(changed, 1)
    with pytest.raises(ValueError, match="current exact registry history"):
        seal_request(state, candidate_request)
    assert sealed.to_json() == before


def test_registered_profile_rejects_continuous_spaces_and_mapped_active_universes():
    original = registered_request(history(count=2))
    vector = VectorSpace(feasible_set=original.snapshot.feasible_set("continuous"), coordinates=(), signed=False)
    request = original.request.model_validate(original.request.model_dump() | {"space": vector})
    candidate_values = original.model_dump() | {"request": request}
    with pytest.raises(ValueError, match="finite member spaces only"):
        RegistryRequest.model_validate(candidate_values)
    producing = execution(original)
    space = ChoiceSpace(feasible_set=ref("mapped-feasible-set"), members=(ref("A"),))
    action = ChoiceAction(schema_version=1, space=space, chosen=ref("A"))
    sampling = producing.executed.sampling.model_validate(
        producing.executed.sampling.model_dump()
        | {
            "action_fingerprint": action.fingerprint,
        }
    )
    mapped = ExecutedDecision(
        schema_version=1,
        proposal=producing.executed.proposal,
        action=action,
        mapping="filtered",
        sampling_disposition="recomputed",
        sampling=sampling,
    )
    with pytest.raises(ValueError, match="exact active contract space"):
        RegisteredExecution(schema_version=1, request=original, executed=mapped)


def test_empty_registered_population_requires_abstain_even_when_core_empty_ranking_is_valid():
    empty = registered_request(history(count=0))
    space = RankingSpace(feasible_set=empty.snapshot.feasible_set("empty-ranking"), members=())
    request = empty.request.model_validate(empty.request.model_dump() | {"space": space})
    envelope = RegistryRequest(schema_version=1, request=request, snapshot=empty.snapshot)
    action = RankingAction(schema_version=1, space=space, order=())
    actual = _unchanged_execution(request, action)
    assert isinstance(actual.action, RankingAction)
    assert not actual.action.order
    with pytest.raises(ValueError, match="Empty active execution requires explicit AbstainAction"):
        RegisteredExecution(schema_version=1, request=envelope, executed=actual)
    assert execution(envelope).executed.action.kind == "abstain"
