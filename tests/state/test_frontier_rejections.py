"""Independent valid histories and hostile wire data exercise retained admission."""

import json
from dataclasses import dataclass
from typing import Any, cast

import pytest

from coleman.core.actions import ChoiceAction
from coleman.core.primitives import fingerprint
from coleman.core.splits import TemporalObservation
from coleman.observations.contracts import ObservationHistory, ObservationRevision
from coleman.replay.engine import CausalReplay
from coleman.replay.retained import RetainedReplay, ordinal_decision_id, rebind_retained
from coleman.replay.retained_codec import read_retained_continuation
from coleman.replay.retained_contracts import OrdinalRetention, RetainedContinuation
from coleman.results.domain import Measurement
from coleman.state import _prepare, _relations
from coleman.state.components import ComponentCodec, ComponentValue, PreparedCollection
from coleman.state.storage_contracts import (
    AcknowledgedHead,
    MigrationReceipt,
    StorageLimits,
    semantic_frontiers_fingerprint,
)
from tests.state.migrations import definition, target_codec, target_layout, wrapped
from tests.state.support import command


@dataclass(frozen=True)
class Histories:
    origin: RetainedContinuation[Any]
    one: RetainedContinuation[Any]
    two: RetainedContinuation[Any]
    sealed: RetainedContinuation[Any]
    rebound: RetainedContinuation[Any]
    with_source: RetainedContinuation[Any]
    sealed_source: RetainedContinuation[Any]


def revision(example, *, value=3.0, available=2):
    """A real future source fact stays invisible to the original decision."""
    return ObservationRevision(
        schema_version=1,
        metadata=TemporalObservation(
            observation_id="authored-build-duration",
            revision_id="original",
            group_id="authored-build",
            event_at=example.instant(0),
            available_at=example.instant(available),
        ),
        entity_id="authored-test",
        received_at=example.instant(available),
        supersedes=None,
        measurement=Measurement(
            name="authored-duration",
            value=value,
            unit="s",
            meaning="authored-test-duration-v1",
            direction="minimize",
            status="observed",
        ),
    )


def source_history(example, *rows):
    return ObservationHistory.model_validate(example.history().model_copy(update={"revisions": rows}))


@pytest.fixture(scope="module")
def histories(state_example):
    example = state_example
    owner = example.ReplayOwner(0)
    origin = owner.export_state()
    owner.step(0.25)
    one = owner.export_state()
    sealed = owner.replay.compact()
    rebound = rebind_retained(
        sealed,
        wrapped(example, sealed.host_state),
        target_codec=target_codec(example),
        migration=definition(example, "authored-wrap-complete-ci-state"),
    )
    owner.restore_state(one)
    owner.step(0.5)
    two = owner.export_state()
    owner.restore_state(one)
    owner.replay.admit_history(source_history(example, revision(example)))
    with_source = owner.export_state()
    sealed_source = owner.replay.compact()
    return Histories(origin, one, two, sealed, rebound, with_source, sealed_source)


def clone(value, **changes):
    """Validate the whole changed value; no unchecked model enters a relation."""
    return type(value).model_validate(value.model_copy(update=changes))


def head(prepared):
    manifest = prepared.manifest
    return AcknowledgedHead(
        schema_version=1,
        session=manifest.definition.session,
        source_epoch=manifest.definition.source_epoch,
        definition_fingerprint=manifest.definition.fingerprint,
        generation=manifest.intent.generation,
        manifest_fingerprint=manifest.fingerprint,
        intent_fingerprint=manifest.intent.fingerprint,
    )


@pytest.fixture(scope="module")
def collections(state_example, histories):
    example = state_example
    layout, bindings = example.collection_layout()
    compatibility = example.execution_compatibility()
    limits = StorageLimits(schema_version=1)
    domain = example.Domain()
    origin_domain = domain.export_state()
    # The authored reservoir is binary100, so its first two emitted plans are
    # both A. Apply those actual public choices to the separate physical book.
    assert histories.one.state.decisions[0].executed.action == histories.two.state.decisions[0].executed.action
    for record in histories.two.state.decisions:
        action = record.executed.action
        assert isinstance(action, ChoiceAction)
        assert action.chosen == example.PLANS[0]
    domain.apply(example.PLANS[0])
    one_domain = domain.export_state()
    domain.apply(example.PLANS[0])
    two_domain = domain.export_state()

    def build(snapshot, domain_state, predecessor, generation, kind):
        return _prepare.build(
            layout,
            compatibility,
            command(generation, predecessor, kind=kind),
            (ComponentValue(bindings[0].codec, snapshot), ComponentValue(bindings[-1].codec, domain_state)),
            bindings,
            limits,
        )

    origin = build(histories.origin, origin_domain, None, 0, "initialize")
    one = build(histories.one, one_domain, head(origin), 1, "transition")
    two = build(histories.two, two_domain, head(one), 2, "transition")
    sealed = build(histories.sealed, one_domain, head(one), 2, "seal")
    return origin, one, two, sealed


def test_valid_complete_causal_and_seal_relations_accept_without_changing_any_value(collections):
    origin, one, two, sealed = collections
    before = tuple(item.manifest.to_json() for item in collections)
    _relations.validate_successor(origin, one, "transition")
    _relations.validate_successor(one, two, "transition")
    _relations.validate_successor(one, sealed, "seal")
    assert tuple(item.manifest.to_json() for item in collections) == before


@pytest.mark.parametrize("kind", ["initialize", "unknown", ""])
def test_publication_relation_rejects_non_successor_kinds_before_acceptance(collections, kind):
    origin, one, _, _ = collections
    selected_kind = cast(Any, kind)
    with pytest.raises(ValueError, match="requires transition"):
        _relations.validate_successor(origin, one, selected_kind)


def test_valid_seal_intent_cannot_be_relabelled_as_transition(collections):
    _, one, _, sealed = collections
    with pytest.raises(ValueError, match="publication intent"):
        _relations.validate_successor(one, sealed, "transition")


def test_ordinary_relation_rejects_an_unrequested_mapping_and_a_foreign_actual_predecessor(state_example, collections):
    origin, one, two, _ = collections
    mapping = definition(state_example, "authored-not-a-transition-mapping")
    with pytest.raises(ValueError, match="Exactly codec_migration"):
        _relations.validate_successor(origin, one, "transition", migration=mapping)
    with pytest.raises(ValueError, match="actual complete acknowledged predecessor"):
        _relations.validate_successor(origin, two, "transition")


@pytest.mark.parametrize("inventory", ["missing", "duplicate", "different_binding"])
def test_prepared_owner_inventory_is_checked_even_when_the_manifest_itself_is_valid(collections, inventory):
    _, one, _, _ = collections
    first, last = one.values
    if inventory == "missing":
        values = (first,)
    elif inventory == "duplicate":
        values = (first, first, last)
    else:
        changed = clone(first.codec.binding, configuration_fingerprint=fingerprint("different-configuration"))
        values = (ComponentValue(ComponentCodec(changed, first.codec.state_type), first.value), last)
    candidate = PreparedCollection(one.manifest, values)
    with pytest.raises(ValueError, match="inventory|duplicate"):
        _relations._owners(candidate)


def test_role_relabelling_cannot_change_the_owned_acceptance_contract(collections):
    origin, _, _, _ = collections
    layout = origin.manifest.definition
    changed = clone(layout.components[-1], roles=("domain", "applied_path", "transform"))
    layout = clone(layout, components=(*layout.components[:-1], changed))
    # Each complete initialization is valid on its own. Comparing their role
    # declarations still forbids relabelling an owner at a later boundary.
    manifest = clone(origin.manifest, definition=layout)
    candidate = PreparedCollection(manifest, origin.values)
    with pytest.raises(ValueError, match="owner/child/role"):
        _relations._definition_identity(origin, candidate)


def test_direct_publication_has_no_implicit_replay_seal_semantics(state_example, collections):
    example = state_example
    old = collections[0].values[-1]
    domain = example.Domain()
    domain.apply(example.PLANS[0])
    new = ComponentValue(old.codec, domain.export_state())
    _relations._component_successor(old, new, "transition", None)
    with pytest.raises(ValueError, match="direct component"):
        _relations._component_successor(old, new, "seal", None)


def test_a_complete_replay_owner_cannot_be_reinterpreted_as_a_direct_host(state_example, collections):
    old = collections[0].values[0]
    replacement = ComponentValue(ComponentCodec(old.codec.binding, state_example.HostState), old.value.host_state)
    with pytest.raises(ValueError, match="add or remove a replay"):
        _relations._component_successor(old, replacement, "transition", None)


def test_an_ordinary_component_cannot_change_its_declared_configuration_binding(collections):
    old = collections[0].values[0]
    changed = clone(old.codec.binding, configuration_fingerprint=fingerprint("a-new-configuration"))
    new = ComponentValue(ComponentCodec(changed, old.codec.state_type), old.value)
    with pytest.raises(ValueError, match="codec/configuration/code"):
        _relations._component_successor(old, new, "transition", None)


@pytest.mark.parametrize("boundary", ["policy", "retention", "codec"])
def test_individually_valid_origins_cannot_change_a_fixed_replay_boundary(state_example, histories, boundary):
    example = state_example
    policy = example.policy(0)
    profile = histories.origin.state.retention
    codec = example.CODEC
    if boundary == "policy":
        policy = clone(policy, stream=example.reference("another-stream"))
    elif boundary == "retention":
        profile = clone(profile, source_epoch=example.reference("another-source-epoch"))
    else:
        codec = example.reference("another-host-codec")
    other = RetainedReplay(
        example.HostConsumer(0), policy, state_type=example.HostState, state_codec=codec, retention=profile
    ).export_continuation()
    with pytest.raises(ValueError, match="policy|retention|host codec"):
        _relations._replay_successor(histories.origin, other, "transition", None)


def test_admitted_feedback_and_source_facts_are_immutable_across_valid_histories(state_example, histories):
    example = state_example
    other = example.ReplayOwner(0)
    other.step(0.5)
    changed_feedback = other.export_state().state
    with pytest.raises(ValueError, match="feedback revision"):
        _relations._feedback_extension(histories.one.state, changed_feedback)
    with pytest.raises(ValueError, match="source revision"):
        _relations._history_extension(histories.with_source.state, histories.one.state)
    other.restore_state(histories.one)
    other.replay.admit_history(source_history(example, revision(example, value=5.0)))
    changed_source = other.export_state().state
    with pytest.raises(ValueError, match="source revision"):
        _relations._history_extension(histories.with_source.state, changed_source)
    _relations._history_extension(histories.one.state, histories.with_source.state)


def test_source_inventory_cannot_be_deleted_or_renamed_at_a_later_generation(state_example, histories):
    example = state_example
    bare = RetainedReplay(
        example.HostConsumer(0),
        example.policy(0),
        state_type=example.HostState,
        state_codec=example.CODEC,
        retention=histories.origin.state.retention,
    )
    with pytest.raises(ValueError, match="complete source history"):
        _relations._history_extension(histories.origin.state, bare.state)
    changed_policy = clone(example.policy(0), source=example.reference("another-source"))
    renamed = RetainedReplay(
        example.HostConsumer(0),
        changed_policy,
        state_type=example.HostState,
        state_codec=example.CODEC,
        retention=histories.origin.state.retention,
    )
    renamed.admit_history(clone(example.history(), source_id="another-source"))
    with pytest.raises(ValueError, match="source identity/version"):
        _relations._history_extension(histories.origin.state, renamed.state)
    _relations._history_extension(bare.state, histories.origin.state)


def test_independently_valid_later_clock_and_counter_states_cannot_be_rewound(state_example, histories):
    example = state_example
    later = example.ReplayOwner(0)
    later.step(0.25, at=3)
    earlier = example.ReplayOwner(0)
    earlier.step(0.25, at=2)
    later_state = later.export_state().state
    earlier_state = earlier.export_state().state
    with pytest.raises(ValueError, match="either replay clock"):
        _relations._monotone_frontiers(later_state, earlier_state)
    with pytest.raises(ValueError, match="global replay counters"):
        _relations._monotone_frontiers(histories.two.state, histories.one.state)
    with pytest.raises(ValueError, match="exact replay counters"):
        _relations._fixed_boundary(histories.one.state, histories.two.state)
    _relations._monotone_frontiers(histories.one.state, histories.two.state)


@pytest.mark.parametrize("change", ["children", "owner_version"])
def test_component_owner_and_child_identity_are_not_migration_relabelling_knobs(collections, change):
    origin = collections[0]
    layout = origin.manifest.definition
    if change == "children":
        child = clone(layout.children[0], version="another-child-version")
        spec = clone(layout.components[0], child=child)
        layout = clone(layout, children=(child,), components=(spec, *layout.components[1:]))
        frontier = clone(origin.manifest.frontiers[0], child=child)
        manifest = clone(origin.manifest, definition=layout, frontiers=(frontier,))
        match = "exact child identities"
    else:
        owner = clone(layout.components[-1].binding.owner, version="another-owner-version")
        binding = clone(layout.components[-1].binding, owner=owner)
        spec = clone(layout.components[-1], binding=binding)
        layout = clone(layout, components=(*layout.components[:-1], spec))
        blob = clone(origin.manifest.components[-1], binding=binding)
        manifest = clone(origin.manifest, definition=layout, components=(*origin.manifest.components[:-1], blob))
        match = "owner/child/role"
    candidate = PreparedCollection(manifest)
    with pytest.raises(ValueError, match=match):
        _relations._definition_identity(origin, candidate)


def migration_candidate(example, source, old):
    """Build a fully validated proposed migration; publication still needs its relation check."""
    source_definition, source_bindings = example.collection_layout()
    target_definition, bindings, compatibility, mapping = target_layout(
        example, source_definition, source_bindings, old.manifest.compatibility
    )
    rebound = rebind_retained(
        source,
        wrapped(example, source.host_state),
        target_codec=target_codec(example),
        migration=mapping,
    )
    values = (
        ComponentValue(bindings[0].codec, rebound),
        ComponentValue(bindings[-1].codec, old.values[-1].value),
    )
    frontiers = _prepare.frontiers(target_definition, values, bindings)
    receipt = MigrationReceipt(
        schema_version=1,
        migration=mapping,
        source_head=head(old),
        source_definition_fingerprint=source_definition.fingerprint,
        target_definition_fingerprint=target_definition.fingerprint,
        preserved_frontiers_fingerprint=semantic_frontiers_fingerprint(frontiers),
    )
    candidate = _prepare.build(
        target_definition,
        compatibility,
        command(old.manifest.intent.generation + 1, head(old), kind="codec_migration"),
        values,
        bindings,
        StorageLimits(schema_version=1),
        receipt,
    )
    return candidate, mapping


def test_a_real_typed_migration_preserves_the_actual_source_semantic_projection(state_example, histories, collections):
    old = collections[-1]
    target, mapping = migration_candidate(state_example, histories.sealed, old)
    _relations.validate_successor(old, target, "codec_migration", migration=mapping)
    another_mapping = definition(state_example, "another-qualified-mapping")
    with pytest.raises(ValueError, match="qualified collection migration"):
        _relations.validate_successor(
            old,
            target,
            "codec_migration",
            migration=another_mapping,
        )
    owner = state_example.ReplayOwner(0)
    owner.restore_state(histories.two)
    different_source = owner.replay.compact()
    changed, mapping = migration_candidate(state_example, different_source, old)
    # Both typed manifests are internally complete, but D2 must not be claimed
    # as a codec-only mapping from the acknowledged D1 predecessor.
    assert changed.manifest.frontiers[0].decision_count == 2
    assert old.manifest.frontiers[0].decision_count == 1
    with pytest.raises(ValueError, match="actual preserved semantic frontiers"):
        _relations.validate_successor(old, changed, "codec_migration", migration=mapping)


@pytest.mark.parametrize("reason", ["no_seal", "live_tail", "no_receipt", "wrong_mapping", "wrong_source"])
def test_codec_migration_requires_the_actual_sealed_source_and_named_mapping(state_example, histories, reason):
    example = state_example
    mapping = definition(example, "authored-wrap-complete-ci-state")
    old, new = histories.sealed.state, histories.rebound.state
    if reason == "no_seal":
        old = new = histories.origin.state
        match = "generation-two seal"
    elif reason == "live_tail":
        owner = example.ReplayOwner(0)
        owner.restore_state(histories.sealed)
        owner.step(0.25)
        old = new = owner.export_state().state
        match = "live tails empty"
    elif reason == "no_receipt":
        new = old
        match = "exactly one appended receipt"
    elif reason == "wrong_mapping":
        mapping = definition(example, "a-different-reviewed-mapping")
        match = "qualified migration definition"
    else:
        receipt = clone(new.rebinds[0], source_state_fingerprint=fingerprint("another-valid-source"))
        new = clone(new, rebinds=(receipt,))
        match = "actual complete acknowledged source"
    with pytest.raises(ValueError, match=match):
        _relations._migration(old, new, mapping)


@pytest.mark.parametrize("reason", ["empty_origin", "live_tail", "wrong_source", "wrong_drain", "wrong_first"])
def test_seal_must_describe_the_actual_complete_predecessor_not_just_equal_counts(histories, reason):
    old, new = histories.one.state, histories.sealed.state
    if reason == "empty_origin":
        old = new = histories.origin.state
        match = "completed quiescence"
    elif reason == "live_tail":
        new = old
        match = "empty live tail"
    else:
        assert new.seal is not None
        seal = new.seal
        if reason == "wrong_source":
            seal = clone(seal, source_state_fingerprint=fingerprint("another-source-checkpoint"))
            match = "actual complete acknowledged predecessor"
        elif reason == "wrong_drain":
            seal = clone(seal, last_drain=clone(seal.last_drain, host_before=fingerprint("another-before-host")))
            match = "actual final closed drain"
        else:
            target = clone(seal.first_opportunity.target, reference=seal.last_drain.lineage)
            seal = clone(seal, first_opportunity=clone(seal.first_opportunity, target=target))
            match = "original first opportunity"
        new = clone(new, seal=seal)
    with pytest.raises(ValueError, match=match):
        _relations._seal(old, new)


def hostile_payload(example, histories, change):
    """Change serialized boundary facts, retaining ordinary parseable JSON."""
    source = histories.rebound if change.startswith("rebind_") else histories.sealed
    if change in {"source_missing", "source_changed"}:
        source = histories.sealed_source
    payload = source.model_dump(mode="json")
    state, seal = payload["state"], payload["state"]["seal"]
    if change == "zero_seal":
        seal["decision_count"] = seal["update_count"] = 0
    elif change == "pending_seal":
        seal["update_count"] = 0
    elif change == "drain_counters":
        seal["decision_count"] = seal["update_count"] = 2
    elif change == "sequence":
        seal["next_sequence"] += 1
    elif change == "host":
        seal["host_fingerprint"] = fingerprint("another-host")
    elif change == "duplicate_ancestry":
        seal["ancestors"].append(seal["ancestors"][0])
    elif change == "policy_lineage":
        state["policy"]["lineage"]["id"] = "another-policy-lineage"
    elif change == "missing_history":
        state["history"] = None
    elif change == "source_identity":
        state["history"]["source_id"] = "another-source"
    elif change == "source_version":
        state["history"]["source_version"] = "another-source-version"
    elif change == "source_missing":
        state["history"]["revisions"] = []
    elif change == "source_changed":
        state["history"]["revisions"][0]["measurement"]["value"] = 5.0
    elif change == "source_backfill":
        state["history"]["revisions"] = [revision(example, available=0).model_dump(mode="json")]
    elif change == "first_ordinal":
        seal["first_opportunity"]["request"]["round"] = 1
    elif change == "first_id":
        seal["first_opportunity"]["request"]["decision_id"] = "not-the-source-ordinal"
    elif change == "first_input":
        seal["first_opportunity"]["request"]["timing"]["eligible_input_fingerprint"] = fingerprint("not-the-view")
    elif change == "first_after_drain":
        timing = seal["first_opportunity"]["request"]["timing"]
        for key in ("information_cutoff", "decision_at", "target_start", "target_end"):
            timing[key] = example.instant(2).isoformat()
        timing["eligible_input_fingerprint"] = example.history().as_of(example.instant(2)).fingerprint
    elif change == "seal_codec":
        state["state_codec"]["id"] = "not-the-authorized-codec"
    elif change == "rebind_boundary":
        state["rebinds"][0]["boundary_fingerprint"] = fingerprint("not-the-boundary")
    elif change == "rebind_source_codec":
        state["rebinds"][0]["source_codec"]["id"] = "not-the-source-codec"
    elif change == "rebind_source_host":
        state["rebinds"][0]["source_host_fingerprint"] = fingerprint("not-the-source-host")
    elif change == "rebind_target_codec":
        state["rebinds"][0]["target_codec"]["id"] = "not-the-target-codec"
    elif change == "rebind_without_seal":
        origin = histories.origin.model_dump(mode="json")
        origin["state"]["rebinds"] = state["rebinds"]
        payload = origin
    else:
        raise AssertionError(change)
    return json.dumps(payload)


@pytest.mark.parametrize(
    "change",
    [
        "zero_seal",
        "pending_seal",
        "drain_counters",
        "sequence",
        "host",
        "duplicate_ancestry",
        "policy_lineage",
        "missing_history",
        "source_identity",
        "source_version",
        "source_missing",
        "source_changed",
        "source_backfill",
        "first_ordinal",
        "first_id",
        "first_input",
        "first_after_drain",
        "seal_codec",
        "rebind_boundary",
        "rebind_source_codec",
        "rebind_source_host",
        "rebind_target_codec",
        "rebind_without_seal",
    ],
)
def test_real_v2_reader_rejects_inconsistent_seals_and_codec_receipt_chains(state_example, histories, change):
    payload = hostile_payload(state_example, histories, change)
    snapshot_type = type(histories.rebound.host_state) if change.startswith("rebind_") else state_example.HostState
    if change == "rebind_without_seal":
        snapshot_type = state_example.HostState
    with pytest.raises(ValueError):
        read_retained_continuation(payload, state_type=snapshot_type)


@pytest.mark.parametrize("ordinal", [-1, True, 0.5, "0"])
def test_ordinal_namespace_refuses_non_integer_or_negative_indices_before_any_host(state_example, ordinal):
    selected_ordinal = cast(Any, ordinal)
    with pytest.raises(ValueError, match="nonnegative integer"):
        ordinal_decision_id(state_example.EPOCH, selected_ordinal)


def test_origin_and_repeated_seal_compaction_are_explicit_idempotent_pure_operations(state_example, histories):
    example = state_example
    replay, _ = example.make_replay(0)
    assert replay.compact().to_json() == histories.origin.to_json()
    replay = RetainedReplay.from_continuation(
        example.HostConsumer(0), histories.sealed, state_type=example.HostState, state_codec=example.CODEC
    )
    assert replay.compact().to_json() == histories.sealed.to_json()
    fresh = example.HostConsumer(0)
    wrong_codec = example.reference("wrong-selected-codec")
    with pytest.raises(ValueError, match="selected codec"):
        RetainedReplay.from_continuation(
            fresh,
            histories.sealed,
            state_type=example.HostState,
            state_codec=wrong_codec,
        )


def test_v1_matched_parent_cannot_be_installed_as_a_v2_origin(state_example):
    example = state_example
    legacy = CausalReplay(
        example.HostConsumer(0), example.policy(0), state_type=example.HostState, state_codec=example.CODEC
    )
    fork = CausalReplay.fork_from_continuation(
        example.HostConsumer(0),
        legacy.export_continuation(),
        lineage=example.reference("authored-v1-child-lineage"),
        state_type=example.HostState,
        state_codec=example.CODEC,
    )
    fresh = example.HostConsumer(0)
    retention = OrdinalRetention(profile="quiescent-ordinal-v1", source_epoch=example.EPOCH, maximum_tail_records=64)
    with pytest.raises(ValueError, match="independent origin"):
        RetainedReplay(
            fresh,
            fork.state.policy,
            state_type=example.HostState,
            state_codec=example.CODEC,
            retention=retention,
        )


def test_rebind_rejects_a_completed_live_suffix_and_a_noncontinuation_without_mutating_source(state_example, histories):
    example = state_example
    owner = example.ReplayOwner(0)
    owner.restore_state(histories.sealed)
    owner.step(0.25)
    live = owner.export_state()
    mapping = definition(example, "authored-wrap-complete-ci-state")
    wrapped_live = wrapped(example, live.host_state)
    codec = target_codec(example)
    with pytest.raises(ValueError, match="empty sealed tail"):
        rebind_retained(live, wrapped_live, target_codec=codec, migration=mapping)
    bare_host = cast(Any, histories.one.host_state)
    with pytest.raises(TypeError, match="concrete retained continuation"):
        rebind_retained(
            bare_host,
            histories.one.host_state,
            target_codec=example.CODEC,
            migration=mapping,
        )
    assert owner.export_state().to_json() == live.to_json()
