"""Known typed layout mappings preserve state without guessing compatibility."""

from functools import partial

import pytest

from coleman.replay.retained import RetainedReplay, rebind_retained
from coleman.state.components import MigrationBindings
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import MigrationRequest, semantic_frontiers_fingerprint
from tests.state.migrations import (
    definition,
    migrate_values,
    target_codec,
    target_layout,
    wrapped,
)
from tests.state.support import command, complete, emit


def initialized_seal(example, root):
    definition_value, bindings = example.collection_layout()
    compatibility = example.execution_compatibility()
    owner = bindings[0].new_owner()
    record = emit(example, owner.replay, 0, 0)
    complete(example, owner.replay, record, 0.25, at=0)
    owner.replay.compact()
    from coleman.state.components import ComponentValue  # pylint: disable=import-outside-toplevel

    values = (ComponentValue(bindings[0].codec, owner.export_state()), *example.origin_values(bindings[1:]))
    store = LocalStateStore.create(root, definition_value, compatibility)
    receipt = store.initialize(command(0, None, kind="initialize"), values, bindings, compatibility=compatibility)
    return store, bindings, compatibility, definition_value, receipt, record


def test_actual_known_replay_codec_migration_preserves_seal_rng_cache_path_and_old_delivery_fence(
    state_example, tmp_path
):
    example = state_example
    store, old_bindings, old_compatibility, old_definition, receipt, old_record = initialized_seal(
        example, tmp_path / "migration"
    )
    before = store.load(old_bindings, compatibility=old_compatibility)
    target_definition, bindings, compatibility, migration = target_layout(
        example, old_definition, old_bindings, old_compatibility
    )
    request = MigrationRequest(
        schema_version=1,
        commit=command(1, receipt.current_head, kind="codec_migration"),
        migration=migration,
        target_definition=target_definition,
    )
    readers = MigrationBindings(tuple(binding.codec for binding in old_bindings), bindings)
    migrate = partial(migrate_values, example, old_bindings=old_bindings, new_bindings=bindings, migration=migration)
    result = store.migrate(request, readers, migrate, compatibility=compatibility)
    after = store.load(bindings, compatibility=compatibility)
    source, target = before.component(old_bindings[0].codec), after.component(bindings[0].codec)
    assert target.host_state.payload.to_json() == source.host_state.to_json()
    assert target.state.seal is not None
    assert source.state.seal is not None
    assert target.state.seal == source.state.seal
    assert target.state.seal.last_drain.to_json() == source.state.seal.last_drain.to_json()
    assert target.state.causal_prefix_fingerprint == source.state.causal_prefix_fingerprint
    assert target.state.rebinds[-1].source_state_fingerprint == source.state.fingerprint
    assert result.current_head.generation == 1
    assert after.manifest.migration is not None
    assert after.manifest.migration.preserved_frontiers_fingerprint == semantic_frontiers_fingerprint(
        before.manifest.frontiers
    )
    calls = []
    old = example.feedback(old_record, decisions=1, available=1, value=0.25)
    followup = command(2, result.current_head)

    def expired(stage):
        calls.append("actual-restored-new-codec")
        stage.owner(bindings[0]).replay.admit_feedback((old,))

    with pytest.raises(ValueError, match="registered|expired"):
        store.commit_transition(followup, bindings, expired, compatibility=compatibility)
    assert calls == ["actual-restored-new-codec"]
    assert store.read_head() == result.current_head


def test_pending_replay_codec_migration_refuses_without_an_invented_quiescent_boundary(state_example):
    example = state_example
    replay, _ = example.make_replay(0)
    emit(example, replay, 0, 0)
    source = replay.export_continuation()
    target = wrapped(example, source.host_state)
    codec = target_codec(example)
    migration = definition(example, "authored-wrap-complete-ci-state")
    before = source.to_json()
    with pytest.raises(ValueError, match="quiescent|seal|empty"):
        rebind_retained(source, target, target_codec=codec, migration=migration)
    assert replay.export_continuation().to_json() == before


def test_two_known_host_layouts_can_alternate64_links_but_not_erase_the_limit_without_a_real_drain(state_example):
    example = state_example
    replay, host = example.make_replay(0)
    record = emit(example, replay, 0, 0)
    complete(example, replay, record, 0.25, at=0)
    current = replay.compact()
    original_host = host.export_state()
    to_wrapped = definition(example, "authored-wrap-complete-ci-state")
    to_original = definition(example, "authored-unwrap-complete-ci-state")
    for index in range(64):
        if index % 2 == 0:
            current = rebind_retained(
                current, wrapped(example, original_host), target_codec=target_codec(example), migration=to_wrapped
            )
        else:
            current = rebind_retained(current, original_host, target_codec=example.CODEC, migration=to_original)
    assert len(current.state.rebinds) == 64
    fresh = example.HostConsumer(0)
    restored = RetainedReplay.from_continuation(fresh, current, state_type=example.HostState, state_codec=example.CODEC)
    with pytest.raises(ValueError, match="new actual closed drain"):
        restored.compact()
    same = restored.export_continuation()
    assert len(same.state.rebinds) == 64
    assert same.to_json() == current.to_json()
    target = wrapped(example, original_host)
    codec = target_codec(example)
    with pytest.raises(ValueError, match="64|receipt|rebind"):
        rebind_retained(same, target, target_codec=codec, migration=to_wrapped)
    restored.advance(eligible_through=example.instant(1), apply_at=example.instant(1))
    new_seal = restored.compact()
    assert new_seal.state.rebinds == ()
    assert new_seal.state.decision_count == 1
    assert new_seal.state.update_count == 1
    assert new_seal.state.seal is not None
    assert new_seal.state.seal.last_drain.apply_at == example.instant(1)
    allowed = rebind_retained(new_seal, target, target_codec=codec, migration=to_wrapped)
    assert len(allowed.state.rebinds) == 1
    assert allowed.host_state.payload.rng == original_host.rng


def test_migration_receipt_cannot_silently_change_a_preserved_semantic_frontier(state_example, tmp_path):
    example = state_example
    store, bindings, compatibility, _, _, _ = initialized_seal(example, tmp_path / "receipt")
    before = store.load(bindings, compatibility=compatibility).manifest.frontiers
    old = before[0]
    changed = type(old).model_validate(old.model_copy(update={"decision_count": old.decision_count + 1}))
    assert semantic_frontiers_fingerprint(before) != semantic_frontiers_fingerprint((changed,))
    updated_codec = type(old).model_validate(old.model_copy(update={"state_codec": target_codec(example)}))
    assert semantic_frontiers_fingerprint(before) == semantic_frontiers_fingerprint((updated_codec,))


def test_known_direct_host_codec_migration_preserves_pending_work_without_replay_seal(state_example, tmp_path):
    from coleman.state.components import ComponentBinding, ComponentCodec, ComponentValue
    from coleman.state.contracts import StateBinding
    from coleman.state.storage_contracts import CollectionDefinition, ComponentSpec, ExecutionCompatibility
    from tests.state.migrations import WrappedConsumer, WrappedState

    example = state_example
    initial_definition, layout = example.collection_layout()
    old = layout[0]
    host_binding = StateBinding.model_validate(old.codec.binding.model_copy(update={"state_codec": example.CODEC}))
    direct = ComponentBinding(
        ComponentCodec(host_binding, example.HostState), partial(example.HostConsumer, 0), lambda value: None
    )
    component = ComponentSpec(
        binding=host_binding,
        child=example.child_reference(0),
        roles=("algorithm", "learner_pending"),
        frontier_profile="none",
    )
    definition_value = CollectionDefinition.model_validate(
        initial_definition.model_copy(update={"components": (component, initial_definition.components[-1])})
    )
    source_bindings = (direct, layout[-1])
    host = example.HostConsumer(0)
    host.decide(example.opportunity(0, 0).request)
    before = host.export_state()
    assert len(before.pending) == 1
    values = (ComponentValue(direct.codec, before), *example.origin_values(layout[-1:]))
    compatibility = example.execution_compatibility()
    store = LocalStateStore.create(tmp_path / "direct-pending-migration", definition_value, compatibility)
    first = store.initialize(command(0, None, kind="initialize"), values, source_bindings, compatibility=compatibility)
    implementation = definition(example, "authored-wrapped-direct-ci-owner")
    mapping = definition(example, "authored-direct-complete-state-mapping")
    target_binding = StateBinding.model_validate(
        host_binding.model_copy(update={"implementation": implementation, "state_codec": target_codec(example)})
    )
    target = ComponentBinding(
        ComponentCodec(target_binding, WrappedState[example.HostState]),
        partial(WrappedConsumer, example, 0),
        lambda value: None,
    )
    target_spec = ComponentSpec.model_validate(component.model_copy(update={"binding": target_binding}))
    target_definition = CollectionDefinition.model_validate(
        definition_value.model_copy(update={"components": (target_spec, definition_value.components[-1])})
    )
    current = ExecutionCompatibility.model_validate(
        compatibility.model_copy(
            update={"owner_implementations": (*compatibility.owner_implementations, implementation, mapping)}
        )
    )
    request = MigrationRequest(
        schema_version=1,
        commit=command(1, first.current_head, kind="codec_migration"),
        migration=mapping,
        target_definition=target_definition,
    )
    readers = MigrationBindings(tuple(binding.codec for binding in source_bindings), (target, layout[-1]))
    calls = []

    def migrate(prepared):
        calls.append("actual-known-pure-mapping")
        return (
            ComponentValue(target.codec, wrapped(example, prepared.component(direct.codec))),
            ComponentValue(layout[-1].codec, prepared.component(layout[-1].codec)),
        )

    receipt = store.migrate(request, readers, migrate, compatibility=current)
    actual = store.load((target, layout[-1]), compatibility=current).component(target.codec)
    assert actual.payload.to_json() == before.to_json()
    assert actual.payload.rng == before.rng
    assert len(actual.payload.pending) == 1
    retry = store.migrate(request, MigrationBindings((), ()), migrate, compatibility=current)
    assert retry.status == "retained"
    assert retry.current_head == receipt.current_head
    assert calls == ["actual-known-pure-mapping"]
    with pytest.raises(ValueError, match="compatibility"):
        store.migrate(request, readers, migrate, compatibility=compatibility)
    assert calls == ["actual-known-pure-mapping"]
