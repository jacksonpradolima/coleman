"""Untrusted manifests preserve complete inventories, ancestry and exact codecs."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from coleman.core.base import CoreContract
from coleman.core.primitives import fingerprint
from coleman.state import _prepare
from coleman.state.components import ComponentCodec, ComponentValue, PreparedCollection, StagedCollection
from coleman.state.contracts import AlgorithmState
from coleman.state.storage_contracts import (
    AcknowledgedHead,
    ChildFrontier,
    CollectionDefinition,
    CollectionManifest,
    CommitReceipt,
    MigrationReceipt,
    MigrationRequest,
    StorageLimits,
    semantic_frontiers_fingerprint,
)
from tests.state.support import command


@pytest.fixture(scope="module")
def manifest_case(state_example):
    example = state_example
    definition, bindings = example.collection_layout()
    values = example.origin_values(bindings)
    compatibility = example.execution_compatibility()
    prepared = _prepare.build(
        definition,
        compatibility,
        command(0, None, kind="initialize"),
        values,
        bindings,
        StorageLimits(schema_version=1),
    )
    manifest = prepared.manifest
    head = AcknowledgedHead(
        schema_version=1,
        session=definition.session,
        source_epoch=definition.source_epoch,
        definition_fingerprint=definition.fingerprint,
        generation=0,
        manifest_fingerprint=manifest.fingerprint,
        intent_fingerprint=manifest.intent.fingerprint,
    )
    migration = MigrationReceipt(
        schema_version=1,
        migration=compatibility.owner_implementations[0],
        source_head=head,
        source_definition_fingerprint=definition.fingerprint,
        target_definition_fingerprint=definition.fingerprint,
        preserved_frontiers_fingerprint=semantic_frontiers_fingerprint(manifest.frontiers),
    )
    successor = CollectionManifest.model_validate(
        manifest.model_copy(update={"predecessor": head, "intent": command(1, head).intent})
    )
    migrated = CollectionManifest.model_validate(
        successor.model_copy(update={"intent": command(1, head, kind="codec_migration").intent, "migration": migration})
    )
    return SimpleNamespace(
        example=example,
        definition=definition,
        bindings=bindings,
        values=values,
        prepared=prepared,
        manifest=manifest,
        head=head,
        migration=migration,
        successor=successor,
        migrated=migrated,
    )


def read_manifest(value):
    """Exercise the actual wire reader, including nested revalidation."""
    return CollectionManifest.from_json(value.model_dump_json())


@pytest.mark.parametrize("change", ["missing", "duplicate", "codec", "implementation", "frontier", "epoch"])
def test_manifest_wire_rejects_partial_or_reinterpreted_complete_inventory(manifest_case, change):
    case = manifest_case
    value = case.manifest
    if change == "missing":
        value = value.model_copy(update={"components": value.components[:-1]})
    elif change == "duplicate":
        value = value.model_copy(update={"components": (*value.components, value.components[0])})
    elif change == "codec":
        blob = value.components[0]
        binding = blob.binding.model_copy(update={"state_codec": case.example.reference("other-codec")})
        value = value.model_copy(
            update={"components": (blob.model_copy(update={"binding": binding}), *value.components[1:])}
        )
    elif change == "implementation":
        implementation = value.compatibility.owner_implementations[0]
        other = implementation.model_copy(update={"reference": case.example.reference("other-implementation")})
        compatibility = value.compatibility.model_copy(update={"owner_implementations": (other,)})
        value = value.model_copy(update={"compatibility": compatibility})
    elif change == "frontier":
        value = value.model_copy(update={"frontiers": ()})
    else:
        frontier = value.frontiers[0].model_copy(update={"source_epoch": case.example.reference("another-epoch")})
        value = value.model_copy(update={"frontiers": (frontier,)})
    with pytest.raises(ValueError, match="inventory|implementation|frontier|epoch"):
        read_manifest(value)
    assert read_manifest(case.manifest) == case.manifest


@pytest.mark.parametrize("change", ["generation", "kind", "migration"])
def test_virgin_manifest_cannot_claim_a_successor_or_mapping(manifest_case, change):
    case = manifest_case
    value = case.manifest
    updates = {
        "generation": {"intent": command(1, None, kind="initialize").intent},
        "kind": {"intent": command(0, None).intent},
        "migration": {"migration": case.migration},
    }
    with pytest.raises(ValueError, match="generation-zero"):
        read_manifest(value.model_copy(update=updates[change]))


@pytest.mark.parametrize("change", ["generation", "initialize", "session", "epoch", "definition", "mapping"])
def test_successor_wire_binds_exact_generation_session_epoch_and_definition(manifest_case, change):
    case = manifest_case
    value = case.successor
    if change == "generation":
        value = value.model_copy(update={"intent": command(2, case.head).intent})
    elif change == "initialize":
        value = value.model_copy(update={"intent": command(1, case.head, kind="initialize").intent})
    elif change in {"session", "epoch", "definition"}:
        field = {"session": "session", "epoch": "source_epoch", "definition": "reference"}[change]
        definition = value.definition.model_copy(update={field: case.example.reference("different-context")})
        updates = {"definition": definition}
        if change == "epoch":
            updates["frontiers"] = tuple(
                row.model_copy(update={"source_epoch": definition.source_epoch}) for row in value.frontiers
            )
        value = value.model_copy(update=updates)
    else:
        value = value.model_copy(update={"migration": case.migration})
    with pytest.raises(ValueError, match="generation|session|epoch|definition"):
        read_manifest(value)
    assert read_manifest(case.successor).predecessor == case.head


@pytest.mark.parametrize("change", ["absent", "head", "source", "target", "semantics"])
def test_migration_wire_requires_exact_source_target_and_preserved_frontiers(manifest_case, change):
    case = manifest_case
    migration = case.migration
    if change == "absent":
        migration = None
    elif change == "head":
        migration = migration.model_copy(update={"source_head": case.head.model_copy(update={"generation": 1})})
    else:
        field = {
            "source": "source_definition_fingerprint",
            "target": "target_definition_fingerprint",
            "semantics": "preserved_frontiers_fingerprint",
        }[change]
        migration = migration.model_copy(update={field: fingerprint("unrelated")})
    with pytest.raises(ValueError, match="source|target|frontiers"):
        read_manifest(case.migrated.model_copy(update={"migration": migration}))
    assert read_manifest(case.migrated).migration == case.migration


@pytest.mark.parametrize("change", ["session", "epoch", "future", "new", "same_generation"])
def test_retry_receipt_never_relabels_a_different_or_future_head(manifest_case, change):
    case = manifest_case
    accepted = case.head
    current = case.head
    status = "retained"
    if change in {"session", "epoch"}:
        field = "session" if change == "session" else "source_epoch"
        current = current.model_copy(update={field: case.example.reference("other-scope")})
    elif change == "future":
        accepted = accepted.model_copy(update={"generation": 1})
    elif change == "new":
        current = current.model_copy(update={"generation": 1})
        status = "new"
    else:
        current = current.model_copy(update={"manifest_fingerprint": fingerprint("other-manifest")})
    with pytest.raises(ValueError, match="session|epoch|advance|identical"):
        CommitReceipt(schema_version=1, status=status, accepted_head=accepted, current_head=current)
    legitimate = CommitReceipt(
        schema_version=1,
        status="retained",
        accepted_head=case.head,
        current_head=case.head.model_copy(update={"generation": 1}),
    )
    assert legitimate.accepted_head.generation == 0
    assert legitimate.current_head.generation == 1


@pytest.mark.parametrize("kind", ["initialize", "transition", "seal"])
def test_migration_request_cannot_execute_under_another_command_kind(manifest_case, kind):
    case = manifest_case
    request = command(0, None, kind=kind) if kind == "initialize" else command(1, case.head, kind=kind)
    with pytest.raises(ValueError, match="acknowledged source"):
        MigrationRequest(
            schema_version=1,
            commit=request,
            migration=case.migration.migration,
            target_definition=case.definition,
        )


@pytest.mark.parametrize("change", ["updates", "floor", "only_information", "only_application", "reversed"])
def test_child_frontier_refuses_impossible_counts_and_partial_or_reversed_clocks(manifest_case, change):
    case = manifest_case
    frontier = case.manifest.frontiers[0]
    changes = {
        "updates": {"update_count": 1},
        "floor": {"sealed_floor": 1},
        "only_information": {"eligible_through": case.example.instant(1)},
        "only_application": {"apply_at": case.example.instant(1)},
        "reversed": {"eligible_through": case.example.instant(2), "apply_at": case.example.instant(1)},
    }
    with pytest.raises(ValueError, match="counts|clocks|clock"):
        ChildFrontier.model_validate(frontier.model_copy(update=changes[change]))


@pytest.mark.parametrize("shape", ["mutable", "too_many", "same_id", "other_version"])
def test_migration_semantics_digest_rejects_ambiguous_or_unbounded_frontiers(manifest_case, shape):
    frontier = manifest_case.manifest.frontiers[0]
    other = frontier.model_copy(update={"child": frontier.child.model_copy(update={"version": "v2"})})
    values = {
        "mutable": [frontier],
        "too_many": (frontier,) * 33,
        "same_id": (frontier, frontier),
        "other_version": (frontier, other),
    }
    with pytest.raises(ValueError, match="bounded tuple|unique"):
        semantic_frontiers_fingerprint(cast(Any, values[shape]))


def test_codec_migration_digest_ignores_only_the_three_declared_reencoding_fields(manifest_case):
    case = manifest_case
    frontier = case.manifest.frontiers[0]
    changed_codec = frontier.model_copy(
        update={
            "state_fingerprint": fingerprint("new-bytes"),
            "host_fingerprint": fingerprint("mapped-host"),
            "state_codec": case.example.reference("known-new-codec"),
        }
    )
    before = semantic_frontiers_fingerprint((frontier,))
    assert semantic_frontiers_fingerprint((changed_codec,)) == before
    for field in ("policy_fingerprint", "retention_fingerprint", "causal_commitment"):
        changed_meaning = changed_codec.model_copy(update={field: fingerprint("changed-semantics")})
        assert semantic_frontiers_fingerprint((changed_meaning,)) != before


@pytest.mark.parametrize(
    "updates",
    [
        {"maximum_component_bytes": 128, "maximum_collection_bytes": 64},
        {"maximum_orphan_bytes": 1},
        {"maximum_orphan_entries": 1},
    ],
)
def test_capacity_is_feasible_for_a_complete_candidate_before_storage(updates):
    with pytest.raises(ValueError, match="ceiling"):
        StorageLimits(schema_version=1, **updates)


def test_one_child_cannot_declare_two_frontier_owners(manifest_case):
    case = manifest_case
    original = case.definition.components[0]
    binding = original.binding.model_copy(update={"owner": case.example.reference("another-frontier-owner")})
    other = original.model_copy(update={"binding": binding})
    with pytest.raises(ValueError, match="frontier|replay"):
        CollectionDefinition.model_validate(
            case.definition.model_copy(update={"components": (*case.definition.components, other)})
        )


@pytest.mark.parametrize("operation", ["bindings", "codecs", "values"])
def test_bad_inventory_entry_rejects_before_any_factory_or_decoder(manifest_case, operation):
    case = manifest_case
    invalid = cast(Any, (object(),))
    with pytest.raises(TypeError, match="concrete"):
        if operation == "bindings":
            _prepare.bound_codecs(invalid)
        elif operation == "codecs":
            _prepare.inventory(case.definition, invalid)
        else:
            _prepare.encoded(case.definition, invalid, StorageLimits(schema_version=1))


def test_concrete_inventory_cannot_omit_or_alias_a_codec(manifest_case):
    case = manifest_case
    for codecs in ((case.bindings[0].codec,), (case.bindings[0].codec,) * 2):
        with pytest.raises(ValueError, match="exact complete"):
            _prepare.inventory(case.definition, codecs)
    assert _prepare.inventory(case.definition, tuple(row.codec for row in reversed(case.bindings))) == tuple(
        row.codec for row in case.bindings
    )


def test_prepared_and_staged_lookups_require_the_exact_requested_interpretation(manifest_case):
    case = manifest_case
    original = case.bindings[-1]
    other_binding = original.codec.binding.model_copy(update={"state_codec": case.example.reference("other-reader")})
    other_codec = ComponentCodec(other_binding, original.codec.state_type)
    with pytest.raises(ValueError, match="absent or incompatible"):
        case.prepared.component(other_codec)
    with pytest.raises(ValueError, match="absent or incompatible"):
        PreparedCollection(case.manifest).component(original.codec)
    other_type = ComponentCodec(original.codec.binding, AlgorithmState[case.example.DomainState])
    with pytest.raises(ValueError, match="absent or incompatible"):
        case.prepared.component(other_type)
    owner = original.new_owner()
    staged = StagedCollection(((original, owner),))
    assert staged.owner(original) is owner
    with pytest.raises(ValueError, match="exact owner"):
        staged.owner(replace(original))


def test_canonical_component_decoder_refuses_mutable_or_reformatted_bytes(manifest_case):
    value = manifest_case.values[-1]
    raw = value.codec.encode(value.value).content
    with pytest.raises(TypeError, match="immutable bytes"):
        value.codec.decode(bytearray(raw))
    with pytest.raises(ValueError, match="canonical"):
        value.codec.decode(raw + b"\n")
    assert value.codec.decode(raw) == value.value


def test_bare_generic_or_embedded_binding_cannot_erase_an_owner_interpretation(manifest_case):
    case = manifest_case
    binding = case.bindings[-1].codec.binding
    with pytest.raises(TypeError, match="generic parameters"):
        ComponentCodec(binding, AlgorithmState)
    with pytest.raises(ValueError, match="concrete payload"):
        AlgorithmState[CoreContract](schema_version=1, binding=binding, payload=CoreContract(schema_version=1))
    state = AlgorithmState[case.example.DomainState](schema_version=1, binding=binding, payload=case.values[-1].value)
    other_binding = binding.model_copy(update={"state_codec": case.example.reference("other-envelope-codec")})
    codec = ComponentCodec(other_binding, type(state))
    with pytest.raises(ValueError, match="embedded binding"):
        ComponentValue(codec, state)


@pytest.mark.parametrize("change", ["concrete_type", "invented", "not_replay", "dishonest"])
def test_frontier_inspectors_cannot_change_concrete_state_or_invent_delivery(manifest_case, change):
    case = manifest_case
    bindings = list(case.bindings)
    values = list(case.values)
    frontier = case.manifest.frontiers[0]
    if change == "concrete_type":
        codec = ComponentCodec(bindings[0].codec.binding, case.example.DomainState)
        bindings[0] = replace(bindings[0], codec=codec)
    elif change == "invented":
        bindings[-1] = replace(bindings[-1], frontier=lambda _state: frontier)
    elif change == "not_replay":
        codec = ComponentCodec(bindings[0].codec.binding, case.example.DomainState)
        bindings[0] = replace(bindings[0], codec=codec, frontier=lambda _state: frontier)
        values[0] = ComponentValue(codec, case.example.DomainState(schema_version=1))
    else:
        forged = frontier.model_copy(update={"host_fingerprint": fingerprint("unrelated-host")})
        bindings[0] = replace(bindings[0], frontier=lambda _state: forged)
    with pytest.raises(ValueError, match="concrete state type|invent|RetainedContinuation|Inspector differs"):
        _prepare.frontiers(case.definition, tuple(values), tuple(bindings))
    assert _prepare.frontiers(case.definition, case.values, case.bindings) == case.manifest.frontiers


def test_manifest_frontier_must_match_the_actual_decoded_component_before_factories(manifest_case):
    case = manifest_case
    forged = case.manifest.frontiers[0].model_copy(update={"host_fingerprint": fingerprint("forged-host")})
    manifest = CollectionManifest.model_validate(case.manifest.model_copy(update={"frontiers": (forged,)}))
    with pytest.raises(ValueError, match="Actual decoded frontiers"):
        _prepare.verify(PreparedCollection(manifest, case.values), case.bindings)


def test_manifest_byte_admission_rejects_before_publication_or_factory_use(manifest_case):
    case = manifest_case

    def forbidden_owner():
        raise AssertionError("Preparing a manifest must not construct an owner")

    bindings = tuple(replace(binding, new_owner=forbidden_owner) for binding in case.bindings)
    with pytest.raises(ValueError, match="manifest exceeds"):
        _prepare.build(
            case.definition,
            case.manifest.compatibility,
            command(0, None, kind="initialize"),
            case.values,
            bindings,
            StorageLimits(schema_version=1, maximum_manifest_bytes=1),
        )


def test_fresh_factories_cannot_alias_one_owner_across_components(manifest_case):
    case = manifest_case
    shared = case.example.Domain()
    before = shared.export_state()
    bindings = tuple(replace(binding, new_owner=lambda: shared) for binding in case.bindings)
    with pytest.raises(ValueError, match="distinct fresh owner"):
        _prepare.stage(case.prepared, bindings)
    assert shared.export_state() == before


def test_restore_must_reexport_path_state_instead_of_silently_using_factory_defaults(manifest_case):
    case = manifest_case
    original = case.example.Domain()
    original.apply(case.example.reference("ci-plan-b"))
    actual = original.export_state()
    domain_binding = case.bindings[-1]
    values = (*case.values[:-1], ComponentValue(domain_binding.codec, actual))
    prepared = _prepare.build(
        case.definition,
        case.manifest.compatibility,
        command(0, None, kind="initialize"),
        values,
        case.bindings,
        StorageLimits(schema_version=1),
    )

    class DiscardsRestoredPath(case.example.Domain):
        def restore_state(self, state, /):
            assert state == actual

    bindings = (*case.bindings[:-1], replace(domain_binding, new_owner=DiscardsRestoredPath))
    with pytest.raises(ValueError, match="exact complete state"):
        _prepare.stage(prepared, bindings)
    assert original.export_state() == actual
    assert actual.book.count_b == 1
