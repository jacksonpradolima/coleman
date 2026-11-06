"""A snapshot must satisfy its own reader before any durable acknowledgement."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import create_model

from coleman.core.base import CoreContract
from coleman.core.primitives import fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.components import ComponentBinding, ComponentCodec, ComponentValue
from coleman.state.contracts import AlgorithmState, StateBinding
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CollectionDefinition, ComponentSpec, ExecutionCompatibility
from tests.state.support import command

MIB = 1024**2


class TextState(CoreContract):
    """Complete plain-text state for the canonical encoded-byte boundary."""

    text: str


def nested_value(levels):
    result = 0
    for _ in range(levels):
        result = (result,)
    return result


def layout(example, levels):
    """Declare a concrete fixed-depth tuple codec, with null as its origin value."""
    annotation: Any = int
    for _ in range(levels):
        annotation = tuple[annotation]
    state_type = create_model(f"NestedState{levels}", __base__=CoreContract, value=(annotation | None, None))
    implementation = DefinitionBinding(
        reference=example.reference("authored-depth-owner"),
        definition_fingerprint="sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    state_binding = StateBinding(
        owner=example.reference("authored-depth-component"),
        implementation=implementation,
        state_codec=example.reference(f"authored-tuple-depth-{levels}"),
        configuration_fingerprint=fingerprint({"tuple_depth": levels, "origin": None}),
    )
    codec = ComponentCodec(state_binding, state_type)
    factories = []

    class Owner:
        def __init__(self):
            factories.append("fresh-owner")
            self.value = state_type.model_validate({"schema_version": 1, "value": None})

        def export_state(self):
            return state_type.model_validate(self.value)

        def restore_state(self, value):
            self.value = state_type.model_validate(value)

    binding = ComponentBinding(codec, Owner, lambda _value: None)
    definition = CollectionDefinition(
        schema_version=1,
        reference=example.reference("authored-depth-definition"),
        session=example.reference("authored-depth-session"),
        source_epoch=example.EPOCH,
        children=(),
        components=(ComponentSpec(binding=state_binding, child=None, roles=("algorithm",), frontier_profile="none"),),
    )
    compatibility = ExecutionCompatibility(
        schema_version=1,
        runtime=example.execution_compatibility().runtime,
        owner_implementations=(implementation,),
    )
    return SimpleNamespace(
        state_type=state_type,
        codec=codec,
        binding=binding,
        definition=definition,
        compatibility=compatibility,
        factories=factories,
    )


@pytest.mark.parametrize("levels", [0, 1, 62, 63])
def test_encoded_snapshot_at_or_below_64_containers_round_trips(state_example, levels):
    case = layout(state_example, levels)
    state = case.state_type(schema_version=1, value=nested_value(levels))
    encoded = case.codec.encode(state)
    assert case.codec.decode(encoded.content) == state
    assert case.factories == []


@pytest.mark.parametrize("levels", [64, 65])
def test_encoding_rejects_a_valid_model_that_its_own_depth_bounded_reader_cannot_read(state_example, levels):
    case = layout(state_example, levels)
    state = case.state_type(schema_version=1, value=nested_value(levels))
    with pytest.raises(ValueError, match="depth"):
        case.codec.encode(state)
    assert case.factories == []


@pytest.mark.parametrize("levels", [62, 63])
def test_depth_budget_includes_the_complete_algorithm_envelope(state_example, levels):
    case = layout(state_example, levels)
    payload = case.state_type(schema_version=1, value=nested_value(levels))
    assert case.codec.decode(case.codec.encode(payload).content) == payload
    state_type = AlgorithmState[case.state_type]
    state = state_type(schema_version=1, binding=case.codec.binding, payload=payload)
    codec = ComponentCodec(case.codec.binding, state_type)
    if levels == 62:
        encoded = codec.encode(state)
        assert codec.decode(encoded.content) == state
    else:
        with pytest.raises(ValueError, match="depth"):
            codec.encode(state)


@pytest.mark.parametrize("levels", [64, 65])
def test_snapshot_admission_cannot_acknowledge_an_unreadable_initial_collection(state_example, tmp_path, levels):
    case = layout(state_example, levels)
    store = LocalStateStore.create(tmp_path / "store", case.definition, case.compatibility)
    state = case.state_type(schema_version=1, value=nested_value(levels))
    request = command(0, None, kind="initialize")

    def snapshot_then_initialize():
        value = ComponentValue(case.codec, state)
        return store.initialize(request, (value,), (case.binding,), compatibility=case.compatibility)

    with pytest.raises(ValueError, match="depth"):
        snapshot_then_initialize()
    assert store.read_head() is None
    assert case.factories == []
    assert list((store.root / "blobs").iterdir()) == []
    assert list((store.root / "manifests").iterdir()) == []


def test_transition_to_an_unreadable_snapshot_keeps_the_acknowledged_origin(state_example, tmp_path):
    case = layout(state_example, 64)
    store = LocalStateStore.create(tmp_path / "store", case.definition, case.compatibility)
    origin = case.state_type(schema_version=1, value=None)
    value = ComponentValue(case.codec, origin)
    first = store.initialize(
        command(0, None, kind="initialize"), (value,), (case.binding,), compatibility=case.compatibility
    )
    deep = case.state_type(schema_version=1, value=nested_value(64))
    request = command(1, first.current_head)

    def transition(stage):
        stage.owner(case.binding).restore_state(deep)

    with pytest.raises(ValueError, match="depth"):
        store.commit_transition(request, (case.binding,), transition, compatibility=case.compatibility)
    assert store.read_head() == first.current_head
    assert store.load((case.binding,), compatibility=case.compatibility).component(case.codec) == origin


def test_the_64_container_boundary_is_readable_after_an_actual_acknowledged_transition(state_example, tmp_path):
    case = layout(state_example, 63)
    store = LocalStateStore.create(tmp_path / "store", case.definition, case.compatibility)
    origin = ComponentValue(case.codec, case.state_type(schema_version=1, value=None))
    first = store.initialize(
        command(0, None, kind="initialize"), (origin,), (case.binding,), compatibility=case.compatibility
    )
    next_state = case.state_type(schema_version=1, value=nested_value(63))
    second = store.commit_transition(
        command(1, first.current_head),
        (case.binding,),
        lambda stage: stage.owner(case.binding).restore_state(next_state),
        compatibility=case.compatibility,
    )
    assert second.current_head.generation == 1
    assert store.read_head() == second.current_head
    assert store.load((case.binding,), compatibility=case.compatibility).component(case.codec) == next_state


@pytest.mark.parametrize("text", ["x" * (16 * MIB), "é" * (8 * MIB)], ids=["ascii", "utf8"])
def test_encoding_checks_complete_utf8_bytes_before_creating_an_unreadable_snapshot(state_example, text):
    binding = layout(state_example, 0).codec.binding.model_copy(
        update={"state_codec": state_example.reference("authored-utf8-text-state")}
    )
    codec = ComponentCodec(binding, TextState)
    state = TextState(schema_version=1, text=text)
    with pytest.raises(ValueError, match="byte|bytes"):
        codec.encode(state)


def test_exact_16_mib_complete_canonical_snapshot_remains_supported(state_example):
    binding = layout(state_example, 0).codec.binding.model_copy(
        update={"state_codec": state_example.reference("authored-utf8-text-state")}
    )
    codec = ComponentCodec(binding, TextState)
    framing = '{"schema_version":1,"text":""}'
    assert TextState(schema_version=1, text="").to_json() == framing
    state = TextState(schema_version=1, text="x" * (16 * MIB - len(framing.encode("utf-8"))))
    encoded = codec.encode(state)
    assert encoded.byte_count == 16 * MIB
    assert codec.decode(encoded.content) == state


def test_byte_budget_includes_the_binding_and_algorithm_envelope(state_example):
    binding = layout(state_example, 0).codec.binding.model_copy(
        update={"state_codec": state_example.reference("authored-text-envelope")}
    )
    framing = '{"schema_version":1,"text":""}'
    payload = TextState(schema_version=1, text="x" * (16 * MIB - len(framing.encode("utf-8"))))
    assert ComponentCodec(binding, TextState).encode(payload).byte_count == 16 * MIB
    state_type = AlgorithmState[TextState]
    state = state_type(schema_version=1, binding=binding, payload=payload)
    codec = ComponentCodec(binding, state_type)
    with pytest.raises(ValueError, match="byte|bytes"):
        codec.encode(state)
