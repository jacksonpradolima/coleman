"""Declared codec data must survive its actual reader before acknowledgement."""

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import PrivateAttr, StrictBool, StrictInt, computed_field, field_serializer, field_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.primitives import fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.components import ComponentBinding, ComponentCodec, ComponentValue
from coleman.state.contracts import StateBinding
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CollectionDefinition, ComponentSpec, ExecutionCompatibility
from tests.state.support import command


class ComputedState(CoreContract):
    """A standard derived serialization field is not accepted by its own reader."""

    value: StrictInt

    @computed_field
    @property
    def doubled(self) -> int:
        return 2 * self.value


class ComputedLeaf(CoreImmutable):
    value: StrictInt

    @computed_field
    @property
    def doubled(self) -> int:
        return 2 * self.value


class NestedComputedState(CoreContract):
    leaf: ComputedLeaf


class OneWayState(CoreContract):
    """A normal JSON serializer changes a strict integer into unreadable text."""

    value: StrictInt

    @field_serializer("value", when_used="json")
    def serialize_value(self, value):
        return str(value)


class LossyFloatState(CoreContract):
    """Readable JSON can still discard required mathematical state."""

    value: float

    @field_serializer("value", when_used="json")
    def serialize_value(self, value):
        return float(int(value))


class BoolState(CoreContract):
    value: StrictBool | StrictInt

    @field_serializer("value", when_used="json")
    def serialize_value(self, value):
        return int(value)


class BoolLeaf(CoreImmutable):
    value: StrictBool | StrictInt

    @field_serializer("value", when_used="json")
    def serialize_value(self, value):
        return int(value)


class NestedBoolState(CoreContract):
    leaf: BoolLeaf


class TupleBoolState(CoreContract):
    values: tuple[StrictBool | StrictInt, ...]

    @field_serializer("values", when_used="json")
    def serialize_values(self, values):
        return tuple(int(value) for value in values)


class ListBoolState(CoreContract):
    values: list[StrictBool | StrictInt]

    @field_serializer("values", when_used="json")
    def serialize_values(self, values):
        return [int(value) for value in values]


class MapBoolState(CoreContract):
    values: dict[str, StrictBool | StrictInt]

    @field_serializer("values", when_used="json")
    def serialize_values(self, values):
        return {key: int(value) for key, value in values.items()}


class SetBoolState(CoreContract):
    values: frozenset[StrictBool | StrictInt]

    @field_serializer("values", when_used="json")
    def serialize_values(self, values):
        return [int(value) for value in values]


class LossyDecimalState(CoreContract):
    """A numeric equality must not erase declared Decimal precision."""

    value: Decimal

    @field_serializer("value", when_used="json")
    def serialize_value(self, value):
        return str(value.normalize())


class ExactDecimalState(CoreContract):
    value: Decimal


class ComplexState(CoreContract):
    value: complex


@dataclass(frozen=True)
class DeclaredData:
    value: StrictBool | StrictInt


class DataclassState(CoreContract):
    """A declared record must not use dataclass True==1 equality as its proof."""

    leaf: DeclaredData

    @field_serializer("leaf", when_used="json")
    def serialize_leaf(self, leaf):
        return {"value": int(leaf.value)}


class ReversibleState(CoreContract):
    """A paired custom JSON serializer/reader preserves its strict signed integer."""

    value: StrictInt

    @field_validator("value", mode="before")
    @classmethod
    def read_value(cls, value):
        if isinstance(value, str) and value.startswith("integer:"):
            return int(value.removeprefix("integer:"))
        return value

    @field_serializer("value", when_used="json")
    def serialize_value(self, value):
        return f"integer:{value}"


class ParserCacheState(CoreContract):
    """Disposable parser memo identity has no effect on any mathematical output."""

    value: StrictInt
    _memo: object = PrivateAttr(default_factory=object)


class NonIdempotentState(CoreContract):
    """A deterministic but non-idempotent normalizer cannot satisfy this codec."""

    value: StrictInt

    @field_validator("value")
    @classmethod
    def change_sign(cls, value):
        return -value


def layout(example, state_type):
    """Select one complete caller-owned codec and actual fresh state owner."""
    implementation = DefinitionBinding(
        reference=example.reference("authored-symmetry-owner"),
        definition_fingerprint="sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    state_binding = StateBinding(
        owner=example.reference("authored-symmetry-component"),
        implementation=implementation,
        state_codec=example.reference("authored-symmetry-" + state_type.__name__),
        configuration_fingerprint=fingerprint({"state_type": state_type.__name__}),
    )
    codec = ComponentCodec(state_binding, state_type)
    factories = []

    class Owner:
        def __init__(self):
            factories.append("fresh-owner")
            self.value = None

        def export_state(self):
            return state_type.model_validate(self.value)

        def restore_state(self, state):
            self.value = state_type.model_validate(state)

    binding = ComponentBinding(codec, Owner, lambda _state: None)
    definition = CollectionDefinition(
        schema_version=1,
        reference=example.reference("authored-symmetry-definition"),
        session=example.reference("authored-symmetry-session"),
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
        codec=codec,
        binding=binding,
        definition=definition,
        compatibility=compatibility,
        factories=factories,
    )


@pytest.mark.parametrize(
    "state",
    [
        ComputedState(schema_version=1, value=2),
        NestedComputedState(schema_version=1, leaf=ComputedLeaf(value=2)),
        OneWayState(schema_version=1, value=2),
    ],
    ids=["computed", "nested-computed", "strict-json-serializer"],
)
def test_encoder_refuses_models_whose_actual_selected_reader_rejects_their_serialization(state_example, state):
    case = layout(state_example, type(state))
    with pytest.raises(ValueError):
        case.codec.encode(state)
    assert case.factories == []


@pytest.mark.parametrize(
    "state",
    [
        LossyFloatState(schema_version=1, value=0.75),
        LossyFloatState(schema_version=1, value=-0.0),
        BoolState(schema_version=1, value=True),
        NestedBoolState(schema_version=1, leaf=BoolLeaf(value=True)),
        TupleBoolState(schema_version=1, values=(True, 2)),
        ListBoolState(schema_version=1, values=[True, 2]),
        MapBoolState(schema_version=1, values={"hold": True, "count": 2}),
        SetBoolState(schema_version=1, values=frozenset({True})),
        LossyDecimalState(schema_version=1, value=Decimal("2.00")),
        DataclassState(schema_version=1, leaf=DeclaredData(True)),
    ],
    ids=[
        "float-loss",
        "negative-zero",
        "bool-to-one",
        "nested-model",
        "tuple",
        "list",
        "mapping",
        "set",
        "decimal-quantum",
        "dataclass",
    ],
)
def test_readable_json_must_preserve_exact_declared_values_and_scalar_types(state_example, state):
    case = layout(state_example, type(state))
    with pytest.raises(ValueError):
        case.codec.encode(state)
    assert case.factories == []


@pytest.mark.parametrize("value", [0, -3, 7])
def test_reversible_custom_json_codec_remains_supported_for_zero_and_signed_values(state_example, value):
    case = layout(state_example, ReversibleState)
    state = ReversibleState(schema_version=1, value=value)
    encoded = case.codec.encode(state)
    decoded = case.codec.decode(encoded.content)
    assert decoded.value == value
    assert type(decoded.value) is int
    assert case.codec.encode(decoded) == encoded
    assert case.factories == []


def test_disposable_private_parser_cache_is_not_mistaken_for_required_algorithm_state(state_example):
    case = layout(state_example, ParserCacheState)
    state = ParserCacheState(schema_version=1, value=2)
    decoded = case.codec.decode(case.codec.encode(state).content)
    assert decoded.value == state.value
    assert decoded._memo is not state._memo
    assert decoded.to_json() == state.to_json()


def test_encoder_also_requires_its_own_decoder_canonical_check_to_succeed(state_example):
    case = layout(state_example, NonIdempotentState)
    state = NonIdempotentState(schema_version=1, value=1)
    with pytest.raises(ValueError):
        case.codec.encode(state)
    assert case.factories == []


@pytest.mark.parametrize("value", [Decimal("2.00"), Decimal("-0.00")])
def test_declared_decimal_digits_quantum_and_signed_zero_round_trip_exactly(state_example, value):
    case = layout(state_example, ExactDecimalState)
    state = ExactDecimalState(schema_version=1, value=value)
    encoded = case.codec.encode(state)
    decoded = case.codec.decode(encoded.content)
    assert decoded.value.as_tuple() == value.as_tuple()
    assert case.codec.encode(decoded) == encoded


def test_reversible_declared_dataclass_fields_remain_supported(state_example):
    case = layout(state_example, DataclassState)
    state = DataclassState(schema_version=1, leaf=DeclaredData(2))
    encoded = case.codec.encode(state)
    decoded = case.codec.decode(encoded.content)
    assert decoded.leaf.value == 2
    assert type(decoded.leaf.value) is int
    assert case.codec.encode(decoded) == encoded


def test_finite_builtin_complex_state_round_trips_both_binary64_coordinates(state_example):
    case = layout(state_example, ComplexState)
    value = complex(1.25, -2.5)
    state = ComplexState(schema_version=1, value=value)
    encoded = case.codec.encode(state)
    decoded = case.codec.decode(encoded.content)
    assert float(decoded.value.real).hex() == "0x1.4000000000000p+0"
    assert float(decoded.value.imag).hex() == "-0x1.4000000000000p+1"
    assert case.codec.encode(decoded) == encoded


@pytest.mark.parametrize("value", [complex(float("inf"), 2.5), complex(1.25, float("inf"))], ids=["real", "imaginary"])
def test_complex_wire_strings_cannot_hide_a_nonfinite_declared_coordinate(state_example, value):
    case = layout(state_example, ComplexState)
    state = ComplexState(schema_version=1, value=value)
    payload = state.to_json().encode("utf-8")
    with pytest.raises(ValueError):
        case.codec.encode(state)
    with pytest.raises(ValueError):
        case.codec.decode(payload)
    assert case.factories == []


@pytest.mark.parametrize(
    "state",
    [ComputedState(schema_version=1, value=2), LossyFloatState(schema_version=1, value=0.75)],
    ids=["unreadable", "readable-but-lossy"],
)
def test_initialization_cannot_acknowledge_an_unreadable_or_lossy_complete_owner(state_example, tmp_path, state):
    case = layout(state_example, type(state))
    store = LocalStateStore.create(tmp_path / "store", case.definition, case.compatibility)
    request = command(0, None, kind="initialize")

    def initialize_snapshot():
        value = ComponentValue(case.codec, state)
        return store.initialize(request, (value,), (case.binding,), compatibility=case.compatibility)

    with pytest.raises(ValueError):
        initialize_snapshot()
    assert store.read_head() is None
    assert case.factories == []
    assert list((store.root / "blobs").iterdir()) == []
    assert list((store.root / "manifests").iterdir()) == []


def test_lossy_transition_retains_the_actual_acknowledged_readable_origin(state_example, tmp_path):
    case = layout(state_example, LossyFloatState)
    store = LocalStateStore.create(tmp_path / "store", case.definition, case.compatibility)
    origin = LossyFloatState(schema_version=1, value=0.0)
    first = store.initialize(
        command(0, None, kind="initialize"),
        (ComponentValue(case.codec, origin),),
        (case.binding,),
        compatibility=case.compatibility,
    )
    changed = LossyFloatState(schema_version=1, value=0.75)
    request = command(1, first.current_head)
    callbacks = []

    def transition(stage):
        callbacks.append("actual-transition")
        stage.owner(case.binding).restore_state(changed)

    with pytest.raises(ValueError):
        store.commit_transition(request, (case.binding,), transition, compatibility=case.compatibility)
    assert callbacks == ["actual-transition"]
    assert store.read_head() == first.current_head
    assert store.load((case.binding,), compatibility=case.compatibility).component(case.codec).value == 0.0
