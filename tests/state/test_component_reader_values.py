"""Own-byte admission retains selected readers, exact state and custom hooks."""

import json
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from itertools import count
from pathlib import Path
from types import MethodType
from typing import Any, ClassVar, Self

import pytest
from pydantic import Field, StrictInt

from coleman.core.base import CoreContract
from coleman.core.identity import Reference
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.components import ComponentBinding, ComponentCodec, ComponentValue
from coleman.state.contracts import StateBinding
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CollectionDefinition, ComponentSpec, ExecutionCompatibility
from tests.state.support import command


class DeclaredState(CoreContract):
    amount: Decimal
    nullable: str | None
    signed: StrictInt
    zero: float


class CountedState(CoreContract):
    value: StrictInt
    events: ClassVar[list[str]] = []
    instances: ClassVar[list[Any]] = []

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        cls.events.append("reader")
        value = super().from_json(payload)
        cls.instances.append(value)
        return value

    def to_json(self) -> str:
        type(self).events.append("writer")
        return super().to_json()


class NestedState(CoreContract):
    value: Any


def codec_for(state_type):
    """Declare independent toy identity without executing any owner or source."""
    binding = StateBinding(
        owner=Reference(id="authored-reader-owner", version="1"),
        implementation=DefinitionBinding(
            reference=Reference(id="authored-reader-implementation", version="1"),
            definition_fingerprint="sha256:" + "1" * 64,
        ),
        state_codec=Reference(id="authored-reader-" + state_type.__name__, version="1"),
        configuration_fingerprint="sha256:" + "2" * 64,
    )
    return ComponentCodec(binding, state_type)


def reset_counted():
    CountedState.events = []
    CountedState.instances = []


@pytest.mark.parametrize("signed", [0, -3, 7])
def test_literal_golden_bytes_preserve_signed_zero_decimal_quantum_null_and_permitted_negatives(signed):
    expected = (
        '{"amount":"2.00","nullable":null,"schema_version":1,"signed":' + str(signed) + ',"zero":-0.0}'
    ).encode()
    codec = codec_for(DeclaredState)
    result = ComponentValue.from_bytes(codec, expected)
    assert result.codec is codec and type(result.value) is DeclaredState
    assert result.value.signed == signed and type(result.value.signed) is int
    assert result.value.nullable is None
    assert result.value.zero.hex() == "-0x0.0p+0"
    assert result.value.amount.as_tuple() == Decimal("2.00").as_tuple()
    assert codec.encode(result.value).content == expected
    assert codec.encode(result.value).fingerprint == "sha256:" + sha256(expected).hexdigest()


def test_stock_route_runs_both_actual_selected_readers_and_wraps_the_first_exact_value():
    reset_counted()
    codec = codec_for(CountedState)
    result = ComponentValue.from_bytes(codec, b'{"schema_version":1,"value":7}')
    assert CountedState.events == ["reader", "writer", "reader", "writer"]
    assert len(CountedState.instances) == 2
    assert result.value is CountedState.instances[0]
    assert result.value is not CountedState.instances[1]
    assert result.value.value == 7


def test_ordinary_constructor_still_performs_its_original_independent_encode_admission():
    reset_counted()
    codec = codec_for(CountedState)
    result = ComponentValue(codec, codec.decode(b'{"schema_version":1,"value":7}'))
    assert result.value.value == 7
    assert CountedState.events == ["reader", "writer", "writer", "reader", "writer"]


@pytest.mark.parametrize("route", ["factory", "ordinary"])
def test_reader_default_excluded_from_json_must_still_repeat_exactly(route):
    numbers = count()
    defaults = []

    def next_value():
        value = next(numbers)
        defaults.append(value)
        return value

    class ChangingDefault(CoreContract):
        value: StrictInt
        excluded: StrictInt = Field(default_factory=next_value, exclude=True)

    codec = codec_for(ChangingDefault)
    payload = b'{"schema_version":1,"value":7}'
    with pytest.raises(ValueError, match="recover the exact complete declared state"):
        if route == "factory":
            ComponentValue.from_bytes(codec, payload)
        else:
            ComponentValue(codec, codec.decode(payload))
    assert defaults == [0, 1]


@pytest.mark.parametrize("route", ["factory", "ordinary"])
def test_first_full_signature_is_captured_before_a_second_reader_can_mutate_that_object(route):
    class AliasedReader(CoreContract):
        value: StrictInt
        excluded: StrictInt = Field(default=0, exclude=True)
        first: ClassVar[Any] = None
        reads: ClassVar[int] = 0

        @classmethod
        def from_json(cls, payload: str | bytes) -> Self:
            value = super().from_json(payload)
            cls.reads += 1
            if cls.first is None:
                cls.first = value
            else:
                object.__setattr__(cls.first, "excluded", 1)
                object.__setattr__(value, "excluded", 1)
            return value

    codec = codec_for(AliasedReader)
    payload = b'{"schema_version":1,"value":7}'
    with pytest.raises(ValueError, match="recover the exact complete declared state"):
        if route == "factory":
            ComponentValue.from_bytes(codec, payload)
        else:
            ComponentValue(codec, codec.decode(payload))
    assert AliasedReader.reads == 2
    assert AliasedReader.first.excluded == 1


@pytest.mark.parametrize("placement", ["class", "instance"])
@pytest.mark.parametrize("name", ["decode", "encode", "_canonical_state"])
def test_codec_method_overrides_use_the_exact_old_constructor_sequence(monkeypatch, placement, name):
    codec = codec_for(CountedState)
    original = getattr(ComponentCodec, name)
    calls = []

    def observed(self, *arguments, **keywords):
        calls.append(name)
        return original(self, *arguments, **keywords)

    if placement == "class":
        monkeypatch.setattr(ComponentCodec, name, observed)
    else:
        object.__setattr__(codec, name, MethodType(observed, codec))
    reset_counted()
    old = ComponentValue(codec, codec.decode(b'{"schema_version":1,"value":7}'))
    expected_calls, expected_events = tuple(calls), tuple(CountedState.events)
    calls.clear()
    reset_counted()
    actual = ComponentValue.from_bytes(codec, b'{"schema_version":1,"value":7}')
    assert actual.value.value == old.value.value == 7
    assert tuple(calls) == expected_calls
    assert tuple(CountedState.events) == expected_events


def test_instance_shadow_of_even_a_stock_bound_method_preserves_constructor_fallback():
    codec = codec_for(CountedState)
    object.__setattr__(codec, "encode", MethodType(ComponentCodec.encode, codec))
    reset_counted()
    result = ComponentValue.from_bytes(codec, b'{"schema_version":1,"value":7}')
    assert result.value.value == 7
    assert CountedState.events == ["reader", "writer", "writer", "reader", "writer"]


@pytest.mark.parametrize("field", ["binding", "state_type"])
def test_codec_field_descriptors_retain_the_original_constructor_access_sequence(monkeypatch, field):
    codec = codec_for(CountedState)
    calls = []

    def read(self):
        calls.append(field)
        return object.__getattribute__(self, "__dict__")[field]

    monkeypatch.setattr(ComponentCodec, field, property(read), raising=False)
    reset_counted()
    old = ComponentValue(codec, codec.decode(b'{"schema_version":1,"value":7}'))
    expected_calls, expected_events = tuple(calls), tuple(CountedState.events)
    calls.clear()
    reset_counted()
    result = ComponentValue.from_bytes(codec, b'{"schema_version":1,"value":7}')
    assert result.value.value == old.value.value == 7
    assert tuple(calls) == expected_calls
    assert tuple(CountedState.events) == expected_events


@pytest.mark.parametrize("hook", ["__init__", "__post_init__"])
def test_monkeypatched_value_construction_hooks_are_never_bypassed(monkeypatch, hook):
    calls = []
    original = getattr(ComponentValue, hook)

    def observed(self, *arguments, **keywords):
        calls.append(hook)
        return original(self, *arguments, **keywords)

    monkeypatch.setattr(ComponentValue, hook, observed)
    codec = codec_for(CountedState)
    reset_counted()
    value = ComponentValue.from_bytes(codec, b'{"schema_version":1,"value":7}')
    assert value.value.value == 7 and calls == [hook]
    assert CountedState.events == ["reader", "writer", "writer", "reader", "writer"]


@pytest.mark.parametrize("mode", ["observed", "none"])
def test_monkeypatched_new_hook_is_preserved_without_leaking_cpython_allocation_slots(mode):
    # Deleting a patched __new__ descriptor can retain a CPython tp_new wrapper;
    # isolate the actual live-class mutation so subsequent ordinary constructors
    # and unrelated tests keep their original process-local allocation slots.
    program = """
import json, sys
sys.path.insert(0, sys.argv[1])
from coleman.state.components import ComponentValue
from tests.state.test_component_reader_values import CountedState, codec_for, reset_counted
calls = []
def observed(cls, *arguments, **keywords):
    calls.append("__new__")
    return object.__new__(cls)
ComponentValue.__new__ = staticmethod(observed) if sys.argv[2] == "observed" else None
reset_counted()
if sys.argv[2] == "observed":
    value = ComponentValue.from_bytes(codec_for(CountedState), b'{"schema_version":1,"value":7}')
    assert value.value.value == 7
    assert calls == ["__new__"]
    assert CountedState.events == ["reader", "writer", "writer", "reader", "writer"]
    print(json.dumps({"calls": calls, "value": value.value.value}))
else:
    try:
        ComponentValue.from_bytes(codec_for(CountedState), b'{"schema_version":1,"value":7}')
    except TypeError:
        assert CountedState.events == ["reader", "writer"]
        print(json.dumps({"custom_new_refused": True}))
    else:
        raise AssertionError("A present noncallable __new__ must not be mistaken for an absent stock descriptor")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(Path(__file__).parents[2]), mode],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {"calls": ["__new__"], "value": 7} if mode == "observed" else {"custom_new_refused": True}
    assert json.loads(result.stdout) == expected


@pytest.mark.parametrize("hook", ["__init__", "__post_init__"])
def test_rejecting_value_constructor_or_post_init_remains_an_actual_admission_guard(monkeypatch, hook):
    def reject(_self, *arguments, **keywords):
        raise ValueError("declared-custom-value-refusal")

    monkeypatch.setattr(ComponentValue, hook, reject)
    with pytest.raises(ValueError, match="declared-custom-value-refusal"):
        ComponentValue.from_bytes(codec_for(CountedState), b'{"schema_version":1,"value":7}')


def test_value_attribute_dispatch_used_by_original_post_init_is_not_bypassed(monkeypatch):
    original = ComponentValue.__getattribute__

    def guarded(self, name):
        if name == "codec":
            raise ValueError("existing-codec-attribute-guard")
        return original(self, name)

    monkeypatch.setattr(ComponentValue, "__getattribute__", guarded)
    with pytest.raises(ValueError, match="existing-codec-attribute-guard"):
        ComponentValue.from_bytes(codec_for(CountedState), b'{"schema_version":1,"value":7}')


@pytest.mark.parametrize("field", ["codec", "value"])
def test_value_field_data_descriptors_preserve_original_post_init_guards(monkeypatch, field):
    def read(_self):
        raise ValueError("existing-value-field-descriptor-guard")

    def write(self, value):
        object.__setattr__(self, "_authored_descriptor_storage", value)

    monkeypatch.setattr(ComponentValue, field, property(read, write), raising=False)
    with pytest.raises(ValueError, match="existing-value-field-descriptor-guard"):
        ComponentValue.from_bytes(codec_for(CountedState), b'{"schema_version":1,"value":7}')


def test_rejecting_codec_encoder_and_value_subclass_keep_their_own_guards():
    class RejectingCodec(ComponentCodec[CountedState]):
        def encode(self, state):
            raise ValueError("declared-custom-codec-refusal")

    codec = codec_for(CountedState)
    rejected = RejectingCodec(codec.binding, CountedState)
    with pytest.raises(ValueError, match="declared-custom-codec-refusal"):
        ComponentValue.from_bytes(rejected, b'{"schema_version":1,"value":7}')

    @dataclass(frozen=True)
    class RejectedValue(ComponentValue[CountedState]):
        def __post_init__(self):
            raise ValueError("declared-value-subclass-refusal")

    with pytest.raises(ValueError, match="declared-value-subclass-refusal"):
        RejectedValue.from_bytes(codec, b'{"schema_version":1,"value":7}')


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":2,"value":7}',
        b'{"schema_version":1,"schema_version":1,"value":7}',
        b'{"schema_version":1,"value":true}',
        b'{"schema_version":1,"value":7.1}',
        b'{"schema_version":1,"value":NaN}',
        b'{"schema_version":1,"value":Infinity}',
        b'{"schema_version":1,"value":7} ',
        b"\xff",
        b"",
    ],
)
def test_future_invalid_nonfinite_duplicate_noncanonical_and_non_utf8_bytes_refuse(payload):
    with pytest.raises((TypeError, ValueError, UnicodeError)):
        ComponentValue.from_bytes(codec_for(CountedState), payload)


@pytest.mark.parametrize("payload", [bytearray(b"{}"), memoryview(b"{}"), "{}"])
def test_own_bytes_factory_never_admits_mutable_or_nonbyte_input(payload):
    with pytest.raises((TypeError, ValueError)):
        ComponentValue.from_bytes(codec_for(CountedState), payload)


def test_unchecked_outer_binding_is_validated_before_wrapping_any_value():
    codec = codec_for(CountedState)
    object.__setattr__(codec, "binding", codec.binding.model_copy(update={"configuration_fingerprint": "invalid"}))
    with pytest.raises(ValueError):
        ComponentValue.from_bytes(codec, b'{"schema_version":1,"value":7}')


@pytest.mark.parametrize("route", ["factory", "ordinary"])
def test_outer_binding_is_still_validated_after_both_selected_readers(route):
    holder = []
    reads = []

    class BindingReader(CoreContract):
        value: StrictInt

        @classmethod
        def from_json(cls, payload: str | bytes) -> Self:
            value = super().from_json(payload)
            reads.append("reader")
            if len(reads) == 2:
                selected = holder[0]
                object.__setattr__(
                    selected,
                    "binding",
                    selected.binding.model_copy(update={"configuration_fingerprint": "invalid"}),
                )
            return value

    codec = codec_for(BindingReader)
    holder.append(codec)
    with pytest.raises(ValueError):
        if route == "factory":
            ComponentValue.from_bytes(codec, b'{"schema_version":1,"value":7}')
        else:
            ComponentValue(codec, codec.decode(b'{"schema_version":1,"value":7}'))
    assert reads == ["reader", "reader"]


def test_embedded_binding_must_match_the_independently_selected_outer_codec():
    class BoundState(CoreContract):
        binding: StateBinding
        value: StrictInt

    codec = codec_for(BoundState)
    changed = codec.binding.model_copy(update={"owner": Reference(id="another-reader-owner", version="1")})
    payload = json.dumps(
        {"schema_version": 1, "binding": changed.model_dump(mode="json"), "value": 7},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with pytest.raises(ValueError, match="embedded binding"):
        ComponentValue.from_bytes(codec, payload)


def test_real_store_load_uses_both_stock_readers_without_an_intervening_encode(state_example, tmp_path):
    codec = codec_for(CountedState)

    class Owner:
        def __init__(self):
            self.value = CountedState(schema_version=1, value=7)

        def export_state(self):
            return CountedState.model_validate(self.value)

        def restore_state(self, value):
            self.value = CountedState.model_validate(value)

    binding = ComponentBinding(codec, Owner, lambda _state: None)
    definition = CollectionDefinition(
        schema_version=1,
        reference=Reference(id="actual-reader-value-definition", version="1"),
        session=Reference(id="actual-reader-value-session", version="1"),
        source_epoch=Reference(id="actual-reader-value-epoch", version="1"),
        children=(),
        components=(ComponentSpec(binding=codec.binding, child=None, roles=("algorithm",), frontier_profile="none"),),
    )
    compatibility = ExecutionCompatibility(
        schema_version=1,
        runtime=state_example.execution_compatibility().runtime,
        owner_implementations=(codec.binding.implementation,),
    )
    store = LocalStateStore.create(tmp_path / "actual-base-store", definition, compatibility)
    store.initialize(
        command(0, None, kind="initialize"),
        (ComponentValue(codec, CountedState(schema_version=1, value=7)),),
        (binding,),
        compatibility=compatibility,
    )
    reset_counted()
    loaded = store.load((binding,), compatibility=compatibility)
    assert loaded.component(codec).value == 7
    assert CountedState.events == ["reader", "writer", "reader", "writer"]


def test_real_store_load_preserves_an_observed_codec_subclass_encoder(state_example, tmp_path):
    from tests.state.test_store_boundaries import TinyState, tiny_layout

    observed = []
    definition, bindings, compatibility = tiny_layout(state_example, count=1, observer=observed)
    codec = bindings[0].codec
    store = LocalStateStore.create(tmp_path / "actual-subclass-store", definition, compatibility)
    store.initialize(
        command(0, None, kind="initialize"),
        (ComponentValue(codec, TinyState(schema_version=1, text="complete-origin")),),
        bindings,
        compatibility=compatibility,
    )
    observed.clear()
    loaded = store.load(bindings, compatibility=compatibility)
    assert observed == ["tiny-00"]
    assert loaded.component(codec).text == "complete-origin"


@pytest.mark.parametrize("levels", [63, 64])
def test_complete_payload_keeps_the_existing_64_container_limit(levels):
    payload = b'{"schema_version":1,"value":' + b"[" * levels + b"0" + b"]" * levels + b"}"
    codec = codec_for(NestedState)
    if levels == 63:
        value = ComponentValue.from_bytes(codec, payload)
        assert codec.encode(value.value).content == payload
    else:
        with pytest.raises(ValueError, match="depth"):
            ComponentValue.from_bytes(codec, payload)


@pytest.mark.parametrize("excess", [0, 1])
def test_complete_payload_keeps_the_existing_exact_16_mib_boundary(excess):
    framing = b'{"schema_version":1,"value":""}'
    payload = b'{"schema_version":1,"value":"' + b"x" * (16 * 1024**2 - len(framing) + excess) + b'"}'
    assert len(payload) == 16 * 1024**2 + excess
    codec = codec_for(NestedState)
    if excess:
        with pytest.raises(ValueError, match="byte|bytes"):
            ComponentValue.from_bytes(codec, payload)
    else:
        value = ComponentValue.from_bytes(codec, payload)
        assert codec.encode(value.value).content == payload
