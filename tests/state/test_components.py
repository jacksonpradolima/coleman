"""Concrete declarative payloads remain complete before any factory or restore."""

from functools import partial
from typing import cast

import pytest

from coleman.core.base import CoreContract
from coleman.core.primitives import fingerprint
from coleman.state.codec import read_algorithm_state
from coleman.state.components import ComponentCodec, ComponentValue, StateBytes, child_frontier
from coleman.state.contracts import AlgorithmState, StateBinding


def direct_binding(example):
    return StateBinding(
        owner=example.child_reference(0),
        implementation=example.owner_implementation(),
        state_codec=example.reference("direct-authored-host-state"),
        configuration_fingerprint=fingerprint({"child": 0}),
    )


@pytest.mark.parametrize("boundary", ["bytes", "codec"])
@pytest.mark.parametrize("kind", ["dictionary", "subclass"])
def test_in_process_codecs_cannot_retain_a_mutable_or_unchecked_binding(state_example, boundary, kind):
    example = state_example
    valid = direct_binding(example)
    if kind == "dictionary":
        value = valid.model_dump()
    else:

        class OtherBinding(StateBinding):
            """Another interpretation despite initially matching field names."""

        value = OtherBinding.model_validate(valid.model_dump())
    invalid = cast(StateBinding, value)
    if boundary == "bytes":
        principal = partial(StateBytes, invalid, b"{}")
    else:
        principal = partial(ComponentCodec, invalid, example.HostState)
    with pytest.raises(TypeError, match="StateBinding|binding"):
        principal()


def test_algorithm_generic_preserves_all_concrete_rng_cache_pending_fields(state_example):
    example = state_example
    host = example.HostConsumer(0)
    host.prewarm(example.instant(0))
    host.decide(example.opportunity(0, 0).request)
    payload = host.export_state()
    binding = direct_binding(example)
    wrapped = AlgorithmState[example.HostState](schema_version=1, binding=binding, payload=payload)
    decoded = read_algorithm_state(wrapped.to_json(), payload_type=example.HostState, expected=binding)
    assert type(decoded.payload) is example.HostState
    assert decoded.payload.to_json() == payload.to_json()
    assert len(decoded.payload.rng.words) == 624
    assert len(decoded.payload.pending) == 1
    assert decoded.payload.cache is not None
    assert decoded.to_json() == wrapped.to_json()


def test_algorithm_reader_rejects_another_exact_binding_without_restoring_a_host(state_example):
    example = state_example
    binding = direct_binding(example)
    wrapped = AlgorithmState[example.HostState](
        schema_version=1, binding=binding, payload=example.HostConsumer().export_state()
    )
    changed = StateBinding.model_validate(
        binding.model_copy(update={"configuration_fingerprint": fingerprint({"child": 1})})
    )
    encoded = wrapped.to_json()
    with pytest.raises(ValueError, match="binding"):
        read_algorithm_state(encoded, payload_type=example.HostState, expected=changed)


def test_component_codec_retains_the_exact_complete_concrete_payload(state_example):
    example = state_example
    host = example.HostConsumer()
    host.decide(example.opportunity(0, 0).request)
    state = host.export_state()
    codec = ComponentCodec(binding=direct_binding(example), state_type=example.HostState)
    encoded = codec.encode(state)
    assert encoded.content == state.to_json().encode()
    assert encoded.byte_count == len(encoded.content)
    assert codec.decode(encoded.content).to_json() == state.to_json()
    assert ComponentValue(codec, state).value.to_json() == state.to_json()
    whitespace = b" " + encoded.content
    with pytest.raises(ValueError, match="canonical"):
        codec.decode(whitespace)


def test_component_codec_refuses_base_or_open_generic_types_before_values_are_acquired(state_example):
    binding = direct_binding(state_example)
    with pytest.raises(TypeError, match="concrete"):
        ComponentCodec(binding=binding, state_type=CoreContract)
    with pytest.raises(TypeError, match="generic|concrete"):
        ComponentCodec(binding=binding, state_type=AlgorithmState)


def test_component_codec_does_not_project_extra_subclass_fields_onto_its_base_codec(state_example):
    example = state_example

    class Extended(example.HostState):
        important_cache_generation: int

    host = example.HostConsumer()
    base = host.export_state()
    extended = Extended.model_validate({**base.model_dump(), "important_cache_generation": 7})
    codec = ComponentCodec(binding=direct_binding(example), state_type=example.HostState)
    with pytest.raises(TypeError, match="exact declared concrete"):
        codec.encode(extended)


@pytest.mark.parametrize("invalid", [b"", bytearray(b"{}"), "{}"])
def test_state_bytes_require_actual_immutable_nonempty_bytes(state_example, invalid):
    binding = direct_binding(state_example)
    value = cast(bytes, invalid)
    with pytest.raises(ValueError, match="immutable bytes"):
        StateBytes(binding, value)


@pytest.mark.parametrize("change", ["host", "version"])
def test_frontier_inspector_revalidates_unchecked_complete_envelopes_before_claiming_counters(state_example, change):
    example = state_example
    replay, host = example.make_replay(0)
    valid = replay.export_continuation()
    if change == "host":
        unchecked = valid.model_copy(update={"host_state": example.HostConsumer(1).export_state()})
    else:
        unchecked = valid.model_copy(update={"schema_version": 3})
    child = example.child_reference(0)
    before = host.export_state().to_json()
    with pytest.raises(ValueError):
        child_frontier(unchecked, child=child)
    assert host.export_state().to_json() == before


def test_outer_component_codec_and_inner_host_codec_are_distinct_valid_bindings(state_example):
    example = state_example
    _, bindings = example.collection_layout()
    values = example.origin_values(bindings)
    child = values[0]
    assert child.codec.binding.state_codec != child.value.state.state_codec
    encoded = child.codec.encode(child.value)
    assert child.codec.decode(encoded.content).to_json() == child.value.to_json()
