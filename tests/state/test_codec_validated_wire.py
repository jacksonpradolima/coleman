"""Preserve inherited, model-serialized and overridden complete wire contracts."""

import hashlib
from decimal import Decimal
from pathlib import Path
from typing import Self

import pytest
from pydantic import StrictInt, model_serializer, model_validator

from coleman.core.base import CoreContract
from coleman.core.primitives import canonical_json, decode_json, fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.components import ComponentCodec
from coleman.state.contracts import StateBinding


class DeclaredWire(CoreContract):
    signed: StrictInt
    amounts: tuple[Decimal, ...]
    nullable: str | None


class ModelWire(CoreContract):
    value: StrictInt

    @model_validator(mode="before")
    @classmethod
    def read_record(cls, value):
        if isinstance(value, dict) and set(value) == {"record"}:
            return value["record"]
        return value

    @model_serializer(mode="wrap")
    def write_record(self, handler):
        return {"record": handler(self)}


class MethodWire(CoreContract):
    value: StrictInt

    def to_json(self) -> str:
        checked = type(self).model_validate(self)
        return canonical_json({"format": "authored-method-v1", "record": checked.model_dump(mode="json")})

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        raw = decode_json(payload, exact_decimals=True)
        if not isinstance(raw, dict) or set(raw) != {"format", "record"} or raw["format"] != "authored-method-v1":
            raise ValueError("The selected method reader requires its exact declared envelope")
        return cls.model_validate(raw["record"])


def codec_for(example, state_type):
    implementation = DefinitionBinding(
        reference=example.reference("authored-wire-owner"),
        definition_fingerprint="sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    binding = StateBinding(
        owner=example.reference("authored-wire-component"),
        implementation=implementation,
        state_codec=example.reference("authored-" + state_type.__name__),
        configuration_fingerprint=fingerprint({"state_type": state_type.__name__}),
    )
    return ComponentCodec(binding, state_type)


@pytest.mark.parametrize("signed", (0, -3, 7))
def test_inherited_writer_retains_independent_bytes_digest_and_all_declared_types(state_example, signed):
    state = DeclaredWire(schema_version=1, signed=signed, amounts=(Decimal("2.00"), Decimal("-0.00")), nullable=None)
    codec = codec_for(state_example, DeclaredWire)
    expected = ('{"amounts":["2.00","-0.00"],"nullable":null,"schema_version":1,"signed":' + str(signed) + "}").encode()
    encoded = codec.encode(state)
    assert encoded.content == expected
    assert encoded.content == state.to_json().encode()
    assert encoded.fingerprint == "sha256:" + hashlib.sha256(expected).hexdigest()
    decoded = codec.decode(expected)
    assert type(decoded.signed) is int
    assert decoded.signed == signed
    assert decoded.nullable is None
    assert tuple(value.as_tuple() for value in decoded.amounts) == tuple(value.as_tuple() for value in state.amounts)
    assert codec.encode(decoded).content == expected


@pytest.mark.parametrize("change", ({"schema_version": 2}, {"signed": True}, {"signed": 1.5}))
def test_inherited_writer_still_revalidates_unchecked_versions_and_declared_types(state_example, change):
    codec = codec_for(state_example, DeclaredWire)
    origin = DeclaredWire(schema_version=1, signed=-3, amounts=(), nullable=None)
    unchecked = origin.model_copy(update=change)
    with pytest.raises(ValueError):
        codec.encode(unchecked)


def test_inherited_writer_keeps_a_reversible_pydantic_model_serializer(state_example):
    codec = codec_for(state_example, ModelWire)
    state = ModelWire(schema_version=1, value=-3)
    expected = b'{"record":{"schema_version":1,"value":-3}}'
    assert codec.encode(state).content == expected
    assert codec.decode(expected).value == -3
    assert state.to_json().encode() == expected


@pytest.mark.parametrize("value", (0, -3, 7))
def test_overridden_method_writer_and_selected_reader_keep_their_exact_envelope(state_example, value):
    codec = codec_for(state_example, MethodWire)
    state = MethodWire(schema_version=1, value=value)
    expected = ('{"format":"authored-method-v1","record":{"schema_version":1,"value":' + str(value) + "}}").encode()
    encoded = codec.encode(state)
    assert encoded.content == expected
    assert encoded.fingerprint == "sha256:" + hashlib.sha256(expected).hexdigest()
    decoded = codec.decode(expected)
    assert type(decoded.value) is int
    assert decoded.value == value
    assert codec.encode(decoded).content == expected
