"""Neutral bounded envelopes retain scientific readers without domain imports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel, model_serializer

from coleman.core import _wire as neutral
from coleman.evaluation.design import _wire as scientific


class _SmallEnvelope(neutral.BoundedContract):
    """Use an independently chosen small byte allowance to exercise framing."""

    _maximum_bytes: ClassVar[int] = 64
    label: str


def test_legacy_names_reexport_the_single_neutral_implementation():
    """Retained imports and their original envelope default keep their meaning."""
    assert scientific.guard_graph is neutral.guard_graph
    assert scientific.model_storage is neutral.model_storage
    assert scientific.wire_size is neutral.wire_size
    assert issubclass(scientific.DesignContract, neutral.BoundedContract)
    assert scientific.DesignContract._maximum_bytes == 64 * 1024**2


def test_neutral_envelope_roundtrip_and_exact_utf8_wire():
    """A new caller can bound canonical UTF-8 without a scientific contract."""
    value = _SmallEnvelope(schema_version=1, label='á\n"')
    expected = '{"label":"á\\n\\"","schema_version":1}'
    assert value.to_json() == expected
    assert neutral.wire_size(value) == len(expected.encode("utf-8"))
    restored = _SmallEnvelope.from_json(expected)
    assert restored == value
    assert restored.fingerprint == value.fingerprint
    with pytest.raises(ValueError, match="budget"):
        _SmallEnvelope(schema_version=1, label="á" * 64)


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":2,"label":"a"}',
        '{"schema_version":true,"label":"a"}',
        '{"schema_version":1,"label":"a","label":"b"}',
        '{"schema_version":1,"label":"a","extra":0}',
    ],
)
def test_shared_bounded_reader_keeps_version_and_duplicate_guards(payload):
    """Changing the namespace does not relax inherited ingress validation."""
    with pytest.raises(ValueError):
        _SmallEnvelope.from_json(payload)


def test_neutral_preflight_rejects_unchecked_storage_before_serializer():
    """A nested legacy serializer cannot run before bounded graph admission."""
    calls = []

    class Hostile(BaseModel):
        """Expose a stored opaque object and an observable unsafe serializer."""

        payload: object

        @model_serializer
        def serialize(self):
            """Record a forbidden call rather than hiding the unchecked object."""
            calls.append("serialized")
            return {"payload": "concealed"}

    class Envelope(neutral.BoundedContract):
        """Exercise the actual neutral guard against a legacy nested model."""

        nested: Hostile

    unchecked = Hostile.model_construct(payload=object())
    with pytest.raises(TypeError, match="unsupported opaque"):
        Envelope(schema_version=1, nested=unchecked)
    assert calls == []


def test_neutral_wire_cold_import_and_use_do_not_load_domains(tmp_path):
    """The new shared base can run with domain and application imports refused."""
    source = r"""
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
forbidden = ("coleman.evaluation", "coleman.application", "coleman.transports",
             "coleman.runner", "coleman.native", "polars", "pyarrow", "mcp")
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == item or fullname.startswith(item + ".") for item in forbidden):
            raise AssertionError("Forbidden import: " + fullname)
sys.meta_path.insert(0, Refuse())
from coleman.core._wire import BoundedContract
class Envelope(BoundedContract):
    value: int
value = Envelope(schema_version=1, value=0)
assert value.to_json() == '{"schema_version":1,"value":0}'
assert Envelope.from_json(value.to_json()).fingerprint == value.fingerprint
assert not any(name == item or name.startswith(item + ".")
               for name in sys.modules for item in forbidden)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", source, str(Path(__file__).parents[2])],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
