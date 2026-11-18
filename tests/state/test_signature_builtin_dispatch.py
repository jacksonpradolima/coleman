"""Keep exact typed signatures with an independent literal legacy walker.

The reference bodies are the committed pre-optimization algorithms, with only
their names and recursive calls changed. Generated cases are metamorphic checks,
not independent performance observations or a public signature wire format.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Hashable
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime, time
from decimal import Decimal
from enum import IntEnum, StrEnum
from math import isfinite
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, Field, PrivateAttr

from coleman.state import components


def _old_scalar_signature(value):
    """Run the original finite and representation-preserving scalar branches."""
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Component state requires finite float values")
        return type(value), value.hex()
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Component state requires finite Decimal values")
        return type(value), value.as_tuple()
    if isinstance(value, complex):
        if not isfinite(value.real) or not isfinite(value.imag):
            raise ValueError("Component state requires finite complex values")
        return type(value), (value.real.hex(), value.imag.hex())
    if isinstance(value, (datetime, time)):
        return type(value), (value.isoformat(), value.fold, getattr(value.tzinfo, "key", None))
    if not isinstance(value, Hashable):
        raise TypeError("Component state requires explicit portable scalar fields")
    return type(value), value


def _old_state_signature(value):
    """Run every original model, dataclass, container and scalar branch."""
    if isinstance(value, BaseModel):
        return type(value), tuple(
            (name, _old_state_signature(getattr(value, name))) for name in type(value).model_fields
        )
    if is_dataclass(value) and not isinstance(value, type):
        return type(value), tuple(
            (item.name, _old_state_signature(getattr(value, item.name))) for item in fields(value)
        )
    if isinstance(value, (tuple, list)):
        return type(value), tuple(_old_state_signature(item) for item in value)
    if isinstance(value, dict):
        return type(value), frozenset(
            (_old_state_signature(key), _old_state_signature(item)) for key, item in value.items()
        )
    if isinstance(value, (set, frozenset)):
        return type(value), frozenset(_old_state_signature(item) for item in value)
    return _old_scalar_signature(value)


def _outcome(function, value):
    try:
        return "value", function(value)
    except (ValueError, TypeError) as error:
        return "error", type(error), str(error)


class _Stored(BaseModel):
    values: tuple[object, ...]
    hidden: int = Field(default=7, exclude=True)
    _cache: object = PrivateAttr(default=None)

    def __eq__(self, other):
        raise AssertionError("A model equality override cannot supply its signature")


@dataclass(frozen=True)
class _Record:
    label: str
    values: object


class _Integer(int):
    """Retain the original subclass type in the scalar fallback."""


class _Text(str):
    """Retain the original subclass type in the scalar fallback."""


class _Bytes(bytes):
    """Retain the original subclass type in the scalar fallback."""


class _Number(IntEnum):
    ONE = 1


class _Label(StrEnum):
    ONE = "one"


@pytest.mark.parametrize(
    "value",
    [
        None,
        False,
        True,
        0,
        -1,
        1 << 4096,
        "",
        'ação/雪/😀/\x00\n"\\',
        b"",
        b"\x00\xff\n",
        0.0,
        -0.0,
        float.fromhex("0x0.0000000000001p-1022"),
        float.fromhex("0x1.fffffffffffffp+1023"),
        Decimal("2.00"),
        Decimal("-0.00"),
        complex(-0.0, -0.0),
        complex(1.25, -2.5),
        datetime(2026, 1, 1, tzinfo=UTC),
        time(1, 2, 3, fold=1, tzinfo=UTC),
        _Integer(1),
        _Text("one"),
        _Bytes(b"one"),
        _Number.ONE,
        _Label.ONE,
    ],
)
def test_all_scalar_representations_match_the_literal_original(value):
    assert components._state_signature(value) == _old_state_signature(value)


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        complex(float("inf"), 1),
        complex(1, float("nan")),
        bytearray(b"mutable"),
    ],
)
def test_nonfinite_and_unhashable_scalars_keep_original_error_type_and_message(value):
    actual = _outcome(components._state_signature, value)
    assert actual[0] == "error"
    assert actual == _outcome(_old_state_signature, value)
    assert _outcome(components._state_signature, ["earlier", value]) == _outcome(
        _old_state_signature, ["earlier", value]
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [(0, False), (1, True), (0.0, -0.0), (Decimal("2.0"), Decimal("2.00")), (Decimal("0"), Decimal("-0.00"))],
)
def test_numeric_equality_cannot_erase_type_signed_zero_or_decimal_quantum(left, right):
    assert left == right
    assert components._state_signature(left) != components._state_signature(right)
    assert components._state_signature([left, right]) == _old_state_signature([left, right])


def test_datetime_fold_and_named_zone_remain_distinct_even_with_equal_offsets():
    values = [
        datetime(2026, 1, 1, tzinfo=UTC, fold=0),
        datetime(2026, 1, 1, tzinfo=UTC, fold=1),
        datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC"), fold=0),
    ]
    assert values[0] == values[1] == values[2]
    signatures = [components._state_signature(value) for value in values]
    assert len(set(signatures)) == 3
    assert signatures == [_old_state_signature(value) for value in values]


@pytest.mark.parametrize("case", range(24))
def test_nested_model_dataclass_and_container_metamorphisms_match_original(case):
    rng = random.Random(378_700 + case)
    atoms = [None, False, 0, -1, 2**130, "same", "雪\n", b"\xff", -0.0, Decimal("3.00")]
    shared = [rng.choice(atoms) for _ in range(8)]
    stored = _Stored(values=tuple(shared))
    value = {"sequence": [shared, shared], "record": _Record("original", stored), "set": {0, "a", b"b"}}
    signature = components._state_signature(value)
    assert signature == _old_state_signature(value)
    assert components._state_signature(dict(reversed(tuple(value.items())))) == signature
    stored._cache = {"private": object()}
    assert components._state_signature(value) == signature
    shared.append("new declared occurrence")
    assert components._state_signature(value) != signature
    assert components._state_signature(value) == _old_state_signature(value)
    changed = components._state_signature(value)
    object.__setattr__(stored, "hidden", 8)
    assert components._state_signature(value) != changed
    assert components._state_signature(value) == _old_state_signature(value)


def test_tuple_list_set_and_frozenset_retain_types_and_array_order():
    values = [(0, "a"), [0, "a"], {0, "a"}, frozenset((0, "a"))]
    signatures = [components._state_signature(value) for value in values]
    assert signatures == [_old_state_signature(value) for value in values]
    assert len(set(signatures)) == 4
    assert components._state_signature([0, "a"]) != components._state_signature(["a", 0])
    assert components._state_signature({"a": 0, "b": 1}) == components._state_signature({"b": 1, "a": 0})


def test_unhashable_scalar_subclasses_keep_the_original_refusal():
    class MutableText(str):
        __hash__ = None

    class MutableInteger(int):
        __hash__ = None

    for value in (MutableText("one"), MutableInteger(1)):
        assert _outcome(components._state_signature, value) == _outcome(_old_state_signature, value)
        assert _outcome(components._state_signature, value)[0] == "error"


def test_custom_container_subclasses_preserve_original_callback_order():
    events = []

    class Sequence(list):
        def __iter__(self):
            events.append("sequence")
            return super().__iter__()

    class Mapping(dict):
        def items(self):
            events.append("mapping")
            return super().items()

    value = Mapping(field=Sequence([0, "a", b"b"]))
    expected = _old_state_signature(value)
    assert events == ["mapping", "sequence"]
    events.clear()
    actual = components._state_signature(value)
    assert events == ["mapping", "sequence"]
    assert actual == expected


def test_hashable_protocol_fallback_is_not_silently_tightened_for_memoryviews():
    """Preserve the old distinction between protocol admission and hashing.

    A writable memoryview advertises ``__hash__`` but refuses an actual hash.
    This optimization changes neither that fallback nor the host's obligation
    to select complete immutable portable state fields.
    """
    value = memoryview(bytearray(b"mutable"))
    signature = components._state_signature(value)
    assert signature == _old_state_signature(value)
    with pytest.raises(ValueError, match="writable memoryview"):
        hash(signature)


@contextmanager
def _scalar_calls():
    """Observe original fallback code without replacing validation or readers."""
    code = components._scalar_signature.__code__
    calls = []
    tool = next(index for index in (4, 3, 2, 5, 0, 1) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "coleman-signature-dispatch-test")

    def started(_code, _offset):
        calls.append("actual-fallback")

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
        yield calls
    finally:
        sys.monitoring.free_tool_id(tool)


def test_only_exact_builtin_leaves_bypass_the_real_scalar_fallback():
    with _scalar_calls() as calls:
        assert components._state_signature([None, False, True, 0, -1, "text", b"bytes"]) == _old_state_signature(
            [None, False, True, 0, -1, "text", b"bytes"]
        )
    assert calls == []
    values = [_Text("one"), _Integer(1), _Bytes(b"one"), 0.0, Decimal("2.00"), complex(1, 2), time(1)]
    with _scalar_calls() as calls:
        signature = components._state_signature(values)
    assert len(calls) == len(values)
    assert signature == _old_state_signature(values)
