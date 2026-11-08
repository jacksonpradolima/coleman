"""Preserve JSON v1 bytes/errors while bounding redundant standard encoding.

The independent reference below is the pre-optimization public function body,
with only its function/self-call name changed. Generated metamorphic cases are
serialization checks, not independent experimental observations or benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
import sys
from contextlib import contextmanager
from decimal import Decimal
from enum import IntEnum, StrEnum
from pathlib import Path
from types import MappingProxyType

import pytest

from coleman.core.primitives import canonical_json, fingerprint


def _reference_json(payload):
    """Run the original recursive public encoding, independently of the SUT."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if not isinstance(key, str):
                raise ValueError("Canonical JSON object keys must be strings")
            _reference_json(value)
    elif isinstance(payload, list):
        for value in payload:
            _reference_json(value)
    elif payload is not None and not isinstance(payload, str | bool | int | float):
        raise TypeError("Canonical input must contain only JSON values")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _failure(serializer, payload):
    """Retain the actual first error without replacing a serializer operation."""
    try:
        serializer(payload)
    except (TypeError, ValueError, RuntimeError) as error:
        return type(error), str(error)
    raise AssertionError("The declared invalid payload unexpectedly serialized")


def _graph(rng, depth=0):
    """Generate bounded ordinary JSON using exact IEEE bit patterns as atoms."""
    floating = struct.unpack("!d", rng.getrandbits(64).to_bytes(8, "big"))[0]
    atoms = [
        None,
        True,
        False,
        rng.getrandbits(128),
        -0.0,
        floating if math.isfinite(floating) else 0.0,
        'ação/雪\\"\n\t',
        "",
    ]
    if depth == 4 or rng.randrange(3) == 0:
        return rng.choice(atoms)
    if rng.randrange(2):
        return [_graph(rng, depth + 1) for _ in range(rng.randrange(5))]
    return {f"key-{i}": _graph(rng, depth + 1) for i in range(rng.randrange(5))}


def _reverse_keys(value):
    """Permute object insertion order while preserving every array coordinate."""
    if isinstance(value, dict):
        return {key: _reverse_keys(item) for key, item in reversed(tuple(value.items()))}
    if isinstance(value, list):
        return [_reverse_keys(item) for item in value]
    return value


@pytest.mark.parametrize("case", range(64), ids=lambda case: f"graph-{case}")
def test_generated_plain_trees_match_legacy_and_key_permutation(case):
    payload = _graph(random.Random(368000 + case))
    expected = _reference_json(payload)
    assert canonical_json(payload) == expected
    assert canonical_json(_reverse_keys(payload)) == expected
    assert fingerprint(payload) == "sha256:" + hashlib.sha256(expected.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    "bits", [0, 1 << 63, 1, (1 << 52) - 1, 1 << 52, 0x7FEFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF, 0x3FF0000000000000]
)
def test_finite_ieee_extremes_and_signed_zero_keep_exact_json_text(bits):
    value = struct.unpack("!d", bits.to_bytes(8, "big"))[0]
    payload = {"value": value}
    assert canonical_json(payload) == _reference_json(payload)
    assert canonical_json(payload) == json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def test_alias_occurrences_and_later_mutation_are_not_memoized_or_normalized():
    shared = [-0.0, None, 0, False]
    payload = {"left": shared, "right": shared}
    original = canonical_json(payload)
    assert original == '{"left":[-0.0,null,0,false],"right":[-0.0,null,0,false]}'
    shared.append(1)
    assert canonical_json(payload) == _reference_json(payload)
    assert canonical_json(payload) != original
    assert payload["left"] is payload["right"]
    assert canonical_json(list(reversed(shared))) != canonical_json(shared)


@pytest.mark.parametrize(
    "case",
    [
        "tuple",
        "set",
        "decimal",
        "mapping",
        "key-int",
        "key-bool",
        "nan-first",
        "inf-first",
        "bad-key-first",
        "object-first",
    ],
)
def test_invalid_first_value_or_key_keeps_legacy_exception_priority(case):
    values = {
        "tuple": {"earlier": (1,), "later": float("nan")},
        "set": {"earlier": {1}, "later": float("nan")},
        "decimal": {"earlier": Decimal("1.0"), "later": float("nan")},
        "mapping": {"earlier": MappingProxyType({"a": 1}), "later": float("nan")},
        "key-int": {1: None, "later": object()},
        "key-bool": {True: None, "later": object()},
        "nan-first": {"earlier": float("nan"), 1: "later bad key"},
        "inf-first": {"earlier": -float("inf"), "later": object()},
        "bad-key-first": {"earlier": {1: "invalid"}, "later": float("nan")},
        "object-first": {"earlier": object(), "later": float("nan")},
    }
    payload = values[case]
    assert _failure(canonical_json, payload) == _failure(_reference_json, payload)


@contextmanager
def _digit_limit(limit):
    """Restore the real process digit limit even if the expected error occurs."""
    original = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(limit)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(original)


def test_oversized_integer_failure_precedes_later_unsupported_value_under_real_digit_limit():
    original = sys.get_int_max_str_digits()
    with _digit_limit(sys.int_info.str_digits_check_threshold):
        payload = {"earlier": 1 << 2200, "later": object()}
        expected = _failure(_reference_json, payload)
        actual = _failure(canonical_json, payload)
        assert expected[0] is ValueError
        assert "limit" in expected[1]
        assert actual == expected
    assert sys.get_int_max_str_digits() == original


@pytest.mark.parametrize("bits", [2048, 2049, 2101])
def test_large_valid_integer_remains_readable_including_legacy_fallback(bits):
    with _digit_limit(sys.int_info.str_digits_check_threshold):
        payload = {"value": (1 << (bits - 1)) - 1}
        assert canonical_json(payload) == _reference_json(payload)


class _IntegerKind(IntEnum):
    ONE = 1


class _StringKind(StrEnum):
    PLAN = "ci-plan"


def test_scalar_subclasses_and_enums_keep_original_admission_without_conversion_hooks():
    class Integer(int):
        def __int__(self):
            raise AssertionError("No integer conversion hook belongs to encoding")

        def __repr__(self):
            raise AssertionError("The stdlib must use its original integer representation")

    class Floating(float):
        def __float__(self):
            raise AssertionError("No float conversion hook belongs to encoding")

        def __repr__(self):
            raise AssertionError("The stdlib must use its original float representation")

    class Text(str):
        def __str__(self):
            raise AssertionError("No text conversion hook belongs to encoding")

    payload = {"values": [Integer(7), Floating(-0.0), Text("ação"), _IntegerKind.ONE, _StringKind.PLAN]}
    assert canonical_json(payload) == _reference_json(payload)
    assert canonical_json(payload) == '{"values":[7,-0.0,"ação",1,"ci-plan"]}'


def _hooked_graph(kind, events):
    """Build fresh equal graphs so stateful legacy hook sequences are observable."""

    class Dictionary(dict):
        def items(self):
            events.append("items")
            return super().items()

    class Sequence(list):
        def __iter__(self):
            events.append("iter")
            if kind == "stateful-list" and events.count("iter") == 2:
                raise RuntimeError("declared-second-iteration-failure")
            return super().__iter__()

    class Key(str):
        def __lt__(self, other):
            events.append("key-sort")
            if kind == "key-raise":
                raise RuntimeError("declared-key-sort-failure")
            return str.__lt__(self, other)

    class ClassProbe:
        @property
        def __class__(self):
            events.append("class-probe")
            return str

    class EqualityProbe(type):
        def __eq__(cls, other):
            events.append("metaclass-equality")
            raise RuntimeError("Type routing cannot invoke equality hooks")

    class Foreign(metaclass=EqualityProbe):
        pass

    if kind == "dict":
        return {"outer": [Dictionary({"b": 2, "a": 1})]}
    if kind in {"list", "stateful-list"}:
        return {"outer": [Sequence([1, 2])], "later": object() if kind == "stateful-list" else None}
    if kind in {"key", "key-raise"}:
        return {"outer": {Key("b"): 2, Key("a"): 1}, "later": object() if kind == "key-raise" else None}
    return {"outer": ClassProbe() if kind == "class-probe" else Foreign()}


@pytest.mark.parametrize(
    "kind", ["dict", "list", "key", "key-raise", "stateful-list", "class-probe", "metaclass-probe"]
)
def test_whole_root_fallback_preserves_subclass_hooks_and_first_error(kind):
    expected_events, actual_events = [], []
    expected = _hooked_graph(kind, expected_events)
    actual = _hooked_graph(kind, actual_events)
    if kind in {"dict", "list", "key"}:
        assert canonical_json(actual) == _reference_json(expected)
    else:
        old_error = _failure(_reference_json, expected)
        assert _failure(canonical_json, actual) == old_error
        if kind == "key-raise":
            assert old_error[0] is RuntimeError
    assert actual_events == expected_events


@pytest.mark.parametrize("depth", [63, 64, 65, 96])
def test_routing_depth_does_not_become_a_new_public_acceptance_limit(depth):
    payload = 0
    for _ in range(depth):
        payload = [payload]
    assert canonical_json(payload) == _reference_json(payload)


@pytest.mark.parametrize("kind", ["list", "dict"])
def test_cycles_remain_legacy_recursion_errors_without_new_policy(kind):
    payload = [] if kind == "list" else {}
    if isinstance(payload, list):
        payload.append(payload)
    else:
        payload["cycle"] = payload
    assert _failure(_reference_json, payload)[0] is RecursionError
    assert _failure(canonical_json, payload)[0] is RecursionError


def test_lone_surrogate_text_is_not_silently_utf8_encoded_or_normalized():
    payload = {"value": "\ud800"}
    assert canonical_json(payload) == _reference_json(payload)
    assert "\ud800" in canonical_json(payload)
    error = _failure(fingerprint, payload)
    assert error[0] is UnicodeEncodeError


@pytest.mark.parametrize(
    ("relative", "digest"),
    [
        (
            "fixtures/learning/v1/controlled-law-v1.json",
            "bd44564a688996b5a92aee2831de29744670a6e8f712d89cbb60515673cd2a99",
        ),
        (
            "fixtures/learning/v1/controlled-pending-state-v1.json",
            "c599fbf30cf865ca163102cce7c30ba7e4236e86aabe869d95384900b712c814",
        ),
        (
            "fixtures/learning/v1/controlled-released-state-v1.json",
            "80afd061878fbe9f541e3cf453ac8c63c9519852404f34b03058b8398e1eb65e",
        ),
        (
            "evaluation/fixtures/bounded-freeze-v1.json",
            "60e1d4044ad502a66a2bcc1fa31a0232c6069c4ee846519f3b94a34274769149",
        ),
        (
            "evaluation/fixtures/bounded-report-v1.json",
            "f7d71f8063d58cf4217a307bd3ccfb611b2a08720545b141ec2cf8fb3f9b243a",
        ),
    ],
)
def test_existing_versioned_golden_bytes_and_digests_remain_exact(relative, digest):
    original = (Path(__file__).parents[1] / relative).read_bytes()
    assert hashlib.sha256(original).hexdigest() == digest
    decoded = json.loads(original)
    assert canonical_json(decoded).encode("utf-8") == original.rstrip(b"\n")
    assert canonical_json(decoded) == _reference_json(decoded)


def test_plain_nested_graph_has_one_complete_stdlib_encoding_without_mocking_it(record_property):
    payload = {"root": [{"index": index, "values": [True, -0.0, None, "ação"]} for index in range(16)]}
    expected = _reference_json(payload)
    encoder_code = json.JSONEncoder.encode.__code__
    calls = 0

    def observe(frame, event, _argument):
        nonlocal calls
        if event == "call" and frame.f_code is encoder_code:
            calls += 1

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        actual = canonical_json(payload)
    finally:
        sys.setprofile(previous)
    record_property("stdlib_encoder_calls", calls)
    assert actual == expected
    assert calls <= 1
    assert sys.getprofile() is previous
