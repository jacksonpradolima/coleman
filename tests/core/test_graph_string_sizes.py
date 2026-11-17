"""Independent JSON arithmetic preserves exact bounded graph admission."""

import json

import pytest

from coleman.core._wire import guard_graph


def encoded_size(value):
    """Measure independent standard JSON bytes without production size helpers."""
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


@pytest.mark.parametrize("character", [chr(code) for code in range(128)])
def test_every_ascii_character_keeps_exact_escaped_byte_boundary(character):
    """Quotes, backslashes, controls and DEL retain their actual JSON sizes."""
    value = character * 11
    size = encoded_size(value)
    guard_graph(value, maximum_bytes=size, maximum_items=1)
    with pytest.raises(ValueError, match="byte budget"):
        guard_graph(value, maximum_bytes=size - 1, maximum_items=1)


@pytest.mark.parametrize("length", (0, 1, 2, 31, 1024, 65536))
@pytest.mark.parametrize("pattern", ("a", '"', "\\", 'sha256:a"b\\c'))
def test_printable_ascii_repetition_has_affine_size_without_normalization(length, pattern):
    """Concatenation adds body bytes while retaining one pair of quotes."""
    value = pattern * length
    expected = 2 + length * (encoded_size(pattern) - 2)
    assert encoded_size(value) == expected
    guard_graph(value, maximum_bytes=expected, maximum_items=1)
    with pytest.raises(ValueError, match="byte budget"):
        guard_graph(value, maximum_bytes=expected - 1, maximum_items=1)


@pytest.mark.parametrize("value", ("á", "λ", "中", "😀", "\ud7ff", "\ue000", 'a\n"中\\😀', "\x00\x1f\x7f"))
def test_unicode_and_control_fallback_preserves_utf8_boundary(value):
    """Multi-byte code points and escaped controls retain the original route."""
    size = encoded_size(value)
    guard_graph(value, maximum_bytes=size, maximum_items=1)
    with pytest.raises(ValueError, match="byte budget"):
        guard_graph(value, maximum_bytes=size - 1, maximum_items=1)


@pytest.mark.parametrize("surrogate", ("\ud800", "\udfff"))
def test_lone_surrogates_still_reject_without_a_serializer(surrogate):
    """A printable prefix cannot hide invalid UTF-8 from the fallback guard."""
    with pytest.raises(ValueError, match="lone surrogates"):
        guard_graph("sha256:" + surrogate, maximum_bytes=128, maximum_items=1)


def test_repeated_aliases_charge_keys_and_values_on_every_occurrence():
    """String acceleration preserves item counts and repeated graph charges."""
    shared = {'quoted"key': "sha256:" + "a" * 64}
    value = [shared, shared]
    size = encoded_size(value)
    guard_graph(value, maximum_bytes=size, maximum_items=7)
    with pytest.raises(ValueError, match="byte budget"):
        guard_graph(value, maximum_bytes=size - 1, maximum_items=7)
    with pytest.raises(ValueError, match="item budget"):
        guard_graph(value, maximum_bytes=size, maximum_items=6)


def test_oversized_ascii_keeps_early_budget_refusal():
    """A tiny allowance cannot admit a large existing string or quote sequence."""
    for value in ("a" * 65536, '"' * 65536):
        with pytest.raises(ValueError, match="byte budget"):
            guard_graph(value, maximum_bytes=8, maximum_items=1)
