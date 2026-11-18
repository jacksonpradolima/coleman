"""Bound built-in JSON input before decoding without importing a domain or store."""

from __future__ import annotations

import re
from typing import Literal

MAXIMUM_DEPTH = 64
ReaderLabel = Literal["Replay", "State"]
_STRUCTURAL = re.compile(r'"(?:[^"\\]++|\\[\s\S])*+(?:"|\\?\Z)|[{}\[\]]')


def bounded_json_bytes(payload: str | bytes, maximum_bytes: int, *, label: ReaderLabel = "State") -> bytes:
    """Check built-in UTF-8 bytes and lexical nesting before recursive parsing.

    Parameters
    ----------
    payload : str or bytes
        Complete JSON content, never a filename or conversion-bearing subclass.
    maximum_bytes : int
        Positive encoded-byte ceiling. Booleans and coerced values reject.
    label : {'Replay', 'State'}, optional
        Fixed diagnostic vocabulary; Replay preserves the existing v1 errors.

    Returns
    -------
    bytes
        Original UTF-8 content after byte/depth checks, without JSON decoding.

    Raises
    ------
    ValueError, TypeError, UnicodeError
        Invalid limit, label, built-in kind, UTF-8, empty input or depth above64.

    Notes
    -----
    String escapes do not count as containers. The actual guarded decoder still
    rejects malformed JSON and duplicate keys. Performs no I/O, migration,
    runtime capture or application callbacks. The caller bounds acquisition too.
    """
    # Exact built-in types prevent invoking arbitrary conversion callbacks.
    # pylint: disable=unidiomatic-typecheck
    if type(label) is not str or label not in {"Replay", "State"}:
        raise ValueError("JSON reader requires a declared diagnostic label")
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError(f"{label} maximum_bytes requires a positive integer")
    if type(payload) not in {str, bytes}:
        raise TypeError(f"{label} reader requires built-in str or bytes, not a path")
    if not payload or len(payload) > maximum_bytes:
        raise ValueError(f"{label} payload is empty or exceeds maximum_bytes")
    encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{label} UTF-8 bytes exceed maximum_bytes")
    _depth(encoded.decode("utf-8"), label)
    return encoded


def _depth(text: str, label: ReaderLabel) -> None:
    """Check lexical containers iteratively, leaving grammar to the JSON reader."""
    depth = 0
    changes = {"[": 1, "{": 1, "]": -1, "}": -1}
    # Disjoint ordinary runs and two-character escapes consume a whole quoted
    # span. Possessive quantifiers prevent repartition/backtracking; the closing
    # quote or optional dangling backslash at EOF makes every opening quote
    # match through its actual end. Work stays linear, with no match list.
    # Python supports these possessive quantifiers since 3.11:
    # https://docs.python.org/3.14/library/re.html#regular-expression-syntax
    # This is lexical depth only: the actual JSON reader still rejects grammar,
    # including invalid escapes, literal newlines and unterminated strings.
    for token in _STRUCTURAL.finditer(text):
        # Inspect the source directly instead of copying a whole quoted span.
        character = text[token.start()]
        if character == '"':
            continue
        depth += changes[character]
        if not 0 <= depth <= MAXIMUM_DEPTH:
            raise ValueError(f"{label} JSON container depth is invalid or exceeds 64")
