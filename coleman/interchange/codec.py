"""Bounded UTF-8 wire acquisition, distinct from typed canonical identities."""

from __future__ import annotations

import hashlib

from pydantic import TypeAdapter

from coleman.core.primitives import Fingerprint
from coleman.interchange.contracts import InterchangeRequest, InterchangeResponse

MAXIMUM_WIRE_BYTES = 8 * 1024**2
MAXIMUM_JSON_DEPTH = 64


def _wire(payload: str | bytes) -> bytes:
    """Bound exact UTF-8 data and lexical nesting before the JSON parser runs."""
    if type(payload) not in {str, bytes}:  # pylint: disable=unidiomatic-typecheck
        raise TypeError("Interchange wire must be built-in str or bytes")
    if len(payload) > MAXIMUM_WIRE_BYTES:
        raise ValueError("Interchange wire exceeds 8 MiB before UTF-8 encoding")
    wire = payload.encode("utf-8") if isinstance(payload, str) else payload
    if not wire or len(wire) > MAXIMUM_WIRE_BYTES:
        raise ValueError("Interchange requires one through 8388608 UTF-8 bytes")
    text = wire.decode("utf-8")
    _check_nesting(text)
    return wire


def _quoted_character(char: str, escaped: bool) -> tuple[bool, bool]:
    """Advance lexical string state without treating escaped quotes as containers."""
    if escaped:
        return True, False
    return char != '"', char == "\\"


def _check_nesting(text: str) -> None:
    """Bound JSON containers before recursive parsing; typed parsing checks grammar."""
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            quoted, escaped = _quoted_character(char, escaped)
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAXIMUM_JSON_DEPTH:
                raise ValueError("Interchange JSON exceeds 64 nested containers")
        elif char in "]}":
            depth -= 1
            if depth < 0:
                raise ValueError("Interchange JSON closes an absent container")


def wire_fingerprint(payload: str | bytes) -> str:
    """Hash supplied exact bounded UTF-8 bytes without numeric reserialization.

    Parameters
    ----------
    payload : str or bytes
        One through 8 MiB of UTF-8 data with at most 64 nested JSON containers.
        This function bounds the wire; typed decoders separately validate JSON.

    Returns
    -------
    str
        Identified raw-byte SHA-256. Whitespace/newline changes alter it even
        when the typed canonical fingerprint is unchanged.

    Raises
    ------
    ValueError, UnicodeError
        Empty/oversized/deep input or invalid UTF-8.
    TypeError
        Unsupported type. No user conversion callbacks or I/O are performed.
    """
    return "sha256:" + hashlib.sha256(_wire(payload)).hexdigest()


def decode_request(payload: str | bytes) -> InterchangeRequest:
    """Decode one bounded v1 request without I/O, execution or migration.

    Parameters
    ----------
    payload : str or bytes
        Exact UTF-8 JSON with at most 8 MiB and 64 container levels.

    Returns
    -------
    InterchangeRequest
        Revalidated immutable typed input; existing mathematical units and
        missingness remain unchanged.

    Raises
    ------
    ValueError, UnicodeError, TypeError
        Invalid wire/JSON, duplicate/unknown keys, unsupported version or typed
        invariants. No dictionary escapes this acquisition boundary.
    """
    return InterchangeRequest.from_json(_wire(payload))


def decode_response(
    payload: str | bytes, *, expected_wire_fingerprint: Fingerprint | None = None
) -> InterchangeResponse:
    """Decode bounded output and optionally verify a separately retained wire pin.

    Parameters
    ----------
    payload : str or bytes
        Complete UTF-8 JSON, bounded to 8 MiB and 64 container levels.
    expected_wire_fingerprint : str or None, optional
        Expected exact-byte digest from a separately retained source. Null omits
        that integrity check; a digest stored beside untrusted bytes authenticates
        neither them nor their producer.

    Returns
    -------
    InterchangeResponse
        Revalidated declared response. ``accept_response`` must still compare
        the actual caller request and explicit receipt/deadline.

    Raises
    ------
    ValueError, UnicodeError, TypeError
        Invalid wire/pin/version/typed fields or a mismatched exact-byte digest.
        Performs no file reads, clocks, algorithms or executable deserialization.
    """
    wire = _wire(payload)
    if expected_wire_fingerprint is not None:
        expected = TypeAdapter(Fingerprint).validate_python(expected_wire_fingerprint)
        if "sha256:" + hashlib.sha256(wire).hexdigest() != expected:
            raise ValueError("Interchange wire digest differs from the retained pin")
    return InterchangeResponse.from_json(wire)
