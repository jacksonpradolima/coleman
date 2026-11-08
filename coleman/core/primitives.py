"""Pure shared JSON, identity and legacy finite-number primitives.

Existing imports remain aliases of these objects. COLEMAN JSON v1 preserves
signed binary64 zero and is not cross-language RFC 8785 canonical JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from typing import Annotated, Any

from pydantic import Field

MAX_MEMBERS = 4096
MAX_DIMENSIONS = 64

Identifier = Annotated[str, Field(strict=True, min_length=1, pattern=r"^\S+(?:[^\r\n]*\S)?$")]
Fingerprint = Annotated[str, Field(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")]
# Preserve legacy Pydantic finite-float coercion; new math uses numeric.Binary64.
Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]


def canonical_json(payload: Any) -> str:
    """Encode JSON values deterministically with finite numbers only.

    Parameters
    ----------
    payload : object
        JSON values: string-keyed dictionaries, lists, strings, booleans,
        integers, finite floats or null. Array order remains significant.

    Returns
    -------
    str
        Compact JSON with sorted keys and unescaped UTF-8 text.

    Raises
    ------
    ValueError
        If an object key is not a string or a float is non-finite.
    TypeError
        If an input contains a non-JSON Python object.

    Notes
    -----
    Performs no filesystem access, rounding or ``default=str`` coercion.
    This is COLEMAN JSON v1, not RFC 8785 cross-language JCS. The caller must
    bound payload size/depth before using untrusted data. Plain built-in trees
    are validated before one complete encoding. Subclasses, large integers,
    non-finite values and deeper trees retain the original recursive encoder;
    the fast-path routing depth is not a new acceptance limit.
    """
    if not _plain_json_tree(payload):
        return _legacy_canonical_json(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _plain_json_tree(payload: Any, remaining: int = 64) -> bool:
    """Select hook-free built-ins without rejecting or normalizing other input.

    Parameters
    ----------
    payload : object
        Original value; no subclass hook or conversion is invoked here.
    remaining : int, optional
        Remaining container budget for this routing pass, default 64. Exhaustion
        selects the original encoder rather than imposing an acceptance limit.

    Returns
    -------
    bool
        Whether one final standard encoding can handle the plain tree. Integers
        above 2048 bits use the original routine, including its digit-limit and
        error-order behavior, even when that integer would encode successfully.

    Notes
    -----
    Exact type checks avoid invoking user container, comparison or conversion
    hooks during this preflight. A false result sends the entire original root
    to the legacy routine, preserving its error order and subtype callbacks.
    The routing budget avoids traversing deep/cyclic inputs before that routine.
    """
    payload_type = type(payload)
    if payload_type is dict:
        return remaining > 0 and _plain_json_object(payload, remaining - 1)
    if payload_type is list:
        return remaining > 0 and all(_plain_json_tree(value, remaining - 1) for value in payload)
    if payload_type is float:
        return math.isfinite(payload)
    if payload_type is int:
        return int.bit_length(payload) <= 2048
    return payload is None or payload_type is str or payload_type is bool


def _plain_json_object(payload: dict[Any, Any], remaining: int) -> bool:
    """Inspect each exact key and value in their original insertion order.

    Parameters
    ----------
    payload : dict
        Already identified built-in dictionary, never a user-defined mapping.
    remaining : int
        Routing budget for its values; it does not reject the original root.

    Returns
    -------
    bool
        False at the first nonplain key/value so original encoding can decide
        which error or subtype hook precedes later siblings.
    """
    for key, value in payload.items():
        key_type = type(key)
        if key_type is not str or not _plain_json_tree(value, remaining):
            return False
    return True


def _legacy_canonical_json(payload: Any) -> str:
    """Retain the original recursive encoding, subtype hooks and error order.

    Parameters
    ----------
    payload : object
        Entire original root, including uncommon integer values or subclasses.

    Returns
    -------
    str
        Original COLEMAN JSON v1 text, with unchanged standard encoder options.

    Raises
    ------
    ValueError, TypeError
        Original invalid key, value, numeric or standard encoding failures.
    RecursionError
        Cyclic or excessively deep recursion under the original runtime policy.

    Notes
    -----
    Subtype callbacks propagate their own failures. Keeping self-recursion in
    this routine preserves their order before later sibling values are visited.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            if not isinstance(key, str):
                raise ValueError("Canonical JSON object keys must be strings")
            _legacy_canonical_json(value)
    elif isinstance(payload, list):
        for value in payload:
            _legacy_canonical_json(value)
    elif payload is not None and not isinstance(payload, str | bool | int | float):
        raise TypeError("Canonical input must contain only JSON values")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def decode_json(payload: str | bytes, *, exact_decimals: bool = False) -> Any:
    """Parse JSON while rejecting duplicate object keys at every depth.

    Parameters
    ----------
    payload : str or bytes
        Complete JSON document. Callers must impose their own byte/depth limits.
    exact_decimals : bool, optional
        Preserve non-integral numeric literals as ``decimal.Decimal`` until
        typed validation. The default retains the legacy binary64 parser.

    Returns
    -------
    object
        Decoded JSON value; no domain/schema validation is performed here.

    Raises
    ------
    ValueError
        If syntax or duplicate keys make the document invalid.
    UnicodeDecodeError
        If input bytes cannot be decoded by the standard JSON reader.

    Notes
    -----
    Accesses no files. Non-standard numeric constants and contract-specific
    finite-number, shape and version constraints must be rejected by the
    consuming typed contract; this helper is not their replacement.
    """

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Reject ambiguous duplicate keys at every object depth."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(payload, object_pairs_hook=unique_object, parse_float=Decimal if exact_decimals else float)


def fingerprint(payload: Any) -> str:
    """Hash canonical JSON input with an identified SHA-256 digest.

    Parameters
    ----------
    payload : object
        JSON-compatible input accepted by ``canonical_json``.

    Returns
    -------
    str
        ``sha256:`` followed by 64 lowercase hexadecimal characters.

    Raises
    ------
    ValueError
        If canonical encoding rejects a key or non-finite value.
    TypeError
        If the input contains a non-JSON object.

    Notes
    -----
    Performs no filesystem access. Hashes content rather than authenticating
    its producer, license, temporal availability or scientific eligibility.
    """
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
