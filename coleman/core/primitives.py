"""Pure shared JSON, identity and legacy finite-number primitives.

Existing imports remain aliases of these objects. COLEMAN JSON v1 preserves
signed binary64 zero and is not cross-language RFC 8785 canonical JSON.
"""

from __future__ import annotations

import hashlib
import json
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
    bound payload size/depth before using untrusted data.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            if not isinstance(key, str):
                raise ValueError("Canonical JSON object keys must be strings")
            canonical_json(value)
    elif isinstance(payload, list):
        for value in payload:
            canonical_json(value)
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
