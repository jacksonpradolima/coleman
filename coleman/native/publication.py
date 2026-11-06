"""Bounded native evidence reading and optional immutable local publication."""

from __future__ import annotations

from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.native.decisions import FrozenNativeDecision

DEFAULT_MAXIMUM_BYTES = 8 * 1024**2
MAXIMUM_DEPTH = 64


class NativeDecisionConflictError(ValueError):
    """Report different complete content for one host logical native decision.

    Notes
    -----
    The previously published winner remains unchanged. A caller must not retry
    native choice to resolve a storage conflict or invent a second content key.
    """


def _quoted_character(character: str, escaped: bool) -> tuple[bool, bool]:
    """Advance quoted-string state without interpreting its data as containers."""
    if escaped:
        return True, False
    return character != '"', character == "\\"


def _depth(text: str) -> None:
    """Bound containers before recursive parsing, respecting escaped string data."""
    depth, quoted, escaped = 0, False, False
    for character in text:
        if quoted:
            quoted, escaped = _quoted_character(character, escaped)
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > MAXIMUM_DEPTH:
                raise ValueError("Native decision JSON exceeds 64 nested containers")
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise ValueError("Native decision JSON closes an absent container")


def _wire(payload: str | bytes, maximum_bytes: int) -> bytes:
    """Bound built-in UTF-8 input before decoding or recursive model validation."""
    # A boolean is not a positive encoded-byte bound.
    # pylint: disable-next=unidiomatic-typecheck
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("Native decision maximum_bytes must be a positive integer")
    if type(payload) not in {str, bytes}:  # pylint: disable=unidiomatic-typecheck
        raise TypeError("Native decision payload must be built-in str or bytes")
    if not payload or len(payload) > maximum_bytes:
        raise ValueError("Native decision payload is empty or exceeds maximum_bytes")
    encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
    if len(encoded) > maximum_bytes:
        raise ValueError("Native decision UTF-8 bytes exceed maximum_bytes")
    _depth(encoded.decode("utf-8"))
    return encoded


def read_native_decision(payload: str | bytes, *, maximum_bytes: int = DEFAULT_MAXIMUM_BYTES) -> FrozenNativeDecision:
    """Read strict versioned native evidence without executing or migrating it.

    Parameters
    ----------
    payload : str or bytes
        Complete UTF-8 JSON, not a path. File/network acquisition belongs to the
        caller and must itself be bounded when its source is untrusted.
    maximum_bytes : int, optional
        Positive encoded-byte bound, default 8 MiB. Nesting is limited to 64
        containers before parsing. There is no implicit candidate-count cap.

    Returns
    -------
    FrozenNativeDecision
        Strict immutable v1 pre-evaluation evidence, retaining exact native
        semantic versions, Unicode names and ranking order.

    Raises
    ------
    ValueError, TypeError, UnicodeError
        Invalid bounds/UTF-8/JSON, duplicate or unknown fields, future/coerced
        versions, incompatible units, non-finite numbers or typed invariants.

    Notes
    -----
    Performs no I/O, catalog lookup, native import, pickle loading, learning or
    update. Legacy CSV/Parquet remains with its existing reader and absent
    metadata; it is never silently promoted into a verified native snapshot.
    Canonical content hashes are consistency checks, not authentication.
    """
    return FrozenNativeDecision.from_json(_wire(payload, maximum_bytes))


def publish_native_decision(
    record: FrozenNativeDecision, writer: ArtifactWriter, *, maximum_bytes: int = DEFAULT_MAXIMUM_BYTES
) -> Path:
    """Publish one complete pre-evaluation record using the existing writer.

    Parameters
    ----------
    record : FrozenNativeDecision
        Retained snapshot from one choice; revalidated before any write. Reuse
        this exact object/content for storage retries instead of choosing again.
    writer : ArtifactWriter
        Explicit existing writer with caller-owned trusted local parents.
    maximum_bytes : int, optional
        Positive canonical UTF-8 byte bound, default 8 MiB; enforced before I/O.

    Returns
    -------
    pathlib.Path
        Complete ``native-decisions/v1/<logical-identity>.json`` artifact. Exact
        retries and cooperative concurrent identical writes are idempotent.

    Raises
    ------
    NativeDecisionConflictError
        Different complete bytes already occupy the same logical decision key.
    ValueError, TypeError, UnicodeError
        Invalid model, unsupported version, excessive bytes/depth or bad names.
    OSError
        Actual local write/link/fsync errors propagate.

    Notes
    -----
    Uses ArtifactWriter's unique fsynced temporary file and atomic hard-link
    create-if-absent publication. A process interrupted before the final name
    leaves no completed evidence; after publication the complete record can be
    acknowledged by identical replay. Temporary files are never completion
    markers. Directory durability across power loss and hostile parent paths
    are outside that writer's contract. Native names never become paths.
    This single-file boundary does not transact with test execution, observe,
    global RNG, a checkpoint, or any legacy result sink.
    """
    content = FrozenNativeDecision.model_validate(record).to_json()
    checked = read_native_decision(content, maximum_bytes=maximum_bytes)
    path = writer.run_root / "native-decisions" / "v1" / (checked.logical_identity.removeprefix("sha256:") + ".json")
    try:
        writer.write_immutable(path, content)
    except ValueError as error:
        raise NativeDecisionConflictError("Conflicting complete native evidence for this logical decision") from error
    return path
