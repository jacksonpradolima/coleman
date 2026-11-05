"""Bounded v1 header projection without decoding outcome-bearing JSON subtrees."""

from coleman.results.domain import decode_json
from coleman.workflows.contracts import WorkflowError

_HEADERS = {"schema_version", "contract", "artifact_type", "plan"}
_MEMBERS = _HEADERS | {"report", "files"}
_CLOSING = {"{": "}", "[": "]"}


def _space(payload: str, index: int) -> int:
    """Skip JSON whitespace only, retaining offsets into the original bounded text."""
    while index < len(payload) and payload[index] in " \t\r\n":
        index += 1
    return index


def _string_end(payload: str, index: int) -> int:
    """Locate a raw quoted span without unescaping or materializing its contents."""
    index += 1
    while index < len(payload):
        if payload[index] == '"':
            return index + 1
        index += 2 if payload[index] == "\\" else 1
    raise WorkflowError("Unterminated portable JSON string")


def _value_end(payload: str, index: int) -> int:
    """Skip an opaque value with at most 128 containers; full JSON validation follows exposure."""
    if index >= len(payload):
        raise WorkflowError("Missing portable JSON value")
    if payload[index] == '"':
        return _string_end(payload, index)
    if payload[index] not in _CLOSING:
        end = index
        while end < len(payload) and payload[end] not in " \t\r\n,}]":
            end += 1
        if end == index:
            raise WorkflowError("Missing portable JSON value")
        return end
    return _container_end(payload, index)


def _container_end(payload: str, index: int) -> int:
    """Locate a bounded raw container end, never decoding its scalar or string values."""
    expected = [_CLOSING[payload[index]]]
    index += 1
    while expected and index < len(payload):
        character = payload[index]
        if character == '"':
            index = _string_end(payload, index)
            continue
        if character in _CLOSING:
            expected.append(_CLOSING[character])
            if len(expected) > 128:
                raise WorkflowError("Portable JSON nesting exceeds 128 containers")
        elif character in "}]" and character != expected.pop():
            raise WorkflowError("Mismatched portable JSON containers")
        index += 1
    if expected:
        raise WorkflowError("Unterminated portable JSON container")
    return index


def _member_name(payload: str, index: int, seen: set[str]) -> tuple[str, int]:
    """Decode only one known top-level member name, retaining its original colon offset."""
    if payload[index] != '"':
        raise WorkflowError("Portable JSON requires quoted member names")
    end = _string_end(payload, index)
    name = decode_json(payload[index:end])
    if name not in _MEMBERS or name in seen:
        raise WorkflowError("Unknown or duplicate portable JSON member")
    seen.add(name)
    separator = _space(payload, end)
    if separator >= len(payload) or payload[separator] != ":":
        raise WorkflowError("Portable JSON member requires a colon")
    return name, separator


def _members(payload: str, index: int) -> tuple[list[str], int]:
    """Project the four header spans, inspecting only the six known top-level key names."""
    pieces, seen = [], set()
    while index < len(payload):
        name, separator = _member_name(payload, index, seen)
        start = _space(payload, separator + 1)
        end = _value_end(payload, start)
        if name in _HEADERS:
            pieces.append(payload[index:separator] + ":" + payload[start:end])
        index = _space(payload, end)
        if index < len(payload) and payload[index] == "}":
            return pieces, index + 1
        if index >= len(payload) or payload[index] != ",":
            raise WorkflowError("Portable JSON members require a comma")
        index = _space(payload, index + 1)
    raise WorkflowError("Unterminated portable JSON object")


def _source_header(payload: str) -> str:
    """Return only raw v1 source/version fields; never decode report or file values.

    At most 2 MiB UTF-8 is structurally scanned using bounded state. Strings
    and numbers inside the opaque body are neither decoded nor interpreted.
    The caller must validate this projected header, reserve source exposure,
    then validate the complete original JSON before retaining any evidence.
    Malformed framing, duplicate/unknown top-level names and excessive nesting
    raise ValueError. Malformed opaque contents may reject only after exposure;
    no scientific source authenticity or body validity is granted here.
    """
    if len(payload.encode()) > 2097152:
        raise WorkflowError("Portable workflow JSON exceeds 2 MiB")
    index = _space(payload, 0)
    if index >= len(payload) or payload[index] != "{":
        raise WorkflowError("Portable workflow evidence requires a JSON object")
    index = _space(payload, index + 1)
    if index < len(payload) and payload[index] == "}":
        pieces, end = [], index + 1
    else:
        pieces, end = _members(payload, index)
    if _space(payload, end) != len(payload):
        raise WorkflowError("Portable JSON contains trailing data")
    return "{" + ",".join(pieces) + "}"
