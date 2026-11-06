"""Consume a pinned finite-result exchange using only Python's standard library.

This deliberately small foreign host supports the public synthetic coverage
operation, not arbitrary COLEMAN contracts. Raw-byte pins must be retained
separately by the caller; they provide integrity, not producer authentication.
It does not import the SDK, dynamically load a payload, or execute an action.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

_LIMIT = 8 * 1024**2


def _unique(pairs):
    """Reject duplicate keys instead of silently retaining the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate foreign-host JSON key")
        result[key] = value
    return result


def _read(path, expected):
    """Bound bytes before JSON parsing and verify a separately retained wire pin."""
    with Path(path).open("rb") as stream:
        wire = stream.read(_LIMIT + 1)
    if not wire or len(wire) > _LIMIT:
        raise ValueError("Foreign host accepts one through 8 MiB of JSON")
    if "sha256:" + hashlib.sha256(wire).hexdigest() != expected:
        raise ValueError("Foreign-host raw wire digest mismatch")
    return json.loads(
        wire.decode("utf-8"),
        object_pairs_hook=_unique,
        parse_float=Decimal,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON: " + value)),
    )


def _equal(left, right):
    """Keep signed decimal zero and JSON scalar types distinct in echoed inputs."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Decimal):
        return left.as_tuple() == right.as_tuple()
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _time(value):
    """Require supplied aware clocks; no current time is consulted."""
    if not isinstance(value, str):
        raise ValueError("Foreign coverage example requires known explicit clocks")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Foreign-host clock must be timezone-aware")
    return result


def _version(contract):
    """Reject missing, coerced and unknown versions at each admitted boundary."""
    if type(contract.get("schema_version")) is not int or contract["schema_version"] != 1:
        raise ValueError("Unsupported foreign-host coverage schema")


def consume(arguments):
    """Read one pinned finite result, preserving declared units and explicit time.

    This consumer checks the retained request and producer declarations. It does
    not independently certify the solver's branch-and-bound proof; the producing
    SDK validates that typed result, and a different foreign host can implement
    additional mathematical verification for its own admitted schema.
    """
    request = _read(arguments.request, arguments.request_wire)
    response = _read(arguments.response, arguments.response_wire)
    _version(request)
    _version(response)
    if not _equal(response["request"], request):
        raise ValueError("Foreign-host reply belongs to another actual request/attempt")
    if response["request_fingerprint"] != arguments.request_fingerprint:
        raise ValueError("Foreign-host request semantic pin mismatch")
    if response["input_fingerprint"] != arguments.input_fingerprint:
        raise ValueError("Foreign-host input semantic pin mismatch")
    if request["implementation"] != response["implementation"] or request["state"] is not None:
        raise ValueError("Foreign coverage producer identity or state mismatch")
    start, end = _time(request["created_at"]), _time(request["expires_at"])
    produced, receipt = _time(response["produced_at"]), _time(arguments.received_at)
    if not start <= produced <= receipt <= end:
        raise ValueError("Foreign-host result is premature, stale or reversed")
    call, reply = request["input"], response["reply"]
    if call["kind"] != "coverage" or reply["kind"] != "coverage":
        raise ValueError("Foreign host supports this finite coverage operation only")
    result = reply["result"]
    _version(call["problem"])
    _version(result)
    _version(result["problem"])
    if (
        not _equal(result["problem"], call["problem"])
        or not _equal(result["limits"], call["limits"])
        or result["method"] != call["method"]
        or response["implementation"] != {"id": call["method"], "version": result["solver_version"]}
        or response["state"] is not None
    ):
        raise ValueError("Foreign-host actual finite result differs from its producing input")
    # Explicit units and status remain visible; an absent result is never zero.
    return {
        "selected": result["selected_ids"],
        "status": result["status"],
        "cost": result["cost"],
        "objective": result["objective"],
        "sdk_imported": any(name == "coleman" or name.startswith("coleman.") for name in sys.modules),
    }


def main():
    """Read caller-provided pins and receipt instead of consulting ambient state."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "request",
        "response",
        "request-wire",
        "response-wire",
        "request-fingerprint",
        "input-fingerprint",
        "received-at",
    ):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(consume(parser.parse_args())))


if __name__ == "__main__":
    main()
