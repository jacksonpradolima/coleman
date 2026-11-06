"""Bounded declarative replay readers with an explicit concrete host-state type."""

from __future__ import annotations

from typing import cast

from coleman.core.base import CoreContract
from coleman.core.reading import _depth as _json_depth
from coleman.core.reading import bounded_json_bytes
from coleman.replay.contracts import ReplayContinuation, ReplayFeedback, ReplayState

DEFAULT_MAXIMUM_BYTES = 16 * 1024**2
MAXIMUM_DEPTH = 64


def _bounded(payload: str | bytes, maximum_bytes: int) -> bytes:
    """Preserve exact v1 ingress diagnostics through the shared bounded leaf."""
    return bounded_json_bytes(payload, maximum_bytes, label="Replay")


def _depth(text: str) -> None:
    """Retain the exact legacy lexical-depth helper without another parser."""
    _json_depth(text, "Replay")


def read_replay_state(payload: str | bytes, *, maximum_bytes: int = DEFAULT_MAXIMUM_BYTES) -> ReplayState:
    """Read one complete bounded immutable replay state without invoking a host.

    Parameters
    ----------
    payload : str or bytes
        Complete UTF-8 JSON, not a filename; caller bounds its acquisition too.
    maximum_bytes : int, optional
        Positive encoded-byte limit, default 16 MiB; depth is at most 64.

    Returns
    -------
    ReplayState
        Revalidated graph, causal records and independent frontiers.

    Raises
    ------
    ValueError, TypeError, UnicodeError
        Invalid bounds/UTF-8/JSON, duplicate keys, versions, units or invariants.

    Notes
    -----
    No I/O, migration, class lookup, native checkpoint or state restore occurs.
    """
    return ReplayState.from_json(_bounded(payload, maximum_bytes))


def read_replay_feedback(payload: str | bytes, *, maximum_bytes: int = DEFAULT_MAXIMUM_BYTES) -> ReplayFeedback:
    """Read a bounded revision without admitting it or inventing a first receipt.

    Parameters
    ----------
    payload : str or bytes
        Complete UTF-8 JSON, acquired separately by an explicitly authorized host.
    maximum_bytes : int, optional
        Positive encoded-byte bound, default 16 MiB; depth is at most 64.

    Returns
    -------
    ReplayFeedback
        Typed values and exact per-coordinate model/generator evidence.

    Raises
    ------
    ValueError, TypeError, UnicodeError
        Bad wire, unsupported version or malformed units/origin/content binding.

    Notes
    -----
    Parent presence and actual producer matching require graph admission later.
    No source lookup, model execution or learner update occurs.
    """
    return ReplayFeedback.from_json(_bounded(payload, maximum_bytes))


def read_continuation[S: CoreContract](
    payload: str | bytes, *, state_type: type[S], maximum_bytes: int = DEFAULT_MAXIMUM_BYTES
) -> ReplayContinuation[S]:
    """Read one declarative continuation using the caller's explicit state codec.

    Parameters
    ----------
    payload : str or bytes
        Complete UTF-8 envelope; it contains no executable pickle or class path.
    state_type : type of CoreContract
        Exact concrete host-owned snapshot model with its version/invariants.
        The bare CoreContract base is not a complete state codec and rejects.
    maximum_bytes : int, optional
        Positive encoded-byte bound, default 16 MiB; nesting is at most 64.

    Returns
    -------
    ReplayContinuation
        Exact typed snapshot and matching replay state; nothing is restored yet.

    Raises
    ------
    ValueError, TypeError, UnicodeError
        Bad bounds/JSON/versions, unspecified state codec or unmatched state digest.

    Notes
    -----
    Calls only the supplied declarative model validation; no import by name,
    filesystem, RNG, session registration or host restore is performed. Digest
    consistency authenticates neither supplied state nor its claimed completeness.
    """
    if not isinstance(state_type, type) or not issubclass(state_type, CoreContract) or state_type is CoreContract:
        raise TypeError("Continuation reader requires a concrete CoreContract state_type")
    contract = cast(type[ReplayContinuation[S]], ReplayContinuation.__class_getitem__(state_type))
    return contract.from_json(_bounded(payload, maximum_bytes))
