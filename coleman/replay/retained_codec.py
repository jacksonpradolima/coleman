"""Bounded explicit generation-two replay readers, without host restoration."""

from __future__ import annotations

from coleman.core.base import CoreContract
from coleman.replay.codec import DEFAULT_MAXIMUM_BYTES, _bounded
from coleman.replay.retained_contracts import RetainedContinuation, RetainedReplayState, _envelope_type


def read_retained_state(payload: str | bytes, *, maximum_bytes: int = DEFAULT_MAXIMUM_BYTES) -> RetainedReplayState:
    """Read a sealed prefix and bounded suffix under the explicit v2 schema.

    Parameters
    ----------
    payload : str or bytes
        Complete UTF-8 JSON, not a path; acquisition remains caller-owned.
    maximum_bytes : int, optional
        Positive encoded-byte ceiling, default16MiB, with maximum depth64.

    Returns
    -------
    RetainedReplayState
        Revalidated counters, identities, source anchor and codec receipt chain.

    Raises
    ------
    ValueError, TypeError, UnicodeError
        Invalid bound, duplicate keys, schema, lineage, graph or content bindings.

    Notes
    -----
    No migration, host call, source lookup, runtime scan or storage I/O occurs.
    A checkpoint commitment cannot authenticate discarded history by itself.
    """
    return RetainedReplayState.from_json(_bounded(payload, maximum_bytes))


def read_retained_continuation[S: CoreContract](
    payload: str | bytes, *, state_type: type[S], maximum_bytes: int = DEFAULT_MAXIMUM_BYTES
) -> RetainedContinuation[S]:
    """Read a complete v2 frontier and explicitly selected concrete host snapshot.

    Parameters
    ----------
    payload : str or bytes
        Complete declarative UTF-8 envelope, never executable pickle/class paths.
    state_type : type[S]
        Exact complete concrete host-owned CoreContract snapshot model.
    maximum_bytes : int, optional
        Joint encoded-byte ceiling, default16MiB, and maximum depth64.

    Returns
    -------
    RetainedContinuation[S]
        Immutable matched typed values. No owner is restored or activated.

    Raises
    ------
    ValueError, TypeError, UnicodeError
        Wrong schema/concrete codec, malformed JSON, bounds or unmatched state.

    Notes
    -----
    Existing v1 readers remain v1 and reject this generation; no implicit upgrade,
    current-runtime attestation, durable acknowledgement or effect replay occurs.
    """
    return _envelope_type(state_type).from_json(_bounded(payload, maximum_bytes))
