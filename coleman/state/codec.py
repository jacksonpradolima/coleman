"""Bounded pure direct-state reading with explicit concrete caller codecs."""

from __future__ import annotations

from typing import cast

from coleman.core.base import CoreContract
from coleman.core.reading import bounded_json_bytes
from coleman.state.contracts import AlgorithmState, StateBinding

DEFAULT_MAXIMUM_BYTES = 16 * 1024**2


def read_algorithm_state[P: CoreContract](
    payload: str | bytes,
    *,
    payload_type: type[P],
    expected: StateBinding,
    maximum_bytes: int = DEFAULT_MAXIMUM_BYTES,
) -> AlgorithmState[P]:
    """Read one strict complete snapshot without restoring a live owner.

    Parameters
    ----------
    payload : str or bytes
        Complete bounded UTF-8 JSON content, not a path or executable pickle.
    payload_type : type of CoreContract
        Exact concrete host payload type selected by trusted caller code.
    expected : StateBinding
        Exact expected owner/configuration/implementation/codec binding.
    maximum_bytes : int, optional
        Positive encoded-byte limit, default16MiB; lexical depth is at most64.

    Returns
    -------
    AlgorithmState
        Revalidated concrete payload with exactly the supplied expected binding.

    Raises
    ------
    ValueError, TypeError, UnicodeError
        Invalid bounds/JSON/versions, unspecified type or incompatible binding.

    Notes
    -----
    No implicit migration, dynamic class lookup, host restore, RNG draw,
    persistence, runtime scan or I/O occurs. The supplied declarative model may
    run its trusted validators; this is not a sandbox for arbitrary Python.
    """
    if not isinstance(payload_type, type) or not issubclass(payload_type, CoreContract) or payload_type is CoreContract:
        raise TypeError("AlgorithmState reader requires an explicit concrete payload_type")
    binding = StateBinding.model_validate(expected)
    contract = cast(type[AlgorithmState[P]], AlgorithmState.__class_getitem__(payload_type))
    state = contract.from_json(bounded_json_bytes(payload, maximum_bytes))
    if state.binding != binding:
        raise ValueError("AlgorithmState differs from its expected complete binding")
    return state
