"""Declarative state for an explicitly owned standard-library MT19937 stream."""

from __future__ import annotations

import random
from typing import Annotated, Literal

from pydantic import Field

from coleman.core.base import CoreContract
from coleman.core.numeric import Binary64

_UInt32 = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]


class PythonRandomState(CoreContract):
    """Retain the complete version-three ``random.Random`` state as JSON data.

    Attributes
    ----------
    schema_version : int
        Required integer 1 for this declarative envelope.
    codec : {'python-random-mt19937-v1'}
        Exact layout of Python's MT19937 state, not a new random algorithm.
    state_version : int
        Required integer 3, the supported ``Random.getstate`` generation.
    words : tuple of int
        Exactly 624 unsigned 32-bit words, in backend order.
    index : int
        Backend index from 0 through 624, inclusive.
    gauss_next : float or None
        Required cached unit-normal variate, including valid zero or negatives;
        null explicitly means that no Gaussian variate is cached.

    Raises
    ------
    ValueError
        Unknown versions, incorrect word count, coercions, nonfinite Gaussian
        cache or out-of-range integer data.

    Notes
    -----
    This pure value performs no draws, mutation, I/O or executable decoding.
    A caller's buffered bits belong to its own host state and must also be
    retained. Restore qualification binds the actual interpreter/runtime; this
    layout is not a blanket cross-version sampling or independence guarantee.
    """

    codec: Literal["python-random-mt19937-v1"]
    state_version: Annotated[int, Field(strict=True, ge=3, le=3)]
    words: Annotated[tuple[_UInt32, ...], Field(min_length=624, max_length=624)]
    index: Annotated[int, Field(strict=True, ge=0, le=624)]
    gauss_next: Binary64 | None


def _owned(generator: random.Random) -> None:
    """Require the concrete backend without running subclass conversion hooks."""
    # Exact type excludes custom getstate/setstate implementations.
    # pylint: disable=unidiomatic-typecheck
    if type(generator) is not random.Random:
        raise TypeError("RNG state requires a concrete caller-owned random.Random")


def capture_random(generator: random.Random) -> PythonRandomState:
    """Copy a caller-owned RNG's complete state without consuming a draw.

    Parameters
    ----------
    generator : random.Random
        Concrete independently owned instance. The caller maintains exclusive
        access; a module-global stream or shared mutable instance is unsuitable.

    Returns
    -------
    PythonRandomState
        Versioned words, index and explicit Gaussian cache. Seeds/draw counts
        are not substituted for the current state.

    Raises
    ------
    TypeError
        The backend is not the concrete standard-library ``Random`` class.
    ValueError
        The actual backend layout or cached value violates this codec.

    Notes
    -----
    Performs no I/O or mutation. It cannot prove the caller's exclusive ownership
    or make concurrent use safe; snapshot at the caller's component barrier.
    """
    _owned(generator)
    version, internal, gaussian = generator.getstate()
    return PythonRandomState(
        schema_version=1,
        codec="python-random-mt19937-v1",
        state_version=version,
        words=internal[:-1],
        index=internal[-1],
        gauss_next=gaussian,
    )


def restore_random(generator: random.Random, state: PythonRandomState) -> None:
    """Validate completely, then replace only the caller-owned RNG's state.

    Parameters
    ----------
    generator : random.Random
        Concrete independently owned target with exclusive caller access.
    state : PythonRandomState
        Complete compatible codec value, including cached Gaussian data.

    Raises
    ------
    TypeError
        Target or snapshot has the wrong concrete type.
    ValueError
        Revalidation rejects a version, layout, word, index or cached value.
        These checks precede ``setstate`` and leave the target unchanged.

    Notes
    -----
    Calls the target's real ``setstate`` once after validation; performs no draw,
    seed initialization, file access or restore of unrelated streams. The host
    restores any separately owned bit reservoir at the same component barrier.
    No migration or unqualified cross-runtime compatibility is inferred.
    """
    _owned(generator)
    # pylint: disable=unidiomatic-typecheck
    if type(state) is not PythonRandomState:
        raise TypeError("RNG restore requires the concrete PythonRandomState codec")
    validated = PythonRandomState.model_validate(state)
    native = validated.state_version, (*validated.words, validated.index), validated.gauss_next
    generator.setstate(native)
