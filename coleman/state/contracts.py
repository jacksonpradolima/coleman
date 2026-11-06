"""Pure concrete algorithm snapshots and exact caller-owned codec bindings."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference
from coleman.core.primitives import Fingerprint
from coleman.objectives.bindings import DefinitionBinding


class StateBinding(CoreImmutable):
    """Identify one complete independently owned state and its interpretation.

    Attributes
    ----------
    owner : Reference
        Stable logical component identity and declared version.
    implementation : DefinitionBinding
        Exact declared algorithm/owner definition, not an executable import.
    state_codec : Reference
        Explicit concrete payload interpretation and codec generation.
    configuration_fingerprint : str
        Content fingerprint of all declared mathematical configuration.

    Notes
    -----
    No runtime is scanned or configuration inferred. These consistency bindings
    neither authenticate an implementation nor prove snapshot completeness.
    """

    owner: Reference
    implementation: DefinitionBinding
    state_codec: Reference
    configuration_fingerprint: Fingerprint


class AlgorithmState[P: CoreContract](CoreContract):
    """Wrap complete typed direct state without requiring durable infrastructure.

    Attributes
    ----------
    binding : StateBinding
        Exact owner, implementation, configuration and codec identities.
    payload : P
        Concrete versioned state owning sufficient statistics, parameters,
        transforms, counters, owned RNG/cache and required pending exactly once.

    Raises
    ------
    ValueError
        The generic payload type is missing or is only the CoreContract base.

    Notes
    -----
    Requires strict schema1 and an explicit concrete P. Preserves the canonical
    JSON codec without ambient metadata, SQLite, pickle, filesystem or transport.
    Snapshot completeness and side-effect-free export are host obligations.
    Existing Stateful implementations may continue using their own concrete
    CoreContract snapshots. A raw update acquires no automatic idempotency.
    """

    binding: StateBinding
    payload: P

    @model_validator(mode="after")
    def _concrete(self) -> Self:
        """Prevent a bare generic envelope from discarding concrete host fields."""
        arguments = type(self).__pydantic_generic_metadata__["args"]
        if (
            not arguments
            or not isinstance(arguments[0], type)
            or not issubclass(arguments[0], CoreContract)
            or arguments[0] is CoreContract
        ):
            raise ValueError("AlgorithmState requires an explicit concrete payload type")
        if type(self.payload) is not arguments[0]:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("AlgorithmState payload differs from its concrete codec")
        return self
