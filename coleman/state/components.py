"""Finite caller-selected codecs and fresh staging for composed state owners."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from coleman.core.base import CoreContract
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.core.protocols import Stateful
from coleman.core.reading import bounded_json_bytes
from coleman.replay.retained_contracts import RetainedContinuation
from coleman.state.contracts import StateBinding
from coleman.state.storage_contracts import ChildFrontier, CollectionManifest


@dataclass(frozen=True)
class StateBytes:
    """Hold canonical complete bytes for one explicitly bound state owner.

    Parameters
    ----------
    binding : StateBinding
        Exact caller-selected owner interpretation.
    content : bytes
        Complete canonical UTF-8 JSON, without a trailing newline. The concrete
        codec validates content; this wrapper validates only immutable inputs.

    Notes
    -----
    Construction performs no file I/O, capture or host restore.
    """

    binding: StateBinding
    content: bytes

    def __post_init__(self) -> None:
        """Reject unchecked bindings and mutable/non-byte content."""
        if type(self.binding) is not StateBinding:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("State bytes require an exact immutable StateBinding")
        StateBinding.model_validate(self.binding)
        if type(self.content) is not bytes or not self.content:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("State bytes require nonempty immutable bytes")

    @property
    def fingerprint(self) -> str:
        """Return exact encoded-content SHA-256, without I/O.

        Returns
        -------
        str
            COLEMAN sha256-prefixed digest of the complete canonical bytes.
        """
        return "sha256:" + hashlib.sha256(self.content).hexdigest()

    @property
    def byte_count(self) -> int:
        """Return the positive exact encoded byte count.

        Returns
        -------
        int
            UTF-8 bytes, independent of character count or allocated memory.
        """
        return len(self.content)


@dataclass(frozen=True)
class ComponentCodec[S: CoreContract]:
    """Select one concrete immutable codec in trusted application code.

    Parameters
    ----------
    binding : StateBinding
        Exact outer interpretation; inner replay host codecs may differ.
    state_type : type of CoreContract
        Concrete complete state type, never a JSON-selected import or base type.

    Notes
    -----
    Trusted model validators must be pure and bounded. No registry is scanned.
    """

    binding: StateBinding
    state_type: type[S]

    def __post_init__(self) -> None:
        """Require an explicit complete model and validated immutable binding."""
        if type(self.binding) is not StateBinding:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Component codec requires an exact immutable StateBinding")
        StateBinding.model_validate(self.binding)
        if (
            not isinstance(self.state_type, type)
            or not issubclass(self.state_type, CoreContract)
            or self.state_type is CoreContract
        ):
            raise TypeError("Component codec requires an explicit concrete state type")
        if self.state_type.__pydantic_generic_metadata__.get("parameters"):
            raise TypeError("Component codec cannot erase concrete generic parameters")

    def decode(self, payload: bytes) -> S:
        """Read exact canonical bounded bytes without constructing an owner.

        Parameters
        ----------
        payload : bytes
            Complete UTF-8 JSON, at most16MiB and lexical depth64.

        Returns
        -------
        S
            Concrete revalidated complete state, retaining every declared field.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Invalid type/version/JSON, noncanonical bytes or embedded binding.

        Notes
        -----
        Performs no migration, runtime scan, filesystem access or host restore.
        """
        if type(payload) is not bytes:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Component decode requires immutable bytes")
        value = self.state_type.from_json(bounded_json_bytes(payload, 16 * 1024**2))
        if self.encode(value).content != payload:
            raise ValueError("Component bytes must use the exact canonical codec representation")
        return value

    def encode(self, state: S) -> StateBytes:
        """Encode one exact concrete state without truncating subclass fields.

        Parameters
        ----------
        state : S
            Exact declared model instance, including all owner state.

        Returns
        -------
        StateBytes
            Canonical bytes and explicit outer binding, without file publication.

        Raises
        ------
        TypeError, ValueError
            Wrong concrete type, invalid fields or incompatible embedded binding.
        """
        if type(state) is not self.state_type:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Component value must have its exact declared concrete type")
        checked = self.state_type.model_validate(state)
        embedded = getattr(checked, "binding", None)
        if embedded is not None and embedded != self.binding:
            raise ValueError("Component embedded binding differs from its declared codec")
        return StateBytes(self.binding, checked.to_json().encode("utf-8"))


@dataclass(frozen=True)
class ComponentValue[S: CoreContract]:
    """Pair a typed complete snapshot with its explicit concrete codec.

    Parameters
    ----------
    codec : ComponentCodec
        Caller-selected finite interpretation.
    value : S
        Exact complete immutable snapshot; construction validates canonical form.

    Notes
    -----
    No owner, file or registry is accessed.
    """

    codec: ComponentCodec[S]
    value: S

    def __post_init__(self) -> None:
        """Validate the actual concrete value without erasing its fields."""
        self.codec.encode(self.value)


@dataclass(frozen=True)
class ComponentBinding[S: CoreContract, H]:
    """Provide a fresh staging owner and pure declared-frontier inspector.

    Parameters
    ----------
    codec : ComponentCodec
        Exact complete state interpretation.
    new_owner : callable
        Trusted zero-argument factory returning a fresh Stateful owner. It must
        not return a live application owner or perform external side effects.
    frontier : callable
        Pure bounded state inspector returning the declared ChildFrontier or
        explicit None for a direct component with publication-only semantics.

    Notes
    -----
    Python callbacks are trusted, not sandboxed. Every component is decoded
    before any factory runs, and every restored owner is re-exported exactly.
    """

    codec: ComponentCodec[S]
    new_owner: Callable[[], H]
    frontier: Callable[[S], ChildFrontier | None]


@dataclass(frozen=True)
class MigrationBindings:
    """Select explicit old readers and new fresh owners for a known mapping.

    Parameters
    ----------
    old_codecs : tuple of ComponentCodec
        Complete finite concrete source inventory.
    new_bindings : tuple of ComponentBinding
        Complete finite target owner inventory.

    Notes
    -----
    Contains no executable import paths and performs no automatic migration.
    """

    old_codecs: tuple[ComponentCodec[Any], ...]
    new_bindings: tuple[ComponentBinding[Any, Any], ...]


@dataclass(frozen=True)
class PreparedCollection:
    """Expose completely validated immutable snapshots without live owners.

    Parameters
    ----------
    manifest : CollectionManifest
        Complete acknowledged manifest already checked by the local adapter.
    values : tuple of ComponentValue, optional
        Concrete decoded values supplied only after complete inventory checks.

    Notes
    -----
    Values retain their actual types. Calling component performs no restore.
    """

    manifest: CollectionManifest
    values: tuple[ComponentValue[Any], ...] = ()

    def component[S: CoreContract](self, codec: ComponentCodec[S]) -> S:
        """Select an exact caller-typed component from the prepared collection.

        Parameters
        ----------
        codec : ComponentCodec
            Exact concrete requested interpretation and binding.

        Returns
        -------
        S
            Complete immutable snapshot.

        Raises
        ------
        ValueError
            Missing owner or different binding/concrete codec type.
        """
        for item in self.values:
            if item.codec.binding == codec.binding and item.codec.state_type is codec.state_type:
                return cast(S, item.value)
        raise ValueError("Requested concrete component is absent or incompatible")


class StagedCollection:  # pylint: disable=too-few-public-methods
    """Hold fresh restored owners for exactly one trusted local transition.

    Parameters
    ----------
    entries : tuple
        Adapter-supplied pairs of exact ComponentBinding and distinct fresh owner.

    Notes
    -----
    Invalidation revokes future lookups, not references retained by arbitrary
    trusted Python. Callbacks must not leak staged owners or external effects.
    """

    def __init__(self, entries: tuple[tuple[ComponentBinding[Any, Any], Any], ...]) -> None:
        """Store fresh owners until the adapter invalidates the stage.

        Parameters
        ----------
        entries : tuple
            Complete checked binding/owner inventory, with no live-owner reuse.
        """
        self._entries = entries
        self._active = True

    def owner[S: CoreContract, H](self, binding: ComponentBinding[S, H]) -> H:
        """Return one exact fresh owner while this transition is active.

        Parameters
        ----------
        binding : ComponentBinding
            Exact original trusted binding, including factory and inspector.

        Returns
        -------
        H
            Fresh restored owner for the current staged transition only.

        Raises
        ------
        ValueError
            Closed stage, missing owner or a different in-process binding.
        """
        if not self._active:
            raise ValueError("State stage is no longer active")
        for expected, owner in self._entries:
            if expected is binding:
                return cast(H, owner)
        raise ValueError("State stage does not contain this exact owner binding")

    def _close(self) -> None:
        """Invalidate lookups after success or failure; never mutate old owners."""
        self._active = False

    def _export(self) -> tuple[ComponentValue[Any], ...]:
        """Read every actual complete staged owner through its explicit codec."""
        return tuple(
            ComponentValue(binding.codec, cast(Stateful[Any], owner).export_state()) for binding, owner in self._entries
        )


def child_frontier(continuation: RetainedContinuation[Any], *, child: Reference) -> ChildFrontier:
    """Extract an actual standard replay frontier without guessing host fields.

    Parameters
    ----------
    continuation : RetainedContinuation
        Complete concrete schema2 replay continuation selected by the host.
    child : Reference
        Exact declared collection consumer.

    Returns
    -------
    ChildFrontier
        Actual global counters, two clocks, exclusive floor and semantic digests.

    Raises
    ------
    TypeError, ValueError
        Wrong generation, incomplete concrete state or invalid child identity.

    Notes
    -----
    Performs no host restore, RNG draw, package capture or I/O.
    """
    if not isinstance(continuation, RetainedContinuation):
        raise TypeError("Replay frontier requires a complete RetainedContinuation")
    continuation = type(continuation).model_validate(continuation)
    state = continuation.state
    return ChildFrontier(
        schema_version=1,
        child=child,
        source_epoch=state.retention.source_epoch,
        stream=state.policy.stream,
        source=state.policy.source,
        lineage=state.policy.lineage,
        decision_count=state.decision_count,
        update_count=state.update_count,
        next_sequence=state.next_sequence,
        sealed_floor=0 if state.seal is None else state.seal.decision_count,
        eligible_through=state.eligible_through,
        apply_at=state.apply_at,
        causal_commitment=state.causal_prefix_fingerprint,
        state_fingerprint=state.fingerprint,
        host_fingerprint=state.host_fingerprint,
        policy_fingerprint=state.policy.fingerprint,
        retention_fingerprint=fingerprint(state.retention.model_dump(mode="json")),
        state_codec=state.state_codec,
    )
