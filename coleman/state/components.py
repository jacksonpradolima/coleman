"""Finite caller-selected codecs and fresh staging for composed state owners."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Hashable
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, time
from decimal import Decimal
from math import isfinite
from types import MethodType
from typing import Any, Self, cast

from pydantic import BaseModel

from coleman.core.base import CoreContract, VersionedContract
from coleman.core.identity import Reference
from coleman.core.primitives import canonical_json, fingerprint
from coleman.core.protocols import Stateful
from coleman.core.reading import bounded_json_bytes
from coleman.replay.retained_contracts import RetainedContinuation
from coleman.state.contracts import StateBinding
from coleman.state.storage_contracts import ChildFrontier, CollectionManifest


def _scalar_signature(value: Any) -> Hashable:
    """Retain exact finite scalar representation during codec admission.

    Parameters
    ----------
    value : object
        Declared scalar field, after model validation and selected-reader parsing.

    Returns
    -------
    collections.abc.Hashable
        Type-tagged value; binary64 signed zero and Decimal quantum survive.

    Raises
    ------
    ValueError, TypeError
        Nonfinite numeric state or an opaque unhashable scalar representation.

    Notes
    -----
    Comparison metadata stays in memory. It captures no private model caches,
    executes no owner and is never substituted for a serialized fingerprint.
    """
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Component state requires finite float values")
        return type(value), value.hex()
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Component state requires finite Decimal values")
        return type(value), value.as_tuple()
    if isinstance(value, complex):
        if not isfinite(value.real) or not isfinite(value.imag):
            raise ValueError("Component state requires finite complex values")
        return type(value), (value.real.hex(), value.imag.hex())
    if isinstance(value, (datetime, time)):
        return type(value), (value.isoformat(), value.fold, getattr(value.tzinfo, "key", None))
    if not isinstance(value, Hashable):
        raise TypeError("Component state requires explicit portable scalar fields")
    return type(value), value


# Preserve a separate return for each original representation and callback/error path.
def _state_signature(value: Any) -> Hashable:  # pylint: disable=too-many-return-statements
    """Compare declared typed fields without invoking model equality overrides.

    Parameters
    ----------
    value : object
        Complete concrete model, nested dataclass/container or scalar field.

    Returns
    -------
    collections.abc.Hashable
        In-memory structural value retaining types, array order and mapping keys.

    Raises
    ------
    ValueError, TypeError
        A scalar cannot satisfy the portable finite-state comparison profile.

    Notes
    -----
    Reads only declared fields; private implementation caches are not captured.
    The host still owns snapshot completeness. Validators and serializers must
    remain pure and deterministic; this is not an arbitrary-Python sandbox.
    Exact builtin text, integers, booleans, bytes and None retain their existing
    type-tagged values without model/container introspection. Subclasses and
    other scalar representations follow the original comparison checks.
    """
    kind = type(value)
    if kind is str or kind is int or kind is bool or value is None or kind is bytes:
        return kind, value
    if isinstance(value, BaseModel):
        return type(value), tuple((name, _state_signature(getattr(value, name))) for name in type(value).model_fields)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value), tuple((item.name, _state_signature(getattr(value, item.name))) for item in fields(value))
    if isinstance(value, (tuple, list)):
        return type(value), tuple(_state_signature(item) for item in value)
    if isinstance(value, dict):
        return type(value), frozenset((_state_signature(key), _state_signature(item)) for key, item in value.items())
    if isinstance(value, (set, frozenset)):
        return type(value), frozenset(_state_signature(item) for item in value)
    return _scalar_signature(value)


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
        if self._canonical_state(value)[0] != payload:
            raise ValueError("Component bytes must use the exact canonical codec representation")
        return value

    def encode(self, state: S) -> StateBytes:
        """Encode one exact concrete state without truncating subclass fields.

        Parameters
        ----------
        state : S
            Exact declared model instance, including all owner state.
            The complete canonical envelope must fit within 16 MiB and 64 nested
            JSON containers, matching this codec's decoder.

        Returns
        -------
        StateBytes
            Canonical bytes and explicit outer binding, without file publication.

        Raises
        ------
        TypeError, ValueError
            Wrong concrete type, invalid fields, incompatible embedded binding,
            an encoded snapshot exceeding the decoder's byte/depth limits, or a
            serializer that the selected reader cannot reverse exactly.

        Notes
        -----
        Bounds apply before creating StateBytes or publishing an artifact. A
        structurally valid model can still exceed this fixed wire profile; it
        must reject before the optional store can acknowledge unreadable state.
        The actual selected model reader must recover the same declared typed
        fields, including nested values. Private implementation caches are not
        inferred; every value needed for continuation belongs in declared state.
        """
        return _encode_with_signature(self, state)[0]

    def _canonical_state(self, state: S) -> tuple[bytes, Hashable]:
        """Validate concrete binding and bounded canonical bytes without recursion.

        Parameters
        ----------
        state : S
            Exact concrete model selected by this codec.

        Returns
        -------
        tuple
            Bounded canonical UTF-8 bytes and declared-state comparison
            signature. The byte profile allows16MiB/depth64.

        Raises
        ------
        ValueError, TypeError
            Invalid model/type/binding, nonfinite or changing declared values,
            unsupported scalar state or encoded reader-profile overflow.

        Notes
        -----
        Calls trusted pure validators/serialization only. It never calls encode,
        decode, an owner or storage; both public methods share this primitive.
        After full validation, the unchanged inherited core writer can dump the
        checked value directly. Custom method-level writers still execute their
        own contract. The inherited route still executes Pydantic field/model
        serializers; selected-reader and typed round-trip checks apply to both.
        """
        if type(state) is not self.state_type:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Component value must have its exact declared concrete type")
        checked = self.state_type.model_validate(state)
        embedded = getattr(checked, "binding", None)
        if embedded is not None and embedded != self.binding:
            raise ValueError("Component embedded binding differs from its declared codec")
        encoded = (
            canonical_json(checked.model_dump(mode="json"))
            if self.state_type.to_json is VersionedContract.to_json
            else checked.to_json()
        )
        content = bounded_json_bytes(encoded, 16 * 1024**2)
        signature = _state_signature(checked)
        if checked is not state and signature != _state_signature(state):
            raise ValueError("Component validation must preserve the exact complete declared state")
        return content, signature


def _encode_with_signature[S: CoreContract](codec: ComponentCodec[S], state: S) -> tuple[StateBytes, Hashable]:
    """Run the full encoder and retain its own signature before its reader.

    Both outputs come from this actual admission call. No caller can supply a
    signature, validation decision or encoded witness. Callback order, complete
    canonical/binding/bounds checks and public encoder errors remain identical.
    """
    content, signature = codec._canonical_state(state)  # pylint: disable=protected-access
    restored = codec.decode(content)
    if signature != _state_signature(restored):
        raise ValueError("Component codec must recover the exact complete declared state")
    return StateBytes(codec.binding, content), signature


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

    @classmethod
    def from_bytes(cls, codec: ComponentCodec[S], payload: bytes) -> Self:
        """Admit complete owned bytes through both actual selected-reader calls.

        Parameters
        ----------
        codec : ComponentCodec
            Explicit complete concrete interpretation and immutable outer binding.
            Custom codec/value subclasses and overridden hooks retain the
            ordinary decode-then-constructor sequence.
        payload : bytes
            Immutable complete canonical UTF-8 JSON, at most 16 MiB/depth 64.

        Returns
        -------
        Self
            The first freshly decoded complete value and its original codec.
            No owner, file, factory or durable acknowledgement is accessed.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Invalid outer binding, mutable/invalid/oversized bytes, unsupported
            fields/versions, noncanonical serialization or nonrepeatable declared
            state, including reader defaults excluded from the JSON writer.

        Notes
        -----
        Both actual selected readers run. The first complete typed signature is
        captured before the second read, so that read cannot hide a change by
        mutating the first object. Each stock decode retains its full concrete
        validation, binding, canonical-byte and typed-preservation checks. Only
        the redundant intervening encode is omitted for unmodified base codecs
        and value construction hooks. No caller-supplied validation flag, cache
        or trust token is accepted. Ordinary construction and encoding retain
        their existing independent admission behavior and exact byte format.
        Selected callbacks retain the existing pure, deterministic codec
        contract; framework implementations remain unchanged during the call.
        """
        if not _stock_value_reader(cls, codec):
            return cls(codec, codec.decode(payload))
        first = codec.decode(payload)
        signature = _state_signature(first)
        second = codec.decode(payload)
        if signature != _state_signature(second):
            raise ValueError("Component codec must recover the exact complete declared state")
        StateBytes(codec.binding, payload)
        result = object.__new__(cls)
        object.__setattr__(result, "codec", codec)
        object.__setattr__(result, "value", first)
        return result


# Immutable dispatch identities, not snapshot/validation caches. Exact class
# descriptors and bound methods preserve all existing custom admission hooks.
_STOCK_CODEC_METHODS = tuple((name, ComponentCodec.__dict__[name]) for name in ("decode", "encode", "_canonical_state"))
_STOCK_CODEC_FIELDS = tuple(
    (name, name in ComponentCodec.__dict__, ComponentCodec.__dict__.get(name)) for name in ("binding", "state_type")
)
_STOCK_VALUE_HOOKS = tuple(
    (name, name in ComponentValue.__dict__, ComponentValue.__dict__.get(name))
    for name in ("__init__", "__post_init__", "__new__", "__getattribute__", "codec", "value")
)
_STOCK_CODEC_GETATTRIBUTE = ComponentCodec.__getattribute__


def _stock_value_reader(cls: type[ComponentValue[Any]], codec: ComponentCodec[Any]) -> bool:
    """Select only unchanged own dispatch; preserve every custom fallback hook."""
    if cls is not ComponentValue or type(codec) is not ComponentCodec:  # pylint: disable=unidiomatic-typecheck
        return False
    if any(
        (name in ComponentValue.__dict__) != present or ComponentValue.__dict__.get(name) is not expected
        for name, present, expected in _STOCK_VALUE_HOOKS
    ):
        return False
    if ComponentCodec.__getattribute__ is not _STOCK_CODEC_GETATTRIBUTE or any(
        (name in ComponentCodec.__dict__) != present or ComponentCodec.__dict__.get(name) is not expected
        for name, present, expected in _STOCK_CODEC_FIELDS
    ):
        return False
    storage = object.__getattribute__(codec, "__dict__")
    for name, expected in _STOCK_CODEC_METHODS:
        if name in storage or ComponentCodec.__dict__.get(name) is not expected:
            return False
        method = getattr(codec, name)
        # Exact stock binding is required; accepting custom dispatch skips its hooks.
        if type(method) is not MethodType or method.__self__ is not codec or method.__func__ is not expected:  # pylint: disable=unidiomatic-typecheck
            return False
    return True


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


@dataclass(frozen=True)
class _RestoreExport:
    """Retain this operation's actual full encode and pre-reader typed state."""

    value: ComponentValue[Any]
    encoded: StateBytes
    signature: Hashable


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

    def _export_for_restore(self) -> tuple[_RestoreExport, ...]:
        """Capture real exports only for this stock restoration comparison.

        Returns
        -------
        tuple
            Complete values, actual canonical encode results and typed signatures
            captured before their selected readers. No byte or field is omitted.

        Raises
        ------
        ValueError, TypeError
            Changed stock dispatch or a value mutated during codec admission.
            The enclosing stage invalidates itself on any failure.

        Notes
        -----
        This private path retains the ordinary constructor's actual full encode.
        Only its already-audited bytes are reused within the same comparison;
        the second selected reader still runs before restoration is accepted.
        Ordinary export and publication paths remain independent full audits.
        """
        if not _stock_restore_stage(self):
            raise ValueError("Stock stage admission changed during owner callbacks")
        exports = []
        for binding, owner in self._entries:
            codec = binding.codec
            if not _stock_value_reader(ComponentValue, codec):
                raise ValueError("Stock restore admission changed during owner callbacks")
            snapshot = cast(Stateful[Any], owner).export_state()
            encoded, signature = _encode_with_signature(codec, snapshot)
            if signature != _state_signature(snapshot):
                raise ValueError("Restored component changed during codec admission")
            value = object.__new__(ComponentValue)
            object.__setattr__(value, "codec", codec)
            object.__setattr__(value, "value", snapshot)
            exports.append(_RestoreExport(value, encoded, signature))
        return tuple(exports)


# Dispatch metadata only: no snapshots, bytes or validation decisions are cached.
_STOCK_RESTORE_NAMESPACES = tuple(
    (cls, tuple(cls.__dict__.items())) for cls in (ComponentValue, ComponentBinding, StagedCollection)
)


def _restore_types_unchanged() -> bool:
    """Check stock descriptors before looking up any caller's component fields."""
    return all(
        len(cls.__dict__) == len(expected)
        and all(name in cls.__dict__ and cls.__dict__[name] is value for name, value in expected)
        for cls, expected in _STOCK_RESTORE_NAMESPACES
    )


def _stock_restore_exports(
    values: tuple[ComponentValue[Any], ...], bindings: tuple[ComponentBinding[Any, Any], ...]
) -> bool:
    """Choose a whole stock operation before any owner export is invoked."""
    # Preserve the existing bounded-inventory error path without enumerating a
    # generator or touching a custom value/binding attribute during selection.
    # pylint: disable=unidiomatic-typecheck
    if type(values) is not tuple or type(bindings) is not tuple or not _restore_types_unchanged():
        return False
    if not 1 <= len(values) <= 128 or not 1 <= len(bindings) <= 128:
        return False
    if any(type(item) is not ComponentValue for item in values) or any(
        type(item) is not ComponentBinding for item in bindings
    ):
        return False
    codecs = tuple(item.codec for item in values) + tuple(item.codec for item in bindings)
    return all(
        _stock_value_reader(ComponentValue, codec) and codec.state_type.to_json is VersionedContract.to_json
        for codec in codecs
    )


def _stock_restore_stage(staged: StagedCollection) -> bool:
    """Preserve subclass or per-instance export hooks through the old path."""
    if type(staged) is not StagedCollection or not _restore_types_unchanged():  # pylint: disable=unidiomatic-typecheck
        return False
    storage = object.__getattribute__(staged, "__dict__")
    return not any(name in storage for name in ("_export", "_export_for_restore"))


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
