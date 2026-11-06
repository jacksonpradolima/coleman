"""Private complete-inventory validation before fresh owner construction."""

from __future__ import annotations

from typing import Any, cast

from coleman.core.protocols import Stateful
from coleman.replay.retained_contracts import RetainedContinuation
from coleman.state import _disk
from coleman.state.components import (
    ComponentBinding,
    ComponentCodec,
    ComponentValue,
    PreparedCollection,
    StagedCollection,
    StateBytes,
    child_frontier,
)
from coleman.state.storage_contracts import (
    ChildFrontier,
    CollectionDefinition,
    CollectionManifest,
    CommitRequest,
    ComponentSpec,
    ExecutionCompatibility,
    MigrationReceipt,
    StateBlobRef,
    StorageLimits,
)

type Bindings = tuple[ComponentBinding[Any, Any], ...]
type Values = tuple[ComponentValue[Any], ...]
type Codecs = tuple[ComponentCodec[Any], ...]


def _bounded(values: object) -> None:
    """Reject non-tuple/oversize in-process inventories before enumeration."""
    if type(values) is not tuple or not 1 <= len(values) <= 128:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Concrete state inventories require a tuple of one through128 components")


def bound_codecs(bindings: Bindings) -> Codecs:
    """Select a finite concrete inventory without invoking any host callback.

    Parameters
    ----------
    bindings : tuple of ComponentBinding
        One through128 explicit trusted in-process component bindings.

    Returns
    -------
    tuple of ComponentCodec
        Concrete codecs in the supplied enumeration order.

    Raises
    ------
    ValueError, TypeError
        Mutable/unbounded inventory or a nonbinding entry, before enumeration
        of an unsupported container or any factory/inspector/restore call.
    """
    _bounded(bindings)
    if any(not isinstance(item, ComponentBinding) for item in bindings):
        raise TypeError("State bindings require concrete ComponentBinding entries")
    return tuple(item.codec for item in bindings)


def inventory(definition: CollectionDefinition, codecs: Codecs) -> Codecs:
    """Require exactly one concrete codec for every declared owner.

    Parameters
    ----------
    definition : CollectionDefinition
        Exact complete component inventory.
    codecs : tuple of ComponentCodec
        Finite caller-selected concrete readers, with no dynamic import lookup.

    Returns
    -------
    tuple of ComponentCodec
        Canonical owner order after exact binding/inventory comparison.

    Raises
    ------
    ValueError
        Missing/duplicate/extra owner or incompatible complete binding.

    Notes
    -----
    Sorts descriptors only; no decoder, factory, owner, file or RNG is called.
    """
    _bounded(codecs)
    if any(not isinstance(item, ComponentCodec) for item in codecs):
        raise TypeError("State inventory requires concrete ComponentCodec entries")
    ordered = tuple(sorted(codecs, key=lambda item: item.binding.owner.id))
    if tuple(item.binding for item in ordered) != tuple(item.binding for item in definition.components):
        raise ValueError("Concrete codecs must match the exact complete declared component inventory")
    return ordered


def encoded(definition: CollectionDefinition, values: Values, limits: StorageLimits) -> tuple[StateBytes, ...]:
    """Encode complete typed values before any fresh owner is constructed.

    Parameters
    ----------
    definition : CollectionDefinition
        Required exact owner inventory.
    values : tuple of ComponentValue
        Actual concrete snapshots selected by trusted caller codecs.
    limits : StorageLimits
        Per-component and total canonical encoded-byte ceilings.

    Returns
    -------
    tuple of StateBytes
        Complete canonical bytes in declared owner order.

    Raises
    ------
    ValueError, TypeError
        Inventory/type/binding mismatch or encoded capacity overflow.

    Notes
    -----
    Calls trusted pure model/codec validation, not factories/restores or file I/O.
    Unknown callback errors propagate; codecs are not a Python sandbox.
    """
    _bounded(values)
    if any(not isinstance(item, ComponentValue) for item in values):
        raise TypeError("State values require concrete ComponentValue entries")
    inventory(definition, tuple(item.codec for item in values))
    ordered = tuple(sorted(values, key=lambda item: item.codec.binding.owner.id))
    result: list[StateBytes] = []
    total = 0
    for item in ordered:
        content = item.codec.encode(item.value)
        if content.byte_count > limits.maximum_component_bytes:
            raise ValueError("Component exceeds its persisted encoded byte ceiling")
        total += content.byte_count
        if total > limits.maximum_collection_bytes:
            raise ValueError("Complete collection exceeds its persisted encoded byte ceiling")
        result.append(content)
    return tuple(result)


def frontiers(definition: CollectionDefinition, values: Values, bindings: Bindings) -> tuple[ChildFrontier, ...]:
    """Verify trusted inspectors against each actual replay continuation.

    Parameters
    ----------
    definition : CollectionDefinition
        Complete owner roles, child identities and explicit frontier profiles.
    values : tuple of ComponentValue
        Already validated complete typed snapshots in the candidate inventory.
    bindings : tuple of ComponentBinding
        Exact concrete codecs and trusted pure frontier inspectors.

    Returns
    -------
    tuple of ChildFrontier
        Actual canonical replay-child frontiers; direct components contribute none.

    Raises
    ------
    ValueError, TypeError, KeyError
        Incomplete/wrong inventory, codec mismatch, invalid replay state or a
        declared inspector differing from the standard concrete frontier.

    Notes
    -----
    Calls inspectors and model validation only; no factory/restore/update/file I/O.
    Custom inspectors remain trusted code and must be pure and bounded.
    """
    _bounded(values)
    inventory(definition, bound_codecs(bindings))
    actual = {item.codec.binding.owner.id: item for item in values}
    inspectors = {item.codec.binding.owner.id: item for item in bindings}
    result = []
    for spec in definition.components:
        item = actual[spec.binding.owner.id]
        binding = inspectors[spec.binding.owner.id]
        if binding.codec.state_type is not item.codec.state_type:
            raise ValueError("Supplied component binding has a different concrete state type")
        inspected = binding.frontier(item.value)
        if spec.frontier_profile == "none":
            if inspected is not None:
                raise ValueError("Publication-only owner cannot invent a replay frontier")
        else:
            expected = _replay_frontier(spec, item, inspected)
            result.append(expected)
    return tuple(sorted(result, key=lambda item: item.child.id))


def _replay_frontier(spec: ComponentSpec, item: ComponentValue[Any], inspected: ChildFrontier | None) -> ChildFrontier:
    """Verify one already-inspected replay owner against its concrete state.

    Parameters
    ----------
    spec : ComponentSpec
        Complete declared replay component and its child identity.
    item : ComponentValue
        Validated supplied owner state with the exact caller-selected codec.
    inspected : ChildFrontier or None
        Result of the trusted inspector already called by ``frontiers``.

    Returns
    -------
    ChildFrontier
        Standard frontier derived from the complete actual continuation.

    Raises
    ------
    ValueError, TypeError
        Missing concrete continuation/child or a mismatched inspector result.

    Notes
    -----
    Revalidates and reads declared state only; calls no inspector, factory,
    restore operation, RNG or filesystem API.
    """
    if not isinstance(item.value, RetainedContinuation) or spec.child is None:
        raise ValueError("Replay owner requires a complete concrete RetainedContinuation")
    expected = child_frontier(item.value, child=spec.child)
    if inspected != expected:
        raise ValueError("Inspector differs from the actual replay state frontier")
    return expected


def build(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    definition: CollectionDefinition,
    compatibility: ExecutionCompatibility,
    request: CommitRequest,
    values: Values,
    bindings: Bindings,
    limits: StorageLimits,
    migration: MigrationReceipt | None = None,
) -> PreparedCollection:
    """Validate one complete candidate before publishing any byte.

    Parameters
    ----------
    definition : CollectionDefinition
        Exact current or explicitly proposed migration target inventory.
    compatibility : ExecutionCompatibility
        Explicit actual target runtime and required copied-owner/mapping bindings.
    request : CommitRequest
        Complete command and original acknowledged predecessor.
    values : tuple of ComponentValue
        Every actual concrete owner snapshot, exactly once.
    bindings : tuple of ComponentBinding
        Every exact concrete codec and pure frontier inspector.
    limits : StorageLimits
        Per-component, collection and manifest encoded-byte ceilings.
    migration : MigrationReceipt or None, optional
        Explicit known mapping evidence only for codec_migration.

    Returns
    -------
    PreparedCollection
        Complete validated manifest and concrete values in canonical owner order.

    Raises
    ------
    ValueError, TypeError
        Version/type/binding/inventory/frontier/ancestry mismatch or capacity overflow.

    Notes
    -----
    Encodes and calls trusted inspectors without owner construction, restore,
    publication or acknowledgement. A prepared manifest is not a committed head.
    """
    content = encoded(definition, values, limits)
    actual = frontiers(definition, values, bindings)
    manifest = CollectionManifest(
        schema_version=1,
        definition=definition,
        compatibility=compatibility,
        intent=request.intent,
        predecessor=request.expected,
        components=tuple(
            StateBlobRef(binding=item.binding, fingerprint=item.fingerprint, byte_count=item.byte_count)
            for item in content
        ),
        frontiers=actual,
        migration=migration,
    )
    if len(manifest.to_json().encode("utf-8")) > limits.maximum_manifest_bytes:
        raise ValueError("Complete manifest exceeds its persisted encoded byte ceiling")
    return PreparedCollection(manifest, tuple(sorted(values, key=lambda item: item.codec.binding.owner.id)))


def decode(store: Any, manifest: CollectionManifest, codecs: Codecs) -> PreparedCollection:
    """Acquire every acknowledged payload before constructing any owner.

    Parameters
    ----------
    store : LocalStateStore
        Owned trusted adapter exposing its root and persisted limits.
    manifest : CollectionManifest
        Complete exact acknowledged manifest already selected by actual metadata.
    codecs : tuple of ComponentCodec
        Finite complete trusted reader inventory matching the manifest.

    Returns
    -------
    PreparedCollection
        All concrete decoded states and the original acknowledged manifest.

    Raises
    ------
    ValueError, TypeError, UnicodeError, OSError
        Missing/corrupt/oversized payload, byte/digest mismatch or invalid codec.

    Notes
    -----
    Reads bounded files and invokes trusted decoder/model validation. It never
    constructs/restores an owner or falls back to an older publication.
    """
    ordered = inventory(manifest.definition, codecs)
    if any(item.byte_count > store.limits.maximum_component_bytes for item in manifest.components):
        raise ValueError("Acknowledged component exceeds its persisted encoded byte ceiling")
    if sum(item.byte_count for item in manifest.components) > store.limits.maximum_collection_bytes:
        raise ValueError("Acknowledged collection exceeds its persisted byte ceiling before acquisition")
    blobs = tuple(
        _disk.acquire(
            _disk.artifact(store.root, "blobs", item.fingerprint),
            min(item.byte_count, store.limits.maximum_component_bytes),
            expected=item.fingerprint,
        )
        for item in manifest.components
    )
    if any(len(blob) != item.byte_count for blob, item in zip(blobs, manifest.components, strict=True)):
        raise ValueError("Acknowledged component byte count differs from its manifest")
    if sum(map(len, blobs)) > store.limits.maximum_collection_bytes:
        raise ValueError("Acknowledged collection exceeds its persisted byte ceiling")
    values = tuple(ComponentValue(codec, codec.decode(blob)) for codec, blob in zip(ordered, blobs, strict=True))
    return PreparedCollection(manifest, values)


def verify(prepared: PreparedCollection, bindings: Bindings) -> None:
    """Check every concrete frontier before a staging owner factory runs.

    Parameters
    ----------
    prepared : PreparedCollection
        Complete decoded values and their manifest.
    bindings : tuple of ComponentBinding
        Exact trusted concrete codecs and pure frontier inspectors.

    Raises
    ------
    ValueError, TypeError
        Actual decoded frontiers differ from the manifest or a binding is invalid.

    Notes
    -----
    Calls pure inspectors/model validation; no factory, restore, update or I/O.
    """
    if frontiers(prepared.manifest.definition, prepared.values, bindings) != prepared.manifest.frontiers:
        raise ValueError("Actual decoded frontiers differ from the acknowledged manifest")


def stage(prepared: PreparedCollection, bindings: Bindings) -> StagedCollection:
    """Restore distinct fresh owners only after complete typed validation.

    Parameters
    ----------
    prepared : PreparedCollection
        Complete decoded candidate, including every pending/cache/RNG field.
    bindings : tuple of ComponentBinding
        Exact trusted fresh factories, codecs and pure frontier inspectors.

    Returns
    -------
    StagedCollection
        Fresh owners after all restores and complete re-exports match exactly.

    Raises
    ------
    ValueError, TypeError
        Frontier/type/binding mismatch, aliased factory owners or unequal re-export.
    Exception
        Trusted factory/export/restore failures propagate; staged lookups are
        invalidated on restore/export failure and no acknowledgement is produced.

    Notes
    -----
    Never mutates a previous live collection. Factories must be fresh and free
    of external effects; trusted Python can retain references and is not sandboxed.
    """
    verify(prepared, bindings)
    ordered = tuple(sorted(bindings, key=lambda item: item.codec.binding.owner.id))
    entries = tuple((binding, binding.new_owner()) for binding in ordered)
    if len({id(owner) for _, owner in entries}) != len(entries):
        raise ValueError("Every component factory must return a distinct fresh owner")
    staged = StagedCollection(entries)
    try:
        for binding, owner in entries:
            cast(Stateful[Any], owner).restore_state(prepared.component(binding.codec))
        actual = staged._export()  # pylint: disable=protected-access
        for expected, restored in zip(prepared.values, actual, strict=True):
            if expected.codec.encode(expected.value) != restored.codec.encode(restored.value):
                raise ValueError("Fresh owner restore did not re-export the exact complete state")
    except BaseException:
        staged._close()  # pylint: disable=protected-access
        raise
    return staged
