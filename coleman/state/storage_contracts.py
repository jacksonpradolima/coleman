"""Pure bounded composed-state manifests, acknowledgements and storage limits."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference, ordered_references
from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint, fingerprint
from coleman.core.temporal import Timestamp
from coleman.objectives.bindings import DefinitionBinding
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from coleman.state.contracts import StateBinding

MIB = 1024**2
GIB = 1024**3
Generation = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
Count = Annotated[int, Field(strict=True, ge=0)]
PositiveCount = Annotated[int, Field(strict=True, ge=1)]
ComponentRole = Literal[
    "algorithm", "learner_pending", "predictor", "transform", "delivery", "domain", "applied_path", "virtual_path"
]
FrontierProfile = Literal["none", "replay-ordinal-v2"]
CommitKind = Literal["initialize", "transition", "seal", "codec_migration"]


class StorageLimits(CoreContract):
    """Persist finite local storage capacities independently of algorithm state.

    Attributes
    ----------
    maximum_children, maximum_components : int, optional
        Component inventory ceilings, at most32 child consumers/128 owners.
    maximum_component_bytes, maximum_collection_bytes : int, optional
        Canonical byte ceilings, at most16MiB per owner and64MiB per collection.
    maximum_manifest_bytes : int, optional
        Manifest byte ceiling, at most2MiB.
    receipt_window : int, optional
        Retained operational generations, default8 and at most64.
    maximum_root_bytes, maximum_entries : int, optional
        Total owned-root ceilings, default1GiB/4096 entries, at most8GiB/16384.
    maximum_orphan_bytes, maximum_orphan_entries : int, optional
        Unreferenced/temp ceilings, default128MiB/512 entries.
    lock_wait_seconds : float, optional
        Finite positive ownership-acquisition allowance, default5seconds.

    Notes
    -----
    Schema1 uses profile local-sqlite-state-v1. These are admission bounds, not
    disk preallocation, free-space assurance or callback CPU preemption. The
    store also validates receipt-window/candidate headroom against the actual
    definition. No storage is opened or modified by this model.
    """

    maximum_children: Annotated[int, Field(strict=True, ge=0, le=32)] = 32
    maximum_components: Annotated[int, Field(strict=True, ge=1, le=128)] = 128
    maximum_component_bytes: Annotated[int, Field(strict=True, ge=1, le=16 * MIB)] = 16 * MIB
    maximum_collection_bytes: Annotated[int, Field(strict=True, ge=1, le=64 * MIB)] = 64 * MIB
    maximum_manifest_bytes: Annotated[int, Field(strict=True, ge=1, le=2 * MIB)] = 2 * MIB
    receipt_window: Annotated[int, Field(strict=True, ge=1, le=64)] = 8
    maximum_root_bytes: Annotated[int, Field(strict=True, ge=1, le=8 * GIB)] = GIB
    maximum_entries: Annotated[int, Field(strict=True, ge=1, le=16384)] = 4096
    maximum_orphan_bytes: Annotated[int, Field(strict=True, ge=1, le=128 * MIB)] = 128 * MIB
    maximum_orphan_entries: Annotated[int, Field(strict=True, ge=1, le=512)] = 512
    lock_wait_seconds: Binary64 = 5.0

    @model_validator(mode="after")
    def _capacity(self) -> Self:
        """Require a possible bounded candidate and finite positive lock wait."""
        if self.maximum_component_bytes > self.maximum_collection_bytes:
            raise ValueError("Component byte ceiling cannot exceed the collection ceiling")
        candidate = self.maximum_collection_bytes + self.maximum_manifest_bytes + self.maximum_component_bytes
        if self.maximum_orphan_bytes < candidate:
            raise ValueError("Orphan byte ceiling must allow one candidate, manifest and temporary")
        if self.maximum_orphan_entries < self.maximum_components + 2:
            raise ValueError("Orphan entry ceiling must allow component/manifest/temporary publication")
        if self.lock_wait_seconds <= 0:
            raise ValueError("State ownership wait must be positive finite seconds")
        return self


class ComponentSpec(CoreImmutable):
    """Declare one typed owner without dynamically importing its implementation.

    Attributes
    ----------
    binding : StateBinding
        Exact stable owner, implementation, outer codec and configuration.
    child : Reference or None
        Child consumer or explicit null for shared domain/application state.
    roles : tuple of str
        Unique declared ownership labels; host code inventories actual fields.
    frontier_profile : {'none', 'replay-ordinal-v2'}
        None supplies publication only. Replay requires a child and delivery role.

    Notes
    -----
    Labels do not prove arbitrary custom snapshot completeness. No lookup,
    registration, restore or I/O occurs. Roles are canonical set declarations.
    """

    binding: StateBinding
    child: Reference | None
    roles: Annotated[tuple[ComponentRole, ...], Field(min_length=1, max_length=8)]
    frontier_profile: FrontierProfile

    @model_validator(mode="after")
    def _roles(self) -> Self:
        """Reject duplicate roles and incomplete causal-owner declarations."""
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("Component roles must be unique")
        if self.frontier_profile == "replay-ordinal-v2" and (self.child is None or "delivery" not in self.roles):
            raise ValueError("Replay components require one child and a delivery role")
        object.__setattr__(self, "roles", tuple(sorted(self.roles)))
        return self


class ExecutionCompatibility(CoreContract):
    """Record explicitly captured runtime and copied-owner definition identities.

    Attributes
    ----------
    runtime : SelectedInstalledRuntime
        Actual schema2 selected-extra environment, captured by the outer host.
    owner_implementations : tuple of DefinitionBinding
        At most256 unique canonical definitions including copied owner/mapping
        code. Definition identities are distinct from the128-component ceiling.

    Notes
    -----
    Reading performs no capture or environment comparison. The optional store
    compares the caller's actual current binding before execution/operational
    receipts. Resource/version hashes are not binary, ABI or authenticity proof.
    """

    runtime: SelectedInstalledRuntime
    owner_implementations: Annotated[tuple[DefinitionBinding, ...], Field(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def _definitions(self) -> Self:
        """Canonicalize declaration order without hiding conflicting identities."""
        ids = [item.reference.id for item in self.owner_implementations]
        if len(set(ids)) != len(ids):
            raise ValueError("Owner implementations require unique logical definition IDs")
        object.__setattr__(
            self, "owner_implementations", tuple(sorted(self.owner_implementations, key=_definition_key))
        )
        return self


def _definition_key(value: DefinitionBinding) -> str:
    """Order exact unique implementation declarations by their logical IDs."""
    return value.reference.id


class CollectionDefinition(CoreContract):
    """Inventory every required state owner in one bounded composed session.

    Attributes
    ----------
    reference, session, source_epoch : Reference
        Exact definition generation, logical session and fixed delivery epoch.
    children : tuple of Reference
        Canonical unique child consumers, from zero through32.
    components : tuple of ComponentSpec
        Canonical unique owners, from one through128; all must be present.

    Notes
    -----
    Declaration enumeration order has no semantic effect. Actual action-path
    order remains inside its typed owner and is never sorted here. A replay
    child has one frontier owner; direct owners get no inferred event acceptance.
    No filesystem, registry or host is consulted.
    """

    reference: Reference
    session: Reference
    source_epoch: Reference
    children: Annotated[tuple[Reference, ...], Field(max_length=32)]
    components: Annotated[tuple[ComponentSpec, ...], Field(min_length=1, max_length=128)]

    @model_validator(mode="after")
    def _inventory(self) -> Self:
        """Check exact children/owner versions and one replay frontier per child."""
        children = ordered_references(self.children)
        owners = [item.binding.owner.id for item in self.components]
        if len(set(owners)) != len(owners):
            raise ValueError("Collection requires unique component owner IDs")
        known = {item.id: item for item in children}
        causal = []
        for item in self.components:
            if item.child is not None and known.get(item.child.id) != item.child:
                raise ValueError("Component child is missing or has an incompatible version")
            if item.frontier_profile == "replay-ordinal-v2":
                causal.append(item.child)
        if {item.child.id for item in self.components if item.child is not None} != {item.id for item in children}:
            raise ValueError("Every declared child requires at least one state owner")
        if len({item.id for item in causal if item is not None}) != len(causal):
            raise ValueError("A replay child requires exactly one declared frontier owner")
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "components", tuple(sorted(self.components, key=lambda item: item.binding.owner.id)))
        return self


class StateBlobRef(CoreImmutable):
    """Bind exact canonical component bytes without embedding executable state.

    Attributes
    ----------
    binding : StateBinding
        Exact concrete outer state-codec and owner interpretation.
    fingerprint : str
        SHA-256 of canonical UTF-8 component bytes; no trailing file newline.
    byte_count : int
        Positive exact encoded bytes, checked again during bounded acquisition.

    Notes
    -----
    The optional adapter derives content-addressed paths from the digest alone.
    This record opens no file and authenticates no producer.
    """

    binding: StateBinding
    fingerprint: Fingerprint
    byte_count: PositiveCount


class ChildFrontier(CoreContract):
    """Bind one actual replay delivery frontier without inferring learner fields.

    Attributes
    ----------
    child, source_epoch, stream, source, lineage : Reference
        Exact consumer, fixed epoch and replay producing identities.
    decision_count, update_count, next_sequence, sealed_floor : int
        Nonnegative global counts. Floor retires ordinals strictly below it.
    eligible_through, apply_at : Timestamp or None
        Paired information/application clocks, null only before a closed phase.
    causal_commitment, state_fingerprint, host_fingerprint : str
        Actual ordered replay commitment and complete replay/host digests.
    policy_fingerprint, retention_fingerprint : str
        Exact immutable producing semantics and retention profile digests.
    state_codec : Reference
        Inner host interpretation; independent of the outer component codec.

    Notes
    -----
    Schema1 is declarative. No host, registry, clock or filesystem is consulted.
    """

    child: Reference
    source_epoch: Reference
    stream: Reference
    source: Reference
    lineage: Reference
    decision_count: Count
    update_count: Count
    next_sequence: Count
    sealed_floor: Count
    eligible_through: Timestamp | None
    apply_at: Timestamp | None
    causal_commitment: Fingerprint
    state_fingerprint: Fingerprint
    host_fingerprint: Fingerprint
    policy_fingerprint: Fingerprint
    retention_fingerprint: Fingerprint
    state_codec: Reference

    @model_validator(mode="after")
    def _frontier(self) -> Self:
        """Reject impossible count order and a partially declared clock pair."""
        if not self.sealed_floor <= self.update_count <= self.decision_count:
            raise ValueError("Replay counts require floor <= updates <= decisions")
        if (self.eligible_through is None) != (self.apply_at is None):
            raise ValueError("Replay frontier clocks must both be present or null")
        if self.eligible_through is not None and self.apply_at is not None and self.eligible_through > self.apply_at:
            raise ValueError("Replay information clock exceeds application clock")
        return self


def semantic_frontiers_fingerprint(frontiers: tuple[ChildFrontier, ...]) -> str:
    """Identify the exact delivery semantics preserved across codec migration.

    Parameters
    ----------
    frontiers : tuple of ChildFrontier
        At most32 unique child frontiers; declaration order is immaterial.

    Returns
    -------
    str
        SHA-256 under preserved-replay-frontiers-v1. Excludes only complete
        replay-state digest, host digest and host codec, which a known typed
        mapping may change. Source head anchors the complete original records.

    Raises
    ------
    ValueError, TypeError
        Invalid/duplicate frontier, unsupported count or non-tuple inventory.

    Notes
    -----
    Includes both clocks, all counters/floors, causal commitment, policy and
    retention identities. Performs no I/O and authenticates no discarded past.
    """
    if type(frontiers) is not tuple or len(frontiers) > 32:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Semantic frontiers require a bounded tuple of at most32 children")
    checked = tuple(ChildFrontier.model_validate(item) for item in frontiers)
    if len({item.child.id for item in checked}) != len(checked):
        raise ValueError("Semantic frontiers require unique child identities")
    return fingerprint(
        {
            "profile": "preserved-replay-frontiers-v1",
            "frontiers": [
                item.model_dump(mode="json", exclude={"state_fingerprint", "host_fingerprint", "state_codec"})
                for item in sorted(checked, key=lambda item: item.child.id)
            ],
        }
    )


class CommitIntent(CoreContract):
    """Identify an explicit command, independently of its Python callback.

    Attributes
    ----------
    generation : int
        Consecutive publication ordinal, zero through2**63-1.
    kind : {'initialize', 'transition', 'seal', 'codec_migration'}
        Exact allowed old-to-new relation.
    content_fingerprint : str
        Caller-declared canonical command digest, not an inferred closure hash.

    Notes
    -----
    Schema1 does not make arbitrary business messages or external effects
    idempotent. Operational retries also bind the exact original predecessor.
    """

    generation: Generation
    kind: CommitKind
    content_fingerprint: Fingerprint


class AcknowledgedHead(CoreContract):
    """Name the sole accepted local publication, never just a written artifact.

    Attributes
    ----------
    session, source_epoch : Reference
        Fixed logical session and source-owned epoch.
    definition_fingerprint, manifest_fingerprint, intent_fingerprint : str
        Exact definition, complete manifest and command digests.
    generation : int
        Accepted consecutive publication ordinal in0..2**63-1.

    Notes
    -----
    Schema1 is metadata evidence, not runtime compatibility or authentication.
    """

    session: Reference
    source_epoch: Reference
    definition_fingerprint: Fingerprint
    generation: Generation
    manifest_fingerprint: Fingerprint
    intent_fingerprint: Fingerprint


class CommitRequest(CoreContract):
    """Bind an explicit intent to its expected accepted predecessor.

    Attributes
    ----------
    expected : AcknowledgedHead or None
        Null only for explicit virgin initialization.
    intent : CommitIntent
        Complete publication identity and caller-owned command digest.

    Notes
    -----
    Schema1 performs no lock, retry, callback or mutation.
    """

    expected: AcknowledgedHead | None
    intent: CommitIntent


class CommitReceipt(CoreContract):
    """Distinguish an accepted retry from the collection's current position.

    Attributes
    ----------
    status : {'new', 'retained'}
        New commit or exact operational retry inside the receipt window.
    accepted_head, current_head : AcknowledgedHead
        Original accepted publication and actual current head respectively.

    Notes
    -----
    Schema1 never rewinds current state when an older receipt is returned.
    """

    status: Literal["new", "retained"]
    accepted_head: AcknowledgedHead
    current_head: AcknowledgedHead

    @model_validator(mode="after")
    def _heads(self) -> Self:
        """Require same-session nonrewinding evidence and exact new heads."""
        accepted, current = self.accepted_head, self.current_head
        if (accepted.session, accepted.source_epoch) != (current.session, current.source_epoch):
            raise ValueError("Receipt heads must belong to the same session and epoch")
        if accepted.generation > current.generation:
            raise ValueError("Receipt cannot advance beyond the actual current head")
        if (self.status == "new" or accepted.generation == current.generation) and accepted != current:
            raise ValueError("New or same-generation receipt heads must be identical")
        return self


class MigrationRequest(CoreContract):
    """Declare a known codec mapping, never a JSON-selected executable import.

    Attributes
    ----------
    commit : CommitRequest
        Nonvirgin codec_migration command and exact source head.
    migration : DefinitionBinding
        Qualified mapping definition and digest supplied by trusted host code.
    target_definition : CollectionDefinition
        Exact proposed owner/code/configuration interpretation.

    Notes
    -----
    Schema1 declares intent; it does not execute or infer a migration.
    """

    commit: CommitRequest
    migration: DefinitionBinding
    target_definition: CollectionDefinition

    @model_validator(mode="after")
    def _kind(self) -> Self:
        """Refuse implicit initialization or migration through another kind."""
        if self.commit.expected is None or self.commit.intent.kind != "codec_migration":
            raise ValueError("Migration requires an acknowledged source and codec_migration intent")
        return self


class MigrationReceipt(CoreContract):
    """Preserve exact mapping and source evidence for an accepted migration.

    Attributes
    ----------
    migration : DefinitionBinding
        Explicit qualified mapping identity.
    source_head : AcknowledgedHead
        Exact accepted source publication.
    source_definition_fingerprint, target_definition_fingerprint : str
        Old and new concrete owner inventories.
    preserved_frontiers_fingerprint : str
        Canonical digest of the preserved semantic replay frontiers.

    Notes
    -----
    Schema1 records a known mapping, not inferred field equivalence.
    """

    migration: DefinitionBinding
    source_head: AcknowledgedHead
    source_definition_fingerprint: Fingerprint
    target_definition_fingerprint: Fingerprint
    preserved_frontiers_fingerprint: Fingerprint


class CollectionManifest(CoreContract):
    """Bind one complete immutable publication before its separate SQL ack.

    Attributes
    ----------
    definition : CollectionDefinition
        Exact complete owner inventory.
    compatibility : ExecutionCompatibility
        Explicit captured runtime and copied-owner implementation identities.
    intent : CommitIntent
        Consecutive command identity, not an acknowledgement itself.
    predecessor : AcknowledgedHead or None
        Exact prior ack, null only at generation zero initialization.
    components : tuple of StateBlobRef
        Every owner exactly once, canonically ordered by logical owner ID.
    frontiers : tuple of ChildFrontier
        Every declared replay child exactly once; direct owners imply none.
    migration : MigrationReceipt or None
        Required only for an explicit known codec migration.

    Notes
    -----
    Schema1 validates inventory and ancestry without opening component files.
    Artifact existence alone never accepts progress. Inner replay host codecs
    and outer component codecs intentionally have distinct interpretations.
    """

    definition: CollectionDefinition
    compatibility: ExecutionCompatibility
    intent: CommitIntent
    predecessor: AcknowledgedHead | None
    components: Annotated[tuple[StateBlobRef, ...], Field(min_length=1, max_length=128)]
    frontiers: Annotated[tuple[ChildFrontier, ...], Field(max_length=32)]
    migration: MigrationReceipt | None

    @model_validator(mode="after")
    def _complete(self) -> Self:
        """Reject partial owner inventories, duplicate frontiers or false ancestry."""
        actual = tuple(sorted(self.components, key=lambda item: item.binding.owner.id))
        if tuple(item.binding for item in actual) != tuple(item.binding for item in self.definition.components):
            raise ValueError("Manifest must contain the exact complete declared owner inventory")
        implementations = self.compatibility.owner_implementations
        if any(item.binding.implementation not in implementations for item in actual):
            raise ValueError("Manifest owner implementation is missing from execution compatibility")
        frontiers = tuple(sorted(self.frontiers, key=lambda item: item.child.id))
        expected = tuple(
            item.child for item in self.definition.components if item.frontier_profile == "replay-ordinal-v2"
        )
        expected = tuple(sorted(expected, key=lambda item: "" if item is None else item.id))
        if tuple(item.child for item in frontiers) != expected:
            raise ValueError("Manifest requires each declared replay child frontier exactly once")
        if any(item.source_epoch != self.definition.source_epoch for item in frontiers):
            raise ValueError("Replay frontier source epoch differs from the collection")
        self._ancestry()
        object.__setattr__(self, "components", actual)
        object.__setattr__(self, "frontiers", frontiers)
        return self

    def _ancestry(self) -> None:
        """Validate the exact initial, successor or explicitly migrated boundary."""
        previous = self.predecessor
        if previous is None:
            if self.intent.generation != 0 or self.intent.kind != "initialize" or self.migration is not None:
                raise ValueError("Only generation-zero initialization has no predecessor")
            return
        if self.intent.kind == "initialize" or self.intent.generation != previous.generation + 1:
            raise ValueError("Manifest must follow its exact predecessor generation")
        if (self.definition.session, self.definition.source_epoch) != (previous.session, previous.source_epoch):
            raise ValueError("Manifest cannot change its session or source epoch")
        if self.intent.kind == "codec_migration":
            self._migration_ancestry(previous)
        elif self.migration is not None or self.definition.fingerprint != previous.definition_fingerprint:
            raise ValueError("Ordinary publication must preserve the exact definition")

    def _migration_ancestry(self, previous: AcknowledgedHead) -> None:
        """Check known migration evidence in its original validation order.

        Parameters
        ----------
        previous : AcknowledgedHead
            Exact predecessor already checked for generation, session and epoch.

        Raises
        ------
        ValueError
            Missing source receipt or conflicting source, target or frontier digest.

        Notes
        -----
        Performs pure comparisons and deterministic fingerprinting only; creates
        no receipt, file or acknowledged progress and never executes a mapping.
        """
        receipt = self.migration
        if receipt is None or receipt.source_head != previous:
            raise ValueError("Codec migration requires its exact source receipt")
        if receipt.source_definition_fingerprint != previous.definition_fingerprint:
            raise ValueError("Migration source definition differs from its acknowledged predecessor")
        if receipt.target_definition_fingerprint != self.definition.fingerprint:
            raise ValueError("Migration target definition differs from its manifest")
        if receipt.preserved_frontiers_fingerprint != semantic_frontiers_fingerprint(self.frontiers):
            raise ValueError("Migration receipt differs from its target's preserved semantic frontiers")


class CleanupReceipt(CoreContract):
    """Report bounded unreachable-file cleanup without advancing child floors.

    Attributes
    ----------
    current_head : AcknowledgedHead
        Exact head protected throughout cleanup.
    removed_blobs, removed_files, removed_bytes : int
        Nonnegative actual removals, bytes measured in encoded file bytes.
    retired_receipt_floor : int
        Earliest retained operational publication generation, not a child floor.

    Notes
    -----
    Schema1 is evidence of local cleanup, not a logical replay transition.
    """

    current_head: AcknowledgedHead
    removed_blobs: Count
    removed_files: Count
    removed_bytes: Count
    retired_receipt_floor: Generation
