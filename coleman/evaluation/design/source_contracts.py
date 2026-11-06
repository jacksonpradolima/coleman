"""Pure scientific source identity, partition manifests and attempted disclosures."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, field_validator, model_validator

from coleman.core.base import CoreImmutable
from coleman.core.identity import Reference
from coleman.core.primitives import Fingerprint, Identifier, fingerprint
from coleman.core.splits import TemporalObservation, TemporalPlan
from coleman.core.temporal import Timestamp
from coleman.evaluation.design._wire import DesignContract
from coleman.objectives.bindings import DefinitionBinding

Partition = Literal["train", "validation", "test", "unknown"]
SourcePurpose = Literal["import", "inspection", "tuning", "protected_evaluation", "result_disclosure"]


class SourceScope(DesignContract):
    """Bind one stable scientific upstream namespace without inventing prehistory.

    Attributes
    ----------
    reference : Reference
        Stable upstream logical ID; version identifies interpretation, not a new
        unexposed namespace. Encoding and candidate changes do not rotate it.
    upstream : DefinitionBinding
        Reviewed upstream interpretation, separate from the selected snapshot.
    partition_definition : DefinitionBinding or None
        Known partition interpretation, or explicit null for unknown.
    prehistory : {'declared_unexposed', 'known_exposed', 'unknown'}
        Explicit host declaration. An empty directory proves none of these.

    Notes
    -----
    Pure identity is not authentication or source admission. The audit adapter
    must retain the exact immutable header at logical_key and reject incompatible
    replacements; aliases/out-of-band reads remain an explicit trust boundary.
    """

    _maximum_bytes: ClassVar[int] = 8192
    reference: Reference
    upstream: DefinitionBinding
    partition_definition: DefinitionBinding | None
    prehistory: Literal["declared_unexposed", "known_exposed", "unknown"]

    @property
    def logical_key(self) -> str:
        """Return the stable source namespace, excluding version and data encoding."""
        return fingerprint(["coleman.source-access-scope-v1", self.reference.id])

    @property
    def holdout_id(self) -> str:
        """Return the new profile's stable legacy-audit holdout identifier."""
        return fingerprint(["coleman.source-access-holdout-v1", self.logical_key])


class SourceSnapshot(CoreImmutable):
    """Identify actual selected input bytes separately from their upstream scope.

    Attributes
    ----------
    reference : Reference
        Selected revision identity; changing it never clears source exposure.
    data_fingerprint : str
        Complete interpreted input identity, not an asserted empirical admission.
    interpretation : DefinitionBinding
        Exact selected input interpretation/adapter definition.
    captured_at : datetime or None
        Actual supplied creation/import instant, or explicit unknown. This is not
        substituted for historical source event or availability clocks.
    """

    reference: Reference
    data_fingerprint: Fingerprint
    interpretation: DefinitionBinding
    captured_at: Timestamp | None


class SourceMember(CoreImmutable):
    """Keep source leakage, task/path and inference-unit membership distinct.

    Attributes
    ----------
    member_id : str
        Stable logical panel member, not a seed, worker or physical retry.
    task, path : Reference
        Declared problem/task and actual path identities. Different independent
        paths may share a task; changing path version cannot split one path into
        several independent units.
    unit_id : str or None
        Declared independent unit for protected members; development rows may
        explicitly omit one. IDs alone do not prove scientific independence.
    metadata : TemporalObservation
        Original observation/revision, leakage group, event/target/label clocks.
        metadata.group_id is not silently reinterpreted as unit_id.
    """

    member_id: Identifier
    task: Reference
    path: Reference
    unit_id: Identifier | None
    metadata: TemporalObservation


def _ordered_members(values: tuple[SourceMember, ...]) -> tuple[SourceMember, ...]:
    """Require one selected revision and logical member per source observation."""
    member_ids = [item.member_id for item in values]
    observations = [item.metadata.observation_id for item in values]
    if len(member_ids) != len(set(member_ids)) or len(observations) != len(set(observations)):
        raise ValueError("Source membership requires unique members and one selected revision per observation")
    return tuple(sorted(values, key=lambda item: item.member_id))


class SourcePartitionManifest(DesignContract):
    """Freeze complete known source metadata before a narrow access claim.

    Attributes
    ----------
    scope, snapshot : SourceScope, SourceSnapshot
        Stable upstream and exact selected revision; neither is inferred.
    source_plan : TemporalPlan
        Original scientific source chronology, independent of later result
        assembly. All nominal partitions participate in leakage validation.
    members : tuple of SourceMember
        Complete bounded selected metadata, including development and excluded
        rows. Protected eligible members require explicit unit_id.

    Raises
    ------
    ValueError
        Duplicate/revised identity, cross-partition leakage, missing protected
        unit or one path split across distinct protected units.

    Notes
    -----
    Unknown/late/outside source rows remain explicit through plan decisions;
    they are not silently assigned another partition. This root performs no
    persistence or access. The adapter verifies its exact fingerprint before
    allowing a train/validation-only disclosure to remain narrow.
    """

    _maximum_bytes: ClassVar[int] = 2 * 1024**2
    scope: SourceScope
    snapshot: SourceSnapshot
    source_plan: TemporalPlan
    members: Annotated[tuple[SourceMember, ...], Field(min_length=1, max_length=4096)]

    @field_validator("members")
    @classmethod
    def _members(cls, values: tuple[SourceMember, ...]) -> tuple[SourceMember, ...]:
        """Canonicalize set membership without erasing duplicate source facts."""
        return _ordered_members(values)

    @model_validator(mode="after")
    def _membership(self) -> Self:
        """Validate all source groups before checking protected path ownership."""
        self.source_plan.split(tuple(item.metadata for item in self.members))
        paths: dict[str, str] = {}
        for member in self.members:
            if self.source_plan.nominal_partition(member.metadata) != "test":
                continue
            if member.unit_id is None:
                raise ValueError("Every protected eligible source member requires an explicit unit_id")
            key = member.path.id
            previous = paths.setdefault(key, member.unit_id)
            if previous != member.unit_id:
                raise ValueError("One logical path cannot be split into several independent units")
        return self

    @property
    def protected_members(self) -> tuple[SourceMember, ...]:
        """Return only fixed source-test members eligible under original clocks."""
        return tuple(
            item
            for item in self.members
            if (decision := self.source_plan.decide(item.metadata)).partition == "test" and decision.eligible
        )


class SourceRequest(DesignContract):
    """Declare one mediated read/disclosure before any result-bearing content.

    Attributes
    ----------
    scope, snapshot : SourceScope, SourceSnapshot
        Actual stable upstream and selected source revision.
    purpose : str
        Import, inspection, tuning, protected evaluation or a cached result view.
    partitions : tuple of str
        Nonempty unique partition claim. Unknown is a complete conservative claim
        and cannot be combined with narrower labels to reduce exposure.
    partition_manifest_fingerprint : str or None
        Exact persisted SourcePartitionManifest supporting a narrow claim. Missing
        or unverified manifests conservatively expose protected data.
    actor : str
        Actual caller/actor identifier, without ambient environment capture.
    selection_fingerprint, evaluation_fingerprint, report_fingerprint : str or None
        Known immutable joins. Pre-selection reads retain explicit nulls;
        protected evaluation requires actual selection/evaluation identity, and
        result disclosure requires the original report identity.

    Notes
    -----
    A cached view is a new disclosure, not a new physical experiment or a fresh
    confirmation. Publication/receipt reconciliation without results is distinct.
    The adapter supplies actual reservation time/ID and verifies all source joins.
    """

    _maximum_bytes: ClassVar[int] = 2048
    scope: SourceScope
    snapshot: SourceSnapshot
    purpose: SourcePurpose
    partitions: Annotated[tuple[Partition, ...], Field(min_length=1, max_length=3)]
    partition_manifest_fingerprint: Fingerprint | None
    actor: Identifier
    selection_fingerprint: Fingerprint | None
    evaluation_fingerprint: Fingerprint | None
    report_fingerprint: Fingerprint | None

    @field_validator("partitions")
    @classmethod
    def _partitions(cls, values: tuple[Partition, ...]) -> tuple[Partition, ...]:
        """Reject ambiguous declarations before sorting partition set labels."""
        if len(values) != len(set(values)) or ("unknown" in values and len(values) != 1):
            raise ValueError("Source partitions must be unique; unknown must stand alone")
        return tuple(sorted(values))

    @model_validator(mode="after")
    def _joins(self) -> Self:
        """Require real known identities for selected protected operations."""
        if self.purpose in {"protected_evaluation", "result_disclosure"} and not (
            "test" in self.partitions or "unknown" in self.partitions
        ):
            raise ValueError("Protected evaluation and result disclosure cannot claim only development partitions")
        if self.purpose == "protected_evaluation" and (
            self.selection_fingerprint is None or self.evaluation_fingerprint is None
        ):
            raise ValueError("Protected evaluation requires actual selected configuration and evaluation bindings")
        if self.purpose == "result_disclosure" and self.report_fingerprint is None:
            raise ValueError("Result disclosure requires its original report binding")
        return self

    @property
    def potential_test_exposure(self) -> bool:
        """Identify claims that cannot be narrow even before manifest verification."""
        return (
            self.purpose in {"protected_evaluation", "result_disclosure"}
            or "test" in self.partitions
            or "unknown" in self.partitions
            or self.partition_manifest_fingerprint is None
            or self.scope.partition_definition is None
        )


class SourceReceipt(DesignContract):
    """Retain one actual attempted disclosure, independently of its completion.

    Attributes
    ----------
    request : SourceRequest
        Complete bounded declaration; no outcome-bearing data is embedded here.
    sequence : int
        Positive source ordinal, at most 512. Exactly full prefixes remain readable.
    access_id : str
        Adapter-owned unique invocation identity, not a seed or scientific unit.
    requested_at : datetime
        Actual reservation instant, never backdated to historical label time.
    previous_fingerprint : str or None
        Exact predecessor link; only ordinal 1 has a null predecessor.
    partition_manifest_verified : bool
        Actual adapter verification of a narrow manifest at reservation. Required
        and immutable; later saving a missing manifest cannot clean an earlier
        unverified access. False does not mean a failed physical experiment.

    Notes
    -----
    This model does not read a clock, publish a receipt or assert success. A
    failed read remains an attempt. The canonical body plus one framing LF must
    fit 2 KiB. Unknown versions reject; no old audit receipt is reinterpreted.
    """

    _maximum_bytes: ClassVar[int] = 2048
    request: SourceRequest
    sequence: Annotated[int, Field(strict=True, ge=1, le=512)]
    access_id: Identifier
    requested_at: Timestamp
    previous_fingerprint: Fingerprint | None
    partition_manifest_verified: Annotated[bool, Field(strict=True)]

    @model_validator(mode="after")
    def _previous(self) -> Self:
        """Preserve the explicit first-ordinal/predecessor distinction."""
        if (self.sequence == 1) != (self.previous_fingerprint is None):
            raise ValueError("Only source receipt ordinal 1 requires a null predecessor")
        if self.partition_manifest_verified and self.request.potential_test_exposure:
            raise ValueError("Only a potentially narrow source request can record verified partition metadata")
        return self


class SourcePrefix(DesignContract):
    """Validate one complete bounded immutable prefix in reservation order.

    Attributes
    ----------
    scope : SourceScope
        Exact immutable scope header; snapshot revisions may legitimately vary.
    receipts : tuple of SourceReceipt
        Zero through512 contiguous attempts. Sequence and links, not timestamp
        sorting, establish order; actual wall clocks may coincide or move back.

    Raises
    ------
    ValueError
        Gap, changed scope, wrong link, duplicate invocation ID or unknown data.

    Notes
    -----
    Checks internal consistency, not authenticity, whole-store rollback or global
    visibility into source files. Confirmatory qualification additionally needs
    the adapter's actual persisted manifests and existing legacy access history.
    """

    _maximum_bytes: ClassVar[int] = 2 * 1024**2
    scope: SourceScope
    receipts: Annotated[tuple[SourceReceipt, ...], Field(max_length=512)]

    @model_validator(mode="after")
    def _chain(self) -> Self:
        """Retain exact complete-prefix order without timestamp-based inference."""
        previous = None
        identities: set[str] = set()
        for sequence, receipt in enumerate(self.receipts, start=1):
            if (
                receipt.sequence != sequence
                or receipt.request.scope != self.scope
                or receipt.previous_fingerprint != previous
                or receipt.access_id in identities
            ):
                raise ValueError("Invalid complete source-access prefix")
            identities.add(receipt.access_id)
            previous = receipt.fingerprint
        return self
