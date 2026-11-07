"""Optional complete protocol reports over existing verified paired inference.

This composition deliberately imports the protected audit/report boundary. Pure
declarations, panels and projections remain usable without storage or transport.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, ValidationInfo, field_validator, model_validator

from coleman.core.primitives import Fingerprint
from coleman.core.temporal import Timestamp
from coleman.evaluation.design._source import audit_capacity
from coleman.evaluation.design._storage import guard_report
from coleman.evaluation.design._wire import DesignContract, wire_size
from coleman.evaluation.design.panel import RawPanel
from coleman.evaluation.design.source_contracts import (
    SourcePartitionManifest,
    SourcePrefix,
    SourceReceipt,
    SourceScope,
    SourceSnapshot,
)
from coleman.evaluation.design.source_qualification import source_reasons
from coleman.evaluation.protected.audit import AccessReceipt
from coleman.evaluation.protected.comparison import FamilyReport


class ReportLocator(DesignContract):
    """Retain metadata needed to reserve a cached disclosure before its body.

    Attributes
    ----------
    scope, snapshot : SourceScope, SourceSnapshot
        Original stable source and selected revision, without outcomes.
    design_fingerprint, selection_fingerprint, evaluation_fingerprint : str
        Exact original immutable declarations.
    report_fingerprint : str
        Exact complete ProtocolReport body. The adapter verifies the persisted
        locator before opening any cached result-bearing content.
    """

    scope: SourceScope
    snapshot: SourceSnapshot
    design_fingerprint: Fingerprint
    selection_fingerprint: Fingerprint
    evaluation_fingerprint: Fingerprint
    report_fingerprint: Fingerprint


class ProtocolReport(DesignContract):
    """Join original cells, complete history and an actually verified paired report.

    Attributes
    ----------
    panel : RawPanel
        Complete raw grid and distinct protected physical/cost ledger.
    source_receipt, source_prefix : SourceReceipt, SourcePrefix
        Actual pre-load access and retained complete immutable prefix.
    source_manifests : tuple of SourcePartitionManifest
        Exact persisted support for narrower historical source accesses.
    family : FamilyReport
        Existing unchanged inference contract, recomputed from derived unit means.
    completed_at : datetime
        Actual complete-envelope assembly, after family calculation and within the
        separately frozen comparison deadline; historical clocks are not changed.

    Notes
    -----
    Internal validation is not proof of persistence or authentic source history.
    ProtocolAudit verifies those joins before atomic complete publication. Unknown
    schemas reject. Old corpus, DomainRecord and FamilyReport formats are neither
    migrated nor reinterpreted. Incomplete source/panel work is not this artifact.
    """

    panel: RawPanel
    source_receipt: SourceReceipt
    source_prefix: SourcePrefix
    source_manifests: Annotated[tuple[SourcePartitionManifest, ...], Field(max_length=512)] = ()
    family: FamilyReport
    completed_at: Timestamp

    @field_validator("family", mode="before")
    @classmethod
    def _family_capacity(cls, value: Any, info: ValidationInfo) -> Any:
        """Apply selected legacy input/family/configuration caps before its validators."""
        panel = info.data.get("panel")
        if not isinstance(panel, RawPanel):
            raise ValueError("A complete validated panel must precede the protected family")
        guard_report(value, audit_capacity(panel.design.limits))
        return value

    @model_validator(mode="after")
    def _complete(self) -> Self:
        """Recompute every aggregate join and reject inflated nested eligibility."""
        panel = self.panel
        request = self.source_receipt.request
        source_join = (
            self.family.frozen,
            request.scope,
            request.snapshot,
            request.partition_manifest_fingerprint,
            request.selection_fingerprint,
            request.evaluation_fingerprint,
        )
        design_join = (
            panel.frozen,
            panel.design.scope,
            panel.design.snapshot,
            panel.design.partition_manifest.fingerprint,
            panel.selection.fingerprint,
            panel.frozen.fingerprint,
        )
        if source_join != design_join:
            raise ValueError("Protocol report source, manifest, selection and evaluation joins differ")
        if self.source_receipt.requested_at < panel.selection.actual_selected_at:
            raise ValueError("Protected source reservation predates actual selection")
        if self.family.access.requested_at < self.source_receipt.requested_at:
            raise ValueError("Legacy protected access predates the source reservation")
        if not self.family.completed_at <= self.completed_at <= panel.design.comparison_plan.test_end:
            raise ValueError("Complete protocol report exceeds its actual frozen analysis deadline")
        if self.source_reasons and (self.family.confirmatory_eligible or self.family.access.purpose != "verification"):
            raise ValueError("Exposed source requires an actual legacy verification receipt")
        self._produced_time()
        expected = panel.paired_observations()
        actual = {(item.candidate_id, item.metadata.group_id): item for item in self.family.inputs}
        if len(expected) != len(actual):
            raise ValueError("Protocol family omits or adds aggregate inputs")
        for row in expected:
            reference = actual.get((row.candidate_id, row.metadata.group_id))
            if reference is None or reference.metadata != row.metadata or reference.record != row.record:
                raise ValueError("Protocol family differs from recomputed original-panel means")
        if wire_size(self) + 1 > panel.design.limits.max_envelope_bytes:
            raise ValueError("Complete protocol report exceeds its declared envelope")
        return self

    def _produced_time(self) -> None:
        """Check every known produced work instant against the actual reservation."""
        if self.panel.mode != "produced":
            return
        reserved = self.source_receipt.requested_at
        if any(row.record.available_at < reserved for row in self.panel.rows):
            raise ValueError("Produced raw outcome predates actual protected source access")
        if any(
            moment is not None and moment < reserved
            for attempt in self.panel.ledger.attempts
            for moment in (attempt.started_at, attempt.completed_at)
        ):
            raise ValueError("Produced physical work predates protected source access")
        if any(cost.record.available_at < reserved for attempt in self.panel.ledger.attempts for cost in attempt.costs):
            raise ValueError("Produced physical cost predates protected source access")

    @property
    def source_reasons(self) -> tuple[str, ...]:
        """Return conservative source qualifications separately from inference."""
        return source_reasons(
            self.source_prefix, self.source_receipt, self.panel.selection, self.panel.mode, self.source_manifests
        )

    @property
    def confirmatory_eligible(self) -> bool:
        """Require both original source eligibility and actual legacy access eligibility."""
        return not self.source_reasons and self.family.confirmatory_eligible

    @property
    def locator(self) -> ReportLocator:
        """Return result-free immutable metadata for subsequent audited disclosures."""
        return ReportLocator(
            schema_version=1,
            scope=self.panel.design.scope,
            snapshot=self.panel.design.snapshot,
            design_fingerprint=self.panel.design.fingerprint,
            selection_fingerprint=self.panel.selection.fingerprint,
            evaluation_fingerprint=self.family.frozen.fingerprint,
            report_fingerprint=self.fingerprint,
        )


class ReportView(DesignContract):
    """Deliver original results with new disclosure receipts, never a new experiment.

    Attributes
    ----------
    original : ProtocolReport
        Unchanged original report, including its historical assessment/receipts.
    source_receipt : SourceReceipt
        New result_disclosure reservation before opening the cached body.
    access : AccessReceipt
        New actual legacy verification access; no new physical work is implied.
    viewed_at : datetime
        Actual delivery assembly after both reservations.
    new_confirmation : bool
        Required false: a cache view never claims another confirmation.
    """

    original: ProtocolReport
    source_receipt: SourceReceipt
    access: AccessReceipt
    viewed_at: Timestamp
    new_confirmation: Literal[False] = False

    @model_validator(mode="after")
    def _disclosure(self) -> Self:
        """Require real matching new verification identities and truthful time."""
        locator = self.original.locator
        request = self.source_receipt.request
        actual = (
            request.purpose,
            request.report_fingerprint,
            request.scope,
            request.snapshot,
            request.selection_fingerprint,
            request.evaluation_fingerprint,
            self.access.purpose,
            self.access.partition,
            self.access.evaluation_fingerprint,
            self.access.holdout_id,
            self.access.data_fingerprint,
        )
        expected = (
            "result_disclosure",
            locator.report_fingerprint,
            locator.scope,
            locator.snapshot,
            locator.selection_fingerprint,
            locator.evaluation_fingerprint,
            "verification",
            "test",
            locator.evaluation_fingerprint,
            locator.scope.holdout_id,
            locator.snapshot.data_fingerprint,
        )
        if actual != expected:
            raise ValueError("Cached disclosure receipt differs from its original report")
        if not self.source_receipt.requested_at <= self.access.requested_at <= self.viewed_at:
            raise ValueError("Cached disclosure clocks contradict reservation order")
        if self.source_receipt.access_id == self.original.source_receipt.access_id:
            raise ValueError("A cached disclosure requires a new source reservation")
        if self.access.access_id == self.original.family.access.access_id:
            raise ValueError("A cached disclosure requires a new legacy verification reservation")
        if wire_size(self) + 1 > self.original.panel.design.limits.max_envelope_bytes:
            raise ValueError("Complete cached view exceeds its declared envelope")
        return self
