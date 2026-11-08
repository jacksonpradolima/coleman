"""Bounded original-cell diagnostics, without a fabricated protected projection."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, field_serializer, model_validator

from coleman.core.primitives import Fingerprint
from coleman.core.temporal import Timestamp
from coleman.evaluation.design._raw_evidence import RawEvidence
from coleman.evaluation.design._wire import guard_graph, model_storage, wire_size
from coleman.evaluation.design.cells import RawCell
from coleman.evaluation.design.contracts import ExperimentDesign, InferenceQualification
from coleman.evaluation.design.limits import ProtocolLimits
from coleman.evaluation.design.projection import qualify_inference
from coleman.evaluation.design.selection import AttemptLedger, SelectionManifest
from coleman.evaluation.design.source_contracts import (
    SourcePartitionManifest,
    SourcePrefix,
    SourceReceipt,
    SourceScope,
    SourceSnapshot,
)
from coleman.evaluation.prequential.contracts import PrequentialContract


def _stored(value: Any) -> dict[str, Any]:
    """Inspect plain/model storage without invoking unchecked serializers."""
    if isinstance(value, BaseModel):
        return model_storage(value)
    kind = type(value)
    return value if kind is dict else {}


def _guard_selected(value: Any) -> Any:
    """Charge the complete supplied graph against its selected profile first."""
    root = _stored(value)
    panel = _stored(root.get("panel", root.get("original", value)))
    if "panel" in panel:
        panel = _stored(panel["panel"])
    raw_limits = _stored(panel.get("design")).get("limits")
    if raw_limits is not None:
        limits = ProtocolLimits.model_validate(raw_limits)
        guard_graph(value, maximum_bytes=limits.max_envelope_bytes, maximum_items=limits.max_graph_items)
    return value


class DiagnosticPanel(PrequentialContract):
    """Retain a complete original grid without requiring supported inference.

    Attributes
    ----------
    design, selection : ExperimentDesign, SelectionManifest
        Exact complete family and earlier selection, including every supplied
        development trial. No source read or protected freeze is performed.
    rows, ledger : tuple of RawCell, AttemptLedger
        All candidate/member/seed cells and separate actual protected work.
        Original records, provenance, omissions and missingness remain intact.
    mode : {'produced', 'imported'}
        Declared provenance handling. Produced evidence still requires actual
        work references; it does not acquire a projected evaluation identity.
    assembled_at : datetime
        Actual assembly after consumed availability and known work completion.

    Notes
    -----
    No p-value, independent mean, confidence interval or confirmatory claim is
    created. A failed partial loader cannot construct this complete artifact.
    """

    design: ExperimentDesign
    selection: SelectionManifest
    mode: Literal["produced", "imported"]
    rows: Annotated[tuple[RawCell, ...], Field(max_length=8192)]
    ledger: AttemptLedger
    assembled_at: Timestamp

    @model_validator(mode="before")
    @classmethod
    def _selected(cls, value: Any) -> Any:
        """Apply the selected graph allowance before nested revalidation."""
        return _guard_selected(value)

    @field_serializer("rows")
    def _original_cells(self, rows: tuple[RawCell, ...]) -> list[dict[str, Any]]:
        """Preserve original optional-field omissions exactly as RawPanel does."""
        return [row.model_dump(mode="json", exclude_unset=True) for row in rows]

    @model_validator(mode="after")
    def _evidence(self) -> Self:
        """Require all original evidence while leaving its inference unsupported."""
        self.selection.require_design(self.design)
        evidence = RawEvidence(self.design, self.selection, self.rows, self.ledger, self.mode, self.assembled_at)
        rows = evidence.require_rows(None)
        metadata = wire_size(self.model_dump(mode="json", exclude={"rows"}))
        if metadata + 1 > self.design.limits.max_metadata_bytes:
            raise ValueError("Diagnostic metadata exceeds its declared capacity")
        if wire_size(self) + 1 > self.design.limits.max_envelope_bytes:
            raise ValueError("Complete diagnostic panel exceeds its declared capacity")
        object.__setattr__(self, "rows", rows)
        return self


class DiagnosticLocator(PrequentialContract):
    """Bind result-free metadata used before a cached diagnostic disclosure.

    Attributes
    ----------
    scope, snapshot : SourceScope, SourceSnapshot
        Original stable upstream and selected source revision.
    design_fingerprint, selection_fingerprint, evidence_fingerprint : str
        Exact persisted declarations and complete diagnostic body. No protected
        evaluation fingerprint is invented for an unsupported request.
    """

    scope: SourceScope
    snapshot: SourceSnapshot
    design_fingerprint: Fingerprint
    selection_fingerprint: Fingerprint
    evidence_fingerprint: Fingerprint


class DiagnosticEvidence(PrequentialContract):
    """Join complete descriptive evidence to its actual pre-load source access.

    Attributes
    ----------
    panel : DiagnosticPanel
        Complete original grid, quantities, development and protected ledgers.
    qualification : InferenceQualification
        Exact pure support assessment; supported alone never means sufficient,
        independent, source-admitted or confirmatory evidence.
    source_receipt, source_prefix : SourceReceipt, SourcePrefix
        Real inspection/import reservation and its complete retained prefix.
    source_manifests : tuple of SourcePartitionManifest, optional
        Exact known support for historical narrow source reads.

    Notes
    -----
    Internal joins do not authenticate upstream history. The optional adapter
    verifies persisted metadata. This is never a ProtocolReport or new empirical
    admission, even when the supplied mathematical method is supported.
    """

    panel: DiagnosticPanel
    qualification: InferenceQualification
    source_receipt: SourceReceipt
    source_prefix: SourcePrefix
    source_manifests: Annotated[tuple[SourcePartitionManifest, ...], Field(max_length=512)] = ()

    @model_validator(mode="before")
    @classmethod
    def _selected(cls, value: Any) -> Any:
        """Bound all repeated evidence copies before nested model work."""
        return _guard_selected(value)

    @model_validator(mode="after")
    def _joins(self) -> Self:
        """Keep diagnostic support, selection and actual access identities exact."""
        panel, receipt = self.panel, self.source_receipt
        design, request = panel.design, receipt.request
        expected = (
            design.scope,
            design.snapshot,
            design.partition_manifest.fingerprint,
            panel.selection.fingerprint,
            None,
            None,
            "import" if panel.mode == "imported" else "inspection",
        )
        actual = (
            request.scope,
            request.snapshot,
            request.partition_manifest_fingerprint,
            request.selection_fingerprint,
            request.evaluation_fingerprint,
            request.report_fingerprint,
            request.purpose,
        )
        if actual != expected or request.partitions != ("test",):
            raise ValueError("Diagnostic reservation differs from its original declarations")
        index = receipt.sequence - 1
        if self.source_prefix.scope != design.scope or self.source_prefix.receipts[index : index + 1] != (receipt,):
            raise ValueError("Diagnostic source reservation is absent from its complete prefix")
        if self.qualification.to_json() != qualify_inference(design, panel.selection).to_json():
            raise ValueError("Diagnostic qualification differs from its complete design")
        if receipt.requested_at < panel.selection.actual_selected_at or (
            panel.mode == "produced" and receipt.requested_at > panel.assembled_at
        ):
            raise ValueError("Diagnostic assembly contradicts actual selection/reservation order")
        if len({item.fingerprint for item in self.source_manifests}) != len(self.source_manifests):
            raise ValueError("Diagnostic source manifests must be unique")
        self._manifest_support()
        self._produced_time()
        metadata = self.model_dump(mode="json", exclude={"panel": {"rows"}})
        if wire_size(metadata) + 1 > design.limits.max_metadata_bytes:
            raise ValueError("Complete diagnostic metadata exceeds its declared capacity")
        if wire_size(self) + 1 > design.limits.max_envelope_bytes:
            raise ValueError("Complete diagnostic evidence exceeds its declared capacity")
        return self

    def _manifest_support(self) -> None:
        """Keep every reservation-time verified narrow manifest in the full body."""
        manifests = {item.fingerprint: item for item in self.source_manifests}
        for receipt in self.source_prefix.receipts:
            if receipt.partition_manifest_verified:
                request = receipt.request
                manifest = manifests.get(str(request.partition_manifest_fingerprint))
                if manifest is None or manifest.scope != request.scope or manifest.snapshot != request.snapshot:
                    raise ValueError("Diagnostic prefix omits exact verified source manifest support")

    def _produced_time(self) -> None:
        """Check actual produced work against the reservation without imputation."""
        if self.panel.mode != "produced":
            return
        reserved = self.source_receipt.requested_at
        times = [row.record.available_at for row in self.panel.rows]
        for attempt in self.panel.ledger.attempts:
            times.extend(moment for moment in (attempt.started_at, attempt.completed_at) if moment is not None)
            times.extend(cost.record.available_at for cost in attempt.costs)
        if any(moment < reserved for moment in times):
            raise ValueError("Produced diagnostic work predates actual source reservation")

    @property
    def locator(self) -> DiagnosticLocator:
        """Return metadata that permits reservation before a cached body read."""
        return DiagnosticLocator(
            schema_version=1,
            scope=self.panel.design.scope,
            snapshot=self.panel.design.snapshot,
            design_fingerprint=self.panel.design.fingerprint,
            selection_fingerprint=self.panel.selection.fingerprint,
            evidence_fingerprint=self.fingerprint,
        )


class DiagnosticView(PrequentialContract):
    """Disclose unchanged original diagnostics with a new actual source receipt.

    Attributes
    ----------
    original : DiagnosticEvidence
        Original complete diagnostic; its historical reservation is unchanged.
    source_receipt : SourceReceipt
        New result_disclosure reservation made before opening the cached body.
    viewed_at : datetime
        Actual delivery after this reservation; no new experiment is claimed.
    """

    original: DiagnosticEvidence
    source_receipt: SourceReceipt
    viewed_at: Timestamp

    @model_validator(mode="before")
    @classmethod
    def _selected(cls, value: Any) -> Any:
        """Charge the original body and new disclosure framing together."""
        return _guard_selected(value)

    @model_validator(mode="after")
    def _disclosure(self) -> Self:
        """Reject a reused receipt, changed result or invented projected identity."""
        locator, receipt = self.original.locator, self.source_receipt
        request = receipt.request
        actual = (
            request.scope,
            request.snapshot,
            request.selection_fingerprint,
            request.report_fingerprint,
            request.evaluation_fingerprint,
            request.purpose,
        )
        expected = (
            locator.scope,
            locator.snapshot,
            locator.selection_fingerprint,
            locator.evidence_fingerprint,
            None,
            "result_disclosure",
        )
        if actual != expected or receipt.access_id == self.original.source_receipt.access_id:
            raise ValueError("Diagnostic disclosure requires a new exact result reservation")
        self._source_extension()
        if receipt.requested_at > self.viewed_at:
            raise ValueError("Diagnostic delivery predates its disclosure reservation")
        if wire_size(self) + 1 > self.original.panel.design.limits.max_envelope_bytes:
            raise ValueError("Diagnostic view exceeds its complete declared capacity")
        return self

    def _source_extension(self) -> None:
        """Refuse an ordinal restart or a changed known predecessor on disclosure."""
        receipt = self.source_receipt
        retained = self.original.source_prefix.receipts
        if receipt.sequence <= len(retained):
            raise ValueError("Diagnostic disclosure cannot restart the retained source ordinal")
        if receipt.sequence == len(retained) + 1 and receipt.previous_fingerprint != retained[-1].fingerprint:
            raise ValueError("Diagnostic disclosure differs from its retained source predecessor")
