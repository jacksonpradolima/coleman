"""Optional source-first diagnostic loading and immutable cached disclosures."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from coleman.artifacts import ArtifactWriter
from coleman.evaluation.design._source import SourceAudit, digest_path
from coleman.evaluation.design._storage import publish_bounded, read_contract
from coleman.evaluation.design._wire import guard_graph, wire_size
from coleman.evaluation.design.audit import ProtocolAudit
from coleman.evaluation.design.contracts import ExperimentDesign
from coleman.evaluation.design.limits import ProtocolLimits
from coleman.evaluation.design.projection import qualify_inference
from coleman.evaluation.design.selection import SelectionManifest
from coleman.evaluation.design.source_contracts import SourcePartitionManifest, SourcePrefix, SourceRequest
from coleman.evaluation.prequential.evidence import (
    DiagnosticEvidence,
    DiagnosticLocator,
    DiagnosticPanel,
    DiagnosticView,
)

Mode = Literal["produced", "imported"]


class DiagnosticAudit:
    """Reserve descriptive access without pretending unsupported inference ran.

    Parameters
    ----------
    artifact_root : str or pathlib.Path
        Existing cooperative artifact/source-audit namespace. Construction does
        not create directories. Reusing the source namespace preserves exposure.
    limits : ProtocolLimits or None, optional
        Exact selected counts, UTF-8 bytes, graph and receipt capacities.

    Notes
    -----
    Loaders are trusted bounded local callbacks, not sandboxed code. Failed loads
    retain real inspection/import reservations. This adapter creates no protected
    freeze, legacy inferential reservation or ProtocolReport. Source and artifact
    publication are separate conservative operations, not a transaction.
    """

    def __init__(self, artifact_root: str | Path, *, limits: ProtocolLimits | None = None) -> None:
        """Select existing source/publication adapters without accessing outcomes."""
        self.root = Path(artifact_root)
        self.limits = ProtocolLimits(schema_version=1) if limits is None else ProtocolLimits.model_validate(limits)
        self.sources = SourceAudit(self.root, limits=self.limits)
        self.writer = ArtifactWriter(self.root)

    def _guard(self, value: object) -> None:
        """Bound the selected complete graph before old validators/serializers."""
        guard_graph(value, maximum_bytes=self.limits.max_envelope_bytes, maximum_items=self.limits.max_graph_items)

    def _manifests(self, prefix: SourcePrefix) -> tuple[SourcePartitionManifest, ...]:
        """Acquire only actual persisted support for verified narrow receipts."""
        manifests = {}
        for receipt in prefix.receipts:
            digest = receipt.request.partition_manifest_fingerprint
            if receipt.partition_manifest_verified and digest not in manifests:
                manifest = self.sources.verified_manifest(receipt.request)
                if manifest is not None:
                    manifests[manifest.fingerprint] = manifest
                    self._guard(tuple(manifests.values()))
        return tuple(value for _, value in sorted(manifests.items()))

    def evaluate(
        self,
        design: ExperimentDesign,
        selection: SelectionManifest,
        loader: Callable[[], DiagnosticPanel],
        *,
        actor: str,
        mode: Mode,
    ) -> DiagnosticEvidence:
        """Reserve actual source access before loading one complete diagnostic.

        Parameters
        ----------
        design, selection : ExperimentDesign, SelectionManifest
            Exact frozen declarations and complete development selection.
        loader : callable
            Trusted zero-argument callback returning its actual DiagnosticPanel.
            No statistical projection is required or fabricated.
        actor : str
            Explicit actor retained in the real source reservation.
        mode : {'produced', 'imported'}
            Declared before loading; import preserves original provenance.

        Returns
        -------
        DiagnosticEvidence
            Complete persisted diagnostic plus actual source prefix/support.

        Raises
        ------
        ValueError, OSError, RuntimeError
            Invalid/capacity/exhausted declarations, callback or publication.
            Reservations survive failures; no false completion is published.
        """
        if mode not in {"produced", "imported"} or not callable(loader):
            raise ValueError("Declare diagnostic mode and a callable loader before access")
        self._guard(design)
        self._guard(selection)
        design = ExperimentDesign.model_validate(design)
        selection = SelectionManifest.model_validate(selection)
        selection.require_design(design)
        if design.limits != self.limits:
            raise ValueError("Diagnostic design and audit capacity profiles differ")
        if datetime.now(UTC) > design.comparison_plan.test_end:
            raise ValueError("Diagnostic comparison deadline has expired")
        qualification = qualify_inference(design, selection)
        ProtocolAudit(self.root, limits=self.limits).save_selection(design, selection)
        receipt = self.sources.access(
            SourceRequest(
                schema_version=1,
                scope=design.scope,
                snapshot=design.snapshot,
                purpose="import" if mode == "imported" else "inspection",
                partitions=("test",),
                partition_manifest_fingerprint=design.partition_manifest.fingerprint,
                actor=actor,
                selection_fingerprint=selection.fingerprint,
                evaluation_fingerprint=None,
                report_fingerprint=None,
            )
        )
        prefix = self.sources.history(design.scope)
        manifests = self._manifests(prefix)
        metadata = {
            "design": design,
            "selection": selection,
            "qualification": qualification,
            "source_receipt": receipt,
            "source_prefix": prefix,
            "source_manifests": manifests,
        }
        self._guard(metadata)
        encoded = {
            "design": design.model_dump(mode="json"),
            "selection": selection.model_dump(mode="json"),
            "qualification": qualification.model_dump(mode="json"),
            "source_receipt": receipt.model_dump(mode="json"),
            "source_prefix": prefix.model_dump(mode="json"),
            "source_manifests": [manifest.model_dump(mode="json") for manifest in manifests],
        }
        self.limits.reserve(design.raw_cells, 0, metadata_bytes=wire_size(encoded), result_bytes=0, audit_receipts=0)
        supplied = loader()
        self._guard(supplied)
        panel = DiagnosticPanel.model_validate(supplied)
        if (panel.design.to_json(), panel.selection.to_json(), panel.mode) != (
            design.to_json(),
            selection.to_json(),
            mode,
        ):
            raise ValueError("Loaded diagnostic differs from its pre-load declarations")
        evidence = DiagnosticEvidence(
            schema_version=1,
            panel=panel,
            qualification=qualification,
            source_receipt=receipt,
            source_prefix=prefix,
            source_manifests=manifests,
        )
        self.save_evidence(evidence)
        return evidence

    def _declarations(self, locator: DiagnosticLocator) -> tuple[ExperimentDesign, SelectionManifest]:
        """Join result-free persisted declarations before opening cached outcomes."""
        design = read_contract(
            digest_path(self.root, "experiment-designs", locator.design_fingerprint),
            ExperimentDesign,
            self.limits.max_design_bytes,
            self.limits.max_graph_items,
        )
        selection = read_contract(
            digest_path(self.root, "experiment-selections", locator.selection_fingerprint),
            SelectionManifest,
            self.limits.max_selection_bytes,
            self.limits.max_graph_items,
        )
        selection.require_design(design)
        if (design.fingerprint, selection.fingerprint, design.scope, design.snapshot, design.limits) != (
            locator.design_fingerprint,
            locator.selection_fingerprint,
            locator.scope,
            locator.snapshot,
            self.limits,
        ):
            raise ValueError("Diagnostic locator differs from persisted declarations")
        return design, selection

    def save_evidence(self, evidence: DiagnosticEvidence) -> Path:
        """Publish complete evidence only after checking actual retained source joins.

        Parameters
        ----------
        evidence : DiagnosticEvidence
            Complete validated original diagnostic, with actual persisted access.

        Returns
        -------
        pathlib.Path
            Immutable body destination; identical publication is idempotent.

        Raises
        ------
        ValueError, OSError, RuntimeError
            Changed declarations/history, capacity or publication failure.
        """
        self._guard(evidence)
        evidence = DiagnosticEvidence.model_validate(evidence)
        locator = evidence.locator
        self._declarations(locator)
        self._require_history(evidence)
        path = digest_path(self.root, "diagnostic-evidence", locator.evidence_fingerprint)
        publish_bounded(self.writer, path, evidence.to_json() + "\n", self.limits.max_envelope_bytes)
        publish_bounded(
            self.writer,
            digest_path(self.root, "diagnostic-locators", locator.evidence_fingerprint),
            locator.to_json() + "\n",
            self.limits.max_design_bytes,
        )
        return path

    def _require_history(self, evidence: DiagnosticEvidence) -> None:
        """Verify original reservations/support against the real retained namespace."""
        locator = evidence.locator
        actual = self.sources.history(locator.scope).receipts
        retained = evidence.source_prefix.receipts
        if actual[: len(retained)] != retained:
            raise ValueError("Diagnostic prefix differs from persisted source history")
        for manifest in evidence.source_manifests:
            if self.sources.load_manifest(manifest.fingerprint) != manifest:
                raise ValueError("Diagnostic source manifest differs from persisted metadata")

    def completed(self, locator: DiagnosticLocator) -> bool:
        """Check result-free publication metadata without opening the body.

        Parameters
        ----------
        locator : DiagnosticLocator
            Exact retained metadata; malformed or changed locator fields reject.

        Returns
        -------
        bool
            Whether matching locator and body names exist, not authentication.

        Raises
        ------
        ValueError, OSError, RuntimeError
            Invalid/mismatched bounded metadata or acquisition failure.
        """
        self._guard(locator)
        locator = DiagnosticLocator.model_validate(locator)
        try:
            stored = read_contract(
                digest_path(self.root, "diagnostic-locators", locator.evidence_fingerprint),
                DiagnosticLocator,
                self.limits.max_design_bytes,
                self.limits.max_graph_items,
            )
        except FileNotFoundError:
            return False
        if stored.to_json() != locator.to_json():
            raise ValueError("Published diagnostic locator differs from its requested metadata")
        return digest_path(self.root, "diagnostic-evidence", locator.evidence_fingerprint).is_file()

    def read(self, locator: DiagnosticLocator, *, actor: str) -> DiagnosticView:
        """Reserve a new disclosure before acquiring its original diagnostic body.

        Parameters
        ----------
        locator : DiagnosticLocator
            Previously retained result-free publication metadata.
        actor : str
            Explicit actor for this new source disclosure.

        Returns
        -------
        DiagnosticView
            Original evidence and a new actual receipt; no new experiment.

        Raises
        ------
        ValueError, OSError, RuntimeError
            Incomplete/corrupt metadata/body, exhausted access or invalid joins.
            A failed body read retains its new reservation.
        """
        self._guard(locator)
        locator = DiagnosticLocator.model_validate(locator)
        if not self.completed(locator):
            raise ValueError("Diagnostic publication is incomplete")
        receipt = self.sources.access(
            SourceRequest(
                schema_version=1,
                scope=locator.scope,
                snapshot=locator.snapshot,
                purpose="result_disclosure",
                partitions=("test",),
                partition_manifest_fingerprint=None,
                actor=actor,
                selection_fingerprint=locator.selection_fingerprint,
                evaluation_fingerprint=None,
                report_fingerprint=locator.evidence_fingerprint,
            )
        )
        # Selection manifests may themselves retain development outcomes; only
        # result-free locator validation precedes this actual reservation.
        self._declarations(locator)
        evidence = read_contract(
            digest_path(self.root, "diagnostic-evidence", locator.evidence_fingerprint),
            DiagnosticEvidence,
            self.limits.max_envelope_bytes,
            self.limits.max_graph_items,
        )
        if evidence.locator.to_json() != locator.to_json():
            raise ValueError("Cached diagnostic body differs from its locator")
        self._require_history(evidence)
        return DiagnosticView(schema_version=1, original=evidence, source_receipt=receipt, viewed_at=datetime.now(UTC))
