"""Source-first optional orchestration over existing protected evaluation.

This boundary connects pure experiment declarations and raw panels to the
existing ArtifactWriter/EvaluationAudit/inference implementation. It launches no
runner, installs no capability and depends on no CLI, MCP or transport.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from coleman.artifacts import ArtifactWriter
from coleman.capabilities import require_capability
from coleman.evaluation.design._source import SourceAudit, audit_capacity, digest_path
from coleman.evaluation.design._storage import (
    BoundedEvaluationAudit,
    ProtocolCapacityError,
    publish_bounded,
    read_contract,
)
from coleman.evaluation.design._wire import guard_graph, wire_size
from coleman.evaluation.design.aggregates import require_projection_capacity
from coleman.evaluation.design.contracts import ExperimentDesign, InferenceQualification
from coleman.evaluation.design.limits import ProtocolLimits
from coleman.evaluation.design.panel import RawPanel
from coleman.evaluation.design.projection import project_frozen, qualify_inference
from coleman.evaluation.design.reports import ProtocolReport, ReportLocator, ReportView
from coleman.evaluation.design.selection import SelectionManifest
from coleman.evaluation.design.source_contracts import SourcePartitionManifest, SourcePrefix, SourceRequest
from coleman.evaluation.design.source_qualification import source_reasons
from coleman.evaluation.protected.audit import AccessReceipt
from coleman.evaluation.protected.comparison import evaluate
from coleman.evaluation.protected.contracts import FrozenEvaluation
from coleman.evaluation.protected.observations import PairedObservation
from coleman.evaluation.protected.temporal import Partition

Purpose = Literal["development", "tuning", "confirmatory", "verification"]
Mode = Literal["produced", "imported"]


class _EvaluationView(BoundedEvaluationAudit):
    """Downgrade actual legacy receipt purpose when source history is exposed."""

    force_verification = False

    def access(self, frozen: FrozenEvaluation, *, partition: Partition, purpose: Purpose, actor: str) -> AccessReceipt:
        """Use the unchanged real CAS with a truthful nonconfirmatory purpose."""
        if self.force_verification and purpose == "confirmatory":
            purpose = "verification"
        return super().access(frozen, partition=partition, purpose=purpose, actor=actor)


class ProtocolAudit:
    """Compose a frozen domain-neutral experiment without replacing its engine.

    Parameters
    ----------
    artifact_root : str or pathlib.Path
        Trusted cooperative local root. Construction creates no files.
    limits : ProtocolLimits or None, optional
        Exact versioned profile shared by each supplied design and both audits.

    Attributes
    ----------
    sources : SourceAudit
        Source reservations usable before configuration selection, including
        import/inspection/tuning. Known metadata can support narrow development
        reads; unknown or test reads conservatively expose the stable source.

    Notes
    -----
    Every loader is trusted application code. This is neither authorization nor
    a sandbox/global monitor. Source and legacy chains are separate reservations:
    interruption may conservatively consume access without a completed result.
    Concurrent contenders can yield zero or one confirmations, never two; first
    reservation is not a promise that the first finisher will confirm. Complete
    publication uses the existing immutable atomic writer and its local limits.
    """

    def __init__(self, artifact_root: str | Path, *, limits: ProtocolLimits | None = None) -> None:
        """Select bounded adapters and local paths without loading or publishing data."""
        self.root = Path(artifact_root)
        self.limits = ProtocolLimits(schema_version=1) if limits is None else ProtocolLimits.model_validate(limits)
        self.sources = SourceAudit(self.root, limits=self.limits)
        self.writer = ArtifactWriter(self.root)

    def _view(self, *, verification: bool = False) -> _EvaluationView:
        """Select bounded old readers and an actual truthful receipt purpose."""
        result = _EvaluationView(self.root, audit_capacity(self.limits))
        result.force_verification = verification
        return result

    def _guard(self, value: object) -> None:
        """Bound the selected whole graph before nested old validators/serializers."""
        guard_graph(value, maximum_bytes=self.limits.max_envelope_bytes, maximum_items=self.limits.max_graph_items)

    def save_design(self, design: ExperimentDesign) -> Path:
        """Persist the exact declaration and complete original source metadata.

        Parameters
        ----------
        design : ExperimentDesign
            Complete bounded declaration with this adapter's exact limits.

        Returns
        -------
        pathlib.Path
            Immutable design path; identical publication is idempotent.

        Raises
        ------
        ValueError, OSError, ProtocolCapacityError
            Invalid/mismatched profile, source header conflict or I/O failure.
            No outcomes are loaded and no exposure is reserved by this operation.
        """
        self._guard(design)
        design = ExperimentDesign.model_validate(design)
        if design.limits != self.limits:
            raise ValueError("Experiment and audit require the same declared capacity profile")
        self.sources.save_manifest(design.partition_manifest)
        path = digest_path(self.root, "experiment-designs", design.fingerprint)
        publish_bounded(self.writer, path, design.to_json() + "\n", self.limits.max_design_bytes)
        return path

    def save_selection(self, design: ExperimentDesign, selection: SelectionManifest) -> Path:
        """Persist development-only selection before any protected loader runs.

        Parameters
        ----------
        design : ExperimentDesign
            Exact final source, law, quantity and complete hypothesis family.
        selection : SelectionManifest
            Exact bound selection with actual supplied selection time and all
            declared development trials/attempts, including nonfinalists.

        Returns
        -------
        pathlib.Path
            Immutable selection path. Identical publication is idempotent.

        Raises
        ------
        ValueError, OSError, ProtocolCapacityError
            Invalid/future selection, mismatched design or publication failure.
        """
        self._guard(design)
        design = ExperimentDesign.model_validate(design)
        self._guard(selection)
        selection = SelectionManifest.model_validate(selection)
        selection.require_design(design)
        if selection.actual_selected_at > datetime.now(UTC):
            raise ValueError("Actual selection cannot be in the future at persistence")
        self.save_design(design)
        path = digest_path(self.root, "experiment-selections", selection.fingerprint)
        publish_bounded(self.writer, path, selection.to_json() + "\n", self.limits.max_selection_bytes)
        return path

    def _manifests(self, prefix: SourcePrefix) -> tuple[SourcePartitionManifest, ...]:
        """Retain actual persisted support for each distinct narrower claim."""
        known = {}
        byte_count = 2
        for receipt in prefix.receipts:
            digest = receipt.request.partition_manifest_fingerprint
            if not receipt.partition_manifest_verified or digest in known:
                continue
            manifest = self.sources.verified_manifest(receipt.request)
            if manifest is not None:
                byte_count += wire_size(manifest) + 1
                if byte_count > self.limits.max_metadata_bytes:
                    raise ProtocolCapacityError("Retained source manifests exceed the complete metadata allowance")
                known[manifest.fingerprint] = manifest
        result = tuple(value for _, value in sorted(known.items()))
        guard_graph(result, maximum_bytes=self.limits.max_metadata_bytes, maximum_items=self.limits.max_graph_items)
        return result

    def _require_metadata(
        self,
        design: ExperimentDesign,
        selection: SelectionManifest,
        frozen: FrozenEvaluation,
        manifests: tuple[SourcePartitionManifest, ...],
    ) -> None:
        """Charge every known repeated metadata occurrence before the outcome callback."""
        metadata = {"design": design, "selection": selection, "frozen": frozen, "source_manifests": manifests}
        guard_graph(metadata, maximum_bytes=self.limits.max_metadata_bytes, maximum_items=self.limits.max_graph_items)
        encoded = {
            "design": design.model_dump(mode="json"),
            "selection": selection.model_dump(mode="json"),
            "frozen": frozen.model_dump(mode="json"),
            "source_manifests": [manifest.model_dump(mode="json") for manifest in manifests],
        }
        design.limits.reserve(design.raw_cells, design.aggregate_inputs, metadata_bytes=wire_size(encoded))

    def evaluate(  # pylint: disable=too-many-locals
        self,
        design: ExperimentDesign,
        selection: SelectionManifest,
        loader: Callable[[], RawPanel],
        *,
        actor: str,
        mode: Mode,
    ) -> ProtocolReport | InferenceQualification:
        """Reserve source access, load one complete panel and publish verified results.

        Parameters
        ----------
        design, selection : ExperimentDesign, SelectionManifest
            Exact frozen declarations. Selection is persisted before source
            reservation; historical eligibility is not moved to the current date.
        loader : callable
            Trusted application callback returning its actual complete RawPanel.
            It runs once only after both real audit reservations. This adapter
            neither launches jobs nor invents physical work/cost measurements.
        actor : str
            Explicit caller identity retained in both access chains.
        mode : {'produced', 'imported'}
            Declared before loading. Imports are always verification and keep
            their original raw evaluation provenance.

        Returns
        -------
        ProtocolReport or InferenceQualification
            Atomically completed report, or a known unsupported inference request
            before loading/persistence. Missing outcomes yield insufficient
            evidence in the complete report rather than dropping cells.

        Raises
        ------
        ValueError, OSError, ProtocolCapacityError, RuntimeError
            Invalid mode/contracts, capacity, loader, capability, inference or
            publication failure. Completed reservations remain visible. No false
            complete protocol locator is published when a later step fails.

        Notes
        -----
        Existing paired bounded/t inference computes the family. Source exposure
        sets an actual legacy verification receipt and, when needed, a real
        current development marker; a wrapper flag cannot bless a nested report.
        """
        if mode not in {"produced", "imported"}:
            raise ValueError("Declare produced or imported mode before source loading")
        self._guard(design)
        self._guard(selection)
        qualification = qualify_inference(design, selection)
        if qualification.status == "unsupported":
            return qualification
        if datetime.now(UTC) > design.comparison_plan.test_end:
            raise ValueError("Frozen comparison deadline has already expired")
        require_projection_capacity(design, selection)
        if design.inference.method == "paired_t_holm_bonferroni_ci_v1":
            require_capability("analytics")
        self.save_selection(design, selection)
        frozen = project_frozen(design, selection)
        source = self.sources.access(
            SourceRequest(
                schema_version=1,
                scope=design.scope,
                snapshot=design.snapshot,
                purpose="import" if mode == "imported" else "protected_evaluation",
                partitions=("test",),
                partition_manifest_fingerprint=design.partition_manifest.fingerprint,
                actor=actor,
                selection_fingerprint=selection.fingerprint,
                evaluation_fingerprint=frozen.fingerprint,
                report_fingerprint=None,
            )
        )
        prefix = self.sources.history(design.scope)
        manifests = self._manifests(prefix)
        self._require_metadata(design, selection, frozen, manifests)
        exposed = bool(source_reasons(prefix, source, selection, mode, manifests))
        view = self._view(verification=exposed)
        if exposed and not any(
            item.partition == "test" and item.purpose == "development" for item in view.history(design.scope.holdout_id)
        ):
            view.access(frozen, partition="test", purpose="development", actor=actor)
        panel: RawPanel | None = None

        def load() -> tuple[PairedObservation, ...]:
            """Load once after both reservations and validate exact pre-load bindings."""
            nonlocal panel
            supplied = loader()
            self._guard(supplied)
            panel = RawPanel.model_validate(supplied)
            if panel.design != design or panel.selection != selection or panel.mode != mode:
                raise ValueError("Loaded panel differs from its pre-load declarations")
            return panel.paired_observations()

        family = evaluate(frozen, view, load, actor=actor)
        if panel is None:
            raise RuntimeError("Protected evaluation completed without its required panel")
        report = ProtocolReport(
            schema_version=1,
            panel=panel,
            source_receipt=source,
            source_prefix=prefix,
            source_manifests=manifests,
            family=family,
            completed_at=datetime.now(UTC),
        )
        self.save_report(report)
        return report

    def _require_declarations(self, locator: ReportLocator) -> tuple[ExperimentDesign, SelectionManifest]:
        """Validate persisted joins; retained development selection can contain outcomes."""
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
        actual = (
            design.fingerprint,
            design.scope,
            design.snapshot,
            design.limits,
            selection.fingerprint,
            project_frozen(design, selection).fingerprint,
        )
        expected = (
            locator.design_fingerprint,
            locator.scope,
            locator.snapshot,
            self.limits,
            locator.selection_fingerprint,
            locator.evaluation_fingerprint,
        )
        if actual != expected:
            raise ValueError("Persisted protocol metadata differs from its locator")
        return design, selection

    def save_report(self, report: ProtocolReport) -> Path:
        """Verify actual persisted joins, then atomically publish body and locator.

        Parameters
        ----------
        report : ProtocolReport
            Complete original result already held by the caller. Repeating this
            publication is idempotent, not a new result disclosure.

        Returns
        -------
        pathlib.Path
            Complete body destination. Completion requires the separate immutable
            result-free locator too; an orphan body is never a completed lookup.

        Raises
        ------
        ValueError, OSError, ProtocolCapacityError
            Unreserved source/legacy access, changed declarations or failed
            publication. Failure may leave reservations or an orphan body.
        """
        self._guard(report)
        report = ProtocolReport.model_validate(report)
        locator = report.locator
        self._require_declarations(locator)
        prefix = self.sources.history(report.panel.design.scope)
        retained = report.source_prefix.receipts
        if prefix.receipts[: len(retained)] != retained:
            raise ValueError("Report source prefix differs from actual persisted history")
        for manifest in report.source_manifests:
            if self.sources.load_manifest(manifest.fingerprint) != manifest:
                raise ValueError("Report source metadata differs from the persisted manifest")
        self._view().save_report(report.family)
        path = digest_path(self.root, "protocol-reports", locator.report_fingerprint)
        publish_bounded(self.writer, path, report.to_json() + "\n", self.limits.max_envelope_bytes)
        publish_bounded(
            self.writer,
            digest_path(self.root, "protocol-locators", locator.report_fingerprint),
            locator.to_json() + "\n",
            self.limits.max_design_bytes,
        )
        return path

    def completed(self, locator: ReportLocator) -> bool:
        """Check publication metadata only, without disclosing result bodies.

        Parameters
        ----------
        locator : ReportLocator
            Caller-retained exact metadata returned by a completed report.

        Returns
        -------
        bool
            False when locator/body publication is absent. True means both names
            are present with matching locator, not that arbitrary external byte
            mutation is authenticated. No outcomes or report body are read.

        Raises
        ------
        ValueError, OSError, ProtocolCapacityError
            Existing metadata is corrupt, mismatched or exceeds capacity.
        """
        self._guard(locator)
        locator = ReportLocator.model_validate(locator)
        path = digest_path(self.root, "protocol-locators", locator.report_fingerprint)
        try:
            stored = read_contract(path, ReportLocator, self.limits.max_design_bytes, self.limits.max_graph_items)
        except FileNotFoundError:
            return False
        if stored != locator:
            raise ValueError("Published protocol locator differs from its requested metadata")
        return digest_path(self.root, "protocol-reports", locator.report_fingerprint).is_file()

    def read_report(self, locator: ReportLocator, *, actor: str) -> ReportView:
        """Reserve a new disclosure before opening the exact original cached body.

        Parameters
        ----------
        locator : ReportLocator
            Exact original metadata, checked against its persisted locator and
            declarations before any result-bearing body is opened.
        actor : str
            Explicit caller identity for this new result disclosure.

        Returns
        -------
        ReportView
            Unchanged original report plus new source/legacy verification
            receipts. No new physical run or confirmation is claimed.

        Raises
        ------
        ValueError, OSError, ProtocolCapacityError
            Incomplete/mismatched metadata, exhausted reservations or invalid
            cached body. Failed delivery still leaves its actual access receipts.
        """
        self._guard(locator)
        locator = ReportLocator.model_validate(locator)
        if not self.completed(locator):
            raise ValueError("Protocol report publication is incomplete")
        source = self.sources.access(
            SourceRequest(
                schema_version=1,
                scope=locator.scope,
                snapshot=locator.snapshot,
                purpose="result_disclosure",
                partitions=("test",),
                partition_manifest_fingerprint=None,
                actor=actor,
                selection_fingerprint=locator.selection_fingerprint,
                evaluation_fingerprint=locator.evaluation_fingerprint,
                report_fingerprint=locator.report_fingerprint,
            )
        )
        view = self._view()
        frozen = view.load_freeze(locator.evaluation_fingerprint)
        if (
            frozen.protocol.holdout_id != locator.scope.holdout_id
            or frozen.data_fingerprint != locator.snapshot.data_fingerprint
        ):
            raise ValueError("Cached evaluation metadata differs from its persisted locator")
        access = view.access(frozen, partition="test", purpose="verification", actor=actor)
        self._require_declarations(locator)
        path = digest_path(self.root, "protocol-reports", locator.report_fingerprint)
        report = read_contract(path, ProtocolReport, self.limits.max_envelope_bytes, self.limits.max_graph_items)
        if report.locator != locator:
            raise ValueError("Cached result body differs from its pre-load locator")
        return ReportView(
            schema_version=1, original=report, source_receipt=source, access=access, viewed_at=datetime.now(UTC)
        )


__all__ = ["ProtocolAudit", "ProtocolCapacityError", "SourceAudit"]
