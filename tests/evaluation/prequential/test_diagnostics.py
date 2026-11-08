"""Actual source reservations survive failures, retries, imports and disclosures."""

import shutil

import pytest

from coleman.evaluation.design.audit import ProtocolAudit
from coleman.evaluation.design.contracts import InferenceQualification
from coleman.evaluation.design.reports import ProtocolReport
from coleman.evaluation.design.source_contracts import SourceRequest
from coleman.evaluation.prequential.diagnostics import DiagnosticAudit
from coleman.evaluation.prequential.evidence import DiagnosticEvidence, DiagnosticView
from coleman.evaluation.protected.audit import EvaluationAudit
from tests.evaluation.design_support import changed
from tests.evaluation.prequential.diagnostic_support import declarations, diagnostic_panel, protected_panel


def test_optional_construction_has_no_filesystem_effect(tmp_path):
    root = tmp_path / "absent"
    DiagnosticAudit(root)
    assert not root.exists()


def test_old_unsupported_route_still_has_zero_callback_and_zero_reservations(tmp_path):
    design, selection = declarations(unsupported=True)
    calls = []
    result = ProtocolAudit(tmp_path).evaluate(design, selection, lambda: calls.append(1), actor="test", mode="produced")
    assert isinstance(result, InferenceQualification)
    assert result.status == "unsupported"
    assert calls == []
    assert not tmp_path.joinpath("source-access").exists()
    assert not tmp_path.joinpath("evaluation-access").exists()


def test_real_descriptive_reservation_precedes_loader_without_legacy_inference(tmp_path):
    design, selection = declarations(unsupported=True)
    audit = DiagnosticAudit(tmp_path)
    calls = []

    def load():
        assert len(audit.sources.history(design.scope).receipts) == 1
        assert EvaluationAudit(tmp_path).history(design.scope.holdout_id) == ()
        calls.append("actual")
        return diagnostic_panel(design, selection)

    evidence = audit.evaluate(design, selection, load, actor="descriptive", mode="produced")
    assert calls == ["actual"]
    assert evidence.qualification.status == "unsupported"
    assert evidence.source_receipt.request.purpose == "inspection"
    assert evidence.source_receipt.request.evaluation_fingerprint is None
    assert audit.completed(evidence.locator)
    assert DiagnosticEvidence.from_json(evidence.to_json()).to_json() == evidence.to_json()
    assert audit.save_evidence(evidence).is_file()
    assert len(audit.sources.history(design.scope).receipts) == 1


def test_failed_loader_retry_then_supported_evaluation_cannot_be_virgin(tmp_path):
    design, selection = declarations()
    audit = DiagnosticAudit(tmp_path)

    def fail():
        raise RuntimeError("actual loader failure")

    with pytest.raises(RuntimeError, match="actual loader failure"):
        audit.evaluate(design, selection, fail, actor="failed", mode="produced")
    assert len(audit.sources.history(design.scope).receipts) == 1
    assert not tmp_path.joinpath("diagnostic-locators").exists()
    evidence = audit.evaluate(
        design, selection, lambda: diagnostic_panel(design, selection), actor="retry", mode="produced"
    )
    assert evidence.source_receipt.sequence == 2
    result = ProtocolAudit(tmp_path).evaluate(
        design, selection, lambda: protected_panel(design, selection), actor="later", mode="produced"
    )
    assert isinstance(result, ProtocolReport)
    assert not result.confirmatory_eligible
    assert "previous_protected_source_access" in result.source_reasons
    assert result.family.access.purpose == "verification"


def test_import_and_cached_disclosure_retain_original_records_and_new_access(tmp_path):
    design, selection = declarations(unsupported=True)
    audit = DiagnosticAudit(tmp_path)
    original = diagnostic_panel(design, selection, mode="imported")
    evidence = audit.evaluate(design, selection, lambda: original, actor="import", mode="imported")
    assert evidence.source_receipt.request.purpose == "import"
    assert tuple(row.record.to_json() for row in evidence.panel.rows) == tuple(
        row.record.to_json() for row in original.rows
    )
    view = audit.read(evidence.locator, actor="reader")
    assert isinstance(view, DiagnosticView)
    assert view.original.to_json() == evidence.to_json()
    assert view.source_receipt.sequence == 2
    assert view.source_receipt.request.purpose == "result_disclosure"
    assert view.source_receipt.request.evaluation_fingerprint is None
    assert DiagnosticView.from_json(view.to_json()).to_json() == view.to_json()
    restarted = view.model_dump(mode="json")
    restarted["source_receipt"]["sequence"] = 1
    restarted["source_receipt"]["previous_fingerprint"] = None
    with pytest.raises(ValueError, match="retained source ordinal"):
        DiagnosticView.model_validate(restarted)
    wrong_previous = view.model_dump(mode="json")
    wrong_previous["source_receipt"]["previous_fingerprint"] = "sha256:" + "a" * 64
    with pytest.raises(ValueError, match="retained source predecessor"):
        DiagnosticView.model_validate(wrong_previous)
    second_view = audit.read(evidence.locator, actor="another-reader")
    assert second_view.source_receipt.sequence == 3
    assert second_view.original.to_json() == evidence.to_json()


def test_corrupt_body_is_read_only_after_new_disclosure_reservation(tmp_path):
    design, selection = declarations(unsupported=True)
    audit = DiagnosticAudit(tmp_path)
    evidence = audit.evaluate(
        design, selection, lambda: diagnostic_panel(design, selection), actor="run", mode="produced"
    )
    path = audit.save_evidence(evidence)
    path.write_bytes(b"{")
    with pytest.raises(ValueError):
        audit.read(evidence.locator, actor="failed-reader")
    assert len(audit.sources.history(design.scope).receipts) == 2


def test_wrong_loaded_selection_retains_attempt_without_false_completion(tmp_path):
    design, selection = declarations(unsupported=True)
    other_design, other_selection = declarations(unsupported=True)
    audit = DiagnosticAudit(tmp_path)
    supplied = diagnostic_panel(other_design, other_selection)
    with pytest.raises(ValueError, match="pre-load declarations"):
        audit.evaluate(design, selection, lambda: supplied, actor="mismatch", mode="produced")
    assert len(audit.sources.history(design.scope).receipts) == 1
    assert not tmp_path.joinpath("diagnostic-locators").exists()


def test_absent_locator_does_not_reserve_a_body_read(tmp_path):
    design, selection = declarations(unsupported=True)
    audit = DiagnosticAudit(tmp_path)
    evidence = audit.evaluate(
        design, selection, lambda: diagnostic_panel(design, selection), actor="run", mode="produced"
    )
    locator = evidence.locator.model_copy(update={"evidence_fingerprint": "sha256:" + "f" * 64})
    with pytest.raises(ValueError, match="incomplete"):
        audit.read(locator, actor="no-body")
    assert len(audit.sources.history(design.scope).receipts) == 1


@pytest.fixture(scope="module")
def retained_diagnostic(tmp_path_factory):
    """One real completed input for pure malformed-artifact refusal tests."""
    design, selection = declarations(unsupported=True)
    root = tmp_path_factory.mktemp("retained-diagnostic")
    audit = DiagnosticAudit(root)
    evidence = audit.evaluate(
        design,
        selection,
        lambda: diagnostic_panel(design, selection),
        actor="complete-before-rejections",
        mode="produced",
    )
    return audit, evidence


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("purpose", "tuning"),
        ("evaluation_fingerprint", "sha256:" + "e" * 64),
        ("report_fingerprint", "sha256:" + "f" * 64),
        ("selection_fingerprint", "sha256:" + "a" * 64),
        ("partitions", ["validation"]),
    ],
)
def test_actual_diagnostic_reader_rejects_wrong_preload_joins(retained_diagnostic, field, value):
    audit, original = retained_diagnostic
    path = audit.save_evidence(original)
    before = path.read_bytes()
    data = original.model_dump(mode="json")
    data["source_receipt"]["request"][field] = value
    with pytest.raises(ValueError, match="Diagnostic reservation differs"):
        DiagnosticEvidence.model_validate(data)
    assert path.read_bytes() == before
    assert len(audit.sources.history(original.panel.design.scope).receipts) == 1


def test_absent_prefix_and_duplicate_metadata_reject(retained_diagnostic):
    _audit, original = retained_diagnostic
    absent = original.model_dump(mode="json")
    absent["source_prefix"]["receipts"] = []
    with pytest.raises(ValueError, match="absent from its complete prefix"):
        DiagnosticEvidence.model_validate(absent)
    duplicated = original.model_dump(mode="python")
    manifest = original.panel.design.partition_manifest
    duplicated["source_manifests"] = (manifest, manifest)
    with pytest.raises(ValueError, match="manifests must be unique"):
        DiagnosticEvidence.model_validate(duplicated)


def test_known_narrow_history_keeps_its_exact_manifest_support(tmp_path):
    design, selection = declarations(unsupported=True)
    audit = DiagnosticAudit(tmp_path)
    manifest = design.partition_manifest
    audit.sources.save_manifest(manifest)
    audit.sources.access(
        SourceRequest(
            schema_version=1,
            scope=design.scope,
            snapshot=design.snapshot,
            purpose="inspection",
            partitions=("validation",),
            partition_manifest_fingerprint=manifest.fingerprint,
            actor="declared-development-inspection",
            selection_fingerprint=None,
            evaluation_fingerprint=None,
            report_fingerprint=None,
        )
    )
    original = audit.evaluate(
        design, selection, lambda: diagnostic_panel(design, selection), actor="diagnostic", mode="produced"
    )
    assert original.source_manifests == (manifest,)
    data = original.model_dump(mode="python")
    data["source_manifests"] = ()
    with pytest.raises(ValueError, match="verified source manifest support"):
        DiagnosticEvidence.model_validate(data)


def test_selected_graph_capacity_refuses_before_any_loader_or_reservation(tmp_path):
    design, selection = declarations(unsupported=True)
    limits = design.limits.model_copy(update={"max_graph_items": 1})
    supplied = design.model_copy(update={"limits": limits})
    audit = DiagnosticAudit(tmp_path, limits=limits)
    calls = []
    with pytest.raises(ValueError, match="Graph item budget"):
        audit.evaluate(supplied, selection, lambda: calls.append("forbidden"), actor="capacity", mode="produced")
    assert calls == []
    assert not tmp_path.joinpath("source-access").exists()


def test_wrong_cached_locator_fails_without_disclosing_body(retained_diagnostic):
    audit, evidence = retained_diagnostic
    locator = changed(evidence.locator, selection_fingerprint="sha256:" + "1" * 64)
    with pytest.raises(ValueError, match="Published diagnostic locator differs"):
        audit.read(locator, actor="wrong-metadata")
    assert len(audit.sources.history(evidence.panel.design.scope).receipts) == 1


def test_cached_body_cannot_replace_original_source_history_with_a_fresh_root(tmp_path, retained_diagnostic):
    audit, evidence = retained_diagnostic
    for namespace in ("experiment-designs", "experiment-selections", "diagnostic-evidence", "diagnostic-locators"):
        shutil.copytree(audit.root / namespace, tmp_path / namespace)
    fresh = DiagnosticAudit(tmp_path)
    with pytest.raises(ValueError, match="persisted source history"):
        fresh.read(evidence.locator, actor="copied-cache-without-original-history")
    history = fresh.sources.history(evidence.panel.design.scope).receipts
    assert len(history) == 1
    assert history[0].request.purpose == "result_disclosure"
    assert history[0].access_id != evidence.source_receipt.access_id


def test_cached_development_selection_is_not_opened_before_disclosure_reservation(tmp_path):
    design, selection = declarations(unsupported=True)
    audit = DiagnosticAudit(tmp_path)
    evidence = audit.evaluate(
        design,
        selection,
        lambda: diagnostic_panel(design, selection),
        actor="before-corrupt-selection",
        mode="produced",
    )
    # A selection can retain inspected development outcomes. Even its failed
    # acquisition must occur after the new result-disclosure reservation.
    path = tmp_path / "experiment-selections" / (selection.fingerprint[7:] + ".json")
    path.write_bytes(b"{")
    with pytest.raises(ValueError):
        audit.read(evidence.locator, actor="failed-selection-read")
    history = audit.sources.history(design.scope).receipts
    assert len(history) == 2
    assert history[-1].request.purpose == "result_disclosure"
