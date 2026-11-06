"""Real source-first orchestration, contamination, cached access and interruption."""

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest
from design_support import changed, make_design, make_raw_panel, make_selection

from coleman.evaluation.design.audit import ProtocolAudit, ProtocolCapacityError, SourceAudit
from coleman.evaluation.design.contracts import InferenceQualification
from coleman.evaluation.design.reports import ProtocolReport, ReportLocator, ReportView
from coleman.evaluation.design.selection import AttemptLedger, ConfigurationTrial
from coleman.evaluation.design.source_contracts import SourceRequest
from coleman.evaluation.protected.audit import EvaluationAudit


def live_declarations(**design_updates):
    """Keep authored historical events while using actual current analysis time."""
    now = datetime.now(UTC)
    original = make_design()
    design = changed(
        original,
        comparison_plan=changed(original.comparison_plan, test_end=now + timedelta(days=1)),
        snapshot=changed(original.snapshot, captured_at=None),
        **design_updates,
    )
    return design, make_selection(design, actual_selected_at=now - timedelta(minutes=1))


def actual_panel(design, selection, *, mode="produced"):
    """Create authored values at actual callback time; no real CI job is claimed."""
    now = datetime.now(UTC)
    times = {
        (candidate.candidate.candidate_id, member.member_id, seed): now
        for candidate in design.candidates
        for member in design.protected_members
        for seed in design.seeds
    }
    return make_raw_panel(design, selection, mode=mode, record_times=times, assembled_at=now)


def source_request(design, *, purpose="inspection", partitions=("test",), manifest=None):
    return SourceRequest(
        schema_version=1,
        scope=design.scope,
        snapshot=design.snapshot,
        purpose=purpose,
        partitions=partitions,
        partition_manifest_fingerprint=manifest,
        actor="real-audit-test",
        selection_fingerprint=None,
        evaluation_fingerprint=None,
        report_fingerprint=None,
    )


def run(audit, design, selection, *, mode="produced"):
    result = audit.evaluate(
        design, selection, lambda: actual_panel(design, selection, mode=mode), actor="actual-callback", mode=mode
    )
    assert isinstance(result, ProtocolReport)
    return result


def test_construction_does_not_create_store(tmp_path):
    root = tmp_path / "absent"
    ProtocolAudit(root)
    SourceAudit(root)
    assert not root.exists()


def test_complete_real_composition_round_trip_and_idempotent_publication(tmp_path):
    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)
    report = run(audit, design, selection)
    assert report.confirmatory_eligible
    assert report.source_reasons == ()
    assert report.family.results[0].conclusion == "insufficient_evidence"
    assert report.family.results[0].observed_pairs == 2
    assert ProtocolReport.from_json(report.to_json()) == report
    assert ProtocolReport.from_json(report.to_json()).fingerprint == report.fingerprint
    assert ReportLocator.from_json(report.locator.to_json()) == report.locator
    first = audit.save_report(report)
    before = first.read_bytes()
    assert audit.save_report(report) == first
    assert first.read_bytes() == before
    assert len(audit.sources.history(design.scope).receipts) == 1
    assert len(EvaluationAudit(tmp_path).history(design.scope.holdout_id)) == 1
    assert audit.completed(report.locator)


def test_both_reservations_and_declarations_exist_before_callback(tmp_path):
    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)
    calls = []

    def load():
        assert len(audit.sources.history(design.scope).receipts) == 1
        assert len(EvaluationAudit(tmp_path).history(design.scope.holdout_id)) == 1
        assert len(tuple((tmp_path / "experiment-designs").glob("*.json"))) == 1
        assert len(tuple((tmp_path / "experiment-selections").glob("*.json"))) == 1
        calls.append("loaded")
        return actual_panel(design, selection)

    result = audit.evaluate(design, selection, load, actor="checks-order", mode="produced")
    assert isinstance(result, ProtocolReport)
    assert calls == ["loaded"]


def test_prior_inspection_downgrades_nested_legacy_report_and_creates_real_marker(tmp_path):
    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)
    prior = audit.sources.access(source_request(design))
    report = run(audit, design, selection)
    assert prior.sequence == 1
    assert report.source_receipt.sequence == 2
    assert "previous_protected_source_access" in report.source_reasons
    assert not report.confirmatory_eligible
    assert not report.family.confirmatory_eligible
    history = EvaluationAudit(tmp_path).history(design.scope.holdout_id)
    assert tuple(item.purpose for item in history) == ("development", "verification")
    assert report.family.access == history[-1]


@pytest.mark.parametrize("prehistory", ["unknown", "known_exposed"])
def test_empty_fresh_directory_does_not_reset_declared_prehistory(tmp_path, prehistory):
    design, _selection = live_declarations()
    design = changed(design, scope=changed(design.scope, prehistory=prehistory))
    selection = make_selection(design, actual_selected_at=datetime.now(UTC) - timedelta(minutes=1))
    report = run(ProtocolAudit(tmp_path), design, selection)
    assert not report.family.confirmatory_eligible
    assert report.family.access.purpose == "verification"
    assert "source_prehistory_not_declared_unexposed" in report.source_reasons


def test_import_preserves_original_raw_digest_and_never_confirms(tmp_path):
    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)
    original = actual_panel(design, selection, mode="imported")
    original_wires = tuple(row.record.to_json() for row in original.rows)
    result = audit.evaluate(design, selection, lambda: original, actor="original-import", mode="imported")
    assert isinstance(result, ProtocolReport)
    assert result.source_receipt.request.purpose == "import"
    assert result.family.access.purpose == "verification"
    assert not result.confirmatory_eligible
    assert tuple(row.record.to_json() for row in result.panel.rows) == original_wires
    assert all(
        row.record.provenance.evaluation_fingerprint != result.family.frozen.fingerprint for row in result.panel.rows
    )


def test_exact_persisted_manifest_allows_narrow_development_read(tmp_path):
    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)
    manifest = design.partition_manifest
    audit.sources.save_manifest(manifest)
    request = source_request(design, partitions=("validation",), manifest=manifest.fingerprint)
    assert audit.sources.verified_manifest(request) == manifest
    audit.sources.access(request)
    report = run(audit, design, selection)
    assert report.confirmatory_eligible
    assert report.source_manifests == (manifest,)


def test_missing_manifest_or_unknown_scope_partition_is_conservative_exposure(tmp_path):
    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)
    request = source_request(design, partitions=("validation",), manifest="sha256:" + "f" * 64)
    assert audit.sources.verified_manifest(request) is None
    audit.sources.access(request)
    assert not run(audit, design, selection).family.confirmatory_eligible


def test_later_manifest_publication_cannot_retroactively_clean_an_unverified_read(tmp_path):
    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)
    manifest = design.partition_manifest
    request = source_request(design, partitions=("validation",), manifest=manifest.fingerprint)
    prior = audit.sources.access(request)
    assert not prior.partition_manifest_verified
    audit.sources.save_manifest(manifest)
    assert audit.sources.verified_manifest(request) == manifest
    report = run(audit, design, selection)
    assert not report.family.confirmatory_eligible
    assert "previous_protected_source_access" in report.source_reasons
    assert not audit.sources.history(design.scope).receipts[0].partition_manifest_verified


def test_scope_version_and_snapshot_changes_cannot_rotate_the_same_chain(tmp_path):
    design, _selection = live_declarations()
    sources = SourceAudit(tmp_path)
    first = sources.access(source_request(design))
    replacement = changed(design.scope, reference=changed(design.scope.reference, version="2"))
    assert replacement.logical_key == design.scope.logical_key
    request = changed(source_request(design), scope=replacement)
    with pytest.raises(ValueError):
        sources.access(request)
    revised_snapshot = changed(design.snapshot, reference=changed(design.snapshot.reference, version="2"))
    second = sources.access(changed(source_request(design), snapshot=revised_snapshot))
    assert second.sequence == 2
    assert second.previous_fingerprint == first.fingerprint
    assert len(sources.history(design.scope).receipts) == 2


def test_development_trial_test_exposure_cannot_be_hidden_by_empty_source_store(tmp_path):
    design, selection = live_declarations()
    trial = ConfigurationTrial(
        trial_id="inspected-on-test",
        candidate=design.candidates[0],
        status="inspected",
        partition="test",
        completed_at=selection.actual_selected_at,
    )
    ledger = AttemptLedger(schema_version=1, stage="development", trials=(trial,))
    selection = changed(selection, ledger=ledger)
    report = run(ProtocolAudit(tmp_path), design, selection)
    assert "development_used_protected_or_unknown_data" in report.source_reasons
    assert report.family.access.purpose == "verification"


def test_loader_failure_keeps_both_attempts_but_no_complete_report(tmp_path):
    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)

    def failed():
        raise RuntimeError("deliberate actual callback interruption")

    with pytest.raises(RuntimeError, match="callback interruption"):
        audit.evaluate(design, selection, failed, actor="interrupted", mode="produced")
    assert len(audit.sources.history(design.scope).receipts) == 1
    assert len(EvaluationAudit(tmp_path).history(design.scope.holdout_id)) == 1
    assert not tuple((tmp_path / "protocol-locators").glob("*.json"))
    resumed = run(audit, design, selection)
    assert not resumed.family.confirmatory_eligible


def test_actual_process_exit_after_reservation_recovers_without_fresh_confirmation(tmp_path):
    design, selection = live_declarations()
    design_path, selection_path = tmp_path / "input-design.json", tmp_path / "input-selection.json"
    design_path.write_text(design.to_json(), encoding="utf-8")
    selection_path.write_text(selection.to_json(), encoding="utf-8")
    root = tmp_path / "interrupted-store"
    script = """
import os,sys
from pathlib import Path
from coleman.evaluation.design.audit import ProtocolAudit
from coleman.evaluation.design.contracts import ExperimentDesign
from coleman.evaluation.design.selection import SelectionManifest
design=ExperimentDesign.from_json(Path(sys.argv[2]).read_bytes())
selection=SelectionManifest.from_json(Path(sys.argv[3]).read_bytes())
def interrupted():
 os._exit(73)
ProtocolAudit(sys.argv[1]).evaluate(design,selection,interrupted,actor='abrupt-process',mode='produced')
raise AssertionError('Interrupted producer returned')
"""
    stopped = subprocess.run(
        [sys.executable, "-c", script, str(root), str(design_path), str(selection_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert stopped.returncode == 73, stopped.stdout + stopped.stderr
    audit = ProtocolAudit(root)
    old = audit.sources.history(design.scope).receipts
    assert len(old) == len(EvaluationAudit(root).history(design.scope.holdout_id)) == 1
    assert not tuple((root / "protocol-locators").glob("*.json"))
    assert not tuple((root / "protocol-reports").glob("*.json"))
    recovered = run(audit, design, selection)
    assert recovered.source_receipt.sequence == 2
    assert recovered.source_receipt.previous_fingerprint == old[0].fingerprint
    assert recovered.family.access.purpose == "verification"
    assert not recovered.family.confirmatory_eligible
    assert audit.completed(recovered.locator)


def test_interrupted_locator_publication_is_incomplete_and_retry_is_metadata_only(tmp_path, monkeypatch):
    from coleman.evaluation.design import audit as audit_module

    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)
    captured = []
    actual_publish = audit_module.publish_bounded

    def interrupted(writer, path, *args, **kwargs):
        if path.parent.name == "protocol-locators":
            raise OSError("deliberate locator publication interruption")
        return actual_publish(writer, path, *args, **kwargs)

    actual_save = audit.save_report

    def retain_and_save(report):
        captured.append(report)
        return actual_save(report)

    monkeypatch.setattr(audit_module, "publish_bounded", interrupted)
    monkeypatch.setattr(audit, "save_report", retain_and_save)
    with pytest.raises(OSError, match="locator publication"):
        run(audit, design, selection)
    report = captured[0]
    assert tuple((tmp_path / "protocol-reports").glob("*.json"))
    assert not audit.completed(report.locator)
    monkeypatch.setattr(audit_module, "publish_bounded", actual_publish)
    actual_save(report)
    assert audit.completed(report.locator)
    assert len(audit.sources.history(design.scope).receipts) == 1
    assert len(EvaluationAudit(tmp_path).history(design.scope.holdout_id)) == 1


def test_cached_view_is_new_disclosure_without_reexecuting_or_rewriting_original(tmp_path):
    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)
    original = run(audit, design, selection)
    original_wire = original.to_json()
    first = audit.read_report(original.locator, actor="first-view")
    second = audit.read_report(original.locator, actor="second-view")
    assert first.original.to_json() == second.original.to_json() == original_wire
    assert not first.new_confirmation and not second.new_confirmation
    assert first.access.purpose == second.access.purpose == "verification"
    assert first.source_receipt.sequence == 2 and second.source_receipt.sequence == 3
    assert ReportView.from_json(first.to_json()) == first
    assert len(first.original.panel.ledger.attempts) == len(original.panel.ledger.attempts)


def test_locator_mismatch_refuses_before_any_new_source_access(tmp_path):
    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)
    original = run(audit, design, selection)
    forged = changed(original.locator, selection_fingerprint="sha256:" + "f" * 64)
    with pytest.raises(ValueError, match="locator"):
        audit.read_report(forged, actor="mismatch")
    assert len(audit.sources.history(design.scope).receipts) == 1


def test_unsupported_request_does_not_touch_store_or_call_loader(tmp_path):
    design, _selection = live_declarations()
    design = changed(design, inference=changed(design.inference, dependence="serial"))
    selection = make_selection(design, actual_selected_at=datetime.now(UTC) - timedelta(minutes=1))
    root = tmp_path / "unsupported"

    def forbidden():
        raise AssertionError("Unsupported inference must not invoke outcomes")

    qualification = ProtocolAudit(root).evaluate(design, selection, forbidden, actor="unsupported", mode="produced")
    assert isinstance(qualification, InferenceQualification)
    assert qualification.status == "unsupported"
    assert qualification.protocol is None
    assert not root.exists()


def test_source_exact_full_prefix_is_readable_and_next_attempt_refuses(tmp_path):
    design, _selection = live_declarations()
    limits = changed(design.limits, max_source_receipts=2, max_audit_receipts=2)
    audit = SourceAudit(tmp_path, limits=limits)
    first = audit.access(source_request(design))
    second = audit.access(source_request(design))
    with pytest.raises(ProtocolCapacityError, match="slots exhausted"):
        audit.access(source_request(design))
    assert audit.history(design.scope).receipts == (first, second)


def test_future_capture_refuses_before_scope_or_receipt_publication(tmp_path):
    design, _selection = live_declarations()
    request = source_request(design)
    request = changed(request, snapshot=changed(request.snapshot, captured_at=datetime.now(UTC) + timedelta(days=1)))
    root = tmp_path / "future-capture"
    with pytest.raises(ValueError, match="future"):
        SourceAudit(root).access(request)
    assert not root.exists()


def test_cached_body_failure_still_records_new_disclosure_before_read(tmp_path, monkeypatch):
    from coleman.evaluation.design import audit as audit_module

    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)
    report = run(audit, design, selection)
    body = next((tmp_path / "protocol-reports").glob("*.json"))
    body.write_text("{deliberately invalid cached JSON}", encoding="utf-8")
    actual_read = audit_module.read_contract
    observed = []

    def observed_read(path, *args):
        if path.parent.name == "protocol-reports":
            assert len(audit.sources.history(design.scope).receipts) == 2
            assert len(EvaluationAudit(tmp_path).history(design.scope.holdout_id)) == 2
            observed.append(path)
        return actual_read(path, *args)

    monkeypatch.setattr(audit_module, "read_contract", observed_read)
    assert audit.completed(report.locator)
    assert observed == []  # Metadata reconciliation must not load any results.
    with pytest.raises(ValueError):
        audit.read_report(report.locator, actor="failed-cached-read")
    assert observed == [body]
    assert len(audit.sources.history(design.scope).receipts) == 2


@pytest.fixture(scope="module")
def completed_protocol(tmp_path_factory):
    design, selection = live_declarations()
    return run(ProtocolAudit(tmp_path_factory.mktemp("pure-report-validation")), design, selection)


def test_unknown_report_versions_and_forged_aggregate_reject(completed_protocol):
    report = completed_protocol
    wire = json.loads(report.to_json())
    wire["schema_version"] = 2
    with pytest.raises(ValueError):
        ProtocolReport.from_json(json.dumps(wire))
    locator = json.loads(report.locator.to_json())
    locator["schema_version"] = 2
    with pytest.raises(ValueError):
        ReportLocator.from_json(json.dumps(locator))
    wire = json.loads(report.to_json())
    wire["family"]["inputs"][0]["record"]["measurements"][0]["value"] = 0.9
    with pytest.raises(ValueError):
        ProtocolReport.from_json(json.dumps(wire))


def test_selected_legacy_input_count_refuses_before_duplicate_record_validation(completed_protocol):
    report = completed_protocol
    excessive = report.family.model_copy(update={"inputs": report.family.inputs * 33})
    supplied = report.model_copy(update={"family": excessive})
    with pytest.raises(ProtocolCapacityError, match="inputs collection exceeds"):
        ProtocolReport.model_validate(supplied)


def test_pre_reservation_produced_outcomes_are_not_published(tmp_path):
    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)
    before = actual_panel(design, selection)
    with pytest.raises(ValueError, match="predates actual protected"):
        audit.evaluate(design, selection, lambda: before, actor="old-outputs", mode="produced")
    assert len(audit.sources.history(design.scope).receipts) == 1
    assert not tuple((tmp_path / "protocol-locators").glob("*.json"))


def test_optional_audit_cold_import_does_not_depend_on_cli_or_mcp(tmp_path):
    script = """
import importlib.abc
import sys
class RejectTransport(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith(('coleman.cli','coleman.mcp','mcp')):
   raise AssertionError('Transport imported: '+fullname)
sys.meta_path.insert(0,RejectTransport())
from coleman.evaluation.design.audit import ProtocolAudit
from coleman.evaluation.design.reports import ProtocolReport
assert callable(ProtocolAudit) and callable(ProtocolReport)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
