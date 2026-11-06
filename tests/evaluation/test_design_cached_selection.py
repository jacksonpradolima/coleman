"""Cached delivery reserves both histories before retained selection outcomes."""

from datetime import UTC, datetime, timedelta

import pytest
from design_support import changed, make_design, make_raw_panel, make_selection

from coleman.evaluation.design.audit import ProtocolAudit
from coleman.evaluation.design.reports import ProtocolReport
from coleman.evaluation.design.selection import AttemptLedger, ConfigurationTrial
from coleman.evaluation.protected.audit import EvaluationAudit


def retained_outcome_report(root):
    """Persist a real complete report whose selection ledger contains outcomes."""
    now = datetime.now(UTC)
    initial = make_design()
    design = changed(
        initial,
        comparison_plan=changed(initial.comparison_plan, test_end=now + timedelta(days=1)),
        snapshot=changed(initial.snapshot, captured_at=None),
    )
    selected_at = now - timedelta(minutes=1)
    initial_selection = make_selection(design, actual_selected_at=selected_at)
    old_times = {
        (candidate.candidate.candidate_id, member.member_id, seed): now - timedelta(minutes=2)
        for candidate in design.candidates
        for member in design.protected_members
        for seed in design.seeds
    }
    inspected = make_raw_panel(
        design,
        initial_selection,
        mode="imported",
        record_times=old_times,
        assembled_at=selected_at,
    ).rows[0]
    trial = ConfigurationTrial(
        trial_id="retained-inspected-configuration",
        candidate=design.candidate(inspected.candidate_id),
        status="inspected",
        partition="test",
        completed_at=now - timedelta(minutes=2),
        results=(inspected.evidence,),
    )
    selection = changed(
        initial_selection,
        ledger=AttemptLedger(schema_version=1, stage="development", trials=(trial,)),
    )
    audit = ProtocolAudit(root)

    def load():
        produced_at = datetime.now(UTC)
        times = {key: produced_at for key in old_times}
        return make_raw_panel(design, selection, record_times=times, assembled_at=produced_at)

    report = audit.evaluate(design, selection, load, actor="retained-outcome-author", mode="produced")
    assert isinstance(report, ProtocolReport)
    assert report.panel.selection.ledger.trials[0].results
    assert report.family.access.purpose == "verification"
    assert not report.confirmatory_eligible
    return audit, report


def test_cached_view_preserves_retained_development_outcomes_and_records_new_disclosure(tmp_path):
    audit, report = retained_outcome_report(tmp_path)
    scope = report.panel.design.scope
    before_source = len(audit.sources.history(scope).receipts)
    before_legacy = len(EvaluationAudit(tmp_path).history(scope.holdout_id))
    view = audit.read_report(report.locator, actor="retained-outcome-reader")
    assert view.original.panel.selection == report.panel.selection
    assert view.original.fingerprint == report.fingerprint
    assert view.source_receipt.sequence == before_source + 1
    assert view.access.sequence == before_legacy + 1
    assert view.access.purpose == "verification"
    assert not view.new_confirmation


def test_missing_result_bearing_selection_fails_after_both_new_reservations(tmp_path):
    audit, report = retained_outcome_report(tmp_path)
    scope = report.panel.design.scope
    before_source = len(audit.sources.history(scope).receipts)
    before_legacy = len(EvaluationAudit(tmp_path).history(scope.holdout_id))
    selection = tmp_path / "experiment-selections" / (report.panel.selection.fingerprint[7:] + ".json")
    selection.unlink()
    assert audit.completed(report.locator)
    with pytest.raises(FileNotFoundError):
        audit.read_report(report.locator, actor="failed-retained-outcome-reader")
    assert len(audit.sources.history(scope).receipts) == before_source + 1
    assert len(EvaluationAudit(tmp_path).history(scope.holdout_id)) == before_legacy + 1
