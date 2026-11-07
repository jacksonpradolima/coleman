"""Known attempt clocks cannot precede produced selection or source admission."""

from datetime import UTC, datetime, timedelta

import pytest
from design_support import changed, make_design, make_raw_panel, make_selection

from coleman.evaluation.design.audit import ProtocolAudit
from coleman.evaluation.design.reports import ProtocolReport
from coleman.evaluation.protected.audit import EvaluationAudit


def live_declarations():
    """Use actual analysis time without changing the authored historical source."""
    now = datetime.now(UTC)
    original = make_design()
    design = changed(
        original,
        snapshot=changed(original.snapshot, captured_at=None),
        comparison_plan=changed(original.comparison_plan, test_end=now + timedelta(days=1)),
    )
    return design, make_selection(design, actual_selected_at=now - timedelta(minutes=1))


def authored_panel(design, selection, *, mode="produced", ledger=None):
    """Retain constructed scalar batches, never fabricated CI jobs or paid costs."""
    now = datetime.now(UTC)
    times = {
        (candidate.candidate.candidate_id, member.member_id, seed): now
        for candidate in design.candidates
        for member in design.protected_members
        for seed in design.seeds
    }
    return make_raw_panel(design, selection, mode=mode, ledger=ledger, record_times=times, assembled_at=now)


def retimed_ledger(panel, *, started_at, completed_at):
    """Change one supplied batch declaration while retaining the complete grid."""
    first, *remaining = panel.ledger.attempts
    replacement = changed(first, started_at=started_at, completed_at=completed_at)
    return changed(panel.ledger, attempts=(replacement, *remaining))


@pytest.mark.parametrize(
    "start_seconds,end_seconds,accepted",
    [
        (None, -1, False),
        (-1, None, False),
        (None, 0, True),
        (0, None, True),
        (0, 1, True),
        (None, None, True),
    ],
    ids=[
        "unknown-start-completion-before-selection",
        "start-before-selection-unknown-completion",
        "unknown-start-completion-equals-selection",
        "start-equals-selection-unknown-completion",
        "known-start-and-later-completion",
        "both-clocks-remain-unknown",
    ],
)
def test_pure_produced_panel_checks_each_known_attempt_clock(start_seconds, end_seconds, accepted):
    """Validate every supplied clock independently and retain unspecified ones."""
    design, selection = live_declarations()
    original = authored_panel(design, selection)
    selected = selection.actual_selected_at
    start = None if start_seconds is None else selected + timedelta(seconds=start_seconds)
    end = None if end_seconds is None else selected + timedelta(seconds=end_seconds)
    ledger = retimed_ledger(original, started_at=start, completed_at=end)

    if not accepted:
        with pytest.raises(ValueError, match="predates actual selected configuration"):
            changed(original, ledger=ledger)
        return

    restored = changed(original, ledger=ledger)
    assert restored.ledger.attempts[0].started_at == start
    assert restored.ledger.attempts[0].completed_at == end
    assert tuple(row.record.to_json() for row in restored.rows) == tuple(row.record.to_json() for row in original.rows)


def test_known_completion_still_cannot_follow_panel_assembly():
    """Keep the independent upper bound even when a start time is unavailable."""
    design, selection = live_declarations()
    original = authored_panel(design, selection)
    ledger = retimed_ledger(original, started_at=None, completed_at=original.assembled_at + timedelta(microseconds=1))

    with pytest.raises(ValueError, match="assembly predates a supplied actual physical attempt time"):
        changed(original, ledger=ledger)


def test_pure_import_preserves_preselection_completion_and_original_provenance():
    """Import historical work without retiming it or changing raw provenance."""
    design, selection = live_declarations()
    produced = authored_panel(design, selection)
    historical = selection.actual_selected_at - timedelta(days=1)
    ledger = retimed_ledger(produced, started_at=None, completed_at=historical)

    imported = authored_panel(design, selection, mode="imported", ledger=ledger)

    assert imported.ledger.attempts[0].completed_at == historical
    assert imported.ledger.attempts[0].started_at is None
    assert all(row.record.provenance.evaluation_fingerprint != imported.frozen.fingerprint for row in imported.rows)
    assert type(imported).from_json(imported.to_json()).ledger.attempts[0].completed_at == historical


@pytest.mark.parametrize("known_clock", ["started_at", "completed_at"])
def test_produced_attempt_between_selection_and_actual_source_reservation_is_rejected(tmp_path, known_clock):
    """Retain both real reservations after refusing premature protected work."""
    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)
    supplied = []

    def load():
        source = audit.sources.history(design.scope).receipts[-1]
        assert len(EvaluationAudit(tmp_path).history(design.scope.holdout_id)) == 1
        assert selection.actual_selected_at < source.requested_at
        intermediate = selection.actual_selected_at + (source.requested_at - selection.actual_selected_at) / 2
        assert selection.actual_selected_at < intermediate < source.requested_at
        original = authored_panel(design, selection)
        ledger = retimed_ledger(
            original,
            started_at=intermediate if known_clock == "started_at" else None,
            completed_at=intermediate if known_clock == "completed_at" else None,
        )
        panel = changed(original, ledger=ledger)
        supplied.append(panel)
        return panel

    with pytest.raises(ValueError, match="physical work predates.*source access"):
        audit.evaluate(design, selection, load, actor="authored-clock-admission", mode="produced")

    assert len(supplied) == 1
    assert len(audit.sources.history(design.scope).receipts) == 1
    assert len(EvaluationAudit(tmp_path).history(design.scope.holdout_id)) == 1
    assert not tuple((tmp_path / "protocol-reports").glob("*.json"))
    assert not tuple((tmp_path / "protocol-locators").glob("*.json"))


@pytest.mark.parametrize("clocks", ["completion_only", "both", "unknown"])
def test_actual_source_boundary_accepts_equal_known_clocks_and_preserves_unknowns(tmp_path, clocks):
    """Accept exact inclusive reservation boundaries without inventing clocks."""
    design, selection = live_declarations()
    audit = ProtocolAudit(tmp_path)

    def load():
        reserved = audit.sources.history(design.scope).receipts[-1].requested_at
        original = authored_panel(design, selection)
        ledger = retimed_ledger(
            original,
            started_at=reserved if clocks == "both" else None,
            completed_at=None if clocks == "unknown" else reserved,
        )
        return changed(original, ledger=ledger)

    report = audit.evaluate(design, selection, load, actor="authored-exact-clock-boundary", mode="produced")

    assert isinstance(report, ProtocolReport)
    assert report.confirmatory_eligible
    assert report.family.results[0].conclusion == "insufficient_evidence"
    first = report.panel.ledger.attempts[0]
    assert first.started_at == (report.source_receipt.requested_at if clocks == "both" else None)
    assert first.completed_at == (None if clocks == "unknown" else report.source_receipt.requested_at)
    assert audit.completed(report.locator)


def test_actual_import_retains_older_completion_as_verification_without_retiming(tmp_path):
    """Persist an unchanged historical batch under actual verification access."""
    design, selection = live_declarations()
    original = authored_panel(design, selection)
    historical = selection.actual_selected_at - timedelta(days=1)
    ledger = retimed_ledger(original, started_at=None, completed_at=historical)
    imported = authored_panel(design, selection, mode="imported", ledger=ledger)
    original_wire = imported.to_json()
    audit = ProtocolAudit(tmp_path)

    report = audit.evaluate(
        design, selection, lambda: imported, actor="authored-historical-completion-import", mode="imported"
    )

    assert isinstance(report, ProtocolReport)
    assert report.panel.to_json() == original_wire
    assert report.panel.ledger.attempts[0].completed_at == historical
    assert report.panel.ledger.attempts[0].started_at is None
    assert report.source_receipt.request.purpose == "import"
    assert report.family.access.purpose == "verification"
    assert not report.confirmatory_eligible
    assert audit.completed(report.locator)
