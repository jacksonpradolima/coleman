"""Complete original panels, truthful independent means and grouped-work evidence."""

import json
from datetime import timedelta

import pytest
from design_support import (
    ASSEMBLED,
    AVAILABLE,
    changed,
    make_design,
    make_finance_panel,
    make_raw_panel,
    make_selection,
)

from coleman.core.primitives import canonical_json
from coleman.evaluation._repetition import repetition_execution_id
from coleman.evaluation.design.aggregates import require_projection_capacity
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.evaluation.design.panel import RawPanel
from coleman.evaluation.design.selection import AttemptLedger, CellIdentity
from coleman.evaluation.protected.audit import EvaluationAudit
from coleman.evaluation.protected.comparison import evaluate
from coleman.evaluation.protected.contracts import FrozenCandidate
from coleman.results.domain import Measurement


def replace_record(cell, **updates):
    return changed(cell, evidence=OriginalRecordEvidence.from_record(changed(cell.record, **updates)))


def replace_rows(panel, first):
    return changed(panel, rows=(first, *panel.rows[1:]))


def test_complete_panel_round_trip_preserves_original_bytes_keys_digests_and_optional_reasons():
    reason_key = "baseline", "build-a", 7
    null_key = "candidate", "build-b", 7
    panel = make_raw_panel(reasons={reason_key: "declared-source-result", null_key: None})
    restored = RawPanel.from_json(panel.to_json())
    assert restored.to_json() == panel.to_json()
    assert restored.fingerprint == panel.fingerprint
    for before, after in zip(panel.rows, restored.rows, strict=True):
        assert after.record.to_json() == before.record.to_json()
        assert after.record_digest == before.record_digest
        assert after.record.logical_key == before.record.logical_key
        assert ("reason" in after.model_fields_set) == ("reason" in before.model_fields_set)
    assert (
        panel.rows[0].record.provenance.spec_fingerprint != panel.frozen.candidate("baseline").configuration_fingerprint
    )
    for derived in panel.paired_observations():
        assert (
            derived.record.provenance.spec_fingerprint
            == panel.frozen.candidate(derived.candidate_id).configuration_fingerprint
        )
        assert derived.record.provenance.source_revision == panel.fingerprint
        assert derived.record.available_at == panel.assembled_at


def test_set_order_changes_no_panel_identity_or_aggregate_identity():
    panel = make_raw_panel()
    reversed_panel = changed(panel, rows=tuple(reversed(panel.rows)))
    assert panel.to_json() == reversed_panel.to_json()
    assert [row.record.digest for row in panel.paired_observations()] == [
        row.record.digest for row in reversed_panel.paired_observations()
    ]


@pytest.mark.parametrize("version", [None, True, "1", 0, 2, 999])
def test_panel_reader_never_guesses_missing_old_coerced_or_future_versions(version):
    data = json.loads(make_raw_panel().to_json())
    if version is None:
        data.pop("schema_version")
    else:
        data["schema_version"] = version
    with pytest.raises(ValueError):
        RawPanel.from_json(canonical_json(data))


def unequal_member_design():
    design = make_design()
    member = design.members[0]
    second = changed(
        member,
        member_id="build-a2",
        metadata=changed(
            member.metadata, observation_id="source-build-a2", event_at=member.metadata.event_at + timedelta(minutes=1)
        ),
    )
    return changed(design, members=(*design.members, second), seeds=(7, 11))


def test_equal_seed_then_member_then_unit_weighting_uses_authored_exact_arithmetic(tmp_path):
    design = unequal_member_design()
    values = {}
    for candidate in design.candidates:
        name = candidate.candidate.candidate_id
        for member in design.protected_members:
            for seed in design.seeds:
                values[name, member.member_id, seed] = 0.25
    values.update({("candidate", "build-a", 7): 0.0, ("candidate", "build-a", 11): 1.0})
    values.update({("candidate", "build-a2", 7): 1.0, ("candidate", "build-a2", 11): 1.0})
    values.update({("candidate", "build-b", 7): 0.0, ("candidate", "build-b", 11): 0.0})
    panel = make_raw_panel(design, values=values)
    means = {
        (row.candidate_id, row.metadata.group_id): row.record.measurement("quality").value
        for row in panel.paired_observations()
    }
    assert means == {
        ("baseline", "family-a"): 0.25,
        ("baseline", "family-b"): 0.25,
        ("candidate", "family-a"): 0.75,
        ("candidate", "family-b"): 0.0,
    }
    report = evaluate(panel.frozen, EvaluationAudit(tmp_path), panel.paired_observations, actor="authored-fixture")
    result = report.results[0]
    assert result.observed_pairs == 2
    assert result.raw_mean_difference == 0.125
    assert result.improvement == 0.125
    assert result.conclusion == "insufficient_evidence"
    assert result.ci_lower is None
    assert panel.design.raw_cells == 12
    assert len(panel.ledger.attempts) == 2


@pytest.mark.parametrize("status", ["unavailable", "unknown", "not_applicable"])
def test_original_missingness_is_retained_and_never_completed_by_dropping_a_seed(status, tmp_path):
    key = "candidate", "build-a", 7
    panel = make_raw_panel(values={key: None}, statuses={key: status})
    original = next(row for row in panel.rows if row.candidate_id == "candidate" and row.member_id == "build-a")
    assert original.record.measurement("quality").status == status
    aggregate = next(
        row
        for row in panel.paired_observations()
        if row.candidate_id == "candidate" and row.metadata.group_id == "family-a"
    )
    assert aggregate.record.measurement("quality").status == "unavailable"
    report = evaluate(panel.frozen, EvaluationAudit(tmp_path), panel.paired_observations, actor="authored-fixture")
    assert report.results[0].conclusion == "insufficient_evidence"
    assert report.results[0].observed_pairs == 1
    assert report.results[0].raw_mean_difference is None


def test_zero_is_observed_and_failed_or_skipped_cells_need_explicit_unavailability():
    key = "candidate", "build-a", 7
    panel = make_raw_panel(values={key: 0.0})
    assert (
        next(
            row
            for row in panel.paired_observations()
            if row.candidate_id == "candidate" and row.metadata.group_id == "family-a"
        )
        .record.measurement("quality")
        .value
        == 0.0
    )
    with pytest.raises(ValueError, match="Incomplete cell"):
        make_raw_panel(states={key: "failed"})
    failed = make_raw_panel(values={key: None}, states={key: "failed"}, reasons={key: "producer-timeout"})
    assert (
        next(row for row in failed.rows if row.candidate_id == "candidate" and row.member_id == "build-a").reason
        == "producer-timeout"
    )
    skipped = make_raw_panel(values={key: None}, states={key: "skipped"})
    outputs = {
        (out.candidate_id, out.member_id, out.seed) for attempt in skipped.ledger.attempts for out in attempt.outputs
    }
    assert key not in outputs


@pytest.mark.parametrize("edit", ["duplicate", "missing", "extra"])
def test_complete_grid_never_silently_drops_duplicates_missing_or_extra_cells(edit):
    panel = make_raw_panel()
    if edit == "duplicate":
        rows = (*panel.rows, panel.rows[0])
    elif edit == "missing":
        rows = panel.rows[1:]
    else:
        rows = (
            *panel.rows,
            changed(
                panel.rows[0],
                seed=11,
                evidence=OriginalRecordEvidence.from_record(
                    changed(panel.rows[0].record, execution_id=repetition_execution_id(panel.rows[0].candidate_id, 11))
                ),
            ),
        )
    with pytest.raises(ValueError):
        changed(panel, rows=rows)


def test_produced_requires_projected_evaluation_but_import_preserves_old_evidence_without_relabeling():
    produced = make_raw_panel()
    imported = make_raw_panel(mode="imported")
    assert not imported.ledger.attempts
    old = "sha256:" + "e" * 64
    assert all(row.record.provenance.evaluation_fingerprint == old for row in imported.rows)
    restored = RawPanel.from_json(imported.to_json())
    assert all(row.record.provenance.evaluation_fingerprint == old for row in restored.rows)
    first = produced.rows[0]
    provenance = changed(first.record.provenance, evaluation_fingerprint=old)
    with pytest.raises(ValueError, match="projected evaluation"):
        replace_rows(produced, replace_record(first, provenance=provenance))
    with pytest.raises(ValueError, match="BASE candidate"):
        replace_rows(
            produced,
            replace_record(
                first,
                provenance=changed(
                    first.record.provenance,
                    spec_fingerprint=produced.frozen.candidate(first.candidate_id).configuration_fingerprint,
                ),
            ),
        )
    with pytest.raises(ValueError, match="source snapshot"):
        replace_rows(produced, replace_record(first, data_fingerprint="sha256:" + "d" * 64))


def test_import_accepts_known_historical_output_times_but_production_must_follow_selection():
    historical = AVAILABLE.replace(hour=11)
    times = {
        (candidate, member, 7): historical
        for candidate in ("baseline", "candidate")
        for member in ("build-a", "build-b")
    }
    imported = make_raw_panel(mode="imported", record_times=times)
    assert all(row.record.available_at == historical for row in imported.rows)
    with pytest.raises(ValueError, match="predates its actual selected"):
        make_raw_panel(record_times=times)


def test_actual_completion_clocks_are_retained_before_common_truthful_assembly():
    early = AVAILABLE
    late = AVAILABLE + timedelta(seconds=30)
    times = {("candidate", "build-a", 7): late}
    panel = make_raw_panel(record_times=times)
    assert {row.record.available_at for row in panel.rows} == {early, late}
    assert {row.record.available_at for row in panel.paired_observations()} == {ASSEMBLED}
    with pytest.raises(ValueError, match="precede assembly"):
        make_raw_panel(record_times={("candidate", "build-a", 7): ASSEMBLED + timedelta(seconds=1)})
    with pytest.raises(ValueError, match="analysis deadline"):
        make_raw_panel(assembled_at=ASSEMBLED.replace(day=8))


def test_too_early_source_availability_and_snapshot_capture_cannot_be_relabelled():
    panel = make_raw_panel()
    first = panel.rows[0]
    with pytest.raises(ValueError, match="original source labels"):
        replace_rows(panel, replace_record(first, available_at=panel.design.members[0].metadata.event_at))
    snapshot = changed(panel.design.snapshot, captured_at=ASSEMBLED + timedelta(seconds=1))
    design = changed(panel.design, snapshot=snapshot)
    with pytest.raises(ValueError, match="source capture"):
        make_raw_panel(design)


def test_physical_groups_can_emit_many_seeds_but_seeds_never_multiply_units(tmp_path):
    design = make_design()
    limits = changed(design.limits, max_seeds=20)
    design = changed(design, seeds=tuple(range(20)), limits=limits)
    panel = make_raw_panel(design)
    assert len(panel.rows) == 80
    assert len(panel.ledger.attempts) == 2
    assert all(len(attempt.outputs) == 40 for attempt in panel.ledger.attempts)
    assert len(panel.paired_observations()) == 4
    report = evaluate(panel.frozen, EvaluationAudit(tmp_path), panel.paired_observations, actor="authored-fixture")
    assert report.results[0].observed_pairs == 2
    assert report.results[0].conclusion == "insufficient_evidence"


def test_attempt_groups_retries_and_known_times_are_validated_without_inventing_missing_work():
    panel = make_raw_panel()
    attempt = panel.ledger.attempts[0]
    failed = changed(attempt, attempt_id="earlier-failed-attempt", state="failed", outputs=())
    ledger = changed(panel.ledger, attempts=(*panel.ledger.attempts, failed))
    restored = changed(panel, ledger=ledger)
    assert len(restored.ledger.attempts) == 3
    assert len(restored.paired_observations()) == 4
    with pytest.raises(ValueError, match="physical-group references"):
        changed(panel, ledger=AttemptLedger(schema_version=1, stage="protected"))
    with pytest.raises(ValueError, match="separate protected"):
        changed(panel, ledger=AttemptLedger(schema_version=1, stage="development"))
    after = changed(attempt, completed_at=ASSEMBLED + timedelta(seconds=1))
    with pytest.raises(ValueError, match="predates a supplied actual"):
        changed(panel, ledger=changed(panel.ledger, attempts=(after,)))
    before = changed(attempt, started_at=panel.selection.actual_selected_at - timedelta(seconds=1))
    with pytest.raises(ValueError, match="predates actual selected"):
        changed(panel, ledger=changed(panel.ledger, attempts=(before,)))
    alien = changed(attempt, outputs=(CellIdentity(candidate_id=attempt.candidate_id, member_id="alien", seed=7),))
    with pytest.raises(ValueError, match="undeclared logical output"):
        changed(panel, ledger=changed(panel.ledger, attempts=(alien,)))


def test_failed_group_retains_separate_cost_without_averaging_or_accepting_future_evidence():
    panel = make_raw_panel()
    cost = changed(
        panel.rows[0].record,
        namespace="constructed.physical-cost-fixture",
        execution_id="authored-failed-invocation",
        decision_id="whole-process-meter",
        measurements=(
            Measurement(
                name="authored_cpu_seconds",
                value=0.125,
                unit="s",
                meaning="authored-whole-process-processor-time-v1",
                direction="minimize",
                status="observed",
            ),
        ),
    )
    attempt = changed(
        panel.ledger.attempts[0],
        attempt_id="authored-failed-with-cost",
        state="failed",
        outputs=(),
        costs=(OriginalRecordEvidence.from_record(cost),),
    )
    ledger = changed(panel.ledger, attempts=(*panel.ledger.attempts, attempt))
    retained = changed(panel, ledger=ledger)
    assert (
        next(item for item in retained.ledger.attempts if item.state == "failed").costs[0].record.to_json()
        == cost.to_json()
    )
    assert [row.record.measurement("quality").value for row in retained.paired_observations()] == [
        row.record.measurement("quality").value for row in panel.paired_observations()
    ]
    for timestamp, message in (
        (ASSEMBLED + timedelta(seconds=1), "predates a supplied actual"),
        (AVAILABLE.replace(hour=11), "predates actual selected"),
    ):
        changed_cost = changed(cost, available_at=timestamp)
        changed_attempt = changed(attempt, costs=(OriginalRecordEvidence.from_record(changed_cost),))
        with pytest.raises(ValueError, match=message):
            changed(panel, ledger=changed(ledger, attempts=(*panel.ledger.attempts, changed_attempt)))


@pytest.mark.parametrize("field", ["max_aggregate_record_bytes", "max_input_header_bytes"])
def test_fixed_output_capacity_is_rejected_purely_before_any_outcome_loader(field):
    design = make_design()
    design = changed(design, limits=changed(design.limits, **{field: 1}))
    with pytest.raises(ValueError, match="capacity"):
        require_projection_capacity(design, make_selection(design))


def test_default_output_capacity_preflight_accepts_full_current_identifiers():
    design = make_design()
    assert require_projection_capacity(design, make_selection(design)) is None


def test_constructed_financial_surrogate_gain_does_not_replace_negative_net_objective(tmp_path):
    net = make_finance_panel()
    loss = make_finance_panel(primary="normalized_loss")
    net_report = evaluate(
        net.frozen, EvaluationAudit(tmp_path / "net"), net.paired_observations, actor="authored-worlds"
    )
    loss_report = evaluate(
        loss.frozen, EvaluationAudit(tmp_path / "loss"), loss.paired_observations, actor="authored-worlds"
    )
    assert net_report.results[0].raw_mean_difference == -0.125
    assert net_report.results[0].improvement == -0.125
    assert loss_report.results[0].raw_mean_difference == -0.0625
    assert loss_report.results[0].improvement == 0.0625
    assert net_report.results[0].observed_pairs == loss_report.results[0].observed_pairs == 4
    assert net_report.results[0].conclusion == loss_report.results[0].conclusion == "insufficient_evidence"
    assert net.frozen.protocol.unit == "USD"
    assert loss.frozen.protocol.unit == "1"
    assert net.frozen.protocol.holdout_id == loss.frozen.protocol.holdout_id
    assert all(not attempt.costs for attempt in net.ledger.attempts)
    assert all(row.record.measurement("modelled_charges").value in {0.125, 0.5} for row in net.rows)
    assert any(row.record.measurement("net_value").value < 0 for row in net.rows)


def test_noisy_per_unit_best_policy_is_retrospective_diagnostic_not_a_deployable_baseline(tmp_path):
    design = make_design()
    oracle = changed(
        design.candidate("candidate"),
        candidate=FrozenCandidate.from_configuration("retrospective-vbs", {"method": "per-unit-outcome-maximum"}),
        role="retrospective_oracle",
    )
    design = changed(design, candidates=(*design.candidates, oracle))
    values = {
        ("baseline", "build-a", 7): 0.0,
        ("baseline", "build-b", 7): 1.0,
        ("candidate", "build-a", 7): 1.0,
        ("candidate", "build-b", 7): 0.0,
        ("retrospective-vbs", "build-a", 7): 1.0,
        ("retrospective-vbs", "build-b", 7): 1.0,
    }
    panel = make_raw_panel(design, values=values)
    report = evaluate(panel.frozen, EvaluationAudit(tmp_path), panel.paired_observations, actor="authored-noisy-worlds")
    assert len(report.inputs) == 6
    assert len(report.results) == 1
    assert report.results[0].raw_mean_difference == 0.0
    assert report.results[0].observed_pairs == 2
    assert report.results[0].conclusion == "insufficient_evidence"
    assert [
        row.record.measurement("quality").value
        for row in panel.paired_observations()
        if row.candidate_id == "retrospective-vbs"
    ] == [1.0, 1.0]


@pytest.mark.parametrize("field", ["max_raw_record_bytes", "max_raw_header_bytes"])
def test_original_record_and_actual_header_wire_limits_are_enforced(field):
    design = make_design()
    design = changed(design, limits=changed(design.limits, **{field: 1}))
    with pytest.raises(ValueError, match="capacity"):
        make_raw_panel(design)
