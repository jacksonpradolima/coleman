"""Independent protected-purpose and complete-metadata admission regressions."""

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest
from design_support import changed, make_design, make_raw_panel, make_selection

from coleman.artifacts import ArtifactWriter
from coleman.evaluation.design._storage import publish_bounded
from coleman.evaluation.design._wire import wire_size
from coleman.evaluation.design.audit import ProtocolAudit, ProtocolCapacityError
from coleman.evaluation.design.contracts import InferenceQualification
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.evaluation.design.projection import project_frozen
from coleman.evaluation.design.selection import AttemptLedger, CellIdentity, ConfigurationTrial, PhysicalAttempt
from coleman.evaluation.design.source_contracts import SourceRequest
from coleman.evaluation.design.source_qualification import source_reasons
from coleman.evaluation.protected.audit import EvaluationAudit
from coleman.evaluation.protected.contracts import FrozenCandidate
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance


def live_design(**updates):
    now = datetime.now(UTC)
    original = make_design()
    return changed(
        original,
        comparison_plan=changed(original.comparison_plan, test_end=now + timedelta(days=1)),
        snapshot=changed(original.snapshot, captured_at=None),
        **updates,
    )


def request_for(design, *, purpose="inspection", partitions=("validation",), manifest=None):
    return SourceRequest(
        schema_version=1,
        scope=design.scope,
        snapshot=design.snapshot,
        purpose=purpose,
        partitions=partitions,
        partition_manifest_fingerprint=manifest,
        actor="independent-admission-test",
        selection_fingerprint="sha256:" + "1" * 64,
        evaluation_fingerprint="sha256:" + "2" * 64,
        report_fingerprint="sha256:" + "3" * 64,
    )


@pytest.mark.parametrize("purpose", ["protected_evaluation", "result_disclosure"])
@pytest.mark.parametrize("partition", ["train", "validation"])
def test_protected_purpose_cannot_masquerade_as_a_verified_narrow_partition(purpose, partition):
    design = live_design()
    with pytest.raises(ValueError, match="protected|Protected"):
        request_for(design, purpose=purpose, partitions=(partition,), manifest=design.partition_manifest.fingerprint)


@pytest.mark.parametrize("purpose", ["protected_evaluation", "result_disclosure"])
@pytest.mark.parametrize("partitions", [("test",), ("unknown",), ("test", "validation")])
def test_protected_purposes_always_remain_potential_exposure(purpose, partitions):
    design = live_design()
    request = request_for(
        design, purpose=purpose, partitions=partitions, manifest=design.partition_manifest.fingerprint
    )
    assert request.potential_test_exposure


@pytest.mark.parametrize("purpose", ["inspection", "tuning", "import"])
@pytest.mark.parametrize("partition", ["train", "validation"])
def test_real_verified_development_reads_remain_narrow_without_broadening_protected_purposes(
    tmp_path, purpose, partition
):
    design = live_design()
    selection = make_selection(design, actual_selected_at=datetime.now(UTC) - timedelta(minutes=1))
    audit = ProtocolAudit(tmp_path)
    manifest = design.partition_manifest
    audit.sources.save_manifest(manifest)
    narrow = audit.sources.access(
        request_for(design, purpose=purpose, partitions=(partition,), manifest=manifest.fingerprint)
    )
    assert narrow.partition_manifest_verified
    frozen = project_frozen(design, selection)
    protected_request = request_for(design, purpose="protected_evaluation", partitions=("test",))
    protected_request = changed(
        protected_request,
        selection_fingerprint=selection.fingerprint,
        evaluation_fingerprint=frozen.fingerprint,
        report_fingerprint=None,
    )
    current = audit.sources.access(protected_request)
    prefix = audit.sources.history(design.scope)
    assert source_reasons(prefix, current, selection, "produced", (manifest,)) == ()


def test_previously_unverified_manifest_is_not_reopened_or_retroactively_promoted(tmp_path, monkeypatch):
    design = live_design()
    audit = ProtocolAudit(tmp_path)
    manifest = design.partition_manifest
    prior = audit.sources.access(request_for(design, manifest=manifest.fingerprint))
    assert not prior.partition_manifest_verified
    audit.sources.save_manifest(manifest)

    def forbidden(_digest):
        raise AssertionError("A frozen unverified receipt must not acquire a later manifest")

    monkeypatch.setattr(audit.sources, "load_manifest", forbidden)
    assert audit._manifests(audit.sources.history(design.scope)) == ()


def test_repeated_verified_manifest_is_acquired_only_once_per_complete_prefix(tmp_path, monkeypatch):
    design = live_design()
    audit = ProtocolAudit(tmp_path)
    manifest = design.partition_manifest
    audit.sources.save_manifest(manifest)
    request = request_for(design, manifest=manifest.fingerprint)
    for _ in range(3):
        assert audit.sources.access(request).partition_manifest_verified
    actual_load = audit.sources.load_manifest
    acquired = []

    def observed(digest):
        acquired.append(digest)
        return actual_load(digest)

    monkeypatch.setattr(audit.sources, "load_manifest", observed)
    assert audit._manifests(audit.sources.history(design.scope)) == (manifest,)
    assert acquired == [manifest.fingerprint]


def large_manifest(design, revision):
    """Use bounded metadata identifiers to exercise actual cumulative byte limits."""
    manifest = design.partition_manifest
    members = tuple(
        changed(member, task=changed(member.task, id="authored-large-task-" + "x" * 5000))
        for member in manifest.members
    )
    return changed(
        manifest,
        members=members,
        snapshot=changed(manifest.snapshot, reference=changed(manifest.snapshot.reference, version=str(revision))),
    )


def test_cumulative_verified_manifests_refuse_before_third_acquisition_or_outcome_loader(tmp_path, monkeypatch):
    original = live_design()
    first = large_manifest(original, 1)
    limits = changed(original.limits, max_metadata_bytes=2 * (wire_size(first) + 1))
    design = changed(original, limits=limits)
    selection = make_selection(design, actual_selected_at=datetime.now(UTC) - timedelta(minutes=1))
    audit = ProtocolAudit(tmp_path, limits=limits)
    manifests = [large_manifest(design, revision) for revision in (1, 2, 3)]
    for manifest in manifests:
        audit.sources.save_manifest(manifest)
        request = changed(request_for(design, manifest=manifest.fingerprint), snapshot=manifest.snapshot)
        assert audit.sources.access(request).partition_manifest_verified
    actual_load = audit.sources.load_manifest
    acquired = []

    def observed(digest):
        acquired.append(digest)
        return actual_load(digest)

    def forbidden():
        raise AssertionError("Known oversized source metadata must refuse before outcomes")

    monkeypatch.setattr(audit.sources, "load_manifest", observed)
    with pytest.raises(ProtocolCapacityError, match="metadata allowance"):
        audit.evaluate(design, selection, forbidden, actor="metadata-capacity", mode="produced")
    assert acquired == [manifest.fingerprint for manifest in manifests[:2]]
    assert len(audit.sources.history(design.scope).receipts) == 4
    assert EvaluationAudit(tmp_path).history(design.scope.holdout_id) == ()
    assert not tuple((tmp_path / "protocol-locators").glob("*.json"))


def test_combined_known_design_selection_and_freeze_metadata_refuses_before_outcomes(tmp_path):
    original = live_design()
    limits = changed(original.limits, max_metadata_bytes=wire_size(original) + 256)
    design = changed(original, limits=limits)
    selection = make_selection(design, actual_selected_at=datetime.now(UTC) - timedelta(minutes=1))
    audit = ProtocolAudit(tmp_path, limits=limits)

    def forbidden():
        raise AssertionError("Known repeated metadata must fit before loading outcomes")

    with pytest.raises(ValueError, match="Graph byte budget"):
        audit.evaluate(design, selection, forbidden, actor="known-metadata", mode="produced")
    assert len(audit.sources.history(design.scope).receipts) == 1
    assert EvaluationAudit(tmp_path).history(design.scope.holdout_id) == ()
    assert not tuple((tmp_path / "protocol-locators").glob("*.json"))


def development_trial(design, selected_at, evidence):
    """Represent actual known activity without overriding its caller-supplied status."""
    trial = ConfigurationTrial(
        trial_id="declared-development-trial",
        candidate=design.candidates[0],
        status="declared",
        partition="validation",
        completed_at=selected_at if evidence == "completion" else None,
    )
    if evidence == "result":
        record = DomainRecord(
            schema_version=1,
            namespace="constructed.development-admission",
            run_id="authored-development-run",
            execution_id="authored-development-execution",
            decision_id="authored-development-unit",
            data_fingerprint=design.snapshot.data_fingerprint,
            available_at=selected_at,
            provenance=ResultProvenance(
                producer="authored-development-fixture",
                producer_version="1",
                spec_fingerprint=trial.candidate.candidate.configuration_fingerprint,
                evaluation_fingerprint="sha256:" + "e" * 64,
            ),
            measurements=(
                Measurement(
                    name="quality",
                    value=0.5,
                    unit="1",
                    meaning="authored-ci-quality-v1",
                    direction="maximize",
                    status="observed",
                ),
            ),
        )
        trial = changed(trial, results=(OriginalRecordEvidence.from_record(record),))
    attempts = ()
    if evidence == "attempt":
        attempts = (
            PhysicalAttempt(
                attempt_id="authored-attempted-development",
                group_id="authored-development-group",
                trial_id=trial.trial_id,
                candidate_id=trial.candidate.candidate.candidate_id,
                state="failed",
                started_at=None,
                completed_at=None,
            ),
        )
    return AttemptLedger(schema_version=1, stage="development", trials=(trial,), attempts=attempts)


@pytest.mark.parametrize("evidence", ["result", "completion", "attempt", "empty"])
def test_declared_status_cannot_hide_actual_outcome_completion_or_physical_attempt(tmp_path, evidence):
    design = live_design()
    selected_at = datetime.now(UTC) - timedelta(minutes=1)
    ledger = development_trial(design, selected_at, evidence)
    selection = make_selection(design, actual_selected_at=selected_at, ledger=ledger)
    audit = ProtocolAudit(tmp_path)
    frozen = project_frozen(design, selection)
    request = changed(
        request_for(design, purpose="protected_evaluation", partitions=("test",)),
        selection_fingerprint=selection.fingerprint,
        evaluation_fingerprint=frozen.fingerprint,
        report_fingerprint=None,
    )
    receipt = audit.sources.access(request)
    reasons = source_reasons(audit.sources.history(design.scope), receipt, selection, "produced", ())
    assert ("development_access_history_unverified" in reasons) == (evidence != "empty")
    assert selection.ledger.trials[0].status == "declared"


def test_development_physical_attempt_cannot_be_attributed_to_another_candidates_trial():
    design = live_design()
    ledger = development_trial(design, datetime.now(UTC) - timedelta(minutes=1), "attempt")
    attempt = ledger.attempts[0]
    assert attempt.candidate_id == "baseline"
    with pytest.raises(ValueError, match="candidate differs from its retained trial"):
        changed(ledger, attempts=(changed(attempt, candidate_id="candidate"),))


def cost_record(design, available_at, execution):
    """Author separate identified evidence; no real processor measurement is claimed."""
    return DomainRecord(
        schema_version=1,
        namespace="constructed.attribution-cost",
        run_id="authored-accounting-fixture",
        execution_id=execution,
        decision_id="whole-process",
        data_fingerprint=design.snapshot.data_fingerprint,
        available_at=available_at,
        provenance=ResultProvenance(
            producer="authored-attribution-fixture",
            producer_version="1",
            spec_fingerprint=design.candidates[0].candidate.configuration_fingerprint,
            evaluation_fingerprint="sha256:" + "e" * 64,
        ),
        measurements=(
            Measurement(
                name="authored_cpu_seconds",
                value=0.125,
                unit="s",
                meaning="authored-whole-process-time-v1",
                direction="minimize",
                status="observed",
            ),
        ),
    )


@pytest.mark.parametrize("overlap", ["attempt", "cost"])
def test_imported_panel_cannot_attribute_one_physical_attempt_or_cost_to_both_stages(overlap):
    design = live_design()
    selected_at = datetime.now(UTC) - timedelta(minutes=1)
    development = development_trial(design, selected_at, "attempt")
    original = cost_record(design, selected_at, "authored-one-meter")
    before = changed(development.attempts[0], costs=(OriginalRecordEvidence.from_record(original),))
    development = changed(development, attempts=(before,))
    selection = make_selection(design, actual_selected_at=selected_at, ledger=development)
    after = changed(
        before,
        attempt_id=before.attempt_id if overlap == "attempt" else "authored-protected-invocation",
        trial_id=None,
        costs=before.costs if overlap == "cost" else (),
    )
    protected = AttemptLedger(schema_version=1, stage="protected", attempts=(after,))
    with pytest.raises(ValueError, match="both development and protected work"):
        make_raw_panel(design, selection, ledger=protected, mode="imported", assembled_at=datetime.now(UTC))


def test_distinct_attempts_and_identified_equal_costs_can_share_a_consistent_work_group():
    design = live_design()
    selected_at = datetime.now(UTC) - timedelta(minutes=1)
    development = development_trial(design, selected_at, "attempt")
    first = cost_record(design, selected_at, "authored-development-meter")
    second = cost_record(design, selected_at, "authored-protected-meter")
    before = changed(development.attempts[0], costs=(OriginalRecordEvidence.from_record(first),))
    development = changed(development, attempts=(before,))
    selection = make_selection(design, actual_selected_at=selected_at, ledger=development)
    after = changed(
        before,
        attempt_id="authored-distinct-protected-invocation",
        trial_id=None,
        costs=(OriginalRecordEvidence.from_record(second),),
    )
    protected = AttemptLedger(schema_version=1, stage="protected", attempts=(after,))
    panel = make_raw_panel(design, selection, ledger=protected, mode="imported", assembled_at=datetime.now(UTC))
    assert before.group_id == after.group_id
    assert before.attempt_id != after.attempt_id
    assert first.logical_key != second.logical_key
    assert first.measurements == second.measurements
    assert panel.selection.ledger.attempts[0].costs[0].record == first
    assert panel.ledger.attempts[0].costs[0].record == second
    assert all(row.record.provenance.evaluation_fingerprint == "sha256:" + "e" * 64 for row in panel.rows)


def test_existing_publication_symlink_refuses_even_when_external_bytes_match(tmp_path):
    target = tmp_path / "unowned-original.json"
    content = '{"schema_version":1}\n'
    target.write_text(content, encoding="utf-8")
    root = tmp_path / "owned-artifacts"
    root.mkdir()
    destination = root / "scope-header.json"
    destination.symlink_to(target)
    with pytest.raises(OSError):
        publish_bounded(ArtifactWriter(root), destination, content, 128)
    assert destination.is_symlink()
    assert target.read_text(encoding="utf-8") == content


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Named pipes require a POSIX filesystem")
def test_real_source_publication_refuses_existing_fifo_without_waiting_for_a_writer(tmp_path):
    design = live_design()
    request = request_for(design, partitions=("test",))
    root = tmp_path / "source-store"
    header = root / "source-scopes" / (design.scope.logical_key[7:] + ".json")
    header.parent.mkdir(parents=True)
    os.mkfifo(header)
    request_file = tmp_path / "source-request.json"
    request_file.write_text(request.to_json(), encoding="utf-8")
    script = r"""
import sys
from pathlib import Path
from coleman.evaluation.design.audit import SourceAudit
from coleman.evaluation.design.source_contracts import SourceRequest
root,request_file=map(Path,sys.argv[1:])
request=SourceRequest.from_json(request_file.read_bytes())
try:
 SourceAudit(root).access(request)
except OSError:
 pass
else:
 raise AssertionError('Nonregular source publication destination was accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(root), str(request_file)],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not tuple((root / "source-access").glob("*/*.json"))


def test_nonfinalist_configuration_obeys_declared_cap_without_dropping_or_truncating_trial(tmp_path):
    design = live_design()
    selected_at = datetime.now(UTC) - timedelta(minutes=1)
    base = FrozenCandidate.from_configuration(
        "authored-nonfinalist", {"method": "authored-trial", "payload": "x" * 5000}
    )
    candidate = changed(design.candidate("candidate"), candidate=base)
    trial = ConfigurationTrial(
        trial_id="retained-failed-nonfinalist",
        candidate=candidate,
        status="failed",
        partition="validation",
        completed_at=selected_at,
    )
    ledger = AttemptLedger(schema_version=1, stage="development", trials=(trial,))
    selection = make_selection(design, actual_selected_at=selected_at, ledger=ledger)
    before = selection.to_json()
    assert len(base.configuration_json.encode("utf-8")) > design.limits.max_configuration_bytes

    def forbidden():
        raise AssertionError("An oversized nonfinalist must refuse before protected outcomes")

    root = tmp_path / "oversized-trial"
    with pytest.raises(ValueError, match="trial configuration exceeds"):
        ProtocolAudit(root).evaluate(design, selection, forbidden, actor="nonfinalist-capacity", mode="produced")
    assert not root.exists()
    assert selection.to_json() == before
    assert selection.ledger.trials[0].candidate.candidate.configuration_json == base.configuration_json


@pytest.mark.parametrize("supported", [True, False])
def test_expired_execution_deadline_never_loads_outcomes_or_creates_a_store(tmp_path, supported):
    original = live_design()
    design = changed(
        original,
        comparison_plan=changed(original.comparison_plan, test_end=datetime.now(UTC) - timedelta(seconds=1)),
    )
    if not supported:
        design = changed(design, inference=changed(design.inference, dependence="serial"))
    selection = make_selection(design, actual_selected_at=datetime.now(UTC) - timedelta(minutes=1))

    def forbidden():
        raise AssertionError("An expired or unsupported request must not load outcomes")

    root = tmp_path / "expired-protocol"
    if supported:
        with pytest.raises(ValueError):
            ProtocolAudit(root).evaluate(design, selection, forbidden, actor="expired-supported", mode="produced")
    else:
        result = ProtocolAudit(root).evaluate(
            design, selection, forbidden, actor="expired-unsupported", mode="produced"
        )
        assert isinstance(result, InferenceQualification)
        assert result.status == "unsupported"
    assert not root.exists()


def test_completed_historical_report_can_still_be_saved_and_disclosed_after_execution_deadline(tmp_path, monkeypatch):
    from coleman.evaluation.design import audit as audit_module
    from coleman.evaluation.design.reports import ProtocolReport

    design = live_design()
    selection = make_selection(design, actual_selected_at=datetime.now(UTC) - timedelta(minutes=1))
    audit = ProtocolAudit(tmp_path)

    def actual_callback():
        now = datetime.now(UTC)
        times = {
            (item.candidate.candidate_id, member.member_id, seed): now
            for item in design.candidates
            for member in design.protected_members
            for seed in design.seeds
        }
        return make_raw_panel(design, selection, record_times=times, assembled_at=now)

    report = audit.evaluate(design, selection, actual_callback, actor="historical-protocol-author", mode="produced")
    assert isinstance(report, ProtocolReport)
    later = design.comparison_plan.test_end + timedelta(days=1)

    class LaterClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return later if tz is None else later.astimezone(tz)

    monkeypatch.setattr(audit_module, "datetime", LaterClock)
    target = audit.save_report(report)
    assert target.exists()
    view = audit.read_report(report.locator, actor="historical-protocol-reader")
    assert view.viewed_at > design.comparison_plan.test_end
    assert view.original.fingerprint == report.fingerprint
    assert not view.new_confirmation


@pytest.mark.parametrize("excess", ["attempts", "outputs"])
def test_complete_experiment_budget_accepts_exact_total_and_rejects_global_plus_one(excess):
    original = live_design()
    design = changed(
        original,
        seeds=(7, 11),
        budget=changed(original.budget, maximum_physical_attempts=2, maximum_output_references=3),
    )
    selected_at = datetime.now(UTC) - timedelta(minutes=1)
    development = development_trial(design, selected_at, "attempt")
    before = changed(
        development.attempts[0],
        outputs=(CellIdentity(candidate_id="baseline", member_id="authored-development-member", seed=7),),
    )
    development = changed(development, attempts=(before,))
    selection = make_selection(design, actual_selected_at=selected_at, ledger=development)
    after = changed(
        before,
        attempt_id="authored-protected-budget-group",
        trial_id=None,
        outputs=tuple(
            CellIdentity(candidate_id="baseline", member_id=member, seed=7) for member in ("build-a", "build-b")
        ),
    )
    protected = AttemptLedger(schema_version=1, stage="protected", attempts=(after,))
    exact = make_raw_panel(design, selection, ledger=protected, mode="imported", assembled_at=datetime.now(UTC))
    assert len(exact.selection.ledger.attempts) + len(exact.ledger.attempts) == 2
    assert exact.selection.ledger.output_references + exact.ledger.output_references == 3
    if excess == "attempts":
        plus_one = changed(after, attempt_id="authored-extra-protected-retry", outputs=())
        excessive = changed(protected, attempts=(after, plus_one))
    else:
        output = CellIdentity(candidate_id="baseline", member_id="build-a", seed=11)
        excessive = changed(protected, attempts=(changed(after, outputs=(*after.outputs, output)),))
    with pytest.raises(ValueError, match="Complete development/protected"):
        make_raw_panel(design, selection, ledger=excessive, mode="imported", assembled_at=datetime.now(UTC))
