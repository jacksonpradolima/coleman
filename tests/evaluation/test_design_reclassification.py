"""Prior narrow reads cannot be laundered through a different protected manifest."""

from datetime import UTC, datetime, timedelta

import pytest
from design_support import changed, make_design, make_raw_panel, make_selection

from coleman.evaluation.design.audit import ProtocolAudit
from coleman.evaluation.design.reports import ProtocolReport
from coleman.evaluation.design.selection import AttemptLedger, CellIdentity, ConfigurationTrial, PhysicalAttempt
from coleman.evaluation.design.source_contracts import SourcePrefix, SourceReceipt, SourceRequest
from coleman.evaluation.protected.audit import EvaluationAudit


def current_design():
    """Declare disjoint authored training and protected CI rows with actual analysis time."""
    original = make_design()
    training = changed(
        original.members[0],
        member_id="authored-training-build",
        path=changed(original.members[0].path, id="authored-training-path"),
        unit_id=None,
        metadata=changed(
            original.members[0].metadata,
            observation_id="authored-training-observation",
            group_id="authored-training-group",
            event_at=datetime(2017, 1, 2, tzinfo=UTC),
            available_at=datetime(2017, 1, 2, 1, tzinfo=UTC),
        ),
    )
    return changed(
        original,
        members=(*original.members, training),
        snapshot=changed(original.snapshot, captured_at=None),
        comparison_plan=changed(original.comparison_plan, test_end=datetime.now(UTC) + timedelta(days=1)),
    )


def prior_manifest(design, alteration):
    """Keep the stable scope while varying one actual source declaration."""
    manifest = design.partition_manifest
    if alteration == "same":
        return manifest
    if alteration == "cutoffs":
        return changed(
            manifest,
            source_plan=changed(
                manifest.source_plan,
                train_end=datetime(2019, 2, 1, tzinfo=UTC),
                validation_end=datetime(2019, 2, 2, tzinfo=UTC),
            ),
        )
    if alteration == "snapshot_revision":
        return changed(
            manifest,
            snapshot=changed(
                manifest.snapshot,
                reference=changed(manifest.snapshot.reference, version="earlier-authored-revision"),
            ),
        )
    if alteration == "membership":
        return changed(manifest, members=tuple(member for member in manifest.members if member.member_id != "build-b"))
    old = next(member for member in manifest.members if member.member_id == "build-a")
    if alteration == "member_identity":
        replacement = changed(old, member_id="earlier-build-a-label")
    elif alteration == "observation_identity":
        replacement = changed(old, metadata=changed(old.metadata, observation_id="earlier-observation-a"))
    elif alteration == "group_identity":
        replacement = changed(old, metadata=changed(old.metadata, group_id="earlier-group-a"))
    else:
        raise ValueError("Unknown authored reclassification fixture")
    return changed(
        manifest,
        members=tuple(replacement if member.member_id == old.member_id else member for member in manifest.members),
    )


def reserve_narrow(audit, manifest):
    """Actually persist and reserve a train-only inspection before selection."""
    audit.sources.save_manifest(manifest)
    request = SourceRequest(
        schema_version=1,
        scope=manifest.scope,
        snapshot=manifest.snapshot,
        purpose="inspection",
        partitions=("train",),
        partition_manifest_fingerprint=manifest.fingerprint,
        actor="authored-train-inspection",
        selection_fingerprint=None,
        evaluation_fingerprint=None,
        report_fingerprint=None,
    )
    receipt = audit.sources.access(request)
    assert receipt.partition_manifest_verified
    return receipt


def selection_after(design, prior=None, *, inspected_trial=False):
    """Record actual selection after the earlier durable source reservation."""
    selected_at = datetime.now(UTC)
    assert prior is None or prior.requested_at <= selected_at
    selection = make_selection(design, actual_selected_at=selected_at)
    if inspected_trial:
        assert prior is not None
        trial = ConfigurationTrial(
            trial_id="authored-inspected-development-trial",
            candidate=design.candidate("candidate"),
            status="inspected",
            partition="train",
            completed_at=selected_at,
            source_accesses=(prior.fingerprint,),
        )
        selection = changed(selection, ledger=AttemptLedger(schema_version=1, stage="development", trials=(trial,)))
    return selection


def computed_panel(design, selection):
    """Compute authored scalar batches in process; no physical CI job or fee is claimed."""
    values, attempts = {}, []
    for candidate in design.candidates:
        started_at = datetime.now(UTC)
        name = candidate.candidate.candidate_id
        outputs = []
        for member in design.protected_members:
            for seed in design.seeds:
                values[name, member.member_id, seed] = 0.25 if name == "baseline" else 0.75
                outputs.append(CellIdentity(candidate_id=name, member_id=member.member_id, seed=seed))
        attempts.append(
            PhysicalAttempt(
                attempt_id="authored-in-process-calculation-" + name,
                group_id="authored-in-process-batch-" + name,
                trial_id=None,
                candidate_id=name,
                state="completed",
                started_at=started_at,
                completed_at=datetime.now(UTC),
                outputs=tuple(outputs),
            )
        )
    available_at = datetime.now(UTC)
    panel = make_raw_panel(
        design,
        selection,
        values=values,
        ledger=AttemptLedger(schema_version=1, stage="protected", attempts=tuple(attempts)),
        record_times={key: available_at for key in values},
        assembled_at=available_at,
    )
    return changed(panel, assembled_at=datetime.now(UTC))


def evaluate(audit, design, selection):
    """Use actual source reservation, bounded evaluator and immutable report storage."""
    report = audit.evaluate(
        design,
        selection,
        lambda: computed_panel(design, selection),
        actor="authored-current-protected-evaluation",
        mode="produced",
    )
    assert isinstance(report, ProtocolReport)
    return report


def assert_prior_unchanged(audit, prior, original_bytes):
    path = audit.root / "source-access" / prior.request.scope.logical_key[7:] / "000000000001.json"
    assert path.read_bytes() == original_bytes
    assert audit.sources.history(prior.request.scope).receipts[0] == prior
    assert prior.partition_manifest_verified


@pytest.mark.parametrize(
    "alteration",
    ["cutoffs", "member_identity", "observation_identity", "group_identity", "snapshot_revision", "membership"],
)
def test_changed_complete_manifest_cannot_reuse_a_prior_verified_narrow_claim(tmp_path, alteration):
    design = current_design()
    previous = prior_manifest(design, alteration)
    assert previous.fingerprint != design.partition_manifest.fingerprint
    audit = ProtocolAudit(tmp_path)
    prior = reserve_narrow(audit, previous)
    prior_bytes = (tmp_path / "source-access" / design.scope.logical_key[7:] / "000000000001.json").read_bytes()
    report = evaluate(audit, design, selection_after(design, prior))

    assert_prior_unchanged(audit, prior, prior_bytes)
    assert "previous_protected_source_access" in report.source_reasons
    assert not report.confirmatory_eligible
    assert not report.family.confirmatory_eligible
    assert report.family.access.purpose == "verification"


def test_disjoint_train_and_test_rows_in_the_same_complete_manifest_remain_eligible(tmp_path):
    design = current_design()
    manifest = design.partition_manifest
    train = tuple(
        member for member in manifest.members if manifest.source_plan.decide(member.metadata).partition == "train"
    )
    protected = manifest.protected_members
    assert len(train) == 1 and len(protected) == 2
    assert {member.metadata.observation_id for member in train}.isdisjoint(
        member.metadata.observation_id for member in protected
    )
    audit = ProtocolAudit(tmp_path)
    prior = reserve_narrow(audit, manifest)
    report = evaluate(audit, design, selection_after(design, prior, inspected_trial=True))

    assert report.source_receipt.request.partition_manifest_fingerprint == manifest.fingerprint
    assert report.source_reasons == ()
    assert report.confirmatory_eligible and report.family.confirmatory_eligible
    assert report.family.access.purpose == "confirmatory"
    assert report.family.results[0].conclusion == "insufficient_evidence"


def test_reclassified_development_receipt_cannot_justify_trial_access_history(tmp_path):
    design = current_design()
    audit = ProtocolAudit(tmp_path)
    prior = reserve_narrow(audit, prior_manifest(design, "cutoffs"))
    report = evaluate(audit, design, selection_after(design, prior, inspected_trial=True))

    assert "development_access_history_unverified" in report.source_reasons
    assert report.family.access.purpose == "verification"
    assert not report.confirmatory_eligible
    assert audit.sources.history(design.scope).receipts[0].partition_manifest_verified


def test_reclassified_report_uses_real_legacy_markers_and_cached_view_preserves_assessment(tmp_path):
    design = current_design()
    audit = ProtocolAudit(tmp_path)
    prior = reserve_narrow(audit, prior_manifest(design, "cutoffs"))
    retained = (tmp_path / "source-access" / design.scope.logical_key[7:] / "000000000001.json").read_bytes()
    report = evaluate(audit, design, selection_after(design, prior))
    original_fingerprint = report.fingerprint
    original_wire = report.to_json()
    before_view = EvaluationAudit(tmp_path).history(design.scope.holdout_id)
    viewed = audit.read_report(report.locator, actor="authored-cached-reclassified-view")

    assert_prior_unchanged(audit, prior, retained)
    assert tuple(receipt.purpose for receipt in before_view) == ("development", "verification")
    assert viewed.original.fingerprint == original_fingerprint
    assert viewed.original.to_json() == original_wire
    assert not viewed.original.confirmatory_eligible and not viewed.original.family.confirmatory_eligible
    assert not viewed.new_confirmation
    assert viewed.access.purpose == "verification"
    assert viewed.source_receipt.request.report_fingerprint == original_fingerprint


@pytest.fixture(scope="module")
def first_protected_report(tmp_path_factory):
    design = current_design()
    return evaluate(ProtocolAudit(tmp_path_factory.mktemp("authored-first-protected")), design, selection_after(design))


@pytest.mark.parametrize("manifest_fingerprint", [None, "sha256:" + "f" * 64])
def test_report_rejects_forged_current_manifest_even_with_an_internally_valid_replaced_prefix(
    first_protected_report, manifest_fingerprint
):
    report = first_protected_report
    assert len(report.source_prefix.receipts) == 1
    assert report.source_receipt.sequence == 1
    request = changed(report.source_receipt.request, partition_manifest_fingerprint=manifest_fingerprint)
    forged_receipt = SourceReceipt.model_validate(report.source_receipt.model_copy(update={"request": request}))
    valid_prefix = SourcePrefix(schema_version=1, scope=report.source_prefix.scope, receipts=(forged_receipt,))
    assert valid_prefix.receipts == (forged_receipt,)
    supplied = report.model_copy(update={"source_receipt": forged_receipt, "source_prefix": valid_prefix})

    with pytest.raises(ValueError, match="manifest"):
        ProtocolReport.model_validate(supplied)
