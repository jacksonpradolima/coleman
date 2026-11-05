"""Real retained evidence rejects inconsistent freeze, scope capacity and checkpoint frontiers."""

import copy
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

from coleman.application.baseline_lock import BaselineBusyError, baseline_lock
from coleman.application.workflow_checkpoint import WorkflowCheckpointStore
from coleman.application.workflow_evaluation import compare_continuation, prepare_evaluation
from coleman.application.workflow_exports import EvidenceBundle
from coleman.application.workflow_exposure import source_lock, synchronize_exposures
from coleman.application.workflow_registry import owned_fixture, snapshot_history
from coleman.application.workflows import WorkflowService
from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.checkpoint.state import CheckpointPayload
from coleman.evaluation.protected import EvaluationAudit, FrozenCandidate, PairedObservation, evaluate, freeze
from coleman.policy import base as policy_base
from coleman.results.domain import fingerprint
from coleman.workflows.contracts import PlanningRequest, WorkflowLimits
from coleman.workflows.lifecycle import ExecutePlan, HostPolicy
from coleman.workflows.repetitions import native_repetition_evidence
from tests.workflows.test_execution import archived_handoff, setup_workflow, wait


@pytest.fixture(scope="module")
def protected_evidence(tmp_path_factory):
    service, preview = setup_workflow(tmp_path_factory.mktemp("review-protected"), evaluation="protected")
    request = ExecutePlan(schema_version=1, request_id="review-protected", plan_id=preview.plan_id)
    job = service.execute(request)
    wait(service, job.job_id, "awaiting_evaluation")
    service.evaluate(job.job_id)
    wait(service, job.job_id)
    return service.export(job.job_id)


def forged_learning_bundle(bundle, mutation, directory):
    """Recompute a genuinely valid generic comparison while falsifying workflow-specific state claims."""
    comparison = bundle.report.comparison
    assert comparison is not None
    original = comparison.frozen
    challenger = original.candidate("challenger")
    configuration = json.loads(challenger.configuration_json)
    learning = configuration["learning_state"]
    if mutation == "missing_seed":
        learning.pop()
    elif mutation == "duplicate_trial":
        learning[1] = copy.deepcopy(learning[0])
    elif mutation == "foreign_trial":
        other = next(t for t in bundle.plan.trials if t.configuration_id != configuration["configuration_id"])
        learning[0]["trial_id"] = other.trial_id
        learning[0]["seed"] = other.seed
    elif mutation == "wrong_seed":
        learning[0]["seed"] = 0
    elif mutation == "foreign_checkpoint":
        pin = learning[0]["checkpoints"][0]
        pin["name"] = "frozen-learning/" + "0" * 64 + "/checkpoint.pkl"
    elif mutation == "duplicate_checkpoint":
        learning[0]["checkpoints"].append(copy.deepcopy(learning[0]["checkpoints"][0]))
    elif mutation == "wrong_design":
        configuration["corpus_repetition_design"] = fingerprint(["invented-repetition-design"])
    candidates = tuple(
        FrozenCandidate.from_configuration(candidate.candidate_id, configuration)
        if candidate.candidate_id == "challenger"
        else candidate
        for candidate in original.candidates
    )
    modified = freeze(
        candidates, original.data_fingerprint, original.plan, original.protocol, original.comparison_family
    )
    observations = []
    for reference in comparison.inputs:
        provenance = reference.record.provenance.model_copy(
            update={
                "spec_fingerprint": modified.candidate(reference.candidate_id).configuration_fingerprint,
                "evaluation_fingerprint": modified.fingerprint,
            }
        )
        record = reference.record.model_copy(update={"provenance": provenance})
        observations.append(
            PairedObservation(candidate_id=reference.candidate_id, metadata=reference.metadata, record=record)
        )
    audit = EvaluationAudit(directory / "audit")
    audit.save_freeze(modified)
    # No fake statistical report: the existing evaluator recomputes the entire
    # family and validates actual native-derived repository means.
    replacement = evaluate(modified, audit, lambda: tuple(observations), actor="forgery-regression")
    data = bundle.model_dump(mode="json")
    data["report"]["comparison"] = replacement.model_dump(mode="json")
    return data


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_seed",
        "duplicate_trial",
        "foreign_trial",
        "wrong_seed",
        "foreign_checkpoint",
        "duplicate_checkpoint",
        "wrong_design",
    ],
)
def test_portable_frozen_learning_membership_is_bound_to_the_actual_plan(protected_evidence, tmp_path, mutation):
    payload = forged_learning_bundle(protected_evidence, mutation, tmp_path)
    with pytest.raises(ValueError):
        EvidenceBundle.model_validate(payload)


def test_protected_report_cannot_be_relabelled_as_descriptive(protected_evidence):
    payload = protected_evidence.model_dump(mode="json")
    payload["report"]["interpretation"] = "descriptive_only"
    with pytest.raises(ValueError, match="interpretation"):
        EvidenceBundle.model_validate(payload)


@pytest.mark.parametrize(
    "updates", [dict(run_id="another-run"), dict(experiment=2), dict(step=1), dict(step=3), dict(experiment=True)]
)
def test_native_payload_must_match_its_marker_before_restoring_global_rng(tmp_path, updates):
    store = WorkflowCheckpointStore(str(tmp_path), "sha256:" + "a" * 64)
    store.save(CheckpointPayload(run_id="owned", experiment=1, step=2))
    actual = LocalCheckpointStore(str(tmp_path)).load("owned", 1)
    assert actual is not None
    # Serialize only an owned native fixture; portable imports never supply or load these bytes.
    path = tmp_path / "owned" / "ckpt_ex1_step2.pkl"
    path.write_bytes(pickle.dumps(replace(actual, **updates)))
    policy_base._rng.random(3)
    before = copy.deepcopy(policy_base._rng.bit_generator.state)
    with pytest.raises(ValueError, match="payload"):
        store.load("owned", 1)
    assert policy_base._rng.bit_generator.state == before


def test_pending_source_exposure_cannot_overfill_protected_audit_capacity(tmp_path):
    service, descriptive = setup_workflow(tmp_path, seeds=(42,), ucb_exploration=(1.0,))
    job = service.execute(ExecutePlan(schema_version=1, request_id="prior-outcomes", plan_id=descriptive.plan_id))
    wait(service, job.job_id)
    protected = service.plan(
        PlanningRequest(
            schema_version=1,
            dataset_id="owned-ci-three-build-v1",
            evaluation="protected",
            seeds=(42,),
            ucb_exploration=(1.0,),
        )
    )
    job = service.execute(ExecutePlan(schema_version=1, request_id="scope-capacity", plan_id=protected.plan_id))
    wait(service, job.job_id, "awaiting_evaluation")
    root = tmp_path / "jobs" / job.job_id
    # Freeze without queueing so a retained coordinator cannot admit protected
    # work before the prior ledger is complete. Admission uses the real service.
    handoff = prepare_evaluation(tmp_path, root, service.saved_plan(protected.plan_id))
    audit = EvaluationAudit(tmp_path / "audit")
    for _ in range(511):
        audit.access(handoff.frozen, partition="test", purpose="development", actor="capacity-regression")
    service.evaluate(job.job_id)
    rejected = wait(service, job.job_id, "failed")
    assert rejected.result_digest is None
    assert rejected.attempts == 2
    assert len(audit.history(handoff.frozen.protocol.holdout_id)) == 512
    assert not list((root / "evidence").glob("*/evaluation/complete.json"))


def test_digest_suffix_cannot_replace_a_complete_owned_plan_identity(tmp_path):
    service, preview = setup_workflow(tmp_path)
    suffix = preview.plan_id[-1]
    with service.store.transaction() as connection:
        connection.execute("UPDATE plans SET id=?", (suffix,))
    with pytest.raises(ValueError, match="identity mismatch"):
        service.saved_plan(suffix)


@pytest.fixture(scope="module", params=["descriptive", "protected"])
def decision_evidence(request, tmp_path_factory):
    service, preview = setup_workflow(
        tmp_path_factory.mktemp("review-selection"),
        evaluation=request.param,
        seeds=(42,),
        ucb_exploration=(1.0,),
    )
    job = service.execute(ExecutePlan(schema_version=1, request_id="selection", plan_id=preview.plan_id))
    if request.param == "protected":
        wait(service, job.job_id, "awaiting_evaluation")
        service.evaluate(job.job_id)
    wait(service, job.job_id)
    return service.export(job.job_id)


@pytest.mark.parametrize("field", ["selected_configuration", "fallback_configuration"])
@pytest.mark.parametrize("replacement", ["null", "other_planned"])
def test_report_choices_must_match_the_native_winner_and_random_fallback(decision_evidence, field, replacement):
    payload = decision_evidence.model_dump(mode="json")
    original = payload["report"][field]
    payload["report"][field] = (
        None
        if replacement == "null"
        else next(
            trial.configuration_id for trial in decision_evidence.plan.trials if trial.configuration_id != original
        )
    )
    with pytest.raises(ValueError):
        EvidenceBundle.model_validate(payload)


def test_protected_planning_requires_two_invocations_before_expansion(tmp_path, monkeypatch):
    service, preview = setup_workflow(tmp_path)
    request = PlanningRequest(
        schema_version=1,
        dataset_id=service.saved_plan(preview.plan_id).dataset.dataset_id,
        evaluation="protected",
        limits=WorkflowLimits(max_attempts=1),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexecutable protected plans must reject before expansion")

    monkeypatch.setattr("coleman.workflows.planning.expand_sweep", forbidden)
    with pytest.raises(ValueError, match="two.*invocations"):
        service.plan(request)


def test_queued_protected_evaluation_uses_a_current_assembly_window(tmp_path, monkeypatch):
    service, preview = setup_workflow(tmp_path, evaluation="protected", seeds=(42,), ucb_exploration=(1.0,))
    job = service.execute(ExecutePlan(schema_version=1, request_id="archived-queue", plan_id=preview.plan_id))
    wait(service, job.job_id, "awaiting_evaluation")
    root = tmp_path / "jobs" / job.job_id
    original = archived_handoff(service, preview, root, monkeypatch)
    before = (root / "evaluation-handoff.json").read_bytes()
    service.evaluate(job.job_id)
    wait(service, job.job_id)
    report = service.report(job.job_id)
    assert report.comparison is not None
    assert report.comparison.frozen.plan.test_end > original.frozen.plan.test_end
    assert report.comparison.frozen.protocol == original.frozen.protocol
    assert report.comparison.access.holdout_id == original.frozen.protocol.holdout_id
    assert report.comparison.access.prior_test_accesses == 0
    assert (root / "evaluation-handoff.json").read_bytes() == before
    for candidate in report.comparison.frozen.candidates:
        previous = json.loads(original.frozen.candidate(candidate.candidate_id).configuration_json)
        current = json.loads(candidate.configuration_json)
        assert current["learning_state"] == previous["learning_state"]
        assert current["configuration_id"] == previous["configuration_id"]


def test_expired_direct_context_rejects_before_reserving_protected_access(tmp_path, monkeypatch):
    service, preview = setup_workflow(tmp_path, evaluation="protected", seeds=(42,), ucb_exploration=(1.0,))
    job = service.execute(ExecutePlan(schema_version=1, request_id="expired-context", plan_id=preview.plan_id))
    wait(service, job.job_id, "awaiting_evaluation")
    root = tmp_path / "jobs" / job.job_id
    initial = archived_handoff(service, preview, root, monkeypatch)
    plan = service.saved_plan(preview.plan_id)

    def forbidden():
        raise AssertionError("Expired analysis must reject before a protected outcome callback")

    with pytest.raises(ValueError, match="assembly window expired"):
        compare_continuation(tmp_path, root, plan, initial, forbidden)
    assert EvaluationAudit(tmp_path / "audit").history(initial.frozen.protocol.holdout_id) == ()


@pytest.fixture(scope="module")
def large_registered_history(tmp_path_factory):
    root = tmp_path_factory.mktemp("large-input-reservation")
    service = WorkflowService(
        root,
        policy=HostPolicy(
            schema_version=1,
            caps=WorkflowLimits(max_configurations=64, max_trials=128, max_output_bytes=67108864),
        ),
    )
    base = service.register(owned_fixture(root))
    frame = pl.read_parquet(root / "datasets" / f"{base.sha256.removeprefix('sha256:')}.parquet")
    payloads = [str(i) + "x" * 2097152 for i in range(frame.height)]
    path = root / "large-ci.parquet"
    frame.with_columns(pl.Series("auxiliary_log", payloads)).write_parquet(path, compression="uncompressed")
    manifest = snapshot_history(
        root,
        path,
        dataset_id="large-ci-history",
        source_fingerprint=base.source_fingerprint,
        projection_fingerprint=fingerprint([base.projection_fingerprint, "auxiliary-log-v1"]),
        evidence=base.evidence,
        assumptions=("Authored CI history with auxiliary log context.",),
    )
    assert manifest.size_bytes > 16777216
    service.register(manifest)
    return service, manifest


@pytest.mark.parametrize("allowance", ["default", "input_size"])
def test_registered_input_must_leave_output_allowance_before_expansion(
    large_registered_history, monkeypatch, allowance
):
    service, manifest = large_registered_history
    limits = WorkflowLimits() if allowance == "default" else WorkflowLimits(max_output_bytes=manifest.size_bytes)
    request = PlanningRequest(schema_version=1, dataset_id=manifest.dataset_id, limits=limits)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexecutable input reservations must reject before expansion")

    monkeypatch.setattr("coleman.workflows.planning.expand_sweep", forbidden)
    with pytest.raises(ValueError, match="input.*output"):
        service.plan(request)


def test_queued_admission_waits_for_a_retiring_supervisor_lock(tmp_path):
    service, preview = setup_workflow(tmp_path, policies=("Random",), seeds=(42,))
    request = ExecutePlan(schema_version=1, request_id="retiring-supervisor", plan_id=preview.plan_id)
    with ThreadPoolExecutor(max_workers=1) as pool:
        # Model the retained descriptor of a retiring owner. A busy inode alone
        # does not prove that any coordinator has observed this new admission.
        with baseline_lock(tmp_path / "supervisor.lock", handoff=True):
            future = pool.submit(service.execute, request)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with closing(service.store.connect()) as connection:
                    jobs = service.store.jobs(connection)
                if jobs:
                    break
                time.sleep(0.01)
            assert jobs
            with pytest.raises(TimeoutError):
                future.result(timeout=0.2)
        job = future.result(timeout=15)
    complete = wait(service, job.job_id)
    assert complete.attempts == 1
    assert len(service.report(job.job_id).trials) == 1


def test_unobserved_queue_timeout_keeps_idempotent_admission(tmp_path, monkeypatch):
    import coleman.application.workflow_supervisor as supervision

    service, preview = setup_workflow(tmp_path, policies=("Random",), seeds=(42,))
    request = ExecutePlan(schema_version=1, request_id="unobserved-queue", plan_id=preview.plan_id)
    with monkeypatch.context() as patch:
        patch.setattr(supervision, "_ACK_WAIT_SECONDS", 0.1)
        with baseline_lock(tmp_path / "supervisor.lock", handoff=True), pytest.raises(ValueError, match="acknowledge"):
            service.execute(request)
    with closing(service.store.connect()) as connection:
        admitted = service.store.jobs(connection)
    assert len(admitted) == 1
    assert admitted[0].state == "queued"
    job = service.execute(request)
    assert job.job_id == admitted[0].job_id
    assert wait(service, job.job_id).attempts == 1


@pytest.mark.parametrize(
    "changes",
    [{"schema_version": 2}, {"requested": 1, "observed": 2}, {"requested": 2**63 - 1}],
)
def test_unknown_or_impossible_coordinator_progress_cannot_acknowledge_work(tmp_path, monkeypatch, changes):
    from coleman.application.workflow_supervisor import ensure_supervisor

    service, preview = setup_workflow(tmp_path, policies=("Random",), seeds=(42,))
    request = ExecutePlan(schema_version=1, request_id="invalid-progress", plan_id=preview.plan_id)
    with monkeypatch.context() as patch:
        patch.setattr(service, "_start", lambda: None)
        job = service.execute(request)
    record = {"schema_version": 1, "contract": "coleman-workflows-v1", "requested": 0, "observed": 0, **changes}
    with service.store.transaction() as connection:
        connection.execute("INSERT INTO settings VALUES ('supervisor_progress_v1',?)", (json.dumps(record),))
    with pytest.raises(ValueError):
        ensure_supervisor(tmp_path)
    assert service.status(job.job_id).state == "queued"
    assert service.status(job.job_id).worker_active is False


def test_supervision_progress_v1_golden():
    from coleman.application.workflow_supervisor import _SupervisionProgress

    path = Path(__file__).with_name("fixtures") / "supervision-progress-v1.json"
    record = _SupervisionProgress.from_json(path.read_bytes())
    assert record.to_json() + "\n" == path.read_text()


@pytest.mark.parametrize("quota", ["slots", "bytes"])
def test_capacity_rejected_import_still_preserves_outcome_exposure(
    protected_evidence, decision_evidence, tmp_path, quota
):
    first = protected_evidence.to_json()
    second = decision_evidence.to_json()
    assert protected_evidence.fingerprint != decision_evidence.fingerprint
    byte_limit = len(first.encode()) + len(second.encode()) - 1
    assert byte_limit >= 65536
    policy = HostPolicy(
        schema_version=1,
        max_jobs=1 if quota == "slots" else 64,
        max_retained_bytes=536870912 if quota == "slots" else byte_limit,
    )
    service = WorkflowService(tmp_path, policy=policy)
    service.import_evidence(first)
    with pytest.raises(ValueError, match="slots/bytes exhausted"):
        service.import_evidence(second)
    comparison = protected_evidence.report.comparison
    assert comparison is not None
    audit = EvaluationAudit(tmp_path / "audit")
    audit.save_freeze(comparison.frozen)
    observations = tuple(
        PairedObservation(candidate_id=item.candidate_id, metadata=item.metadata, record=item.record)
        for item in comparison.inputs
    )
    with source_lock(tmp_path, protected_evidence.plan.dataset.source_fingerprint):
        synchronize_exposures(tmp_path, comparison.frozen)
        result = evaluate(comparison.frozen, audit, lambda: observations, actor="rejected-import-regression")
    assert result.access.prior_test_accesses == 2


@pytest.mark.parametrize("allowance", [-1, 0, 1])
def test_import_quota_charges_canonical_bytes_across_owners(protected_evidence, tmp_path, allowance):
    compact = protected_evidence.model_dump_json(exclude_defaults=True)
    canonical = protected_evidence.to_json()
    size = len(canonical.encode("utf-8"))
    assert len(compact.encode("utf-8")) < size - 1
    policy = HostPolicy(schema_version=1, max_retained_bytes=2 * size + allowance)
    first = WorkflowService(tmp_path, owner="first", policy=policy)
    retained = first.import_evidence(compact)
    assert first.import_evidence(canonical) == retained
    second = WorkflowService(tmp_path, owner="second")
    if allowance < 0:
        with pytest.raises(ValueError, match="slots/bytes exhausted"):
            second.import_evidence(compact)
        expected = 1
    else:
        other = second.import_evidence(compact)
        assert second.import_evidence(canonical) == other
        assert other.evidence_id != retained.evidence_id
        expected = 2
    with closing(first.store.connect()) as connection:
        count, stored_size = connection.execute(
            "SELECT count(*),sum(length(CAST(payload AS BLOB))) FROM imports"
        ).fetchone()
    assert count == expected
    assert stored_size == expected * size
    assert stored_size <= policy.max_retained_bytes
    assert first.import_evidence(compact) == retained
    assert first.imported_evidence(retained.evidence_id).to_json() == canonical


def test_import_reserves_exposure_before_native_outcome_materialization(protected_evidence, tmp_path, monkeypatch):
    import coleman.application.workflow_exports as exports

    def forbidden(*args, **kwargs):
        raise AssertionError("A busy exposure scope must reject before native outcome decoding")

    service = WorkflowService(tmp_path)
    payload = protected_evidence.to_json()
    monkeypatch.setattr(exports, "_native_table", forbidden)
    with (
        source_lock(tmp_path, protected_evidence.plan.dataset.source_fingerprint),
        pytest.raises(BaselineBusyError),
    ):
        service.import_evidence(payload)


@pytest.mark.parametrize("mutation", ["full", "future_version", "duplicate", "overfull"])
def test_import_attempt_capacity_rejects_before_outcome_decoding(protected_evidence, tmp_path, monkeypatch, mutation):
    import coleman.application.workflow_exports as exports

    def forbidden(*args, **kwargs):
        raise RuntimeError("Attempt admission must reject before native outcome decoding")

    service = WorkflowService(tmp_path)
    payload = protected_evidence.to_json()
    identities = [fingerprint(["prior-import", index])[7:] for index in range(1024)]
    record = {"schema_version": 1, "contract": "coleman-workflows-v1", "identities": sorted(identities)}
    if mutation == "future_version":
        record["schema_version"] = 2
    elif mutation == "duplicate":
        record["identities"] = [identities[0], identities[0]]
    elif mutation == "overfull":
        record["identities"] = sorted([*identities, "extra-attempt"])
    with service.store.transaction() as connection:
        connection.execute("INSERT INTO settings VALUES ('import_exposure_attempts_v1',?)", (json.dumps(record),))
    monkeypatch.setattr(exports, "_native_table", forbidden)
    with pytest.raises(ValueError):
        service.import_evidence(payload)
    assert not (tmp_path / "source-exposures").exists()


def test_full_import_attempt_index_preserves_exact_retry(protected_evidence, tmp_path):
    service = WorkflowService(tmp_path)
    payload = protected_evidence.to_json()
    identity = fingerprint(["workflow.import-attempt.v1", service.owner, payload])[7:]
    identities = [identity, *(fingerprint(["prior-import", index])[7:] for index in range(1023))]
    record = {"schema_version": 1, "contract": "coleman-workflows-v1", "identities": sorted(identities)}
    with service.store.transaction() as connection:
        connection.execute("INSERT INTO settings VALUES ('import_exposure_attempts_v1',?)", (json.dumps(record),))
    retained = service.import_evidence(payload)
    marker = next((tmp_path / "source-exposures").rglob("*.json"))
    original = marker.read_bytes()
    assert service.import_evidence(payload) == retained
    assert marker.read_bytes() == original
    with closing(service.store.connect()) as connection:
        saved = connection.execute("SELECT payload FROM settings WHERE name='import_exposure_attempts_v1'").fetchone()
    assert json.loads(saved[0]) == record


def test_rejected_import_retry_retains_one_attempt_and_exposure(protected_evidence, tmp_path):
    service = WorkflowService(tmp_path)
    data = protected_evidence.model_dump(mode="json")
    data["files"] = []
    payload = json.dumps(data)
    with pytest.raises(ValueError, match="membership"):
        service.import_evidence(payload)
    marker = next((tmp_path / "source-exposures").rglob("*.json"))
    original = marker.read_bytes()
    with pytest.raises(ValueError, match="membership"):
        service.import_evidence(payload)
    assert marker.read_bytes() == original
    assert len(list((tmp_path / "source-exposures").rglob("*.json"))) == 1
    with closing(service.store.connect()) as connection:
        saved = connection.execute("SELECT payload FROM settings WHERE name='import_exposure_attempts_v1'").fetchone()
        assert connection.execute("SELECT count(*) FROM imports").fetchone()[0] == 0
    assert len(json.loads(saved[0])["identities"]) == 1


@pytest.mark.parametrize("mode", ["canonical", "reordered", "busy"])
def test_import_does_not_decode_outcome_json_before_exposure(protected_evidence, tmp_path, monkeypatch, mode):
    payload = protected_evidence.to_json()
    if mode == "reordered":
        data = json.loads(payload)
        payload = json.dumps(dict(reversed(tuple(data.items()))), indent=2)
    service = WorkflowService(tmp_path)
    original = json.JSONDecoder.raw_decode
    decoded = []

    def guarded(decoder, document, idx=0):
        if document == payload:
            if not list((tmp_path / "source-exposures").rglob("*.json")):
                raise RuntimeError("Outcome-bearing JSON was decoded before its source exposure")
            decoded.append(document)
        return original(decoder, document, idx)

    monkeypatch.setattr(json.JSONDecoder, "raw_decode", guarded)
    if mode == "busy":
        with (
            source_lock(tmp_path, protected_evidence.plan.dataset.source_fingerprint),
            pytest.raises(BaselineBusyError),
        ):
            service.import_evidence(payload)
        assert not decoded
    else:
        assert service.import_evidence(payload).read_only
        assert decoded


@pytest.mark.parametrize(
    "mutation",
    ["observation_id", "revision_id", "event_at", "target_end", "decision_id", "namespace", "source_revision"],
)
def test_protected_inputs_must_retain_native_unit_metadata_and_provenance(protected_evidence, tmp_path, mutation):
    comparison = protected_evidence.report.comparison
    assert comparison is not None
    rows = []
    for reference in comparison.inputs:
        metadata, record = reference.metadata, reference.record
        if mutation in ("observation_id", "revision_id"):
            metadata = metadata.model_copy(update={mutation: "invented-source-identity"})
        elif mutation == "event_at":
            metadata = metadata.model_copy(update={"event_at": metadata.event_at + timedelta(minutes=1)})
        elif mutation == "target_end":
            metadata = metadata.model_copy(update={"target_end": metadata.event_at + timedelta(minutes=2)})
        elif mutation == "decision_id":
            record = record.model_copy(update={"decision_id": "invented-source-unit"})
        elif mutation == "namespace":
            record = record.model_copy(update={"namespace": "ci.invented.unit_means"})
        else:
            provenance = record.provenance.model_copy(update={"source_revision": fingerprint(["invented-raw-source"])})
            record = record.model_copy(update={"provenance": provenance})
        rows.append(PairedObservation(candidate_id=reference.candidate_id, metadata=metadata, record=record))
    audit = EvaluationAudit(tmp_path / "audit")
    audit.save_freeze(comparison.frozen)
    # The actual generic evaluator accepts these otherwise eligible means. The
    # workflow reader must additionally bind them to the registered native source.
    rebuilt = evaluate(comparison.frozen, audit, lambda: tuple(rows), actor="source-metadata-regression")
    data = protected_evidence.model_dump(mode="json")
    data["report"]["comparison"] = rebuilt.model_dump(mode="json")
    assert data["files"] == protected_evidence.model_dump(mode="json")["files"]
    with pytest.raises(ValueError):
        EvidenceBundle.model_validate(data)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "unplanned_trial", "wrong_plan", "wrong_stage", "unselected_configuration"]
)
def test_native_repetition_projection_rejects_invalid_receipt_membership(protected_evidence, mutation):
    plan, report = protected_evidence.plan, protected_evidence.report
    comparison = report.comparison
    assert comparison is not None
    first = report.trials[0]
    if mutation == "duplicate":
        changed = first.model_copy(update={"rows": (*first.rows, first.rows[0])})
    elif mutation == "missing":
        changed = first.model_copy(update={"rows": tuple(row for row in first.rows if row.source_build_id != 3)})
    elif mutation == "wrong_plan":
        changed = first.model_copy(update={"plan_fingerprint": fingerprint(["another-plan"])})
    elif mutation == "wrong_stage":
        changed = first.model_copy(update={"stage": "tuning"})
    elif mutation == "unselected_configuration":
        selected = {item.trial_id for item in report.trials}
        other = next(trial for trial in plan.trials if trial.trial_id not in selected)
        changed = first.model_copy(update={"trial_id": other.trial_id})
    else:
        changed = first.model_copy(update={"trial_id": fingerprint(["another-trial"])})
    receipts = (changed, *report.trials[1:])
    assembled = comparison.inputs[0].record.available_at
    with pytest.raises(ValueError):
        native_repetition_evidence(plan, comparison.frozen, receipts, assembled_at=assembled)
