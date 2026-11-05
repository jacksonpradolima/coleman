"""Public evidence trust boundaries and deterministic worker-entry admission."""

import pytest

from coleman.application.workflow_exports import EvidenceBundle
from coleman.workflows import PlanningRequest, WorkflowLimits, make_plan
from coleman.workflows.lifecycle import ExecutePlan, TrialSummary
from coleman.workflows.planning import artifact_entry_bound
from tests.workflows.test_execution import setup_workflow, wait
from tests.workflows.test_planning import manifest, recorded_runtime


@pytest.fixture(scope="module", params=("descriptive", "protected"))
def completed_evidence(request, tmp_path_factory):
    service, preview = setup_workflow(tmp_path_factory.mktemp(request.param), evaluation=request.param)
    job = service.execute(ExecutePlan(schema_version=1, request_id="final-review", plan_id=preview.plan_id))
    if request.param == "protected":
        wait(service, job.job_id, "awaiting_evaluation")
        service.evaluate(job.job_id)
    wait(service, job.job_id)
    bundle = service.export(job.job_id)
    entries = sum(1 for _ in (service.store.root / "jobs" / job.job_id).rglob("*"))
    assert entries <= artifact_entry_bound(bundle.plan.request, trials=len(bundle.plan.trials), builds=3)
    return bundle


@pytest.mark.parametrize("assumptions", [[], ["An unqualified claim of production superiority."]])
def test_mode_assumptions_cannot_be_removed_or_replaced(completed_evidence, assumptions):
    payload = completed_evidence.model_dump(mode="json")
    payload["report"]["assumptions"] = assumptions
    with pytest.raises(ValueError, match="assumptions"):
        EvidenceBundle.model_validate(payload)


def test_native_run_ids_are_explicitly_unverified_portable_claims(completed_evidence, tmp_path):
    payload = completed_evidence.model_dump(mode="json")
    # Descriptive claims have no inferred unit records additionally referencing the ID.
    if completed_evidence.plan.request.evaluation == "protected":
        assert completed_evidence.native_run_ids_verified is False
        return
    payload["report"]["trials"][0]["native_run_id"] = "0123456789ab"
    imported = EvidenceBundle.model_validate(payload)
    assert imported.native_run_ids_verified is False
    summary = TrialSummary.from_result(imported.report.trials[0])
    assert summary.native_run_id_verification == "producer_asserted_unverified"
    assert summary.native_run_id == "0123456789ab"
    service, _ = setup_workflow(tmp_path)
    handle = service.import_evidence(imported.to_json())
    assert handle.native_run_ids_verified is False
    assert service.imported_evidence(handle.evidence_id).native_run_ids_verified is False
    assert service.imported_evidence(handle.evidence_id).to_json() == imported.to_json()


def test_checkpoint_entries_reject_before_sweep_expansion(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Deterministically impossible job expanded its sweep")

    monkeypatch.setattr("coleman.workflows.planning.expand_sweep", forbidden)
    data = manifest().model_copy(update={"build_ids": tuple(range(1, 258)), "rows": 771})
    request = PlanningRequest(
        schema_version=1,
        dataset_id=data.dataset_id,
        seeds=tuple(range(32)),
        limits=WorkflowLimits(max_trials=128, max_attempts=1),
    )
    runtime = recorded_runtime()
    with pytest.raises(ValueError, match="artifact.*entries"):
        make_plan(request, data, runtime)


@pytest.mark.parametrize("builds,accepted", [(35, True), (36, False)])
def test_entry_reservation_exact_boundary(builds, accepted):
    data = manifest().model_copy(update={"build_ids": tuple(range(1, builds + 1)), "rows": 3 * builds})
    request = PlanningRequest(
        schema_version=1,
        dataset_id=data.dataset_id,
        seeds=tuple(range(32)),
        limits=WorkflowLimits(max_trials=128, max_attempts=1),
    )
    if accepted:
        assert artifact_entry_bound(request, trials=128, builds=builds) == 32768
        assert len(make_plan(request, data, recorded_runtime()).trials) == 128
    else:
        runtime = recorded_runtime()
        with pytest.raises(ValueError, match="artifact.*entries"):
            make_plan(request, data, runtime)


def test_entry_reservation_grows_with_retries_and_native_work():
    request = PlanningRequest(schema_version=1, dataset_id="owned-ci", limits=WorkflowLimits(max_attempts=1))
    baseline = artifact_entry_bound(request, trials=8, builds=3)
    assert artifact_entry_bound(request, trials=8, builds=4) > baseline
    assert artifact_entry_bound(request, trials=9, builds=3) > baseline
    retry = request.model_copy(update={"limits": WorkflowLimits(max_attempts=2)})
    assert artifact_entry_bound(retry, trials=8, builds=3) > baseline
