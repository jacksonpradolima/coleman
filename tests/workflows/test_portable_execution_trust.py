"""Portable metric integrity does not authenticate a producer's trial/seed/treatment attribution."""

from coleman.application.workflow_downloads import report_overview
from coleman.application.workflow_exports import EvidenceBundle
from coleman.workflows.lifecycle import ExecutePlan, TrialSummary
from tests.workflows.test_execution import setup_workflow, wait


def test_unchanged_native_bytes_with_swapped_seed_claims_remain_explicitly_unverified(tmp_path):
    service, preview = setup_workflow(tmp_path)
    job = service.execute(ExecutePlan(schema_version=1, request_id="seed-claim", plan_id=preview.plan_id))
    wait(service, job.job_id)
    original = service.export(job.job_id)
    first = original.plan.trials[0]
    second = next(
        t for t in original.plan.trials if t.configuration_id == first.configuration_id and t.seed != first.seed
    )
    swaps = {first.trial_id: second.trial_id, second.trial_id: first.trial_id}
    payload = original.model_dump(mode="json")
    for receipt in payload["report"]["trials"]:
        receipt["trial_id"] = swaps.get(receipt["trial_id"], receipt["trial_id"])
    for event in payload["report"]["physical_attempts"]:
        event["trial_id"] = swaps.get(event["trial_id"], event["trial_id"])
    bundle = EvidenceBundle.model_validate(payload)
    assert bundle.files == original.files
    assert bundle.native_execution_bindings_verified is False
    for receipt in bundle.report.trials:
        assert TrialSummary.from_result(receipt).native_execution_bindings_verified is False
    handle = service.import_evidence(bundle.to_json())
    assert handle.native_execution_bindings_verified is False
    assert service.imported_evidence(handle.evidence_id).to_json() == bundle.to_json()
    assert report_overview(service, job.job_id).native_execution_bindings_verified is False
