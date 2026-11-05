"""Actual protected continuation refuses reset and preserves earlier descriptive source exposure."""

import hashlib
import json
import shutil

import polars as pl
import pytest

from coleman.application.workflow_registry import snapshot_history
from coleman.evaluation.protected import EvaluationAudit
from coleman.results.domain import fingerprint
from coleman.workflows.contracts import PlanningRequest
from coleman.workflows.lifecycle import ExecutePlan
from tests.workflows.test_execution import archived_handoff, setup_workflow, wait


def execute(service, plan, key):
    return service.execute(ExecutePlan(schema_version=1, plan_id=plan.plan_id, request_id=key))


def protected_plan(service, dataset_id="owned-ci-three-build-v1"):
    return service.plan(
        PlanningRequest(
            schema_version=1, dataset_id=dataset_id, evaluation="protected", seeds=(42,), ucb_exploration=(1.0,)
        )
    )


def test_descriptive_and_imported_outcomes_cannot_become_a_virgin_holdout(tmp_path):
    service, preview = setup_workflow(tmp_path)
    descriptive = execute(service, preview, "descriptive")
    wait(service, descriptive.job_id)
    service.import_evidence(service.export(descriptive.job_id).to_json())
    assert len(list((tmp_path / "source-exposures").rglob("*.json"))) == 2
    with pytest.raises(ValueError, match="Descriptive replay"):
        service.evaluate(descriptive.job_id)
    protected = execute(service, protected_plan(service), "after-descriptive")
    wait(service, protected.job_id, "awaiting_evaluation")
    service.evaluate(protected.job_id)
    wait(service, protected.job_id)
    report = service.report(protected.job_id)
    assert report.comparison is not None
    assert report.comparison.access.prior_test_accesses == 2
    assert report.interpretation == "insufficient_evidence"
    audit = EvaluationAudit(tmp_path / "audit")
    history = audit.history(report.comparison.access.holdout_id)
    assert tuple(receipt.purpose for receipt in history) == ("development", "development", "confirmatory")
    # Re-encoding/projecting the same upstream observations changes data identity,
    # while the stable source scope still counts those earlier accesses.
    original = service.saved_plan(preview.plan_id).dataset
    source = tmp_path / "datasets" / f"{original.sha256[7:]}.parquet"
    reencoded = tmp_path / "reencoded.parquet"
    pl.read_parquet(source).write_parquet(reencoded, compression="uncompressed")
    assert hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(reencoded.read_bytes()).digest()
    manifest = snapshot_history(
        tmp_path,
        reencoded,
        dataset_id="same-source-new-storage",
        source_fingerprint=original.source_fingerprint,
        projection_fingerprint=fingerprint([original.projection_fingerprint, "reencoded"]),
        evidence=original.evidence,
        assumptions=original.assumptions,
        metadata=original.metadata,
        temporal_plan=original.temporal_plan,
    )
    service.register(manifest)
    second = execute(service, protected_plan(service, manifest.dataset_id), "same-upstream")
    wait(service, second.job_id, "awaiting_evaluation")
    service.evaluate(second.job_id)
    wait(service, second.job_id)
    comparison = service.report(second.job_id).comparison
    assert comparison is not None
    assert comparison.access.holdout_id == report.comparison.access.holdout_id
    assert comparison.access.prior_test_accesses == 3


def test_missing_native_state_cannot_silently_restart_protected_learning(tmp_path, monkeypatch):
    service, _ = setup_workflow(tmp_path)
    preview = protected_plan(service)
    job = execute(service, preview, "state-handoff")
    wait(service, job.job_id, "awaiting_evaluation")
    root = tmp_path / "jobs" / job.job_id
    original = archived_handoff(service, preview, root, monkeypatch)
    handoff = json.loads((root / "evaluation-handoff.json").read_text())
    before = {str(path): path.read_bytes() for path in (root / "evidence").rglob("*.parquet")}
    checkpoints = list((root / "native").glob("*/*/checkpoints"))
    assert len(checkpoints) == 2
    for path in checkpoints:
        shutil.rmtree(path)
    service.evaluate(job.job_id)
    failed = wait(service, job.job_id, "failed")
    assert failed.error is not None
    assert "reset missing native learning state" in failed.error
    assert failed.result_digest is None
    assert not (root / "complete.json").exists()
    assert not list((root / "evidence").glob("*/evaluation/complete.json"))
    assert all(path.read_bytes() == before[str(path)] for path in (root / "evidence").rglob("*.parquet"))
    # Restore the exact original private checkpoint bytes as a host recovery action;
    # no imported evidence or tool can supply these executable files.
    for pin in handoff["checkpoints"]:
        source = root / pin["name"]
        parts = source.relative_to(root / "frozen-learning").parts
        target = root / "native" / parts[0]
        for part in parts[1:]:
            target = target / part
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    service.resume(job.job_id)
    completed = wait(service, job.job_id)
    report = service.report(job.job_id)
    assert completed.invocations == 3
    assert completed.attempts == 4
    assert report.comparison is not None
    assert report.comparison.access.prior_test_accesses == 1
    assert report.comparison.frozen.plan.test_end > original.frozen.plan.test_end
    assert report.comparison.frozen.protocol == original.frozen.protocol
    assert sum(len(trial.rows) for trial in report.trials) == 6
    assert report.fallback_configuration is not None
