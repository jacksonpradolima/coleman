"""Retained native facade refusals use actual CI evidence and source bytes."""

from __future__ import annotations

from contextlib import closing

import pytest

from coleman.application import workflow_artifacts
from coleman.application.workflow_registry import owned_fixture
from coleman.application.workflows import WorkflowService
from coleman.evaluation.protected import EvaluationAudit
from coleman.workflows.contracts import DatasetManifest, WorkflowError
from coleman.workflows.lifecycle import ExecutePlan, HostPolicy
from tests.workflows.test_execution import archived_handoff, setup_workflow, wait


@pytest.fixture(name="actual_bundle", scope="module")
def _actual_bundle(tmp_path_factory):
    """Execute the actual native runner; this is constructed CI evidence."""
    service, preview = setup_workflow(
        tmp_path_factory.mktemp("native-typed-host-refusals"), policies=("Random",), seeds=(42,)
    )
    job = service.execute(ExecutePlan(schema_version=1, request_id="real-ci", plan_id=preview.plan_id))
    wait(service, job.job_id)
    return service.export(job.job_id)


def test_registered_source_change_between_pin_and_read_is_not_published(tmp_path, monkeypatch):
    """Recheck actual bytes after a real source update, without a forged digest."""
    source_root = tmp_path / "prepared"
    manifest = owned_fixture(source_root)
    service = WorkflowService(tmp_path / "workspace")
    actual_pin = workflow_artifacts.pin_file
    observed = []

    def pin_then_change(root, path):
        pin = actual_pin(root, path)
        observed.append(pin)
        path.write_bytes(path.read_bytes() + b"changed after actual pin")
        return pin

    monkeypatch.setattr(workflow_artifacts, "pin_file", pin_then_change)
    with pytest.raises(WorkflowError, match="changed during validation"):
        service.register(manifest, snapshot_root=source_root)
    assert len(observed) == 1
    with closing(service.store.connect()) as connection:
        assert not service.store.datasets(connection)
    assert not list((service.store.root / "datasets").glob("*.parquet"))


@pytest.mark.parametrize("payload", (" " * (2097152 + 1), "é" * (1048576 + 1)), ids=("ascii", "utf8"))
def test_excessive_readonly_import_refuses_before_any_source_exposure(tmp_path, payload):
    """Limit complete UTF-8 ingress before interpreting even a source header."""
    service = WorkflowService(tmp_path)
    with closing(service.store.connect()) as connection:
        before = tuple(connection.iterdump())
    with pytest.raises(WorkflowError, match="exceeds 2 MiB"):
        service.import_evidence(payload)
    with closing(service.store.connect()) as connection:
        assert tuple(connection.iterdump()) == before
    assert not (tmp_path / "source-exposures").exists()


def test_valid_readonly_evidence_under_another_handle_refuses_without_repair(tmp_path, actual_bundle):
    """A coherent actual bundle cannot authenticate an altered indexed handle."""
    service = WorkflowService(tmp_path)
    wrong_handle = "wrong-retained-handle"
    body = actual_bundle.to_json()
    with service.store.transaction() as connection:
        connection.execute("INSERT INTO imports VALUES (?,?,?)", (wrong_handle, service.owner, body))
    with pytest.raises(WorkflowError, match="handle/content binding mismatch"):
        service.imported_evidence(wrong_handle)
    with closing(service.store.connect()) as connection:
        assert connection.execute("SELECT payload FROM imports WHERE id=?", (wrong_handle,)).fetchone()[0] == body
        assert not service.store.records(connection)


def test_readonly_import_pool_is_shared_and_exact_retries_retain_one_slot(tmp_path, actual_bundle):
    """Read-only capacity is shared across owners without becoming execution."""
    policy = HostPolicy(schema_version=1, max_jobs=1)
    first = WorkflowService(tmp_path, owner="first", policy=policy)
    second = WorkflowService(tmp_path, owner="second", policy=policy)
    payload = actual_bundle.to_json()
    retained = first.import_evidence(payload)
    assert first.import_evidence(payload) == retained
    with pytest.raises(WorkflowError, match="Read-only import slots/bytes exhausted"):
        second.import_evidence(payload)
    with closing(first.store.connect()) as connection:
        assert connection.execute("SELECT count(*) FROM imports").fetchone()[0] == 1
        assert not first.store.records(connection)
    assert first.imported_evidence(retained.evidence_id).to_json() == payload


def test_excessive_valid_source_metadata_refuses_before_snapshot_acquisition(tmp_path, monkeypatch):
    """Bound even valid native metadata before opening its source snapshot."""
    original = owned_fixture(tmp_path / "prepared")
    observation = original.metadata[0].observation
    expanded = type(observation).model_validate({**observation.model_dump(), "observation_id": "x" * 262144})
    first = type(original.metadata[0]).model_validate({**original.metadata[0].model_dump(), "observation": expanded})
    manifest = DatasetManifest.model_validate({**original.model_dump(), "metadata": (first, *original.metadata[1:])})
    service = WorkflowService(tmp_path / "workspace")

    def forbidden_pin(*_args, **_kwargs):
        raise AssertionError("Oversized metadata reached snapshot acquisition")

    monkeypatch.setattr(workflow_artifacts, "pin_file", forbidden_pin)
    with pytest.raises(WorkflowError, match="metadata exceeds 256 KiB"):
        service.register(manifest, snapshot_root=tmp_path / "prepared")
    with closing(service.store.connect()) as connection:
        assert not service.store.datasets(connection)


def test_actual_protected_inspection_requires_retained_handoff_and_never_reopens_holdout(tmp_path, monkeypatch):
    """Inspect real continued CI evidence; missing metadata is never recreated."""
    service, preview = setup_workflow(
        tmp_path, policies=("Random", "UCB"), seeds=(42,), ucb_exploration=(1.0,), evaluation="protected"
    )
    job = service.execute(ExecutePlan(schema_version=1, request_id="protected-inspection", plan_id=preview.plan_id))
    wait(service, job.job_id, "awaiting_evaluation")
    root = service.store.root / "jobs" / job.job_id
    archived_handoff(service, preview, root, monkeypatch)
    service.evaluate(job.job_id)
    wait(service, job.job_id)
    report = service.report(job.job_id)
    assert report.comparison is not None
    audit = EvaluationAudit(service.store.root / "audit")
    before = audit.history(report.comparison.access.holdout_id)
    page = service.inspect(job.job_id)
    assert page.job.stage == "evaluation"
    assert len(page.coordinates) == len(page.trials) == 2
    assert page.next_offset is None
    handoff = root / "evaluation-handoff.json"
    backup = root / "held-handoff.json"
    handoff.rename(backup)
    with pytest.raises(FileNotFoundError):
        service.inspect(job.job_id)
    assert not handoff.exists()
    backup.rename(handoff)
    again = service.inspect(job.job_id)
    assert again.coordinates == page.coordinates
    assert again.trials == page.trials
    assert audit.history(report.comparison.access.holdout_id) == before
