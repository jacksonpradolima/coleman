"""Real native preflight and retained-marker refusals, without executing a trial.

Native progress files are weaker legacy checkpoint evidence, never local-state
ACKs. These tests create actual owned files/SQLite metadata and preserve that
distinction; no native mathematical result or restored RNG is manufactured.
"""

from __future__ import annotations

import json
import time
from contextlib import closing
from typing import Any, cast

import pytest

from coleman.application import execution_lifecycle as lifecycle
from coleman.application import native_executor as native
from coleman.application.execution_views import changed
from coleman.application.native_executor import _NativePrepared
from coleman.application.workflow_registry import owned_fixture
from coleman.application.workflow_store import JobRecord
from coleman.application.workflows import WorkflowService
from coleman.artifacts import ArtifactWriter
from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.checkpoint.state import CheckpointPayload
from coleman.tasks.protocols import CommittedProgress, CoordinateContext
from coleman.workflows.contracts import (
    DatasetManifest,
    ExperimentPlan,
    PlanningRequest,
    Trial,
    WorkflowError,
    WorkflowLimits,
)
from coleman.workflows.lifecycle import HostPolicy, WorkflowReport
from coleman.workflows.planning import make_plan
from tests.tasks.test_storage_records import _native, _plans


@pytest.fixture(name="admitted")
def _admitted(tmp_path):
    """Retain an actual source and current native plan, without starting a worker."""
    service = WorkflowService(tmp_path, owner="owner")
    data = service.register(owned_fixture(tmp_path))
    preview = service.plan(
        PlanningRequest(schema_version=1, dataset_id=data.dataset_id, policies=("Random",), seeds=(42,))
    )
    plan = service.saved_plan(preview.plan_id)
    job = changed(_native(plan), state="running", invocations=1)
    with service.store.transaction() as connection:
        service.store.save_record(connection, job, create=True)
    return service, plan, job


def _prepared(admitted) -> tuple[_NativePrepared, Trial, CoordinateContext]:
    """Prepare the real adapter; no source/checkpoint/runtime guard is replaced."""
    service, plan, job = admitted
    program = native.prepare_native_program(
        service.store, job, plan, deadline=time.monotonic() + 30, guard=lambda: None
    )
    work = program.coordinates[0]
    context = CoordinateContext(
        program.context, work.coordinate_fingerprint, work.reference, work.target, work.artifact_directory
    )
    assert isinstance(program.executor, _NativePrepared)
    return program.executor, work.value, context


def _minimum_verifier(admitted, context, trial, minimum):
    """Persist the actual outer minimum separately from the native marker."""
    service, plan, job = admitted
    retained = JobRecord.model_validate(changed(job, current_trial=trial.trial_id, checkpoint_step=minimum))
    with service.store.transaction() as connection:
        service.store.save_record(connection, retained)
    selected = native._NativeContext(service.store.root, context.preparation.job_root, plan, retained, lambda: None)
    return native._NativeProgress(selected)


def _checkpoint(root, trial, *, run="owned", step=1):
    """Publish a genuine native checkpoint and return its existing progress file."""
    directory = root / "native" / trial[7:] / "checkpoints"
    LocalCheckpointStore(str(directory)).save(CheckpointPayload(run_id=run, experiment=1, step=step))
    return directory / run / "progress_ex1.json"


@pytest.mark.parametrize("position", [True, 0, -1, 513, 1.5])
def test_native_marker_rejects_invalid_build_frontier_without_reading_pickle(tmp_path, position):
    """Native work is a strict bounded build ordinal, not a coerced counter."""
    trial = "sha256:" + "a" * 64
    marker = _checkpoint(tmp_path, trial)
    original = marker.read_bytes()
    payload = json.loads(original)
    payload["step_committed"] = position
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkflowError, match="frontier is incompatible"):
        native._marker_position(tmp_path, trial)
    assert marker.read_bytes() != original
    assert (marker.parent / "ckpt_ex1_step1.pkl").is_file()


@pytest.mark.parametrize("field,value", [("run_id", "foreign"), ("experiment", True), ("experiment", 2)])
def test_native_marker_requires_exact_run_and_execution_ownership(tmp_path, field, value):
    """Changing only a retained identity cannot relabel an existing checkpoint."""
    trial = "sha256:" + "b" * 64
    marker = _checkpoint(tmp_path, trial)
    payload = json.loads(marker.read_bytes())
    payload[field] = value
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkflowError, match="exact workflow root/execution"):
        native._marker_position(tmp_path, trial)


def test_native_marker_refuses_ambiguous_and_oversized_retained_progress(tmp_path):
    """Neither multiple real checkpoints nor an excessive marker selects work."""
    trial = "sha256:" + "c" * 64
    marker = _checkpoint(tmp_path, trial)
    original = marker.read_bytes()
    marker.write_bytes(original + b" " * 16385)
    with pytest.raises(WorkflowError, match="16 KiB"):
        native._marker_position(tmp_path, trial)
    marker.write_bytes(original)
    _checkpoint(tmp_path, trial, run="second-owned")
    with pytest.raises(WorkflowError, match="implicit reset is forbidden"):
        native._marker_position(tmp_path, trial)


def test_changed_registered_source_refuses_before_native_input_publication(admitted):
    """A valid frozen plan must not copy changed original source bytes."""
    service, plan, job = admitted
    source = service.store.root / "datasets" / f"{plan.dataset.sha256[7:]}.parquet"
    source.write_bytes(source.read_bytes() + b"changed after registration")
    destination = service.store.root / "jobs" / job.job_id
    deadline = time.monotonic() + 30
    with pytest.raises(WorkflowError, match="Registered dataset changed"):
        native.prepare_native_program(service.store, job, plan, deadline=deadline, guard=lambda: None)
    assert not (destination / "inputs").exists()
    assert not (destination / "native").exists()


@pytest.mark.parametrize("failure", ["threads", "deadline", "historical_runtime"])
def test_native_prepare_rejects_incompatible_context_before_source_or_state(admitted, failure):
    """Current actual runtime checks precede source copy and native RNG restore."""
    service, plan, job = admitted
    deadline = time.monotonic() + 30
    if failure == "threads":
        plan = ExperimentPlan.model_validate(plan.model_copy(update={"native_threads": plan.native_threads + 1}))
        expected = "thread profile"
    elif failure == "deadline":
        deadline = 0.0
        expected = "monotonic deadline"
    else:
        plan = _plans()[0]
        expected = "pinned v4"
    with pytest.raises(ValueError, match=expected):
        native.prepare_native_program(service.store, job, plan, deadline=deadline, guard=lambda: None)
    assert not (service.store.root / "jobs" / job.job_id / "native").exists()
    assert not (service.store.root / "jobs" / job.job_id / "inputs").exists()


def test_native_progress_requires_real_matching_marker_and_keeps_its_minimum(admitted):
    """A declared integer cannot replace actual marker ownership or rewind it."""
    prepared, trial, context = _prepared(admitted)
    root = context.preparation.job_root
    _checkpoint(root, trial.trial_id)
    actual = prepared.progress_verifier.reconcile(context)
    assert actual is not None
    prepared.progress_verifier.validate(actual, context)
    wrong = CommittedProgress[int](actual.plan_fingerprint, actual.coordinate_fingerprint, actual.frontier, True)
    with pytest.raises(WorkflowError, match="exact build ordinal"):
        prepared.progress_verifier.validate(wrong, context)
    later = actual.frontier.model_copy(update={"position": 2})
    wrong = CommittedProgress(actual.plan_fingerprint, actual.coordinate_fingerprint, later, 2)
    with pytest.raises(WorkflowError, match="actual committed marker"):
        prepared.progress_verifier.validate(wrong, context)
    verifier = _minimum_verifier(admitted, context, trial, 2)
    with pytest.raises(WorkflowError, match="older than"):
        verifier.reconcile(context)
    assert native._marker_position(root, trial.trial_id) == 1


def test_missing_native_active_minimum_and_unpublished_coordinate_never_become_complete(admitted):
    """Missing partial evidence cannot be promoted into a completed coordinate."""
    prepared, trial, context = _prepared(admitted)
    assert prepared.read_completed(trial, context) is None
    verifier = _minimum_verifier(admitted, context, trial, 1)
    with pytest.raises(WorkflowError, match="implicit reset"):
        verifier.reconcile(context)
    # The absent retained publication is checked before a candidate value is used.
    missing = cast(Any, None)
    with pytest.raises(WorkflowError, match="Published native coordinate"):
        prepared.publish_coordinate(trial, missing, context)
    assert not (context.artifact_directory / "complete.json").exists()


def test_protected_prefix_requires_explicit_temporal_declaration(admitted):
    """The adapter cannot invent a protected split from native build ordinals."""
    _, plan, _ = admitted
    dataset = DatasetManifest.model_validate(plan.dataset.model_copy(update={"temporal_plan": None}))
    missing = make_plan(plan.request, dataset, plan.runtime, native_threads=plan.native_threads)
    with pytest.raises(WorkflowError, match="temporal plan"):
        native._prefix_steps(missing)


@pytest.mark.parametrize("case", ["changed_bytes", "dimensions", "replacement_id", "registry_full"])
def test_native_source_registration_preserves_original_bytes_and_catalogue(tmp_path, case):
    """Real registry admission rejects source or identity changes atomically."""
    policy = HostPolicy(schema_version=1, max_datasets=1)
    service = WorkflowService(tmp_path, policy=policy)
    original = service.register(owned_fixture(tmp_path))
    proposed = original
    if case == "changed_bytes":
        source = tmp_path / "datasets" / f"{original.sha256[7:]}.parquet"
        source.write_bytes(source.read_bytes() + b"changed")
        expected = "input bytes"
    elif case == "dimensions":
        proposed = DatasetManifest.model_validate(original.model_copy(update={"rows": original.rows + 1}))
        expected = "dimensions"
    elif case == "replacement_id":
        proposed = DatasetManifest.model_validate(
            original.model_copy(update={"assumptions": ("Different declaration.",)})
        )
        expected = "silently replaced"
    else:
        proposed = DatasetManifest.model_validate(original.model_copy(update={"dataset_id": "another-source"}))
        expected = "registry is full"
    with pytest.raises(WorkflowError, match=expected):
        service.register(proposed)
    with closing(service.store.connect()) as connection:
        assert service.store.datasets(connection) == (original,)


def test_native_facade_rejects_unregistered_planning_and_expired_or_unknown_evidence(tmp_path):
    """Read-only facade errors neither schedule work nor create native state."""
    service = WorkflowService(tmp_path, owner="owner")
    request = PlanningRequest(schema_version=1, dataset_id="not-registered")
    with pytest.raises(KeyError, match="not registered"):
        service.plan(request)
    native_plan = _plans()[0]
    lifecycle.save_plan(service.store, native_plan, service.owner)
    job = changed(_native(native_plan), state="expired")
    with service.store.transaction() as connection:
        service.store.save_record(connection, job, create=True)
    with pytest.raises(WorkflowError, match="Expired workflow evidence"):
        service.inspect(job.job_id)
    with pytest.raises(KeyError, match="read-only evidence"):
        service.imported_evidence("a" * 64)
    assert not (tmp_path / "jobs").exists()


def test_native_planning_cannot_exceed_immutable_host_caps(tmp_path):
    """Reject a valid requested worker allowance above the explicit host ceiling."""
    caps = WorkflowLimits(max_attempts=1)
    service = WorkflowService(tmp_path, policy=HostPolicy(schema_version=1, caps=caps))
    request = PlanningRequest(schema_version=1, dataset_id="not-opened", limits=WorkflowLimits(max_attempts=2))
    with pytest.raises(WorkflowError, match="host cap: max_attempts"):
        service.plan(request)


def test_native_planning_requires_worker_to_fit_shared_memory_pool(tmp_path):
    """A worker allowed by its individual cap must also fit the actual host pool."""
    service = WorkflowService(tmp_path, policy=HostPolicy(schema_version=1, max_active_memory_bytes=67108864))
    request = PlanningRequest(schema_version=1, dataset_id="not-opened")
    with pytest.raises(WorkflowError, match="aggregate active memory"):
        service.plan(request)


def test_native_resume_rechecks_frozen_thread_context_before_requeue(admitted):
    """Failed retained work cannot resume under an incompatible thread profile."""
    service, original, job = admitted
    incompatible = make_plan(
        original.request, original.dataset, original.runtime, native_threads=original.native_threads + 1
    )
    lifecycle.save_plan(service.store, incompatible, service.owner)
    retained = changed(
        job,
        job_id="incompatible-thread-job",
        request_id="incompatible-thread-request",
        plan_id=incompatible.fingerprint[7:],
        state="failed",
    )
    with service.store.transaction() as connection:
        service.store.save_record(connection, retained, create=True)
    with pytest.raises(WorkflowError, match="Incompatible runtime"):
        service.resume(retained.job_id)
    with closing(service.store.connect()) as connection:
        assert service.store.records(connection) == (job, retained)
    assert not (service.store.root / "jobs" / retained.job_id).exists()


def test_protected_native_boundary_requires_retained_handoff_before_loading_coordinates(admitted):
    """Evaluation metadata without its real frozen handoff does not expose tests."""
    service, original, job = admitted
    request = PlanningRequest(schema_version=1, dataset_id=original.dataset.dataset_id, evaluation="protected")
    preview = service.plan(request)
    protected = service.saved_plan(preview.plan_id)
    claimed = JobRecord.model_validate(
        changed(job, plan_id=preview.plan_id, stage="evaluation", total_trials=len(protected.trials))
    )
    deadline = time.monotonic() + 30
    with pytest.raises(FileNotFoundError):
        native.prepare_native_program(service.store, claimed, protected, deadline=deadline, guard=lambda: None)
    assert not (service.store.root / "jobs" / job.job_id / "native").exists()
    assert not (service.store.root / "jobs" / job.job_id / "complete.json").exists()


def test_protected_evaluation_requires_remaining_actual_invocation_allowance(admitted):
    """An awaiting label cannot bypass the frozen worker-start allowance."""
    service, original, job = admitted
    preview = service.plan(
        PlanningRequest(schema_version=1, dataset_id=original.dataset.dataset_id, evaluation="protected")
    )
    plan = service.saved_plan(preview.plan_id)
    retained = changed(
        job,
        job_id="exhausted-protected-job",
        request_id="exhausted-protected-request",
        plan_id=preview.plan_id,
        stage="tuning",
        state="awaiting_evaluation",
        total_trials=len(plan.trials),
        invocations=plan.request.limits.max_attempts,
    )
    with service.store.transaction() as connection:
        service.store.save_record(connection, retained, create=True)
    with pytest.raises(WorkflowError, match="exhausted invocation allowance"):
        service.evaluate(retained.job_id)
    with closing(service.store.connect()) as connection:
        assert service.store.records(connection) == (job, retained)
    assert not (service.store.root / "jobs" / retained.job_id / "evaluation-handoff.json").exists()


def test_complete_looking_report_with_foreign_plan_is_not_readable_native_completion(admitted):
    """A correct file digest and operational label cannot authenticate another plan."""
    service, _, job = admitted
    # This empty authored envelope is deliberately not scientific evidence.
    report = WorkflowReport(
        schema_version=1,
        plan_fingerprint="sha256:" + "f" * 64,
        trials=(),
        selected_configuration=None,
        interpretation="descriptive_only",
        assumptions=("Deliberately foreign plan, not an actual execution.",),
    )
    path = service.store.root / "jobs" / job.job_id / "complete.json"
    ArtifactWriter(path.parent).write_immutable(path, report.to_json())
    retained = changed(job, state="completed", completed_trials=job.total_trials, result_digest=report.fingerprint)
    with service.store.transaction() as connection:
        service.store.save_record(connection, retained)
    with pytest.raises(WorkflowError, match="completion binding mismatch"):
        service.report(job.job_id)
    assert path.read_text(encoding="utf-8") == report.to_json()
