"""Negative compatibility, lifecycle and scientific eligibility cases with explicit units."""

import json
from datetime import UTC, datetime

import pytest

from coleman.application.workflow_registry import owned_fixture
from coleman.application.workflow_store import JobRecord
from coleman.application.workflows import WorkflowService
from coleman.budget import BudgetMode
from coleman.results.domain import Measurement
from coleman.workflows import ExperimentPlan, PlanningRequest, ScheduleBudget, WorkflowLimits, make_plan
from coleman.workflows.lifecycle import JobView, NativeRow
from tests.workflows.test_planning import manifest, recorded_runtime


def changed(model, **updates):
    return type(model).model_validate({**model.model_dump(), **updates})


@pytest.mark.parametrize(
    "updates",
    [
        dict(policies=("UCB",)),
        dict(policies=("Random",), ucb_exploration=(1.0,)),
        dict(seeds=(0, 0)),
        dict(policies=("Random", "Random")),
    ],
)
def test_incomplete_baselines_or_irrelevant_axes_reject(updates):
    with pytest.raises(ValueError):
        PlanningRequest.model_validate({"schema_version": 1, "dataset_id": "owned-ci", **updates})


def test_work_and_zip_bounds_before_materialization():
    request = PlanningRequest(
        schema_version=1, dataset_id="owned-ci", mode="zip", budget=ScheduleBudget(values=(0.2, 0.8))
    )
    zip_manifest = manifest()
    zip_runtime = recorded_runtime()
    with pytest.raises(ValueError, match="matching"):
        make_plan(request, zip_manifest, zip_runtime)
    request = PlanningRequest(schema_version=1, dataset_id="owned-ci", limits=WorkflowLimits(max_build_work=1))
    work_manifest = manifest()
    work_runtime = recorded_runtime()
    with pytest.raises(ValueError, match="Worst-case"):
        make_plan(request, work_manifest, work_runtime)
    fixed_time_request = changed(
        request,
        limits=WorkflowLimits(),
        evaluation="protected",
        budget=ScheduleBudget(mode=BudgetMode.FIXED_TIME, unit="s", values=(2.0,)),
    )
    fixed_time_manifest = manifest()
    fixed_time_runtime = recorded_runtime()
    with pytest.raises(ValueError, match="one ratio"):
        make_plan(fixed_time_request, fixed_time_manifest, fixed_time_runtime)


def test_saved_plan_cannot_drop_or_substitute_resolved_coordinates():
    plan = make_plan(PlanningRequest(schema_version=1, dataset_id="owned-ci"), manifest(), recorded_runtime())
    with pytest.raises(ValueError, match="declared dataset"):
        changed(plan, trials=plan.trials[:-1])
    payload = json.loads(plan.to_json())
    payload["trials"][0]["exploration"] = None
    changed_plan_json = json.dumps(payload)
    with pytest.raises(ValueError):
        ExperimentPlan.from_json(changed_plan_json)


def test_protected_admission_clocks_missing_partitions_and_order(tmp_path):
    data = owned_fixture(tmp_path)
    request = PlanningRequest(schema_version=1, dataset_id=data.dataset_id, evaluation="protected")
    descriptive_manifest = changed(data, evidence="descriptive_source")
    descriptive_runtime = recorded_runtime()
    with pytest.raises(ValueError, match="admission"):
        make_plan(request, descriptive_manifest, descriptive_runtime)
    late = changed(
        data.metadata[0],
        observation=changed(data.metadata[0].observation, available_at=datetime(2026, 1, 3, tzinfo=UTC)),
    )
    late_manifest = changed(data, metadata=(late, *data.metadata[1:]))
    late_runtime = recorded_runtime()
    with pytest.raises(ValueError, match="late/unavailable"):
        make_plan(request, late_manifest, late_runtime)
    all_train = tuple(
        changed(
            item,
            observation=changed(
                item.observation,
                event_at=datetime(2026, 1, 1, tzinfo=UTC),
                available_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            ),
        )
        for item in data.metadata
    )
    all_train_manifest = changed(data, metadata=all_train)
    all_train_runtime = recorded_runtime()
    with pytest.raises(ValueError, match="nonempty"):
        make_plan(request, all_train_manifest, all_train_runtime)
    backward = tuple(
        changed(
            item,
            observation=changed(
                item.observation, event_at=other.observation.event_at, available_at=other.observation.available_at
            ),
        )
        for item, other in zip(data.metadata, reversed(data.metadata), strict=True)
    )
    backward_manifest = changed(data, metadata=backward)
    backward_runtime = recorded_runtime()
    with pytest.raises(ValueError, match="chronological"):
        make_plan(request, backward_manifest, backward_runtime)
    with pytest.raises(ValueError, match="every native build"):
        changed(data, metadata=data.metadata[:-1])
    with pytest.raises(ValueError, match="complete build"):
        changed(data, metadata=())
    with pytest.raises(ValueError, match="sorted and unique"):
        changed(data, build_ids=(2, 1, 3))


def test_zero_null_negative_and_nonfinite_native_projection():
    row = NativeRow(
        step=1,
        source_build_id=-7,
        napfd=-0.25,
        apfdc=0.0,
        tests_ran=0,
        tests_not_ran=3,
        modeled_full_seconds=0.0,
        process_cpu_seconds=0.0,
        process_wall_seconds=None,
        ranking=("a", "b", "c"),
        attempt_id="physical-1",
    )
    assert row.napfd == -0.25
    assert row.process_cpu_seconds == 0.0
    assert row.process_wall_seconds is None
    for name, value in (
        ("napfd", float("nan")),
        ("apfdc", float("inf")),
        ("modeled_full_seconds", -1.0),
        ("process_cpu_seconds", -1.0),
        ("tests_ran", None),
    ):
        with pytest.raises(ValueError):
            changed(row, **{name: value})


def test_completion_cannot_be_claimed_without_a_receipt_or_valid_progress():
    eta = Measurement(
        name="remaining_wall_seconds",
        value=None,
        unit="s",
        meaning="remaining-wall-v1",
        direction="none",
        status="unavailable",
    )
    view = JobView(
        schema_version=1,
        job_id="job",
        request_id="request",
        plan_id="plan",
        state="interrupted",
        stage="descriptive",
        completed_trials=0,
        total_trials=1,
        elapsed_seconds=0.0,
        eta_seconds=eta,
        worker_active=False,
    )
    for updates in (dict(state="completed"), dict(completed_trials=2), dict(result_digest="sha256:" + "a" * 64)):
        with pytest.raises(ValueError):
            changed(view, **updates)


def test_corrupt_metadata_identity_and_missing_update_reject(tmp_path):
    service = WorkflowService(tmp_path)
    with service.store.transaction() as connection:
        record = JobRecord(
            schema_version=1,
            job_id="missing",
            owner="local-workspace",
            request_id="key",
            request_digest="sha256:" + "a" * 64,
            plan_id="plan",
            state="queued",
            stage="descriptive",
            created=0.0,
            changed=0.0,
            total_trials=1,
        )
        with pytest.raises(ValueError, match="identity"):
            service.store.save_job(connection, record)
        service.store.save_job(connection, record, create=True)
        connection.execute("UPDATE jobs SET owner='forged'")
    with pytest.raises(ValueError, match="identity"):
        service.status(record.job_id)
