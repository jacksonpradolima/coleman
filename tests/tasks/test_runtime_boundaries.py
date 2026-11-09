"""Real source-provider compilation and complete-last refusal boundaries.

The shared fixture selects a declared image and detached-start substitution,
plus an explicit successful 1 MiB unit memory sample through trusted WorkerHooks.
No ACK, codec, math, cap, guard predicate or common driver is replaced.
Deliberate publication faults delegate the actual write first.
"""

from __future__ import annotations

import json

import pytest

from coleman.application import task_runtime
from coleman.application.tasks import TaskService
from coleman.core.identity import Reference
from coleman.learning.reference import Hedge
from coleman.tasks.contracts import EncodedValue
from coleman.tasks.protocols import CoordinateContext
from coleman.workflows.contracts import WorkflowError

from .test_service_flow import admit, changed, direct_state, records, run_claimed
from .test_service_flow import service_case as service_case
from .test_service_flow import source as source


def compile_program(item, job, plan=None, profile=None, guard=lambda: None):
    """Invoke the actual public compilation entry with complete known metadata."""
    return task_runtime.prepare_task_program(
        item.service.store,
        job,
        item.service.saved_plan(job.plan_id) if plan is None else plan,
        bootstrap_profile=item.profile if profile is None else profile,
        guard=guard,
    )


@pytest.fixture
def completed_case(service_case):
    item = service_case
    _, queued = admit(item)
    run_claimed(item, queued.job.job_id)
    done = item.service.status(queued.job.job_id).job
    assert done.state == "completed"
    return item, done


def test_compilation_has_concrete_values_and_exact_context_but_creates_no_math_owner(service_case, monkeypatch):
    item = service_case
    _, queued = admit(item)
    plan = item.service.saved_plan(queued.job.plan_id)
    calls = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("prepare constructed a live mathematical owner")

    monkeypatch.setattr(item.source.state.HedgeOwner, "__init__", forbidden)
    program = compile_program(item, queued.job, guard=lambda: calls.append("guard"))
    assert calls == ["guard", "guard", "guard"]
    assert program.context.plan_fingerprint == plan.fingerprint
    assert program.context.source == item.request.source
    assert program.context.implementation == item.image.binding
    assert program.context.compatibility.owner_implementations == (item.image.binding,)
    assert program.coordinates[0].value.to_json() == item.source.sample.inputs().trajectories[0].to_json()
    assert program.coordinates[0].target == plan.coordinates[0].target
    assert program.output_path == program.context.job_root / "complete.json"
    assert program.executor.progress_verifier is not None
    assert (
        program.executor.progress_verifier.reconcile(
            CoordinateContext(
                program.context,
                program.coordinates[0].coordinate_fingerprint,
                program.coordinates[0].reference,
                program.coordinates[0].target,
                program.coordinates[0].artifact_directory,
            )
        )
        is None
    )
    assert not program.context.job_root.exists()


@pytest.mark.parametrize("failure", ("no_profile", "job_join", "profile_join", "guard", "future_input"))
def test_compile_refusals_precede_prepare_reservation_and_state_access(service_case, failure):
    item = service_case
    _, queued = admit(item)
    job = queued.job
    plan = item.service.saved_plan(job.plan_id)
    profile = item.profile

    def guard() -> None:
        if failure == "guard":
            raise ValueError("controlled pre-work cancellation")

    expected = "bootstrap|plan|stage|schema_version|controlled pre-work"
    if failure == "job_join":
        job = changed(job, stage=Reference(id="wrong-stage", version="1"))
    elif failure == "profile_join":
        profile = changed(profile, reference=Reference(id="another-host", version="1"))
    elif failure == "future_input":
        raw = json.loads(plan.inputs.body)
        raw["schema_version"] = 2
        plan = changed(plan, inputs=EncodedValue(codec=plan.inputs.codec, body=json.dumps(raw)))
        job = changed(job, plan_id=plan.fingerprint[7:])
    before = records(item.service)
    item.calls.clear()
    selected_profile = None if failure == "no_profile" else profile
    with pytest.raises(ValueError, match=expected):
        task_runtime.prepare_task_program(
            item.service.store, job, plan, bootstrap_profile=selected_profile, guard=guard
        )
    assert "prepare" not in item.calls
    assert records(item.service) == before
    assert not (item.service.store.root / "jobs" / queued.job.job_id).exists()


def test_lost_coordinate_publication_readback_cannot_complete_or_repeat_math(service_case, monkeypatch):
    item = service_case
    _, queued = admit(item)
    original = item.source.executor.PreparedHedge.publish_coordinate
    accepted = []

    def lose_after_actual_write(prepared, coordinate, value, context):
        original(prepared, coordinate, value, context)
        accepted.append(value)
        (context.artifact_directory / "coordinate-result.json").unlink()

    with monkeypatch.context() as patch:
        patch.setattr(item.source.executor.PreparedHedge, "publish_coordinate", lose_after_actual_write)
        run_claimed(item, queued.job.job_id)
    failed = item.service.status(queued.job.job_id).job
    assert failed.state == "failed"
    assert len(accepted) == 1
    assert accepted[0].checkpoint.learner.to_json() == direct_state(item).to_json()
    assert accepted[0].proof.frontier.position == 3
    complete = item.service.store.root / "jobs" / failed.job_id / "complete.json"
    assert not complete.exists()
    with pytest.raises(WorkflowError, match="no completed report"):
        item.service.report(failed.job_id)
    assert item.service.resume(failed.job_id).job.state == "queued"

    def forbidden(*_args):
        raise AssertionError("accepted panels must not be observed again")

    with monkeypatch.context() as patch:
        patch.setattr(Hedge, "observe", forbidden)
        run_claimed(item, failed.job_id)
    result = item.service.decode_result(failed.job_id)
    assert result.results[0].checkpoint.to_json() == accepted[0].checkpoint.to_json()
    assert result.results[0].proof.observed_head == accepted[0].proof.observed_head
    assert item.service.status(failed.job_id).job.attempts == 2


@pytest.mark.parametrize("damage", ("missing", "binding", "future", "digest"))
def test_completed_status_does_not_authorize_missing_or_incompatible_report(completed_case, damage):
    item, job = completed_case
    complete = item.service.report(job.job_id)
    path = item.service.store.root / "jobs" / job.job_id / "complete.json"
    before = records(item.service)
    if damage == "missing":
        path.unlink()
        error = FileNotFoundError
        match = "complete.json"
    elif damage == "binding":
        altered = changed(complete, source=item.source.recipe.binding("wrong-source", {"version": 1}))
        path.write_text(altered.to_json())
        error = ValueError
        match = "source|plan"
    elif damage == "future":
        raw = json.loads(complete.to_json())
        raw["schema_version"] = 2
        path.write_text(json.dumps(raw))
        error = ValueError
        match = "schema_version"
    else:
        raw = json.loads(complete.result.body)
        raw["schema_version"] = 2
        altered = changed(complete, result=EncodedValue(codec=complete.result.codec, body=json.dumps(raw)))
        path.write_text(altered.to_json())
        error = WorkflowError
        match = "completion binding mismatch"
    observer = TaskService(item.service.store.root, owner=item.service.owner)
    item.calls.clear()
    with pytest.raises(error, match=match):
        observer.report(job.job_id)
    assert item.calls == []
    assert records(item.service) == before


def test_concrete_result_codec_still_refuses_after_valid_outer_report_join(completed_case):
    item, job = completed_case
    completion = item.service.report(job.job_id)
    raw = json.loads(completion.result.body)
    raw["schema_version"] = 2
    altered = changed(completion, result=EncodedValue(codec=completion.result.codec, body=json.dumps(raw)))
    path = item.service.store.root / "jobs" / job.job_id / "complete.json"
    path.write_text(altered.to_json())
    with item.service.store.transaction() as connection:
        item.service.store.save_record(connection, changed(job, result_digest=altered.fingerprint))
    # Deliberately corrupted concrete evidence can still have valid outer bindings.
    assert item.service.report(job.job_id).to_json() == altered.to_json()
    before = records(item.service)
    with pytest.raises(ValueError, match="schema_version"):
        item.service.decode_result(job.job_id)
    assert records(item.service) == before


def test_complete_outer_codec_roundtrip_preserves_original_concrete_bytes(completed_case):
    item, job = completed_case
    completion = item.service.report(job.job_id)
    codec = task_runtime.CompletionCodec()
    assert codec.declaration.maximum_bytes == 16777216
    assert codec.decode(codec.encode(completion)).to_json() == completion.to_json()
    assert codec.validate(completion).result.body == completion.result.body
