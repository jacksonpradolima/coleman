"""Real two-Hedge recovery preserves independently retained prefix minima.

The mathematical owners, local state ACKs, common coordinate driver, attempt
files and SQL metadata are real. The identity below is an explicitly authored
source-compatibility fixture, never an installed-capture or detached-worker
claim. The controlled interruption delegates actual progress verification and
metadata persistence before stopping at B U2. No source or capture mock is used.
The trusted WorkerHooks supplies a successful 1 MiB unit RSS observation so the
unrelated pytest process's lifetime peak is not charged to one worker. Actual
resource checks, output sampling and the declared 1 GiB cap remain unchanged.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from coleman.application import execution_driver as driver
from coleman.application import execution_views as views
from coleman.application.execution_program import CoordinateWork, ExecutionProgram
from coleman.application.workflow_store import WorkflowStore
from coleman.core.primitives import fingerprint
from coleman.learning.reference import Hedge
from coleman.state.storage_contracts import ExecutionCompatibility
from coleman.tasks.contracts import TaskCoordinate, TaskFrontier, TaskJobRecord, TaskLimits, TaskPlan
from coleman.tasks.identity import (
    ApprovedExecutor,
    DistributionInventory,
    ExecutorImplementationIdentity,
    HostBootstrapProfile,
    InstalledFile,
)
from coleman.tasks.protocols import CoordinateContext, PreparationContext, encode_value
from coleman.workflows.contracts import WorkflowError

from .test_reference_executor import ROOT
from .test_reference_executor import source as source


def _declared_image(example):
    template = TaskPlan.from_json((ROOT / "tests/fixtures/tasks/v1/plan.json").read_bytes())
    files = (
        ("coleman", ROOT / "coleman/__init__.py", "coleman/__init__.py"),
        (
            "coleman-typed-hedge-reference",
            ROOT / "examples/typed_executor/provider/src/coleman_typed_hedge/__init__.py",
            "coleman_typed_hedge/__init__.py",
        ),
        (
            "coleman-typed-hedge-support",
            ROOT / "examples/typed_executor/support/src/coleman_typed_hedge_support/__init__.py",
            "coleman_typed_hedge_support/__init__.py",
        ),
    )
    return ExecutorImplementationIdentity(
        contract="coleman-tasks-v1",
        schema_version=1,
        artifact_type="executor_implementation",
        reference=example.recipe.reference("declared-source-prefix-image"),
        recipe=example.package.recipe(),
        runtime=template.implementation.runtime,
        distributions=tuple(
            DistributionInventory(
                name=name,
                version="4.20.0" if name == "coleman" else "1.0.0",
                extras=(),
                metadata_fingerprint=fingerprint({"declared-unit-fixture-only": name}),
            )
            for name, _, _ in files
        ),
        files=tuple(
            InstalledFile(
                distribution=name,
                path=relative,
                size_bytes=path.stat().st_size,
                sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for name, path, relative in files
        ),
    )


def _source(example, config):
    original = example.sample.inputs(config)
    return example.contracts.HedgeInputs.model_validate(
        {
            **original.model_dump(),
            "trajectories": tuple(
                example.contracts.HedgeTrajectory(
                    contract="coleman-typed-hedge-reference-v1",
                    schema_version=1,
                    reference=owner,
                    initial_state=Hedge(config.reference, owner=owner).export_state(),
                    panels=original.trajectories[0].panels,
                )
                for owner in (example.recipe.reference("trajectory-a"), example.recipe.reference("trajectory-b"))
            ),
        }
    )


def _case(example, tmp_path):
    config = example.sample.configuration()
    inputs = _source(example, config)
    image = _declared_image(example)
    profile = HostBootstrapProfile(
        contract="coleman-tasks-v1",
        schema_version=1,
        artifact_type="host_bootstrap",
        reference=example.recipe.reference("declared-prefix-host"),
        executors=(ApprovedExecutor(recipe=image.recipe, implementation=image.binding),),
    )
    coordinates = tuple(
        TaskCoordinate(
            reference=item.reference,
            body=encode_value(example.recipe.COORDINATE, item),
            seed=None,
            target=TaskFrontier(profile=image.recipe.frontier, coordinate=item.reference, position=len(item.panels)),
        )
        for item in inputs.trajectories
    )
    plan = TaskPlan(
        contract="coleman-tasks-v1",
        schema_version=1,
        artifact_type="plan",
        executor=image.recipe.binding,
        implementation=image,
        bootstrap=profile.binding,
        configuration=encode_value(example.recipe.CONFIGURATION, config),
        inputs=encode_value(example.recipe.INPUTS, inputs),
        source=inputs.binding,
        objective=example.recipe.binding("constructed-prefix-loss", {"unit": "1", "role": "loss"}),
        protocol=None,
        stage=example.recipe.reference("constructed-prefix"),
        coordinates=coordinates,
        limits=TaskLimits(
            max_coordinates=2, max_work_units=12, max_attempts=2, max_invocations=3, attempt_seconds=120.0
        ),
    )
    store = WorkflowStore(tmp_path)
    now = time.time()
    job = TaskJobRecord(
        contract="coleman-tasks-v1",
        schema_version=1,
        artifact_type="job",
        job_id="actual-prefix-job",
        owner="owner",
        request_id="actual-prefix-request",
        request_digest=fingerprint({"plan": plan.fingerprint}),
        plan_id=plan.fingerprint[7:],
        state="running",
        stage=plan.stage,
        created=now,
        changed=now,
        total_coordinates=2,
        invocations=1,
    )
    with store.transaction() as connection:
        connection.execute("INSERT INTO plans VALUES (?,?,?)", (job.plan_id, job.owner, plan.to_json()))
        store.save_record(connection, job, create=True)
    context = PreparationContext(
        plan.fingerprint,
        inputs.binding,
        image.binding,
        ExecutionCompatibility(schema_version=1, runtime=image.runtime, owner_implementations=(image.binding,)),
        store.root / "jobs" / job.job_id,
    )
    prepared = example.package.bootstrap().prepare(config, inputs, context)
    works = tuple(
        CoordinateWork(
            declaration.fingerprint,
            declaration.reference,
            declaration.target,
            trajectory,
            context.job_root / "evidence" / declaration.fingerprint[7:],
        )
        for declaration, trajectory in zip(coordinates, inputs.trajectories, strict=True)
    )
    program = ExecutionProgram(
        context,
        works,
        prepared,
        example.recipe.FINAL_RESULT,
        lambda value: value.fingerprint,
        context.job_root / "unused-complete.json",
    )
    worker = driver._Worker(
        store, job, plan, time.monotonic() + 120.0, driver.WorkerHooks(peak_rss_mib=lambda: 1.0), None
    )
    return SimpleNamespace(
        config=config, inputs=inputs, plan=plan, store=store, context=context, program=program, worker=worker
    )


def _context(item, work):
    return CoordinateContext(
        item.context, work.coordinate_fingerprint, work.reference, work.target, work.artifact_directory
    )


def _snapshot(example, item, index):
    work = item.program.coordinates[index]
    layout = example.state.Layout(item.config, work.value, _context(item, work))
    store = layout.open()
    assert store is not None
    return layout.checkpoint(store), store.read_head()


def _interrupt_after_b_ack(example, item, monkeypatch):
    original = driver._Worker.record_progress
    stopped = []

    def after_actual_progress(worker, program, context, progress, **kwargs):
        original(worker, program, context, progress, **kwargs)
        if context.reference == item.program.coordinates[1].reference and progress.frontier.position == 2:
            stopped.append(progress.proof)
            raise InterruptedError("Controlled stop after actual B ACK and outer U2")

    with monkeypatch.context() as patch:
        patch.setattr(driver._Worker, "record_progress", after_actual_progress)
        with pytest.raises(InterruptedError, match="after actual B ACK"):
            item.worker.drive_coordinates(item.program)
    assert len(stopped) == 1
    retained = driver.read_job(item.store, item.worker.job.job_id)
    assert isinstance(retained, TaskJobRecord)
    assert retained.completed_coordinates == 1
    assert retained.current_coordinate == item.program.coordinates[1].coordinate_fingerprint
    assert retained.frontier is not None
    assert retained.frontier.position == 2
    first, first_head = _snapshot(example, item, 0)
    second, second_head = _snapshot(example, item, 1)
    assert first.learner.payload.update_count == 3
    assert second.learner.payload.update_count == 2
    assert first_head.generation == 2
    assert second_head.generation == 1
    assert stopped[0].observed_head == second_head
    assert views.retained_attempts(item.context.job_root, item.plan, job_id=retained.job_id) == 2
    return retained, first, first_head, second, second_head


def _restart(example, item):
    with item.store.transaction() as connection:
        current = item.store.records(connection)[0]
        job = views.changed(current, invocations=current.invocations + 1)
        item.store.save_record(connection, job)
    prepared = example.package.bootstrap().prepare(item.config, item.inputs, item.context)
    program = replace(item.program, executor=prepared)
    worker = driver._Worker(
        item.store, job, item.plan, time.monotonic() + 120.0, driver.WorkerHooks(peak_rss_mib=lambda: 1.0), None
    )
    return worker, program, job


def _observe_actual_calls(example, monkeypatch):
    calls = SimpleNamespace(advance=[], factory=[], observe=[])
    advance = example.executor.PreparedHedge.advance
    factory = example.state.HedgeOwner.__init__
    observe = Hedge.observe

    def actual_advance(prepared, coordinate, *args):
        calls.advance.append(coordinate.reference)
        return advance(prepared, coordinate, *args)

    def actual_factory(owner, initial):
        calls.factory.append(initial.binding.owner)
        factory(owner, initial)

    def actual_observe(learner, panel):
        calls.observe.append(learner.binding.owner)
        return observe(learner, panel)

    monkeypatch.setattr(example.executor.PreparedHedge, "advance", actual_advance)
    monkeypatch.setattr(example.state.HedgeOwner, "__init__", actual_factory)
    monkeypatch.setattr(Hedge, "observe", actual_observe)
    return calls


@pytest.mark.parametrize("missing", [0, 1], ids=["completed-prefix-a", "active-frontier-b"])
def test_missing_acknowledged_coordinate_refuses_before_fresh_work_and_preserves_independent_minimum(
    source,
    tmp_path,
    monkeypatch,
    missing,
):
    item = _case(source, tmp_path)
    _, a, a_head, b, b_head = _interrupt_after_b_ack(source, item, monkeypatch)
    removed = item.program.coordinates[missing].artifact_directory
    shutil.rmtree(removed)  # Only this coordinate's subtree; outer metadata and both attempt ledgers remain.
    worker, program, expected = _restart(source, item)
    calls = _observe_actual_calls(source, monkeypatch)
    refusal = "completed task coordinate evidence is missing" if missing == 0 else "acknowledged task state is missing"
    with pytest.raises(WorkflowError, match=refusal):
        worker.drive_coordinates(program)
    assert calls.advance == []
    assert calls.factory == []
    assert calls.observe == []
    assert not removed.exists()
    assert driver.read_job(item.store, expected.job_id).to_json() == expected.to_json()
    assert views.retained_attempts(item.context.job_root, item.plan, job_id=expected.job_id) == 2
    surviving, head = _snapshot(source, item, 1 - missing)
    assert surviving.to_json() == (b if missing == 0 else a).to_json()
    assert head == (b_head if missing == 0 else a_head)


def test_valid_cached_prefix_preserves_active_frontier_and_continues_only_the_remaining_original_panel(
    source,
    tmp_path,
    monkeypatch,
):
    item = _case(source, tmp_path)
    expected = []
    for trajectory in item.inputs.trajectories:
        direct = Hedge(item.config.reference, owner=trajectory.reference)
        for panel in trajectory.panels:
            assert direct.observe(panel).status == "accepted"
        expected.append(direct.export_state().to_json())
    _, a, a_head, _, _ = _interrupt_after_b_ack(source, item, monkeypatch)
    worker, program, _ = _restart(source, item)
    calls = _observe_actual_calls(source, monkeypatch)
    before_reconcile = []
    reconcile = driver._Worker._reconcile

    def inspect_before_actual_reconcile(actual_worker, actual_program, context):
        retained = driver.read_job(item.store, actual_worker.job.job_id)
        assert isinstance(retained, TaskJobRecord)
        assert retained.current_coordinate == item.program.coordinates[1].coordinate_fingerprint
        assert retained.frontier is not None
        before_reconcile.append(retained.frontier.position)
        return reconcile(actual_worker, actual_program, context)

    monkeypatch.setattr(driver._Worker, "_reconcile", inspect_before_actual_reconcile)
    results = worker.drive_coordinates(program)
    second_owner = item.program.coordinates[1].reference
    assert calls.advance == [second_owner]
    assert calls.observe == [second_owner]
    assert calls.factory == [second_owner]
    assert before_reconcile == [2]
    assert results[0].checkpoint.to_json() == a.to_json()
    assert _snapshot(source, item, 0)[1] == a_head
    for result, direct_state in zip(results, expected, strict=True):
        assert result.checkpoint.learner.to_json() == direct_state
        assert result.proof.frontier.position == 3
        assert result.proof.observed_head.generation == 2
    final = driver.read_job(item.store, item.worker.job.job_id)
    assert isinstance(final, TaskJobRecord)
    assert final.completed_coordinates == 2
    assert final.current_coordinate is None
    assert final.frontier is None
    assert final.state == "running"  # This test exercises the actual coordinate loop, not final job publication.
    assert views.retained_attempts(item.context.job_root, item.plan, job_id=final.job_id) == 3
