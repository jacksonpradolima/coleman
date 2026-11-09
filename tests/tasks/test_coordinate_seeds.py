"""Frozen coordinate seeds reach real compiled execution and owned callbacks."""

from __future__ import annotations

import random
import time

import pytest

from coleman.application import execution_driver as driver
from coleman.application.execution_program import CoordinateWork
from coleman.application.task_runtime import prepare_task_program
from coleman.learning.reference import Hedge
from coleman.tasks.contracts import TaskCoordinate, TaskFrontier
from coleman.tasks.protocols import CoordinateContext, encode_value

from .test_completed_prefix import _source
from .test_native_boundary_refusals import _admitted as _admitted
from .test_reference_executor import source as source
from .test_service_flow import admit, changed, run_claimed
from .test_service_flow import service_case as service_case


@pytest.mark.parametrize("seed", (None, 0, 7, 2**64 - 1))
def test_compilation_retains_the_declared_coordinate_seed(service_case, seed):
    """The seed remains separate from the body and cannot disappear at compile."""
    item = service_case
    coordinates = tuple(changed(coordinate, seed=seed) for coordinate in item.request.coordinates)
    item.request = changed(item.request, coordinates=coordinates)
    command, view = admit(item)
    plan = item.service.saved_plan(command.plan_id)
    program = prepare_task_program(
        item.service.store, view.job, plan, bootstrap_profile=item.profile, guard=lambda: None
    )
    assert tuple(getattr(work, "seed", None) for work in program.coordinates) == (seed,)
    assert plan.coordinates[0].seed == seed
    assert program.coordinates[0].value.to_json() == item.source.sample.inputs().trajectories[0].to_json()


def _two_trajectories(item, seeds):
    """Construct typed public loss inputs; seeds stay outside trajectory bodies."""
    config = item.source.sample.configuration()
    inputs = _source(item.source, config)
    coordinates = tuple(
        TaskCoordinate(
            reference=trajectory.reference,
            body=encode_value(item.source.recipe.COORDINATE, trajectory),
            seed=seed,
            target=TaskFrontier(
                profile=item.image.recipe.frontier,
                coordinate=trajectory.reference,
                position=len(trajectory.panels),
            ),
        )
        for trajectory, seed in zip(inputs.trajectories, seeds, strict=True)
    )
    item.request = changed(
        item.request,
        source=inputs.binding,
        inputs=encode_value(item.source.recipe.INPUTS, inputs),
        coordinates=coordinates,
        limits=changed(item.request.limits, max_coordinates=2, max_work_units=12),
    )
    return config, inputs


def _observe_callbacks(item, monkeypatch, expected):
    """Record and delegate every real provider callback, without drawing RNG."""
    observed = []

    def observer(name, original, position):
        def delegated(instance, *args, **kwargs):
            context = args[position]
            if name == "advance":
                context = context.coordinate
            selected = expected[context.reference]
            if selected is None:
                assert context.seed is None
            else:
                assert isinstance(context.seed, int)
                assert not isinstance(context.seed, bool)
                assert context.seed == selected
            observed.append((name, context.reference, context.seed, context.coordinate_fingerprint))
            return original(instance, *args, **kwargs)

        return delegated

    positions = {"read_completed": 1, "reconcile": 0, "validate": 1, "advance": 1, "publish_coordinate": 2}
    for name, position in positions.items():
        original = getattr(item.source.executor.PreparedHedge, name)
        monkeypatch.setattr(item.source.executor.PreparedHedge, name, observer(name, original, position))
    return observed


def _interrupt_b_after_actual_ack(item, reference, monkeypatch):
    """Accept A and B's U2 state before the controlled outer response loss."""
    command, queued = admit(item)
    original = driver._Worker.record_progress

    def after_actual_ack(worker, program, context, progress, **kwargs):
        original(worker, program, context, progress, **kwargs)
        if context.reference == reference and progress.frontier.position == 2:
            raise InterruptedError("controlled seeded B interruption after actual U2 ACK")

    with monkeypatch.context() as stop:
        stop.setattr(driver._Worker, "record_progress", after_actual_ack)
        run_claimed(item, queued.job.job_id)
    return command, queued.job.job_id, item.service.status(queued.job.job_id).job


def _assert_direct_report(item, job_id, config, inputs):
    """Compare each real restored complete mathematical state to direct Hedge."""
    report = item.service.decode_result(job_id)
    for result, trajectory in zip(report.results, inputs.trajectories, strict=True):
        direct = Hedge(config.reference, owner=trajectory.reference)
        for panel in trajectory.panels:
            assert direct.observe(panel).status == "accepted"
        assert result.checkpoint.learner.to_json() == direct.export_state().to_json()


def _assert_resumed_callbacks(observed, start, inputs):
    """Require A's cached read and B's acknowledged-head reconciliation."""
    resumed = observed[start:]
    assert any(
        name == "read_completed" and reference == inputs.trajectories[0].reference for name, reference, _, _ in resumed
    )
    assert any(
        name == "reconcile" and reference == inputs.trajectories[1].reference for name, reference, _, _ in resumed
    )


@pytest.mark.parametrize("seeds", ((None, 0), (0, 7), (7, 2**64 - 1)))
def test_distinct_seeds_survive_cached_prefix_and_real_acknowledged_resume(service_case, monkeypatch, seeds):
    """Resume B's actual U2 head while A stays cached and both seeds stay exact.

    The selected Hedge remains deterministic. These observations prove seed
    delivery and ACK continuity, not that this no-RNG learner consumes a seed.
    """
    item = service_case
    config, inputs = _two_trajectories(item, seeds)
    observed = _observe_callbacks(
        item, monkeypatch, dict(zip((trajectory.reference for trajectory in inputs.trajectories), seeds, strict=True))
    )
    ambient_before = random.getstate()
    command, job_id, interrupted = _interrupt_b_after_actual_ack(item, inputs.trajectories[1].reference, monkeypatch)
    fingerprints = {
        coordinate.reference: coordinate.fingerprint
        for coordinate in item.service.saved_plan(command.plan_id).coordinates
    }
    assert interrupted.state == "interrupted"
    assert interrupted.completed_coordinates == 1
    assert interrupted.frontier is not None
    assert interrupted.frontier.position == 2
    resumed_at = len(observed)
    assert item.service.resume(interrupted.job_id).job.state == "queued"
    run_claimed(item, interrupted.job_id)
    done = item.service.status(interrupted.job_id).job
    assert done.state == "completed"
    assert done.completed_coordinates == 2
    assert done.attempts == 3
    assert done.invocations == 2
    assert done.job_id == job_id
    _assert_direct_report(item, done.job_id, config, inputs)
    assert {row[0] for row in observed} == {"read_completed", "advance", "publish_coordinate", "validate", "reconcile"}
    assert all(fingerprint == fingerprints[reference] for _, reference, _, fingerprint in observed)
    _assert_resumed_callbacks(observed, resumed_at, inputs)
    assert random.getstate() == ambient_before
    assert item.service.execute(command).job.to_json() == done.to_json()


def test_original_five_positional_carriers_keep_an_undeclared_seed(service_case):
    """The additive ephemeral field leaves existing constructor calls valid."""
    item = service_case
    command, view = admit(item)
    plan = item.service.saved_plan(command.plan_id)
    before = plan.to_json()
    program = prepare_task_program(
        item.service.store, view.job, plan, bootstrap_profile=item.profile, guard=lambda: None
    )
    selected = program.coordinates[0]
    work = CoordinateWork(
        selected.coordinate_fingerprint,
        selected.reference,
        selected.target,
        selected.value,
        selected.artifact_directory,
    )
    context = CoordinateContext(
        program.context, work.coordinate_fingerprint, work.reference, work.target, work.artifact_directory
    )
    assert work.seed is None
    assert context.seed is None
    assert plan.to_json() == before


def test_native_compilation_exposes_the_original_trial_seed(admitted):
    """The neutral carrier mirrors the native declaration without changing it."""
    from coleman.application.native_executor import prepare_native_program

    service, plan, job = admitted
    before = plan.to_json()
    program = prepare_native_program(service.store, job, plan, deadline=time.monotonic() + 30, guard=lambda: None)
    assert program.coordinates
    assert {work.seed for work in program.coordinates} == {42}
    assert all(work.seed == work.value.seed for work in program.coordinates)
    assert plan.to_json() == before
