"""Optional scopes preserve the concrete host and the one real metadata driver.

The small proof is a codec-boundary fixture, not a claim of a domain storage ACK.
Real domain continuation and concurrent store tests belong to the provider.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Annotated, Any, Literal, cast

import pytest
from pydantic import Field

from coleman.application import execution_driver as driver
from coleman.application import execution_views as views
from coleman.application.execution_program import CoordinateWork, ExecutionProgram
from coleman.application.task_runtime import _Prepared, _progress, _require_prepared
from coleman.application.workflow_store import WorkflowStore
from coleman.core.base import VersionedContract
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.storage_contracts import ExecutionCompatibility
from coleman.tasks.contracts import TaskJobRecord, TaskPlan
from coleman.tasks.identity import CodecDeclaration
from coleman.tasks.protocols import (
    AttemptContext,
    CommittedProgress,
    CoordinateContext,
    PreparationContext,
    coordinate_operation,
)

FIXTURES = Path(__file__).parents[1] / "fixtures/tasks/v1"
PROFILE = Reference(id="coleman-coordinate-operation", version="1")


class _Value(VersionedContract):
    """Keep version, signed zero, units and nonfinite admission observable."""

    value: Annotated[float, Field(strict=True, allow_inf_nan=False)]
    unit: Literal["seconds"] = "seconds"


@dataclass(frozen=True)
class _Codec:
    @property
    def declaration(self):
        schema = fingerprint(_Value.model_json_schema())
        return CodecDeclaration(
            binding=DefinitionBinding(
                reference=Reference(id="coordinate-scope-fixture", version="1"), definition_fingerprint=schema
            ),
            schema_fingerprint=schema,
            maximum_bytes=1024,
        )

    def validate(self, value, /):
        return _Value.model_validate(value)

    def encode(self, value, /):
        return self.validate(value).to_json()

    def decode(self, body, /):
        return _Value.from_json(body)


class _Verifier:
    def __init__(self, events):
        self.events = events

    def validate(self, progress, context, /):
        assert isinstance(progress.proof, _Value)
        self.events.append(("verify", context.reference.id))

    def reconcile(self, context, /):
        self.events.append(("reconcile", context.reference.id))
        return None


class _Operations:
    def __init__(self, events):
        self.events = events
        self.boundary = SimpleNamespace(produce=lambda load: load())
        self.progress_verifier = _Verifier(events)
        self.result = None
        self.callback = None

    def read_completed(self, coordinate, context, /):
        self.events.append(("read", context.reference.id))
        return self.result

    def advance(self, coordinate, context, committed, /):
        self.events.append(("advance", context.coordinate.reference.id))
        self.callback = committed
        context.guard()
        value = _Value(schema_version=1, value=0.0)
        committed(
            CommittedProgress(
                context.coordinate.preparation.plan_fingerprint,
                context.coordinate.coordinate_fingerprint,
                context.coordinate.target,
                value,
            )
        )
        self.result = value
        return value

    def publish_coordinate(self, coordinate, value, context, /):
        self.events.append(("publish", context.reference.id))
        (context.artifact_directory / "publication.txt").parent.mkdir(parents=True, exist_ok=True)
        (context.artifact_directory / "publication.txt").write_text(value.to_json())


class _Handler:
    profile = PROFILE

    def __init__(self, events):
        self.events = events
        self.children = []
        self.factory = _Operations

    @contextmanager
    def open(self, coordinate, context, /):
        self.events.append(("enter", context.reference.id))
        selected = self.factory(self.events)
        self.children.append(selected)
        try:
            yield selected
        finally:
            self.events.append(("exit", context.reference.id))


def _values(tmp_path):
    plan = TaskPlan.from_json((FIXTURES / "plan.json").read_bytes())
    original = TaskJobRecord.from_json((FIXTURES / "job.json").read_bytes())
    job = TaskJobRecord.model_validate(
        {**original.model_dump(), "state": "running", "current_coordinate": None, "frontier": None, "attempts": 0}
    )
    preparation = PreparationContext(
        plan.fingerprint,
        plan.source,
        plan.implementation.binding,
        ExecutionCompatibility(
            schema_version=1,
            runtime=plan.implementation.runtime,
            owner_implementations=(plan.implementation.binding,),
        ),
        tmp_path / "jobs" / job.job_id,
    )
    work = plan.coordinates[0]
    context = CoordinateContext(
        preparation, work.fingerprint, work.reference, work.target, preparation.job_root / work.reference.id
    )
    return plan, job, context


def _wrapped(selected):
    return _Prepared(selected, _Codec(), selected.boundary, _progress(selected, _Codec()), _Codec())


def _selected():
    events = []
    selected = _Operations(events)
    cast(Any, selected).coordinate_operation = _Handler(events)
    return cast(Any, selected)


def _scope(selected, context):
    capability = coordinate_operation(_wrapped(selected))
    assert capability is not None
    return capability.open("coordinate", context)


@pytest.mark.parametrize("explicit_none", [False, True])
def test_absent_or_none_keeps_same_program_and_no_scope_callbacks(explicit_none, tmp_path):
    _, _, context = _values(tmp_path)
    selected = _Operations([])
    if explicit_none:
        cast(Any, selected).coordinate_operation = None
    wrapped = _wrapped(selected)
    program = SimpleNamespace(executor=wrapped)
    work = SimpleNamespace(value="coordinate")
    with driver._coordinate_scope(cast(Any, program), cast(Any, work), context) as actual:
        assert actual is program
        assert actual.executor is wrapped
    assert selected.events == []


@pytest.mark.parametrize(
    "handler",
    [
        SimpleNamespace(profile=Reference(id="other", version="1"), open=lambda *_: None),
        SimpleNamespace(profile=Reference(id=PROFILE.id, version="2"), open=lambda *_: None),
        SimpleNamespace(profile=None, open=lambda *_: None),
        SimpleNamespace(profile=Reference.model_construct(id=PROFILE.id, version=True), open=lambda *_: None),
        SimpleNamespace(profile=PROFILE),
        SimpleNamespace(profile=PROFILE, open=0),
    ],
    ids=["other", "future", "absent-profile", "unchecked", "missing-open", "not-callable"],
)
def test_unknown_or_malformed_profile_refuses_before_scope_or_reservation(handler, tmp_path):
    plan, _, context = _values(tmp_path)
    selected = _selected()
    selected.coordinate_operation = handler
    with pytest.raises((ValueError, TypeError)):
        _require_prepared(selected, plan)
    wrapped = _wrapped(selected)
    with pytest.raises((ValueError, TypeError)):
        coordinate_operation(wrapped)
    assert selected.events == []
    assert not context.artifact_directory.exists()


def test_profile_is_rechecked_before_open_and_after_preparation(tmp_path):
    plan, _, context = _values(tmp_path)
    selected = _selected()
    _require_prepared(selected, plan)
    selected.coordinate_operation.profile = Reference(id=PROFILE.id, version="2")
    with pytest.raises(ValueError, match="profile"):
        _scope(selected, context)
    assert selected.events == []


@pytest.mark.parametrize("damage", ["missing-operation", "missing-verifier", "bad-verifier", "bad-publication"])
def test_acquired_shape_refuses_and_closes_before_any_work(damage, tmp_path):
    _, _, context = _values(tmp_path)
    selected = _selected()

    def invalid(events):
        child = _Operations(events)
        if damage == "missing-operation":
            cast(Any, child).advance = None
        elif damage == "missing-verifier":
            child.progress_verifier = None
        elif damage == "bad-verifier":
            child.progress_verifier = SimpleNamespace(validate=None, reconcile=None)
        else:
            cast(Any, child).publication_readback = SimpleNamespace(profile=Reference(id="future", version="2"))
        return child

    selected.coordinate_operation.factory = invalid
    with pytest.raises((ValueError, TypeError)), _scope(selected, context):
        pytest.fail("Invalid acquired executor reached the common driver")
    assert [event[0] for event in selected.events] == ["enter", "exit"]


@pytest.mark.parametrize("damage", ["future", "unit", "nonfinite"])
def test_scope_retains_same_result_and_proof_codecs_before_selected_verifier(damage, tmp_path):
    _, _, context = _values(tmp_path)
    selected = _selected()
    changes = {"future": {"schema_version": 2}, "unit": {"unit": "milliseconds"}, "nonfinite": {"value": float("inf")}}
    damaged = _Value(schema_version=1, value=0.0).model_copy(update=changes[damage])
    with _scope(selected, context) as actual:
        selected.coordinate_operation.children[-1].result = damaged
        with pytest.raises(ValueError):
            actual.read_completed("coordinate", context)
        verifier = actual.progress_verifier
        assert verifier is not None
        proof = CommittedProgress(
            context.preparation.plan_fingerprint, context.coordinate_fingerprint, context.target, damaged
        )
        with pytest.raises(ValueError):
            verifier.validate(proof, context)
    assert not any(event[0] == "verify" for event in selected.events)


def test_scoped_operations_and_captured_verifier_refuse_after_close(tmp_path):
    _, _, context = _values(tmp_path)
    selected = _selected()
    with _scope(selected, context) as actual:
        verifier = actual.progress_verifier
        assert verifier is not None
        assert actual.read_completed("coordinate", context) is None
    proof = CommittedProgress(
        context.preparation.plan_fingerprint,
        context.coordinate_fingerprint,
        context.target,
        _Value(schema_version=1, value=0.0),
    )
    attempt = AttemptContext(context, "fixture-reservation", 1, time.monotonic() + 1, lambda: None)
    calls = [
        lambda: actual.read_completed("coordinate", context),
        lambda: actual.advance("coordinate", attempt, lambda _: None),
        lambda: actual.publish_coordinate("coordinate", proof.proof, context),
        lambda: verifier.validate(proof, context),
        lambda: verifier.reconcile(context),
    ]
    before = list(selected.events)
    for call in calls:
        with pytest.raises(ValueError, match="closed"):
            call()
    assert selected.events == before


def test_two_overlapping_scopes_have_independent_lifetimes_and_acquisitions(tmp_path):
    _, _, context = _values(tmp_path)
    selected = _selected()
    with _scope(selected, context) as first:
        with _scope(selected, context) as second:
            assert first is not second
            child1, child2 = selected.coordinate_operation.children
            assert child1 is not child2
            child2.result = _Value(schema_version=1, value=-0.0)
            assert first.read_completed("coordinate", context) is None
            assert second.read_completed("coordinate", context).to_json() == child2.result.to_json()
        with pytest.raises(ValueError, match="closed"):
            second.read_completed("coordinate", context)
        assert first.read_completed("coordinate", context) is None


def test_two_actual_threads_keep_their_own_scope_alive_until_its_exit(tmp_path):
    _, _, context = _values(tmp_path)
    selected = _selected()
    entered = Barrier(2)
    closing = Barrier(2)

    def use_scope(ordinal):
        with _scope(selected, context) as actual:
            entered.wait(timeout=5)
            assert actual.read_completed("coordinate", context) is None
            closing.wait(timeout=5)
        with pytest.raises(ValueError, match="closed"):
            actual.read_completed("coordinate", context)
        return ordinal

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(use_scope, 1)
        second = pool.submit(use_scope, 2)
        assert first.result(timeout=10) == 1
        assert second.result(timeout=10) == 2
    assert len(selected.coordinate_operation.children) == 2
    assert selected.coordinate_operation.children[0] is not selected.coordinate_operation.children[1]


@pytest.mark.parametrize("phase", ["enter", "exit"])
def test_scope_entry_and_cleanup_failures_propagate_without_fallback(phase, tmp_path):
    _, _, context = _values(tmp_path)
    selected = _selected()

    @contextmanager
    def failing(coordinate, selected_context):
        selected.events.append(("enter", selected_context.reference.id))
        if phase == "enter":
            raise OSError("actual enter refusal")
        try:
            yield _Operations(selected.events)
        finally:
            selected.events.append(("exit", selected_context.reference.id))
            raise OSError("actual cleanup refusal")

    selected.coordinate_operation.open = failing
    captured = None
    with pytest.raises(OSError, match="refusal"), _scope(selected, context) as actual:
        captured = actual
    if captured is not None:
        with pytest.raises(ValueError, match="closed"):
            captured.read_completed("coordinate", context)
    assert not any(event[0] in {"read", "advance", "publish"} for event in selected.events)


def test_provider_cannot_suppress_driver_failure_and_no_fallback_is_attempted(tmp_path):
    _, _, context = _values(tmp_path)
    selected = _selected()

    @contextmanager
    def suppressing(coordinate, selected_context):
        selected.events.append(("enter", selected_context.reference.id))
        try:
            yield _Operations(selected.events)
        except RuntimeError:
            selected.events.append(("suppressed", selected_context.reference.id))

    selected.coordinate_operation.open = suppressing
    failure = RuntimeError("actual common guard refused")
    with pytest.raises(RuntimeError) as caught, _scope(selected, context):
        raise failure
    assert caught.value is failure
    assert [event[0] for event in selected.events] == ["enter", "suppressed"]


def _driver(tmp_path, selected):
    plan, job, context = _values(tmp_path)
    store = WorkflowStore(tmp_path)
    with store.transaction() as connection:
        connection.execute("INSERT INTO plans VALUES (?,?,?)", (job.plan_id, job.owner, plan.to_json()))
        store.save_record(connection, job, create=True)
    works = tuple(
        CoordinateWork(c.fingerprint, c.reference, c.target, c, context.preparation.job_root / c.reference.id)
        for c in plan.coordinates
    )
    program = ExecutionProgram(
        context.preparation,
        works,
        _wrapped(selected),
        _Codec(),
        fingerprint,
        context.preparation.job_root / "complete.json",
    )
    # Resource sample is explicitly a small successful host-boundary fixture,
    # not this pytest process's lifetime RSS or a paid physical-cost claim.
    worker = driver._Worker(store, job, plan, time.monotonic() + 30, driver.WorkerHooks(peak_rss_mib=lambda: 1.0), None)
    return worker, program


def test_real_common_loop_keeps_reservations_progress_publication_and_scope_order(tmp_path):
    selected = _selected()
    worker, program = _driver(tmp_path, selected)
    results = worker.drive_coordinates(program)
    assert len(results) == len(program.coordinates)
    expected = ["enter", "read", "advance", "verify", "publish", "read", "exit"]
    assert [event[0] for event in selected.events] == expected * len(program.coordinates)
    actual = driver.read_job(worker.store, worker.job.job_id)
    assert actual.attempts == len(program.coordinates)
    assert views.completed(actual) == len(program.coordinates)
    assert views.retained_attempts(worker.store.root / "jobs" / actual.job_id, worker.plan, job_id=actual.job_id) == 2
    assert not program.output_path.exists()  # Coordinate completion is not final job publication.
    callback = selected.coordinate_operation.children[0].callback
    context = CoordinateContext(
        program.context,
        program.coordinates[0].coordinate_fingerprint,
        program.coordinates[0].reference,
        program.coordinates[0].target,
        program.coordinates[0].artifact_directory,
    )
    proof = CommittedProgress(
        program.context.plan_fingerprint, context.coordinate_fingerprint, context.target, results[0]
    )
    with pytest.raises(ValueError, match="closed"):
        callback(proof)  # Must stay bound to its own closed scope, never the next coordinate.


def test_actual_cancellation_after_publication_closes_scope_and_preserves_the_artifact(tmp_path):
    selected = _selected()
    worker, program = _driver(tmp_path, selected)
    factory = selected.coordinate_operation.factory

    def cancelling(events):
        child = factory(events)
        publish = child.publish_coordinate

        def cancel(coordinate, value, context):
            publish(coordinate, value, context)
            with worker.store.transaction() as connection:
                job = worker.store.records(connection)[0]
                worker.store.save_record(connection, views.changed(job, state="cancelled"))

        child.publish_coordinate = cancel
        return child

    selected.coordinate_operation.factory = cancelling
    with pytest.raises(InterruptedError, match="no longer running"):
        worker.drive_coordinates(program)
    actual = driver.read_job(worker.store, worker.job.job_id)
    assert actual.state == "cancelled"
    assert views.completed(actual) == 0
    assert selected.events[-1][0] == "exit"
    assert (program.coordinates[0].artifact_directory / "publication.txt").exists()
    assert not program.output_path.exists()


def test_actual_reconcile_uses_same_proof_codec_before_another_reservation(tmp_path):
    selected = _selected()
    worker, program = _driver(tmp_path, selected)
    work = program.coordinates[0]
    worker.reserve(work)
    factory = selected.coordinate_operation.factory

    def bad_reconcile(events):
        child = factory(events)

        def reconcile(context):
            events.append(("reconcile", context.reference.id))
            return CommittedProgress(
                context.preparation.plan_fingerprint,
                context.coordinate_fingerprint,
                context.target,
                _Value(schema_version=1, value=0.0).model_copy(update={"schema_version": 2}),
            )

        child.progress_verifier.reconcile = reconcile
        return child

    selected.coordinate_operation.factory = bad_reconcile
    with pytest.raises(ValueError):
        worker.drive_coordinates(program)
    assert [event[0] for event in selected.events] == ["enter", "read", "reconcile", "exit"]
    assert driver.read_job(worker.store, worker.job.job_id).attempts == 1
    assert views.completed(driver.read_job(worker.store, worker.job.job_id)) == 0


@pytest.mark.parametrize("phase", ["read", "advance", "callback", "publish", "post-publish"])
def test_real_driver_closes_on_failure_without_marking_coordinate_complete(phase, tmp_path):
    selected = _selected()
    original = selected.coordinate_operation.factory

    def failing(events):
        child = original(events)
        name = {
            "read": "read_completed",
            "advance": "advance",
            "callback": "advance",
            "publish": "publish_coordinate",
            "post-publish": "publish_coordinate",
        }[phase]
        operation = getattr(child, name)

        def refuse(*args):
            if phase == "post-publish":
                operation(*args)
            if phase == "callback":
                bad = _Value(schema_version=1, value=0.0).model_copy(update={"schema_version": 2})
                attempt = args[1]
                args[2](
                    CommittedProgress(
                        attempt.coordinate.preparation.plan_fingerprint,
                        attempt.coordinate.coordinate_fingerprint,
                        attempt.coordinate.target,
                        bad,
                    )
                )
            raise RuntimeError(phase)

        setattr(child, name, refuse)
        return child

    selected.coordinate_operation.factory = failing
    worker, program = _driver(tmp_path, selected)
    with pytest.raises((RuntimeError, ValueError)):
        worker.drive_coordinates(program)
    assert selected.events[-1][0] == "exit"
    assert views.completed(driver.read_job(worker.store, worker.job.job_id)) == 0
    assert not program.output_path.exists()
    if phase == "post-publish":
        assert (program.coordinates[0].artifact_directory / "publication.txt").exists()
