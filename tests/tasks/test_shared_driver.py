"""Shared lifecycle boundary units; unchanged workflow tests run real native work."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import sysconfig
import time
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from coleman.application import execution_driver as driver
from coleman.application import execution_views as views
from coleman.application.execution_program import CoordinateWork, ExecutionProgram
from coleman.application.native_executor import _marker_position, _NativePrepared
from coleman.application.workflow_store import JobRecord, WorkflowStore
from coleman.artifacts import ArtifactWriter
from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.checkpoint.state import CheckpointPayload
from coleman.core.base import CoreContract
from coleman.core.primitives import fingerprint
from coleman.state.storage_contracts import ExecutionCompatibility
from coleman.tasks.contracts import TaskJobRecord, TaskPlan
from coleman.tasks.identity import CodecDeclaration
from coleman.tasks.protocols import CoordinateContext, PreparationContext
from coleman.workflows.contracts import ExperimentPlan, WorkflowError
from coleman.workflows.lifecycle import PhysicalAttempt

FIXTURES = Path(__file__).parents[1] / "fixtures" / "tasks" / "v1"


def _task_values():
    """Use constructed declarations only; these fixtures claim no installation."""
    plan = TaskPlan.from_json((FIXTURES / "plan.json").read_bytes())
    original = TaskJobRecord.from_json((FIXTURES / "job.json").read_bytes())
    job = TaskJobRecord.model_validate(
        {
            **original.model_dump(),
            "state": "running",
            "current_coordinate": None,
            "frontier": None,
            "attempts": 0,
        }
    )
    return plan, job


def test_common_driver_cold_import_does_not_import_native_runner_or_optional_scientific_stack():
    paths = [str(Path(__file__).parents[2]), sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    code = """
import importlib.abc,sys
sys.path[:0]=PATHS
blocked=('coleman.application.native_executor','coleman.application.workflow_native',
         'coleman.runner','coleman.cli','coleman.transports','polars','pyarrow','numpy','mcp')
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==p or fullname.startswith(p+'.') for p in blocked):
            raise AssertionError('Unexpected provider import: '+fullname)
sys.meta_path.insert(0,Boundary())
import coleman.application.execution_driver
assert not any(any(n==p or n.startswith(p+'.') for p in blocked) for n in sys.modules)
print('neutral common driver')
""".replace("PATHS", repr(paths))
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "neutral common driver"


def test_generic_attempt_retains_actual_ordinal_clock_and_separate_identity_law(tmp_path, monkeypatch):
    plan, job = _task_values()
    monkeypatch.setattr(views.time, "time", lambda: 0.0)
    coordinate = plan.coordinates[0]
    identity, payload = views.attempt_payload(job, plan, coordinate.fingerprint, 1)
    event = views.CoordinateAttempt.from_json(payload)
    expected = fingerprint(["task.physical.v1", job.job_id, coordinate.fingerprint, 1])[7:]
    assert identity == expected
    assert event.attempt_id == expected
    assert event.started_epoch_seconds == 0.0
    assert event.ordinal == 1
    assert event.worker_invocation == 1
    assert event.stage == plan.stage
    target = views.attempt_directory(tmp_path, coordinate.fingerprint) / "0001.json"
    ArtifactWriter(tmp_path).write_immutable(target, payload)
    assert views.retained_attempts(tmp_path, plan, job_id=job.job_id) == 1
    raw = json.loads(payload)
    raw["ordinal"] = 2
    target.write_text(json.dumps(raw))
    with pytest.raises(WorkflowError, match="binding mismatch"):
        views.retained_attempts(tmp_path, plan, job_id=job.job_id)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("ordinal", 0),
        ("worker_invocation", 9),
        ("started_epoch_seconds", -1.0),
        ("started_epoch_seconds", float("nan")),
        ("started_epoch_seconds", float("inf")),
    ],
)
def test_generic_attempt_rejects_future_coercion_bounds_and_nonfinite_clocks(field, value):
    plan, job = _task_values()
    _, payload = views.attempt_payload(job, plan, plan.coordinates[0].fingerprint, 1)
    raw = json.loads(payload)
    raw[field] = value
    with pytest.raises(ValueError):
        views.CoordinateAttempt.model_validate(raw)


def test_native_attempt_retains_exact_original_physical_attempt_bytes_and_identity(monkeypatch):
    native_fixture = Path(__file__).parents[1] / "workflows" / "fixtures" / "experiment-plan-v1.json"
    plan = ExperimentPlan.from_json(native_fixture.read_bytes())
    job = JobRecord(
        schema_version=1,
        job_id="native",
        owner="owner",
        request_id="request",
        request_digest="sha256:" + "a" * 64,
        plan_id=plan.fingerprint[7:],
        state="running",
        stage="descriptive",
        created=0.0,
        changed=0.0,
        total_trials=len(plan.trials),
        invocations=1,
    )
    trial = plan.trials[0]
    monkeypatch.setattr(views.time, "time", lambda: 123.0)
    identity, payload = views.attempt_payload(job, plan, trial.trial_id, 1)
    expected = fingerprint(["workflow.physical.v1", job.job_id, trial.trial_id, 1])[7:]
    original = PhysicalAttempt(
        schema_version=1,
        trial_id=trial.trial_id,
        attempt_id=expected,
        plan_fingerprint=plan.fingerprint,
        stage="descriptive",
        worker_invocation=1,
        ordinal=1,
        started_epoch_seconds=123.0,
    )
    assert identity == expected
    assert payload == original.to_json()


def test_known_metadata_projection_keeps_logical_frontiers_and_monotone_completion_separate():
    plan, job = _task_values()
    first, second = plan.coordinates
    active = views.reserved(job, first.fingerprint)
    origin = first.target.model_copy(update={"position": 1})
    progressed = views.progressed(active, first.fingerprint, origin)
    assert progressed.attempts == 1
    assert views.position(progressed, first.fingerprint) == 1
    assert views.position(progressed, second.fingerprint) == 0
    zero = first.target.model_copy(update={"position": 0})
    with pytest.raises(WorkflowError, match="rewind"):
        views.progressed(progressed, first.fingerprint, zero)
    with pytest.raises(WorkflowError, match="active coordinate"):
        views.progressed(progressed, second.fingerprint, origin)
    done = views.coordinate_completed(progressed, 1)
    repeated = views.coordinate_completed(done, 0)
    assert views.completed(repeated) == 1
    assert views.current_coordinate(repeated) is None
    assert isinstance(repeated, TaskJobRecord)
    assert repeated.frontier is None


class _PublicationValue(CoreContract):
    """Concrete publication-only value; no learner execution is claimed."""

    count: int


def _outside_writer(store, observed, callback):
    """A second writer must acquire ownership during every concrete callback."""
    with closing(sqlite3.connect(store.path, timeout=0, isolation_level=None)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.rollback()
    observed.append(callback)


@dataclass(frozen=True)
class _PublicationCodec:
    declaration: CodecDeclaration
    store: WorkflowStore
    observed: list[str]

    def validate(self, value, /):
        _outside_writer(self.store, self.observed, "validate")
        return _PublicationValue.model_validate(value)

    def encode(self, value, /):
        _outside_writer(self.store, self.observed, "encode")
        return value.to_json()

    def decode(self, body, /):
        _outside_writer(self.store, self.observed, "decode")
        return _PublicationValue.from_json(body)


def _publication_boundary(tmp_path):
    """Create real SQL metadata with constructed declarations, without execution."""
    store = WorkflowStore(tmp_path)
    plan, job = _task_values()
    with store.transaction() as connection:
        connection.execute("INSERT INTO plans VALUES (?,?,?)", (job.plan_id, job.owner, plan.to_json()))
        store.save_record(connection, job, create=True)
    compatibility = ExecutionCompatibility.model_validate(
        json.loads((FIXTURES / "owned-proof.json").read_text())["compatibility"]
    )
    context = PreparationContext(
        plan.fingerprint, plan.source, plan.implementation.binding, compatibility, tmp_path / "jobs" / job.job_id
    )
    coordinates = tuple(
        CoordinateWork(c.fingerprint, c.reference, c.target, c, context.job_root / c.reference.id)
        for c in plan.coordinates
    )
    observed = []
    declaration = CodecDeclaration.model_validate(plan.implementation.recipe.codec_for("final_result"))
    codec = _PublicationCodec(declaration, store, observed)
    program = ExecutionProgram(
        context, coordinates, cast(Any, None), codec, fingerprint, context.job_root / "complete.json"
    )
    # Publication/SQL isolation uses an explicit successful 1 MiB sample, not
    # the unrelated pytest process's lifetime peak or a physical-cost claim.
    hooks = driver.WorkerHooks(peak_rss_mib=lambda: 1.0)
    worker = driver._Worker(store, job, plan, time.monotonic() + 30, hooks, None)
    return worker, program, observed


def test_selected_concrete_callbacks_and_final_digest_run_before_the_completion_transaction(tmp_path):
    worker, program, observed = _publication_boundary(tmp_path)

    def content_identity(value):
        _outside_writer(worker.store, observed, "fingerprint")
        return value.fingerprint

    program = replace(program, output_fingerprint=content_identity)
    value = _PublicationValue(schema_version=1, count=2)
    worker.publish(program, value)
    retained = driver.read_job(worker.store, worker.job.job_id)
    assert retained.state == "completed"
    assert retained.result_digest == value.fingerprint
    assert views.completed(retained) == 2
    assert program.output_path.read_text() == value.to_json()
    assert set(observed) == {"validate", "encode", "decode", "fingerprint"}


def test_cancellation_during_final_codec_work_wins_before_any_complete_publication(tmp_path):
    worker, program, _ = _publication_boundary(tmp_path)

    def cancel(value):
        with worker.store.transaction() as connection:
            current = worker.store.records(connection)[0]
            worker.store.save_record(connection, views.changed(current, state="cancelled"))
        return value.fingerprint

    program = replace(program, output_fingerprint=cancel)
    value = _PublicationValue(schema_version=1, count=2)
    with pytest.raises(InterruptedError, match="no longer running"):
        worker.publish(program, value)
    assert driver.read_job(worker.store, worker.job.job_id).state == "cancelled"
    assert not program.output_path.exists()


def test_missing_actual_owned_state_cannot_reset_acknowledged_outer_progress(tmp_path):
    worker, program, _ = _publication_boundary(tmp_path)
    coordinate = program.coordinates[0]
    frontier = coordinate.target.model_copy(update={"position": 1})
    with worker.store.transaction() as connection:
        original = views.progressed(
            views.reserved(worker.job, coordinate.coordinate_fingerprint), coordinate.coordinate_fingerprint, frontier
        )
        worker.store.save_record(connection, original)

    class AbsentState:
        def reconcile(self, _context):
            return None

    # Deliberate absent-capability boundary input; this does not claim an ACK.
    executor = SimpleNamespace(progress_verifier=AbsentState())
    program = replace(program, executor=cast(Any, executor))
    context = CoordinateContext(
        program.context,
        coordinate.coordinate_fingerprint,
        coordinate.reference,
        coordinate.target,
        coordinate.artifact_directory,
    )
    with pytest.raises(WorkflowError, match="implicit reset is forbidden"):
        worker._reconcile(program, context)
    assert driver.read_job(worker.store, worker.job.job_id).to_json() == original.to_json()


def test_stateless_executor_cannot_submit_an_unverified_progress_integer(tmp_path):
    from coleman.tasks.protocols import CommittedProgress

    worker, program, _ = _publication_boundary(tmp_path)
    coordinate = program.coordinates[0]
    program = replace(program, executor=cast(Any, SimpleNamespace(progress_verifier=None)))
    context = CoordinateContext(
        program.context,
        coordinate.coordinate_fingerprint,
        coordinate.reference,
        coordinate.target,
        coordinate.artifact_directory,
    )
    progress = CommittedProgress(
        program.context.plan_fingerprint, coordinate.coordinate_fingerprint, coordinate.target, 2
    )
    with pytest.raises(WorkflowError, match="cannot submit unverified"):
        worker.record_progress(program, context, progress)
    assert driver.read_job(worker.store, worker.job.job_id).to_json() == worker.job.to_json()


def test_legacy_worker_read_and_resource_symbols_stay_live_after_hook_selection(tmp_path, monkeypatch):
    from coleman.application import workflow_worker as legacy

    worker, _, _ = _publication_boundary(tmp_path)
    hooks = driver.WorkerHooks(
        read_job=legacy._live_read_job,
        peak_rss_mib=legacy._live_peak,
        output_exceeds=legacy._live_output,
    )
    reads = []

    def observed_read(store, identity):
        reads.append(identity)
        return driver.read_job(store, identity)

    monkeypatch.setattr(legacy, "_read_job", observed_read)
    monkeypatch.setattr(legacy, "_worker_peak_rss_mib", lambda: None)
    monkeypatch.setattr(legacy, "_output_exceeds", lambda *_args, **_kwargs: True)
    assert hooks.read_job(worker.store, worker.job.job_id).to_json() == worker.job.to_json()
    assert reads == [worker.job.job_id]
    assert hooks.peak_rss_mib() is None
    assert hooks.output_exceeds(0, root=tmp_path) is True


@pytest.mark.parametrize("loads", [0, 1, 2])
def test_scientific_boundary_may_reuse_zero_or_load_once_and_rejects_duplicate_effects(loads):
    effects = []

    class Scope:
        def produce(self, load):
            for _ in range(loads):
                load()
            return "complete"

    worker = SimpleNamespace(drive_coordinates=lambda _program: effects.append("load"))
    program = SimpleNamespace(executor=SimpleNamespace(boundary=Scope()))
    if loads == 2:
        typed_worker = cast(Any, worker)
        typed_program = cast(Any, program)
        with pytest.raises(WorkflowError, match="more than once"):
            driver._produce(typed_worker, typed_program)
        assert effects == ["load"]
    else:
        assert driver._produce(cast(Any, worker), cast(Any, program)) == "complete"
        assert len(effects) == loads


def test_native_marker_rejects_moved_or_missing_actual_checkpoint_without_loading_pickle(tmp_path):
    trial_id = "sha256:" + "a" * 64
    base = tmp_path / "native" / trial_id[7:] / "checkpoints"
    store = LocalCheckpointStore(str(base))
    store.save(CheckpointPayload(run_id="owned", experiment=1, step=1))
    marker = base / "owned" / "progress_ex1.json"
    assert _marker_position(tmp_path, trial_id) == 1
    raw = json.loads(marker.read_text())
    raw["checkpoint_path"] = str(tmp_path / "foreign.pkl")
    marker.write_text(json.dumps(raw))
    with pytest.raises(WorkflowError, match="exact workflow root"):
        _marker_position(tmp_path, trial_id)
    raw["checkpoint_path"] = str(base / "owned" / "ckpt_ex1_step1.pkl")
    marker.write_text(json.dumps(raw))
    Path(raw["checkpoint_path"]).unlink()
    with pytest.raises(WorkflowError, match="implicit reset"):
        _marker_position(tmp_path, trial_id)


def test_missing_protected_native_prefix_refuses_at_read_preflight_before_an_attempt(tmp_path):
    from coleman.application.workflow_registry import owned_fixture

    fixture = Path(__file__).parents[1] / "workflows" / "fixtures" / "experiment-plan-v1.json"
    trial = ExperimentPlan.from_json(fixture.read_bytes()).trials[0]
    # Only this absent-receipt preflight is exercised. No running job, captured
    # installation, learned prefix, actual ACK or executable program is claimed.
    native = SimpleNamespace(
        job=SimpleNamespace(stage="evaluation"),
        root=tmp_path,
        plan=SimpleNamespace(dataset=owned_fixture(tmp_path)),
    )
    prepared = _NativePrepared(cast(Any, native), cast(Any, None), cast(Any, None))
    context = SimpleNamespace(artifact_directory=tmp_path / "evidence" / "absent")
    selected = cast(Any, context)
    with pytest.raises(WorkflowError, match="cannot reset missing native learning state"):
        prepared.read_completed(trial, selected)
    assert not (tmp_path / "attempts").exists()
