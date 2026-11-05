"""Actual native global-RNG checkpoint extension, legacy rejection and mathematical continuation."""

import json
import pickle
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from coleman.application.baseline_lock import baseline_lock
from coleman.application.workflow_checkpoint import WorkflowCheckpointStore, WorkflowRandomState
from coleman.application.workflow_store import JobRecord
from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.checkpoint.state import CheckpointPayload
from coleman.policy import base as policy_base
from coleman.workflows.lifecycle import ExecutePlan
from tests.workflows.test_execution import _CRASH_AT_REAL_CHECKPOINT, mathematics, setup_workflow, wait


def test_actual_global_generator_and_cached_uint32_restore_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(policy_base, "_rng", np.random.default_rng(42))
    # Consume one 32-bit sample so the generator can retain a cached half-word.
    policy_base._rng.integers(0, 2**31)
    store = WorkflowCheckpointStore(str(tmp_path), "sha256:" + "a" * 64)
    payload = CheckpointPayload(run_id="owned", experiment=1, step=2)
    store.save(payload)
    assert payload.extra == {}
    expected = policy_base._rng.permutation(31).tolist()
    policy_base._rng.permutation(31)
    restored = store.load("owned", 1)
    assert restored is not None
    assert policy_base._rng.permutation(31).tolist() == expected
    state = WorkflowRandomState.from_json(restored.extra["coleman_workflow_rng_v1"])
    assert state.to_json() == WorkflowRandomState.from_json(state.to_json()).to_json()
    assert state.increment > 2**64
    before = policy_base._rng.bit_generator.state
    incompatible_store = WorkflowCheckpointStore(str(tmp_path), "sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="binding"):
        incompatible_store.load("owned", 1)
    assert policy_base._rng.bit_generator.state == before
    future_state = {**state.model_dump(), "schema_version": 2}
    with pytest.raises(ValueError):
        WorkflowRandomState.model_validate(future_state)


def test_ordinary_native_checkpoints_are_not_implicitly_reinterpreted(tmp_path):
    LocalCheckpointStore(str(tmp_path)).save(CheckpointPayload(run_id="ordinary", experiment=1, step=1))
    legacy_workflow_store = WorkflowCheckpointStore(str(tmp_path), "sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="implicit reseeding"):
        legacy_workflow_store.load("ordinary", 1)
    assert WorkflowCheckpointStore(str(tmp_path), "sha256:" + "a" * 64).load("not-started", 1) is None
    marker = tmp_path / "ordinary" / "progress_ex1.json"
    raw = json.loads(marker.read_text())
    raw["checkpoint_path"] = str(tmp_path / "elsewhere.pkl")
    marker.write_text(json.dumps(raw))
    redirected_store = WorkflowCheckpointStore(str(tmp_path), "sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="not owned"):
        redirected_store.load("ordinary", 1)


def test_random_resume_after_consumed_rng_matches_uninterrupted_native_run(tmp_path, monkeypatch):
    service, preview = setup_workflow(tmp_path, policies=("Random",))
    with monkeypatch.context() as patch:
        patch.setattr(service, "_start", lambda: None)
        job = service.execute(ExecutePlan(schema_version=1, request_id="random-interrupted", plan_id=preview.plan_id))
    with baseline_lock(tmp_path / "locks" / f"{job.job_id}.lock", handoff=True) as descriptor:
        with service.store.transaction() as connection:
            record = service._job(connection, job.job_id)
            service.store.save_job(
                connection, JobRecord.model_validate({**record.model_dump(), "state": "running", "invocations": 1})
            )
        # Unlike the first build's agent-local shuffle, step 2 has consumed the
        # native policy module's global generator; resetting it changes future orders.
        code = _CRASH_AT_REAL_CHECKPOINT.replace("step==1", "step==2")
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(tmp_path), job.job_id],
            pass_fds=(descriptor,),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    _, error = process.communicate(timeout=30)
    assert process.returncode == 87, error.decode()
    assert service.status(job.job_id).checkpoint_step == 2
    service.resume(job.job_id)
    wait(service, job.job_id)
    resumed = service.report(job.job_id)
    fresh = service.execute(ExecutePlan(schema_version=1, request_id="random-uninterrupted", plan_id=preview.plan_id))
    wait(service, fresh.job_id)
    assert mathematics(resumed) == mathematics(service.report(fresh.job_id))
    assert len(resumed.physical_attempts) == 3
    assert len(resumed.trials) == 2


def test_descriptive_resume_cannot_reset_a_previously_committed_native_checkpoint(tmp_path, monkeypatch):
    service, preview = setup_workflow(tmp_path, policies=("Random",), seeds=(42,))
    with monkeypatch.context() as patch:
        patch.setattr(service, "_start", lambda: None)
        job = service.execute(ExecutePlan(schema_version=1, request_id="missing-frontier", plan_id=preview.plan_id))
    with baseline_lock(tmp_path / "locks" / f"{job.job_id}.lock", handoff=True) as descriptor:
        with service.store.transaction() as connection:
            record = service._job(connection, job.job_id)
            service.store.save_job(
                connection, JobRecord.model_validate({**record.model_dump(), "state": "running", "invocations": 1})
            )
        code = _CRASH_AT_REAL_CHECKPOINT.replace("completed_trials==1", "completed_trials==0").replace(
            "step==1", "step==2"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(tmp_path), job.job_id],
            pass_fds=(descriptor,),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    _, error = process.communicate(timeout=30)
    assert process.returncode == 87, error.decode()
    assert service.status(job.job_id).checkpoint_step == 2
    root = tmp_path / "jobs" / job.job_id
    marker = next((root / "native").rglob("progress_ex1.json"))
    retained = marker.read_bytes()
    checkpoint = marker.parent / "ckpt_ex1_step2.pkl"
    original_checkpoint = checkpoint.read_bytes()
    payload = LocalCheckpointStore(str(marker.parent.parent)).load(marker.parent.name, 1)
    assert payload is not None
    checkpoint.write_bytes(pickle.dumps(replace(payload, step=1)))
    service.resume(job.job_id)
    incompatible = wait(service, job.job_id, "failed")
    assert incompatible.attempts == 2
    assert incompatible.invocations == 2
    assert incompatible.checkpoint_step == 2
    checkpoint.write_bytes(original_checkpoint)
    native_bytes = {path: path.read_bytes() for path in (root / "native").rglob("*.parquet")}
    marker.unlink()
    service.resume(job.job_id)
    rejected = wait(service, job.job_id, "failed")
    assert rejected.attempts == 2
    assert rejected.invocations == 3
    assert rejected.error is not None
    assert "implicit reset" in rejected.error
    assert all(path.read_bytes() == value for path, value in native_bytes.items())
    marker.write_bytes(retained)
    service.resume(job.job_id)
    final = wait(service, job.job_id)
    assert final.attempts == 3
    assert final.invocations == 4
    reference = service.execute(ExecutePlan(schema_version=1, request_id="fresh-reference", plan_id=preview.plan_id))
    wait(service, reference.job_id)
    assert mathematics(service.report(job.job_id)) == mathematics(service.report(reference.job_id))


def test_valid_older_checkpoint_pair_cannot_rewind_the_acknowledged_frontier(tmp_path, monkeypatch):
    service, preview = setup_workflow(tmp_path, policies=("Random",), seeds=(42,))
    with monkeypatch.context() as patch:
        patch.setattr(service, "_start", lambda: None)
        job = service.execute(ExecutePlan(schema_version=1, request_id="rolled-back-pair", plan_id=preview.plan_id))
    capture = """    if step==1:
        marker=next((root/'jobs'/job_id/'native').rglob('progress_ex1.json'))
        (root/'earlier-marker.json').write_bytes(marker.read_bytes())
"""
    code = _CRASH_AT_REAL_CHECKPOINT.replace("    if job.completed_trials==1 and step==1:", capture + "    if step==2:")
    with baseline_lock(tmp_path / "locks" / f"{job.job_id}.lock", handoff=True) as descriptor:
        with service.store.transaction() as connection:
            record = service._job(connection, job.job_id)
            service.store.save_job(
                connection, JobRecord.model_validate({**record.model_dump(), "state": "running", "invocations": 1})
            )
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(tmp_path), job.job_id],
            pass_fds=(descriptor,),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    _, error = process.communicate(timeout=30)
    assert process.returncode == 87, error.decode()
    assert service.status(job.job_id).checkpoint_step == 2
    root = tmp_path / "jobs" / job.job_id
    marker = next((root / "native").rglob("progress_ex1.json"))
    acknowledged = marker.read_bytes()
    native_bytes = {p: p.read_bytes() for p in (root / "native").rglob("*.parquet")}
    marker.write_bytes((tmp_path / "earlier-marker.json").read_bytes())
    assert json.loads(marker.read_text())["step_committed"] == 1
    service.resume(job.job_id)
    rejected = wait(service, job.job_id, "failed")
    assert rejected.checkpoint_step == 2
    assert all(p.read_bytes() == payload for p, payload in native_bytes.items())
    marker.write_bytes(acknowledged)
    service.resume(job.job_id)
    wait(service, job.job_id)
    reference = service.execute(ExecutePlan(schema_version=1, request_id="rollback-reference", plan_id=preview.plan_id))
    wait(service, reference.job_id)
    assert mathematics(service.report(job.job_id)) == mathematics(service.report(reference.job_id))


@pytest.mark.parametrize("minimum", [-1, True, 1.5, 513])
def test_acknowledged_frontier_requires_a_bounded_ordinal(tmp_path, minimum):
    with pytest.raises(ValueError, match="frontier"):
        WorkflowCheckpointStore(str(tmp_path), "sha256:" + "a" * 64, minimum_step=minimum)


def test_older_or_missing_frontier_rejects_before_pickle_or_rng_restore(tmp_path, monkeypatch):
    store = WorkflowCheckpointStore(str(tmp_path), "sha256:" + "a" * 64)
    store.save(CheckpointPayload(run_id="owned", experiment=1, step=1))
    strict = WorkflowCheckpointStore(str(tmp_path), "sha256:" + "a" * 64, minimum_step=2)

    def forbidden(*args):
        raise AssertionError("An older frontier must reject before loading executable state")

    monkeypatch.setattr(LocalCheckpointStore, "load", forbidden)
    before = policy_base._rng.bit_generator.state
    with pytest.raises(ValueError, match="acknowledged"):
        strict.load("owned", 1)
    assert policy_base._rng.bit_generator.state == before
    with pytest.raises(ValueError, match="acknowledged"):
        strict.load("absent", 1)
