"""A real spawned reader cannot mistake an uncommitted running claim for a queued no-op."""

import subprocess
import sys
import time
from contextlib import closing

import pytest

import coleman.application.workflow_supervisor as supervision
from coleman.application.baseline_lock import baseline_lock
from coleman.workflows.contracts import WorkflowLimits
from coleman.workflows.lifecycle import ExecutePlan
from tests.workflows.test_execution import setup_workflow

_OBSERVE_INITIAL_SNAPSHOT = """
import sys
from pathlib import Path
from coleman.artifacts import ArtifactWriter
import coleman.application.workflow_worker as worker
from coleman.application.workflow_supervisor import main
original_read = worker._read_job
marker = Path(sys.argv[2]) / 'initial-worker-snapshot.txt'
def observe(store, job_id):
    job = original_read(store, job_id)
    if not marker.exists():
        ArtifactWriter(marker.parent).write_immutable(marker, job.state)
    return job
worker._read_job = observe
main()
"""


@pytest.mark.parametrize("commit_claim", [True, False])
def test_spawned_worker_waits_for_the_real_claim_commit_or_rollback(tmp_path, monkeypatch, commit_claim):
    service, preview = setup_workflow(
        tmp_path, policies=("Random",), seeds=(42,), limits=WorkflowLimits(max_attempts=1, attempt_seconds=30.0)
    )
    with monkeypatch.context() as patch:
        patch.setattr(service, "_start", lambda: None)
        job = service.execute(ExecutePlan(schema_version=1, request_id="claim-visibility", plan_id=preview.plan_id))
    children = []

    def observe_spawn(root, role, descriptor, job_id=None):
        assert role == "work"
        assert job_id is not None
        child = subprocess.Popen(
            [sys.executable, "-c", _OBSERVE_INITIAL_SNAPSHOT, role, str(root), str(descriptor), job_id],
            cwd=root,
            pass_fds=(descriptor,),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        children.append(child)
        marker = root / "initial-worker-snapshot.txt"
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            assert child.poll() is None
            time.sleep(0.01)
        assert marker.read_text() == "queued"
        if not commit_claim:
            raise RuntimeError("controlled claim rollback")

    monkeypatch.setattr(supervision, "_spawn", observe_spawn)
    with baseline_lock(tmp_path / "supervisor.lock", handoff=True) as descriptor:
        if commit_claim:
            assert supervision._schedule(service.store, descriptor)
        else:
            with pytest.raises(RuntimeError, match="controlled claim rollback"):
                supervision._schedule(service.store, descriptor)
    for child in children:
        assert child.wait(timeout=30) == 0
    state = service.status(job.job_id)
    if commit_claim:
        assert state.state == "completed"
        assert state.invocations == state.attempts == 1
        assert len(service.report(job.job_id).trials[0].rows) == 3
    else:
        assert state.state == "queued"
        assert state.invocations == state.attempts == 0
        assert not (tmp_path / "jobs" / job.job_id / "native").exists()
        with closing(service.store.connect()) as connection:
            assert len(service.store.jobs(connection)) == 1
