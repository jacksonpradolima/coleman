"""Real owned-process death/cancellation tests; installed source bytes stay intact."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coleman.application.baseline_lock import baseline_lock
from coleman.application.execution_views import changed
from coleman.application.tasks import TaskService
from coleman.application.workflow_store import WorkflowStore
from coleman.tasks.requests import TaskExecuteRequest
from coleman.workflows.contracts import WorkflowError

if TYPE_CHECKING:
    from tests.tasks import _installed_support as shared
else:
    shared = import_module("task_installed_support")


@dataclass(frozen=True)
class Fault:
    """One fixed trusted injection site and its parent-owned observation file."""

    kind: str
    barrier: Path


def _pause() -> None:
    """Wait only while a parent observes an actual barrier and kills/cancels us."""
    deadline = time.monotonic() + shared.JOB_OBSERVATION_SECONDS
    while time.monotonic() < deadline:
        time.sleep(0.025)
    raise TimeoutError("Fault parent did not act within the original task allowance")


def fault_worker(root: Path, approval: Path, identity: str, descriptor: int, fault: Fault) -> None:
    """Run the REAL shared driver under a verified inherited actual job lock.

    Notes
    -----
    This trusted fault-injection launcher is separate from normal fixed
    supervisor startup, which detached_parent exercises without patching spawn.
    Wrappers call the principal original methods exactly once. No installed
    SDK/provider file, learner, ACK response or mathematical result is replaced.
    """
    from coleman.application import execution_driver
    from coleman.application.workflow_supervisor import _verify_inherited_lock
    from coleman.state.local import LocalStateStore

    _verify_inherited_lock(root / "locks" / f"{identity}.lock", descriptor)
    if fault.kind == "inner_ack":
        original = LocalStateStore.commit_transition

        def after_ack(store, *args, **kwargs):
            receipt = original(store, *args, **kwargs)
            if receipt.status == "new" and receipt.accepted_head.generation == 1:
                actual = store.read_head()
                assert actual == receipt.current_head
                shared.save(
                    fault.barrier,
                    {
                        "boundary": "returned_inner_ack_before_outer_progress",
                        "pid": os.getpid(),
                        "receipt": json.loads(receipt.to_json()),
                    },
                )
                _pause()
            return receipt

        shared.replace_callback(LocalStateStore, "commit_transition", after_ack)
    elif fault.kind == "outer_progress":
        original_progress = execution_driver._Worker.record_progress

        def after_progress(worker, program, context, progress, **kwargs):
            original_progress(worker, program, context, progress, **kwargs)
            if progress.frontier.position == 2:
                shared.save(
                    fault.barrier,
                    {
                        "boundary": "verified_outer_progress_after_actual_ack",
                        "pid": os.getpid(),
                        "proof": json.loads(progress.proof.to_json()),
                    },
                )
                _pause()

        shared.replace_callback(execution_driver._Worker, "record_progress", after_progress)
    else:
        raise ValueError("Unknown trusted fault boundary")
    execution_driver.run_job(root, identity, bootstrap_profile=shared.profile(approval))


def _wait_barrier(path: Path, child: subprocess.Popen) -> dict[str, Any]:
    """Observe a real live owned child rather than assuming a sleep is a barrier."""
    deadline = time.monotonic() + shared.JOB_OBSERVATION_SECONDS
    while time.monotonic() < deadline:
        if path.exists():
            assert child.poll() is None
            observed = json.loads(path.read_text())
            assert observed["pid"] == child.pid
            return observed
        if child.poll() is not None:
            output, errors = child.communicate(timeout=shared.PROCESS_OBSERVATION_SECONDS)
            raise AssertionError("Fault child ended before its actual barrier: " + output + errors)
        time.sleep(0.025)
    raise TimeoutError("Actual fault boundary was not reached within the task allowance")


@contextmanager
def _launch_fault(root: Path, approval: Path, key: str, kind: str):
    """Admit, actually own and reserve one physical child with original semantics."""
    admitted = shared.prepare(root, approval, key)
    identity = admitted["job_id"]
    store = WorkflowStore(root)
    barrier = root.parent / f"{root.name}-barrier.json"
    with ExitStack() as children:
        with baseline_lock(root / "locks" / f"{identity}.lock", handoff=True) as descriptor:
            with store.transaction() as connection:
                job = next((item for item in store.records(connection) if item.job_id == identity), None)
                assert job is not None
                assert job.state == "queued"
                store.save_record(connection, changed(job, state="running", invocations=job.invocations + 1))
            arguments = [
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).with_name("driver.py")),
                "fault-worker",
                str(root),
                str(approval),
                identity,
                str(descriptor),
                kind,
                str(barrier),
            ]
            child = children.enter_context(
                subprocess.Popen(
                    arguments,
                    cwd=Path.cwd(),
                    pass_fds=(descriptor,),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        try:
            yield child, admitted, barrier
        finally:
            _stop(child)


def _stop(child: subprocess.Popen) -> None:
    """Reap the actual fault child even if a parent assertion fails."""
    if child.poll() is None:
        child.kill()
    if child.stdout is not None and not child.stdout.closed:
        child.communicate(timeout=shared.PROCESS_OBSERVATION_SECONDS)
    else:
        child.wait(timeout=shared.PROCESS_OBSERVATION_SECONDS)


def interrupted(root: Path, approval: Path, key: str) -> dict[str, Any]:
    """Kill after a REAL accepted batch but before outer progress is recorded."""
    with _launch_fault(root, approval, key, "inner_ack") as (child, admitted, barrier_path):
        identity = admitted["job_id"]
        barrier = _wait_barrier(barrier_path, child)
        before = shared.state_snapshot(root, identity)
        assert before["job"]["state"] == "running"
        assert before["job"]["frontier"] is None
        assert before["job"]["attempts"] == 1
        assert [item["U"] for item in before["chain"]] == [0, 2]
        assert before["head"] == barrier["receipt"]["current_head"]
        child.kill()
        output, errors = child.communicate(timeout=shared.PROCESS_OBSERVATION_SECONDS)
        assert child.returncode == -signal.SIGKILL, output + errors
    view = TaskService(root).status(identity)
    assert view.job.state == "interrupted"
    assert not view.worker_active
    assert view.job.result_digest is None
    assert not (root / "jobs" / identity / "complete.json").exists()
    evidence = {
        "job_id": identity,
        "barrier": barrier,
        "before": before,
        "after": json.loads(view.job.to_json()),
        "exit_code": child.returncode,
    }
    shared.save(root.parent / f"{root.name}-interruption.json", evidence)
    return evidence


def recover(root: Path, approval: Path) -> dict[str, Any]:
    """Use normal fixed supervisor resume; accepted original panels are not replayed."""
    interrupted_case = interrupted(root, approval, "ack-gap")
    identity = interrupted_case["job_id"]
    service = TaskService(root, bootstrap_profile=shared.profile(approval))
    assert service.resume(identity).job.job_id == identity
    complete = shared.inspect_complete(root, approval, identity)
    assert complete["attempts"] == 2
    assert complete["invocations"] == 2
    return {"interruption": interrupted_case, "complete": complete}


def cancel_resume(root: Path, approval: Path) -> dict[str, Any]:
    """Cancel a real live owner after verified ACK/metadata, then resume exactly."""
    with _launch_fault(root, approval, "cancel-prefix", "outer_progress") as (child, admitted, barrier_path):
        identity = admitted["job_id"]
        barrier = _wait_barrier(barrier_path, child)
        service = TaskService(root)
        live = service.status(identity)
        assert live.worker_active
        assert live.job.frontier is not None
        assert live.job.frontier.position == 2
        cancelled = service.cancel(identity)
        assert cancelled.job.state == "cancelled"
        assert cancelled.job.result_digest is None
        output, errors = child.communicate(timeout=shared.PROCESS_OBSERVATION_SECONDS)
        assert child.returncode == 75, output + errors
    stopped = TaskService(root).status(identity)
    assert not stopped.worker_active
    assert stopped.job.state == "cancelled"
    try:
        TaskService(root).report(identity)
    except WorkflowError as error:
        assert "completed report" in str(error)
    else:
        raise AssertionError("A cancelled partial job must not expose complete evidence")
    assert not (root / "jobs" / identity / "complete.json").exists()
    assert TaskService(root, bootstrap_profile=shared.profile(approval)).resume(identity).job.job_id == identity
    complete = shared.inspect_complete(root, approval, identity)
    assert complete["attempts"] == 2
    assert complete["invocations"] == 2
    return {"barrier": barrier, "cancelled": json.loads(stopped.job.to_json()), "complete": complete}


def detached_parent(root: Path, approval: Path, request: str, evidence: Path) -> None:
    """Exit a real submitting parent while its NORMAL fixed supervisor is live.

    Notes
    -----
    An actual existing job lock gates queued work until this parent exits. The
    genuine _start/ensure_supervisor/_spawn/ACK path runs unchanged; no worker
    can complete before exit because the parent still owns that kernel lock.
    """
    service = TaskService(root, bootstrap_profile=shared.profile(approval))
    original = service._start
    current = TaskExecuteRequest.from_json(request)
    with closing(service.store.connect()) as connection:
        record = next(item for item in service.store.records(connection) if item.request_id == current.request_id)

    def gated_start():
        with baseline_lock(root / "locks" / f"{record.job_id}.lock"):
            original()
            observed = service.status(record.job_id)
            assert observed.job.state == "queued"
            assert observed.worker_active
            shared.save(
                evidence,
                {
                    "job_id": record.job_id,
                    "parent_pid": os.getpid(),
                    "state_before_parent_exit": "queued",
                    "actual_owned_gate": True,
                },
            )
            os._exit(0)

    shared.replace_callback(service, "_start", gated_start)
    service.execute(current)
    raise AssertionError("The submitting parent must exit inside its real ownership gate")


def refuse_changed_image(root: Path, approval: Path, identity: str) -> dict[str, object]:
    """Prove actual dependency bytes reject before the provider import/factory."""
    from coleman.application import executor_capture

    assert not shared.provider_loaded()
    imports = []
    original_import = executor_capture.import_module

    def observed(name):
        imports.append(name)
        return original_import(name)

    shared.replace_callback(executor_capture, "import_module", observed)
    service = TaskService(root, bootstrap_profile=shared.profile(approval))
    before = service.status(identity).job.to_json()
    try:
        service.resume(identity)
    except ValueError as error:
        assert "installed executor identity" in str(error)
    else:
        raise AssertionError("Changed same-version effective dependency must refuse continuation")
    assert not imports
    assert not shared.provider_loaded()
    assert service.status(identity).job.to_json() == before
    return {
        "before_factory": True,
        "provider_loaded": False,
        "attempts_unchanged": True,
        "version": metadata.version("coleman-typed-hedge-support"),
    }


def same_version_image_change(root: Path, approval: Path) -> dict[str, object]:
    """Mutate only an actual installed dependency; restore exact original bytes."""
    partial = interrupted(root, approval, "same-version-image")
    identity = partial["job_id"]
    distribution = metadata.distribution("coleman-typed-hedge-support")
    path = Path(str(distribution.locate_file("coleman_typed_hedge_support/__init__.py")))
    original, version = path.read_bytes(), distribution.version
    try:
        path.write_bytes(original + b"\n# Deliberate same-version installed dependency mutation.\n")
        result = shared.run_checked(
            [
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).with_name("driver.py")),
                "refuse-image",
                str(root),
                str(approval),
                identity,
            ],
        )
        refusal = json.loads(result.stdout)
        assert refusal["version"] == version
    finally:
        path.write_bytes(original)
    assert path.read_bytes() == original
    assert metadata.version("coleman-typed-hedge-support") == version
    assert shared.state_snapshot(root, identity)["head"] == partial["before"]["head"]
    current = TaskService(root, bootstrap_profile=shared.profile(approval))
    current.resume(identity)
    complete = shared.inspect_complete(root, approval, identity)
    return {"partial": partial, "refusal": refusal, "same_version": version, "complete": complete}
