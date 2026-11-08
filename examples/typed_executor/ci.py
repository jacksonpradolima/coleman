"""Approve a selected installed recipe, then run its public constructed task.

Copy this file outside the checkout after installing the three ordinary wheels.
Run with the base consumer interpreter using ``-I -B`` and optimization zero.
The explicit approval command is a trusted operator action, never task content.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sqlite3
import time
from math import isfinite
from pathlib import Path

from coleman.application.executor_capture import capture_executor_identity, load_approved_executor
from coleman.application.tasks import TaskService
from coleman.artifacts import ArtifactWriter
from coleman.core.identity import Reference
from coleman.learning.reference import Hedge
from coleman.tasks.identity import ApprovedExecutor, ExecutorRecipe, HostBootstrapProfile
from coleman.tasks.requests import TaskExecuteRequest, TaskJobView


def read_artifact(path: Path, maximum_bytes: int) -> str:
    """Acquire exact bounded UTF-8 operator metadata before model parsing.

    Parameters
    ----------
    path : pathlib.Path
        Explicit trusted operator-selected local artifact.
    maximum_bytes : int
        Strict integer from one through16KiB, in UTF-8 bytes; checked before I/O.

    Returns
    -------
    str
        Complete unmodified text; invalid or excessive content raises.

    Raises
    ------
    ValueError
        Invalid capacity refuses before opening the operator-selected path.
    """
    if type(maximum_bytes) not in (int,) or not 1 <= maximum_bytes <= 16 * 1024:
        raise ValueError("Operator metadata requires a strict positive capacity at most16KiB")
    with path.open("rb") as stream:
        content = stream.read(maximum_bytes + 1)
    if len(content) > maximum_bytes:
        raise ValueError("Operator artifact exceeds its selected byte capacity")
    return content.decode("utf-8")


def approve(recipe_path: Path, profile_path: Path) -> HostBootstrapProfile:
    """Explicitly approve the actual installed image for one static recipe.

    Parameters
    ----------
    recipe_path : pathlib.Path
        Independently selected shipped static recipe JSON, read without provider
        import. Its declaration alone grants no authority.
    profile_path : pathlib.Path
        Trusted immutable output for this deliberate operator approval.

    Returns
    -------
    HostBootstrapProfile
        Exact captured approval. Capture reads bounded installed resources and
        dependency metadata without installing packages or loading the provider.

    Notes
    -----
    Requires the supported immutable directory-wheel CPython profile. Reading
    this saved object later proves equality only: the operator must explicitly
    supply it to each new service/supervisor after daemon death.
    """
    selected = ExecutorRecipe.from_json(read_artifact(recipe_path, 16 * 1024))
    image = capture_executor_identity(selected)
    profile = HostBootstrapProfile(
        contract="coleman-tasks-v1",
        schema_version=1,
        artifact_type="host_bootstrap",
        reference=Reference(id="approved-constructed-hedge", version="1"),
        executors=(ApprovedExecutor(recipe=selected, implementation=image.binding),),
    )
    ArtifactWriter(profile_path.parent).write_immutable(profile_path, profile.to_json())
    return profile


def _direct_states(source) -> tuple[str, ...]:
    """Run the same original panels through actual independent direct kernels."""
    states = []
    for trajectory in source.trajectories:
        learner = Hedge(trajectory.initial_state.payload.configuration, owner=trajectory.reference)
        for panel in trajectory.panels:
            if learner.observe(panel).status != "accepted":
                raise ValueError("Direct reference refused an original constructed loss panel")
        states.append(learner.export_state().to_json())
    return tuple(states)


def wait_for_completion(
    service: TaskService, request: TaskExecuteRequest, *, observation_seconds: float = 120.0
) -> TaskJobView:
    """Observe one retained request through transient numeric SQLite contention.

    Parameters
    ----------
    service : TaskService
        Independently configured common facade and trusted local principal.
    request : TaskExecuteRequest
        Exact immutable request, reused if admission committed before its reply
        failed. A retry never invents a new key or another logical experiment.
    observation_seconds : float, optional
        Finite positive monotonic observation allowance, at most the original
        120 seconds. Each metadata operation keeps its own one-second wait.

    Returns
    -------
    TaskJobView
        Actual terminal snapshot after kernel ownership ends. Non-completion
        stays explicit; the caller separately interprets failure/cancellation.

    Raises
    ------
    ValueError
        Invalid observation allowance. No storage capacity/timeout is changed.
    TimeoutError
        The original observation window ended; retry this same request later.
    sqlite3.Error, OSError
        Non-contention metadata/storage errors propagate, without text matching
        or an unbounded retry. Only numeric BUSY/LOCKED primary/extended codes
        are retried, and no transaction body is replayed by this observer.
    """
    if isinstance(observation_seconds, bool) or not isfinite(observation_seconds) or not 0 < observation_seconds <= 120:
        raise ValueError("Observation allowance must be finite, positive and at most 120 seconds")
    deadline, view = time.monotonic() + observation_seconds, None
    while time.monotonic() < deadline:
        try:
            current = service.execute(request) if view is None else service.status(view.job.job_id)
        except sqlite3.OperationalError as error:
            code = getattr(error, "sqlite_errorcode", 0)
            if not isinstance(code, int) or code & 0xFF not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise
        else:
            view = current
            if (
                view.job.state in {"completed", "failed", "cancelled", "expired", "interrupted"}
                and not view.worker_active
            ):
                return view
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.025, remaining))
    raise TimeoutError("Observation window ended; retry the same retained request ID")


def run(root: Path, profile_path: Path) -> dict[str, object]:
    """Execute and recheck complete direct/durable mathematical-state parity.

    Parameters
    ----------
    root : pathlib.Path
        Trusted persistent workspace using the existing common coordinator.
    profile_path : pathlib.Path
        Operator-selected current approval, not inferred from workspace data.

    Returns
    -------
    dict
        Actual job/plan and accepted-panel/generation observations. Results are
        constructed mechanism evidence, not empirical efficacy or inference.

    Raises
    ------
    ValueError, TimeoutError, OSError
        Changed installation, failed work, unavailable/corrupt result, state
        mismatch or incomplete observation within the finite 120-second window.
    """
    profile = HostBootstrapProfile.from_json(read_artifact(profile_path, 16 * 1024))
    if len(profile.executors) != 1:
        raise ValueError("Public example requires its one explicit selected executor")
    approved = profile.executors[0]
    image = capture_executor_identity(approved.recipe)
    executor = load_approved_executor(approved, expected=image)  # Validate before the fixture import below.
    sample = importlib.import_module("coleman_typed_hedge.sample")
    request = sample.planning_request()
    service = TaskService(root, bootstrap_profile=profile)
    preview = service.plan(request)
    command = TaskExecuteRequest(
        contract="coleman-tasks-v1",
        schema_version=1,
        artifact_type="execute_request",
        request_id="constructed-hedge-v1",
        plan_id=preview.plan_id,
    )
    view = wait_for_completion(service, command)
    if view.job.state != "completed":
        raise ValueError("Constructed Hedge task did not complete: " + view.job.state)
    report = executor.final_result_codec.validate(service.decode_result(view.job.job_id))
    source = sample.inputs(sample.configuration())
    if tuple(item.checkpoint.learner.to_json() for item in report.results) != _direct_states(source):
        raise ValueError("Durable result differs from complete actual direct mathematical state")
    if service.execute(command).job.job_id != view.job.job_id:
        raise ValueError("Retained request unexpectedly created a different logical job")
    return {
        "job_id": view.job.job_id,
        "plan": preview.digest,
        "interpretation": report.interpretation,
        "updates": [item.proof.frontier.position for item in report.results],
        "generations": [item.proof.observed_head.generation for item in report.results],
        "complete_direct_state_equal": True,
    }


def main() -> None:
    """Run the explicit operator approval or the installed constructed example.

    Notes
    -----
    This is example host tooling. Task/CLI/MCP request DTOs contain no profile,
    package path, import string, loader or installation command.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    approval = subparsers.add_parser("approve")
    approval.add_argument("recipe", type=Path)
    approval.add_argument("profile", type=Path)
    execution = subparsers.add_parser("run")
    execution.add_argument("root", type=Path)
    execution.add_argument("profile", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "approve":
        result = {"approved": approve(arguments.recipe, arguments.profile).fingerprint}
    else:
        result = run(arguments.root, arguments.profile)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
