"""Measure real bounded queued tasks and naturally different artifact sizes.

Install the existing typed-Hedge provider/support wheels and independently
approve their recipe as described in the typed-executor guide. Copy this file,
task_records.py and task_inputs.py beside operational ci.py outside the checkout;
run with the reviewed minimal wheel interpreter using -I -B. No scheduler,
provider or guard is substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import os
import platform
import sqlite3
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from coleman.application import execution_lifecycle
from coleman.application.executor_capture import capture_executor_identity, load_approved_executor
from coleman.application.tasks import TaskService
from coleman.artifacts import ArtifactWriter
from coleman.core.primitives import Fingerprint, fingerprint
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CollectionManifest
from coleman.tasks.contracts import TaskJobRecord, TaskPlan
from coleman.tasks.identity import HostBootstrapProfile
from coleman.tasks.requests import TaskExecuteRequest, TaskJobView
from coleman.telemetry.resources import ProcessResourceTracker
from coleman.workflows.contracts import MAX_WORKFLOW_ARTIFACT_ENTRIES
from coleman.workflows.lifecycle import HostPolicy


def _sibling(name: str):
    """Load an explicitly copied helper even under Python isolated mode."""
    spec = importlib.util.spec_from_file_location("operational_" + name, Path(__file__).with_name(name + ".py"))
    if spec is None or spec.loader is None:
        raise ImportError("The explicitly copied operational helper is required")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if TYPE_CHECKING:
    from examples.operational.task_records import (
        CASES,
        CELL_SECONDS,
        ENTRY_RESERVE_SECONDS,
        OUTER_SECONDS,
        RESOURCE_ERRORS,
        THREAD_VARIABLES,
        JobObservation,
        NotStartedCell,
        TaskCell,
        TaskLoadReport,
        task_limits,
    )
else:
    _records = _sibling("task_records")
    CASES, CELL_SECONDS = _records.CASES, _records.CELL_SECONDS
    ENTRY_RESERVE_SECONDS, OUTER_SECONDS = _records.ENTRY_RESERVE_SECONDS, _records.OUTER_SECONDS
    RESOURCE_ERRORS, THREAD_VARIABLES = _records.RESOURCE_ERRORS, _records.THREAD_VARIABLES
    JobObservation, NotStartedCell = _records.JobObservation, _records.NotStartedCell
    TaskCell, TaskLoadReport, task_limits = _records.TaskCell, _records.TaskLoadReport, _records.task_limits


def _example():
    """Load only the consciously copied, source-bound configuration sibling."""
    spec = importlib.util.spec_from_file_location("operational_task_ci", Path(__file__).with_name("ci.py"))
    if spec is None or spec.loader is None:
        raise ImportError("The explicitly copied operational ci.py is required")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _occupancy(directory: Path) -> tuple[int, int]:
    """Inspect only inactive terminal job artifacts, without following links."""
    count = size = 0
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Inactive task artifact tree unexpectedly contains a symlink")
        if path.is_file():
            count += 1
            size += path.stat().st_size
            if count > MAX_WORKFLOW_ARTIFACT_ENTRIES or size > 4 * 1024**2:
                raise ValueError("Actual task output exceeds the unchanged bounded inspection profile")
    return count, size


def _views(service, ids):
    """Observe real status, retrying only numeric SQLiteBUSY/LOCKED later."""
    try:
        return tuple(service.status(identity) for identity in ids)
    except sqlite3.OperationalError as error:
        code = getattr(error, "sqlite_errorcode", 0)
        if not isinstance(code, int) or code & 0xFF not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            raise
        return ()


def _state_digest(states: tuple[str, ...]) -> Fingerprint:
    """Bind ordered complete state strings through the actual JSON-array seam."""
    return fingerprint(list(states))


def _sample_peaks(current: tuple[TaskJobView, ...], previous: tuple[int, bool]) -> tuple[int, bool]:
    """Accumulate nonatomic sweep observations without claiming concurrency."""
    queued = sum(view.job.state == "queued" and not view.worker_active for view in current)
    active = any(view.worker_active for view in current)
    return max(previous[0], queued), previous[1] or active


@dataclass(frozen=True)
class _Run:
    """Keep one explicit host measurement context, never checkpoint authority."""

    service: TaskService
    plan: TaskPlan
    direct: tuple[str, ...]
    commands: tuple[TaskExecuteRequest, ...]
    tracker: ProcessResourceTracker
    started: float
    deadline: float


def _prepare(output, profile, case, example, deadline) -> _Run:
    """Plan real work and retain the exact source before any queue submission."""
    tracker, started = ProcessResourceTracker(), time.perf_counter()
    request, direct = _sibling("task_inputs").planning_request(case[1], example, task_limits(case[1]))
    service = TaskService(
        output, bootstrap_profile=profile, policy=HostPolicy(schema_version=1, max_running=1, max_queued=3)
    )
    preview = service.plan(request)
    commands = tuple(
        TaskExecuteRequest(
            schema_version=1,
            contract="coleman-tasks-v1",
            artifact_type="execute_request",
            request_id=f"observed-{index}",
            plan_id=preview.plan_id,
        )
        for index in range(case[0])
    )
    return _Run(service, service.saved_plan(preview.plan_id), direct, commands, tracker, started, deadline)


@dataclass(frozen=True)
class _Window:
    """Freeze the last eligible views and the actual observation-window event."""

    before: tuple[TaskJobView, ...]
    peaks: tuple[int, bool]
    disposition: Literal["completed", "observation_deadline", "resource_limit"]
    seconds: float


@dataclass(frozen=True)
class _Observed:
    """Join eligible views to verified post-cleanup quiescent actual outcomes."""

    window: _Window
    after: tuple[TaskJobView, ...]


def _disposition(current: tuple[TaskJobView, ...]) -> Literal["completed", "pending", "resource_limit"]:
    """Classify only known resource dispositions; ordinary defects still raise."""
    resource_limited = False
    for view in current:
        if view.job.state == "failed" and view.job.error in RESOURCE_ERRORS:
            resource_limited = True
        elif view.job.state not in {"queued", "running", "completed"}:
            raise RuntimeError("Unexpected actual task failure cannot become a LIMITED observation")
    if resource_limited:
        return "resource_limit"
    return "completed" if all(v.job.state == "completed" and not v.worker_active for v in current) else "pending"


def _settled(run: _Run, ids: list[str]) -> tuple[TaskJobView, ...]:
    """Verify every cancelled/completed owner inactive within original180cap."""
    while time.monotonic() < run.deadline:
        current = _views(run.service, ids)
        if current and all(not view.worker_active for view in current):
            if any(view.job.state not in {"completed", "cancelled", "failed"} for view in current):
                raise RuntimeError("Cleanup left an unexpected nonterminal task disposition")
            return current
        time.sleep(min(0.025, max(0.0, run.deadline - time.monotonic())))
    raise TimeoutError("Actual workers did not become inactive within the original180-second outer cap")


def _window(run: _Run, ids, known, peaks, *, began: float | None = None) -> _Window:
    """Refuse a completion whose real status read crosses the original window."""
    began = time.monotonic() if began is None else began
    deadline = began + CELL_SECONDS
    while time.monotonic() < min(deadline, run.deadline):
        current = _views(run.service, ids)
        sampled_at = time.monotonic()
        if sampled_at >= min(deadline, run.deadline):
            break  # Late status cannot replace the last eligible pre-cleanup view.
        known.update((view.job.job_id, view) for view in current)
        peaks = _sample_peaks(current, peaks)
        before = tuple(known[identity] for identity in ids)
        state = _disposition(before)
        if state != "pending":
            return _Window(before, peaks, state, sampled_at - began)
        time.sleep(min(0.025, max(0.0, deadline - time.monotonic())))
    if time.monotonic() >= run.deadline:
        raise TimeoutError("The unchanged180-second outer task measurement allowance ended")
    return _Window(tuple(known[identity] for identity in ids), peaks, "observation_deadline", time.monotonic() - began)


def _cell(output, profile, case, example, deadline) -> TaskCell:
    """Retain an unchanged workload's completed or explicitly LIMITED outcome."""
    run = _prepare(output, profile, case, example, deadline)
    with ExitStack() as cleanup:
        began = time.monotonic()
        ids, known, peaks = _admit(run, cleanup, min(run.deadline, began + CELL_SECONDS))
        window = _window(run, ids, known, peaks, began=began)
        if window.disposition == "completed":
            result = _cell_observation(run, _Observed(window, window.before), case)
            cleanup.pop_all()
            return result
        cleanup.close()
        return _cell_observation(run, _Observed(window, _settled(run, ids)), case)


def _admit(run: _Run, cleanup: ExitStack, deadline: float):
    """Submit the fixed real burst and retain its actual individual views."""
    ids, known, peaks = [], {}, (0, False)
    for command in run.commands:
        # Admission can commit before startup/status returns an uncertain error.
        cleanup.callback(_cancel_command, run, command)
        view = _retry_metadata(run, lambda command=command: run.service.execute(command), deadline=deadline)
        if time.monotonic() >= deadline:
            raise TimeoutError("Uncertain admission crossed the original60-second observation window")
        ids.append(view.job.job_id)
        known[view.job.job_id] = view
        peaks = _sample_peaks((view,), peaks)
        current = _views(run.service, ids)
        if time.monotonic() >= deadline:
            raise TimeoutError("Admission status crossed the original60-second observation window")
        known.update((view.job.job_id, view) for view in current)
        peaks = _sample_peaks(current, peaks)
    return ids, known, peaks


def _retry_metadata(run: _Run, operation, *, deadline: float | None = None):
    """Retry only numeric metadata contention within the original180-second cap.

    Callers supply the same retained request or cancel ID on every attempt;
    this is never a retry of a mathematical update or a transaction body.
    Unexpected faults and deadline exhaustion still prevent any aggregate.
    """
    deadline = run.deadline if deadline is None else min(deadline, run.deadline)
    while time.monotonic() < deadline:
        try:
            return operation()
        except sqlite3.OperationalError as error:
            code = getattr(error, "sqlite_errorcode", 0)
            if not isinstance(code, int) or code & 0xFF not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise
            time.sleep(min(0.025, max(0.0, deadline - time.monotonic())))
    raise TimeoutError("Metadata contention exhausted the original observation/aggregate allowance")


def _cancel_command(run: _Run, command: TaskExecuteRequest) -> None:
    """Locate only this exact uncertain admission, then use real cancellation."""
    job = _retry_metadata(
        run, lambda: execution_lifecycle.retained_request(run.service.store, run.service.owner, command.request_id)
    )
    if job is None:
        return  # A refused pre-admission request did not invent a job to cancel.
    if not isinstance(job, TaskJobRecord) or (
        job.owner,
        job.request_id,
        job.request_digest,
        job.plan_id,
    ) != (run.service.owner, command.request_id, command.fingerprint, command.plan_id):
        raise ValueError("Cleanup refuses another family/owner/payload instead of cancelling an unrelated job")
    _retry_metadata(run, lambda: run.service.cancel(job.job_id))


def _read_verified(path: Path, maximum: int, expected: Fingerprint) -> bytes:
    """Acquire only the selected bounded actual ACK content, never a glob."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Acknowledged content must remain an actual regular owned file")
    with path.open("rb") as stream:
        content = stream.read(maximum + 1)
    if len(content) > maximum or "sha256:" + hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Acknowledged content differs from its exact selected bytes or capacity")
    return content


def _ack_component(directory, plan, coordinate):
    """Read the actual SQL-selected current component without restoring owners."""
    store = LocalStateStore.open(directory)
    head = store.read_head()
    if head is None:
        return None
    manifest = CollectionManifest.from_json(
        _read_verified(
            directory / "manifests" / (head.manifest_fingerprint[7:] + ".json"),
            store.limits.maximum_manifest_bytes,
            head.manifest_fingerprint,
        )
    )
    if manifest.intent.generation != head.generation or len(manifest.components) != 1:
        raise ValueError("The actual typed-Hedge acknowledgement requires its one original current owner")
    selected = manifest.components[0]
    content = _read_verified(
        directory / "blobs" / (selected.fingerprint[7:] + ".json"),
        store.limits.maximum_component_bytes,
        selected.fingerprint,
    )
    if len(content) != selected.byte_count:
        raise ValueError("Acknowledged component byte count differs from its retained manifest")
    checkpoint = importlib.import_module("coleman_typed_hedge.recipe").STATE.decode(content.decode())
    if (
        checkpoint.plan_fingerprint,
        checkpoint.coordinate_fingerprint,
        checkpoint.source,
        checkpoint.binding.owner,
        checkpoint.binding.implementation,
        checkpoint.binding,
    ) != (
        plan.fingerprint,
        coordinate.fingerprint,
        plan.source,
        coordinate.reference,
        plan.implementation.binding,
        selected.binding,
    ) or checkpoint.learner.payload.update_count > coordinate.target.position:
        raise ValueError("Acknowledged checkpoint does not belong to the exact actual task/coordinate/source")
    return checkpoint.learner.payload.update_count, len(content), selected.fingerprint


@dataclass(frozen=True)
class _Ack:
    """Retain observed accepted work and actual selected current blob bytes."""

    updates: int
    byte_count: int
    digest: Fingerprint | None


@dataclass(frozen=True)
class _Final:
    """Keep completed-only state/result verification separate from partial ACKs."""

    mathematical_digest: Fingerprint
    result_bytes: int
    checkpoint_bytes: int
    committed_digest: Fingerprint


def _acknowledged(run: _Run, view: TaskJobView) -> _Ack:
    """Count actual SQL ACKs, including partial coordinates and real U0 states."""
    if run.service.bootstrap_profile is None:
        raise ValueError("Actual acknowledged state requires the explicit current approved host profile")
    run.plan.validate_bootstrap(run.service.bootstrap_profile)
    view.job.validate_for(run.plan)
    updates = byte_count = 0
    records = []
    for index, coordinate in enumerate(run.plan.coordinates):
        directory = (
            run.service.store.root / "jobs" / view.job.job_id / "evidence" / coordinate.fingerprint[7:] / "state"
        )
        selected = _ack_component(directory, run.plan, coordinate) if directory.exists() else None
        required = coordinate.target.position if index < view.job.completed_coordinates else None
        if coordinate.fingerprint == view.job.current_coordinate and view.job.frontier is not None:
            required = view.job.frontier.position if required is None else max(required, view.job.frontier.position)
        if required is not None and (selected is None or selected[0] < required):
            raise ValueError("Retained outer ACK progress has missing or regressed actual acknowledged state")
        if selected is not None:
            updates += selected[0]
            byte_count += selected[1]
            records.append([coordinate.fingerprint, selected[2]])
    return _Ack(updates, byte_count, fingerprint(records) if records else None)


def _final_result(run: _Run, command: TaskExecuteRequest, view: TaskJobView):
    """Validate complete results/direct parity only when completion is actual."""
    if view.job.state != "completed":
        return None
    completion = run.service.report(view.job.job_id)
    report = cast(Any, run.service.decode_result(view.job.job_id))
    states = tuple(item.checkpoint.learner.to_json() for item in report.results)
    if states != run.direct:
        raise ValueError("Real task complete states differ from original direct Hedge states")
    retried = _retry_metadata(run, lambda: run.service.execute(command))
    if retried.job.job_id != view.job.job_id or retried.job.result_digest != view.job.result_digest:
        raise ValueError("Actual retained retry changed logical identity or completed result")
    return _Final(
        mathematical_digest=_state_digest(states),
        result_bytes=len(completion.to_json().encode()),
        checkpoint_bytes=sum(len(item.checkpoint.to_json().encode()) for item in report.results),
        committed_digest=fingerprint(
            [[item.checkpoint.coordinate_fingerprint, item.checkpoint.fingerprint] for item in report.results]
        ),
    )


def _job_observation(run: _Run, command: TaskExecuteRequest, views: tuple[TaskJobView, TaskJobView]) -> JobObservation:
    """Retain actual quiescent state, ACK work and completed-only verification."""
    before, view = views
    progress = _acknowledged(run, view)
    result = _final_result(run, command, view)
    if result is not None and result.committed_digest != progress.digest:
        raise ValueError("Completed result checkpoints differ from the actually selected current SQL acknowledgements")
    files, size_bytes = _occupancy(run.service.store.root / "jobs" / view.job.job_id)
    return JobObservation(
        schema_version=1,
        job=view.job,
        before_cleanup=before,
        worker_active=view.worker_active,
        planned_updates=sum(row.target.position for row in run.plan.coordinates),
        accepted_updates=progress.updates,
        acknowledged_checkpoint_bytes=progress.byte_count,
        committed_state_digest=progress.digest,
        complete_state_status="verified" if result is not None else "not_applicable",
        mathematical_digest=None if result is None else result.mathematical_digest,
        result_bytes=None if result is None else result.result_bytes,
        final_checkpoint_representation_bytes=None if result is None else result.checkpoint_bytes,
        age_seconds=view.elapsed_seconds,
        artifact_files=files,
        artifact_bytes=size_bytes,
    )


def _cell_observation(run, observed: _Observed, case) -> TaskCell:
    """Join every actual job after complete or declared-limit cleanup."""
    observations = tuple(
        _job_observation(run, command, (before, after))
        for command, before, after in zip(run.commands, observed.window.before, observed.after, strict=True)
    )
    resources = run.tracker.sample()
    return TaskCell(
        schema_version=1,
        complete=observed.window.disposition == "completed",
        disposition=observed.window.disposition,
        observation_status="observed",
        batch=case[0],
        size=case[1],
        planned_updates=case[0] * sum(row.target.position for row in run.plan.coordinates),
        input_bytes=len(run.plan.inputs.body.encode()),
        max_queued_views_per_sweep=observed.window.peaks[0],
        worker_owner_observed=observed.window.peaks[1],
        elapsed_seconds=time.perf_counter() - run.started,
        observation_elapsed_seconds=observed.window.seconds,
        observation_limit_seconds=60.0,
        host_cpu_seconds=resources.cpu_time_seconds,
        host_peak_rss_mib=resources.peak_rss_mib,
        worker_cpu_seconds=None,
        worker_peak_rss_mib=None,
        worker_resource_status="unavailable",
        task_limits=run.plan.limits,
        jobs=observations,
    )


def _not_started(case, remaining: float) -> NotStartedCell:
    """Retain an unfunded original case without inventing admitted jobs/work."""
    return NotStartedCell(
        schema_version=1,
        complete=False,
        disposition="not_started",
        reason="outer_budget",
        observation_status="unavailable",
        batch=case[0],
        size=case[1],
        planned_updates=case[0] * (1 if case[1] == "small" else 32),
        remaining_seconds=max(0.0, remaining),
        jobs=(),
        input_bytes=None,
        elapsed_seconds=None,
        observation_elapsed_seconds=None,
        observation_limit_seconds=60.0,
        host_cpu_seconds=None,
        host_peak_rss_mib=None,
        max_queued_views_per_sweep=None,
        worker_owner_observed=None,
    )


def _measure_cells(output, profile, example, deadline):
    """Observe exactly the original grid, retaining each outcome before the next."""
    cells = []
    for batch, size in CASES:
        remaining = deadline - time.monotonic()
        cell = (
            _not_started((batch, size), remaining)
            if remaining < CELL_SECONDS + ENTRY_RESERVE_SECONDS
            else _cell(output / f"batch-{batch}-{size}", profile, (batch, size), example, deadline)
        )
        ArtifactWriter(output).write_immutable(output / f"cell-{batch}-{size}.json", cell.to_json())
        cells.append(cell)
    return tuple(cells)


def measure(profile_path: Path, output: Path) -> TaskLoadReport:
    """Measure four real installed task/queue/input/artifact workload cells.

    Parameters
    ----------
    profile_path : pathlib.Path
        Independently operator-approved existing typed-Hedge host profile,
        bounded to16KiB. It is explicit host authority, never task-provided code.
    output : pathlib.Path
        New private root. Actual task workspaces/logs survive failure, without a
        completed report. Every cell retains original60-second worker guards.

    Returns
    -------
    TaskLoadReport
        All four actual cells with completed or explicitly LIMITED outcomes,
        planned versus SQL-acknowledged work, inactive owners and host-only
        CPU/RSS. Only completed jobs receive full direct-state verification.
        Repetitions are not independent histories or evidence of superiority.

    Raises
    ------
    ValueError, OSError, RuntimeError, TimeoutError
        Invalid/changed installed approval, provider/input decoding refusal,
        unavailable runtime, unexpected worker/validation defects or failure to
        clean up within the unchanged180-second outer allowance. A declared
        observation deadline/resource disposition instead yields LIMITED.

    Notes
    -----
    Uses shipped TaskService/SQLite/supervisor/worker/state ACK/resource guards
    and existing independently installed public typed provider unchanged. No
    start callback, sampler, memory observation or principal is mocked. Every
    case keeps its60-second observation/worker caps. Cleanup and verification
    consume only the remaining original180-second aggregate allowance.
    Partial cells publish limited.json with complete=False, never complete.json.
    Unexpected exceptions still propagate without an aggregate report.
    """
    deadline = time.monotonic() + OUTER_SECONDS
    with profile_path.open("rb") as stream:
        profile = HostBootstrapProfile.from_json(stream.read(16 * 1024 + 1))
    if len(profile.executors) != 1:
        raise ValueError("Task measurement requires exactly its explicitly approved typed-Hedge recipe")
    approval = profile.executors[0]
    if approval.recipe.reference.id != "external-constant-hedge":
        raise ValueError("Only the existing public typed-Hedge executor is supported")
    image = capture_executor_identity(approval.recipe)
    load_approved_executor(approval, expected=image)
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    cells = _measure_cells(output, profile, _example(), deadline)
    complete = all(cell.complete for cell in cells)
    observed = all(
        isinstance(cell, TaskCell) and cell.max_queued_views_per_sweep > 0 for cell in cells if cell.batch == 3
    )
    report = TaskLoadReport(
        schema_version=1,
        complete=complete,
        outcome="completed" if complete else "limited",
        planned_updates=132,
        outer_limit_seconds=180.0,
        entry_reserve_seconds=30.0,
        acknowledged_update_lower_bound=sum(row.accepted_updates for cell in cells for row in cell.jobs),
        runtime=InstalledRuntime.capture(),
        bootstrap=profile,
        implementation=image,
        hardware=f"{platform.platform()}; processor={platform.processor()}; logical_cpus={os.cpu_count()}",
        native_threads=tuple(sorted((name, value) for name, value in os.environ.items() if name in THREAD_VARIABLES)),
        host_policy=HostPolicy(schema_version=1, max_running=1, max_queued=3),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        example_sha256=hashlib.sha256(Path(__file__).with_name("ci.py").read_bytes()).hexdigest(),
        helper_sources=tuple(
            (name, hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest())
            for name in ("task_inputs.py", "task_records.py")
        ),
        queue_evidence="observed" if observed else "insufficient",
        cells=cells,
    )
    if time.monotonic() >= deadline:
        raise TimeoutError("The original180-second task measurement allowance ended before publication")
    _publish(report, output)
    return report


def _publish(report: TaskLoadReport, output: Path) -> Path:
    """Validate the whole grid before choosing complete versus LIMITED output."""
    current = TaskLoadReport.model_validate(report)
    target = "complete.json" if current.complete else "limited.json"
    path = output / target
    ArtifactWriter(output).write_immutable(path, current.to_json())
    return path


def main() -> None:
    """Run explicit trusted local load qualification, without package installs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(measure(arguments.profile, arguments.output).to_json())


if __name__ == "__main__":
    main()
