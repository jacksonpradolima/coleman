"""One bounded job, coordinate, attempt and complete-last execution lifecycle."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import closing, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coleman.application import execution_views as views
from coleman.application import resource_usage
from coleman.application._workflow_records import require_record_plan
from coleman.application.execution_program import CoordinateWork, ExecutionProgram
from coleman.application.workflow_store import WorkflowStore
from coleman.artifacts import ArtifactWriter
from coleman.core.primitives import canonical_json
from coleman.tasks.contracts import EncodedValue, TaskFrontier, TaskJobRecord, TaskPlan
from coleman.tasks.identity import HostBootstrapProfile
from coleman.tasks.protocols import AttemptContext, CommittedProgress, CoordinateContext, decode_value
from coleman.workflows.contracts import MAX_WORKFLOW_ARTIFACT_ENTRIES, WorkflowError

type Program = ExecutionProgram[Any, Any, Any, Any]
type Work = CoordinateWork[Any]
type ProgressHook = Callable[[str, int], None]


def read_job(store: WorkflowStore, job_id: str) -> views.Job:
    """Read exact known metadata through the shared store, without providers.

    Parameters
    ----------
    store : WorkflowStore
        Existing bounded metadata owner; opens one short-lived read connection.
    job_id : str
        Exact scheduler-issued identity, never an executable selector.

    Returns
    -------
    JobRecord or TaskJobRecord
        Complete retained known record, with indexed identities checked.

    Raises
    ------
    StopIteration, ValueError, OSError, sqlite3.Error
        Missing identity or corrupt/unavailable storage. No source or provider
        is loaded and no metadata is changed.
    """
    with closing(store.connect()) as connection:
        return next(job for job in store.records(connection) if job.job_id == job_id)


@dataclass(frozen=True)
class WorkerHooks:
    """Preserve live legacy read/resource seams without importing native code.

    Attributes
    ----------
    read_job : callable
        Bounded known metadata reader; legacy wrappers retain their actual hook.
    peak_rss_mib : callable
        Own-executable peak MiB or unavailable None; zero remains observed.
    output_exceeds : callable
        Existing sampled byte/entry/symlink refusal predicate, without writes.

    Notes
    -----
    Trusted callbacks are never selected from request JSON or invoked in SQL.
    """

    read_job: Callable[[WorkflowStore, str], views.Job] = read_job
    peak_rss_mib: Callable[[], float | None] = resource_usage.peak_rss_mib
    output_exceeds: Callable[..., bool] = resource_usage.output_exceeds


@dataclass(frozen=True)
class _Worker:
    """Hold one claimed job's immutable plan, common caps and trusted hooks."""

    store: WorkflowStore
    job: views.Job
    plan: views.Plan
    deadline: float
    hooks: WorkerHooks
    progress_hook: ProgressHook | None

    def transition(self, transform: Callable[[views.Job], views.Job], *, deadline: float | None = None) -> views.Job:
        """Serialize pure metadata; cancellation wins after its transaction commits."""
        with self.store.transaction(deadline=deadline) as connection:
            current = next(item for item in self.store.records(connection) if item.job_id == self.job.job_id)
            if current.state != "running":
                raise InterruptedError("Workflow is no longer running")
            require_record_plan(current, self.plan)
            updated = transform(current)
            self.store.save_record(connection, updated)
        return updated

    def check_output(self) -> None:
        """Refuse sampled output/deadline overshoot at common phase boundaries."""
        if time.monotonic() > self.deadline:
            raise WorkflowError("Workflow observed wall-time allowance exhausted")
        if self.hooks.output_exceeds(
            views.limits(self.plan).max_output_bytes,
            root=self.store.root / "jobs" / self.job.job_id,
            maximum_entries=MAX_WORKFLOW_ARTIFACT_ENTRIES,
        ):
            raise WorkflowError("Workflow output byte/file limit exceeded")

    def check(self) -> None:
        """Check actual cancellation and sampled common memory/output/deadline."""
        if self.hooks.read_job(self.store, self.job.job_id).state != "running":
            raise InterruptedError("Workflow is no longer running")
        peak = self.hooks.peak_rss_mib()
        if peak is None or peak * 1048576 > views.limits(self.plan).max_memory_bytes:
            raise WorkflowError("Workflow sampled memory allowance exhausted or unavailable")
        self.check_output()

    def guard(self, stopped: threading.Event) -> None:
        """Retain the actual fail-closed process guard and native exit codes."""
        while not stopped.wait(0.02):
            try:
                peak = self.hooks.peak_rss_mib()
                refused = (
                    peak is None
                    or peak * 1048576 > views.limits(self.plan).max_memory_bytes
                    or time.monotonic() > self.deadline
                    or self.hooks.output_exceeds(
                        views.limits(self.plan).max_output_bytes,
                        root=self.store.root / "jobs" / self.job.job_id,
                        maximum_entries=MAX_WORKFLOW_ARTIFACT_ENTRIES,
                    )
                )
                state = self.hooks.read_job(self.store, self.job.job_id).state
                if state == "cancelled":
                    os._exit(75)
                if state != "running":
                    return
                if refused:
                    self.transition(lambda job: views.changed(job, state="failed", error="sampled_resource_limit"))
                    os._exit(74)
            except (OSError, ValueError, sqlite3.Error):
                os._exit(74)

    def reserve(self, work: Work) -> str:
        """Prepare bytes before the serialized actual pre-work reservation."""
        root = self.store.root / "jobs" / self.job.job_id
        directory = views.attempt_directory(root, work.coordinate_fingerprint)
        count = len(list(directory.glob("*.json")))
        if count >= views.limits(self.plan).max_attempts:
            raise WorkflowError("Physical trial attempt limit exhausted")
        current = self.hooks.read_job(self.store, self.job.job_id)
        identity, payload = views.attempt_payload(current, self.plan, work.coordinate_fingerprint, count + 1)
        with self.store.transaction(deadline=self.deadline) as connection:
            selected = next(item for item in self.store.records(connection) if item.job_id == self.job.job_id)
            if selected.state != "running":
                raise InterruptedError("Workflow cancelled before native invocation")
            require_record_plan(selected, self.plan)
            if (selected.stage, selected.invocations) != (current.stage, current.invocations):
                raise WorkflowError("Worker reservation changed its actual invocation or stage")
            ArtifactWriter(root).write_immutable(directory / f"{count + 1:04d}.json", payload)
            self.store.save_record(connection, views.reserved(selected, work.coordinate_fingerprint))
        return identity

    def record_progress(
        self,
        program: Program,
        context: CoordinateContext,
        progress: CommittedProgress[Any],
        *,
        notify_hook: bool = True,
    ) -> None:
        """Verify actual progress outside SQL, then retain its logical work."""
        if (progress.plan_fingerprint, progress.coordinate_fingerprint) != (
            program.context.plan_fingerprint,
            context.coordinate_fingerprint,
        ):
            raise WorkflowError("Committed progress differs from the retained plan or coordinate")
        frontier = TaskFrontier.model_validate(progress.frontier)
        if (
            frontier.profile != context.target.profile
            or frontier.coordinate != context.reference
            or frontier.position > context.target.position
        ):
            raise WorkflowError("Committed progress differs from the declared target frontier")
        verifier = program.executor.progress_verifier
        if verifier is None:
            raise WorkflowError("Stateless executors cannot submit unverified committed progress")
        verifier.validate(progress, context)
        self.transition(
            lambda job: views.progressed(job, context.coordinate_fingerprint, frontier), deadline=self.deadline
        )
        self.check()
        if notify_hook and self.progress_hook is not None and not isinstance(self.plan, TaskPlan):
            self.progress_hook(context.coordinate_fingerprint, frontier.position)

    def drive_coordinates(self, program: Program) -> tuple[Any, ...]:
        """Use one finite coordinate loop for both native and generic work."""
        receipts = []
        for index, work in enumerate(program.coordinates):
            self.check()
            context = CoordinateContext(
                program.context,
                work.coordinate_fingerprint,
                work.reference,
                work.target,
                work.artifact_directory,
                seed=work.seed,
            )
            result = program.executor.read_completed(work.value, context)
            if result is None:
                views.require_pending_coordinate(self.job, index)
                self._reconcile(program, context)
                identity = self.reserve(work)
                attempt = AttemptContext(context, identity, self.job.invocations, self.deadline, self.check)
                result = program.executor.advance(
                    work.value, attempt, lambda progress, ctx=context: self.record_progress(program, ctx, progress)
                )
                self.check()
                program.executor.publish_coordinate(work.value, result, context)
                self.check()
            receipts.append(result)
            self.transition(lambda job: views.coordinate_completed(job, len(receipts)), deadline=self.deadline)
        return tuple(receipts)

    def _reconcile(self, program: Program, context: CoordinateContext) -> None:
        """Close a verified inner-ACK/outer-progress gap without another update."""
        verifier = program.executor.progress_verifier
        if verifier is None:
            return
        current = self.hooks.read_job(self.store, self.job.job_id)
        if views.current_coordinate(current) != context.coordinate_fingerprint:
            return
        actual = verifier.reconcile(context)
        if actual is None:
            if views.position(current, context.coordinate_fingerprint):
                raise WorkflowError("Previously acknowledged task state is missing; implicit reset is forbidden")
            return
        self.record_progress(program, context, actual, notify_hook=False)

    def publish(self, program: Program, result: Any) -> None:
        """Compute complete typed bytes/digest before the cancellation transaction."""
        if program.finish_state == "awaiting_evaluation":
            self.transition(lambda job: views.changed(job, state="awaiting_evaluation"), deadline=self.deadline)
            return
        value = program.output_codec.validate(result)
        payload = program.output_codec.encode(value)
        if len(payload.encode("utf-8")) > 16777216:
            raise WorkflowError("Workflow report exceeds its 16 MiB reader limit; reduce the plan/history")
        decode_value(program.output_codec, EncodedValue(codec=program.output_codec.declaration, body=payload))
        identity = program.output_fingerprint(value)
        self.check()
        with self.store.transaction(deadline=self.deadline) as connection:
            current = next(item for item in self.store.records(connection) if item.job_id == self.job.job_id)
            if current.state != "running":
                raise InterruptedError("Cancellation won before report publication")
            require_record_plan(current, self.plan)
            ArtifactWriter(self.store.root).write_immutable(program.output_path, payload)
            self.check_output()
            done = views.coordinate_completed(current, len(program.coordinates))
            self.store.save_record(connection, views.changed(done, state="completed", result_digest=identity))


def _worker_context(
    store: WorkflowStore, job_id: str, began: float, hooks: WorkerHooks
) -> tuple[views.Job, views.Plan, float] | None:
    """Observe the committed startup claim through the original bounded barrier."""
    job = hooks.read_job(store, job_id)
    if job.state not in ("queued", "running"):
        return None
    with closing(store.connect()) as connection:
        plan = store.read_plan(connection, job.plan_id, job.owner)
    require_record_plan(job, plan)
    deadline = began + views.limits(plan).attempt_seconds
    if job.state == "queued":
        with store.transaction(deadline=deadline) as connection:
            current = next(item for item in store.records(connection) if item.job_id == job_id)
            if (current.plan_id, current.owner) != (job.plan_id, job.owner):
                raise WorkflowError("Worker startup claim differs from its immutable admitted plan")
            job = current
    if job.state != "running":
        return None
    return job, plan, deadline


def _program(worker: _Worker, bootstrap_profile: HostBootstrapProfile | None) -> Program:
    """Resolve known metadata only; never import code named by stored JSON."""
    if isinstance(worker.plan, TaskPlan):
        from coleman.application.task_runtime import prepare_task_program

        if not isinstance(worker.job, TaskJobRecord):
            raise WorkflowError("Task plan requires its exact task metadata record")
        return prepare_task_program(
            worker.store,
            worker.job,
            worker.plan,
            bootstrap_profile=bootstrap_profile,
            guard=worker.check,
        )
    from coleman.application.native_executor import prepare_native_program

    if isinstance(worker.job, TaskJobRecord):
        raise WorkflowError("Native plan requires its exact native metadata record")
    return prepare_native_program(worker.store, worker.job, worker.plan, deadline=worker.deadline, guard=worker.check)


def _produce(worker: _Worker, program: Program) -> Any:
    """Install the at-most-once common loader before adapter scope callbacks."""
    called = False

    def load_completed() -> tuple[Any, ...]:
        """Reject a second stage call before additional coordinate effects."""
        nonlocal called
        if called:
            raise WorkflowError("Execution boundary cannot load its coordinates more than once")
        called = True
        return worker.drive_coordinates(program)

    return program.executor.boundary.produce(load_completed)


def _failure(worker: _Worker, error: Exception) -> None:
    """Retain bounded private failure details without manufacturing completion."""
    root, job = worker.store.root, worker.job
    ArtifactWriter(root).write_immutable(
        root / "jobs" / job.job_id / f"failure-{job.invocations:04d}.json",
        canonical_json({"exception": type(error).__name__, "detail": str(error)[:1000]}),
    )
    prefix = "task_execution_" if isinstance(worker.plan, TaskPlan) else "native_workflow_"
    with suppress(InterruptedError, WorkflowError, sqlite3.Error):
        worker.transition(
            lambda current: views.changed(
                current,
                state="interrupted" if isinstance(error, InterruptedError) else "failed",
                error=str(error)[:128] if isinstance(error, WorkflowError) else prefix + type(error).__name__,
            ),
            deadline=worker.deadline if time.monotonic() < worker.deadline else None,
        )


def run_with_hooks(
    root: Path,
    job_id: str,
    *,
    progress_hook: ProgressHook | None = None,
    bootstrap_profile: HostBootstrapProfile | None = None,
    hooks: WorkerHooks,
) -> None:
    """Run the one engine with trusted compatibility hooks, never a second loop.

    Parameters
    ----------
    root : pathlib.Path
        Existing trusted workspace; caller already owns its actual job lock.
    job_id : str
        Exact scheduler-issued metadata identity.
    progress_hook : callable or None, optional
        Native-only post-checkpoint/metadata/guard interruption hook.
    bootstrap_profile : HostBootstrapProfile or None, optional
        Independently approved generic host profile, never loaded from SQL.
    hooks : WorkerHooks
        Host-selected live read/resource functions preserving legacy seams.

    Notes
    -----
    Uses the same startup, coordinate, resource and publication implementation
    as run_job. Later errors publish bounded private failure evidence; initial
    workspace/claim failures propagate. No provider callback executes in SQL.
    """
    began = time.monotonic()
    store = WorkflowStore(root)
    context = _worker_context(store, job_id, began, hooks)
    if context is None:
        return
    worker = _Worker(store, *context, hooks, progress_hook)
    stopped = threading.Event()
    guard = threading.Thread(target=worker.guard, args=(stopped,), daemon=True)
    try:
        worker.check_output()
        guard.start()
        program = _program(worker, bootstrap_profile)
        if isinstance(worker.plan, TaskPlan) and progress_hook is not None:
            raise WorkflowError("The native build hook cannot receive generic task progress")
        count = views.retained_attempts(store.root / "jobs" / job_id, worker.plan, job_id=job_id)
        worker.transition(lambda current: views.changed(current, attempts=count), deadline=worker.deadline)
        worker.publish(program, _produce(worker, program))
    except Exception as error:
        _failure(worker, error)
    finally:
        stopped.set()
        if guard.is_alive():
            guard.join(timeout=1)


def run_job(
    root: Path,
    job_id: str,
    *,
    progress_hook: ProgressHook | None = None,
    bootstrap_profile: HostBootstrapProfile | None = None,
) -> None:
    """Execute an already kernel-owned native or generic job through one driver.

    Parameters
    ----------
    root : pathlib.Path
        Trusted persistent workspace; the caller already owns the actual job lock.
    job_id : str
        Scheduler-admitted identity; queued startup waits for claim commit/rollback.
    progress_hook : callable or None, optional
        Native-only hook after actual checkpoint, outer metadata and resource
        guard. Generic work must not use native build-ordinal semantics.
    bootstrap_profile : HostBootstrapProfile or None, optional
        Independently operator-selected generic approval; stored equality metadata
        never supplies authority. Native execution retains its original route.

    Raises
    ------
    ValueError, OSError, sqlite3.Error
        Invalid initial workspace/claim. Later failures retain bounded private
        evidence and failed/interrupted metadata, never a false complete result.

    Notes
    -----
    One loop owns receipt reuse, physical reservations, resources, actual progress
    verification and complete-last publication. Runtime/source checks precede
    restore. Codecs/factories/progress verification and final bytes/digest execute
    outside SQL bodies. Cancellation/completion serialize in existing SQLite.
    A crash before a durable commit may repeat computation; physical exactly-once
    and hostile-code sandbox guarantees are not claimed. Native RNG is reused.
    """
    run_with_hooks(root, job_id, progress_hook=progress_hook, bootstrap_profile=bootstrap_profile, hooks=WorkerHooks())
