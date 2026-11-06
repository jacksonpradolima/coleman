"""Bounded physical workflow attempts using actual native execution and checkpoints."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import closing, suppress
from pathlib import Path

import polars as pl

from coleman.application.worker import _output_exceeds, _worker_peak_rss_mib
from coleman.application.workflow_artifacts import pin_file, publish_trial, read_attempts, verify_report, verify_trial
from coleman.application.workflow_native import execute_native
from coleman.application.workflow_store import JobRecord, WorkflowStore
from coleman.artifacts import ArtifactWriter
from coleman.results.domain import canonical_json, fingerprint
from coleman.spec.selected_runtime import require_native_runtime
from coleman.workflows.contracts import MAX_WORKFLOW_ARTIFACT_ENTRIES, ExperimentPlan, Trial, WorkflowError
from coleman.workflows.lifecycle import PhysicalAttempt, TrialResult, WorkflowReport, report_assumptions

_COMPLETE_RECEIPT = "complete.json"


def _read_job(store: WorkflowStore, job_id: str) -> JobRecord:
    """Read a bounded validated private record without modifying ownership."""
    with closing(store.connect()) as connection:
        return next(job for job in store.jobs(connection) if job.job_id == job_id)


def _transition(store: WorkflowStore, job_id: str, *, deadline: float | None = None, **changes) -> JobRecord:
    """Serialize one worker transition; cancellation wins after its transaction commits."""
    with store.transaction(deadline=deadline) as connection:
        job = next(job for job in store.jobs(connection) if job.job_id == job_id)
        if job.state != "running":
            raise InterruptedError("Workflow is no longer running")
        if "completed_trials" in changes:
            changes["completed_trials"] = max(job.completed_trials, changes["completed_trials"])
        updated = JobRecord.model_validate({**job.model_dump(), **changes, "changed": time.time()})
        store.save_job(connection, updated)
        return updated


def _guard(store: WorkflowStore, job_id: str, plan: ExperimentPlan, deadline: float, stopped: threading.Event) -> None:
    """Fail closed on sampled resource exhaustion/cancellation, leaving no false completion."""
    while not stopped.wait(0.02):
        try:
            peak = _worker_peak_rss_mib()
            refused = (
                peak is None
                or peak * 1048576 > plan.request.limits.max_memory_bytes
                or time.monotonic() > deadline
                or _output_exceeds(
                    plan.request.limits.max_output_bytes,
                    root=store.root / "jobs" / job_id,
                    maximum_entries=MAX_WORKFLOW_ARTIFACT_ENTRIES,
                )
            )
            state = _read_job(store, job_id).state
            if state == "cancelled":
                os._exit(75)
            if state != "running":
                return
            if refused:
                _transition(store, job_id, state="failed", error="sampled_resource_limit")
                os._exit(74)
        except (OSError, ValueError, sqlite3.Error):
            os._exit(74)


def _input(store: WorkflowStore, root: Path, plan: ExperimentPlan) -> None:
    """Verify the original snapshot before copying it into a stable private native path."""
    source = store.root / "datasets" / f"{plan.dataset.sha256[7:]}.parquet"
    pin = pin_file(store.root, source)
    if (pin.sha256, pin.size_bytes) != (plan.dataset.sha256, plan.dataset.size_bytes):
        raise WorkflowError("Registered dataset changed; continuation refused")
    destination = root / "inputs" / plan.dataset.dataset_id / "features-engineered.parquet"
    with source.open("rb") as stream:
        ArtifactWriter(root).write_immutable_stream(
            destination, iter(lambda: stream.read(1048576), b""), maximum_bytes=plan.dataset.size_bytes
        )


def _check_output(store: WorkflowStore, job_id: str, plan: ExperimentPlan, deadline: float) -> None:
    """Reject cooperative deadline and publication-byte overshoot at actual phase boundaries."""
    if time.monotonic() > deadline:
        raise WorkflowError("Workflow observed wall-time allowance exhausted")
    if _output_exceeds(
        plan.request.limits.max_output_bytes,
        root=store.root / "jobs" / job_id,
        maximum_entries=MAX_WORKFLOW_ARTIFACT_ENTRIES,
    ):
        raise WorkflowError("Workflow output byte/file limit exceeded")


def _require_committed_progress(root: Path, job: JobRecord, trial: Trial) -> None:
    """Reject losing a native frontier already acknowledged by durable job metadata.

    A physical attempt interrupted before its first checkpoint may retry from
    the beginning. Once the application has recorded an actual native commit,
    missing progress cannot silently turn continuation into fresh learning.
    Existing native recovery still validates the checkpoint and its byte binding.
    """
    if job.current_trial == trial.trial_id and job.checkpoint_step > 0:
        markers = list((root / "native" / trial.trial_id[7:]).rglob("progress_ex1.json"))
        if len(markers) != 1:
            raise WorkflowError("Previously committed native checkpoint is missing; implicit reset is forbidden")


def _prefix_steps(plan: ExperimentPlan) -> int:
    """Count the already eligible chronological train/validation frontier without reading outcomes."""
    assert plan.dataset.temporal_plan is not None
    return sum(item.partition != "test" for item in plan.dataset.split_by_build())


def _reserve_attempt(
    store: WorkflowStore, job: JobRecord, plan: ExperimentPlan, trial: Trial, *, deadline: float | None = None
) -> str:
    """Reserve one bounded physical invocation before native work; cancellation wins transactionally."""
    root = store.root / "jobs" / job.job_id
    with store.transaction(deadline=deadline) as connection:
        current = next(item for item in store.jobs(connection) if item.job_id == job.job_id)
        if current.state != "running":
            raise InterruptedError("Workflow cancelled before native invocation")
        events = root / "attempts" / trial.trial_id[7:]
        count = len(list(events.glob("*.json")))
        if count >= plan.request.limits.max_attempts:
            raise WorkflowError("Physical trial attempt limit exhausted")
        attempt_id = fingerprint(["workflow.physical.v1", job.job_id, trial.trial_id, count + 1])[7:]
        ArtifactWriter(root).write_immutable(
            events / f"{count + 1:04d}.json",
            PhysicalAttempt(
                schema_version=1,
                trial_id=trial.trial_id,
                attempt_id=attempt_id,
                plan_fingerprint=plan.fingerprint,
                stage=current.stage,
                worker_invocation=current.invocations,
                ordinal=count + 1,
                started_epoch_seconds=time.time(),
            ).to_json(),
        )
        store.save_job(
            connection,
            JobRecord.model_validate(
                {
                    **current.model_dump(),
                    "attempts": current.attempts + 1,
                    "current_trial": trial.trial_id,
                    "checkpoint_step": job.checkpoint_step if job.current_trial == trial.trial_id else 0,
                    "changed": time.time(),
                }
            ),
        )

    return attempt_id


def _execute_trial(store, job, plan, trial, steps, progress_hook, deadline) -> TrialResult:
    """Continue actual owned native state and publish one immutable stage receipt last."""
    root = store.root / "jobs" / job.job_id
    _require_committed_progress(root, job, trial)
    if job.stage == "evaluation":
        from coleman.application.workflow_evaluation import require_continuation

        require_continuation(root, trial, _prefix_steps(plan))
    attempt_id = _reserve_attempt(store, job, plan, trial, deadline=deadline)

    def progress(step: int) -> None:
        """Record only a real native commit before invoking the trusted interruption hook."""
        _transition(store, job.job_id, deadline=deadline, checkpoint_step=step)
        _check_output(store, job.job_id, plan, deadline)
        if progress_hook is not None:
            progress_hook(trial.trial_id, step)

    minimum_step = job.checkpoint_step if job.current_trial == trial.trial_id else 0
    if job.stage == "evaluation":
        minimum_step = max(minimum_step, _prefix_steps(plan))
    native = execute_native(root, plan, trial, steps, progress, attempt_id=attempt_id, minimum_step=minimum_step)
    _check_output(store, job.job_id, plan, deadline)
    result = publish_trial(root, native, plan, trial, job.stage, steps)
    _check_output(store, job.job_id, plan, deadline)
    return result


def _trials(
    store: WorkflowStore,
    job: JobRecord,
    plan: ExperimentPlan,
    progress_hook: Callable[[str, int], None] | None,
    trials: tuple[Trial, ...] | None = None,
    *,
    deadline: float,
) -> tuple[TrialResult, ...]:
    """Reuse complete receipts and invoke only unfinished coordinates with retained attempts."""
    root = store.root / "jobs" / job.job_id
    steps = _prefix_steps(plan) if job.stage == "tuning" else len(plan.dataset.build_ids)
    receipts = []
    for trial in plan.trials if trials is None else trials:
        receipt = root / "evidence" / trial.trial_id[7:] / job.stage / _COMPLETE_RECEIPT
        if receipt.exists():
            result = verify_trial(root, receipt, plan, trial, job.stage)
            if len(result.rows) != steps:
                raise WorkflowError("Completed workflow stage has an incompatible frontier")
        else:
            result = _execute_trial(store, job, plan, trial, steps, progress_hook, deadline)
        receipts.append(result)
        _transition(store, job.job_id, deadline=deadline, completed_trials=len(receipts), current_trial=None)
    return tuple(receipts)


def _produce_report(store, job, plan, progress_hook, deadline):
    """Reuse a valid complete report or execute the actual declared native/scientific stage."""
    root, job_id = store.root, job.job_id
    from coleman.application.baseline_storage import read_baseline_artifact
    from coleman.application.workflow_evaluation import (
        EvaluationHandoff,
        choose_configuration,
        compare_continuation,
        evaluation_trials,
        prepare_evaluation_attempt,
    )

    complete_path = root / "jobs" / job_id / _COMPLETE_RECEIPT
    if complete_path.exists():
        report = read_baseline_artifact(complete_path, WorkflowReport)
        verify_report(root / "jobs" / job_id, plan, report)
        if report.comparison is not None:
            from coleman.evaluation.protected import EvaluationAudit

            EvaluationAudit(root / "audit").save_report(report.comparison)
        _transition(store, job_id, deadline=deadline, completed_trials=len(report.trials), current_trial=None)
    elif job.stage == "evaluation":
        handoff = read_baseline_artifact(root / "jobs" / job_id / "evaluation-handoff.json", EvaluationHandoff)
        handoff = prepare_evaluation_attempt(root, root / "jobs" / job_id, plan, handoff, job.invocations)
        _check_output(store, job_id, plan, deadline)
        report = compare_continuation(
            root,
            root / "jobs" / job_id,
            plan,
            handoff,
            lambda: _trials(store, job, plan, progress_hook, evaluation_trials(plan, handoff), deadline=deadline),
        )
    else:
        results = _trials(store, job, plan, progress_hook, deadline=deadline)
        selected = choose_configuration(plan, results)
        selected_budget = next(trial.budget_value for trial in plan.trials if trial.configuration_id == selected)
        fallback = next(
            trial.configuration_id
            for trial in plan.trials
            if trial.policy == "Random" and trial.budget_value == selected_budget
        )
        report = WorkflowReport(
            schema_version=1,
            plan_fingerprint=plan.fingerprint,
            trials=results,
            selected_configuration=selected,
            fallback_configuration=fallback,
            physical_attempts=read_attempts(root / "jobs" / job_id, plan),
            interpretation="descriptive_only",
            assumptions=report_assumptions(plan),
        )
    return report


def _publish_completion(store, job_id, plan, report, deadline) -> None:
    """Publish complete bytes before the serialized completion transition; prior cancellation wins."""
    root = store.root
    with store.transaction(deadline=deadline) as connection:
        current = next(item for item in store.jobs(connection) if item.job_id == job_id)
        if current.state != "running":
            raise InterruptedError("Cancellation won before report publication")
        payload = report.to_json()
        if len(payload.encode()) > 16777216:
            raise WorkflowError("Workflow report exceeds its 16 MiB reader limit; reduce the plan/history")
        ArtifactWriter(root).write_immutable(root / "jobs" / job_id / _COMPLETE_RECEIPT, payload)
        _check_output(store, job_id, plan, deadline)
        store.save_job(
            connection,
            JobRecord.model_validate(
                {
                    **current.model_dump(),
                    "state": "completed",
                    "result_digest": report.fingerprint,
                    "changed": time.time(),
                }
            ),
        )


def _worker_context(store: WorkflowStore, job_id: str, began: float) -> tuple[JobRecord, ExperimentPlan, float] | None:
    """Synchronize a queued startup snapshot with the scheduler's commit before deciding.

    The immutable admitted plan may be read while the scheduler owns its write
    transaction. A queued snapshot then obtains a deadline-bounded write barrier
    and rereads the job: a committed claim runs; rollback/cancellation does no
    native work. A running snapshot already witnesses a committed claim. The
    inherited kernel lock stays owned throughout, without a PID or timing guess.
    """
    job = _read_job(store, job_id)
    if job.state not in ("queued", "running"):
        return None
    with closing(store.connect()) as connection:
        plan = store.plan(connection, job.plan_id, job.owner)
    deadline = began + plan.request.limits.attempt_seconds
    if job.state == "queued":
        with store.transaction(deadline=deadline) as connection:
            current = next(item for item in store.jobs(connection) if item.job_id == job_id)
            if (current.plan_id, current.owner) != (job.plan_id, job.owner):
                raise WorkflowError("Worker startup claim differs from its immutable admitted plan")
            job = current
    if job.state != "running":
        return None
    return job, plan, deadline


def run_job(root: Path, job_id: str, *, progress_hook: Callable[[str, int], None] | None = None) -> None:
    """Execute an already kernel-owned durable job; never acquire work from arbitrary directories.

    Parameters
    ----------
    root : pathlib.Path
        Trusted persistent workspace created by WorkflowStore.
    job_id : str
        Scheduler-admitted running identity. The fixed child inherits its native
        execution lock before the running transition becomes visible.
    progress_hook : callable or None
        Trusted embedding/fault-test hook after actual checkpoint publication;
        never accepted through CLI/MCP. Exceptions leave honest partial evidence.

    Notes
    -----
    Rechecks exact source/runtime binding before executing the existing runner.
    Native recovery owns incomplete-tail reconciliation and checkpoint rejection.
    Attempts precede execution; trial receipts and the workflow report publish last.
    Cancellation/completion serialize in SQLite. No exactly-once physical work is
    promised across a crash between computation and receipt publication.
    """
    began = time.monotonic()
    store = WorkflowStore(root)
    context = _worker_context(store, job_id, began)
    if context is None:
        return
    job, plan, deadline = context
    stopped = threading.Event()
    guard = threading.Thread(target=_guard, args=(store, job_id, plan, deadline, stopped), daemon=True)
    try:
        _check_output(store, job_id, plan, deadline)
        guard.start()
        require_native_runtime(plan.runtime)
        if plan.native_threads != pl.thread_pool_size():
            raise WorkflowError("Native Polars thread profile differs from the frozen plan")
        _transition(store, job_id, deadline=deadline, attempts=len(read_attempts(root / "jobs" / job_id, plan)))
        _input(store, root / "jobs" / job_id, plan)
        if job.stage == "descriptive" and not (root / "jobs" / job_id / _COMPLETE_RECEIPT).exists():
            from coleman.application.workflow_exposure import mark_exposure

            mark_exposure(root, plan, job_id, job.invocations, "descriptive_replay")
        report = _produce_report(store, job, plan, progress_hook, deadline)
        if job.stage == "tuning":
            _transition(store, job_id, deadline=deadline, state="awaiting_evaluation")
        else:
            _publish_completion(store, job_id, plan, report, deadline)
    except Exception as error:  # boundary retains a stable bounded failure, not a false result
        ArtifactWriter(root).write_immutable(
            root / "jobs" / job_id / f"failure-{job.invocations:04d}.json",
            canonical_json({"exception": type(error).__name__, "detail": str(error)[:1000]}),
        )
        with suppress(InterruptedError, WorkflowError, sqlite3.Error):
            _transition(
                store,
                job_id,
                deadline=deadline if time.monotonic() < deadline else None,
                state="interrupted" if isinstance(error, InterruptedError) else "failed",
                error=str(error)[:128]
                if isinstance(error, WorkflowError)
                else f"native_workflow_{type(error).__name__}",
            )
    finally:
        stopped.set()
        if guard.is_alive():
            guard.join(timeout=1)
