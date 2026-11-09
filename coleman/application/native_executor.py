"""Adapt unchanged native trials and scientific evidence to the shared work loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from coleman.application.baseline_storage import read_baseline_artifact
from coleman.application.execution_program import CoordinateWork, ExecutionProgram
from coleman.application.workflow_artifacts import pin_file, publish_trial, read_attempts, verify_report, verify_trial
from coleman.application.workflow_evaluation import (
    EvaluationHandoff,
    choose_configuration,
    compare_continuation,
    evaluation_trials,
    prepare_evaluation_attempt,
    require_continuation,
)
from coleman.application.workflow_native import execute_native
from coleman.application.workflow_store import JobRecord, WorkflowStore
from coleman.artifacts import ArtifactWriter
from coleman.core.identity import Reference
from coleman.core.primitives import decode_json, fingerprint
from coleman.corpus.materialization import validate_source_storage
from coleman.objectives.bindings import DefinitionBinding
from coleman.spec.selected_runtime import SelectedInstalledRuntime, require_native_runtime
from coleman.state.storage_contracts import ExecutionCompatibility
from coleman.tasks.contracts import TaskFrontier
from coleman.tasks.identity import CodecDeclaration
from coleman.tasks.protocols import AttemptContext, CommittedProgress, CoordinateContext, PreparationContext
from coleman.workflows.contracts import ExperimentPlan, Trial, WorkflowError
from coleman.workflows.lifecycle import TrialResult, WorkflowReport, report_assumptions

_COMPLETE_RECEIPT = "complete.json"
_MISSING_COMMITTED_CHECKPOINT = "Previously committed native checkpoint is missing; implicit reset is forbidden"
_FRONTIER = DefinitionBinding(
    reference=Reference(id="coleman-native-processed-builds", version="1"),
    definition_fingerprint=fingerprint({"profile": "native-workflow-build-ordinal-v1", "unit": "processed_build"}),
)


def _prefix_steps(plan: ExperimentPlan) -> int:
    """Count eligible chronological train/validation builds without outcomes."""
    if plan.dataset.temporal_plan is None:
        raise WorkflowError("Protected native execution requires its temporal plan")
    return sum(item.partition != "test" for item in plan.dataset.split_by_build())


def _input(workspace: Path, root: Path, plan: ExperimentPlan) -> None:
    """Verify the original snapshot before copying into its stable native path."""
    source = workspace / "datasets" / f"{plan.dataset.sha256[7:]}.parquet"
    pin = pin_file(workspace, source)
    if (pin.sha256, pin.size_bytes) != (plan.dataset.sha256, plan.dataset.size_bytes):
        raise WorkflowError("Registered dataset changed; continuation refused")
    destination = root / "inputs" / plan.dataset.dataset_id / "features-engineered.parquet"
    with source.open("rb") as stream:
        ArtifactWriter(root).write_immutable_stream(
            destination, iter(lambda: stream.read(1048576), b""), maximum_bytes=plan.dataset.size_bytes
        )


@dataclass(frozen=True)
class _NativeContext:
    """Keep native declarations and trusted guards out of metadata transactions."""

    workspace: Path
    root: Path
    plan: ExperimentPlan
    job: JobRecord
    guard: Callable[[], None]


def _marker_position(root: Path, trial_id: str) -> int | None:
    """Read only the bounded native progress marker, never pickle/RNG state."""
    markers = list((root / "native" / trial_id[7:]).rglob("progress_ex1.json"))
    if not markers:
        return None
    if len(markers) != 1:
        raise WorkflowError(_MISSING_COMMITTED_CHECKPOINT)
    marker = markers[0]
    validate_source_storage(marker)
    with marker.open("rb") as stream:
        raw = stream.read(16385)
    if len(raw) > 16384:
        raise WorkflowError("Native checkpoint progress exceeds 16 KiB")
    progress = decode_json(raw)
    position = progress.get("step_committed")
    if not _strict_int(position) or not 1 <= position <= 512:
        raise WorkflowError("Native checkpoint frontier is incompatible with this workflow")
    expected = marker.parent / f"ckpt_ex1_step{position}.pkl"
    if (
        progress.get("run_id") != marker.parent.name
        or not _strict_int(progress.get("experiment"))
        or progress["experiment"] != 1
        or progress.get("checkpoint_path") != str(expected)
    ):
        raise WorkflowError("Native checkpoint is not owned by this exact workflow root/execution")
    validate_source_storage(expected)
    if not expected.is_file():
        raise WorkflowError(_MISSING_COMMITTED_CHECKPOINT)
    return position


def _strict_int(value: object) -> bool:
    """Keep exact native JSON counters distinct from bool and subclasses."""
    kind = type(value)
    return kind is int


@dataclass(frozen=True)
class _NativeProgress:
    """Verify weaker native marker evidence without inventing an owned-store ACK."""

    native: _NativeContext

    def validate(self, progress: CommittedProgress[int], context: CoordinateContext, /) -> None:
        """Match an actual post-checkpoint callback to its retained build ordinal."""
        if not _strict_int(progress.proof) or progress.proof != progress.frontier.position:
            raise WorkflowError("Native committed progress requires its exact build ordinal")
        if _marker_position(self.native.root, context.coordinate_fingerprint) != progress.proof:
            raise WorkflowError("Native progress differs from its actual committed marker")

    def reconcile(self, context: CoordinateContext, /) -> CommittedProgress[int] | None:
        """Observe marker progress; existing native recovery verifies payload/RNG."""
        step = _marker_position(self.native.root, context.coordinate_fingerprint)
        job = self.native.job
        minimum = job.checkpoint_step if job.current_trial == context.coordinate_fingerprint else 0
        if step is None:
            if minimum:
                raise WorkflowError(_MISSING_COMMITTED_CHECKPOINT)
            return None
        if step < minimum:
            raise WorkflowError("Native checkpoint is older than its acknowledged workflow frontier")
        frontier = TaskFrontier(profile=context.target.profile, coordinate=context.reference, position=step)
        return CommittedProgress(context.preparation.plan_fingerprint, context.coordinate_fingerprint, frontier, step)


@dataclass(frozen=True)
class _NativeBoundary:
    """Wrap the one common coordinate loader in the existing scientific scope."""

    native: _NativeContext
    cached: WorkflowReport | None
    handoff: EvaluationHandoff | None

    def produce(self, load_completed: Callable[[], tuple[TrialResult, ...]], /) -> WorkflowReport:
        """Reuse a verified report or reserve protected access before loading."""
        native = self.native
        if self.cached is not None:
            if self.cached.comparison is not None:
                from coleman.evaluation.protected import EvaluationAudit

                EvaluationAudit(native.workspace / "audit").save_report(self.cached.comparison)
            return self.cached
        if self.handoff is not None:
            return compare_continuation(native.workspace, native.root, native.plan, self.handoff, load_completed)
        return self._descriptive(load_completed())

    def _descriptive(self, results: tuple[TrialResult, ...]) -> WorkflowReport:
        """Retain native selection, conservative ties and physical-attempt fields."""
        native, plan = self.native, self.native.plan
        selected = choose_configuration(plan, results)
        selected_budget = next(trial.budget_value for trial in plan.trials if trial.configuration_id == selected)
        fallback = next(
            trial.configuration_id
            for trial in plan.trials
            if trial.policy == "Random" and trial.budget_value == selected_budget
        )
        return WorkflowReport(
            schema_version=1,
            plan_fingerprint=plan.fingerprint,
            trials=results,
            selected_configuration=selected,
            fallback_configuration=fallback,
            physical_attempts=read_attempts(native.root, plan),
            interpretation="descriptive_only",
            assumptions=report_assumptions(plan),
        )


@dataclass(frozen=True)
class _NativePrepared:
    """Delegate mathematical work/checkpoints; own no queue or coordinate loop."""

    native: _NativeContext
    boundary: _NativeBoundary
    progress_verifier: _NativeProgress

    def read_completed(self, trial: Trial, context: CoordinateContext, /) -> TrialResult | None:
        """Read original complete native bytes and validate the required frontier."""
        path = context.artifact_directory / _COMPLETE_RECEIPT
        if not path.exists():
            if self.native.job.stage == "evaluation":
                require_continuation(self.native.root, trial, _prefix_steps(self.native.plan))
            return None
        result = verify_trial(self.native.root, path, self.native.plan, trial, self.native.job.stage)
        if len(result.rows) != context.target.position:
            raise WorkflowError("Completed workflow stage has an incompatible frontier")
        return result

    def advance(
        self, trial: Trial, context: AttemptContext, committed: Callable[[CommittedProgress[int]], None], /
    ) -> TrialResult:
        """Continue actual native learning and construct its original stage receipt."""
        native, coordinate = self.native, context.coordinate
        minimum = native.job.checkpoint_step if native.job.current_trial == trial.trial_id else 0
        if native.job.stage == "evaluation":
            minimum = max(minimum, _prefix_steps(native.plan))

        def progress(step: int) -> None:
            """Notify only after the existing runner publishes its checkpoint."""
            frontier = TaskFrontier(profile=coordinate.target.profile, coordinate=coordinate.reference, position=step)
            committed(CommittedProgress(coordinate.preparation.plan_fingerprint, trial.trial_id, frontier, step))

        output = execute_native(
            native.root,
            native.plan,
            trial,
            coordinate.target.position,
            progress,
            attempt_id=context.attempt_id,
            minimum_step=minimum,
        )
        context.guard()
        result = publish_trial(native.root, output, native.plan, trial, native.job.stage, coordinate.target.position)
        context.guard()
        return result

    def publish_coordinate(self, trial: Trial, value: TrialResult, context: CoordinateContext, /) -> None:
        """Verify the original immutable publication made by publish_trial."""
        retained = self.read_completed(trial, context)
        if retained is None or retained.to_json() != value.to_json():
            raise WorkflowError("Published native coordinate differs from its complete returned evidence")


@dataclass(frozen=True)
class _ReportCodec:
    """Preserve the existing report serializer, defaults and 16 MiB reader cap."""

    declaration: CodecDeclaration

    def validate(self, value: object, /) -> WorkflowReport:
        """Revalidate the complete existing native report without schema conversion."""
        return WorkflowReport.model_validate(value)

    def encode(self, value: WorkflowReport, /) -> str:
        """Retain native canonical bytes rather than dumping an adapter wrapper."""
        return self.validate(value).to_json()

    def decode(self, body: str, /) -> WorkflowReport:
        """Read the unchanged native v1 report; no migration or inference occurs."""
        return WorkflowReport.from_json(body)


def _stage_context(native: _NativeContext) -> tuple[_NativeBoundary, tuple[Trial, ...]]:
    """Freeze native stage membership before the common loader can execute."""
    complete_path = native.root / _COMPLETE_RECEIPT
    if complete_path.exists():
        cached = read_baseline_artifact(complete_path, WorkflowReport)
        verify_report(native.root, native.plan, cached)
        ids = {item.trial_id for item in cached.trials}
        return _NativeBoundary(native, cached, None), tuple(t for t in native.plan.trials if t.trial_id in ids)
    if native.job.stage == "evaluation":
        initial = read_baseline_artifact(native.root / "evaluation-handoff.json", EvaluationHandoff)
        handoff = prepare_evaluation_attempt(
            native.workspace, native.root, native.plan, initial, native.job.invocations
        )
        native.guard()
        return _NativeBoundary(native, None, handoff), evaluation_trials(native.plan, handoff)
    return _NativeBoundary(native, None, None), native.plan.trials


def prepare_native_program(
    store: WorkflowStore, job: JobRecord, plan: ExperimentPlan, *, deadline: float, guard: Callable[[], None]
) -> ExecutionProgram[Trial, TrialResult, WorkflowReport, int]:
    """Prepare native declarations and source before any checkpoint restoration.

    Parameters
    ----------
    store : WorkflowStore
        Existing shared metadata owner; no transaction is entered by this adapter.
    job : JobRecord
        Already claimed native record, retaining original stage/counter meanings.
    plan : ExperimentPlan
        Exact frozen native source/spec/runtime; never converted to TaskPlan.
    deadline : float
        Actual finite common-driver monotonic deadline in seconds.
    guard : callable
        Common cancellation/deadline/RSS/output check, outside SQL bodies.

    Returns
    -------
    ExecutionProgram
        Original Trial coordinates, native report codec and scientific boundary.

    Raises
    ------
    ValueError, OSError
        Changed runtime/threads/source/evidence or unavailable protected prefix.
        Existing native checkpoint loading remains inside execute_native later.

    Notes
    -----
    Checks actual installation and threads before copying source or reading any
    native pickle/RNG. Writes only the existing pinned input/exposure/evaluation
    bookkeeping. Owns no job mutation, attempt reservation or coordinate loop.
    """
    require_native_runtime(plan.runtime)
    if plan.native_threads != pl.thread_pool_size():
        raise WorkflowError("Native Polars thread profile differs from the frozen plan")
    if not isinstance(plan.runtime, SelectedInstalledRuntime):
        raise WorkflowError("Native execution requires the selected runtime generation")
    if deadline <= 0:
        raise WorkflowError("Native execution requires its common monotonic deadline")
    root = store.root / "jobs" / job.job_id
    _input(store.root, root, plan)
    if job.stage == "descriptive" and not (root / _COMPLETE_RECEIPT).exists():
        from coleman.application.workflow_exposure import mark_exposure

        mark_exposure(store.root, plan, job.job_id, job.invocations, "descriptive_replay")
    native = _NativeContext(store.root, root, plan, job, guard)
    boundary, trials = _stage_context(native)
    implementation = DefinitionBinding(
        reference=Reference(id="coleman-native-tcpci", version="1"),
        definition_fingerprint=plan.runtime.package_fingerprint,
    )
    context = PreparationContext(
        plan.fingerprint,
        DefinitionBinding(
            reference=Reference(id=plan.dataset.dataset_id, version="1"),
            definition_fingerprint=plan.dataset.source_fingerprint,
        ),
        implementation,
        ExecutionCompatibility(schema_version=1, runtime=plan.runtime, owner_implementations=(implementation,)),
        root,
    )
    steps = _prefix_steps(plan) if job.stage == "tuning" else len(plan.dataset.build_ids)
    work = tuple(
        CoordinateWork(
            trial.trial_id,
            Reference(id=trial.trial_id, version="1"),
            TaskFrontier(profile=_FRONTIER, coordinate=Reference(id=trial.trial_id, version="1"), position=steps),
            trial,
            root / "evidence" / trial.trial_id[7:] / job.stage,
            seed=trial.seed,
        )
        for trial in trials
    )
    codec = _ReportCodec(
        CodecDeclaration(
            binding=implementation,
            schema_fingerprint=fingerprint(WorkflowReport.model_json_schema()),
            maximum_bytes=16777216,
        )
    )
    return ExecutionProgram(
        context,
        work,
        _NativePrepared(native, boundary, _NativeProgress(native)),
        codec,
        lambda report: report.fingerprint,
        root / _COMPLETE_RECEIPT,
        "awaiting_evaluation" if job.stage == "tuning" else "completed",
    )
