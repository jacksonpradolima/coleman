"""Complete-last workflow evidence over unchanged native Parquet/checkpoint formats."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from coleman.application.baseline_storage import _bounded_native_table, read_baseline_artifact
from coleman.artifacts import ArtifactWriter
from coleman.corpus.materialization import validate_source_storage
from coleman.results.domain import decode_json
from coleman.reward.rnfail import RNFailReward
from coleman.workflows.contracts import ExperimentPlan, Trial, WorkflowError
from coleman.workflows.lifecycle import FilePin, NativeRow, PhysicalAttempt, Stage, TrialResult, WorkflowReport

_COMPLETE_RECEIPT = "complete.json"


def pin_file(root: Path, path: Path) -> FilePin:
    """Identify bounded immutable bytes without opening executable checkpoints.

    Parameters
    ----------
    root, path : pathlib.Path
        Trusted workspace artifact root and contained regular file.

    Returns
    -------
    FilePin
        Canonical relative name, exact byte count and SHA-256 fingerprint.

    Raises
    ------
    ValueError, OSError
        Unsafe path, file over 64 MiB or read failure; no receipt is published.
    """
    validate_source_storage(path)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1048576):
            size += len(block)
            if size > 67108864:
                raise WorkflowError("Workflow evidence file exceeds 64 MiB")
            digest.update(block)
    return FilePin(name=path.relative_to(root).as_posix(), sha256="sha256:" + digest.hexdigest(), size_bytes=size)


def verify_trial(root: Path, path: Path, plan: ExperimentPlan, trial: Trial, stage: Stage) -> TrialResult:
    """Revalidate complete stage membership and actual native bytes before reuse.

    Parameters
    ----------
    root, path : pathlib.Path
        Job-owned root and published stage receipt.
    plan : ExperimentPlan
        Complete saved data/runtime/configuration context.
    trial : Trial
        Expected logical coordinate.
    stage : str
        Expected native stage; later continuation cannot mutate a prefix receipt.

    Returns
    -------
    TrialResult
        Verified immutable completed result, suitable for idempotent reuse.

    Raises
    ------
    ValueError, OSError
        Wrong binding, duplicate/missing steps, altered evidence or unavailable bytes.
    """
    receipt = read_baseline_artifact(path, TrialResult)
    if (receipt.plan_fingerprint, receipt.trial_id, receipt.stage) != (plan.fingerprint, trial.trial_id, stage):
        raise WorkflowError("Workflow trial receipt binding mismatch")
    if tuple(row.step for row in receipt.rows) != tuple(range(1, len(receipt.rows) + 1)):
        raise WorkflowError("Workflow receipt requires exactly one contiguous row per build")
    if tuple(row.source_build_id for row in receipt.rows) != plan.dataset.build_ids[: len(receipt.rows)]:
        raise WorkflowError("Workflow receipt source membership mismatch")
    if len({item.name for item in receipt.files}) != len(receipt.files):
        raise WorkflowError("Duplicate workflow evidence membership")
    for pin in receipt.files:
        if pin_file(root, root / pin.name) != pin:
            raise WorkflowError("Workflow native evidence bytes changed")
    if _receipt_native_rows(root, receipt, plan, trial) != receipt.rows:
        raise WorkflowError("Workflow receipt metrics differ from actual native bytes")
    return receipt


def _receipt_native_rows(root: Path, receipt: TrialResult, plan: ExperimentPlan, trial: Trial) -> tuple[NativeRow, ...]:
    """Check aggregate stage footers before decoding only the pinned native files."""
    paths, rows, decoded, encoded = [], 0, 0, 0
    for pin in receipt.files:
        path = root / pin.name
        if path.suffix != ".parquet":
            raise WorkflowError("Workflow native evidence must be Parquet")
        encoded += pin.size_bytes
        with pq.ParquetFile(path) as source:
            rows += source.metadata.num_rows
            decoded += sum(source.metadata.row_group(i).total_byte_size for i in range(source.metadata.num_row_groups))
            if rows > len(receipt.rows) or decoded > 67108864 or len(source.schema_arrow) > 64:
                raise WorkflowError("Workflow native receipt exceeds aggregate row/column/decoded-byte bounds")
        if encoded > plan.request.limits.max_output_bytes:
            raise WorkflowError("Workflow native receipt exceeds its encoded byte allowance")
        paths.append(path)
    if rows != len(receipt.rows):
        raise WorkflowError("Workflow native receipt has missing rows")
    try:
        table = pl.read_parquet(paths, hive_partitioning=True, glob=False)
    except pl.exceptions.PolarsError as error:
        raise WorkflowError("Workflow native receipt schema is incompatible") from error
    return native_rows(table, plan, trial)


def publish_trial(
    root: Path, native: Path, plan: ExperimentPlan, trial: Trial, stage: Stage, steps: int
) -> TrialResult:
    """Validate real native rows/frontier and publish a stage receipt last.

    Parameters
    ----------
    root, native : pathlib.Path
        Job-owned root and actual existing native run directory.
    plan : ExperimentPlan
        Frozen source/runtime/configuration binding.
    trial : Trial
        Logical policy/parameter/budget/seed coordinate.
    stage : str
        Descriptive, tuning prefix or protected continuation.
    steps : int
        Required native committed frontier, in processed build ordinals.

    Returns
    -------
    TrialResult
        Typed rows referencing an immutable snapshot of unchanged native output.

    Raises
    ------
    ValueError, OSError
        Incomplete/mismatched native output/checkpoint or immutable conflict.

    Notes
    -----
    Copies completed Parquet bytes into stage-specific immutable evidence because
    native recovery may reconcile its mutable output during later continuation.
    Native checkpoint pickles remain private to the owning job and are not exported.
    """
    progress_files = list((native / "checkpoints").rglob("progress_ex1.json"))
    if len(progress_files) != 1:
        raise WorkflowError("Native workflow requires one committed checkpoint frontier")
    validate_source_storage(progress_files[0])
    with progress_files[0].open("rb") as stream:
        raw = stream.read(16385)
    if len(raw) > 16384 or decode_json(raw).get("step_committed") != steps:
        raise WorkflowError("Native checkpoint has not committed every required workflow step")
    table = _bounded_native_table(native, steps)
    rows = native_rows(table, plan, trial)
    if len(rows) != steps:
        raise WorkflowError("Native workflow has missing or duplicated build rows")
    target = root / "evidence" / trial.trial_id[7:] / stage
    pins = []
    for source in sorted((native / "results").rglob("*.parquet")):
        destination = target / source.relative_to(native / "results")
        with source.open("rb") as stream:
            ArtifactWriter(root).write_immutable_stream(
                destination,
                iter(lambda stream=stream: stream.read(1048576), b""),
                maximum_bytes=plan.request.limits.max_output_bytes,
            )
        pins.append(pin_file(root, destination))
    receipt = TrialResult(
        schema_version=1,
        trial_id=trial.trial_id,
        plan_fingerprint=plan.fingerprint,
        stage=stage,
        native_run_id=native.name,
        rows=tuple(rows),
        files=tuple(pins),
    )
    payload = receipt.to_json()
    if len(payload.encode()) > 16777216:
        raise WorkflowError("Native trial receipt exceeds its 16 MiB reader limit; reduce the declared history")
    ArtifactWriter(root).write_immutable(target / _COMPLETE_RECEIPT, payload)
    return verify_trial(root, target / _COMPLETE_RECEIPT, plan, trial, stage)


def native_rows(table, plan: ExperimentPlan, trial: Trial) -> tuple[NativeRow, ...]:
    """Project bounded native rows with explicit original metric meanings.

    Parameters
    ----------
    table : polars.DataFrame
        Footer-bounded unchanged native table sorted by processed step.
    plan : ExperimentPlan
        Exact registered source membership and native budget semantics.
    trial : Trial
        Expected actual policy, RNFail and budget coordinate.

    Returns
    -------
    tuple[NativeRow, ...]
        Finite contiguous summaries; native cost is APFDc, never physical cost.

    Raises
    ------
    ValueError
        Wrong coordinate, duplicate/missing steps, invalid ranking or finite values.
    """
    required = {
        "step",
        "experiment",
        "scenario",
        "budget_value",
        "reward_function",
        "policy",
        "budget_mode",
        "fitness",
        "cost",
        "tests_ran",
        "tests_not_ran",
        "total_build_duration",
        "process_cpu_time_seconds",
        "wall_time_seconds",
        "prioritization_order_top_k",
        "execution_id",
    }
    if not required.issubset(table.columns):
        raise WorkflowError("Native workflow table schema is incomplete or incompatible")
    table = table.sort("step")
    if table["step"].to_list() != list(range(1, table.height + 1)) or table.height > len(plan.dataset.build_ids):
        raise WorkflowError("Native workflow row membership is incomplete or duplicated")
    rows = []
    for item in table.iter_rows(named=True):
        policy = "Random" if trial.policy == "Random" else f"UCB (C={trial.exploration})"
        if (
            item["budget_value"] != trial.budget_value
            or item["reward_function"] != str(RNFailReward())
            or item["policy"] != policy
            or item["budget_mode"] != plan.request.budget.mode.value
            or item["scenario"] != plan.dataset.dataset_id
            or item["experiment"] != 1
        ):
            raise WorkflowError("Native coordinate differs from frozen workflow")
        rows.append(
            NativeRow(
                step=item["step"],
                source_build_id=plan.dataset.build_ids[item["step"] - 1],
                napfd=item["fitness"],
                apfdc=item["cost"],
                tests_ran=item["tests_ran"],
                tests_not_ran=item["tests_not_ran"],
                modeled_full_seconds=item["total_build_duration"],
                process_cpu_seconds=item["process_cpu_time_seconds"],
                process_wall_seconds=item["wall_time_seconds"],
                ranking=tuple(decode_json(item["prioritization_order_top_k"])),
                attempt_id=item["execution_id"],
            )
        )
    return tuple(rows)


def verify_report(root: Path, plan: ExperimentPlan, report: WorkflowReport) -> None:
    """Verify complete planned membership and all retained native bytes without loading checkpoints.

    Parameters
    ----------
    root : pathlib.Path
        Trusted job directory; no imported bundle is extracted into it.
    plan : ExperimentPlan
        Saved complete context used to validate every coordinate and expected count.
    report : WorkflowReport
        Revalidated proposed completion, including a possible audited comparison.

    Raises
    ------
    ValueError, OSError
        Wrong plan/stage, missing/duplicate trials/builds or changed native evidence.
    """
    from coleman.application.workflow_validation import expected_trials_for_report

    stage = "descriptive" if plan.request.evaluation == "descriptive" else "evaluation"
    expected = expected_trials_for_report(plan, report)
    if len(report.trials) != len(expected) or {receipt.trial_id for receipt in report.trials} != expected:
        raise WorkflowError("Workflow complete report has missing/duplicate/unplanned trials")
    coordinates = {trial.trial_id: trial for trial in plan.trials}
    for receipt in report.trials:
        if receipt.stage != stage or len(receipt.rows) != len(plan.dataset.build_ids):
            raise WorkflowError("Workflow complete report requires every native build at the declared stage")
        trial = coordinates[receipt.trial_id]
        path = root / "evidence" / trial.trial_id[7:] / stage / _COMPLETE_RECEIPT
        if verify_trial(root, path, plan, trial, stage) != receipt:
            raise WorkflowError("Workflow report differs from actual completed native evidence")
    if report.physical_attempts != read_attempts(root, plan):
        raise WorkflowError("Workflow report differs from retained physical attempts")


def read_attempts(root: Path, plan: ExperimentPlan) -> tuple[PhysicalAttempt, ...]:
    """Read bounded typed physical reservations, including interruptions and prior stages.

    Parameters
    ----------
    root : pathlib.Path
        Private trusted job directory. Events are never inferred from native timing.
    plan : ExperimentPlan
        Complete expected coordinates and finite per-trial attempt cap.

    Returns
    -------
    tuple[PhysicalAttempt, ...]
        Canonical trial/ordinal order, with exact plan/trial/version binding.

    Raises
    ------
    ValueError, OSError
        Extra, missing-ordinal, conflicting or incompatible event records.
    """
    return tuple(event for trial in plan.trials for event in read_trial_attempts(root, plan, trial))


def read_trial_attempts(root: Path, plan: ExperimentPlan, trial: Trial) -> tuple[PhysicalAttempt, ...]:
    """Read one coordinate's bounded physical reservations without loading learning state.

    Parameters
    ----------
    root : pathlib.Path
        Trusted private job directory, never a client-supplied extraction path.
    plan : ExperimentPlan
        Exact context and finite per-trial invocation allowance, at most eight.
    trial : Trial
        Saved coordinate whose retained event files are inspected.

    Returns
    -------
    tuple[PhysicalAttempt, ...]
        Canonical ordinal order. A reservation does not prove native computation
        completed; no files are written and no checkpoint pickle is loaded.

    Raises
    ------
    ValueError, OSError
        Excess events, nonconsecutive filenames/ordinals, incompatible versions,
        changed binding or unavailable private metadata.
    """
    paths = sorted((root / "attempts" / trial.trial_id[7:]).glob("*.json"))
    if len(paths) > plan.request.limits.max_attempts:
        raise WorkflowError("Retained physical attempts exceed the saved cap")
    events = []
    for ordinal, path in enumerate(paths, 1):
        event = read_baseline_artifact(path, PhysicalAttempt)
        if (event.ordinal, event.trial_id, event.plan_fingerprint, path.name) != (
            ordinal,
            trial.trial_id,
            plan.fingerprint,
            f"{ordinal:04d}.json",
        ):
            raise WorkflowError("Retained physical attempt binding mismatch")
        events.append(event)
    return tuple(events)
