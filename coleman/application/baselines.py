"""Sequential development matrices through the existing native RunSpec application API."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from coleman.api import run_with_extension
from coleman.application.baseline_execution import (
    NativeBaselineBinding,
    baseline_dataset_name,
    baseline_extension,
    baseline_input_bytes,
)
from coleman.application.baseline_lock import BaselineBusyError, baseline_lock
from coleman.application.baseline_resources import BaselineLimitError
from coleman.application.baseline_storage import (
    load_baseline_observations,
    read_baseline_artifact,
    read_baseline_payload,
    validate_native_baseline_completion,
)
from coleman.artifacts import ArtifactWriter
from coleman.corpus.baseline_contracts import BaselinePlan
from coleman.corpus.baseline_eligibility import baseline_eligibility
from coleman.corpus.baseline_results import (
    BaselineAttempt,
    BaselineCellResult,
    BaselineCoordinate,
    BaselineMatrixReceipt,
    BaselineMatrixReport,
    baseline_coordinates,
    baseline_measurement,
)
from coleman.corpus.materialization import validate_source_storage
from coleman.spec.models import RunSpec
from coleman.spec.run_id import compute_run_id
from coleman.spec.selected_runtime import (
    RuntimeIdentity,
    SelectedInstalledRuntime,
    capture_native_runtime,
    read_runtime,
)

_PLAN_FILE = "plan.json"


class _MatrixInterruptedError(RuntimeError):
    """Carry an audited aborted cell until the complete partial grid is durably published."""

    def __init__(self, cell: BaselineCellResult, interruption: KeyboardInterrupt | SystemExit):
        """Retain the original stop request; orchestration re-raises it after publishing incomplete evidence."""
        super().__init__("Baseline matrix invocation interrupted")
        self.cell, self.interruption = cell, interruption


def _publish(writer: ArtifactWriter, path: Path, value) -> None:
    """Bound and atomically publish immutable strict JSON; conflicting bytes raise without replacement."""
    payload = value.to_json() + "\n"
    if len(payload.encode()) > 16 * 1024**2:
        raise ValueError("Baseline JSON artifact exceeds 16 MiB")
    writer.write_immutable(path, payload)


def _native_spec(root: Path, binding: NativeBaselineBinding, dataset: str) -> RunSpec:
    """Use legacy orchestration's validated placeholder; the qualified extension supplies the actual policy."""
    return RunSpec.model_validate(
        {
            "execution": {
                "independent_executions": 1,
                "parallel_pool_size": 2,
                "seed": binding.coordinate.seed,
                "force_sequential_under_scalene": False,
            },
            "experiment": {
                "datasets_dir": str(root / "inputs"),
                "datasets": [dataset],
                "policies": ["Random"],
                "rewards": ["RNFail"],
                "budget": {"mode": "ratio", "values": [binding.coordinate.budget.value]},
            },
            "checkpoint": {"enabled": True, "interval": 1},
            "results": {
                "out_dir": str(root / "native"),
                "sink": "parquet",
                "enabled": True,
                "top_k_prioritization": 10000,
            },
            "extensions": {"baseline_matrix_v1": binding.to_json()},
        }
    )


def _attempts(root: Path, cell: BaselineCoordinate) -> tuple[BaselineAttempt, ...]:
    """Retain every physical start and terminal event, including orphaned starts after process death."""
    values = []
    for index, path in enumerate(sorted((root / "attempts" / cell.cell_id[7:]).glob("*.json"))):
        if index >= 4096:
            raise ValueError("Baseline attempt event bound exceeded")
        event = read_baseline_artifact(path, BaselineAttempt)
        if event.cell_id != cell.cell_id or path.name != f"{event.attempt_id}-{event.event}.json":
            raise ValueError("Baseline attempt identity differs from its artifact location")
        values.append(event)
    starts = {value.attempt_id for value in values if value.event == "started"}
    terminals = {value.attempt_id for value in values if value.event == "terminal"}
    if not terminals <= starts:
        raise ValueError("Baseline terminal event has no durable start")
    return tuple(sorted(values, key=lambda value: (value.recorded_at, value.attempt_id, value.event)))


def _event(cell: BaselineCoordinate, attempt_id: str, state, reasons, elapsed, completion=None, error=None):
    """Record actual invocation costs; unknown worker costs remain unavailable after interruption or failure."""
    worker_elapsed = completion.worker_elapsed.value if completion else None
    worker_peak = completion.worker_peak.value if completion else None
    if isinstance(error, BaselineLimitError):
        worker_elapsed, worker_peak = error.elapsed_seconds, error.peak_mib
    return BaselineAttempt(
        schema_version=1,
        cell_id=cell.cell_id,
        attempt_id=attempt_id,
        event="started" if state == "running" else "terminal",
        state=state,
        recorded_at=datetime.now(UTC),
        reasons=reasons,
        host_elapsed=baseline_measurement("host_attempt_elapsed_seconds", elapsed),
        worker_elapsed=baseline_measurement("native_process_elapsed_seconds", worker_elapsed),
        worker_peak=baseline_measurement("native_process_peak_mib", worker_peak),
    )


def _attempt_summary(events: tuple[BaselineAttempt, ...]) -> tuple[int, bool]:
    """Count missing terminals and require actual worker costs for every terminal invocation."""
    starts = {event.attempt_id for event in events if event.event == "started"}
    terminals = [event for event in events if event.event == "terminal"]
    unclosed = len(starts - {event.attempt_id for event in terminals})
    observed = (
        bool(terminals)
        and not unclosed
        and all(event.worker_elapsed.value is not None and event.worker_peak.value is not None for event in terminals)
    )
    return unclosed, observed


def _cell_result(root, plan, binding, spec, state, reasons) -> BaselineCellResult:
    """Load actual partial records and complete physical-attempt history without erasing missing costs."""
    cell = binding.coordinate
    native_id = compute_run_id(spec)
    observations = load_baseline_observations(root / "native" / native_id, plan, cell, native_id, binding.fingerprint)
    events = _attempts(root, cell)
    unclosed, observed = _attempt_summary(events)
    starts = {event.attempt_id for event in events if event.event == "started"}
    if any(value.attempt_id not in starts for value in observations):
        raise ValueError("Baseline observation has no matching physical attempt")
    return BaselineCellResult(
        **cell.model_dump(),
        state=state,
        reasons=reasons,
        native_run_id=native_id,
        observations=tuple(value.fingerprint for value in observations),
        mathematical_observations=tuple(value.mathematical_fingerprint for value in observations),
        attempts=tuple(value.fingerprint for value in events),
        unclosed_attempts=unclosed,
        resource_evidence="observed" if observed else "unavailable",
    )


def _execute_cell(root, plan, runtime, binding, dataset, writer) -> BaselineCellResult:
    """Reuse verified completion or terminal failure; resume only interrupted logical work using native checkpoints."""
    cell = binding.coordinate
    spec = _native_spec(root, binding, dataset)
    native_root = root / "native" / compute_run_id(spec)
    completion_path = native_root / "baseline-native-complete.json"
    events = _attempts(root, cell)
    if completion_path.exists():
        validate_native_baseline_completion(
            native_root, plan, cell, native_root.name, binding.fingerprint, runtime.fingerprint, dataset
        )
        return _cell_result(root, plan, binding, spec, "completed", ())
    failed = [event for event in events if event.state == "failed"]
    if failed:
        return _cell_result(root, plan, binding, spec, "failed", failed[-1].reasons)
    attempt_id = uuid.uuid4().hex
    event_root = root / "attempts" / cell.cell_id[7:]
    _publish(writer, event_root / f"{attempt_id}-started.json", _event(cell, attempt_id, "running", (), None))
    started = time.perf_counter()
    state, reasons, completion, error = "completed", (), None, None
    try:
        run_with_extension(spec, baseline_extension(attempt_id))
    except BaselineBusyError as caught:
        state, reasons, error = "blocked", ("native_execution_active",), caught
    except (KeyboardInterrupt, SystemExit) as caught:
        reasons = ("host_invocation_interrupted",)
        terminal = _event(cell, attempt_id, "aborted", reasons, time.perf_counter() - started)
        _publish(writer, event_root / f"{attempt_id}-terminal.json", terminal)
        aborted = _cell_result(root, plan, binding, spec, "aborted", reasons)
        raise _MatrixInterruptedError(aborted, caught) from caught
    except Exception as caught:  # preserve actual execution failures, never completion-integrity failures
        if completion_path.exists():
            raise
        logging.getLogger(__name__).exception("Native baseline cell failed")
        state, reasons, error = "failed", ("native_execution_failed",), caught
        if isinstance(caught, BaselineLimitError):
            reasons = (caught.reason,)
    if state == "completed":
        completion = validate_native_baseline_completion(
            native_root, plan, cell, native_root.name, binding.fingerprint, runtime.fingerprint, dataset
        )
    terminal = _event(cell, attempt_id, state, reasons, time.perf_counter() - started, completion, error)
    _publish(writer, event_root / f"{attempt_id}-terminal.json", terminal)
    return _cell_result(root, plan, binding, spec, "unavailable" if state == "blocked" else state, reasons)


def run_baseline_matrix(plan: BaselinePlan, output: str | Path) -> BaselineMatrixReport:
    """Execute the frozen development grid sequentially through native process-isolated RunSpec extensions.

    Writes immutable plan/runtime/input, per-build evidence and physical attempt
    events under output/baseline-matrices/<plan>/<installed-runtime>. A completion
    receipt publishes last only for a terminal grid. Same-runtime retries verify
    completed bytes, retain failed cells and resume interrupted native checkpoints.
    Unknown versions, conflicting/corrupt evidence and active matrix locks raise;
    cell execution failures remain explicit report entries. Keyboard interruption
    records the current cell, leaves remaining eligible cells unavailable, and
    re-raises the original interruption after publishing the incomplete report.

    Frozen cooperative limits apply per invocation, not as an OS sandbox. Prior
    interrupted costs stay visible and may be unavailable. A runtime change creates
    a new scope. No protected test access, statistical promotion or source
    admission is granted; current timing is not portable reproducible evidence.
    """
    plan = BaselinePlan.model_validate(plan)
    runtime = capture_native_runtime()
    root = Path(output).resolve() / "baseline-matrices" / plan.fingerprint[7:] / runtime.fingerprint[7:]
    validate_source_storage(root)
    with baseline_lock(root / "matrix.lock"):
        return _run_locked(root, plan, runtime)


def _run_locked(root: Path, plan: BaselinePlan, runtime: SelectedInstalledRuntime) -> BaselineMatrixReport:
    """Serialize inventory and completion publication while native children separately own their execution locks."""
    writer = ArtifactWriter(root)
    _publish(writer, root / _PLAN_FILE, plan)
    _publish(writer, root / "runtime.json", runtime)
    eligibility = {item.method: item for item in baseline_eligibility(plan)}
    data = baseline_input_bytes(plan) if any(item.compatible for item in eligibility.values()) else b""
    dataset = baseline_dataset_name(plan)
    if data:
        writer.write_immutable_stream(
            root / "inputs" / dataset / "features-engineered.parquet", (data,), maximum_bytes=len(data)
        )
    cells = []
    interruption: KeyboardInterrupt | SystemExit | None = None
    for coordinate in baseline_coordinates(plan):
        reasons = eligibility[coordinate.method].reasons
        if reasons or interruption is not None:
            cells.append(_unexecuted_cell(coordinate, reasons))
            continue
        binding = NativeBaselineBinding(
            schema_version=1,
            plan_file=str(root / _PLAN_FILE),
            plan_fingerprint=plan.fingerprint,
            runtime_fingerprint=runtime.fingerprint,
            coordinate=coordinate,
            input_sha256="sha256:" + hashlib.sha256(data).hexdigest(),
            input_size_bytes=len(data),
        )
        try:
            result = _execute_cell(root, plan, runtime, binding, dataset, writer)
        except _MatrixInterruptedError as stopped:
            result, interruption = stopped.cell, stopped.interruption
        cells.append(result)
    report = BaselineMatrixReport(
        schema_version=1, plan=plan, producer_fingerprint=runtime.fingerprint, cells=tuple(cells)
    )
    _publish(writer, root / "reports" / f"{report.fingerprint[7:]}.json", report)
    if report.complete:
        receipt = BaselineMatrixReceipt(
            schema_version=1,
            plan_fingerprint=plan.fingerprint,
            runtime_fingerprint=runtime.fingerprint,
            report_fingerprint=report.fingerprint,
            mathematical_fingerprint=report.mathematical_fingerprint,
            report_size_bytes=len((report.to_json() + "\n").encode()),
        )
        _publish(writer, root / "complete.json", receipt)
    if interruption is not None:
        raise interruption
    return report


def _unexecuted_cell(coordinate: BaselineCoordinate, reasons: tuple[str, ...]) -> BaselineCellResult:
    """Distinguish declared incompatibility from unattempted interrupted work without inventing observations."""
    return BaselineCellResult(
        **coordinate.model_dump(),
        state="incompatible" if reasons else "unavailable",
        reasons=reasons or ("earlier_cell_interrupted",),
        native_run_id=None,
        observations=(),
        mathematical_observations=(),
        attempts=(),
        unclosed_attempts=0,
        resource_evidence="not_applicable" if reasons else "unavailable",
    )


def read_baseline_matrix(root: str | Path) -> BaselineMatrixReport:
    """Verify completed matrix bytes, every attempt and actual native evidence without executing checkpoints.

    The local directory may move: recorded input paths are inspected as data and
    never followed. Unknown versions, missing completion, extra/changed raw files,
    forged identities and invalid mathematics raise ValueError or I/O errors.
    This verifies internal integrity, not external authorship or source admission.
    No installed-runtime equality, network, CLI/MCP import or pickle loading is
    required to inspect an older completed matrix. No implicit migration occurs.
    """
    root = Path(root).resolve()
    validate_source_storage(root)
    with baseline_lock(root / "matrix.lock"):
        receipt = read_baseline_artifact(root / "complete.json", BaselineMatrixReceipt)
        report_path = root / "reports" / f"{receipt.report_fingerprint[7:]}.json"
        report = read_baseline_artifact(report_path, BaselineMatrixReport)
        plan = read_baseline_artifact(root / _PLAN_FILE, BaselinePlan)
        runtime = read_runtime(read_baseline_payload(root / "runtime.json"))
        if (
            not report.complete
            or report.fingerprint != receipt.report_fingerprint
            or report.mathematical_fingerprint != receipt.mathematical_fingerprint
            or report_path.stat().st_size != receipt.report_size_bytes
            or plan != report.plan
            or receipt.plan_fingerprint != plan.fingerprint
            or receipt.runtime_fingerprint != runtime.fingerprint
            or report.producer_fingerprint != runtime.fingerprint
        ):
            raise ValueError("Baseline matrix completion differs from its actual frozen report/context")
        _verify_report_cells(root, report, runtime)
        return report


def _verify_report_cells(root: Path, report: BaselineMatrixReport, runtime: RuntimeIdentity) -> None:
    """Verify native cell membership separately from the matrix completion envelope."""
    plan = report.plan
    coordinates = {cell.cell_id: cell for cell in baseline_coordinates(plan)}
    for cell in report.cells:
        if cell.state == "incompatible":
            continue
        coordinate = coordinates[cell.cell_id]
        events = _verify_cell_attempts(root, cell, coordinate)
        native_root = root / "native" / str(cell.native_run_id)
        spec_path = native_root / "spec.resolved.json"
        if not spec_path.exists() and not cell.observations and cell.state == "failed":
            continue
        binding = _read_cell_binding(spec_path, coordinate, plan, runtime, cell.native_run_id)
        observations = load_baseline_observations(
            native_root, plan, coordinate, str(cell.native_run_id), binding.fingerprint
        )
        if (
            tuple(value.fingerprint for value in observations) != cell.observations
            or tuple(value.mathematical_fingerprint for value in observations) != cell.mathematical_observations
        ):
            raise ValueError("Baseline report raw observation membership changed")
        _verify_observed_attempts(observations, events)
        if cell.state == "completed" or (native_root / "baseline-native-complete.json").exists():
            validate_native_baseline_completion(
                native_root,
                plan,
                coordinate,
                str(cell.native_run_id),
                binding.fingerprint,
                runtime.fingerprint,
                baseline_dataset_name(plan),
            )


def _verify_observed_attempts(observations, events: tuple[BaselineAttempt, ...]) -> None:
    """Require each raw build's recorded physical invocation to exist in the retained ledger."""
    starts = {event.attempt_id for event in events if event.event == "started"}
    if any(value.attempt_id not in starts for value in observations):
        raise ValueError("Baseline raw evidence lacks its physical attempt")


def _read_cell_binding(spec_path, coordinate, plan, runtime, native_run_id) -> NativeBaselineBinding:
    """Read recorded native configuration as bounded data; never follow its original host paths."""
    validate_source_storage(spec_path)
    with spec_path.open("rb") as stream:
        payload = stream.read(16 * 1024**2 + 1)
    if len(payload) > 16 * 1024**2:
        raise ValueError("Baseline saved native specification exceeds 16 MiB")
    spec = RunSpec.model_validate_json(payload)
    binding = NativeBaselineBinding.from_json(spec.extensions["baseline_matrix_v1"])
    if (
        binding.coordinate != coordinate
        or binding.plan_fingerprint != plan.fingerprint
        or binding.runtime_fingerprint != runtime.fingerprint
        or compute_run_id(spec) != native_run_id
    ):
        raise ValueError("Baseline saved native specification differs from its matrix cell")
    return binding


def _verify_cell_attempts(root, cell, coordinate) -> tuple[BaselineAttempt, ...]:
    """Verify the complete physical cost ledger independently of saved native build evidence."""
    events = _attempts(root, coordinate)
    if tuple(event.fingerprint for event in events) != cell.attempts:
        raise ValueError("Baseline matrix physical attempt membership changed")
    unclosed, observed = _attempt_summary(events)
    if unclosed != cell.unclosed_attempts or cell.resource_evidence != ("observed" if observed else "unavailable"):
        raise ValueError("Baseline report resource completeness differs from actual attempts")
    if cell.state == "failed" and not any(
        event.state == "failed" and event.reasons == cell.reasons for event in events
    ):
        raise ValueError("Baseline failed cell lacks its actual failed physical attempt")
    return events
