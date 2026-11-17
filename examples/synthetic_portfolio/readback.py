"""Resolve compact task results through trusted storage and original strict readers."""

from hashlib import sha256

from coleman.application.tasks import TaskService
from coleman.state.storage_contracts import ExecutionCompatibility
from coleman.tasks.protocols import CoordinateContext, PreparationContext, decode_value

from ._state import read_text
from .executor import PortfolioExecutor
from .recipe import CONFIGURATION, COORDINATE, COORDINATE_RESULT, FINAL_RESULT, INPUTS
from .task_contracts import PortfolioCheckpoint
from .task_results import PortfolioCoordinateResultV2, PortfolioReportItemV2


def _read_coordinate_artifact(context: CoordinateContext, item: PortfolioReportItemV2) -> PortfolioCoordinateResultV2:
    """Read one fixed owned artifact and match its complete published identity."""
    selected = PortfolioReportItemV2.model_validate(item)
    text = read_text(context.artifact_directory / "coordinate-result.json", COORDINATE_RESULT.maximum_bytes)
    raw = text.encode("utf-8")
    if (len(raw), "sha256:" + sha256(raw).hexdigest()) != (
        selected.artifact_byte_count,
        selected.artifact_fingerprint,
    ):
        raise ValueError("Portfolio coordinate artifact differs from its complete published reference")
    actual = COORDINATE_RESULT.decode(text)
    if actual.to_json() != selected.coordinate.to_json():
        raise ValueError("Portfolio completion differs from its actual original coordinate artifact")
    if actual.proof.coordinate_fingerprint != context.coordinate_fingerprint:
        raise ValueError("Portfolio coordinate artifact belongs to another original owned coordinate")
    return actual


def read_full_task_result(service: TaskService, job_id: str) -> tuple[PortfolioCheckpoint, ...]:
    """Read every complete original checkpoint named by an owned compact result.

    Parameters
    ----------
    service : TaskService
        Trusted local operator-owned service with explicit current approval.
        Its workspace supplies all paths; request/result JSON supplies no path.
    job_id : str
        Actual completed owned job handle, never a file or import specification.

    Returns
    -------
    tuple of PortfolioCheckpoint
        Complete schema1 checkpoints in original coordinate order, retaining
        every book receipt, predictor statistic, control state and pending label.

    Raises
    ------
    ValueError, TypeError, OSError, ImportError
        Missing/incomplete/expired task, changed or unapproved implementation,
        incompatible codecs, altered summaries/bytes/identity/size, stale ACK or
        unavailable/corrupt original evidence.

    Notes
    -----
    Uses the existing approved concrete-result decoder and complete selected
    state reader. No model is updated, no state command is accepted and no code
    is selected by JSON. This is a local library/artifact operation, not an MCP
    download tool or permission to access another host's filesystem. Content
    hashes establish consistency, not source authentication.
    """
    view = service.status(job_id)
    completion = service.report(job_id)
    plan = service.saved_plan(view.job.plan_id)
    completion.validate_for(plan)
    report = FINAL_RESULT.validate(service.decode_result(job_id))
    if len(report.results) != len(plan.coordinates):
        raise ValueError("Portfolio report lost an original owned coordinate")
    job_root = service.store.root / "jobs" / view.job.job_id
    preparation = PreparationContext(
        plan.fingerprint,
        plan.source,
        plan.implementation.binding,
        ExecutionCompatibility(
            schema_version=1,
            runtime=plan.implementation.runtime,
            owner_implementations=(plan.implementation.binding,),
        ),
        job_root,
    )
    prepared = PortfolioExecutor().prepare(
        decode_value(CONFIGURATION, plan.configuration), decode_value(INPUTS, plan.inputs), preparation
    )
    checkpoints = []
    for coordinate, item in zip(plan.coordinates, report.results, strict=True):
        context = CoordinateContext(
            preparation,
            coordinate.fingerprint,
            coordinate.reference,
            coordinate.target,
            job_root / "evidence" / coordinate.fingerprint.removeprefix("sha256:"),
            seed=coordinate.seed,
        )
        selected = decode_value(COORDINATE, coordinate.body)
        if selected.reference != context.reference:
            raise ValueError("Portfolio evidence differs from its original coordinate")
        actual = _read_coordinate_artifact(context, item)
        checkpoints.append(prepared.resolve_result(actual, context))
    return tuple(checkpoints)
