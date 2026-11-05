"""Typed optional MCP tools over the shared durable application service."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, ValidationError

from coleman.application.baseline_lock import BaselineBusyError
from coleman.application.workflow_downloads import (
    ArtifactChunk,
    ArtifactPage,
    ReportOverview,
    artifact_page,
    read_artifact,
    report_overview,
)
from coleman.application.workflow_exports import EvidenceBundle, ImportedEvidence
from coleman.application.workflows import WorkflowService
from coleman.workflows.contracts import PlanningRequest, WorkflowContract, WorkflowError, WorkflowId
from coleman.workflows.lifecycle import (
    ExecutePlan,
    JobView,
    PlanPreview,
    WorkflowDiscovery,
    WorkflowInspection,
)

if TYPE_CHECKING:
    from mcp.server import MCPServer


class WorkflowReply[T](WorkflowContract):
    """Structured bounded tool response with stable errors and missing-field paths.

    Parameters
    ----------
    data : object or None
        Typed successful response; never a JSON string requiring model interpretation.
    error : str or None
        Stable classification; storage/worker exceptions and host paths stay private.
    missing_fields : tuple[str, ...]
        Missing required planning fields, rather than invented conversation values.
    reason : str or None
        Safe package-defined eligibility/cap explanation. Arbitrary exception
        messages and host paths are never exposed.
    """

    data: T | None = None
    error: Literal["invalid_input", "not_found", "conflict_or_limit", "storage_unavailable"] | None = None
    missing_fields: tuple[str, ...] = ()
    reason: Annotated[str, Field(max_length=512)] | None = None


class PlanningInput(PlanningRequest):
    """Typed conversation draft, converted to a strict executable request before planning.

    Parameters
    ----------
    dataset_id : str or None
        Discovered host-approved ID. Missing/None returns structured missing-field
        information with no plan or job admission. All other versioned fields,
        parameter constraints and discovered defaults inherit PlanningRequest.

    Notes
    -----
    This transport-only draft never enters persistence or the native runner.
    The domain PlanningRequest still requires a valid explicit dataset identity.
    """

    dataset_id: WorkflowId | None = None


class JobCommand(WorkflowContract):
    """Versioned caller-owned job operation; a stable ID replaces arbitrary paths.

    Parameters
    ----------
    job_id : str
        Host-issued durable identity. Caller ownership is supplied by the host,
        never this request. Unknown/future versions and unknown fields reject.
    """

    job_id: WorkflowId


class InspectionCommand(JobCommand):
    """Bounded evidence-page request; inherited job identity and version rules apply.

    Parameters
    ----------
    offset : int
        Zero-based retained logical trial offset, at most 128.
    limit : int
        One through eight receipts per response. Context overflows ask for a smaller page.
    """

    offset: Annotated[int, Field(strict=True, ge=0, le=128)] = 0
    limit: Annotated[int, Field(strict=True, ge=1, le=8)] = 4


class EvidenceCommand(WorkflowContract):
    """Read-only import handle; never an executable job/checkpoint identity.

    Parameters
    ----------
    evidence_id : str
        Caller-owned imported evidence ID returned by import_experiment_evidence.
    """

    evidence_id: WorkflowId


class ArtifactPageCommand(JobCommand):
    """Bounded immutable artifact listing; inherited job ownership/version rules apply.

    Parameters
    ----------
    offset, limit : int
        Zero-based membership offset and one through 64 descriptors per page.
    """

    offset: Annotated[int, Field(strict=True, ge=0, le=65538)] = 0
    limit: Annotated[int, Field(strict=True, ge=1, le=64)] = 32


class ArtifactReadCommand(JobCommand):
    """Byte-range request by discovered content handle; arbitrary paths are forbidden.

    Parameters
    ----------
    artifact_id : str
        Discovered caller-owned content handle, never a server filesystem path.
    offset, maximum_bytes : int
        Zero-based byte offset and one through 65,536 response bytes.
    """

    artifact_id: WorkflowId
    offset: Annotated[int, Field(strict=True, ge=0, le=67108864)] = 0
    maximum_bytes: Annotated[int, Field(strict=True, ge=1, le=65536)] = 65536


def _reply[T](operation: Callable[[], T]) -> WorkflowReply[T]:
    """Translate only expected boundary failures; never expose host exception strings."""
    try:
        return WorkflowReply[T](schema_version=1, data=operation())
    except ValidationError as error:
        missing = tuple(".".join(map(str, item["loc"])) for item in error.errors() if item["type"] == "missing")
        return WorkflowReply[T](schema_version=1, error="invalid_input", missing_fields=missing)
    except KeyError:
        return WorkflowReply[T](schema_version=1, error="not_found")
    except WorkflowError as error:
        return WorkflowReply[T](schema_version=1, error="conflict_or_limit", reason=str(error))
    except BaselineBusyError:
        return WorkflowReply[T](
            schema_version=1,
            error="conflict_or_limit",
            reason="Source evidence is active; retry after its owner finishes",
        )
    except ValueError:
        return WorkflowReply[T](schema_version=1, error="conflict_or_limit")
    except (OSError, sqlite3.Error):
        return WorkflowReply[T](schema_version=1, error="storage_unavailable")


def register_workflow_tools(server: MCPServer, service: WorkflowService) -> None:
    """Add typed opt-in durable tools without changing the seven application-v1 tools.

    Parameters
    ----------
    server : mcp.server.MCPServer
        Existing optional SDK server. No SDK/model dependency enters core modules.
    service : WorkflowService
        Host-selected persistent workspace/principal. Tools accept no host paths,
        arbitrary native specifications, plugins, URLs or execution commands.

    Notes
    -----
    Registration alone starts no experiment. The typed planner exposes discovered
    defaults and rejects invalid inputs before admission. Disconnect only detaches
    clients; explicit cancellation/resume use the same application service.
    """

    @server.tool()
    def workflow_discover() -> WorkflowReply[WorkflowDiscovery]:
        """Discover approved history IDs, typed choices/defaults, evidence restrictions and resource units."""
        return _reply(service.discover)

    @server.tool()
    def plan_experiment(request: PlanningInput) -> WorkflowReply[PlanPreview]:
        """Freeze Random/UCB RNFail grid/zip, native budget, seeds and finite work limits; execute nothing."""
        return _reply(lambda: service.plan(PlanningRequest.model_validate(request.model_dump(exclude_none=True))))

    @server.tool()
    def execute_plan(request: ExecutePlan) -> WorkflowReply[JobView]:
        """Execute a saved plan idempotently by retained request ID; no additional confirmation step."""
        return _reply(lambda: service.execute(request))

    @server.tool()
    def experiment_status(request: JobCommand) -> WorkflowReply[JobView]:
        """Inspect durable progress, checkpoints and attempts after reconnect; ETA can be unavailable."""
        return _reply(lambda: service.status(request.job_id))

    @server.tool()
    def cancel_experiment(request: JobCommand) -> WorkflowReply[JobView]:
        """Cancel owned work; inspect worker_active before resuming an interrupted physical attempt."""
        return _reply(lambda: service.cancel(request.job_id))

    @server.tool()
    def resume_experiment(request: JobCommand) -> WorkflowReply[JobView]:
        """Continue compatible native checkpoints and reuse complete trials under the same logical job."""
        return _reply(lambda: service.resume(request.job_id))

    @server.tool()
    def inspect_experiment(request: InspectionCommand) -> WorkflowReply[WorkflowInspection]:
        """Read bounded completed-trial pages with explicit NAPFD/APFDc, modeled seconds and observed runtime."""
        return _reply(lambda: service.inspect(request.job_id, offset=request.offset, limit=request.limit))

    @server.tool()
    def evaluate_experiment(request: JobCommand) -> WorkflowReply[JobView]:
        """Freeze validation choice/state/protocol, then reserve audited holdout access before native continuation."""
        return _reply(lambda: service.evaluate(request.job_id))

    @server.tool()
    def experiment_report(request: JobCommand) -> WorkflowReply[ReportOverview]:
        """Retrieve completed descriptive/protected evidence; tuning success never implies scientific superiority."""
        return _reply(lambda: report_overview(service, request.job_id))

    @server.tool()
    def export_experiment(request: JobCommand) -> WorkflowReply[EvidenceBundle]:
        """Export at most 2 MiB of complete configuration/provenance/report/Parquet; checkpoints stay private."""
        return _reply(lambda: service.export(request.job_id))

    @server.tool()
    def import_experiment_evidence(
        bundle_json: Annotated[str, Field(max_length=2097152)],
    ) -> WorkflowReply[ImportedEvidence]:
        """Import bounded read-only evidence; validate versions/digests/native rows and never execute imports."""
        return _reply(lambda: service.import_evidence(bundle_json))

    @server.tool()
    def read_experiment_evidence(request: EvidenceCommand) -> WorkflowReply[EvidenceBundle]:
        """Read/export a caller-owned imported bundle; never register or execute its plans/checkpoints."""
        return _reply(lambda: service.imported_evidence(request.evidence_id))

    @server.tool()
    def list_experiment_artifacts(request: ArtifactPageCommand) -> WorkflowReply[ArtifactPage]:
        """List immutable complete plan/report/native file handles; omit private checkpoints and raw source inputs."""
        return _reply(lambda: artifact_page(service, request.job_id, offset=request.offset, limit=request.limit))

    @server.tool()
    def read_experiment_artifact(request: ArtifactReadCommand) -> WorkflowReply[ArtifactChunk]:
        """Download at most 64 KiB by an owned artifact handle, with exact byte offsets and SHA-256 digests."""
        return _reply(
            lambda: read_artifact(
                service, request.job_id, request.artifact_id, offset=request.offset, maximum_bytes=request.maximum_bytes
            )
        )
