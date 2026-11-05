"""Bounded report summaries and immutable artifact handles, independent of MCP framing."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, model_validator

from coleman.application.workflow_artifacts import pin_file
from coleman.evaluation.protected import AccessReceipt
from coleman.evaluation.protected.observations import ComparisonResult
from coleman.results.domain import Fingerprint, fingerprint
from coleman.workflows.contracts import WorkflowContract, WorkflowError, WorkflowId

_FINGERPRINT_PREFIX = "sha256:"

if TYPE_CHECKING:
    from coleman.application.workflows import WorkflowService


class ReportOverview(WorkflowContract):
    """Small completed evidence summary; full native data remains in verified artifacts.

    Parameters
    ----------
    job_id, digest, plan_fingerprint : str
        Owned completion and full source/runtime/configuration identity.
    trial_count, native_build_rows : int
        Completed stage trials and cumulative rows, separately from inference units.
    selected_configuration, fallback_configuration : str or None
        Validation/descriptive choice and conservative same-budget Random fallback.
    interpretation : str
        Explicit descriptive-only or insufficient-evidence product outcome.
    native_execution_bindings_verified : bool
        Always false: portable native-byte validation does not independently prove
        trial/seed, treatment/configuration or source attribution. The producer
        must be trusted separately; logical plan and metric checks still apply.
    comparisons, access : tuple or None
        Existing bounded family results and actual persistent protected access receipt.
        Raw inputs/complete report can be retrieved through immutable artifact handles.
    """

    job_id: WorkflowId
    digest: Fingerprint
    plan_fingerprint: Fingerprint
    trial_count: Annotated[int, Field(strict=True, ge=1, le=128)]
    native_build_rows: Annotated[int, Field(strict=True, ge=1, le=65536)]
    selected_configuration: Fingerprint | None
    fallback_configuration: Fingerprint | None
    interpretation: Literal["descriptive_only", "insufficient_evidence"]
    native_execution_bindings_verified: Literal[False] = False
    comparisons: Annotated[tuple[ComparisonResult, ...], Field(max_length=1)] = ()
    access: AccessReceipt | None = None


class EvidenceArtifact(WorkflowContract):
    """Stable content-bound artifact descriptor; never an arbitrary server path.

    Parameters
    ----------
    artifact_id, sha256 : str
        Retained handle and actual complete-byte fingerprint.
    name, kind : str
        Safe relative evidence label and plan/report/native-Parquet role.
    size_bytes : int
        Complete actual artifact size, bounded to 64 MiB.
    """

    artifact_id: WorkflowId
    sha256: Fingerprint
    name: Annotated[str, Field(strict=True, max_length=512)]
    kind: Literal["plan", "report", "native_parquet", "source_receipt"]
    size_bytes: Annotated[int, Field(strict=True, ge=0, le=67108864)]


@dataclass(frozen=True)
class _ArtifactManifest:
    """One connection-local, digest-bound manifest; never persistent executable state.

    At most one completed job is retained by a service, with at most 65,538
    descriptors and the existing 16 MiB limits on each plan/report payload.
    Native/source files remain paths and are rehashed only when requested.
    """

    owner: str
    job_id: str
    result_digest: Fingerprint
    entries: tuple[tuple[EvidenceArtifact, bytes | Path], ...]
    by_id: Mapping[str, tuple[EvidenceArtifact, bytes | Path]]


class ArtifactPage(WorkflowContract):
    """Bounded manifest page, excluding private checkpoints and arbitrary local files.

    Parameters
    ----------
    job_id, result_digest : str
        Owned completed job and report content binding.
    artifacts : tuple[EvidenceArtifact, ...]
        At most 64 approved logical artifact descriptors; no source history bytes.
    total_artifacts : int
        Full retained approved membership, including complete plan and report.
    next_offset : int or None
        Next manifest page, or explicit end of membership.
    """

    job_id: WorkflowId
    result_digest: Fingerprint
    artifacts: Annotated[tuple[EvidenceArtifact, ...], Field(max_length=64)]
    total_artifacts: Annotated[int, Field(strict=True, ge=2, le=65538)]
    next_offset: Annotated[int, Field(strict=True, ge=0, le=65538)] | None


class ArtifactChunk(WorkflowContract):
    """Bounded byte-range response; the consumer verifies the complete artifact digest.

    Parameters
    ----------
    artifact : EvidenceArtifact
        Original immutable descriptor, including total encoded bytes and SHA-256.
    offset : int
        Zero-based byte offset, never a row offset or filesystem path.
    content_base64 : str
        At most 64 KiB decoded bytes. No checkpoint or executable import is served.
    chunk_sha256 : str
        Actual returned bytes' digest; whole-file verification remains required.
    next_offset : int or None
        Next byte offset, or end of the already completed artifact.
    """

    artifact: EvidenceArtifact
    offset: Annotated[int, Field(strict=True, ge=0, le=67108864)]
    content_base64: Annotated[str, Field(strict=True, max_length=87384)]
    chunk_sha256: Fingerprint
    next_offset: Annotated[int, Field(strict=True, ge=0, le=67108864)] | None

    @model_validator(mode="after")
    def _range(self):
        """Reject corrupted/noncanonical bytes, impossible ranges and stalled partial downloads."""
        payload = base64.b64decode(self.content_base64, validate=True)
        end = self.offset + len(payload)
        expected = end if end < self.artifact.size_bytes else None
        if (
            len(payload) > 65536
            or end > self.artifact.size_bytes
            or self.next_offset != expected
            or (not payload and expected is not None)
            or base64.b64encode(payload).decode("ascii") != self.content_base64
            or _FINGERPRINT_PREFIX + hashlib.sha256(payload).hexdigest() != self.chunk_sha256
        ):
            raise WorkflowError("Artifact chunk bytes/digest/range are inconsistent")
        return self


def report_overview(service: WorkflowService, job_id: str) -> ReportOverview:
    """Project verified completion into a bounded summary without statistical recomputation.

    Parameters
    ----------
    service : WorkflowService
        Caller-bound shared application service.
    job_id : str
        Owned completed identity; normal evidence/expiry checks apply.

    Returns
    -------
    ReportOverview
        Counts, choices, conservative outcome and existing audited comparisons.
    """
    report = service.report(job_id)
    return ReportOverview(
        schema_version=1,
        job_id=job_id,
        digest=report.fingerprint,
        plan_fingerprint=report.plan_fingerprint,
        trial_count=len(report.trials),
        native_build_rows=sum(len(trial.rows) for trial in report.trials),
        selected_configuration=report.selected_configuration,
        fallback_configuration=report.fallback_configuration,
        interpretation=report.interpretation,
        comparisons=report.comparison.results if report.comparison else (),
        access=report.comparison.access if report.comparison else None,
    )


def _membership(service: WorkflowService, job_id: str):
    """Resolve only fully verified immutable plan/report/native evidence; exclude learning pickles."""
    report = service.report(job_id)
    plan = service.saved_plan(report.plan_fingerprint[7:])
    root = service.store.root / "jobs" / job_id
    values = []
    for name, kind, payload in (
        ("plan.json", "plan", plan.to_json().encode()),
        ("report.json", "report", report.to_json().encode()),
    ):
        digest = _FINGERPRINT_PREFIX + hashlib.sha256(payload).hexdigest()
        item = EvidenceArtifact(
            schema_version=1,
            artifact_id=fingerprint([kind, digest])[7:],
            sha256=digest,
            name=name,
            kind="plan" if kind == "plan" else "report",
            size_bytes=len(payload),
        )
        values.append((item, payload))
    for receipt in report.trials:
        for pin in receipt.files:
            item = EvidenceArtifact(
                schema_version=1,
                artifact_id=fingerprint([pin.name, pin.sha256])[7:],
                sha256=pin.sha256,
                name=pin.name,
                kind="native_parquet",
                size_bytes=pin.size_bytes,
            )
            values.append((item, root / pin.name))
    source_receipt = service.store.root / "source-receipts" / f"{plan.dataset.projection_fingerprint[7:]}.json"
    if source_receipt.exists():
        pin = pin_file(service.store.root, source_receipt)
        if pin.sha256 != plan.dataset.projection_fingerprint:
            raise WorkflowError("Source projection receipt differs from its registered fingerprint")
        values.append(
            (
                EvidenceArtifact(
                    schema_version=1,
                    artifact_id=fingerprint(["source_receipt", pin.sha256])[7:],
                    sha256=pin.sha256,
                    name="source-projection-receipt.json",
                    kind="source_receipt",
                    size_bytes=pin.size_bytes,
                ),
                source_receipt,
            )
        )
    return report, tuple(values)


def artifact_page(service: WorkflowService, job_id: str, *, offset: int = 0, limit: int = 32) -> ArtifactPage:
    """List stable completed evidence handles without putting entire datasets/reports into tool context.

    Parameters
    ----------
    service : WorkflowService
        Caller-bound shared application service.
    job_id : str
        Owned completed identity.
    offset : int
        Zero-based artifact membership offset, at most 65,538.
    limit : int
        One through 64 descriptors per page.

    Returns
    -------
    ArtifactPage
        Safe plan/report/native handles; private native checkpoints are excluded.

    Raises
    ------
    ValueError, OSError
        Invalid paging, incomplete/expired/corrupt evidence or storage failure.
    """
    if type(offset) is not int or type(limit) is not int or not 0 <= offset <= 65538 or not 1 <= limit <= 64:
        raise WorkflowError("Artifact pages require bounded integer offsets and one through 64 descriptors")
    manifest = _resolved_membership(service, job_id)
    values = manifest.entries
    return ArtifactPage(
        schema_version=1,
        job_id=job_id,
        result_digest=manifest.result_digest,
        artifacts=tuple(item for item, _ in values[offset : offset + limit]),
        total_artifacts=len(values),
        next_offset=offset + limit if offset + limit < len(values) else None,
    )


def _resolved_membership(service: WorkflowService, job_id: str) -> _ArtifactManifest:
    """Verify once per retained job/digest while checking current ownership/state on every read."""
    view = service.status(job_id)
    if view.state != "completed" or view.result_digest is None:
        service._artifact_manifest = None
        raise WorkflowError("Workflow has no completed report")
    manifest = service._artifact_manifest
    binding = (service.owner, job_id, view.result_digest)
    if manifest is not None and (manifest.owner, manifest.job_id, manifest.result_digest) == binding:
        return manifest
    report, entries = _membership(service, job_id)
    if report.fingerprint != view.result_digest or len(entries) > 65538:
        raise WorkflowError("Workflow artifact manifest binding/count mismatch")
    by_id = {item.artifact_id: (item, source) for item, source in entries}
    if len(by_id) != len(entries):
        raise WorkflowError("Workflow artifact manifest has duplicate handles")
    manifest = _ArtifactManifest(service.owner, job_id, report.fingerprint, entries, MappingProxyType(by_id))
    service._artifact_manifest = manifest
    return manifest


def read_artifact(
    service: WorkflowService, job_id: str, artifact_id: str, *, offset: int = 0, maximum_bytes: int = 65536
) -> ArtifactChunk:
    """Read a bounded owned artifact range by content handle, without arbitrary path access.

    Parameters
    ----------
    service : WorkflowService
        Caller-bound shared application service.
    job_id, artifact_id : str
        Owned completed job and discovered immutable artifact handle.
    offset : int
        Zero-based byte offset, within the completed artifact.
    maximum_bytes : int
        One through 65,536 decoded response bytes. The protocol adds base64/framing overhead.

    Returns
    -------
    ArtifactChunk
        Exact bytes, complete-file/chunk digests and next offset. Reassemble and
        verify the whole-file digest; chunks are not executable checkpoint imports.

    Notes
    -----
    Each service retains at most one verified manifest. Subsequent ranges check
    live ownership/completion and the requested file's hash without reparsing
    the report or rehashing unrelated files. Plan/report bytes are immutable
    verified snapshots. Reconnecting or closing the service drops this cache.

    Raises
    ------
    KeyError, ValueError, OSError
        Unknown/foreign handle, invalid bounds, altered/incomplete/expired evidence
        or read failure. No filesystem extraction or native execution occurs.
    """
    if type(offset) is not int or type(maximum_bytes) is not int or offset < 0 or not 1 <= maximum_bytes <= 65536:
        raise WorkflowError("Artifact reads require bounded integer byte offsets and at most 64 KiB")
    manifest = _resolved_membership(service, job_id)
    found = manifest.by_id.get(artifact_id)
    if found is None:
        raise KeyError("No owned workflow evidence artifact")
    item, source = found
    if offset > item.size_bytes:
        raise WorkflowError("Artifact offset exceeds its completed byte size")
    if isinstance(source, bytes):
        payload = source[offset : offset + maximum_bytes]
    else:
        current = pin_file(service.store.root, source)
        if (current.sha256, current.size_bytes) != (item.sha256, item.size_bytes):
            raise WorkflowError("Completed workflow artifact changed")
        with source.open("rb") as stream:
            stream.seek(offset)
            payload = stream.read(maximum_bytes)
    end = offset + len(payload)
    return ArtifactChunk(
        schema_version=1,
        artifact=item,
        offset=offset,
        content_base64=base64.b64encode(payload).decode("ascii"),
        chunk_sha256=_FINGERPRINT_PREFIX + hashlib.sha256(payload).hexdigest(),
        next_offset=end if end < item.size_bytes else None,
    )
