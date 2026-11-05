"""Bounded portable workflow evidence; imports are read-only and never executable."""

from __future__ import annotations

import base64
from typing import Annotated, Literal, Self
from urllib.parse import unquote

from pydantic import Field, model_validator

from coleman.application.contracts import PortableFile
from coleman.application.workflow_import_header import _source_header
from coleman.application.workflow_validation import expected_trials_for_report
from coleman.results.domain import Fingerprint
from coleman.workflows.contracts import ExperimentPlan, WorkflowContract, WorkflowError, WorkflowId
from coleman.workflows.lifecycle import WorkflowReport


class WorkflowPortableFile(PortableFile):
    """Longer workflow Hive names; application-v1 file contracts stay unchanged.

    Parameters
    ----------
    name : str
        At most 512 canonical relative path characters. Existing byte-count,
        SHA-256 and base64 invariants are inherited without reinterpretation.
    """

    name: Annotated[str, Field(max_length=512, strict=True)]


class _EvidenceSourceHeader(WorkflowContract):
    """Strict source/version projection; the opaque body is validated separately before retention."""

    artifact_type: Literal["coleman-workflow-evidence-v1"] = "coleman-workflow-evidence-v1"
    plan: ExperimentPlan


def evidence_source_plan(payload: str) -> ExperimentPlan:
    """Validate a bounded source header before any native outcome materialization.

    Parameters
    ----------
    payload : str
        At most 2 MiB of UTF-8 portable JSON. Other body fields remain opaque to
        this preflight; it does not validate or admit their native measurements.

    Returns
    -------
    ExperimentPlan
        Strict declared source identity. Callers must reserve conservative exposure
        before full EvidenceBundle validation and must validate the full body before
        retention. No Parquet, checkpoint or execution is opened by this function.

    Raises
    ------
    ValueError
        Oversized/nonobject JSON, duplicate keys, unknown versions or invalid plan.
        No files are written and no scientific source authenticity is granted.
    """
    return _EvidenceSourceHeader.from_json(_source_header(payload)).plan


def _native_table(file: WorkflowPortableFile, maximum_decoded: int = 67108864):
    """Decode bounded original bytes using their unchanged Hive partition metadata."""
    from io import BytesIO
    from pathlib import PurePosixPath

    import polars as pl
    import pyarrow.parquet as pq

    content = base64.b64decode(file.content_base64, validate=True)
    with pq.ParquetFile(BytesIO(content)) as source:
        decoded = sum(source.metadata.row_group(i).total_byte_size for i in range(source.metadata.num_row_groups))
        if source.metadata.num_rows > 512 or decoded > maximum_decoded or len(source.schema_arrow) > 64:
            raise ValueError("Portable native decode bounds exceeded")
    parts = [part.partition("=") for part in PurePosixPath(file.name).parts if "=" in part]
    values = {key: unquote(value) for key, _, value in parts}
    required = {"scenario", "policy", "reward_function", "budget_mode", "budget_value"}
    if set(values) != required or len(parts) != len(required):
        raise ValueError("Portable native Hive metadata is missing or duplicated")
    table = pl.read_parquet(content)
    if required.intersection(table.columns):
        raise ValueError("Ambiguous portable inline/Hive partition columns")
    return table.with_columns(
        pl.lit(float(value) if key == "budget_value" else value).alias(key) for key, value in values.items()
    ), decoded


def _verify_receipt(plan, receipt, files, maximum_decoded=67108864):
    """Require exact planned coordinate, full build membership and original native data."""
    import polars as pl

    from coleman.application.workflow_artifacts import native_rows

    trial = next((trial for trial in plan.trials if trial.trial_id == receipt.trial_id), None)
    stage = "descriptive" if plan.request.evaluation == "descriptive" else "evaluation"
    if trial is None or receipt.plan_fingerprint != plan.fingerprint or receipt.stage != stage:
        raise ValueError("Portable report contains an unplanned coordinate or stage")
    if len(receipt.rows) != len(plan.dataset.build_ids):
        raise ValueError("Portable completion requires every declared build")
    tables, rows, decoded = [], 0, 0
    for pin in receipt.files:
        file = files[pin.name]
        if ("sha256:" + file.sha256, file.size_bytes) != (pin.sha256, pin.size_bytes):
            raise ValueError("Portable native artifact differs from its receipt pin")
        if not file.name.endswith(".parquet"):
            raise ValueError("Only opaque native Parquet evidence can be imported")
        table, size = _native_table(file, maximum_decoded - decoded)
        rows, decoded = rows + table.height, decoded + size
        if rows > len(plan.dataset.build_ids) or decoded > 67108864:
            raise ValueError("Portable native decode bounds exceeded")
        tables.append(table)
    try:
        combined = pl.concat(tables)
    except pl.exceptions.PolarsError as error:
        raise WorkflowError("Portable native table schema is incompatible across its files") from error
    if native_rows(combined, plan, trial) != receipt.rows:
        raise ValueError("Portable report metrics differ from actual unchanged native bytes")
    return decoded


class EvidenceBundle(WorkflowContract):
    """Portable plan/report/native-byte membership, with no executable checkpoints.

    Parameters
    ----------
    plan : ExperimentPlan
        Complete abstract native specifications, producer/runtime and source identity.
        Importing retains evidence only; it does not register datasets or executable plans.
    report : WorkflowReport
        Completed evidence with byte-checked native metric values and internally
        consistent declared coordinates. Attribution to trial, seed, policy,
        budget, source and native run identity remains producer asserted; native
        Parquet does not independently bind these execution claims.
    files : tuple[PortableFile, ...]
        Exact completed native Parquet bytes. Every report pin must be represented
        once; at most 1 MiB original native bytes and 64 MiB aggregate decoded
        Parquet row groups. Footer checks precede materialization. Encoded portable
        JSON is bounded to 2 MiB at export/import boundaries.

    Notes
    -----
    Digests check consistency, not authenticity, empirical admission or a trusted
    producer. Imported evidence can be inspected/exported but never resumed/run.
    Native ``cost`` remains APFDc and is never relabeled as physical/money cost.
    """

    artifact_type: Literal["coleman-workflow-evidence-v1"] = "coleman-workflow-evidence-v1"
    plan: ExperimentPlan
    report: WorkflowReport
    files: Annotated[tuple[WorkflowPortableFile, ...], Field(max_length=4096)]

    @property
    def native_run_ids_verified(self) -> Literal[False]:
        """Return false: native path-derived IDs are producer claims, not byte-verified provenance.

        This nonserialized trust declaration leaves existing v1 canonical bytes
        and digests unchanged. File membership, native metrics and logical trial
        definitions are checked against the declared plan. The association of
        those coordinates to actual executions is not independently verified;
        this property never authenticates a producer. Consumers must not promote these IDs to trusted provenance.
        """
        return False

    @property
    def native_execution_bindings_verified(self) -> Literal[False]:
        """Return false: native trial/seed/treatment/source attribution is a producer claim.

        Logical coordinate definitions, native metric values and declared
        membership are checked for consistency. Legacy Parquet has no seed or
        authoritative treatment/source binding; Hive labels are producer supplied.
        This nonserialized qualification preserves existing v1 bytes and digests.
        Producer trust or independent reproduction must establish the attribution.
        """
        return False

    @model_validator(mode="after")
    def _membership(self) -> Self:
        """Validate full logical/native membership without extracting or executing imports."""
        if self.report.plan_fingerprint != self.plan.fingerprint:
            raise ValueError("Portable workflow plan/report binding mismatch")
        files = {file.name: file for file in self.files}
        if len(files) != len(self.files):
            raise ValueError("Duplicate portable native artifact")
        if sum(file.size_bytes for file in self.files) > 1048576:
            raise ValueError("Portable native evidence exceeds 1 MiB")
        expected = {pin.name for trial in self.report.trials for pin in trial.files}
        if set(files) != expected:
            raise ValueError("Portable native membership differs from the completed report")
        seen, decoded = set(), 0
        for receipt in self.report.trials:
            if receipt.trial_id in seen:
                raise ValueError("Duplicate portable logical trial")
            seen.add(receipt.trial_id)
            decoded += _verify_receipt(self.plan, receipt, files, 67108864 - decoded)
        expected_trials = expected_trials_for_report(self.plan, self.report)
        if seen != expected_trials:
            raise ValueError("Portable completion requires every planned/selected trial")
        return self


class ImportedEvidence(WorkflowContract):
    """Caller-owned retained handle to a validated read-only portable bundle.

    Parameters
    ----------
    evidence_id : str
        Caller-scoped content handle; never an executable job or plan ID.
    digest : str
        Complete canonical evidence fingerprint, not authentication.
    native_run_ids_verified : bool
        Always false. Original path-derived native IDs are retained as explicit
        producer claims; portable validation does not authenticate those claims.
    native_execution_bindings_verified : bool
        Always false. Trial/seed, treatment/configuration and source attribution
        remain producer assertions; byte integrity does not prove execution.
    read_only : bool
        Always true; importing neither grants dataset admission nor native resume.
    """

    evidence_id: WorkflowId
    digest: Fingerprint
    read_only: Literal[True] = True
    native_run_ids_verified: Literal[False] = False
    native_execution_bindings_verified: Literal[False] = False
