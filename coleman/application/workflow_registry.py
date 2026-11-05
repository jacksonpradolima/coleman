"""Trusted local CI snapshot registration; clients receive IDs, never host paths."""

from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
import pyarrow.parquet as pq

from coleman.application.registry import _validate_native_auxiliary_columns, _validate_native_durations
from coleman.artifacts import ArtifactWriter
from coleman.corpus.materialization import validate_source_storage
from coleman.evaluation.protected import TemporalObservation, TemporalPlan
from coleman.results.domain import fingerprint
from coleman.workflows.contracts import BuildMetadata, DatasetManifest, WorkflowError

if TYPE_CHECKING:
    from coleman.application.workflow_store import WorkflowStore

_REQUIRED = {"BuildId", "Name", "Duration", "Verdict", "CalcPrio", "LastRun"}


def _publish_snapshot(store: WorkflowStore, manifest: DatasetManifest, payload: bytes) -> None:
    """Publish validated bytes under the caller's registry transaction, counting orphan files.

    Existing exact bytes are reusable without another physical reservation. Every
    flat snapshot-directory entry, including publication debris, consumes one
    physical dataset slot and its encoded byte size. The immutable host policy
    bounds both before a new write. Capacity/storage failures propagate without
    deleting shared files or claiming metadata admission. Call only after logical
    ID/count/byte checks while holding WorkflowStore.transaction().
    """
    directory = store.root / "datasets"
    target = directory / f"{manifest.sha256[7:]}.parquet"
    validate_source_storage(directory)
    if not target.exists():
        entries = tuple(directory.iterdir()) if directory.exists() else ()
        for entry in entries:
            validate_source_storage(entry)
        if any(not entry.is_file() for entry in entries):
            raise WorkflowError("Snapshot directory contains unsupported entries")
        if len(entries) >= store.policy.max_datasets:
            raise WorkflowError("Physical snapshot slots exhausted, including unregistered files")
        if sum(entry.stat().st_size for entry in entries) + len(payload) > store.policy.max_registered_bytes:
            raise WorkflowError("Physical snapshot byte limit exceeded, including unregistered files")
    ArtifactWriter(store.root).write_immutable_stream(target, (payload,), maximum_bytes=len(payload))


def inspect_history(payload: bytes) -> pl.DataFrame:
    """Validate trusted native Parquet before preserving its original bytes.

    Parameters
    ----------
    payload : bytes
        At most 64 MiB encoded Parquet. Footer bounds precede decoding: at most
        128 MiB decoded row groups, one million rows and 64 columns.

    Returns
    -------
    polars.DataFrame
        Validated native rows; no values, units, names or source bytes are changed.

    Raises
    ------
    ValueError
        Invalid shape/types, missing fields, non-finite/negative duration,
        native dtype underflow/overflow, nulls or duplicate build/test identities.
    """
    if not 0 < len(payload) <= 67108864:
        raise ValueError("History exceeds its 64 MiB encoded limit")
    with pq.ParquetFile(io.BytesIO(payload)) as source:
        size = sum(source.metadata.row_group(i).total_byte_size for i in range(source.metadata.num_row_groups))
        if source.metadata.num_rows > 1000000 or size > 134217728 or len(source.schema_arrow) > 64:
            raise ValueError("History exceeds decoded row/byte/column bounds")
    frame = pl.read_parquet(payload)
    _native_columns(frame)
    if any(frame[name].null_count() for name in _REQUIRED):
        raise ValueError("Null native source cells cannot be silently converted to zero")
    if not frame["Name"].str.len_chars().is_between(1, 4096).all():
        raise ValueError("Native testcase names require one through 4096 characters")
    if frame["BuildId"].cast(pl.Int64, strict=False).null_count():
        raise ValueError("Native build IDs must fit Int64")
    _validate_native_durations(frame["Duration"])
    _validate_native_auxiliary_columns(frame)
    if (
        not frame["Verdict"].is_in([0, 1]).all()
        or frame.select(pl.struct("BuildId", "Name").is_duplicated().any()).item()
    ):
        raise ValueError("History requires binary outcomes and unique build/test coordinates")
    if frame["BuildId"].n_unique() > 512 or int(max(frame.group_by("BuildId").len()["len"].to_list())) > 10000:
        raise ValueError("History exceeds native build/test candidate limits")
    return frame


def _native_columns(frame: pl.DataFrame) -> None:
    """Require a nonempty native schema before reductions; never coerce values or units."""
    if not frame.height or not _REQUIRED.issubset(frame.columns):
        raise ValueError("History requires nonempty native testcase columns")
    if not frame.schema["BuildId"].is_integer() or not frame.schema["Verdict"].is_integer():
        raise ValueError("Native build IDs and binary verdicts require integer columns")
    if frame.schema["Name"] != pl.String or not frame.schema["Duration"].is_numeric():
        raise ValueError("Native names require strings and durations numeric seconds")


def snapshot_history(
    root: Path,
    path: Path,
    *,
    dataset_id: str,
    source_fingerprint: str,
    projection_fingerprint: str,
    evidence: Literal["constructed_mechanism", "descriptive_source"],
    assumptions: tuple[str, ...],
    metadata: tuple[BuildMetadata, ...] = (),
    temporal_plan: TemporalPlan | None = None,
) -> DatasetManifest:
    """Snapshot bounded host-approved bytes without fetching sources or granting admission.

    Parameters
    ----------
    root : pathlib.Path
        Trusted preparation root; completed bytes publish immutably below datasets/.
        Prefer a temporary root outside the workspace, then pass snapshot_root to
        WorkflowService.register. This preparation function has no host quota or
        admission authority and deliberately performs no registry mutation.
    path : pathlib.Path
        Trusted host input, read once with a 64 MiB cap; tools cannot supply it.
    dataset_id : str
        Stable public registry label, validated before artifact publication.
    source_fingerprint, projection_fingerprint : str
        Actual host-verified source/projection bindings, not inferred from paths.
    evidence : {'constructed_mechanism', 'descriptive_source'}
        Explicit evidentiary scope; no automatic empirical source admission.
    assumptions : tuple[str, ...]
        Source/chronology/scheduling declarations retained by the plan.
    metadata : tuple[BuildMetadata, ...]
        Complete temporal/group/cost declarations, or absent for descriptive data.
    temporal_plan : TemporalPlan or None
        Optional predeclared protected split, validated with existing machinery.

    Returns
    -------
    DatasetManifest
        Exact input digest, dimensions, units and declared evidence metadata.

    Raises
    ------
    ValueError
        Unsafe paths, invalid inputs, incompatible declarations or conflicting bytes.
    OSError
        Input or atomic publication failure. No completed registration is implied.
    """
    validate_source_storage(path)
    with path.open("rb") as stream:
        payload = stream.read(67108865)
    table = inspect_history(payload)
    manifest = DatasetManifest(
        schema_version=1,
        dataset_id=dataset_id,
        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        build_ids=tuple(sorted(table["BuildId"].unique().to_list())),
        rows=table.height,
        maximum_tests=int(max(table.group_by("BuildId").len()["len"].to_list())),
        source_fingerprint=source_fingerprint,
        projection_fingerprint=projection_fingerprint,
        evidence=evidence,
        assumptions=assumptions,
        metadata=metadata,
        temporal_plan=temporal_plan,
    )
    writer = ArtifactWriter(root)
    writer.write_immutable_stream(
        root / "datasets" / f"{manifest.sha256[7:]}.parquet", (payload,), maximum_bytes=len(payload)
    )
    return manifest


def owned_fixture(root: Path) -> DatasetManifest:
    """Register the owned three-build/three-test example with explicitly authored clocks.

    Parameters
    ----------
    root : pathlib.Path
        Trusted workspace; creates only owned deterministic fixture/snapshot files.

    Returns
    -------
    DatasetManifest
        Constructed mechanism evidence, three source builds and one repository
        family. The authored clocks are not historical observations.

    Raises
    ------
    OSError, ValueError
        Publication failure, conflicting prior fixture bytes or invalid metadata.
    """
    rows = [
        {
            "BuildId": build,
            "Name": name,
            "Duration": duration,
            "CalcPrio": 0,
            "LastRun": "2026-01-01",
            "Verdict": verdict,
        }
        for build in (1, 2, 3)
        for name, duration, verdict in (("unit", 1.0, 0), ("integration", 2.0, build % 2), ("system", 4.0, 0))
    ]
    stream = io.BytesIO()
    pl.DataFrame(rows).write_parquet(stream, compression="zstd")
    path = root / "fixture" / "owned-ci-three-build.parquet"
    ArtifactWriter(root).write_immutable_stream(path, (stream.getvalue(),), maximum_bytes=65536)
    metadata = tuple(
        BuildMetadata(
            build_id=build,
            observation=TemporalObservation(
                observation_id=f"owned-build-{build}",
                revision_id="v1",
                group_id=f"owned-build-{build}",
                event_at=datetime(2026, 1, build, tzinfo=UTC),
                available_at=datetime(2026, 1, build, 1, tzinfo=UTC),
            ),
            unit_id="owned-repository-family",
            costs_known_at_decision=True,
        )
        for build in (1, 2, 3)
    )
    return snapshot_history(
        root,
        path,
        dataset_id="owned-ci-three-build-v1",
        source_fingerprint=fingerprint(rows),
        projection_fingerprint=fingerprint(["owned-ci-native-seconds-v1", rows]),
        evidence="constructed_mechanism",
        assumptions=("Authored clocks, fixed predecision costs and synchronous feedback; not historical CI.",),
        metadata=metadata,
        temporal_plan=TemporalPlan(
            train_end=datetime(2026, 1, 1, 12, tzinfo=UTC),
            validation_end=datetime(2026, 1, 2, 12, tzinfo=UTC),
            test_end=datetime(2026, 1, 3, 12, tzinfo=UTC),
        ),
    )
