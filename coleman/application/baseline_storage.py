"""Actual-byte and independent-metric validation for native baseline completion artifacts."""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from coleman.application.baseline_resources import BaselineResourceGuard
from coleman.artifacts import ArtifactWriter
from coleman.corpus.baseline_contracts import BaselinePlan
from coleman.corpus.baseline_reference import baseline_reference
from coleman.corpus.baseline_results import (
    BaselineCoordinate,
    BaselineFilePin,
    BaselineNativeCompletion,
    BaselineObservation,
    baseline_measurement,
)
from coleman.corpus.materialization import validate_source_storage
from coleman.results.domain import decode_json, fingerprint
from coleman.results.domain_artifacts import DomainArtifacts
from coleman.results.versioned import VersionedContract

_NATIVE_COMPLETION = "baseline-native-complete.json"
_OPERATIONAL = frozenset(
    {"manifest.json", "spec.resolved.json", "provenance.json", "native-execution.lock", _NATIVE_COMPLETION}
)


def read_baseline_payload(path: Path) -> bytes:
    """Read bounded trusted local artifact bytes without interpreting an executable object.

    Parameters
    ----------
    path : pathlib.Path
        Local artifact location, checked for unsafe source storage before opening.

    Returns
    -------
    bytes
        At most 16 MiB of original content. The caller selects the explicit
        versioned reader; no migration, repair or pickle decoding occurs.

    Raises
    ------
    ValueError, OSError
        Unsafe location, unreadable content or artifact exceeding 16 MiB.
    """
    validate_source_storage(path)
    with path.open("rb") as stream:
        payload = stream.read(16 * 1024**2 + 1)
    if len(payload) > 16 * 1024**2:
        raise ValueError("Baseline JSON artifact exceeds 16 MiB")
    return payload


def read_baseline_artifact[Contract: VersionedContract](path: Path, contract: type[Contract]) -> Contract:
    """Read at most 16 MiB of a strict versioned local artifact; do not execute/repair/migrate it."""
    return contract.from_json(read_baseline_payload(path))


def _same(actual: float | None, expected: float | None) -> bool:
    """Compare nullable finite quantities without turning unknown values into zero."""
    return (
        actual is None
        if expected is None
        else actual is not None and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    )


def _check_science(observation: BaselineObservation, plan: BaselinePlan, cell: BaselineCoordinate) -> None:
    """Recompute source-based ratios/work/censoring rather than trusting self-consistent JSON digests."""
    build = plan.source.builds[observation.step - 1]
    measured = {item.name: item.value for item in observation.record.measurements}
    source_work = math.fsum(test.duration.value for test in build.tests if test.duration.value is not None)
    if measured["source_full_runner_seconds"] != source_work:
        raise ValueError("Baseline raw source work does not match the frozen universe")
    work = measured["source_full_runner_seconds"]
    budget = measured["declared_native_work_budget_seconds"]
    if work is None or budget is None or budget != work * float(cell.budget.value):
        raise ValueError("Baseline raw budget differs from the frozen native ratio")
    reference = baseline_reference(build, observation.ranking, budget)
    if observation.selected != reference.selected:
        raise ValueError("Baseline raw selection differs from its independent source reference")
    expected = {
        "full_universe_napfd": reference.napfd,
        "full_universe_apfdc": reference.apfdc,
        "failing_test_recall": reference.recall,
        "selected_runner_seconds": reference.selected_work,
        "projected_full_runner_seconds": reference.projected_full_work,
        "censored_first_failure_seconds": reference.first_failure,
        "native_budget_excess_seconds": reference.budget_excess,
    }
    if any(not _same(measured[name], value) for name, value in expected.items()):
        raise ValueError("Baseline raw mathematical measurements failed independent source verification")


def load_baseline_observations(
    root: Path, plan: BaselinePlan, cell: BaselineCoordinate, native_run_id: str, binding_fingerprint: str
) -> tuple[BaselineObservation, ...]:
    """Read all committed raw observations, including partial evidence after a failed/interrupted execution.

    Checks frozen context and independent mathematics without requiring a native
    checkpoint for each partial sidecar. Full completion additionally validates
    actual native Parquet, domain-sidecar and checkpoint correspondence.
    """
    validate_source_storage(root)
    plan_digest, source_digest = plan.fingerprint, plan.source.fingerprint
    builds = {build.step: build for build in plan.source.builds}
    build_digests = {step: fingerprint(build.model_dump(mode="json")) for step, build in builds.items()}
    found = []
    for index, path in enumerate(sorted((root / "baseline-observations").glob("*.json")), start=1):
        if index > len(builds):
            raise ValueError("Native baseline has extra raw build artifacts")
        observation = read_baseline_artifact(path, BaselineObservation)
        if (
            observation.step not in builds
            or path.name != f"step-{observation.step:04d}.json"
            or observation.cell_id != cell.cell_id
            or observation.plan_fingerprint != plan_digest
            or observation.source_build_fingerprint != build_digests[observation.step]
            or observation.member_id != builds[observation.step].member_id
            or observation.record.run_id != native_run_id
            or observation.record.data_fingerprint != source_digest
            or observation.record.provenance.source_revision != plan.source.source_fingerprint
            or observation.record.provenance.spec_fingerprint != binding_fingerprint
        ):
            raise ValueError("Native baseline raw artifact is not bound to its actual frozen source/cell/specification")
        _check_science(observation, plan, cell)
        found.append(observation)
    return tuple(sorted(found, key=lambda item: item.step))


def _protected_files(root: Path) -> tuple[Path, ...]:
    """Inventory completed native files; operational root metadata and interrupted temporary files are separate."""
    paths = []
    for index, path in enumerate(root.rglob("*"), start=1):
        if index > 65536:
            raise ValueError("Native baseline file inventory exceeds its supported bound")
        if path.is_symlink():
            raise ValueError("Native baseline artifact tree contains a symlink")
        if path.is_file() and path.relative_to(root).as_posix() not in _OPERATIONAL and not path.name.endswith(".tmp"):
            paths.append(path)
    return tuple(sorted(paths))


def _pin_file(root: Path, path: Path) -> BaselineFilePin:
    """Hash complete local bytes with a 1 GiB per-file bound; never load a learning pickle."""
    validate_source_storage(path)
    count, digest = 0, hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            count += len(block)
            if count > 1073741824:
                raise ValueError("Native baseline artifact exceeds its supported byte bound")
            digest.update(block)
    return BaselineFilePin(
        path=path.relative_to(root).as_posix(), sha256="sha256:" + digest.hexdigest(), size_bytes=count
    )


def _check_native_rows(
    root: Path, observations: tuple[BaselineObservation, ...], plan: BaselinePlan, cell: BaselineCoordinate
) -> None:
    """Check the actual unchanged native table against raw source-derived evidence, excluding variable retry timings."""
    table = _bounded_native_table(root, len(plan.source.builds))
    if table.height != len(plan.source.builds) or set(table["step"].to_list()) != {
        build.step for build in plan.source.builds
    }:
        raise ValueError("Completed baseline native table requires exactly one row per declared build")
    indexed = {observation.step: observation for observation in observations}
    for row in table.iter_rows(named=True):
        observation = indexed[row["step"]]
        if row["policy"] != cell.method or tuple(decode_json(row["prioritization_order_top_k"])) != observation.ranking:
            raise ValueError("Native baseline candidate/ranking differs from its raw evidence")
        measured = {item.name: item.value for item in observation.record.measurements}
        expected = {
            "fitness": measured["full_universe_napfd"] if measured["full_universe_napfd"] is not None else 1.0,
            "cost": measured["full_universe_apfdc"] if measured["full_universe_apfdc"] is not None else 1.0,
            "budget_value": float(cell.budget.value),
            "total_build_duration": measured["source_full_runner_seconds"],
            "tests_ran": len(observation.selected),
            "tests_not_ran": len(observation.ranking) - len(observation.selected),
        }
        if any(not _same(row[name], value) for name, value in expected.items()):
            raise ValueError("Native baseline table metrics differ from independently checked raw evidence")


def _check_native_complete(
    root: Path, plan: BaselinePlan, cell: BaselineCoordinate, observations, dataset_id: str
) -> None:
    """Require all source steps, their original domain records and a fully committed native checkpoint frontier."""
    if tuple(item.step for item in observations) != tuple(range(1, len(plan.source.builds) + 1)):
        raise ValueError("Completed native baseline has missing raw builds")
    domains = DomainArtifacts(root / "results")
    for observation in observations:
        record = observation.record
        if domains.read(record.run_id, record.execution_id, record.decision_id, namespace=record.namespace) != record:
            raise ValueError("Completed native baseline has missing or conflicting typed domain sidecars")
    progress_path = root / "checkpoints" / dataset_id / "progress_ex1.json"
    validate_source_storage(progress_path)
    with progress_path.open("rb") as stream:
        payload = stream.read(16385)
    if len(payload) > 16384:
        raise ValueError("Native baseline checkpoint progress metadata exceeds 16 KiB")
    progress = decode_json(payload)
    if type(progress.get("step_committed")) is not int or progress["step_committed"] != len(plan.source.builds):
        raise ValueError("Completed native baseline has an incomplete native checkpoint frontier")
    _check_native_rows(root, observations, plan, cell)


def publish_native_baseline_completion(
    root: Path,
    plan: BaselinePlan,
    cell: BaselineCoordinate,
    binding_fingerprint: str,
    runtime_fingerprint: str,
    dataset_id: str,
    guard: BaselineResourceGuard,
) -> BaselineNativeCompletion:
    """Validate actual native work, then publish completion last while the child still owns its execution lock."""
    native_run_id = root.name
    path = root / _NATIVE_COMPLETION
    if path.exists():
        return validate_native_baseline_completion(
            root, plan, cell, native_run_id, binding_fingerprint, runtime_fingerprint, dataset_id
        )
    observations = load_baseline_observations(root, plan, cell, native_run_id, binding_fingerprint)
    _check_native_complete(root, plan, cell, observations, dataset_id)
    pins = tuple(_pin_file(root, entry) for entry in _protected_files(root))
    guard.check()
    resources = guard.tracker.sample()
    receipt = BaselineNativeCompletion(
        schema_version=1,
        cell_id=cell.cell_id,
        plan_fingerprint=plan.fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        native_run_id=native_run_id,
        recorded_at=datetime.now(UTC),
        completed_steps=len(observations),
        observations=tuple(item.fingerprint for item in observations),
        mathematical_observations=tuple(item.mathematical_fingerprint for item in observations),
        files=pins,
        worker_elapsed=baseline_measurement("native_process_elapsed_seconds", resources.wall_time_seconds),
        worker_cpu=baseline_measurement("native_process_cpu_seconds", resources.cpu_time_seconds),
        worker_peak=baseline_measurement("native_process_peak_mib", guard.peak_mib()),
    )
    payload = receipt.to_json() + "\n"
    if len(payload.encode("utf-8")) > 16 * 1024**2:
        raise ValueError("Native baseline completion receipt exceeds 16 MiB")
    guard.check_publication(len(payload.encode("utf-8")))
    ArtifactWriter(root).write_immutable(path, payload)
    return receipt


def validate_native_baseline_completion(
    root: Path,
    plan: BaselinePlan,
    cell: BaselineCoordinate,
    native_run_id: str,
    binding_fingerprint: str,
    runtime_fingerprint: str,
    dataset_id: str,
) -> BaselineNativeCompletion:
    """Reverify native completion against actual files/source, without repairs, network or pickle execution."""
    receipt = read_baseline_artifact(root / _NATIVE_COMPLETION, BaselineNativeCompletion)
    if (receipt.cell_id, receipt.plan_fingerprint, receipt.runtime_fingerprint, receipt.native_run_id) != (
        cell.cell_id,
        plan.fingerprint,
        runtime_fingerprint,
        native_run_id,
    ):
        raise ValueError("Native baseline completion context differs from its frozen execution")
    actual = tuple(_pin_file(root, path) for path in _protected_files(root))
    if actual != receipt.files:
        raise ValueError("Native baseline completion bytes or artifact membership changed")
    observations = load_baseline_observations(root, plan, cell, native_run_id, binding_fingerprint)
    _check_native_complete(root, plan, cell, observations, dataset_id)
    if receipt.observations != tuple(
        item.fingerprint for item in observations
    ) or receipt.mathematical_observations != tuple(item.mathematical_fingerprint for item in observations):
        raise ValueError("Native baseline completion raw evidence digests changed")
    return receipt


def _bounded_native_table(root: Path, build_count: int) -> pl.DataFrame:
    """Check every native Parquet footer before decoding bounded table bytes, rows and columns."""
    rows, decoded = 0, 0
    paths = list((root / "results").rglob("*.parquet"))
    for path in paths:
        with pq.ParquetFile(path) as source:
            rows += source.metadata.num_rows
            decoded += sum(source.metadata.row_group(i).total_byte_size for i in range(source.metadata.num_row_groups))
            if rows > build_count or decoded > 64 * 1024**2 or len(source.schema_arrow) > 64:
                raise ValueError("Native baseline table exceeds its bounded row/column/decoded-byte limits")
    return pl.read_parquet(paths, hive_partitioning=True, glob=False)
