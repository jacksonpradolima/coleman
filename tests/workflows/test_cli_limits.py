"""Shared CLI lifecycle, trusted registration and finite admission/compatibility boundaries."""

import argparse
import json
import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import polars as pl
import pytest

from coleman.application.workflow_cli import configure_parser, run_command
from coleman.application.workflow_registry import owned_fixture, snapshot_history
from coleman.application.workflows import WorkflowService
from coleman.results.domain import fingerprint
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.workflows.contracts import PlanningRequest, WorkflowLimits
from coleman.workflows.lifecycle import ExecutePlan, HostPolicy


def staged_variant(root, offset):
    """Prepare distinct authored native bytes outside the workspace for admission tests."""
    manifest = owned_fixture(root)
    source = root / "input.parquet"
    pl.read_parquet(root / "datasets" / f"{manifest.sha256[7:]}.parquet").with_columns(
        (pl.col("Duration") + offset).alias("Duration")
    ).write_parquet(source)
    return snapshot_history(
        root,
        source,
        dataset_id=f"authored-{offset}",
        source_fingerprint=fingerprint(["authored-native-source", offset]),
        projection_fingerprint=fingerprint(["authored-native-projection", offset]),
        evidence="descriptive_source",
        assumptions=("Authored source, not historical evidence.",),
    )


@pytest.mark.parametrize("quota", ["count", "bytes"])
def test_staged_admission_counts_unregistered_snapshot_debris(tmp_path, quota):
    root = tmp_path / "workspace"
    service = WorkflowService(
        root,
        policy=HostPolicy(schema_version=1, max_datasets=1 if quota == "count" else 16, max_registered_bytes=65536),
    )
    directory = root / "datasets"
    directory.mkdir()
    orphan = directory / "interrupted-publication.parquet"
    original = b"x" * (65536 if quota == "bytes" else 1)
    orphan.write_bytes(original)
    staging = tmp_path / "staging"
    manifest = staged_variant(staging, 1)
    with pytest.raises(ValueError, match="Physical snapshot"):
        service.register(manifest, snapshot_root=staging)
    assert tuple(directory.iterdir()) == (orphan,)
    assert orphan.read_bytes() == original
    with closing(service.store.connect()) as connection:
        assert service.store.datasets(connection) == ()


def test_metadata_failure_leaves_bounded_reusable_snapshot(tmp_path):
    root = tmp_path / "workspace"
    service = WorkflowService(root, policy=HostPolicy(schema_version=1, max_datasets=1))
    staging = tmp_path / "first"
    first = staged_variant(staging, 1)
    with service.store.transaction() as connection:
        connection.execute(
            "CREATE TRIGGER fail_registration BEFORE INSERT ON datasets "
            "BEGIN SELECT RAISE(ABORT,'injected metadata failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected metadata"):
        service.register(first, snapshot_root=staging)
    artifact = root / "datasets" / f"{first.sha256[7:]}.parquet"
    original = artifact.read_bytes()
    with service.store.transaction() as connection:
        assert service.store.datasets(connection) == ()
        connection.execute("DROP TRIGGER fail_registration")
    other_staging = tmp_path / "second"
    second = staged_variant(other_staging, 2)
    with pytest.raises(ValueError, match="Physical snapshot slots"):
        service.register(second, snapshot_root=other_staging)
    assert service.register(first, snapshot_root=staging) == first
    assert artifact.read_bytes() == original
    assert tuple((root / "datasets").iterdir()) == (artifact,)


def test_competing_staged_registrations_publish_only_the_admitted_snapshot(tmp_path):
    root = tmp_path / "workspace"
    services = [WorkflowService(root, policy=HostPolicy(schema_version=1, max_datasets=1)), WorkflowService(root)]
    staged = [(tmp_path / "first", 1), (tmp_path / "second", 2)]
    manifests = [staged_variant(path, offset) for path, offset in staged]

    def register(index):
        try:
            return services[index].register(manifests[index], snapshot_root=staged[index][0])
        except ValueError as error:
            assert "registry is full" in str(error)
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        admitted = [value for value in pool.map(register, (0, 1)) if value is not None]
    assert len(admitted) == 1
    assert {p.name for p in (root / "datasets").iterdir()} == {f"{admitted[0].sha256[7:]}.parquet"}


def test_original_source_etl_output_cannot_bypass_workspace_quotas(tmp_path):
    from coleman.application.workflow_presets import register_karaf
    from coleman.corpus.materialization import SourceCache

    service = WorkflowService(tmp_path / "workspace")
    cache = SourceCache(tmp_path / "cache")
    output = service.store.root / "etl"
    with pytest.raises(ValueError, match="outside the workflow workspace"):
        register_karaf(service, cache, output)


@pytest.mark.parametrize("reason", ["count", "bytes", "conflict", "manifest"])
def test_rejected_cli_registration_leaves_no_unregistered_snapshot(tmp_path, reason):
    root = tmp_path / "workspace"
    policy = HostPolicy(schema_version=1, max_datasets=1 if reason == "count" else 16, max_registered_bytes=65536)
    service = WorkflowService(root, policy=policy)
    existing = service.register(owned_fixture(root))
    baseline = {p.name: p.read_bytes() for p in (root / "datasets").iterdir()}
    frame = pl.read_parquet(root / "datasets" / f"{existing.sha256[7:]}.parquet")
    frame = frame.with_columns((pl.col("Duration") + 1.0).alias("Duration"))
    if reason == "bytes":
        frame = frame.with_columns(pl.lit(random.Random(17).randbytes(100000).hex()).alias("HostAnnotation"))
    source = tmp_path / "input.parquet"
    frame.write_parquet(source)
    if reason == "bytes":
        assert source.stat().st_size > policy.max_registered_bytes
    manifest = snapshot_history(
        tmp_path / "staging",
        source,
        dataset_id=existing.dataset_id if reason == "conflict" else "additional-history",
        source_fingerprint=fingerprint(["owned-host-history", reason]),
        projection_fingerprint=fingerprint(["owned-host-projection", reason]),
        evidence="descriptive_source",
        assumptions=("Authored fixture, not historical evidence.",),
    )
    if reason == "manifest":
        manifest = manifest.model_copy(update={"rows": manifest.rows + 1})
    reviewed = tmp_path / "reviewed.json"
    reviewed.write_text(manifest.to_json())
    with pytest.raises(ValueError):
        invoke(root, "register", "--input", str(source), "--manifest", str(reviewed))
    assert {p.name: p.read_bytes() for p in (root / "datasets").iterdir()} == baseline


def invoke(root, operation, *arguments):
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    run_command(parser.parse_args(["--workspace", str(root), operation, *arguments]))


def test_cli_actual_count_budget_native_lifecycle_and_portable_evidence(tmp_path, capsys):
    invoke(tmp_path, "init-fixture")
    manifest = json.loads(capsys.readouterr().out)
    reviewed = tmp_path / "reviewed.json"
    reviewed.write_text(json.dumps(manifest))
    source = tmp_path / "datasets" / f"{manifest['sha256'][7:]}.parquet"
    invoke(tmp_path, "register", "--input", str(source), "--manifest", str(reviewed))
    assert json.loads(capsys.readouterr().out) == manifest
    request = {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "policies": ["Random"],
        "seeds": [0],
        "budget": {"mode": "subset_size", "unit": "count", "values": [2.0]},
    }
    invoke(tmp_path, "plan", "--request", json.dumps(request))
    plan = json.loads(capsys.readouterr().out)
    invoke(
        tmp_path,
        "execute",
        "--request",
        json.dumps({"schema_version": 1, "request_id": "cli", "plan_id": plan["plan_id"]}),
    )
    job = json.loads(capsys.readouterr().out)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        invoke(tmp_path, "status", "--job", job["job_id"])
        job = json.loads(capsys.readouterr().out)
        if job["state"] == "completed" and not job["worker_active"]:
            break
        assert job["state"] in ("queued", "running", "completed"), job
        time.sleep(0.05)
    assert job["state"] == "completed"
    invoke(tmp_path, "inspect", "--job", job["job_id"], "--limit", "1")
    inspection = json.loads(capsys.readouterr().out)
    assert all(row["tests_ran"] == 2 for row in inspection["trials"][0]["rows"])
    invoke(tmp_path, "report", "--job", job["job_id"])
    report = json.loads(capsys.readouterr().out)
    assert len(report["trials"]) == 1
    assert len(report["physical_attempts"]) == 1
    for operation in ("cancel", "resume"):
        invoke(tmp_path, operation, "--job", job["job_id"])
        assert json.loads(capsys.readouterr().out)["state"] == "completed"
    invoke(tmp_path, "export", "--job", job["job_id"])
    bundle = capsys.readouterr().out
    portable = tmp_path / "portable.json"
    portable.write_text(bundle)
    invoke(tmp_path, "import", "--input", str(portable))
    assert json.loads(capsys.readouterr().out)["read_only"] is True
    invoke(tmp_path, "cleanup")
    assert json.loads(capsys.readouterr().out)["expired_jobs"] == []
    manifest["rows"] += 1
    reviewed.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs"):
        invoke(tmp_path, "register", "--input", str(source), "--manifest", str(reviewed))


@pytest.mark.parametrize("operation", ["register", "register-karaf", "plan", "execute", "import", "status", "inspect"])
def test_missing_cli_configuration_is_never_guessed(tmp_path, operation):
    with pytest.raises(ValueError):
        invoke(tmp_path, operation)


def test_cli_rejects_oversized_portable_input_before_decode(tmp_path):
    path = tmp_path / "too-large.json"
    path.write_bytes(b"x" * 2097153)
    with pytest.raises(ValueError, match="2 MiB"):
        invoke(tmp_path, "import", "--input", str(path))
    with pytest.raises(ValueError, match="bounded"):
        invoke(tmp_path, "plan", "--request", "x" * 65537)


def test_queue_tombstone_registry_and_active_memory_caps(tmp_path, monkeypatch):
    policy = HostPolicy(schema_version=1, max_jobs=2, max_datasets=1, max_running=1, max_queued=0)
    service = WorkflowService(tmp_path, policy=policy)
    manifest = service.register(owned_fixture(tmp_path))
    another_manifest = manifest.model_copy(update={"dataset_id": "another-label"})
    with pytest.raises(ValueError, match="registry is full"):
        service.register(another_manifest)
    preview = service.plan(PlanningRequest(schema_version=1, dataset_id=manifest.dataset_id))
    monkeypatch.setattr(service, "_start", lambda: None)
    job = service.execute(ExecutePlan(schema_version=1, request_id="one", plan_id=preview.plan_id))
    second_request = ExecutePlan(schema_version=1, request_id="two", plan_id=preview.plan_id)
    with pytest.raises(ValueError, match="queue capacity"):
        service.execute(second_request)
    assert service.cancel(job.job_id).state == "cancelled"
    second = service.execute(ExecutePlan(schema_version=1, request_id="two", plan_id=preview.plan_id))
    service.cancel(second.job_id)
    third_request = ExecutePlan(schema_version=1, request_id="three", plan_id=preview.plan_id)
    with pytest.raises(ValueError, match="tombstone"):
        service.execute(third_request)
    constrained = WorkflowService(
        tmp_path / "memory", policy=HostPolicy(schema_version=1, max_active_memory_bytes=67108864)
    )
    constrained.register(owned_fixture(constrained.store.root))
    memory_request = PlanningRequest(schema_version=1, dataset_id=manifest.dataset_id)
    with pytest.raises(ValueError, match="aggregate active memory"):
        constrained.plan(memory_request)


def test_changed_runtime_and_native_thread_profile_reject_before_admission(tmp_path, monkeypatch):
    import coleman.application.workflows as module

    service = WorkflowService(tmp_path)
    manifest = service.register(owned_fixture(tmp_path))
    preview = service.plan(PlanningRequest(schema_version=1, dataset_id=manifest.dataset_id))
    request = ExecutePlan(schema_version=1, request_id="compatible-only", plan_id=preview.plan_id)
    actual = InstalledRuntime.capture()
    incompatible = actual.model_copy(update={"package_fingerprint": "sha256:" + "0" * 64})
    with monkeypatch.context() as patch:
        patch.setattr(InstalledRuntime, "capture", lambda: incompatible)
        with pytest.raises(ValueError, match="incompatible"):
            service.execute(request)
    threads = module.pl.thread_pool_size()
    with monkeypatch.context() as patch:
        patch.setattr(module.pl, "thread_pool_size", lambda: threads + 1)
        with pytest.raises(ValueError, match="incompatible"):
            service.execute(request)
    with service.store.transaction() as connection:
        assert not service.store.jobs(connection)


@pytest.mark.parametrize(
    "limits",
    [
        WorkflowLimits(attempt_seconds=0.0001),
        WorkflowLimits(max_memory_bytes=67108864),
        WorkflowLimits(max_output_bytes=65536),
    ],
)
def test_real_worker_resource_exhaustion_never_certifies_completion(tmp_path, limits):
    from tests.workflows.test_execution import setup_workflow, wait

    service, preview = setup_workflow(tmp_path, limits=limits)
    job = service.execute(ExecutePlan(schema_version=1, request_id="finite-resource-budget", plan_id=preview.plan_id))
    view = wait(service, job.job_id, "failed")
    assert view.result_digest is None
    with pytest.raises(ValueError, match="completed report"):
        service.report(job.job_id)
