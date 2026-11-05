"""Retained workspace corruption, source preflight and evidence compatibility boundaries."""

import copy
import io
import json
from contextlib import closing
from functools import partial

import polars as pl
import pytest

from coleman.application.workflow_artifacts import pin_file, verify_trial
from coleman.application.workflow_checkpoint import WorkflowRandomState
from coleman.application.workflow_downloads import artifact_page
from coleman.application.workflow_exports import EvidenceBundle
from coleman.application.workflow_exposure import mark_exposure
from coleman.application.workflow_registry import inspect_history, owned_fixture
from coleman.application.workflow_store import WorkflowStore
from coleman.application.workflows import WorkflowService
from coleman.artifacts import ArtifactWriter
from coleman.workflows.contracts import PlanningRequest
from coleman.workflows.lifecycle import ExecutePlan
from tests.workflows.test_execution import setup_workflow, wait


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "missing",
        "float_build",
        "float_verdict",
        "numeric_name",
        "string_duration",
        "null_name",
        "huge_build",
        "duplicate",
        "too_many_builds",
    ],
)
def test_source_shape_is_rejected_before_registration(tmp_path, kind):
    manifest = owned_fixture(tmp_path)
    source = tmp_path / "datasets" / f"{manifest.sha256[7:]}.parquet"
    frame = pl.read_parquet(source)
    if kind == "empty":
        frame = frame.head(0)
    elif kind == "missing":
        frame = frame.drop("LastRun")
    elif kind == "float_build":
        frame = frame.with_columns(pl.col("BuildId").cast(pl.Float64))
    elif kind == "float_verdict":
        frame = frame.with_columns(pl.col("Verdict").cast(pl.Float64))
    elif kind == "numeric_name":
        frame = frame.with_columns(pl.lit(1).alias("Name"))
    elif kind == "string_duration":
        frame = frame.with_columns(pl.lit("1.0").alias("Duration"))
    elif kind == "null_name":
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("Name"))
    elif kind == "huge_build":
        frame = frame.with_columns(pl.lit(2**64 - 1, dtype=pl.UInt64).alias("BuildId"))
    elif kind == "duplicate":
        frame = pl.concat([frame, frame.head(1)])
    else:
        frame = pl.concat([frame.head(1)] * 513).with_columns(pl.int_range(0, 513).alias("BuildId"))
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    encoded_source = buffer.getvalue()
    with pytest.raises(ValueError):
        inspect_history(encoded_source)


def test_empty_encoded_source_and_invalid_rng_stream_reject():
    with pytest.raises(ValueError, match="encoded"):
        inspect_history(b"")
    with pytest.raises(ValueError, match="odd stream"):
        WorkflowRandomState(
            schema_version=1, binding="sha256:" + "a" * 64, state=0, increment=2, has_uint32=0, uinteger=0
        )


@pytest.mark.parametrize(
    "corruption", ["policy", "dataset_key", "dataset_bound", "plan_key", "plan_missing", "job_bound"]
)
def test_corrupt_retained_indexes_do_not_change_owned_identity(tmp_path, corruption):
    service, preview = setup_workflow(tmp_path)
    with service.store.transaction() as connection:
        if corruption == "policy":
            connection.execute("DELETE FROM settings WHERE name='policy'")
        elif corruption == "dataset_key":
            connection.execute("UPDATE datasets SET id='another-id'")
        elif corruption == "dataset_bound":
            row = connection.execute("SELECT payload FROM datasets").fetchone()[0]
            connection.executemany("INSERT INTO datasets VALUES (?,?)", [(f"extra-{i}", row) for i in range(32)])
        elif corruption == "plan_key":
            connection.execute("UPDATE plans SET id=?", ("0" * 64,))
        elif corruption == "job_bound":
            connection.executemany(
                "INSERT INTO jobs VALUES (?,?,?,?,?)",
                [(f"job-{i}", "owner", f"request-{i}", "plan", "{}") for i in range(129)],
            )
    if corruption == "policy":
        operation = partial(WorkflowStore, tmp_path)
    elif corruption.startswith("dataset"):
        operation = service.discover
    elif corruption.startswith("plan"):
        operation = partial(service.saved_plan, "0" * 64)
    else:
        operation = partial(service.status, "job-0")
    with pytest.raises((ValueError, KeyError)):
        operation()


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    service, preview = setup_workflow(tmp_path_factory.mktemp("retained-guards"))
    job = service.execute(ExecutePlan(schema_version=1, request_id="retained", plan_id=preview.plan_id))
    wait(service, job.job_id)
    return service, job, service.saved_plan(preview.plan_id), service.report(job.job_id)


@pytest.mark.parametrize("change", ["binding", "steps", "source", "duplicate_files", "missing_bytes", "napfd", "apfdc"])
def test_local_receipt_corruption_cannot_be_reused(complete, tmp_path, change):
    service, job, plan, report = complete
    receipt = report.trials[0]
    trial = next(trial for trial in plan.trials if trial.trial_id == receipt.trial_id)
    data = copy.deepcopy(receipt.model_dump(mode="json"))
    if change == "binding":
        data["plan_fingerprint"] = "sha256:" + "0" * 64
    elif change == "steps":
        data["rows"][0]["step"] = 2
    elif change == "source":
        data["rows"][0]["source_build_id"] = 99
    elif change == "duplicate_files":
        data["files"].append(data["files"][0])
    elif change in ("napfd", "apfdc"):
        data["rows"][0][change] = 0.25 if data["rows"][0][change] != 0.25 else 0.75
    else:
        data["files"][0]["name"] = "absent.parquet"
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(data))
    with pytest.raises((ValueError, OSError)):
        verify_trial(service.store.root / "jobs" / job.job_id, path, plan, trial, "descriptive")


def test_local_receipt_bounds_aggregate_decoding_before_reading_rows(complete, tmp_path, monkeypatch):
    _, _, plan, report = complete
    receipt = report.trials[0]
    trial = next(t for t in plan.trials if t.trial_id == receipt.trial_id)
    target = tmp_path / receipt.files[0].name
    target.parent.mkdir(parents=True)
    pl.DataFrame({"oversized": [str(i) + "x" * (24 * 1024**2) for i in range(3)]}).write_parquet(
        target, use_pyarrow=True, pyarrow_options={"use_dictionary": False}
    )
    pin = pin_file(tmp_path, target)
    changed = receipt.model_copy(update={"files": (pin,)})
    path = tmp_path / "receipt.json"
    path.write_text(changed.to_json())

    def forbidden(*args, **kwargs):
        raise AssertionError("Oversized native footers must reject before row decoding")

    monkeypatch.setattr(pl, "read_parquet", forbidden)
    with pytest.raises(ValueError, match="decoded-byte"):
        verify_trial(tmp_path, path, plan, trial, "descriptive")


def test_imported_descriptive_evidence_rejects_duplicate_physical_reservations(complete):
    service, job, _, _ = complete
    bundle = service.export(job.job_id)
    data = bundle.model_dump(mode="json")
    data["report"]["physical_attempts"].append(data["report"]["physical_attempts"][0])
    with pytest.raises(ValueError, match="physical attempts"):
        EvidenceBundle.model_validate(data)


def test_source_exposure_retry_keeps_original_clock_and_rejects_rebinding(complete, tmp_path):
    _, _, plan, _ = complete
    mark_exposure(tmp_path, plan, "owned-exposure", 1, "descriptive_replay")
    before = next((tmp_path / "source-exposures").rglob("*.json")).read_bytes()
    mark_exposure(tmp_path, plan, "owned-exposure", 1, "descriptive_replay")
    assert next((tmp_path / "source-exposures").rglob("*.json")).read_bytes() == before
    with pytest.raises(ValueError, match="conflicts"):
        mark_exposure(tmp_path, plan, "owned-exposure", 1, "read_only_import")


def test_source_receipt_handle_binds_actual_registered_projection(complete):
    service, job, plan, _ = complete
    path = service.store.root / "source-receipts" / f"{plan.dataset.projection_fingerprint[7:]}.json"
    ArtifactWriter(service.store.root).write_immutable(path, "{}")
    try:
        with pytest.raises(ValueError, match="projection receipt"):
            artifact_page(service, job.job_id)
    finally:
        path.unlink()


def test_offline_approved_preset_uses_real_owned_etl_and_retains_receipt(tmp_path, monkeypatch):
    from coleman.application import workflow_presets as presets
    from coleman.application.workflow_downloads import _membership
    from tests.corpus.test_lrts_etl import prepare

    recipe, cache = prepare(tmp_path / "source")
    resource = tmp_path / "resources"
    resource.mkdir()
    (resource / "lrts-karaf-first-eight-v1.json").write_text(recipe.to_json())
    # Only choose owned source bytes. The actual ETL, validation, registration,
    # snapshot storage and receipt publication run without substitutes.
    monkeypatch.setattr(presets, "files", lambda package: resource)
    service = WorkflowService(tmp_path / "workspace")
    result = presets.register_karaf(service, cache, tmp_path / "etl")
    assert result.receipt.observed_cases == 10
    assert result.receipt.replay_cases == 4
    assert service.discover().datasets[0].protected_eligible is False
    receipt = service.store.root / "source-receipts" / f"{result.receipt.fingerprint[7:]}.json"
    assert receipt.read_text() == result.receipt.to_json()
    assert presets.register_karaf(service, cache, tmp_path / "etl") == result
    preview = service.plan(
        PlanningRequest(schema_version=1, dataset_id=result.manifest.dataset_id, policies=("Random",), seeds=(0,))
    )
    job = service.execute(ExecutePlan(schema_version=1, request_id="source-receipt", plan_id=preview.plan_id))
    wait(service, job.job_id)
    _, items = _membership(service, job.job_id)
    assert sum(item.kind == "source_receipt" for item, _ in items) == 1
    with closing(service.store.connect()) as connection:
        assert service.store.datasets(connection) == (result.manifest,)
