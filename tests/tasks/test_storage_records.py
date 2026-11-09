"""Actual five-table metadata joins for mixed native and typed task records."""

from __future__ import annotations

import json
import sqlite3
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from coleman.application import _workflow_records
from coleman.application.workflow_store import JobRecord, WorkflowStore
from coleman.core.identity import Reference
from coleman.tasks.contracts import EncodedValue, TaskFrontier, TaskJobRecord, TaskPlan
from coleman.workflows.contracts import ExperimentPlan
from coleman.workflows.lifecycle import HostPolicy

FIXTURES = Path(__file__).parents[1] / "fixtures" / "tasks" / "v1"
NATIVE = Path(__file__).parents[1] / "workflows" / "fixtures" / "experiment-plan-v1.json"


def _plans():
    return ExperimentPlan.from_json(NATIVE.read_bytes()), TaskPlan.from_json((FIXTURES / "plan.json").read_bytes())


def _put_plan(connection, plan, owner="owner", payload=None):
    identity = plan.fingerprint.removeprefix("sha256:")
    connection.execute(
        "INSERT INTO plans VALUES (?,?,?)", (identity, owner, plan.to_json() if payload is None else payload)
    )
    return identity


def _native(plan, *, job_id="native", owner="owner", request_id="native-request"):
    return JobRecord(
        schema_version=1,
        job_id=job_id,
        owner=owner,
        request_id=request_id,
        request_digest="sha256:" + "1" * 64,
        plan_id=plan.fingerprint[7:],
        state="queued",
        stage="descriptive",
        created=0.0,
        changed=0.0,
        total_trials=len(plan.trials),
    )


def _task(plan, *, job_id="task", owner="owner", request_id="task-request", **updates):
    return TaskJobRecord(
        contract="coleman-tasks-v1",
        schema_version=1,
        artifact_type="job",
        job_id=job_id,
        owner=owner,
        request_id=request_id,
        request_digest="sha256:" + "2" * 64,
        plan_id=plan.fingerprint[7:],
        state="queued",
        stage=plan.stage,
        created=0.0,
        changed=0.0,
        total_coordinates=len(plan.coordinates),
        **updates,
    )


def _payloads(connection):
    return tuple(tuple(row) for row in connection.execute("SELECT * FROM jobs ORDER BY rowid"))


def test_both_known_plan_arms_and_missing_native_contract_keep_exact_bytes(tmp_path):
    native, task = _plans()
    store = WorkflowStore(tmp_path)
    without_tag = json.loads(native.to_json())
    without_tag.pop("contract")
    with store.transaction() as connection:
        native_id = _put_plan(connection, native, payload=json.dumps(without_tag))
        task_id = _put_plan(connection, task)
        assert store.read_plan(connection, native_id, "owner").to_json() == native.to_json()
        assert store.read_plan(connection, task_id, "owner").to_json() == task.to_json()
        assert store.plan(connection, native_id, "owner").to_json() == NATIVE.read_text().rstrip("\n")
        with pytest.raises(ValueError):
            store.plan(connection, task_id, "owner")
        with pytest.raises(KeyError, match="owned"):
            store.read_plan(connection, task_id, "other-owner")


def test_mixed_record_scan_keeps_all_owners_and_native_facade_refuses_instead_of_filtering(tmp_path):
    native, task = _plans()
    store = WorkflowStore(tmp_path)
    first, second = _native(native), _task(task, owner="other")
    with store.transaction() as connection:
        _put_plan(connection, native)
        _put_plan(connection, task, "other")
        store.save_job(connection, first, create=True)
        store.save_record(connection, second, create=True)
        records = store.records(connection)
        assert tuple(type(item) for item in records) == (JobRecord, TaskJobRecord)
        assert tuple(item.owner for item in records) == ("owner", "other")
        with pytest.raises(ValueError):
            store.jobs(connection)
        assert _payloads(connection) == (
            (first.job_id, first.owner, first.request_id, first.plan_id, first.to_json()),
            (second.job_id, second.owner, second.request_id, second.plan_id, second.to_json()),
        )
    assert WorkflowStore(tmp_path).policy == store.policy
    with store.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {
            "settings",
            "datasets",
            "plans",
            "jobs",
            "imports",
        }


@pytest.mark.parametrize(
    "edit",
    ["future", "missing_version", "coerced_version", "unknown_namespace", "wrong_artifact", "missing_task_namespace"],
)
def test_known_routing_rejects_invalid_task_tags_before_provider_code(tmp_path, monkeypatch, edit):
    _, plan = _plans()
    raw = json.loads(plan.to_json())
    if edit == "future":
        raw["schema_version"] = 2
    elif edit == "missing_version":
        raw.pop("schema_version")
    elif edit == "coerced_version":
        raw["schema_version"] = True
    elif edit == "unknown_namespace":
        raw["contract"] = "unapproved-provider-v1"
    elif edit == "wrong_artifact":
        raw["artifact_type"] = "job"
    else:
        raw.pop("contract")
    calls = []

    def forbidden(_entrypoint):
        calls.append("load")
        raise AssertionError("Metadata must not load a provider")

    monkeypatch.setattr(EntryPoint, "load", forbidden)
    store = WorkflowStore(tmp_path)
    with store.transaction() as connection:
        identity = _put_plan(connection, plan, payload=json.dumps(raw))
        before = _payloads(connection)
        with pytest.raises(ValueError):
            store.read_plan(connection, identity, "owner")
        assert _payloads(connection) == before
    assert calls == []


def test_duplicate_tag_and_wrong_content_handle_refuse(tmp_path):
    _, plan = _plans()
    store = WorkflowStore(tmp_path)
    duplicate = plan.to_json().replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1)
    with store.transaction() as connection:
        identity = _put_plan(connection, plan, payload=duplicate)
        with pytest.raises(ValueError, match="Duplicate"):
            store.read_plan(connection, identity, "owner")
        connection.execute("UPDATE plans SET payload=?", (plan.to_json(),))
        connection.execute("UPDATE plans SET id=?", ("0" * 64,))
        with pytest.raises(ValueError, match="identity"):
            store.read_plan(connection, "0" * 64, "owner")


@pytest.mark.parametrize("column", ["id", "owner", "request", "plan"])
def test_mixed_record_scan_rejects_each_index_binding_without_loading_any_plan(tmp_path, column):
    _, plan = _plans()
    store = WorkflowStore(tmp_path)
    record = _task(plan)
    with store.transaction() as connection:
        _put_plan(connection, plan)
        store.save_record(connection, record, create=True)
        connection.execute(f"UPDATE jobs SET {column}=?", ("forged",))
        connection.execute("DELETE FROM plans")
        before = _payloads(connection)
        with pytest.raises(ValueError, match="identity"):
            store.records(connection)
        assert _payloads(connection) == before


def test_record_scan_is_not_a_plan_or_provider_scan_but_transition_requires_plan(tmp_path):
    _, plan = _plans()
    store = WorkflowStore(tmp_path)
    record = _task(plan)
    with store.transaction() as connection:
        _put_plan(connection, plan)
        store.save_record(connection, record, create=True)
        connection.execute("DELETE FROM plans")
        assert store.records(connection) == (record,)
        before = _payloads(connection)
        with pytest.raises(KeyError, match="owned"):
            store.save_record(connection, record)
        assert _payloads(connection) == before


def test_task_update_retains_exact_identity_and_real_named_frontier(tmp_path):
    _, plan = _plans()
    store = WorkflowStore(tmp_path)
    original = _task(plan)
    selected = plan.coordinates[0]
    updated = TaskJobRecord.model_validate(
        original.model_copy(
            update={
                "state": "running",
                "changed": 1.0,
                "invocations": 1,
                "current_coordinate": selected.fingerprint,
                "frontier": TaskFrontier(profile=selected.target.profile, coordinate=selected.reference, position=1),
            }
        )
    )
    with store.transaction() as connection:
        _put_plan(connection, plan)
        store.save_record(connection, original, create=True)
        store.save_record(connection, updated)
        assert store.records(connection) == (updated,)
        assert _payloads(connection)[0][:4] == (original.job_id, original.owner, original.request_id, original.plan_id)
        assert _payloads(connection)[0][4] == updated.to_json()


@pytest.mark.parametrize("change", ["future", "unknown_namespace", "wrong_artifact", "missing_task_namespace"])
def test_record_routing_refuses_malformed_task_arm_without_changing_prior_sql(tmp_path, change):
    _, plan = _plans()
    store = WorkflowStore(tmp_path)
    record = _task(plan)
    raw = json.loads(record.to_json())
    if change == "future":
        raw["schema_version"] = 2
    elif change == "unknown_namespace":
        raw["contract"] = "unapproved-record-v1"
    elif change == "wrong_artifact":
        raw["artifact_type"] = "plan"
    else:
        raw.pop("contract")
    with store.transaction() as connection:
        _put_plan(connection, plan)
        store.save_record(connection, record, create=True)
        connection.execute("UPDATE jobs SET payload=?", (json.dumps(raw),))
        before = _payloads(connection)
        with pytest.raises(ValueError):
            store.records(connection)
        assert _payloads(connection) == before


@pytest.mark.parametrize("foreign", ["owner", "family"])
def test_task_save_requires_its_owned_task_plan_before_insertion(tmp_path, foreign):
    native, plan = _plans()
    store = WorkflowStore(tmp_path)
    original = _task(plan)
    if foreign == "owner":
        saved, owner, bad = plan, "other-owner", original
    else:
        saved, owner = native, "owner"
        bad = original.model_copy(update={"plan_id": native.fingerprint[7:]})
    with store.transaction() as connection:
        _put_plan(connection, saved, owner)
        before = _payloads(connection)
        expected = KeyError if foreign == "owner" else ValueError
        with pytest.raises(expected):
            store.save_record(connection, bad, create=True)
        assert _payloads(connection) == before


@pytest.mark.parametrize("change", ["stage", "coordinate", "position", "profile", "counts"])
def test_task_transition_joins_complete_plan_frontier_and_retains_prior_bytes(tmp_path, change):
    _, plan = _plans()
    store = WorkflowStore(tmp_path)
    selected = plan.coordinates[0]
    frontier = TaskFrontier(profile=selected.target.profile, coordinate=selected.reference, position=1)
    record = _task(plan, current_coordinate=selected.fingerprint, frontier=frontier)
    if change == "stage":
        bad = record.model_copy(update={"stage": Reference(id="other", version="1")})
    elif change == "coordinate":
        bad = record.model_copy(update={"current_coordinate": "sha256:" + "0" * 64})
    elif change == "position":
        bad = record.model_copy(
            update={"frontier": frontier.model_copy(update={"position": selected.target.position + 1})}
        )
    elif change == "profile":
        bad = record.model_copy(update={"frontier": frontier.model_copy(update={"profile": plan.source})})
    else:
        bad = record.model_copy(update={"total_coordinates": 1})
    bad = TaskJobRecord.model_validate(bad)
    with store.transaction() as connection:
        _put_plan(connection, plan)
        store.save_record(connection, record, create=True)
        before = _payloads(connection)
        with pytest.raises(ValueError):
            store.save_record(connection, bad)
        assert _payloads(connection) == before


def test_same_request_key_is_global_across_families_and_failed_insert_keeps_records(tmp_path):
    native, task = _plans()
    store = WorkflowStore(tmp_path)
    first, second = _native(native, request_id="same"), _task(task, request_id="same")
    with store.transaction() as connection:
        _put_plan(connection, native)
        _put_plan(connection, task)
        store.save_record(connection, first, create=True)
        before = _payloads(connection)
        with pytest.raises(sqlite3.IntegrityError):
            store.save_record(connection, second, create=True)
        assert _payloads(connection) == before


def test_shared_record_capacity_counts_generic_rows_and_tombstones(tmp_path):
    native, task = _plans()
    store = WorkflowStore(tmp_path, policy=HostPolicy(schema_version=1, max_jobs=1))
    first = _native(native).model_copy(update={"state": "expired"})
    second = _task(task)
    with store.transaction() as connection:
        _put_plan(connection, task)
        store.save_record(connection, first, create=True)
        store.save_record(connection, second, create=True)
        with pytest.raises(ValueError, match="exceeds its policy"):
            store.records(connection)


def test_family_switch_update_and_native_save_of_task_both_refuse(tmp_path):
    native, task = _plans()
    store = WorkflowStore(tmp_path)
    original = _task(task)
    forged_native = _native(native).model_copy(
        update={"job_id": original.job_id, "request_id": original.request_id, "plan_id": original.plan_id}
    )
    unchecked_native = JobRecord.model_construct(**original.model_dump())
    with store.transaction() as connection:
        _put_plan(connection, task)
        store.save_record(connection, original, create=True)
        before = _payloads(connection)
        with pytest.raises(ValueError):
            store.save_job(connection, unchecked_native)
        with pytest.raises(ValueError):
            store.save_record(connection, forged_native)
        assert _payloads(connection) == before


def test_pure_selected_record_plan_join_rejects_both_foreign_arms():
    native, task = _plans()
    native_job, task_job = _native(native), _task(task)
    with pytest.raises(ValueError, match="task plan family"):
        _workflow_records.require_record_plan(task_job, native)
    with pytest.raises(ValueError, match="native plan family"):
        _workflow_records.require_record_plan(native_job, task)
    wrong = native_job.model_copy(update={"plan_id": "0" * 64})
    with pytest.raises(ValueError, match="retained plan"):
        _workflow_records.require_record_plan(wrong, native)


def test_opaque_task_body_is_retained_without_concrete_codec_execution(tmp_path, monkeypatch):
    _, plan = _plans()
    encoded = EncodedValue(codec=plan.configuration.codec, body='{"not_the_provider_schema":true}')
    plan = TaskPlan.model_validate(plan.model_copy(update={"configuration": encoded}))

    def forbidden(_entrypoint):
        raise AssertionError("Outer metadata cannot select code")

    monkeypatch.setattr(EntryPoint, "load", forbidden)
    store = WorkflowStore(tmp_path)
    with store.transaction() as connection:
        identity = _put_plan(connection, plan)
        restored = store.read_plan(connection, identity, "owner")
        assert isinstance(restored, TaskPlan)
        assert restored.configuration.body == encoded.body


def test_generic_typed_body_limit_precedes_model_validation_and_preserves_sql(tmp_path):
    _, plan = _plans()
    payload = '{"contract":"coleman-tasks-v1","schema_version":1,"padding":"' + "x" * (16 * 1024**2) + '"}'
    store = WorkflowStore(tmp_path)
    with store.transaction() as connection:
        identity = _put_plan(connection, plan, payload=payload)
        with pytest.raises(ValueError, match="16 MiB typed-reader"):
            store.read_plan(connection, identity, "owner")
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0


def test_original_native_reader_accepts_padded_wire_beyond_new_shared_route(tmp_path):
    native, _ = _plans()
    payload = native.to_json() + " " * _workflow_records.ROUTING_MAXIMUM_BYTES
    store = WorkflowStore(tmp_path)
    with store.transaction() as connection:
        identity = _put_plan(connection, native, payload=payload)
        with pytest.raises(ValueError, match="maximum_bytes"):
            store.read_plan(connection, identity, "owner")
        assert store.plan(connection, identity, "owner").to_json() == native.to_json()
