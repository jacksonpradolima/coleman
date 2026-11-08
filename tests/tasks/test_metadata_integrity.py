"""Actual SQLite corruption and complete task-declaration refusal boundaries.

All persistence is real and the nested models are validated complete values.
The authored proof fixture declares receipt joins; it is not evidence of an
executed provider or ACK. No provider, codec, factory or mathematical callback
is substituted or executed in these metadata tests.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from coleman.application.execution_bootstrap import bind_host, validate_queued_tasks
from coleman.application.workflow_store import WorkflowStore
from coleman.core.identity import Reference
from coleman.tasks.contracts import OwnedCommitProof, TaskJobRecord, TaskPlan
from coleman.tasks.identity import HostBootstrapProfile
from coleman.workflows.lifecycle import HostPolicy

from .test_storage_records import FIXTURES, _native, _payloads, _plans, _put_plan, _task


def changed(value, **updates):
    """Revalidate a complete altered model; never construct an unchecked DTO."""
    return type(value).model_validate({**value.model_dump(), **updates})


def snapshot(path):
    """Read exact logical schema/rows/version independently of WorkflowStore."""
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute("PRAGMA user_version").fetchone()[0], tuple(connection.iterdump())


def plan():
    """Read the independently authored complete two-coordinate task declaration."""
    return TaskPlan.from_json((FIXTURES / "plan.json").read_bytes())


@pytest.mark.parametrize("damage", ("future", "nonempty_virgin", "missing_table", "extra_table"))
def test_workspace_schema_refusal_preserves_exact_corrupt_input_without_repair(tmp_path, damage):
    store = WorkflowStore(tmp_path)
    with closing(sqlite3.connect(store.path)) as connection:
        if damage == "future":
            connection.execute("PRAGMA user_version=2")
        elif damage == "nonempty_virgin":
            connection.execute("PRAGMA user_version=0")
        elif damage == "missing_table":
            connection.execute("DROP TABLE imports")
        else:
            connection.execute("CREATE TABLE unexpected(payload TEXT)")
        connection.commit()
    before = snapshot(store.path)
    with pytest.raises(ValueError, match="Unknown/corrupt workflow workspace schema"):
        WorkflowStore(tmp_path)
    assert snapshot(store.path) == before


def test_missing_immutable_policy_is_not_replaced_by_constructor_defaults(tmp_path):
    store = WorkflowStore(tmp_path)
    with store.transaction() as connection:
        connection.execute("DELETE FROM settings WHERE name='policy'")
    before = snapshot(store.path)
    with pytest.raises(ValueError, match="missing its immutable policy"):
        WorkflowStore(tmp_path)
    assert snapshot(store.path) == before


def test_existing_v1_delete_journal_reopen_preserves_full_policy_and_rows(tmp_path):
    policy = HostPolicy(schema_version=1, max_jobs=3, retention_seconds=123.0)
    store = WorkflowStore(tmp_path, policy)
    native, _ = _plans()
    with store.transaction() as connection:
        _put_plan(connection, native)
        store.save_job(connection, _native(native), create=True)
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    before = snapshot(store.path)
    reopened = WorkflowStore(tmp_path, policy)
    assert reopened.policy.to_json() == policy.to_json()
    assert snapshot(store.path) == before
    with closing(reopened.connect()) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert reopened.jobs(connection)[0].to_json() == _native(native).to_json()


def test_real_exclusive_metadata_lock_propagates_connect_error_and_recovers_after_release(tmp_path):
    store = WorkflowStore(tmp_path)
    with closing(sqlite3.connect(store.path)) as owner:
        owner.execute("PRAGMA locking_mode=EXCLUSIVE")
        owner.execute("BEGIN EXCLUSIVE")
        before = owner.execute("SELECT payload FROM settings WHERE name='policy'").fetchone()[0]
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            store.connect()
        assert owner.execute("SELECT payload FROM settings WHERE name='policy'").fetchone()[0] == before
        owner.rollback()
    with closing(store.connect()) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("damage", ("capacity", "index"))
def test_strict_native_scan_refuses_corrupt_counts_or_identity_without_filtering(tmp_path, damage):
    store = WorkflowStore(tmp_path, HostPolicy(schema_version=1, max_jobs=1))
    native, _ = _plans()
    with store.transaction() as connection:
        _put_plan(connection, native)
        original = _native(native)
        store.save_job(connection, original, create=True)
        if damage == "capacity":
            store.save_job(connection, changed(original, job_id="second", request_id="second-command"), create=True)
        else:
            connection.execute("UPDATE jobs SET owner='different-index-owner'")
        before = _payloads(connection)
        with pytest.raises(ValueError, match="exceeds its policy|metadata identity mismatch"):
            store.jobs(connection)
        assert _payloads(connection) == before


@pytest.mark.parametrize("damage", ("missing_row", "payload_identity", "ignored_update"))
def test_failed_task_update_never_accepts_a_lost_or_misbound_metadata_write(tmp_path, damage):
    store = WorkflowStore(tmp_path)
    _, selected = _plans()
    original = _task(selected)
    with store.transaction() as connection:
        _put_plan(connection, selected)
        store.save_record(connection, original, create=True)
        candidate = changed(original, state="cancelled")
        if damage == "missing_row":
            candidate = changed(candidate, job_id="absent")
        elif damage == "payload_identity":
            corrupt = changed(original, owner="payload-only-foreign-owner")
            connection.execute("UPDATE jobs SET payload=?", (corrupt.to_json(),))
        else:
            # A real SQLite no-op must not be mistaken for accepted metadata.
            connection.execute("CREATE TRIGGER suppress_update BEFORE UPDATE ON jobs BEGIN SELECT RAISE(IGNORE); END")
        before = _payloads(connection)
        with pytest.raises(ValueError, match="Cannot replace workflow job identity|metadata identity mismatch"):
            store.save_record(connection, candidate)
        assert _payloads(connection) == before
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1


def test_queued_task_cannot_acquire_a_native_plan_through_corrupted_retained_indices(tmp_path):
    store = WorkflowStore(tmp_path)
    native, selected = _plans()
    profile = HostBootstrapProfile.from_json((FIXTURES / "bootstrap.json").read_bytes())
    bind_host(store, profile, None)
    job = changed(_task(selected), plan_id=native.fingerprint[7:])
    with store.transaction() as connection:
        _put_plan(connection, native)
        # Deliberate raw persistence corruption bypasses save_record's join.
        connection.execute(
            "INSERT INTO jobs VALUES (?,?,?,?,?)",
            (job.job_id, job.owner, job.request_id, job.plan_id, job.to_json()),
        )
        assert store.records(connection)[0].to_json() == job.to_json()
        before = _payloads(connection)
        with pytest.raises(ValueError, match="Queued task requires its exact task plan family"):
            validate_queued_tasks(store, connection, profile)
        assert _payloads(connection) == before
    assert not (store.root / "jobs").exists()


@pytest.mark.parametrize("damage", ("executor", "coordinate_capacity"))
def test_complete_plan_refuses_an_unrelated_recipe_or_underdeclared_catalogue_capacity(damage):
    selected = plan()
    raw = selected.model_dump()
    if damage == "executor":
        raw["executor"] = changed(selected.executor, reference=Reference(id="unrelated-recipe", version="1"))
    else:
        raw["limits"] = changed(selected.limits, max_coordinates=1)
    before = selected.to_json()
    with pytest.raises(ValueError, match="captured complete recipe|coordinate count exceeds"):
        TaskPlan.model_validate(raw)
    assert selected.to_json() == before


def test_complete_job_refuses_more_completed_coordinates_than_its_own_total():
    original = TaskJobRecord.from_json((FIXTURES / "job.json").read_bytes())
    raw = {**original.model_dump(), "completed_coordinates": original.total_coordinates + 1}
    before = original.to_json()
    with pytest.raises(ValueError, match="Completed task coordinates exceed"):
        TaskJobRecord.model_validate(raw)
    assert original.to_json() == before


@pytest.mark.parametrize("damage", ("generation", "current_definition"))
def test_complete_receipt_proof_cannot_join_a_different_generation_or_current_collection(damage):
    original = OwnedCommitProof.from_json((FIXTURES / "owned-proof.json").read_bytes())
    if damage == "generation":
        head = changed(original.receipt.accepted_head, generation=3)
        receipt = changed(original.receipt, accepted_head=head, current_head=head)
        observed = head
    else:
        observed = changed(original.observed_head, generation=3, definition_fingerprint="sha256:" + "f" * 64)
        receipt = changed(original.receipt, status="retained", current_head=observed)
    raw = {**original.model_dump(), "receipt": receipt, "observed_head": observed}
    before = original.to_json()
    with pytest.raises(ValueError, match="acknowledged generation differ|current head changed its collection"):
        OwnedCommitProof.model_validate(raw)
    assert original.to_json() == before


def test_proof_fingerprint_join_does_not_make_a_foreign_coordinate_a_plan_member():
    selected = plan()
    original = OwnedCommitProof.from_json((FIXTURES / "owned-proof.json").read_bytes())
    coordinate = selected.coordinates[0]
    foreign = changed(coordinate, body=changed(coordinate.body, body=coordinate.body.body + " "))
    proof = changed(original, coordinate_fingerprint=foreign.fingerprint)
    before = proof.to_json()
    with pytest.raises(ValueError, match="coordinate is not a member"):
        proof.validate_for(selected, foreign)
    assert proof.to_json() == before


@pytest.mark.parametrize("damage", ("duplicate_distribution", "missing_root", "missing_selected_extra"))
def test_complete_image_requires_unique_roots_and_every_selected_dependency_extra(damage):
    original = plan().implementation
    raw = original.model_dump()
    if damage == "duplicate_distribution":
        raw["distributions"] = (*original.distributions, original.distributions[0])
    elif damage == "missing_root":
        raw["distributions"] = tuple(item for item in original.distributions if item.name != "coleman")
    else:
        root = original.recipe.roots[-1]
        selected_root = changed(root, extras=("optional-feature",))
        raw["recipe"] = changed(
            original.recipe,
            roots=tuple(selected_root if item.name == root.name else item for item in original.recipe.roots),
        )
    before = original.to_json()
    with pytest.raises(ValueError, match="unique complete root distributions|include every declared selection"):
        type(original).model_validate(raw)
    assert original.to_json() == before


def test_selected_extra_order_canonicalizes_but_duplicate_selection_never_does():
    original = plan().implementation.distributions[-1]
    first = changed(original, extras=("z-feature", "a-feature"))
    second = changed(original, extras=("a-feature", "z-feature"))
    assert first.model_dump_json() == second.model_dump_json()
    duplicate = {**first.model_dump(), "extras": ("a-feature", "a-feature")}
    with pytest.raises(ValueError, match="extras must be unique"):
        type(first).model_validate(duplicate)
