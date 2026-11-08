"""Independent hostile-input and operational refusal cases for the shared host."""

from __future__ import annotations

import json
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, time, timedelta, timezone, tzinfo
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest

from coleman.application import execution_driver as driver
from coleman.application import execution_lifecycle as lifecycle
from coleman.application import execution_views as views
from coleman.application._workflow_records import read_plan_payload
from coleman.application.workflow_store import WorkflowStore
from coleman.core._wire import guard_graph
from coleman.core.identity import Reference
from coleman.tasks.contracts import TaskJobRecord, TaskPlan
from coleman.tasks.identity import DistributionInventory, ExecutorImplementationIdentity
from coleman.tasks.protocols import CommittedProgress, CoordinateContext
from coleman.workflows.contracts import WorkflowError
from coleman.workflows.lifecycle import HostPolicy
from tests.tasks.test_shared_driver import _publication_boundary, _task_values
from tests.tasks.test_storage_records import _native, _plans


@pytest.mark.parametrize("body", ("null", "0", '"a plan"', "[]"))
def test_mixed_plan_reader_rejects_nonobjects_without_selecting_code(body):
    """Refuse nonobjects before any family or provider can be selected."""
    with pytest.raises(ValueError, match="requires an object"):
        read_plan_payload(body)


@pytest.mark.parametrize("value", (False, "é", '\\"\n\x00', [0, None, -1], {"unit": "1", "value": -0.0}))
def test_graph_preflight_matches_independent_exact_utf8_budget_for_builtin_json(value):
    """Account for every UTF-8 and escape byte against a stdlib encoder oracle."""
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    guard_graph(value, maximum_bytes=len(encoded), maximum_items=100)
    with pytest.raises(ValueError, match="byte budget"):
        guard_graph(value, maximum_bytes=len(encoded) - 1, maximum_items=100)


@pytest.mark.parametrize("value", (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")))
def test_nonfinite_decimal_is_rejected_before_conversion(value):
    """Reject nonfinite decimals before a caller can normalize or serialize them."""
    with pytest.raises(ValueError, match="finite Decimal"):
        guard_graph(value, maximum_bytes=128, maximum_items=10)


@pytest.mark.parametrize("value", ("\ud800", 10**500, 1j))
def test_unrepresentable_or_excessive_scalars_refuse_before_serialization(value):
    """Reject unsafe scalar representations within the declared finite budget."""
    with pytest.raises((ValueError, TypeError)):
        guard_graph(value, maximum_bytes=16, maximum_items=10)


def test_untrusted_timezone_hooks_never_run_during_graph_admission():
    """Reject custom timezone behavior before invoking any user callback."""
    calls = []

    class CallbackTimezone(tzinfo):
        """Trap every custom timezone callback rather than supply a real timezone."""

        def utcoffset(self, dt):
            """Record an unsafe callback if admission accidentally invokes it."""
            calls.append(dt)
            raise AssertionError("An unapproved timezone callback ran")

        def dst(self, dt):
            """Record an unsafe daylight-saving callback."""
            calls.append(dt)
            raise AssertionError("An unapproved timezone callback ran")

        def tzname(self, dt):
            """Record an unsafe name callback."""
            calls.append(dt)
            raise AssertionError("An unapproved timezone callback ran")

    for value in (datetime(2026, 1, 1, tzinfo=CallbackTimezone()), time(12, tzinfo=CallbackTimezone())):
        with pytest.raises(TypeError, match="standard explicit timezone"):
            guard_graph(value, maximum_bytes=128, maximum_items=10)
    assert not calls
    for value in (datetime(2026, 1, 1, tzinfo=UTC), time(12, tzinfo=timezone(timedelta(hours=-3)))):
        guard_graph(value, maximum_bytes=128, maximum_items=10)


@pytest.mark.parametrize("maximum_bytes,maximum_items", ((True, 10), (128, False), (0, 10), (128, -1)))
def test_graph_allowances_are_strict_positive_counts(maximum_bytes, maximum_items):
    """Keep booleans and missing capacity out of graph resource declarations."""
    with pytest.raises(ValueError, match="strict positive integers"):
        guard_graph(None, maximum_bytes=maximum_bytes, maximum_items=maximum_items)


def test_graph_alias_occurrences_and_keys_consume_real_admission_capacity():
    """A shared object is charged per occurrence, and JSON keys stay strings."""
    shared = {"value": 0}
    with pytest.raises(ValueError, match="item budget"):
        guard_graph([shared, shared], maximum_bytes=128, maximum_items=4)
    with pytest.raises(TypeError, match="keys must be built-in strings"):
        guard_graph({1: "untyped-key"}, maximum_bytes=128, maximum_items=10)


def test_plan_publication_is_idempotent_and_orphan_artifacts_retain_slots(tmp_path):
    """Reuse exact plans while interrupted publication still consumes a finite slot."""
    plan, _ = _task_values()
    store = WorkflowStore(tmp_path / "normal", HostPolicy(schema_version=1, max_plans=1))
    lifecycle.save_plan(store, plan, "owner")
    lifecycle.save_plan(store, plan, "owner")
    assert len(list((store.root / "plans").glob("*"))) == 1
    with closing(store.connect()) as connection:
        assert connection.execute("SELECT count(*) FROM plans").fetchone()[0] == 1
    alternative = TaskPlan.model_validate({**plan.model_dump(), "stage": Reference(id="other-stage", version="1")})
    with pytest.raises(WorkflowError, match="plan slots are full"):
        lifecycle.save_plan(store, alternative, "owner")

    orphan_store = WorkflowStore(tmp_path / "orphan", HostPolicy(schema_version=1, max_plans=1))
    orphan = orphan_store.root / "plans" / "interrupted-publication.tmp"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"not a completed plan")
    with pytest.raises(WorkflowError, match="artifacts/orphans"):
        lifecycle.save_plan(orphan_store, plan, "owner")
    assert orphan.read_bytes() == b"not a completed plan"
    with closing(orphan_store.connect()) as connection:
        assert connection.execute("SELECT count(*) FROM plans").fetchone()[0] == 0


@pytest.mark.parametrize("state", ("running", "interrupted", "failed"))
def test_admission_rejects_records_that_claim_existing_work(tmp_path, state):
    """Reject false fresh admissions before creating any metadata row."""
    plan, original = _task_values()
    proposed = TaskJobRecord.model_validate({**original.model_dump(), "state": state})
    store = WorkflowStore(tmp_path)
    with pytest.raises(WorkflowError, match="new admission must be queued"):
        lifecycle.admit_record(store, plan, proposed)
    with closing(store.connect()) as connection:
        assert not store.records(connection)


@pytest.mark.parametrize("state,starts", (("expired", 1), ("interrupted", 8)))
def test_resume_refuses_expired_or_exhausted_workers_without_new_reservation(tmp_path, state, starts):
    """Retain the exact failed resume metadata and physical attempt catalogue."""
    worker, _, _ = _publication_boundary(tmp_path)
    current = TaskJobRecord.model_validate({**worker.job.model_dump(), "state": state, "invocations": starts})
    with worker.store.transaction() as connection:
        worker.store.save_record(connection, current)
    before = current.to_json()
    with pytest.raises(WorkflowError, match="Expired work|invocation allowance exhausted"):
        lifecycle.requeue_record(worker.store, current, worker.plan)
    with closing(worker.store.connect()) as connection:
        assert worker.store.records(connection)[0].to_json() == before
    assert not list((worker.store.root / "jobs").rglob("attempts/*.json"))


def test_expired_cleanup_failure_preserves_tombstone_and_retries_actual_bytes(tmp_path, monkeypatch):
    """A failed deletion keeps its bytes and retained request identity until retry."""
    worker, _, _ = _publication_boundary(tmp_path)
    expired = TaskJobRecord.model_validate({**worker.job.model_dump(), "state": "expired"})
    with worker.store.transaction() as connection:
        worker.store.save_record(connection, expired)
    directory = worker.store.root / "jobs" / expired.job_id
    directory.mkdir(parents=True, exist_ok=True)
    retained = directory / "retained.txt"
    retained.write_text("accepted evidence")
    real_delete = lifecycle.shutil.rmtree

    def unavailable(_path):
        raise PermissionError("constructed storage failure")

    monkeypatch.setattr(lifecycle.shutil, "rmtree", unavailable)
    with pytest.raises(PermissionError):
        lifecycle.cleanup_records(worker.store, expired.owner, TaskJobRecord)
    assert retained.read_text() == "accepted evidence"
    with closing(worker.store.connect()) as connection:
        assert worker.store.records(connection)[0].to_json() == expired.to_json()
    monkeypatch.setattr(lifecycle.shutil, "rmtree", real_delete)
    assert lifecycle.cleanup_records(worker.store, expired.owner, TaskJobRecord) == (expired.job_id,)
    assert not directory.exists()


def test_inventory_rejects_duplicate_extras_and_changed_runtime_versions():
    """An installation declaration must preserve its actual closure identities."""
    plan, _ = _task_values()
    inventory = plan.implementation
    raw = inventory.distributions[0].model_dump()
    with pytest.raises(ValueError, match="extras must be unique"):
        DistributionInventory.model_validate({**raw, "extras": ("ci", "ci")})
    changed = inventory.model_dump()
    original = next(item for item in changed["distributions"] if item["name"] == "coleman")
    original["version"] = "999.0.0"
    with pytest.raises(ValueError, match="exact COLEMAN runtime closure"):
        ExecutorImplementationIdentity.model_validate(changed)


@pytest.mark.parametrize("substitution", ("plan", "coordinate", "profile", "reference", "beyond_target"))
def test_foreign_progress_refuses_before_verification_or_outer_metadata(tmp_path, substitution):
    """Reject substituted progress before storage verification or outer acceptance."""
    worker, program, _ = _publication_boundary(tmp_path)
    assert isinstance(worker.plan, TaskPlan)
    coordinate = program.coordinates[0]
    context = CoordinateContext(
        program.context,
        coordinate.coordinate_fingerprint,
        coordinate.reference,
        coordinate.target,
        coordinate.artifact_directory,
    )
    calls = []
    verifier = SimpleNamespace(validate=lambda *_: calls.append("verify"))
    program = replace(program, executor=cast(Any, SimpleNamespace(progress_verifier=verifier)))
    frontier = coordinate.target
    plan_id, coordinate_id = program.context.plan_fingerprint, coordinate.coordinate_fingerprint
    if substitution == "plan":
        plan_id = "sha256:" + "0" * 64
    elif substitution == "coordinate":
        coordinate_id = "sha256:" + "0" * 64
    elif substitution == "profile":
        frontier = frontier.model_copy(update={"profile": worker.plan.objective})
    elif substitution == "reference":
        frontier = frontier.model_copy(update={"coordinate": Reference(id="another-owner", version="1")})
    else:
        frontier = frontier.model_copy(update={"position": coordinate.target.position + 1})
    unverified = CommittedProgress(plan_id, coordinate_id, frontier, None)
    with pytest.raises(WorkflowError, match="retained plan or coordinate|declared target frontier"):
        worker.record_progress(program, context, unverified)
    assert not calls
    assert driver.read_job(worker.store, worker.job.job_id).to_json() == worker.job.to_json()


def test_physical_reservation_cap_prevents_one_more_real_ledger_entry(tmp_path):
    """Enforce the physical allowance before another actual reservation is written."""
    worker, program, _ = _publication_boundary(tmp_path)
    assert isinstance(worker.plan, TaskPlan)
    coordinate = program.coordinates[0]
    identities = [worker.reserve(coordinate) for _ in range(worker.plan.limits.max_attempts)]
    assert len(set(identities)) == worker.plan.limits.max_attempts
    before = driver.read_job(worker.store, worker.job.job_id).to_json()
    entries = sorted((worker.store.root / "jobs").rglob("*.json"))
    assert len(entries) == worker.plan.limits.max_attempts
    with pytest.raises(WorkflowError, match="attempt limit exhausted"):
        worker.reserve(coordinate)
    assert driver.read_job(worker.store, worker.job.job_id).to_json() == before
    assert sorted((worker.store.root / "jobs").rglob("*.json")) == entries


@pytest.mark.parametrize("interleaving", ("cancelled", "new_invocation"))
def test_reservation_rechecks_actual_transactional_claim_after_concurrent_change(tmp_path, interleaving):
    """A concurrent cancellation or new invocation wins over an earlier observation."""
    worker, program, _ = _publication_boundary(tmp_path)
    changed = []

    def advance_claim(store, job_id):
        observed = driver.read_job(store, job_id)
        changes = {"state": "cancelled"} if interleaving == "cancelled" else {"invocations": observed.invocations + 1}
        current = views.changed(observed, **changes)
        with store.transaction() as connection:
            store.save_record(connection, current)
        changed.append(current)
        return observed

    hooks = driver.WorkerHooks(read_job=advance_claim, peak_rss_mib=lambda: 1.0)
    worker = replace(worker, hooks=hooks)
    with pytest.raises((InterruptedError, WorkflowError), match="cancelled before|reservation changed"):
        worker.reserve(program.coordinates[0])
    assert len(changed) == 1
    assert driver.read_job(worker.store, worker.job.job_id).to_json() == changed[0].to_json()
    assert not list((worker.store.root / "jobs").rglob("*.json"))


def test_existing_common_policy_cannot_be_replaced_on_reopen(tmp_path):
    """Changing host setup must not rewrite retained operational allowances."""
    policy = HostPolicy(schema_version=1, max_jobs=3)
    store = WorkflowStore(tmp_path, policy)
    with closing(store.connect()) as connection:
        before = tuple(connection.iterdump())
    with pytest.raises(ValueError, match="cannot be silently replaced"):
        WorkflowStore(tmp_path, HostPolicy(schema_version=1, max_jobs=4))
    with closing(store.connect()) as connection:
        assert tuple(connection.iterdump()) == before
    assert WorkflowStore(tmp_path, policy).policy == policy


def test_terminal_expiry_deletes_owned_bytes_but_keeps_request_tombstone(tmp_path):
    """Expire constructed failed metadata without inventing accepted work."""
    worker, _, _ = _publication_boundary(tmp_path)
    failed = TaskJobRecord.model_validate({**worker.job.model_dump(), "state": "failed"})
    with worker.store.transaction() as connection:
        worker.store.save_record(connection, failed)
    directory = worker.store.root / "jobs" / failed.job_id
    directory.mkdir(parents=True)
    (directory / "partial.txt").write_text("incomplete output")
    assert lifecycle.cleanup_records(worker.store, failed.owner, TaskJobRecord) == (failed.job_id,)
    assert not directory.exists()
    with closing(worker.store.connect()) as connection:
        retained = worker.store.records(connection)[0]
    assert retained.state == "expired"
    assert retained.request_id == failed.request_id
    assert retained.request_digest == failed.request_digest
    assert retained.result_digest is None


def test_native_stage_change_after_resume_observation_refuses_requeue(tmp_path):
    """A stale continuation cannot overwrite a different committed stage."""
    native, _ = _plans()
    store = WorkflowStore(tmp_path)
    expected = views.changed(_native(native), state="interrupted", invocations=1)
    current = views.changed(expected, stage="evaluation")
    lifecycle.save_plan(store, native, expected.owner)
    with store.transaction() as connection:
        store.save_record(connection, current, create=True)
    with pytest.raises(WorkflowError, match="Continuation changed"):
        lifecycle.requeue_record(store, expected, native)
    with closing(store.connect()) as connection:
        assert store.records(connection)[0].to_json() == current.to_json()
