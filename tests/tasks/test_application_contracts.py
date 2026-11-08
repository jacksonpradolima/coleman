"""Independent application wires and provider-free host authority boundaries."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier

import pytest

from coleman.application.execution_bootstrap import (
    acknowledge_profile,
    bind_host,
    profile_observed,
    require_current_profile,
    stored_policy,
    stored_profile,
    validate_queued_tasks,
    validate_task_caps,
)
from coleman.application.task_runtime import CompletionCodec
from coleman.application.tasks import TaskService
from coleman.application.workflow_store import WorkflowStore
from coleman.core.identity import Reference
from coleman.tasks.contracts import TaskJobRecord, TaskPlan
from coleman.tasks.identity import HostBootstrapProfile
from coleman.tasks.requests import (
    TaskDiscovery,
    TaskExecuteRequest,
    TaskHostPolicy,
    TaskJobView,
    TaskPlanningRequest,
    TaskPlanPreview,
)
from coleman.tasks.results import TaskCompletion

FIXTURES = Path(__file__).parents[1] / "fixtures" / "tasks"
APPLICATION = FIXTURES / "application-v1"
READERS = (
    ("completion.json", TaskCompletion),
    ("planning-request.json", TaskPlanningRequest),
    ("execute-request.json", TaskExecuteRequest),
    ("policy.json", TaskHostPolicy),
    ("view.json", TaskJobView),
    ("preview.json", TaskPlanPreview),
    ("discovery.json", TaskDiscovery),
)


def _profile():
    return HostBootstrapProfile.from_json((FIXTURES / "v1" / "bootstrap.json").read_bytes())


def _plan():
    return TaskPlan.from_json((FIXTURES / "v1" / "plan.json").read_bytes())


def _changed(value, **changes):
    return type(value).model_validate({**value.model_dump(), **changes})


@pytest.mark.parametrize(("name", "reader"), READERS)
def test_independently_authored_application_goldens_round_trip_and_digests(name, reader):
    """Match independent canonical fixture bytes and digests without execution."""
    body = (APPLICATION / name).read_text().removesuffix("\n")
    value = reader.from_json(body)
    independent = "sha256:" + hashlib.sha256(body.encode()).hexdigest()
    hand = json.loads((APPLICATION / "hand.json").read_text())
    assert value.to_json() == body
    assert reader.model_validate_json(body) == value
    assert value.fingerprint == independent == hand["fingerprints"][name]


@pytest.mark.parametrize(("name", "reader"), READERS)
@pytest.mark.parametrize("edit", ("future", "missing", "boolean_version", "unknown", "duplicate"))
def test_application_contracts_reject_unknown_interpretation_before_execution(name, reader, edit):
    """Refuse changed wire generations and ignored executable-looking fields."""
    raw = json.loads((APPLICATION / name).read_text())
    if edit == "future":
        raw["schema_version"] = 2
    elif edit == "missing":
        del raw["contract"]
    elif edit == "boolean_version":
        raw["schema_version"] = True
    elif edit == "unknown":
        raw["arbitrary_factory"] = "unapproved.module"
    body = json.dumps(raw)
    if edit == "duplicate":
        body = body.replace('"schema_version": 1', '"schema_version": 1,"schema_version": 1', 1)
    with pytest.raises(ValueError):
        reader.from_json(body)


def test_completion_preserves_exact_body_and_refuses_every_changed_plan_join():
    """Keep selected result bytes and reject substituted plan interpretations."""
    plan = _plan()
    completion = TaskCompletion.from_json((APPLICATION / "completion.json").read_bytes())
    completion.validate_for(plan)
    codec = CompletionCodec()
    assert codec.decode(codec.encode(completion)) == codec.validate(completion)
    assert completion.result.body == '{"loss":0.0,"status":"observed","unit":"1"}'
    variants = (
        {"plan_fingerprint": "sha256:" + "0" * 64},
        {"source": _changed(plan.source, reference=Reference(id="another/source", version="1"))},
        {"implementation": _changed(completion.implementation, definition_fingerprint="sha256:" + "0" * 64)},
        {"stage": Reference(id="another/stage", version="1")},
        {"coordinates": tuple(reversed(completion.coordinates))},
        {"result": _changed(completion.result, codec=plan.configuration.codec)},
    )
    for changes in variants:
        altered = _changed(completion, **changes)
        with pytest.raises(ValueError, match="complete retained plan"):
            altered.validate_for(plan)
        assert altered.fingerprint != completion.fingerprint
    with pytest.raises(ValueError, match="unique"):
        _changed(completion, coordinates=(completion.coordinates[0],) * 2)


def test_provider_free_open_never_reconstitutes_stored_approval(tmp_path):
    """Retained equality metadata never becomes current execution authority."""
    service = TaskService(tmp_path, bootstrap_profile=_profile())
    metadata_only = TaskService(tmp_path)
    assert service.discover().bootstrap == _profile().binding
    assert metadata_only.discover().bootstrap is None
    assert metadata_only.discover().executors == ()
    with closing(service.store.connect()) as connection:
        assert stored_profile(connection) == _profile()
        assert stored_policy(connection) == service.policy
        with pytest.raises(ValueError, match="explicit current"):
            require_current_profile(connection, None)
    request = TaskPlanningRequest.from_json((APPLICATION / "planning-request.json").read_bytes())
    with pytest.raises(ValueError, match="explicit current"):
        metadata_only.plan(request)
    reconnected, expired = metadata_only.reconnect(), metadata_only.cleanup()
    assert isinstance(reconnected, tuple) and not reconnected
    assert isinstance(expired, tuple) and not expired


def test_first_profile_race_has_one_winner_and_no_partial_policy(tmp_path):
    """Serialize competing real registrations with one complete retained pair."""
    store = WorkflowStore(tmp_path)
    first = _profile()
    second = _changed(first, reference=Reference(id="other/operator-profile", version="1"))
    barrier = Barrier(2)

    def register(profile):
        barrier.wait(timeout=5)
        try:
            return bind_host(store, profile, None)
        except ValueError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(register, (first, second)))
    assert sum(isinstance(result, TaskHostPolicy) for result in results) == 1
    assert sum(isinstance(result, ValueError) for result in results) == 1
    with closing(store.connect()) as connection:
        retained = stored_profile(connection)
        assert retained in (first, second)
        assert stored_policy(connection) is not None
        require_current_profile(connection, retained)
        wrong = second if retained == first else first
        with pytest.raises(ValueError, match="explicit current"):
            require_current_profile(connection, wrong)


def test_profile_observation_requires_actual_matching_generation(tmp_path):
    """Require this profile's actual observation rather than a busy lock or age."""
    store = WorkflowStore(tmp_path)
    profile = _profile()
    bind_host(store, profile, None)
    with store.transaction() as connection:
        assert not profile_observed(connection, profile, 1)
        acknowledge_profile(connection, profile, 2)
        assert profile_observed(connection, profile, 1)
        assert profile_observed(connection, profile, 2)
        assert not profile_observed(connection, profile, 3)
        other = _changed(profile, reference=Reference(id="other/host", version="1"))
        assert not profile_observed(connection, other, 2)
        with pytest.raises(ValueError):
            acknowledge_profile(connection, profile, True)


def test_generic_policy_is_immutable_and_incomplete_settings_refuse(tmp_path):
    """Reject silent capacity changes and incomplete registration metadata."""
    store = WorkflowStore(tmp_path)
    policy = bind_host(store, _profile(), None)
    with store.transaction() as connection:
        validate_task_caps(connection, _plan())
        larger = _changed(_plan(), limits=_changed(_plan().limits, max_invocations=policy.caps.max_invocations))
        validate_task_caps(connection, larger)
    different = _changed(policy, caps=_changed(policy.caps, max_output_bytes=policy.caps.max_output_bytes + 1))
    with pytest.raises(ValueError, match="policy cannot"):
        bind_host(store, _profile(), different)
    with store.transaction() as connection:
        connection.execute("DELETE FROM settings WHERE name='task_policy_v1'")
    with pytest.raises(ValueError, match="Incomplete"):
        bind_host(store, None, None)
    with closing(store.connect()) as connection, pytest.raises(ValueError, match="not registered"):
        validate_task_caps(connection, _plan())


def test_queued_generic_job_requires_current_profile_even_if_sql_contains_it(tmp_path):
    """Refuse scheduling from persisted declarations without current approval."""
    store = WorkflowStore(tmp_path)
    plan = _plan()
    bind_host(store, _profile(), None)
    job = TaskJobRecord.from_json((FIXTURES / "v1" / "job.json").read_bytes())
    job = _changed(job, state="queued", result_digest=None)
    with store.transaction() as connection:
        connection.execute("INSERT INTO plans VALUES (?,?,?)", (plan.fingerprint[7:], job.owner, plan.to_json()))
        store.save_record(connection, job, create=True)
        with pytest.raises(ValueError, match="explicit current"):
            validate_queued_tasks(store, connection, None)
        validate_queued_tasks(store, connection, _profile())
        other = _changed(_profile(), reference=Reference(id="other/host", version="1"))
        with pytest.raises(ValueError, match="explicit current"):
            validate_queued_tasks(store, connection, other)
