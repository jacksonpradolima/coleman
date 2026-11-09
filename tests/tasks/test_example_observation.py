"""Bounded public observer regressions from actual installed contention."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from coleman.tasks.contracts import TaskJobRecord
from coleman.tasks.requests import TaskExecuteRequest, TaskJobView


@pytest.fixture(name="example")
def _example():
    """Load only the public host observer; no provider, store or job is created."""
    path = Path(__file__).parents[2] / "examples" / "typed_executor" / "ci.py"
    spec = importlib.util.spec_from_file_location("typed_ci_observer_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _view(state="queued"):
    root = Path(__file__).parents[1] / "fixtures" / "tasks" / "v1"
    original = TaskJobRecord.from_json((root / "job.json").read_bytes())
    changes: dict[str, object] = {"state": state, "result_digest": None}
    if state == "completed":
        changes.update(result_digest="sha256:" + "1" * 64, completed_coordinates=original.total_coordinates)
    job = TaskJobRecord.model_validate(original.model_copy(update=changes))
    return TaskJobView(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="job_view",
        job=job,
        worker_active=False,
        elapsed_seconds=0.0,
    )


def _request():
    view = _view()
    return TaskExecuteRequest(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="execute_request",
        request_id=view.job.request_id,
        plan_id=view.job.plan_id,
    )


def _busy(code):
    error = sqlite3.OperationalError("Localized text is deliberately not used for classification")
    error.sqlite_errorcode = code
    return error


@pytest.mark.parametrize("code", (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, (1 << 8) | sqlite3.SQLITE_BUSY))
def test_numeric_contention_retries_exact_admission_key_and_known_job(example, monkeypatch, code):
    """Keep the same request and job when numeric contention loses responses."""
    calls = []
    request = _request()
    admission = iter((_busy(code), _view()))
    observations = iter((_busy(code), _view("completed")))

    def execute(value):
        assert value == request
        calls.append(("execute", value.request_id))
        outcome = next(admission)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def status(identity):
        assert identity == _view().job.job_id
        calls.append(("status", identity))
        outcome = next(observations)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(example.time, "sleep", lambda _seconds: None)
    result = example.wait_for_completion(cast(Any, SimpleNamespace(execute=execute, status=status)), request)
    assert result.job.state == "completed"
    assert [operation for operation, _ in calls] == ["execute", "execute", "status", "status"]


@pytest.mark.parametrize("code", (None, sqlite3.SQLITE_ERROR, "database is locked"))
def test_other_sqlite_failures_propagate_without_retries(example, code):
    """Refuse text-based retry guesses and preserve real schema/storage errors."""
    calls = []
    error = _busy(code)

    def execute(_request):
        calls.append("execute")
        raise error

    service = cast(Any, SimpleNamespace(execute=execute))
    request = _request()
    with pytest.raises(sqlite3.OperationalError) as caught:
        example.wait_for_completion(service, request)
    assert caught.value is error
    assert calls == ["execute"]


def test_contention_exhausts_original_observation_window_without_budget_growth(example, monkeypatch):
    """Stop at the original allowance and never mutate a storage timeout."""
    now = [0.0]
    calls = []
    monkeypatch.setattr(example.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(example.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))

    def execute(request):
        calls.append(request.request_id)
        raise _busy(sqlite3.SQLITE_BUSY)

    request = _request()
    service = cast(Any, SimpleNamespace(execute=execute))
    with pytest.raises(TimeoutError, match="same retained request"):
        example.wait_for_completion(service, request, observation_seconds=0.05)
    assert now[0] == 0.05
    assert calls == [request.request_id, request.request_id]


@pytest.mark.parametrize("seconds", (0.0, -1.0, 120.001, float("inf"), float("nan"), True))
def test_observation_capacity_rejects_invalid_values_before_work(example, seconds):
    """Keep positive finite seconds distinct from bool, zero and nonfinite input."""
    service = cast(Any, None)
    request = _request()
    with pytest.raises(ValueError, match="allowance"):
        example.wait_for_completion(service, request, observation_seconds=seconds)
