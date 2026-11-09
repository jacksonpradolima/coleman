"""Retained generic attempts must belong to the actual job and saved caps."""

import json

import pytest

from coleman.application import execution_views as views
from coleman.artifacts import ArtifactWriter
from coleman.core.primitives import fingerprint
from coleman.tasks.contracts import TaskPlan
from coleman.workflows.contracts import WorkflowError
from tests.tasks.test_storage_records import _plans, _task


def _attempt(tmp_path):
    _, original = _plans()
    plan = TaskPlan.model_validate(
        original.model_copy(update={"limits": original.limits.model_copy(update={"max_invocations": 1})})
    )
    job = _task(plan, job_id="actual-job", invocations=1)
    coordinate = plan.coordinates[0]
    _, body = views.attempt_payload(job, plan, coordinate.fingerprint, 1)
    root = tmp_path / job.job_id
    target = views.attempt_directory(root, coordinate.fingerprint) / "0001.json"
    return plan, job, coordinate, root, target, json.loads(body)


@pytest.mark.parametrize("alteration", ("another_job", "wrong_id", "invocation_above_saved_cap"))
def test_retained_attempt_cannot_borrow_another_job_identity_or_global_invocation_ceiling(tmp_path, alteration):
    """Refuse borrowed attempt identity and a worker start above saved caps."""
    plan, job, coordinate, root, target, raw = _attempt(tmp_path)
    if alteration == "another_job":
        raw["attempt_id"] = fingerprint(["task.physical.v1", "another-job", coordinate.fingerprint, 1])[7:]
    elif alteration == "wrong_id":
        raw["attempt_id"] = "0" * 64
    else:
        raw["worker_invocation"] = 2  # Valid global field; invalid for this saved plan's cap of one.
    ArtifactWriter(root).write_immutable(target, json.dumps(raw))
    with pytest.raises(WorkflowError, match="binding mismatch"):
        views.retained_attempts(root, plan, job_id=job.job_id)


def test_actual_job_bound_reservation_retains_count_and_original_bytes(tmp_path):
    """Keep one correctly joined physical declaration without rewriting it."""
    plan, job, _, root, target, raw = _attempt(tmp_path)
    body = json.dumps(raw)
    ArtifactWriter(root).write_immutable(target, body)
    assert views.retained_attempts(root, plan, job_id=job.job_id) == 1
    assert target.read_text() == body
