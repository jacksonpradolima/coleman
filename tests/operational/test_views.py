"""Actual delivery and checkpoint observations never alter causal host state."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from coleman.replay.engine import CausalReplay
from coleman.replay.retained_contracts import RetainedReplayState
from coleman.tasks.contracts import TaskJobRecord, TaskPlan
from coleman.telemetry.views import action_snapshot, replay_snapshot, task_snapshot
from tests.replay.support import emit, session


def points(snapshot):
    return {p.name: p for p in snapshot.points}


def test_pending_age_missing_feedback_delivery_dedup_and_exact_checkpoint_continuation(replay_example):
    ex = replay_example
    replay, host = session(ex)
    empty = points(replay_snapshot(replay.state, at=ex.instant(0)))
    assert empty["pending"].value == 0
    assert empty["update_age"].status == "not_applicable"
    first = emit(ex, replay, 0, at=0, end=1)
    second = emit(ex, replay, 1, at=1, end=2)
    before = replay.export_continuation().to_json()
    pending = points(replay_snapshot(replay.state, at=ex.instant(2)))
    assert (pending["decisions"].value, pending["updates"].value) == (2, 0)
    assert pending["pending_age"].value == 120
    assert pending["missing_feedback"].value == 2
    outcome = ex.feedback(first, (1.0, 0.0, 0.5), visible=2, delay=1)
    replay.admit_feedback((outcome, outcome))
    assert points(replay_snapshot(replay.state, at=ex.instant(2)))["missing_feedback"].value == 1
    replay.advance(eligible_through=ex.instant(2), apply_at=ex.instant(3))
    after = replay.export_continuation()
    observed = replay_snapshot(replay.state, at=ex.instant(4))
    assert points(observed)["pending_age"].value == 180
    assert points(observed)["update_age"].value == 60
    assert host.export_state().update_count == 1
    fresh = ex.HostConsumer()
    restored = CausalReplay.from_continuation(fresh, after, state_type=ex.HostConsumerState, state_codec=ex.CODEC)
    assert replay_snapshot(restored.state, at=ex.instant(4)).to_json() == observed.to_json()
    replay.admit_feedback((outcome,))
    replay.advance(eligible_through=ex.instant(2), apply_at=ex.instant(3))
    assert host.export_state().update_count == 1
    assert before != replay.export_continuation().to_json()
    assert second.executed.fingerprint != first.executed.fingerprint


def test_observing_readonly_source_and_explicit_clock_does_not_apply_a_pending_label(replay_example):
    ex = replay_example
    replay, host = session(ex)
    first = emit(ex, replay, 0, at=0, end=1)
    replay.admit_feedback((ex.feedback(first, (1, 0, 0), visible=2, delay=0),))
    before = replay.export_continuation().to_json()
    for at in (2, 3, 4):
        assert points(replay_snapshot(replay.state, at=ex.instant(at)))["updates"].value == 0
    assert replay.export_continuation().to_json() == before
    assert host.export_state().update_count == 0
    with pytest.raises(ValueError):
        replay_snapshot(replay.state, at=ex.instant(-1))
    with pytest.raises(ValueError):
        replay_snapshot(replay.state, at=datetime(2026, 1, 1))
    action = action_snapshot(first.executed, at=ex.instant(2), outcome="accepted")
    assert points(action)["concentration"].status == "not_applicable"
    assert first.executed.fingerprint == action.source_fingerprint


def test_old_task_goldens_remain_readable_and_false_completion_or_plan_join_rejects():
    directory = Path(__file__).parents[1] / "fixtures/tasks/v1"
    plan = TaskPlan.from_json((directory / "plan.json").read_bytes())
    job = TaskJobRecord.from_json((directory / "job.json").read_bytes())
    observation = task_snapshot(job, plan, at=datetime(2026, 1, 1, tzinfo=UTC))
    assert observation.outcome == "pending"
    assert points(observation)["task_progress"].value == 0
    assert TaskJobRecord.from_json(job.to_json()).to_json() == job.to_json()
    with pytest.raises(ValueError):
        task_snapshot(job.model_copy(update={"state": "completed"}), plan, at=datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError):
        task_snapshot(job.model_copy(update={"plan_id": "0" * 64}), plan, at=datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError):
        task_snapshot(job, plan, at=datetime(1970, 1, 1, tzinfo=UTC))


@pytest.mark.parametrize(
    "filename,decisions,updates,pending",
    (
        ("retained-sealed.json", 2, 2, 0),
        ("retained-one-pending.json", 1, 0, 1),
    ),
)
def test_existing_retained_v2_preserves_lifetime_counts_and_lost_update_clock(filename, decisions, updates, pending):
    path = Path(__file__).parents[1] / "fixtures/state/v2" / filename
    state = RetainedReplayState.model_validate(json.loads(path.read_text())["state"])
    before = state.to_json()
    snapshot = replay_snapshot(state, at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=2))
    actual = points(snapshot)
    assert (actual["decisions"].value, actual["updates"].value, actual["pending"].value) == (
        decisions,
        updates,
        pending,
    )
    assert actual["update_age"].status == ("unavailable" if updates else "not_applicable")
    assert actual["update_age"].value is None
    assert (
        replay_snapshot(RetainedReplayState.from_json(before), at=snapshot.observed_at).to_json() == snapshot.to_json()
    )
    assert state.to_json() == before
    with pytest.raises(ValueError):
        replay_snapshot(state.model_copy(update={"schema_version": 3}), at=snapshot.observed_at)
