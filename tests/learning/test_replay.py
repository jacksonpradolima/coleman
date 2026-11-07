"""Actual existing causal replay drives shipped reference/controlled learners."""

from datetime import timedelta

import pytest

from coleman.learning.controlled import ControlledHedge
from coleman.learning.direct import DirectMixture, DirectOptions, LearningApplicationError
from coleman.learning.reference import AnytimeHedge, FixedShare, Hedge
from coleman.learning.replay import ReplayMixture
from coleman.learning.schedule import ControlledDirect, ControlSchedule
from coleman.objectives.controls import ReleaseInterval
from coleman.replay.codec import read_continuation
from coleman.replay.contracts import ReplayPolicy
from coleman.replay.engine import CausalReplay, ReplayInvalidatedError


def application(example, *, strength=0.05):
    calendar = example.reference("authored-ci-interval-clock")
    config = example.controlled_configuration(cadence=calendar, strength=strength)
    point = ReleaseInterval(
        interval=example.reference("ci-scheduled-release"),
        start=example.NOW,
        end=example.NOW + timedelta(minutes=5),
        cutoff=example.NOW + timedelta(minutes=3),
        release_at=example.NOW + timedelta(minutes=4),
    )
    schedule = ControlSchedule(schema_version=1, calendar=calendar, intervals=(point,))
    method = ControlledHedge(config, owner=example.reference("actual-controlled-ci-owner"))
    return ControlledDirect(
        method, schedule, DirectOptions(task=example.TASK, objective=example.OBJECTIVE, horizon=example.HORIZON)
    )


def session(example, direct):
    host = ReplayMixture(direct)
    policy = ReplayPolicy(
        schema_version=1,
        stream=example.TASK,
        lineage=example.reference("actual-ci-mixture-live"),
        parent=None,
        source=example.SOURCE,
        capability=host.capability,
        delivery="delayed",
        specialists=False,
    )
    replay = CausalReplay(host, policy, state_type=direct.state_type, state_codec=direct.binding.state_codec)
    replay.admit_history(example.history())
    return replay


def emit(example, replay, direct, index, at, cutoff=None):
    return replay.emit(replay.prepare(example.opportunity(direct, index, at, cutoff)), execution="offline_identity_v1")


def test_public_ci_example_runs_real_penalty_replay_and_separate_clocks(learning_example):
    actual = learning_example.run()
    assert (actual["decisions_D"], actual["accepted_panels_U"], actual["successful_releases_R"]) == (2, 1, 1)
    assert actual["disposition"] == "released"
    assert actual["initial_coefficients"] == [0.5, 0.25, 0.25]
    assert actual["coefficients"] == pytest.approx([0.5102288952285365, 0.2397711047714635, 0.25], abs=2e-16)
    assert actual["feedback_received_at"] < actual["actual_applied_at"] < actual["release_at"]
    assert actual["paid_cost"] == actual["executed_book_turnover"] == "not_measured"
    assert learning_example.run() == actual


@pytest.mark.parametrize("strength", [None, 0.05], ids=["projection", "positive-l1"])
def test_complete_pending_continuation_resumes_after_drain_before_release_identically(learning_example, strength):
    example = learning_example
    direct = application(example, strength=strength)
    replay = session(example, direct)
    first = emit(example, replay, direct, 0, 0)
    delayed = example.feedback(first, direct.learner.config, (0.0, 1.0, 0.5), visible=2)
    replay.admit_feedback((delayed, delayed))
    replay.advance(eligible_through=example.NOW + timedelta(minutes=2), apply_at=example.NOW + timedelta(minutes=3))
    assert direct.learner.update_count == 1 and direct.learner.release_count == 0
    pending = replay.export_continuation()
    decoded = read_continuation(pending.to_json(), state_type=direct.state_type)
    fresh = application(example, strength=strength)
    resumed = CausalReplay.from_continuation(
        ReplayMixture(fresh), decoded, state_type=fresh.state_type, state_codec=fresh.binding.state_codec
    )
    original_next = emit(example, replay, direct, 1, 4, 3)
    resumed_next = emit(example, resumed, fresh, 1, 4, 3)
    assert resumed_next.to_json() == original_next.to_json()
    assert direct.export_state().to_json() == fresh.export_state().to_json()
    assert replay.export_continuation().to_json() == resumed.export_continuation().to_json()
    assert direct.learner.release_count == 1
    before = resumed.export_continuation().to_json()
    resumed.admit_feedback((delayed, delayed))
    assert resumed.export_continuation().to_json() == before
    prepared = resumed.prepare(original_next.opportunity)
    assert resumed.emit(prepared, execution="offline_identity_v1").to_json() == original_next.to_json()
    assert resumed.export_continuation().to_json() == before


@pytest.mark.parametrize(
    "profile,constructor",
    [
        ("hedge-constant", Hedge),
        ("hedge-anytime-cumulative", AnytimeHedge),
        ("fixed-share-excluding-origin", FixedShare),
    ],
)
def test_every_reference_consumes_actual_delayed_replay_application_with_original_producer(
    learning_example, profile, constructor
):
    example = learning_example
    config = example.reference_configuration(profile)
    direct = DirectMixture(
        constructor(config, owner=example.reference("replay-reference-owner")),
        task=example.TASK,
        objective=example.OBJECTIVE,
        horizon=example.HORIZON,
    )
    replay = session(example, direct)
    first = emit(example, replay, direct, 0, 0)
    emit(example, replay, direct, 1, 1)
    delayed = example.feedback(first, config, (0.0, 1.0, 0.5), visible=2, delay=1)
    replay.admit_feedback((delayed,))
    replay.advance(eligible_through=example.NOW + timedelta(minutes=2), apply_at=example.NOW + timedelta(minutes=3))
    assert direct.learner.update_count == 1
    state = direct.export_state()
    assert state.decision_count == 2 and state.update_count == 1
    assert len(state.pending) == 1
    accepted = replay.state.updates[0].application
    assert accepted.executed.to_json() == first.executed.to_json()
    assert accepted.eligible_at != accepted.apply_at
    assert accepted.update_count_before == 0 and accepted.decision_count_before == 2
    assert direct.learner.weights != config.prior.weights


def test_release_outside_captured_host_boundary_invalidates_existing_replay(learning_example):
    example = learning_example
    direct = application(example)
    replay = session(example, direct)
    first = emit(example, replay, direct, 0, 0)
    replay.admit_feedback((example.feedback(first, direct.learner.config, (0.0, 1.0, 0.5), visible=2),))
    replay.advance(eligible_through=example.NOW + timedelta(minutes=2), apply_at=example.NOW + timedelta(minutes=3))
    schedule = direct.export_state().schedule
    direct.learner.release(schedule.intervals[0], calendar=schedule.calendar, apply_at=schedule.intervals[0].release_at)
    with pytest.raises(LearningApplicationError, match="outside"):
        replay.export_continuation()
    with pytest.raises(ReplayInvalidatedError):
        replay.export_continuation()


def test_closed_drain_can_apply_after_information_cutoff_before_the_same_release(learning_example):
    example = learning_example
    direct = application(example)
    replay = session(example, direct)
    first = emit(example, replay, direct, 0, 0)
    replay.admit_feedback((example.feedback(first, direct.learner.config, (0.0, 1.0, 0.5), visible=2),))
    # No advance call: the real emit drain applies eligible<=3 feedback at4,
    # then the captured decision hook releases the complete block at4.
    emit(example, replay, direct, 1, 4, 3)
    accepted = replay.state.updates[0].application
    assert accepted.eligible_at == example.NOW + timedelta(minutes=2)
    assert accepted.apply_at == example.NOW + timedelta(minutes=4)
    assert accepted.apply_at > replay.state.decisions[-1].opportunity.request.timing.information_cutoff
    assert direct.learner.update_count == direct.learner.release_count == 1
    result = direct.learner.last_outcome
    assert result is not None and result.disposition == "released"
    assert direct.export_state().direct.method.payload.pending == ()
