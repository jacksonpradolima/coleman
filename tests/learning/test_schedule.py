"""Real scheduled producer adapter rejects incompatible clocks and continuations."""

from datetime import timedelta

import pytest

from coleman.core.actions import AbstainAction, MixtureSpace
from coleman.core.decisions import DecisionRequest, ExecutedDecision
from coleman.core.feedback import FeedbackBatch
from coleman.learning.controlled import ControlledHedge
from coleman.learning.direct import DirectOptions, LearningApplicationError
from coleman.learning.schedule import ControlledDirect, ControlledDirectState, ControlSchedule
from coleman.objectives.controls import ReleaseInterval
from tests.learning._controlled_cases import configuration, panel, ref
from tests.learning.test_controlled import START, period


def scheduled(config=None, *, schedule=None, options=None):
    config = configuration() if config is None else config
    schedule = ControlSchedule(schema_version=1, calendar=None) if schedule is None else schedule
    options = (
        DirectOptions(task=ref("scheduled-ci-task"), objective=ref("ci-loss-objective"), horizon=None)
        if options is None
        else options
    )
    return ControlledDirect(ControlledHedge(config, owner=ref("scheduled-owner")), schedule, options)


def request(application, index):
    return DecisionRequest(
        schema_version=1,
        task=ref("scheduled-ci-task"),
        objective=ref("ci-loss-objective"),
        decision_id="build-" + str(index),
        round=index,
        space=MixtureSpace(
            feasible_set=ref("full-ci-mixture"), members=application.learner.members, profile="binary64-simplex-v1"
        ),
    )


def outcome(application, proposal):
    executed = ExecutedDecision(
        schema_version=1,
        proposal=proposal,
        action=proposal.action,
        mapping="unchanged",
        sampling_disposition="preserved",
        sampling=proposal.sampling,
    )
    return FeedbackBatch(
        schema_version=1,
        decision_fingerprint=executed.fingerprint,
        feedback_id="observed-ci-loss",
        revision_id="initial",
        supersedes=None,
        horizon=None,
        feedback=panel(application.learner.config, (0.0, 1.0, 0.5)),
    )


def test_controlled_direct_round_mode_restarts_original_pending_and_emits_actual_negative_abstention():
    config = configuration(prior=(1 / 3, 1 / 3, 1 / 3), movement=(0.0, 0.0, 0.0))
    application = scheduled(config)
    first = application.decide(request(application, 0))
    snapshot = application.export_state()
    restored = scheduled(config)
    restored.restore_state(ControlledDirectState.from_json(snapshot.to_json()))
    batch = outcome(application, first)
    result = application.update(batch)
    assert result.disposition == "infeasible"
    assert restored.update(batch).to_json() == result.to_json()
    second = application.decide(request(application, 1))
    assert isinstance(second.action, AbstainAction)
    assert second.action.reason == "controlled-output-infeasible"
    assert application.learner.release_count == 0 and application.learner.update_count == 1
    assert restored.decide(request(restored, 1)).to_json() == second.to_json()
    assert restored.export_state().to_json() == application.export_state().to_json()


def test_schedule_complete_outer_bytes_are_admitted_before_any_successful_decision():
    first = scheduled()
    state = first.export_state()
    total = len(state.to_json().encode())
    inner = len(state.direct.to_json().encode())
    assert total - inner > 64
    limit = DirectOptions(
        task=ref("scheduled-ci-task"),
        objective=ref("ci-loss-objective"),
        horizon=None,
        maximum_bytes=(total + inner) // 2,
    )
    with pytest.raises(ValueError, match="complete state"):
        scheduled(options=limit)


def test_schedule_reader_rejects_unknown_version_and_compatible_new_plan_cannot_rebind_existing_owner():
    first = scheduled()
    first.decide(request(first, 0))
    state = first.export_state()
    before = state.to_json()
    with pytest.raises(ValueError):
        ControlledDirectState.model_validate_json(before.replace('"schema_version":1', '"schema_version":2', 1))
    other_config = configuration(strength=0.1)
    foreign = scheduled(other_config).export_state()
    with pytest.raises(ValueError, match="binding"):
        first.restore_state(foreign)
    assert first.export_state().to_json() == before


def test_scheduled_period_is_explicit_required_and_does_not_exist_for_per_release_controls():
    config = configuration(movement=(0.1, 0.1, 0.1), scope="per_interval")
    with pytest.raises(ValueError, match="scheduled movement periods"):
        scheduled(config)
    plan = ControlSchedule(schema_version=1, calendar=None, periods=(period(config, 0),))
    with pytest.raises(ValueError, match="scheduled movement periods"):
        scheduled(configuration(), schedule=plan)
    actual = scheduled(config, schedule=plan)
    before = actual.export_state().to_json()
    with pytest.raises(ValueError, match="explicit application time"):
        actual.decide(request(actual, 0))
    with pytest.raises(LearningApplicationError, match="invalidated"):
        actual.export_state()
    fresh = scheduled(config, schedule=plan)
    fresh.restore_state(ControlledDirectState.from_json(before))
    assert fresh.export_state().to_json() == before


def test_scheduled_interval_clock_must_match_the_fixed_law_and_cannot_overlap():
    calendar = ref("explicit-ci-clock")
    config = configuration(cadence=calendar)
    point = ReleaseInterval(
        interval=ref("point-one"),
        start=START,
        end=START + timedelta(hours=2),
        cutoff=START,
        release_at=START + timedelta(hours=1),
    )
    wrong = ControlSchedule(schema_version=1, calendar=ref("another-clock"), intervals=(point,))
    with pytest.raises(ValueError, match="fixed cadence calendar"):
        scheduled(config, schedule=wrong)
    duplicate = (point, point)
    with pytest.raises(ValueError, match="unique"):
        ControlSchedule(schema_version=1, calendar=calendar, intervals=duplicate)
    overlapping = ReleaseInterval.model_validate(point.model_dump() | {"interval": ref("point-two")})
    with pytest.raises(ValueError, match="overlap"):
        ControlSchedule(schema_version=1, calendar=calendar, intervals=(point, overlapping))
