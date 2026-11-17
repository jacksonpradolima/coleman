"""Complete immutable controlled views leave audited export and ingress intact."""

from datetime import UTC, datetime, timedelta

import pytest

from coleman.learning.control_contracts import ControlledPayload, MovementPeriod
from coleman.learning.controlled import ControlledHedge
from coleman.objectives.controls import ReleaseInterval
from coleman.state.contracts import AlgorithmState
from tests.learning._controlled_cases import configuration, panel, ref

START = datetime(2026, 1, 1, tzinfo=UTC)


def _assert_audited(learner):
    view = learner.view_state()
    assert type(view) is AlgorithmState[ControlledPayload]
    assert view.to_json() == learner.export_state().to_json()
    return view


def _interval(index):
    start = START + timedelta(days=index)
    return ReleaseInterval(
        interval=ref("view-release-" + str(index)),
        start=start,
        end=start + timedelta(days=1),
        cutoff=start,
        release_at=start + timedelta(hours=12),
    )


def test_initial_held_and_released_views_are_complete_and_keep_earlier_values():
    config = configuration(cadence=2)
    learner = ControlledHedge(config, owner=ref("immutable-view-owner"))
    initial = _assert_audited(learner)
    initial_wire = initial.to_json()
    assert initial.payload.update_count == 0
    assert learner.observe(panel(config, (0.0, 1.0, 1.0))).disposition == "held"
    held = _assert_audited(learner)
    held_wire = held.to_json()
    assert len(held.payload.pending) == held.payload.update_count == 1
    assert initial.to_json() == initial_wire
    assert learner.observe(panel(config, (0.0, 1.0, 1.0))).disposition == "released"
    released = _assert_audited(learner)
    assert released.payload.update_count == released.payload.consumed_panels == 2
    assert released.payload.release_count == 1 and not released.payload.pending
    assert held.to_json() == held_wire and initial.to_json() == initial_wire


def test_infeasible_view_retains_original_failed_block_and_retry_lineage():
    config = configuration(prior=(1 / 3,) * 3, movement=(0.0,) * 3, strength=0.1)
    learner = ControlledHedge(config, owner=ref("failed-view-owner"))
    assert learner.observe(panel(config, (0.0, 0.5, 1.0))).disposition == "infeasible"
    failed = _assert_audited(learner)
    original_wire = failed.to_json()
    assert failed.payload.unresolved_failure is not None
    assert len(failed.payload.pending) == 1 and failed.payload.release_count == 0
    learner.retry_release()
    _assert_audited(learner)
    assert failed.to_json() == original_wire


def test_empty_and_closed_interval_retry_views_preserve_every_original_value():
    calendar = ref("view-release-calendar")
    config = configuration(cadence=calendar)
    learner = ControlledHedge(config, owner=ref("interval-view-owner"))
    empty_interval = _interval(0)
    assert learner.release(empty_interval, calendar=calendar, apply_at=empty_interval.release_at).disposition == "empty"
    empty = _assert_audited(learner)
    empty_wire = empty.to_json()
    at = START + timedelta(days=1)
    learner.observe(panel(config, (0.0, 1.0, 1.0)), eligible_at=at, apply_at=at)
    release = _interval(1)
    outcome = learner.release(release, calendar=calendar, apply_at=release.release_at)
    assert outcome.disposition == "released"
    completed = _assert_audited(learner)
    complete_wire = completed.to_json()
    assert len(completed.payload.intervals) == 2
    assert learner.release(release, calendar=calendar, apply_at=release.release_at).to_json() == outcome.to_json()
    assert _assert_audited(learner).to_json() == complete_wire
    assert empty.to_json() == empty_wire


def test_numerical_failure_view_survives_held_feedback_and_later_successful_recovery():
    calendar = ref("view-numerical-calendar")
    config = configuration(cadence=calendar, eta=1e308)
    learner = ControlledHedge(config, owner=ref("numerical-view-owner"))
    learner.observe(panel(config, (0.0, 1.0, 1.0)), eligible_at=START, apply_at=START)
    release = _interval(0)
    outcome = learner.release(release, calendar=calendar, apply_at=release.release_at)
    assert outcome.disposition == "numerical_failure" and outcome.failure.code == "positive_underflow"
    failed = _assert_audited(learner)
    failed_wire = failed.to_json()
    at = START + timedelta(days=1)
    assert learner.observe(panel(config, (1.0, 0.0, 0.0)), eligible_at=at, apply_at=at).disposition == "held"
    held = _assert_audited(learner)
    assert held.payload.unresolved_failure == outcome and len(held.payload.pending) == 2
    next_release = _interval(1)
    assert learner.release(next_release, calendar=calendar, apply_at=next_release.release_at).disposition == "released"
    recovered = _assert_audited(learner)
    assert recovered.payload.unresolved_failure is None and not recovered.payload.pending
    assert failed.to_json() == failed_wire


@pytest.mark.parametrize("measure", ["net_displacement", "accumulated_absolute"])
def test_period_and_restored_views_keep_usage_and_the_complete_original_frontier(measure):
    config = configuration(cadence=2, movement=(0.2,) * 3, scope="per_interval", measure=measure)
    learner = ControlledHedge(config, owner=ref("period-view-owner"))
    period = MovementPeriod(
        schema_version=1,
        calendar=config.period_clock,
        interval=ref("first-view-period"),
        start=START,
        end=START + timedelta(days=1),
    )
    learner.begin_period(period, apply_at=START)
    learner.observe(panel(config, (0.0, 1.0, 1.0)), apply_at=START)
    before = _assert_audited(learner)
    before_wire = before.to_json()
    restored = ControlledHedge(config, owner=learner.binding.owner)
    restored.restore_state(AlgorithmState[ControlledPayload].from_json(before_wire))
    assert _assert_audited(restored).to_json() == before_wire
    loss = panel(config, (0.0, 1.0, 1.0))
    assert learner.observe(loss, apply_at=START).to_json() == restored.observe(loss, apply_at=START).to_json()
    assert _assert_audited(restored).to_json() == _assert_audited(learner).to_json()
    assert before.to_json() == before_wire


def test_shared_view_and_nested_values_reject_ordinary_mutation():
    config = configuration(cadence=2)
    learner = ControlledHedge(config, owner=ref("frozen-view-owner"))
    learner.observe(panel(config, (0.0, 1.0, 1.0)))
    view = _assert_audited(learner)
    original = view.to_json()
    with pytest.raises(ValueError):
        view.schema_version = 2
    with pytest.raises(ValueError):
        view.payload.update_count = 99
    with pytest.raises(ValueError):
        view.payload.origin.weights[0].value = 0.9
    with pytest.raises(ValueError):
        view.payload.pending[0].panel.rows[0].values[0].value = 0.75
    assert _assert_audited(learner).to_json() == original


@pytest.mark.parametrize("change", ["future_state", "future_payload", "bad_counter", "wrong_owner"])
def test_view_copies_and_constructed_inputs_still_get_complete_atomic_restore_validation(change):
    config = configuration(cadence=2)
    learner = ControlledHedge(config, owner=ref("strict-view-owner"))
    learner.observe(panel(config, (0.0, 1.0, 1.0)))
    view = _assert_audited(learner)
    original = view.to_json()
    if change == "future_state":
        candidate = view.model_copy(update={"schema_version": 2})
    elif change == "future_payload":
        candidate = view.model_copy(update={"payload": view.payload.model_copy(update={"schema_version": 2})})
    elif change == "bad_counter":
        payload = view.payload.model_copy(update={"update_count": 99})
        candidate = AlgorithmState[ControlledPayload].model_construct(
            schema_version=1,
            binding=view.binding,
            payload=payload,
        )
    else:
        candidate = ControlledHedge(config, owner=ref("other-view-owner")).view_state()
    with pytest.raises(ValueError):
        learner.restore_state(candidate)
    assert view.to_json() == original
    assert _assert_audited(learner).to_json() == original
