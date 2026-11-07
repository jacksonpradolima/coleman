"""Real controlled learner recurrence, interval frontiers and complete recovery."""

from datetime import UTC, datetime, timedelta
from fractions import Fraction

import pytest

from coleman.core.feedback import FullPanel
from coleman.learning.control_contracts import ControlledPayload, MovementPeriod
from coleman.learning.controlled import ControlledHedge
from coleman.objectives.controls import L1MovementPenalty, ReleaseInterval
from coleman.state.contracts import AlgorithmState
from tests.learning._controlled_cases import configuration, panel, ref, values

START = datetime(2026, 1, 1, tzinfo=UTC)


def interval(index, *, cutoff=None):
    start = START + timedelta(days=index)
    return ReleaseInterval(
        interval=ref("ci-interval-" + str(index)),
        start=start,
        end=start + timedelta(days=1),
        cutoff=start if cutoff is None else cutoff,
        release_at=start + timedelta(hours=12),
    )


def period(config, index):
    start = START + timedelta(days=index)
    return MovementPeriod(
        schema_version=1,
        calendar=config.period_clock,
        interval=ref("movement-period-" + str(index)),
        start=start,
        end=start + timedelta(days=1),
    )


def clone(learner):
    state = AlgorithmState[ControlledPayload].from_json(learner.export_state().to_json())
    restored = ControlledHedge(learner.config, owner=learner.binding.owner)
    restored.restore_state(state)
    return restored


def test_penalty_is_once_per_released_sum_block_and_next_step_uses_emitted_origin():
    config = configuration(cadence=2, strength=0.05)
    learner = ControlledHedge(config, owner=ref("block-owner"))
    initial = learner.export_state().to_json()
    held = learner.observe(panel(config, (0.0, 1.0, 1.0)))
    assert held.disposition == "held"
    assert held.output is held.proposal is None
    assert held.summed_losses == ()
    assert learner.update_count == 1 and learner.release_count == 0
    restored = clone(learner)
    loss = panel(config, (0.0, 1.0, 1.0))
    first = learner.observe(loss)
    assert first.to_json() == restored.observe(loss).to_json()
    assert first.disposition == "released"
    assert tuple(item.value for item in first.summed_losses) == (0.0, 2.0, 2.0)
    assert first.origin.weights == config.controls.initial_prior.weights
    assert isinstance(first.output, L1MovementPenalty)
    assert first.emitted_origin is not None
    assert first.output.origin.to_json() == first.origin.to_json()
    assert first.emitted_origin.weights == learner.weights
    assert learner.config.to_json() == config.to_json()
    assert learner.update_count == 2 and learner.release_count == 1
    assert initial != learner.export_state().to_json()
    learner.observe(panel(config, (1.0, 0.0, 0.0)))
    second = learner.observe(panel(config, (1.0, 0.0, 0.0)))
    assert isinstance(second.output, L1MovementPenalty)
    assert second.origin.to_json() == first.emitted_origin.to_json()
    assert second.output.origin.to_json() == first.emitted_origin.to_json()
    assert second.release != first.release
    assert learner.release_count == 2


@pytest.mark.parametrize("strength", [None, 0.0, 0.05, 100.0])
def test_interval_cadence_anchored_bounded_movement_and_penalty_combine(strength):
    calendar = ref("explicit-ci-release-clock")
    config = configuration(
        strength=strength,
        cadence=calendar,
        target=(0.1, (0.3, 0.3, 0.4)),
        lower=(0.1, 0.1, 0.1),
        upper=(0.6, 0.6, 0.6),
        movement=(0.1, 0.1, 0.1),
    )
    learner = ControlledHedge(config, owner=ref("interval-owner"))
    learner.observe(panel(config, (0.0, 1.0, 0.5)), eligible_at=START, apply_at=START)
    one = interval(0)
    result = learner.release(one, calendar=calendar, apply_at=one.release_at)
    assert result.disposition == "released"
    assert learner.release_count == 1
    assert all(0.1 <= item.value <= 0.6 for item in learner.weights)
    assert all(
        abs(Fraction(item.value) - Fraction(old.value)) <= Fraction(0.1)
        for item, old in zip(learner.weights, config.controls.initial_prior.weights, strict=True)
    )
    before = learner.export_state().to_json()
    assert learner.release(one, calendar=calendar, apply_at=one.release_at).to_json() == result.to_json()
    assert learner.export_state().to_json() == before
    restored = clone(learner)
    empty = interval(1)
    empty_result = restored.release(empty, calendar=calendar, apply_at=empty.release_at)
    assert empty_result.disposition == "empty"
    assert empty_result.proposal_status == "not_applicable"
    assert restored.release_count == 1
    assert restored.weights == learner.weights


def test_high_penalty_holds_but_consumes_nonempty_release_opportunity():
    config = configuration(strength=100.0)
    learner = ControlledHedge(config, owner=ref("hold-owner"))
    before = learner.weights
    result = learner.observe(panel(config, (0.0, 1.0, 1.0)))
    assert result.disposition == "released"
    assert learner.weights == before
    assert learner.release_count == learner.update_count == 1
    assert result.emitted_origin is not None
    assert result.emitted_origin.binding != result.origin.binding
    assert result.failure is None


def test_zero_strength_matches_original_projection_by_identical_emitted_bytes():
    penalized_config, plain_config = configuration(strength=0.0), configuration(strength=None)
    penalized = ControlledHedge(penalized_config, owner=ref("l1-zero"))
    plain = ControlledHedge(plain_config, owner=ref("projection"))
    for loss in ((0.0, 1.0, 0.5), (1.0, 0.0, 0.25), (0.5, 0.5, 0.0)):
        assert penalized.observe(panel(penalized_config, loss)).disposition == "released"
        assert plain.observe(panel(plain_config, loss)).disposition == "released"
        assert values(penalized) == values(plain)


def test_exact_empty_intersection_retains_failed_block_and_rejects_fake_acceptance():
    config = configuration(prior=(1 / 3, 1 / 3, 1 / 3), movement=(0.0, 0.0, 0.0), strength=0.1)
    learner = ControlledHedge(config, owner=ref("infeasible-owner"))
    result = learner.observe(panel(config, (0.0, 0.5, 1.0)))
    assert result.disposition == "infeasible"
    assert result.failure is not None
    assert result.failure.code == "infeasible"
    assert learner.update_count == 1 and learner.release_count == 0
    before = learner.export_state().to_json()
    with pytest.raises(ValueError, match="retried"):
        learner.observe(panel(config, (0.0, 0.5, 1.0)))
    assert learner.export_state().to_json() == before
    assert clone(learner).retry_release().to_json() == result.to_json()


@pytest.mark.parametrize("measure", ["net_displacement", "accumulated_absolute"])
def test_period_budget_round_trip_historical_retry_never_resets_usage_or_flushes_partial_block(measure):
    config = configuration(cadence=2, movement=(0.2, 0.2, 0.2), scope="per_interval", measure=measure)
    learner = ControlledHedge(config, owner=ref("period-owner"))
    one, two = period(config, 0), period(config, 1)
    assert learner.begin_period(one, apply_at=START)
    learner.observe(panel(config, (0.0, 1.0, 1.0)), apply_at=START)
    first = learner.observe(panel(config, (0.0, 1.0, 1.0)), apply_at=START)
    assert first.disposition == "released"
    learner.observe(panel(config, (1.0, 0.0, 0.0)), apply_at=START)
    restored = clone(learner)
    before = restored.export_state().to_json()
    assert not restored.begin_period(one, apply_at=START)
    assert restored.export_state().to_json() == before
    assert restored.begin_period(two, apply_at=two.start)
    assert restored.release_count == 1 and restored.update_count == 3
    if measure == "accumulated_absolute":
        assert all(item.atoms_decimal == "0" for item in restored.export_state().payload.exact_path.used)
    final = restored.observe(panel(config, (1.0, 0.0, 0.0)), apply_at=two.start)
    assert final.disposition == "released"
    assert restored.update_count == 4 and restored.release_count == 2
    after = restored.export_state().to_json()
    assert not restored.begin_period(one, apply_at=START)
    assert restored.export_state().to_json() == after


def test_late_panel_does_not_reopen_closed_cutoff_and_future_interval_retains_it():
    calendar = ref("late-clock")
    config = configuration(cadence=calendar)
    learner = ControlledHedge(config, owner=ref("late-owner"))
    first = interval(0)
    empty = learner.release(first, calendar=calendar, apply_at=first.release_at)
    learner.observe(panel(config, (0.0, 1.0, 1.0)), eligible_at=START, apply_at=first.release_at)
    before = learner.export_state().to_json()
    assert learner.release(first, calendar=calendar, apply_at=first.release_at).to_json() == empty.to_json()
    assert learner.export_state().to_json() == before
    second = interval(1)
    assert learner.release(second, calendar=calendar, apply_at=second.release_at).disposition == "released"
    assert learner.release_count == 1 and learner.update_count == 1


def test_unknown_future_version_and_wrong_owner_cannot_change_live_state():
    learner = ControlledHedge(configuration(), owner=ref("strict-owner"))
    state = learner.export_state()
    before = state.to_json()
    with pytest.raises(ValueError):
        AlgorithmState[ControlledPayload].from_json(before.replace('"schema_version":1', '"schema_version":2', 1))
    other = ControlledHedge(learner.config, owner=ref("foreign-owner")).export_state()
    with pytest.raises(ValueError, match="owner"):
        learner.restore_state(other)
    assert learner.export_state().to_json() == before


@pytest.mark.parametrize("target", [None, (1.0, (0.5, 0.5))])
def test_structural_support_mask_is_explicit_and_target_can_restore_positive_mass(target):
    config = configuration(prior=(0.0, 1.0), target=target, strength=0.0)
    learner = ControlledHedge(config, owner=ref("zero-support-owner"))
    assert learner.export_state().payload.support == (False, True)
    result = learner.observe(panel(config, (0.0, 1.0)))
    assert result.disposition == "released"
    expected = (0.0, 1.0) if target is None else (0.5, 0.5)
    assert values(learner) == expected
    assert learner.export_state().payload.support == tuple(value > 0 for value in expected)
    restored = clone(learner)
    assert restored.export_state().to_json() == learner.export_state().to_json()
    loss = panel(config, (1.0, 0.0))
    assert restored.observe(loss).to_json() == learner.observe(loss).to_json()


def test_single_expert_and_positive_penalty_have_finite_exact_hold_state():
    config = configuration(prior=(1.0,), strength=100.0)
    learner = ControlledHedge(config, owner=ref("one-controlled-expert"))
    for value in (0.0, 1.0, 0.25):
        result = learner.observe(panel(config, (value,)))
        assert result.disposition == "released" and values(learner) == (1.0,)
        assert learner.export_state().payload.support == (True,)
    assert learner.release_count == learner.update_count == 3


def test_historical_interval_retry_still_rejects_a_conflicting_application_instant():
    calendar = ref("retry-clock")
    config = configuration(cadence=calendar)
    learner = ControlledHedge(config, owner=ref("interval-retry-owner"))
    first = interval(0)
    learner.release(first, calendar=calendar, apply_at=first.release_at)
    before = learner.export_state().to_json()
    with pytest.raises(ValueError, match="single release point"):
        learner.release(first, calendar=calendar, apply_at=first.release_at + timedelta(seconds=1))
    assert learner.export_state().to_json() == before


@pytest.mark.parametrize("missing_status", ["unknown", "unavailable", "not_applicable"])
def test_zero_is_observed_but_every_explicit_missing_kind_refuses_without_advancing(missing_status):
    config = configuration()
    learner = ControlledHedge(config, owner=ref("explicit-missingness-owner"))
    original = panel(config, (0.0, 0.5, 1.0))
    data = original.model_dump()
    rows = list(data["rows"])
    values = list(rows[0]["values"])
    values[0] = values[0] | {"value": None, "status": missing_status}
    rows[0] = rows[0] | {"values": tuple(values)}
    missing = FullPanel.model_validate(data | {"rows": tuple(rows)})
    before = learner.export_state().to_json()
    with pytest.raises(ValueError, match="observed"):
        learner.observe(missing)
    assert learner.export_state().to_json() == before
    assert learner.observe(original).disposition == "released"
    assert learner.update_count == 1


@pytest.mark.parametrize("clock_error", ["eligibility_after_application", "rewound_application", "unknown_eligibility"])
def test_explicit_processing_admission_rejects_inconsistent_clocks_without_changing_state(clock_error):
    calendar = ref("clock-admission")
    config = configuration(cadence=calendar)
    learner = ControlledHedge(config, owner=ref("clock-owner"))
    learner.observe(panel(config, (0.0, 1.0, 0.5)), eligible_at=START, apply_at=START)
    arguments = {
        "eligibility_after_application": {"eligible_at": START + timedelta(seconds=1), "apply_at": START},
        "rewound_application": {"eligible_at": START - timedelta(seconds=1), "apply_at": START - timedelta(seconds=1)},
        "unknown_eligibility": {"eligible_at": None, "apply_at": START},
    }[clock_error]
    before = learner.export_state().to_json()
    with pytest.raises(ValueError):
        learner.observe(panel(config, (1.0, 0.0, 0.5)), **arguments)
    assert learner.export_state().to_json() == before


@pytest.mark.parametrize("admission", ["unknown_application", "no_period", "outside_period"])
def test_period_limited_panel_requires_a_real_current_period_and_its_processing_instant(admission):
    config = configuration(movement=(0.1, 0.1, 0.1), scope="per_interval")
    learner = ControlledHedge(config, owner=ref("period-admission-owner"))
    current = period(config, 0)
    if admission != "no_period":
        learner.begin_period(current, apply_at=START)
    apply_at = None if admission == "unknown_application" else current.end if admission == "outside_period" else START
    before = learner.export_state().to_json()
    with pytest.raises(ValueError, match="period|application"):
        learner.observe(panel(config, (0.0, 1.0, 0.5)), apply_at=apply_at)
    assert learner.export_state().to_json() == before


def test_partial_round_block_cannot_be_flushed_by_explicit_retry():
    config = configuration(cadence=2)
    learner = ControlledHedge(config, owner=ref("partial-retry-owner"))
    learner.observe(panel(config, (0.0, 1.0, 0.5)))
    before = learner.export_state().to_json()
    with pytest.raises(ValueError, match="complete retained round block"):
        learner.retry_release()
    assert learner.export_state().to_json() == before


@pytest.mark.parametrize("change", ["wrong_calendar", "conflicting_identity", "overlapping_interval"])
def test_interval_release_history_cannot_be_rebound_or_reopened(change):
    calendar = ref("release-admission-clock")
    config = configuration(cadence=calendar)
    learner = ControlledHedge(config, owner=ref("release-admission-owner"))
    first = interval(0)
    learner.release(first, calendar=calendar, apply_at=first.release_at)
    selected_calendar = ref("foreign-calendar") if change == "wrong_calendar" else calendar
    changes = {} if change == "wrong_calendar" else {"end": first.end + timedelta(days=1)}
    if change == "overlapping_interval":
        changes |= {"interval": ref("overlapping-identity")}
    offered = ReleaseInterval.model_validate(first.model_dump() | changes)
    before = learner.export_state().to_json()
    with pytest.raises(ValueError):
        learner.release(offered, calendar=selected_calendar, apply_at=offered.release_at)
    assert learner.export_state().to_json() == before


@pytest.mark.parametrize(
    "change", ["wrong_clock", "outside_period", "conflicting_identity", "rewound_application", "overlapping_period"]
)
def test_movement_period_admission_cannot_reset_or_rewind_consumed_allowances(change):
    config = configuration(movement=(0.1, 0.1, 0.1), scope="per_interval")
    learner = ControlledHedge(config, owner=ref("movement-clock-owner"))
    current = period(config, 0)
    learner.begin_period(current, apply_at=START)
    at = START + timedelta(hours=12)
    learner.observe(panel(config, (0.0, 1.0, 0.5)), apply_at=at)
    values = current.model_dump()
    apply_at = at
    if change == "wrong_clock":
        values |= {"calendar": ref("foreign-movement-clock")}
    elif change == "outside_period":
        apply_at = current.end
    elif change == "conflicting_identity":
        values |= {"end": current.end + timedelta(days=1)}
    else:
        values |= {
            "interval": ref("new-overlapping-period"),
            "start": START + timedelta(hours=1),
            "end": current.end + timedelta(days=1),
        }
        apply_at = START + timedelta(hours=1) if change == "rewound_application" else current.end
    offered = MovementPeriod.model_validate(values)
    before = learner.export_state().to_json()
    with pytest.raises(ValueError):
        learner.begin_period(offered, apply_at=apply_at)
    assert learner.export_state().to_json() == before
