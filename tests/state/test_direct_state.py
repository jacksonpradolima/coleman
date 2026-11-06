"""A real bounded consumer preserves every owned state field without a store."""

import json

import pytest

from coleman.core.actions import ChoiceAction
from tests.state.support import complete, emit


@pytest.mark.parametrize("decisions", [2, 3, 4])
def test_partial_reservoir_and_complete_pending_restore_reproduce_literal_next_choices(state_example, decisions):
    example = state_example
    host = example.HostConsumer(0)
    choices = []
    for ordinal in range(decisions):
        proposal = host.decide(example.opportunity(0, ordinal).request)
        assert isinstance(proposal.action, ChoiceAction)
        choices.append(proposal.action.chosen.id)
    assert choices[:3] == ["ci-plan-a", "ci-plan-a", "ci-plan-b"][:decisions]
    state = host.export_state()
    decoded = example.HostState.from_json(state.to_json())
    resumed = example.HostConsumer(0)
    resumed.restore_state(decoded)
    actual = []
    for ordinal in range(decisions, 8):
        request = example.opportunity(0, ordinal).request
        before, after = host.decide(request), resumed.decide(request)
        assert before.to_json() == after.to_json()
        assert isinstance(after.action, ChoiceAction)
        actual.append(after.action.chosen.id)
    expected = ["ci-plan-a", "ci-plan-a", "ci-plan-b", "ci-plan-b", "ci-plan-b", "ci-plan-a", "ci-plan-a", "ci-plan-b"]
    assert actual == expected[decisions:]
    assert resumed.export_state().to_json() == host.export_state().to_json()
    assert len(resumed.export_state().pending) == 8
    assert len(resumed.export_state().book.recent) == 8


def test_real_chosen_update_changes_only_observed_stats_and_eager_model_cache(state_example):
    example = state_example
    replay, host = example.make_replay(0)
    record = emit(example, replay, 0, 0)
    before = host.export_state()
    complete(example, replay, record, 0.0, at=1)
    after = host.export_state()
    assert after.observed_counts == (1, 0)
    assert after.loss_sums == (0.0, 0.0)
    assert after.update_count == 1
    assert after.pending == ()
    assert after.rng.to_json() == before.rng.to_json()
    assert after.book == before.book
    assert after.reservoir_word == before.reservoir_word
    assert after.cache is not None
    assert after.cache.value == 3.0
    assert after.prediction_misses == 2
    next_record = emit(example, replay, 0, 1)
    assert next_record.executed.action.kind == "choice"
    assert host.export_state().prediction_hits == 1


def test_cache_hits_and_cold_misses_are_owned_state_not_recomputed_from_seed(state_example):
    example = state_example
    source = example.HostConsumer(5)
    at = example.instant(0)
    assert source.prewarm(at) == 12.0
    snapshot = source.export_state()
    target = example.HostConsumer(5)
    target.restore_state(example.HostState.from_json(snapshot.to_json()))
    assert target.prewarm(at) == 12.0
    assert target.export_state().prediction_hits == 1
    assert target.export_state().prediction_misses == 1
    assert target.prewarm(example.instant(1)) == 12.0
    assert target.export_state().prediction_misses == 2
    assert target.export_state().rng == snapshot.rng


def test_path_charge_cannot_replace_last_plan_or_actual_path_state(state_example):
    example = state_example
    first, second = example.PathBook(), example.PathBook()
    for plan in (0, 0, 1, 0):
        first = first.release(example.PLANS[plan])
    for plan in (0, 0, 1, 1):
        second = second.release(example.PLANS[plan])
    assert first.spent_seconds == 18
    assert second.spent_seconds == 18
    assert first.last_plan != second.last_plan
    assert first.recent != second.recent
    assert first.release(example.PLANS[0]).spent_seconds == 21
    assert second.release(example.PLANS[0]).spent_seconds == 23


def test_unknown_codec_and_different_child_refuse_before_replacing_an_actual_owner(state_example):
    example = state_example
    owner = example.HostConsumer(0)
    other = example.HostConsumer(1).export_state()
    before = owner.export_state().to_json()
    with pytest.raises(ValueError, match="child configuration"):
        owner.restore_state(other)
    payload = json.loads(before)
    payload["schema_version"] = 2
    encoded = json.dumps(payload)
    with pytest.raises(ValueError):
        example.HostState.from_json(encoded)
    assert owner.export_state().to_json() == before


@pytest.mark.parametrize(
    "changed",
    [
        {"decision_count": 1},
        {"update_count": 1},
        {"observed_counts": (1, 0)},
        {"loss_sums": (-0.25, 0.0)},
        {"reservoir_word": 8},
        {"reservoir_remaining": 2},
        {"backend_refills": 1},
        {"prediction_hits": True},
    ],
)
def test_malformed_owned_snapshot_cannot_partially_restore(state_example, changed):
    example = state_example
    owner = example.HostConsumer(0)
    before = owner.export_state()
    invalid = before.model_copy(update=changed)
    with pytest.raises(ValueError):
        owner.restore_state(invalid)
    assert owner.export_state().to_json() == before.to_json()


def test_pending_capacity_refuses_before_consuming_a_bit_or_charging_an_action(state_example):
    example = state_example
    host = example.HostConsumer(0)
    for ordinal in range(128):
        host.decide(example.opportunity(0, ordinal).request)
    before = host.export_state().to_json()
    request = example.opportunity(0, 128).request
    with pytest.raises(ValueError, match="pending capacity"):
        host.decide(request)
    assert host.export_state().to_json() == before


def test_demonstration_uses_actual_replay_and_literal_resource_loss_oracles(state_example):
    assert state_example.run() == {
        "evidence": "authored-synthetic-ci",
        "decisions": 3,
        "updates": 3,
        "loss_sum": 0.75,
        "virtual_seconds": 13,
        "backend_refills": 0,
    }
