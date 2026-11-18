"""Actual causal engine, original mark delivery and feedback-ready continuation."""

import json
from datetime import timedelta

import pytest

from coleman.core.actions import VectorAction
from coleman.replay.codec import read_continuation
from examples.synthetic_portfolio import model
from examples.synthetic_portfolio.experiment import Experiment, ExperimentState
from examples.synthetic_portfolio.replay import (
    PortfolioReplayHost,
    PortfolioReplayState,
    feedback,
    opportunity,
    start,
)


def _all_validated_fields(experiment):
    """Compare every exported field using independent standard JSON serialization."""
    snapshot = experiment.export_state()
    return json.dumps(
        snapshot.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def test_real_causal_replay_matches_direct_at_original_feedback_ready_checkpoint():
    host, replay = start()
    direct = Experiment()
    assert host.experiment.rows is None
    first = replay.emit(replay.prepare(opportunity(replay, 0)), execution="offline_identity_v1")
    direct.step(0)
    record = first
    for index in range(1, 16):
        event = feedback(record, model.PRICE_ROWS[index])
        replay.admit_feedback((event,))
        at = model.START + timedelta(days=index)
        replay.advance(eligible_through=at, apply_at=at)
        direct.apply_feedback(index)
        assert _all_validated_fields(host.experiment) == _all_validated_fields(direct)
        if index == 8:
            # Save after actual eligible label application and before the new
            # decision. This is the phase a generic replay consumer must own.
            snapshot = read_continuation(replay.export_continuation().to_json(), state_type=PortfolioReplayState)
            assert snapshot.host_state.experiment.ready_observation is not None
            assert snapshot.host_state.experiment.ready_observation.index == 8
            host = PortfolioReplayHost(source_fingerprint=direct.source_fingerprint)
            replay = host.resume(snapshot)
            previous = host.experiment.export_state().fingerprint
            replay.admit_feedback((event,))
            replay.advance(eligible_through=at, apply_at=at)
            assert host.experiment.export_state().fingerprint == previous
            direct.restore_state(ExperimentState.from_json(direct.export_state().to_json()))
        record = replay.emit(replay.prepare(opportunity(replay, index)), execution="offline_identity_v1")
        direct.step(index)
        assert _all_validated_fields(host.experiment) == _all_validated_fields(direct)
        component = next(row for row in direct.periods[-1].candidates if row.name == "composed-online")
        assert isinstance(record.executed.action, VectorAction)
        assert component.solver is not None and component.solver.executed is not None
        assert tuple(value.value for value in record.executed.action.values) == component.solver.executed
        # The core vector represents the real pre-funding mathematical point;
        # the exact funded asset fractions live in the separate Target record.
        assert component.proposed_target is not None
        assert component.proposed_target.weights[0].value.fraction != component.solver.executed[0]
    assert host.experiment.ridge.export_state().payload.sample_count == 15
    assert host.experiment.export_state().to_json() == direct.export_state().to_json()


def test_future_feedback_and_late_correction_do_not_enter_live_training():
    host, replay = start()
    record = replay.emit(replay.prepare(opportunity(replay, 0)), execution="offline_identity_v1")
    event = feedback(record, model.PRICE_ROWS[1])
    replay.admit_feedback((event,))
    replay.advance(eligible_through=model.START, apply_at=model.START)
    assert host.experiment.ridge.export_state().payload.sample_count == 0
    at = model.START + timedelta(days=1)
    replay.advance(eligible_through=at, apply_at=at)
    before = host.experiment.export_state().fingerprint
    correction = feedback(record, (150, 50), revision="late-correction", supersedes="initial", receipt_days=2)
    replay.admit_feedback((correction,))
    replay.advance(eligible_through=model.START + timedelta(days=3), apply_at=model.START + timedelta(days=3))
    assert host.experiment.export_state().fingerprint == before
    assert host.experiment.ridge.export_state().payload.sample_count == 1


def test_missing_current_mark_refuses_without_a_zero_or_changed_host():
    host, replay = start()
    before = host.export_state().fingerprint
    with pytest.raises(ValueError, match="missing or incompatible"):
        opportunity(replay, 16)
    assert host.export_state().fingerprint == before


def test_owned_snapshot_reuse_keeps_full_wire_and_revalidates_every_unowned_ingress():
    host, replay = start()
    replay.emit(replay.prepare(opportunity(replay, 0)), execution="offline_identity_v1")
    owned = host.export_state()
    assert host.state_type.model_validate(owned) is owned
    # Neither a copied object nor a constructed object has the owning instance's
    # authority, even when its concrete class happens to match the owner.
    for invalid in (
        owned.model_copy(update={"schema_version": 2}),
        host.state_type.model_construct(**(owned.model_dump() | {"schema_version": 2})),
        owned.model_dump() | {"schema_version": 2},
    ):
        with pytest.raises(ValueError):
            host.state_type.model_validate(invalid)
    raw = json.loads(owned.to_json())
    raw["experiment"]["periods"][0]["candidates"][0]["settlement"]["net_value"]["numerator"] = 999999
    with pytest.raises(ValueError):
        host.state_type.from_json(json.dumps(raw))
    assert PortfolioReplayState.from_json(owned.to_json()).to_json() == owned.to_json()
    assert host.state_type.model_validate(owned.model_dump()).to_json() == owned.to_json()
    assert _all_validated_fields(host.experiment) == host.experiment.export_state().to_json()


@pytest.mark.parametrize("through_json", [False, True])
def test_cold_owner_restores_in_memory_and_json_without_wire_selected_type(through_json):
    host, replay = start()
    record = replay.emit(replay.prepare(opportunity(replay, 0)), execution="offline_identity_v1")
    replay.admit_feedback((feedback(record, model.PRICE_ROWS[1]),))
    at = model.START + timedelta(days=1)
    replay.advance(eligible_through=at, apply_at=at)
    original = replay.export_continuation()
    saved = read_continuation(original.to_json(), state_type=PortfolioReplayState) if through_json else original
    other = PortfolioReplayHost(source_fingerprint=host.experiment.source_fingerprint)
    assert other.state_type is not host.state_type
    restored = other.resume(saved)
    assert restored.export_continuation().to_json() == original.to_json()
    next_record = restored.emit(restored.prepare(opportunity(restored, 1)), execution="offline_identity_v1")
    assert next_record.executed.proposal.request.round == 1
    assert other.experiment.ridge.export_state().payload.sample_count == 1


def test_cached_snapshot_does_not_conceal_an_advanced_component_or_changed_archive():
    host, replay = start()
    replay.emit(replay.prepare(opportunity(replay, 0)), execution="offline_identity_v1")
    host.experiment.step(1, prices=model.prices(1))
    with pytest.raises(ValueError, match="outside their owning"):
        host.export_state()
