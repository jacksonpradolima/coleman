"""A restored continuation must preserve the actual acknowledged closed views."""

from functools import partial

import pytest

from coleman.core.splits import TemporalObservation
from coleman.observations.contracts import ObservationRevision
from coleman.replay.retained import RetainedReplay
from coleman.replay.retained_contracts import OrdinalRetention
from coleman.replay.revisions import validate_feedback_graph
from coleman.results.domain import Measurement
from coleman.state import _relations
from coleman.state.components import ComponentValue
from coleman.state.local import LocalStateStore
from tests.state.support import command


def checked(value, **updates):
    return type(value).model_validate(value.model_copy(update=updates))


def row(example, *, available=2, received=2):
    return ObservationRevision(
        schema_version=1,
        metadata=TemporalObservation(
            observation_id="authored-closed-build-duration",
            revision_id="original",
            group_id="authored-closed-build",
            event_at=example.instant(0),
            available_at=example.instant(available),
        ),
        entity_id="authored-ci-test",
        received_at=example.instant(received),
        supersedes=None,
        measurement=Measurement(
            name="authored-duration",
            value=3.0,
            unit="s",
            meaning="authored-ci-duration-v1",
            direction="minimize",
            status="observed",
        ),
    )


def correction(example, source, *, available=2, received=2):
    original = source.state.feedback[0]
    timing = checked(
        original.batch.timing,
        available_at=example.instant(available),
        received_at=None if received is None else example.instant(received),
    )
    batch = checked(
        original.batch,
        revision_id="authored-later-revision",
        supersedes=original.batch.revision_id,
        timing=timing,
    )
    return checked(original, batch=batch)


@pytest.fixture(scope="module")
def closed(state_example):
    example = state_example
    owner = example.ReplayOwner(0)
    action = owner.step(0.25, at=0)
    owner.replay.advance(eligible_through=example.instant(5), apply_at=example.instant(10))
    domain = example.Domain()
    domain.apply(action.chosen)
    return owner.export_state(), domain.export_state()


def candidate(example, source, kind, *, available=2, received=2):
    if kind == "history":
        addition = row(example, available=available, received=received)
        history = checked(source.state.history, revisions=(*source.state.history.revisions, addition))
        state = checked(source.state, history=history)
    else:
        addition = correction(example, source, available=available, received=received)
        feedback = validate_feedback_graph(
            (*source.state.feedback, addition), source.state.decisions, source.state.policy.capability
        )
        state = checked(source.state, feedback=feedback)
    return checked(source, state=state), addition


@pytest.mark.parametrize("kind", ["history", "feedback"])
def test_live_and_restored_successors_reject_the_same_closed_frontier_backfill(state_example, closed, kind):
    source, _ = closed
    restored, addition = candidate(state_example, source, kind)
    assert restored.state.decisions[0].snapshot == source.state.decisions[0].snapshot
    assert restored.host_state.to_json() == source.host_state.to_json()
    live = state_example.ReplayOwner(0)
    live.restore_state(source)
    if kind == "history":
        principal = partial(live.replay.admit_history, restored.state.history)
    else:
        principal = partial(live.replay.admit_feedback, (addition,))
    with pytest.raises(ValueError, match="backfill|frontier|cohort"):
        principal()
    with pytest.raises(ValueError, match="backfill|frontier|cohort"):
        _relations._replay_successor(source, restored, "transition", None)


@pytest.mark.parametrize("kind", ["history", "feedback"])
def test_actual_store_cannot_acknowledge_a_restored_backfill_across_its_closed_source(
    state_example, closed, tmp_path, kind
):
    example = state_example
    source, domain = closed
    restored, _ = candidate(example, source, kind)
    definition, bindings = example.collection_layout()
    compatibility = example.execution_compatibility()
    store = LocalStateStore.create(tmp_path / "store", definition, compatibility)
    values = (ComponentValue(bindings[0].codec, source), ComponentValue(bindings[-1].codec, domain))
    first = store.initialize(command(0, None, kind="initialize"), values, bindings, compatibility=compatibility)
    request = command(1, first.current_head)
    callbacks = []

    def restore(stage):
        callbacks.append("actual-restore")
        stage.owner(bindings[0]).restore_state(restored)

    with pytest.raises(ValueError, match="backfill|frontier|cohort"):
        store.commit_transition(request, bindings, restore, compatibility=compatibility)
    assert callbacks == ["actual-restore"]
    assert store.read_head() == first.current_head
    actual = store.load(bindings, compatibility=compatibility).component(bindings[0].codec)
    assert actual.to_json() == source.to_json()


@pytest.mark.parametrize("kind", ["history", "feedback"])
def test_the_acknowledged_information_cutoff_is_inclusive(state_example, closed, kind):
    source, _ = closed
    restored, addition = candidate(state_example, source, kind, available=5, received=5)
    live = state_example.ReplayOwner(0)
    live.restore_state(source)
    if kind == "history":
        principal = partial(live.replay.admit_history, restored.state.history)
    else:
        principal = partial(live.replay.admit_feedback, (addition,))
    with pytest.raises(ValueError, match="backfill|frontier|cohort"):
        principal()
    with pytest.raises(ValueError, match="backfill|frontier|cohort"):
        _relations._replay_successor(source, restored, "transition", None)


@pytest.mark.parametrize("kind", ["history", "feedback"])
@pytest.mark.parametrize("available,received", [(6, 6), (2, 6)], ids=["future-source", "first-receipt-after-cutoff"])
@pytest.mark.parametrize("advance", [False, True], ids=["pending", "next-information-frontier"])
def test_future_information_uses_old_information_clock_and_not_processing_clock(
    state_example, closed, kind, available, received, advance
):
    example = state_example
    source, _ = closed
    staged, addition = candidate(example, source, kind, available=available, received=received)
    live = example.ReplayOwner(0)
    live.restore_state(source)
    if kind == "history":
        live.replay.admit_history(staged.state.history)
    else:
        live.replay.admit_feedback((addition,))
    if advance:
        live.replay.advance(eligible_through=example.instant(6), apply_at=example.instant(10))
    restored = live.export_state()
    assert source.state.eligible_through == example.instant(5)
    assert source.state.apply_at == example.instant(10)
    assert restored.state.update_count == source.state.update_count
    assert restored.host_state.to_json() == source.host_state.to_json()
    _relations._replay_successor(source, restored, "transition", None)
    if kind == "history":
        assert restored.state.history.as_of(example.instant(5)) == source.state.history.as_of(example.instant(5))
        assert restored.state.history.as_of(example.instant(6)) != source.state.history.as_of(example.instant(6))


@pytest.mark.parametrize("kind", ["history", "feedback"])
def test_actual_store_accepts_pending_future_information_without_reapplying_a_consumed_round(
    state_example, closed, tmp_path, kind
):
    example = state_example
    source, domain = closed
    restored, _ = candidate(example, source, kind, available=2, received=6)
    definition, bindings = example.collection_layout()
    compatibility = example.execution_compatibility()
    store = LocalStateStore.create(tmp_path / "store", definition, compatibility)
    values = (ComponentValue(bindings[0].codec, source), ComponentValue(bindings[-1].codec, domain))
    first = store.initialize(command(0, None, kind="initialize"), values, bindings, compatibility=compatibility)
    request = command(1, first.current_head)

    def restore(stage):
        stage.owner(bindings[0]).restore_state(restored)

    second = store.commit_transition(request, bindings, restore, compatibility=compatibility)
    assert second.current_head != first.current_head
    assert store.read_head() == second.current_head
    actual = store.load(bindings, compatibility=compatibility).component(bindings[0].codec)
    assert actual.to_json() == restored.to_json()
    assert actual.state.update_count == source.state.update_count
    assert actual.host_state.to_json() == source.host_state.to_json()


def test_unknown_first_receipt_remains_pending_in_a_successor(state_example, closed):
    example = state_example
    source, _ = closed
    restored, addition = candidate(example, source, "feedback", available=2, received=None)
    live = example.ReplayOwner(0)
    live.restore_state(source)
    live.replay.admit_feedback((addition,))
    assert live.export_state().to_json() == restored.to_json()
    assert addition.batch.timing.received_at is None
    _relations._replay_successor(source, restored, "transition", None)


def test_already_admitted_future_feedback_may_become_visible_in_a_later_drain(state_example, closed):
    example = state_example
    source, _ = closed
    _, addition = candidate(example, source, "feedback", available=6, received=6)
    live = example.ReplayOwner(0)
    live.restore_state(source)
    live.replay.admit_feedback((addition,))
    pending = live.export_state()
    live.replay.advance(eligible_through=example.instant(6), apply_at=example.instant(10))
    visible = live.export_state()
    assert visible.state.feedback == pending.state.feedback
    assert visible.state.update_count == pending.state.update_count
    assert visible.state.eligible_through > pending.state.eligible_through
    _relations._replay_successor(pending, visible, "transition", None)


def test_new_same_clock_producer_after_the_old_drain_can_admit_its_own_feedback(state_example):
    example = state_example
    owner = example.ReplayOwner(0)
    owner.step(0.25, at=0)
    owner.replay.advance(eligible_through=example.instant(5), apply_at=example.instant(5))
    source = owner.export_state()
    owner.step(0.5, at=5)
    restored = owner.export_state()
    assert source.state.eligible_through == restored.state.eligible_through
    assert restored.state.decisions[-1].sequence > source.state.drains[-1].sequence
    assert restored.state.decision_count == source.state.decision_count + 1
    assert restored.state.update_count == source.state.update_count + 1
    _relations._replay_successor(source, restored, "transition", None)


def test_first_complete_history_does_not_invent_a_previous_empty_source_view(state_example):
    example = state_example
    host = example.HostConsumer(0)
    replay = RetainedReplay(
        host,
        example.policy(0),
        state_type=example.HostState,
        state_codec=example.CODEC,
        retention=OrdinalRetention(
            profile="quiescent-ordinal-v1", source_epoch=example.EPOCH, maximum_tail_records=512
        ),
    )
    replay.advance(eligible_through=example.instant(5), apply_at=example.instant(10))
    source = replay.export_continuation()
    history = checked(example.history(), revisions=(row(example),))
    replay.admit_history(history)
    restored = replay.export_continuation()
    assert source.state.history is None
    assert restored.state.history == history
    assert restored.state.decision_count == 0
    assert restored.state.update_count == 0
    _relations._replay_successor(source, restored, "transition", None)
