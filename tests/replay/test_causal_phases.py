"""Equal timestamps retain separate producer and pre-decision drain phases."""

import json
from functools import partial

import pytest

from coleman.replay.codec import read_continuation, read_replay_state
from tests.replay.support import chosen, emit, session


def same_clock_trace(example):
    replay, host = session(example)
    first = emit(example, replay, 0, cutoff=0, at=0, end=0)
    second = emit(example, replay, 1, cutoff=0, at=0, end=0)
    feedback = example.feedback(first, (1.0, 0.0, 0.5), visible=0, delay=0)
    return replay, host, first, second, feedback


def test_real_consecutive_same_clock_decisions_retain_two_distinct_empty_drains(example):
    replay, host, first, second, _ = same_clock_trace(example)
    assert [record.decision_index for record in (first, second)] == [0, 1]
    assert [record.accepted_at for record in (first, second)] == [example.instant(0), example.instant(0)]
    assert [drain.decision_count for drain in replay.state.drains] == [0, 1]
    assert [drain.sequence for drain in replay.state.drains] == [0, 2]
    assert replay.state.update_count == 0
    assert host.export_state().decision_count == 2
    assert read_replay_state(replay.state.to_json()).to_json() == replay.state.to_json()


def test_retained_second_drain_already_rejects_an_omitted_same_instant_ready_round(example):
    replay, host, _, _, feedback = same_clock_trace(example)
    document = json.loads(replay.state.to_json())
    document["feedback"] = [feedback.model_dump(mode="json")]
    wire = json.dumps(document)
    before = host.export_state().to_json()
    with pytest.raises(ValueError, match="omits or changes a complete eligible"):
        read_replay_state(wire)
    assert host.export_state().to_json() == before


@pytest.mark.parametrize("reader", ["state", "continuation"])
def test_deleting_the_second_same_clock_drain_cannot_hide_a_ready_round_from_the_reader(example, reader):
    replay, host, first, second, feedback = same_clock_trace(example)
    document = json.loads(replay.export_continuation().to_json())
    state = document["state"]
    assert first.sequence == 1
    assert second.sequence == 3
    # The valid empty trace has drain0,D0,drain1,D1. Inject source-visible F0,
    # then remove drain1 and close the sequence gap: only drain0 precedes D1.
    # Its clocks match D1, but its decision_count=0 cannot drain D0's feedback.
    state["feedback"] = [feedback.model_dump(mode="json")]
    state["drains"] = state["drains"][:1]
    state["decisions"][1]["sequence"] = 2
    wire = json.dumps(state if reader == "state" else document)
    before = host.export_state().to_json()
    principal = (
        read_replay_state if reader == "state" else partial(read_continuation, state_type=example.HostConsumerState)
    )
    with pytest.raises(ValueError, match="completed decision/update-count boundary"):
        principal(wire)
    assert host.export_state().to_json() == before


def test_actual_same_instant_feedback_between_decisions_is_applied_in_the_second_drain(example):
    replay, host = session(example)
    first = emit(example, replay, 0, cutoff=0, at=0, end=0)
    feedback = example.feedback(first, (1.0, 0.0, 0.5), visible=0, delay=0)
    replay.admit_feedback((feedback,))
    second = emit(example, replay, 1, cutoff=0, at=0, end=0)
    assert chosen(first) == "suite-a"
    assert chosen(second) == "suite-b"
    assert second.update_count_before == 1
    assert [drain.decision_count for drain in replay.state.drains] == [0, 1]
    assert [drain.update_count_after for drain in replay.state.drains] == [0, 1]
    application = replay.state.updates[0].application
    assert application.apply_at == example.instant(0)
    assert application.decision_count_before == 1
    assert application.feedback.batch.delay_rounds == 0
    assert host.export_state().observed_counts == (1, 0, 0)
    continuation = replay.export_continuation()
    decoded = read_continuation(continuation.to_json(), state_type=type(host.export_state()))
    assert decoded.to_json() == continuation.to_json()
