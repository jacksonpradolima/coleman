"""Retained replay preserves real causal phases, versions and global counters."""

from typing import Any

import pytest

from coleman.core.base import CoreContract
from coleman.replay.codec import read_continuation, read_replay_state
from coleman.replay.engine import CausalReplay
from coleman.replay.retained import RetainedReplay, ordinal_decision_id
from coleman.replay.retained_codec import read_retained_continuation, read_retained_state
from coleman.replay.retained_contracts import RetainedContinuation
from tests.state.support import complete, emit


def rounds(example, replay, start, stop):
    result = []
    for ordinal in range(start, stop):
        record = emit(example, replay, 0, ordinal)
        complete(example, replay, record, (ordinal % 4) / 4, at=ordinal)
        result.append(record)
    return result


@pytest.mark.parametrize("argument", [Any, object, CoreContract])
@pytest.mark.parametrize("boundary", ["constructor", "reader", "unchecked_export"])
def test_v2_cannot_erase_concrete_owned_state_through_open_or_base_generic_parameters(
    state_example, argument, boundary
):
    example = state_example
    replay, host = example.make_replay(0)
    source = replay.export_continuation()
    selected = RetainedContinuation[argument]
    if boundary == "constructor":
        principal = selected
        with pytest.raises(ValueError, match="concrete"):
            principal(schema_version=2, state=source.state, host_state=source.host_state)
    elif boundary == "reader":
        payload = source.to_json()
        with pytest.raises(ValueError, match="concrete"):
            selected.from_json(payload)
    else:
        unchecked = selected.model_construct(schema_version=2, state=source.state, host_state=source.host_state)
        copied = unchecked.model_copy()
        with pytest.raises(ValueError, match="concrete"):
            copied.to_json()
    assert host.export_state().decision_count == 0


def test_origin_state_is_explicit_v2_and_readers_do_not_implicitly_convert_versions(state_example):
    example = state_example
    replay, _ = example.make_replay(0)
    state_payload = replay.state.to_json()
    continuation_payload = replay.export_continuation().to_json()
    assert read_retained_state(state_payload).to_json() == state_payload
    assert (
        read_retained_continuation(continuation_payload, state_type=example.HostState).to_json() == continuation_payload
    )
    with pytest.raises(ValueError):
        read_replay_state(state_payload)
    with pytest.raises(ValueError):
        read_continuation(continuation_payload, state_type=example.HostState)
    legacy = CausalReplay(
        example.HostConsumer(), example.policy(0), state_type=example.HostState, state_codec=example.CODEC
    )
    old_state, old_continuation = legacy.state.to_json(), legacy.export_continuation().to_json()
    with pytest.raises(ValueError):
        read_retained_state(old_state)
    with pytest.raises(ValueError):
        read_retained_continuation(old_continuation, state_type=example.HostState)


def test_pending_compaction_refuses_without_mutating_the_real_host_or_replay(state_example):
    replay, host = state_example.make_replay(0)
    emit(state_example, replay, 0, 0)
    before = replay.export_continuation().to_json()
    with pytest.raises(ValueError, match="quiescent|pending|complete"):
        replay.compact()
    assert replay.export_continuation().to_json() == before
    assert host.export_state().update_count == 0


def test_ordinal_opportunity_gap_rejects_before_the_actual_consumer_draws_or_charges(state_example):
    example = state_example
    replay, host = example.make_replay(0)
    before = host.export_state().to_json()
    request = example.opportunity(0, 2)
    with pytest.raises(ValueError):
        replay.prepare(request)
    assert host.export_state().to_json() == before
    assert replay.state.decision_count == 0
    assert ordinal_decision_id(example.EPOCH, 0) != ordinal_decision_id(example.EPOCH, 1)


def test_seal_floor_is_exclusive_and_same_instant_new_producer_feedback_remains_legal(state_example):
    example = state_example
    replay, host = example.make_replay(0)
    old_records = rounds(example, replay, 0, 2)
    before_host = host.export_state().to_json()
    sealed = replay.compact()
    assert sealed.state.decision_count == 2
    assert sealed.state.update_count == 2
    assert sealed.state.decisions == ()
    assert sealed.host_state.to_json() == before_host
    next_record = emit(example, replay, 0, 2, at=1)
    complete(example, replay, next_record, 0.5, at=1)
    assert replay.state.decision_count == 3
    assert replay.state.update_count == 3
    assert host.export_state().loss_sums == (0.25, 0.5)
    old = example.feedback(old_records[1], decisions=3, available=2, value=0.25)
    before = replay.export_continuation().to_json()
    with pytest.raises(ValueError, match="registered|expired|retired|seal"):
        replay.admit_feedback((old,))
    assert replay.export_continuation().to_json() == before


def test_profile_two_causal_commitment_is_independent_of_actual_seal_chunking(state_example):
    example = state_example
    first, first_host = example.make_replay(0)
    second, second_host = example.make_replay(0)
    rounds(example, first, 0, 2)
    first.compact()
    rounds(example, first, 2, 4)
    first.compact()
    rounds(example, second, 0, 4)
    second.compact()
    assert first.state.causal_prefix_fingerprint == second.state.causal_prefix_fingerprint
    assert first_host.export_state().to_json() == second_host.export_state().to_json()
    assert read_retained_state(first.state.to_json()).to_json() == first.state.to_json()


def test_compacted_continuation_refuses_inherited_fork_before_restoring_a_new_owner(state_example):
    example = state_example
    replay, _ = example.make_replay(0)
    rounds(example, replay, 0, 1)
    continuation = replay.compact()
    fresh = example.HostConsumer(0)
    before = fresh.export_state().to_json()
    lineage = example.reference("forbidden-retained-fork")
    with pytest.raises(ValueError, match="fork|compacted|seal"):
        RetainedReplay.fork_from_continuation(
            fresh, continuation, lineage=lineage, state_type=example.HostState, state_codec=example.CODEC
        )
    assert fresh.export_state().to_json() == before
