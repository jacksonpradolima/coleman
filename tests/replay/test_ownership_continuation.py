"""Real host ownership, whole continuations and independently timed child paths."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from coleman.replay.codec import read_continuation
from coleman.replay.engine import CausalReplay, ReplayBusyError, ReplayInvalidatedError
from tests.replay.support import chosen, emit, session


def populated(example):
    replay, host = session(example)
    first = emit(example, replay, 0, at=0, end=1)
    emit(example, replay, 1, at=1, end=2)
    replay.admit_feedback((example.feedback(first, (1.0, 0.0, 0.5), visible=2, delay=1),))
    emit(example, replay, 2, at=3, end=4)
    return replay, host


def restore(example, continuation, *, host=None, codec=None):
    host = example.HostConsumer() if host is None else host
    replay = CausalReplay.from_continuation(
        host,
        continuation,
        state_type=example.HostConsumerState,
        state_codec=example.CODEC if codec is None else codec,
    )
    return replay, host


def test_populated_concrete_continuation_serializes_every_host_field_and_resumes_exactly(example):
    replay, host = populated(example)
    continuation = replay.export_continuation()
    document = json.loads(continuation.to_json())
    assert set(document["host_state"]) == set(example.HostConsumerState.model_fields)
    assert document["host_state"]["decision_count"] == 3
    assert document["host_state"]["update_count"] == 1
    assert len(document["host_state"]["executions"]) == 3
    assert len(document["host_state"]["applications"]) == 1
    assert len(document["host_state"]["pending"]) == 2
    assert document["host_state"]["loss_sums"] == [1.0, 0.0, 0.0]
    decoded = read_continuation(continuation.to_json(), state_type=example.HostConsumerState)
    assert type(decoded.host_state) is example.HostConsumerState
    assert decoded.to_json() == continuation.to_json()
    resumed, fresh = restore(example, decoded)
    original_next = emit(example, replay, 3, at=5, end=6)
    resumed_next = emit(example, resumed, 3, at=5, end=6)
    assert resumed_next.to_json() == original_next.to_json()
    assert fresh.export_state().to_json() == host.export_state().to_json()
    assert resumed.export_continuation().to_json() == replay.export_continuation().to_json()


@pytest.mark.parametrize("field,value", [("decision_count", 2), ("loss_sums", [0.0, 0.0, 0.0]), ("pending", [])])
def test_reader_rejects_forged_complete_host_facts_before_restoration(example, field, value):
    replay, _ = populated(example)
    document = json.loads(replay.export_continuation().to_json())
    document["host_state"][field] = value
    wire = json.dumps(document)
    with pytest.raises(ValueError):
        read_continuation(wire, state_type=example.HostConsumerState)


def test_restore_rejects_wrong_codec_without_changing_the_fresh_host(example):
    replay, _ = populated(example)
    continuation = replay.export_continuation()
    fresh = example.HostConsumer()
    before = fresh.export_state().to_json()
    codec = example.reference("another-host-codec")
    with pytest.raises(ValueError, match="codec"):
        restore(example, continuation, host=fresh, codec=codec)
    assert fresh.export_state().to_json() == before


def test_accepted_emit_retry_returns_the_original_after_later_frontiers_without_another_decide(example):
    replay, host = session(example)
    opportunity = example.opportunity(0, at=0, end=1)
    prepared = replay.prepare(opportunity)
    original = replay.emit(prepared, execution="offline_identity_v1")
    emit(example, replay, 1, at=3, end=4)
    before = host.export_state().to_json()
    assert replay.emit(prepared, execution="offline_identity_v1").to_json() == original.to_json()
    assert host.export_state().to_json() == before
    target = opportunity.target.model_copy(update={"reference": example.reference("other")})
    changed = type(opportunity).model_validate(opportunity.model_copy(update={"target": target}))
    with pytest.raises(ValueError, match="Conflicting"):
        replay.prepare(changed)
    assert host.export_state().to_json() == before


def test_new_preparation_becomes_stale_after_inventory_admission_without_host_effect(example):
    replay, host = session(example)
    first = emit(example, replay, 0, at=0, end=1)
    prepared = replay.prepare(example.opportunity(1, at=3, end=4))
    replay.admit_feedback((example.feedback(first, (1.0, 0.0, 0.5), visible=2, delay=0),))
    before = host.export_state().to_json()
    with pytest.raises(ValueError, match="stale"):
        replay.emit(prepared, execution="offline_identity_v1")
    assert host.export_state().to_json() == before


def test_out_of_band_actual_host_transition_invalidates_the_bound_session(example):
    replay, host = session(example)
    host.decide(example.opportunity(0, at=0, end=1).request)
    with pytest.raises(ReplayInvalidatedError, match="outside"):
        replay.export_continuation()
    with pytest.raises(ReplayInvalidatedError):
        replay.state  # noqa: B018 - exercising the public guarded property


def test_reentrant_real_decider_is_rejected_and_the_outer_session_fails_stop(example):
    class Reentrant(example.HostConsumer):
        owner: CausalReplay | None = None

        def decide(self, request, /):
            assert self.owner is not None
            self.owner.export_continuation()
            return super().decide(request)

    host = Reentrant()
    replay, _ = session(example, host=host)
    host.owner = replay
    prepared = replay.prepare(example.opportunity(0, at=0, end=1))
    with pytest.raises(ReplayBusyError):
        replay.emit(prepared, execution="offline_identity_v1")
    assert host.export_state().decision_count == 0
    with pytest.raises(ReplayInvalidatedError):
        replay.export_continuation()


def test_concurrent_call_is_refused_while_the_real_decider_has_the_single_owner(example):
    entered, release = Event(), Event()

    class Waiting(example.HostConsumer):
        def decide(self, request, /):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("Test failed to release actual decider")
            return super().decide(request)

    host = Waiting()
    replay, _ = session(example, host=host)
    prepared = replay.prepare(example.opportunity(0, at=0, end=1))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(replay.emit, prepared, execution="offline_identity_v1")
        try:
            assert entered.wait(10)
            with pytest.raises(ReplayBusyError):
                replay.export_continuation()
        finally:
            release.set()
        record = future.result(timeout=10)
    assert chosen(record) == "suite-a"
    assert host.export_state().decision_count == 1
    assert replay.state.decision_count == 1


def test_independent_sessions_have_independent_real_hosts_under_concurrent_execution(example):
    def run(_):
        replay, host = populated(example)
        return replay.export_continuation().to_json(), host.export_state().to_json()

    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = pool.map(run, (0, 1))
    assert left == right


def test_actual_partial_host_update_failure_never_exposes_a_successful_continuation(example):
    class FailedAfterUpdate(example.HostConsumer):
        def update(self, application, /):
            super().update(application)
            raise RuntimeError("Host failed after its real state transition")

    host = FailedAfterUpdate()
    replay, _ = session(example, host=host)
    first = emit(example, replay, 0, at=0, end=1)
    replay.admit_feedback((example.feedback(first, (1.0, 0.0, 0.5), visible=2, delay=0),))
    instant = example.instant(2)
    with pytest.raises(RuntimeError, match="after its real"):
        replay.advance(eligible_through=instant, apply_at=instant)
    assert host.export_state().update_count == 1
    with pytest.raises(ReplayInvalidatedError):
        replay.export_continuation()
    with pytest.raises(ReplayInvalidatedError):
        replay.advance(eligible_through=instant, apply_at=instant)


def test_post_decide_export_failure_invalidates_even_after_the_host_count_increased(example):
    class ExportFailure(example.HostConsumer):
        fail = False

        def export_state(self):
            if self.fail:
                raise RuntimeError("Snapshot unavailable after mutation")
            return super().export_state()

        def decide(self, request, /):
            result = super().decide(request)
            self.fail = True
            return result

    host = ExportFailure()
    replay, _ = session(example, host=host)
    prepared = replay.prepare(example.opportunity(0, at=0, end=1))
    with pytest.raises(RuntimeError, match="Snapshot unavailable"):
        replay.emit(prepared, execution="offline_identity_v1")
    assert host._state.decision_count == 1
    with pytest.raises(ReplayInvalidatedError):
        replay.export_continuation()


@pytest.mark.parametrize("cutoff,processing", [(1, 4), (3, 2), (5, 4)])
def test_reversed_independent_frontiers_are_rejected_without_host_mutation(example, cutoff, processing):
    replay, host = session(example)
    emit(example, replay, 0, at=0, end=1)
    replay.advance(eligible_through=example.instant(2), apply_at=example.instant(3))
    before = replay.export_continuation().to_json()
    eligible, actual = example.instant(cutoff), example.instant(processing)
    with pytest.raises(ValueError):
        replay.advance(eligible_through=eligible, apply_at=actual)
    assert replay.export_continuation().to_json() == before
    assert host.export_state().decision_count == 1


def test_synchronous_profile_requires_completed_feedback_before_next_real_decision(example):
    policy = type(example.policy()).model_validate(example.policy().model_copy(update={"delivery": "synchronous"}))
    replay, host = session(example, policy=policy)
    first = emit(example, replay, 0, at=0, end=1)
    prepared = replay.prepare(example.opportunity(1, at=3, end=4))
    with pytest.raises(ValueError, match="Synchronous"):
        replay.emit(prepared, execution="offline_identity_v1")
    assert host.export_state().decision_count == 1
    replay.admit_feedback((example.feedback(first, (1.0, 0.0, 0.5), visible=2, delay=0),))
    emit(example, replay, 1, at=3, end=4)
    assert host.export_state().update_count == 1
    assert host.export_state().decision_count == 2


def test_matched_child_replays_correction_from_a_complete_prefix_without_changing_parent(example):
    parent, parent_host = session(example)
    first = emit(example, parent, 0, at=0, end=1)
    root = example.feedback(first, (1.0, 0.0, 0.5), visible=4, delay=0)
    parent.admit_feedback((root,))
    prefix = parent.export_continuation()
    prefix_bytes = prefix.to_json()
    parent.advance(eligible_through=example.instant(4), apply_at=example.instant(4))
    parent_bytes = parent.export_continuation().to_json()
    child_host = example.HostConsumer()
    child_lineage = example.reference("authored-ci-revision-child")
    child = CausalReplay.fork_from_continuation(
        child_host, prefix, lineage=child_lineage, state_type=example.HostConsumerState, state_codec=example.CODEC
    )
    correction = example.feedback(first, (0.0, 0.0, 0.5), visible=5, delay=0, revision="corrected", supersedes="root")
    child.admit_feedback((correction,))
    child.advance(eligible_through=example.instant(6), apply_at=example.instant(10))
    assert parent_host.export_state().loss_sums == (1.0, 0.0, 0.0)
    assert child_host.export_state().loss_sums == (0.0, 0.0, 0.0)
    application = child.state.updates[0].application
    assert application.lineage == child_lineage
    assert application.feedback.lineage == first.lineage
    assert application.apply_at == example.instant(10)
    assert application.eligible_at == example.instant(5)
    assert child.state.decisions[0].to_json() == first.to_json()
    assert parent.export_continuation().to_json() == parent_bytes
    assert prefix.to_json() == prefix_bytes
    parent_binding = child.state.policy.parent
    assert parent_binding is not None
    assert parent_binding.state_fingerprint == prefix.state.fingerprint


def test_child_cannot_reuse_a_retained_lineage(example):
    replay, _ = populated(example)
    continuation = replay.export_continuation()
    fresh = example.HostConsumer()
    before = fresh.export_state().to_json()
    with pytest.raises(ValueError):
        CausalReplay.fork_from_continuation(
            fresh,
            continuation,
            lineage=continuation.state.policy.lineage,
            state_type=example.HostConsumerState,
            state_codec=example.CODEC,
        )
    assert fresh.export_state().to_json() == before


def test_empty_nested_fork_cannot_reuse_the_original_root_lineage_after_a_drain(example):
    root, _ = session(example)
    root.advance(eligible_through=example.instant(0), apply_at=example.instant(0))
    root_lineage = root.state.policy.lineage
    child = CausalReplay.fork_from_continuation(
        example.HostConsumer(),
        root.export_continuation(),
        lineage=example.reference("first-empty-child"),
        state_type=example.HostConsumerState,
        state_codec=example.CODEC,
    )
    child_snapshot = child.export_continuation()
    fresh = example.HostConsumer()
    before = fresh.export_state().to_json()
    with pytest.raises(ValueError):
        CausalReplay.fork_from_continuation(
            fresh,
            child_snapshot,
            lineage=root_lineage,
            state_type=example.HostConsumerState,
            state_codec=example.CODEC,
        )
    assert fresh.export_state().to_json() == before


def test_truly_empty_two_generation_fork_retains_and_rejects_the_original_ancestor(example):
    root, _ = session(example)
    lineage = root.state.policy.lineage
    child = CausalReplay.fork_from_continuation(
        example.HostConsumer(),
        root.export_continuation(),
        lineage=example.reference("empty-first-generation"),
        state_type=example.HostConsumerState,
        state_codec=example.CODEC,
    )
    grandchild = CausalReplay.fork_from_continuation(
        example.HostConsumer(),
        child.export_continuation(),
        lineage=example.reference("empty-second-generation"),
        state_type=example.HostConsumerState,
        state_codec=example.CODEC,
    )
    snapshot = grandchild.export_continuation()
    assert snapshot.state.decisions == ()
    assert snapshot.state.drains == ()
    fresh = example.HostConsumer()
    with pytest.raises(ValueError):
        CausalReplay.fork_from_continuation(
            fresh,
            snapshot,
            lineage=lineage,
            state_type=example.HostConsumerState,
            state_codec=example.CODEC,
        )
    document = json.loads(snapshot.to_json())
    document["state"]["policy"]["lineage"] = lineage.model_dump(mode="json")
    wire = json.dumps(document)
    with pytest.raises(ValueError):
        read_continuation(wire, state_type=example.HostConsumerState)
