"""Actual acknowledgements anchor the first new causal record to the old host."""

import pytest

from coleman.replay.retained import RetainedReplay
from coleman.replay.retained_contracts import OrdinalRetention
from coleman.state.local import LocalStateStore
from tests.state.support import command


def origin_store(example, root):
    """Initialize real complete origin snapshots, without replacing store work."""
    definition, bindings = example.collection_layout()
    compatibility = example.execution_compatibility()
    store = LocalStateStore.create(root, definition, compatibility)
    receipt = store.initialize(
        command(0, None, kind="initialize"),
        example.origin_values(bindings),
        bindings,
        compatibility=compatibility,
    )
    return store, bindings, compatibility, receipt.current_head


def changed_origin(example, owner):
    """Build another internally valid origin with changed cache/accounting."""
    host = example.HostConsumer(0)
    host.prewarm(example.instant(0))
    replay = RetainedReplay(
        host,
        example.policy(0),
        state_type=example.HostState,
        state_codec=example.CODEC,
        retention=OrdinalRetention(
            profile="quiescent-ordinal-v1", source_epoch=example.EPOCH, maximum_tail_records=512
        ),
    )
    replay.admit_history(example.history())
    owner.host, owner.replay = host, replay


@pytest.mark.parametrize("add_causal_records", [False, True])
def test_acknowledged_origin_rejects_unrecorded_host_change_before_no_or_first_new_record(
    state_example, tmp_path, add_causal_records
):
    example = state_example
    store, bindings, compatibility, head = origin_store(example, tmp_path / "origin")
    before = store.load(bindings, compatibility=compatibility)
    candidate_was_complete = []

    def rewrite(stage):
        owner = stage.owner(bindings[0])
        changed_origin(example, owner)
        if add_causal_records:
            action = owner.step(0.25)
            stage.owner(bindings[-1]).apply(action.chosen)
        # Both candidates are individually valid complete continuations. The
        # failing boundary must compare them with the actual prior ack host.
        candidate = owner.export_state()
        assert candidate.host_state.cache is not None
        assert candidate.host_state.prediction_hits == int(add_causal_records)
        candidate_was_complete.append(candidate.fingerprint)

    request = command(1, head)
    with pytest.raises(ValueError, match="host|causal"):
        store.commit_transition(request, bindings, rewrite, compatibility=compatibility)
    assert len(candidate_was_complete) == 1
    assert store.read_head() == head
    after = store.load(bindings, compatibility=compatibility)
    assert tuple(item.value.to_json() for item in after.values) == tuple(item.value.to_json() for item in before.values)


def test_unchanged_host_and_legitimate_first_causal_extension_remain_publishable(state_example, tmp_path):
    example = state_example
    store, bindings, compatibility, head = origin_store(example, tmp_path / "positive")
    unchanged = store.commit_transition(command(1, head), bindings, lambda stage: None, compatibility=compatibility)
    before = store.load(bindings, compatibility=compatibility).component(bindings[0].codec)

    def actual_round(stage):
        action = stage.owner(bindings[0]).step(0.25)
        stage.owner(bindings[-1]).apply(action.chosen)

    accepted = store.commit_transition(
        command(2, unchanged.current_head), bindings, actual_round, compatibility=compatibility
    )
    after = store.load(bindings, compatibility=compatibility).component(bindings[0].codec)
    assert accepted.current_head.generation == 2
    assert after.state.decision_count == 1
    assert after.state.update_count == 1
    assert after.state.drains[0].host_before == before.state.host_fingerprint
    assert after.host_state.loss_sums == (0.25, 0.0)
