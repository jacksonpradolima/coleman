"""Independent literal arithmetic qualifies the actual 32-child pending state."""

import json
from pathlib import Path

import pytest

from coleman.core.actions import ChoiceAction
from coleman.replay.retained import RetainedReplay
from coleman.replay.retained_codec import read_retained_continuation
from tests.state.support import emit

FIXTURE = Path(__file__).parents[1] / "fixtures/state/v1/ci-32-oracle.json"
ORACLE = json.loads(FIXTURE.read_text())


@pytest.mark.parametrize("child", range(32))
def test_every_actual_pending_child_matches_the_authored_table_and_full_snapshot(
    state_example, pending_evidence, child
):
    example = state_example
    continuations, _, _ = pending_evidence
    snapshot = continuations[child]
    state = snapshot.host_state
    expected = ORACLE["children"][child]
    assert state.decision_count == expected["decisions"]
    assert state.update_count == expected["updates"]
    assert len(state.pending) == expected["pending"]
    assert sum(state.loss_sums) == expected["loss_quarters"] / 4
    assert state.cadence_count == expected["cadence"]
    assert state.book.spent_seconds == expected["virtual_seconds"]
    assert ["A" if plan == example.PLANS[0] else "B" for plan in state.book.recent] == expected["path"]
    assert state.cache is not None
    assert state.cache.value == expected["priority"]
    assert state.prediction_misses == state.decision_count + state.update_count
    decoded = read_retained_continuation(snapshot.to_json(), state_type=example.HostState)
    assert decoded.to_json() == snapshot.to_json()


def test_global_applied_path_is_distinct_from_the_sum_of_virtual_path_charges(state_example, pending_evidence):
    continuations, _, domain = pending_evidence
    totals = ORACLE["totals"]
    assert sum(row.host_state.decision_count for row in continuations) == totals["decisions"]
    assert sum(row.host_state.update_count for row in continuations) == totals["updates"]
    assert sum(len(row.host_state.pending) for row in continuations) == totals["pending"]
    assert sum(sum(row.host_state.loss_sums) for row in continuations) == 16
    assert sum(row.host_state.cadence_count for row in continuations) == 42
    assert sum(row.host_state.book.spent_seconds for row in continuations) == 389
    assert domain.book.releases == 95
    assert domain.book.count_b == 25
    assert domain.book.switches == 8
    assert domain.book.spent_seconds == 351
    assert domain.book.last_plan == state_example.PLANS[0]


@pytest.mark.parametrize("child", [0, 1, 2, 15, 31])
def test_actual_resumed_child_uses_its_own_buffered_suffix_and_not_factory_seed_zero(
    state_example, pending_evidence, child
):
    example = state_example
    snapshots, _, _ = pending_evidence
    snapshot = snapshots[child]
    host = example.HostConsumer(child)
    replay = RetainedReplay.from_continuation(host, snapshot, state_type=example.HostState, state_codec=example.CODEC)
    choices = []
    ordinal = snapshot.state.decision_count
    expected = ORACLE["next_bits"][f"recipe-{child:02d}"]
    for offset in range(len(expected)):
        record = emit(example, replay, child, ordinal + offset, at=ordinal + offset + 1)
        assert record.executed.action == record.executed.proposal.action
        assert isinstance(record.executed.action, ChoiceAction)
        choices.append(int(record.executed.action.chosen == example.PLANS[1]))
    assert choices == expected
    assert host.export_state().update_count == snapshot.host_state.update_count
    assert host.export_state().book.releases == ordinal + len(expected)


def test_completing_actual47_pending_labels_changes_no_rng_release_or_path_book(state_example, pending_evidence):
    example = state_example
    snapshots, records, _ = pending_evidence
    restored = []
    for child, snapshot in enumerate(snapshots):
        host = example.HostConsumer(child)
        replay = RetainedReplay.from_continuation(
            host, snapshot, state_type=example.HostState, state_codec=example.CODEC
        )
        before = host.export_state()
        remaining = records[child][before.update_count :]
        if remaining:
            offered = tuple(
                example.feedback(
                    record,
                    decisions=before.decision_count,
                    available=before.decision_count + 1,
                    value=((child + record.decision_index) % 4) / 4,
                )
                for record in remaining
            )
            replay.admit_feedback(offered)
            at = example.instant(before.decision_count + 1)
            replay.advance(eligible_through=at, apply_at=at)
        after = host.export_state()
        assert after.decision_count == before.decision_count
        assert after.cadence_count == before.cadence_count
        assert after.rng.to_json() == before.rng.to_json()
        assert after.book == before.book
        assert after.pending == ()
        assert after.update_count == after.decision_count
        assert sum(after.loss_sums) == ORACLE["children"][child]["resolved_loss_quarters"] / 4
        sealed = replay.compact()
        assert sealed.state.seal is not None
        assert sealed.state.seal.decision_count == after.decision_count
        restored.append(after)
    assert sum(row.update_count for row in restored) == 95
    assert sum(sum(row.loss_sums) for row in restored) == 35.5
    assert sum(row.cadence_count for row in restored) == 42
