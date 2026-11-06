"""Versioned synthetic wire fixtures retain independently worked CI facts."""

import hashlib
import json

import pytest

from coleman.replay.codec import read_continuation, read_replay_state
from coleman.replay.engine import CausalReplay
from tests.replay.support import FIXTURES, chosen


@pytest.mark.parametrize("profile", ["chosen", "component", "full_panel"])
def test_v1_golden_is_deterministic_complete_readable_and_restorable(example, profile):
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    name = profile + "-continuation.json"
    receipt = manifest["files"][name]
    encoded = (FIXTURES / name).read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == receipt["sha256"]
    continuation = read_continuation(encoded, state_type=example.HostConsumerState)
    assert continuation.to_json().encode() + b"\n" == encoded
    assert continuation.fingerprint == receipt["continuation_fingerprint"]
    assert continuation.state.causal_prefix_fingerprint == receipt["causal_prefix_fingerprint"]
    assert read_replay_state(continuation.state.to_json()).to_json() == continuation.state.to_json()
    state = continuation.host_state
    assert state.decision_count == receipt["expected_decisions"]
    assert state.update_count == receipt["expected_updates"]
    assert list(state.observed_counts) == receipt["expected_counts"]
    assert list(state.loss_sums) == receipt["expected_loss_sums"]
    host = example.HostConsumer(profile)
    replay = CausalReplay.from_continuation(
        host, continuation, state_type=example.HostConsumerState, state_codec=example.CODEC
    )
    assert replay.export_continuation().to_json().encode() + b"\n" == encoded
    assert host.export_state().fingerprint == continuation.host_state.fingerprint


def test_chosen_wire_golden_preserves_two_pending_decisions_and_the_actual_lagged_application(example):
    continuation = read_continuation(
        (FIXTURES / "chosen-continuation.json").read_bytes(), state_type=example.HostConsumerState
    )
    assert [chosen(row) for row in continuation.state.decisions] == ["suite-a", "suite-a", "suite-b"]
    assert len(continuation.host_state.pending) == 2
    application = continuation.host_state.applications[0]
    assert application.feedback.batch.delay_rounds == 1
    assert application.eligible_at == example.instant(2)
    assert application.apply_at == example.instant(3)


def test_component_wire_golden_retains_partial_root_and_complete_successor_without_a_second_round(example):
    continuation = read_continuation(
        (FIXTURES / "component-continuation.json").read_bytes(), state_type=example.HostConsumerState
    )
    assert len(continuation.state.feedback) == 2
    assert continuation.host_state.update_count == 1
    assert continuation.host_state.applications[0].feedback.batch.revision_id == "a-complete"
    assert continuation.host_state.observed_counts == (1, 1, 0)
    assert continuation.host_state.loss_sums == (0.5, 0.0, 0.0)


def test_full_panel_wire_golden_preserves_all_three_per_value_evidence_origins(example):
    continuation = read_continuation(
        (FIXTURES / "full_panel-continuation.json").read_bytes(), state_type=example.HostConsumerState
    )
    evidence = continuation.host_state.applications[0].feedback.origins
    assert tuple(item.origin for item in evidence) == ("realized", "modeled_counterfactual", "synthetic")
    assert evidence[0].producer is None
    assert evidence[1].producer == example.GENERATOR
    assert evidence[2].producer == example.GENERATOR
    assert continuation.host_state.loss_sums == (0.0, 0.5, 1.0)
