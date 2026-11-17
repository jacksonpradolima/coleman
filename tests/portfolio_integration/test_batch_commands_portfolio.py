"""Head-bound commands obtain their period frontier from real staged state."""

import json
from hashlib import sha256
from typing import cast

import pytest

from coleman.state.local import LocalStateStore
from examples.synthetic_portfolio import model
from tests.portfolio_integration.test_task_state_portfolio import ObservedStore, advance_one
from tests.portfolio_integration.test_task_state_portfolio import real_layout as real_layout


def test_batch_binds_exact_head_and_observes_one_complete_postcommit_state(real_layout):
    layout, store = real_layout
    original = store.read_head()
    observed = ObservedStore(store)
    periods = []
    progress = layout.advance(cast(LocalStateStore, observed), lambda: periods.append("guarded-period"))
    assert observed.loads == 1
    assert len(periods) == progress.frontier.position == 8
    expected = {
        "command": "portfolio-staged-batch-v2",
        "expected": original.model_dump(mode="json"),
        "trajectory": layout.trajectory.fingerprint,
        "source": layout.context.preparation.source.model_dump(mode="json"),
        "configuration": layout.configuration.fingerprint,
        "maximum_periods": 8,
    }
    raw = json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert progress.proof.request.intent.content_fingerprint == "sha256:" + sha256(raw).hexdigest()
    assert progress.proof.request.expected == original

    def forbidden(_stage):
        raise AssertionError("An acknowledged original batch cannot observe its periods again")

    retained = store.commit_transition(
        progress.proof.request,
        (layout.binding,),
        forbidden,
        compatibility=layout.context.preparation.compatibility,
    )
    assert retained.status == "retained"
    assert retained.current_head == progress.proof.receipt.current_head
    reconciled = layout.reconcile(store)
    assert reconciled.frontier.position == 8
    assert reconciled.proof.request.to_json() == progress.proof.request.to_json()
    assert reconciled.proof.receipt.status == "retained"


def test_foreign_complete_definition_with_identical_bindings_refuses_before_staging(real_layout, tmp_path):
    layout, _store = real_layout
    foreign = layout.definition.model_copy(update={"reference": model.ref("different-collection-definition")})
    store = LocalStateStore.create(tmp_path / "foreign-store", foreign, layout.context.preparation.compatibility)
    layout.initialize(store)
    original = store.read_head()
    assert original.definition_fingerprint != layout.definition.fingerprint
    observed = ObservedStore(store)
    calls = []
    with pytest.raises(ValueError, match="complete acknowledged collection definition"):
        layout.advance(cast(LocalStateStore, observed), lambda: calls.append("unexpected-body"))
    assert not calls and observed.loads == 0
    assert store.read_head() == original
    assert (
        store.load((layout.binding,), compatibility=layout.context.preparation.compatibility)
        .component(layout.binding.codec)
        .experiment.next_index
        == 0
    )


def test_competing_accepted_head_is_not_silently_substituted_before_our_batch(real_layout):
    layout, store = real_layout
    original = store.read_head()
    observed = ObservedStore(store, before_receipt=lambda: advance_one(layout, store.root, 0))
    calls = []
    with pytest.raises(ValueError, match="[Ss]tale|predecessor|expected|generation"):
        layout.advance(cast(LocalStateStore, observed), lambda: calls.append("unexpected-body"))
    assert not calls and observed.loads == 0
    assert store.read_head() != original
    assert layout.checkpoint(store).experiment.next_index == 1
    # Reconciliation still reissues the actual original accepted one-period
    # command, rather than relabeling it as the new eight-period command.
    reconciled = layout.reconcile(store)
    assert reconciled.frontier.position == 1
    assert reconciled.proof.receipt.status == "retained"
    assert reconciled.proof.request.intent.content_fingerprint != original.manifest_fingerprint


def test_terminal_staged_batch_refuses_without_another_ack_or_guard(real_layout):
    layout, store = real_layout
    assert layout.advance(store, lambda: None).frontier.position == 8
    assert layout.advance(store, lambda: None).frontier.position == 16
    original = store.read_head()
    calls = []
    with pytest.raises(ValueError, match="terminal original period"):
        layout.advance(store, lambda: calls.append("unexpected-period"))
    assert not calls and store.read_head() == original
    assert layout.checkpoint(store).experiment.next_index == 16
