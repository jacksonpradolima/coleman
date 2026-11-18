"""Opening and resolving share actual evidence only inside one operation."""

import sys
from contextlib import contextmanager
from dataclasses import replace

import pytest

from coleman.state.local import LocalStateStore
from tests.portfolio_integration.test_compact_results_portfolio import complete_evidence as complete_evidence
from tests.portfolio_integration.test_task_state_portfolio import advance_one, make_layout


@contextmanager
def _actual_loads(root, *, before_retained=None):
    """Observe original code and optionally interleave a real second client."""
    load = LocalStateStore.load.__code__
    commands = {LocalStateStore.initialize.__code__, LocalStateStore.commit_transition.__code__}
    calls = []
    action = before_retained
    interleaving = False
    tool = next(index for index in (4, 3, 2, 5, 0, 1) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "portfolio-opened-evidence-test")

    def started(code, _offset):
        nonlocal action, interleaving
        instance = sys._getframe(1).f_locals["self"]
        if instance.root != root or interleaving:
            return
        if code is load:
            calls.append("actual-selected-load")
        elif action is not None:
            callback, action = action, None
            interleaving = True
            try:
                callback()
            finally:
                interleaving = False

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
        for code in (load, *commands):
            sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
        yield calls
    finally:
        sys.monitoring.free_tool_id(tool)


def _absent_layout(layout, path):
    return replace(layout, context=replace(layout.context, artifact_directory=path))


def test_open_reconciled_distinguishes_absence_and_unacknowledged_virgin_store(tmp_path):
    layout, _ = make_layout(tmp_path / "original")
    absent = _absent_layout(layout, tmp_path / "absent")
    assert absent.open_reconciled() == (None, None, None)
    assert not absent.context.artifact_directory.exists()
    virgin = absent.open(create=True)
    assert virgin is not None and virgin.read_head() is None
    with _actual_loads(virgin.root) as calls:
        reopened, progress, checkpoint = absent.open_reconciled()
    assert reopened is not None and reopened.root == virgin.root
    assert progress is checkpoint is None
    assert calls == []
    assert virgin.read_head() is None


@pytest.mark.parametrize("position", [0, 1])
def test_open_reconciled_joins_one_full_read_to_real_original_retained_receipt(tmp_path, position):
    layout, store = make_layout(tmp_path)
    if position:
        assert advance_one(layout, store.root, 0).status == "new"
    before = store.read_head()
    with _actual_loads(store.root) as calls:
        reopened, progress, checkpoint = layout.open_reconciled()
    assert calls == ["actual-selected-load"]
    assert reopened is not None and progress is not None and checkpoint is not None
    assert progress.proof.receipt.status == "retained"
    assert progress.proof.receipt.accepted_head == progress.proof.observed_head == before
    assert progress.frontier.position == checkpoint.experiment.next_index == position
    independent = layout.reconcile(store)
    assert progress.proof.to_json() == independent.proof.to_json()
    assert progress.frontier == independent.frontier
    assert checkpoint.to_json() == layout.checkpoint(store).to_json()
    with _actual_loads(store.root) as calls:
        again = layout.open_reconciled()
        third = layout.open_reconciled()
    assert len(calls) == 2  # An earlier read never certifies a later operation.
    assert again[1] is not None and third[1] is not None
    assert again[1].proof.to_json() == third[1].proof.to_json() == progress.proof.to_json()
    assert store.read_head() == before


@pytest.mark.parametrize("position", [0, 1])
def test_open_reconciled_reloads_after_actual_concurrent_advance(tmp_path, position):
    layout, store = make_layout(tmp_path)
    if position:
        assert advance_one(layout, store.root, 0).status == "new"
    before = store.read_head()
    bodies = []

    def competing_client():
        assert advance_one(layout, store.root, position).status == "new"
        bodies.append("actual-independent-transition")

    with _actual_loads(store.root, before_retained=competing_client) as calls:
        reopened, progress, checkpoint = layout.open_reconciled()
    assert bodies == ["actual-independent-transition"]
    assert reopened is not None and progress is not None and checkpoint is not None
    assert len(calls) == 2
    assert progress.proof.receipt.status == "retained"
    assert progress.proof.receipt.accepted_head == before
    assert progress.proof.observed_head == store.read_head() != before
    assert progress.frontier.position == checkpoint.experiment.next_index == position + 1
    assert checkpoint.to_json() == layout.checkpoint(store).to_json()


def test_open_resolved_keeps_all_terminal_evidence_with_one_actual_selected_read(complete_evidence):
    layout, store, result, expected = complete_evidence
    before = store.read_head()
    with _actual_loads(store.root) as calls:
        reopened, checkpoint = layout.open_resolved(result)
    assert calls == ["actual-selected-load"]
    assert reopened.root == store.root
    assert checkpoint.to_json() == expected.to_json()
    assert checkpoint.experiment.next_index == 16
    assert len(checkpoint.experiment.observations) == 15
    assert checkpoint.to_json() == layout.resolve_result(store, result).to_json()
    assert store.read_head() == before


def test_open_resolved_refuses_false_result_identity_and_missing_evidence(complete_evidence, tmp_path):
    layout, store, result, _ = complete_evidence
    before = store.read_head()
    changed = result.model_copy(update={"experiment_fingerprint": "sha256:" + "0" * 64})
    with pytest.raises(ValueError):
        layout.open_resolved(changed)
    assert store.read_head() == before
    absent = _absent_layout(layout, tmp_path / "missing-owned-evidence")
    with pytest.raises(ValueError):
        absent.open_resolved(result)
    assert not absent.context.artifact_directory.exists()
    assert store.read_head() == before
