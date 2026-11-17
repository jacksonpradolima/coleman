"""Combined advancement joins actual acceptance, retained SQL and full state."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace

import pytest

from coleman.core.primitives import fingerprint
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import StorageLimits
from examples.synthetic_portfolio.experiment import Experiment
from examples.synthetic_portfolio.recipe import binding
from tests.portfolio_integration.test_acquired_snapshot_portfolio import _acquisitions
from tests.portfolio_integration.test_task_state_portfolio import advance_one
from tests.portfolio_integration.test_task_state_portfolio import real_layout as real_layout


@contextmanager
def _actual_commits(store, *, before=None, after=None):
    """Observe original public SQL operations without replacing their methods."""
    code = LocalStateStore.commit_transition.__code__
    requests, receipts = [], []
    tool = next(index for index in (4, 3, 2, 5, 0, 1) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "portfolio-advance-receipt-test")

    def started(_code, _offset):
        arguments = sys._getframe(1).f_locals
        if arguments["self"] is not store:
            return
        requests.append(arguments["request"])
        if before is not None:
            before(
                len(requests),
                arguments["request"],
                arguments["bindings"],
                arguments["transition"],
                arguments["compatibility"],
            )

    def returned(_code, _offset, receipt):
        if sys._getframe(1).f_locals["self"] is not store:
            return
        receipts.append(receipt)
        if after is not None:
            after(len(receipts), receipt)

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, returned)
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN)
        yield requests, receipts
    finally:
        sys.monitoring.free_tool_id(tool)


def _competing_period(layout, store, position):
    """Interleave a real independent client at the selected operation boundary.

    The public helper opens a separate store and commits through real SQL and
    staged math. This controlled interleaving makes no thread-latency claim;
    the store's ownership limit and the caller's aggregate bound still apply.
    """
    receipt = advance_one(layout, store.root, position)
    assert receipt.status == "new"
    return receipt


def test_new_and_real_retained_receipts_share_one_full_post_ack_acquisition(real_layout):
    layout, store = real_layout
    original = store.read_head()
    periods = []
    with _actual_commits(store) as (requests, receipts), _acquisitions() as (calls, returned):
        accepted, retained, checkpoint = layout.advance_with_checkpoint(store, lambda: periods.append("period"))
    assert len(periods) == 8
    assert calls == ["actual-load"]
    assert [receipt.status for receipt in receipts] == ["new", "retained"]
    assert requests[0].to_json() == requests[1].to_json()
    assert accepted.proof.request.to_json() == retained.proof.request.to_json() == requests[0].to_json()
    assert accepted.proof.receipt.to_json() == receipts[0].to_json()
    assert retained.proof.receipt.to_json() == receipts[1].to_json()
    assert accepted.frontier.position == retained.frontier.position == checkpoint.experiment.next_index == 8
    assert accepted.proof.request.expected == original
    assert accepted.proof.receipt.accepted_head == retained.proof.receipt.accepted_head
    assert retained.proof.receipt.accepted_head == retained.proof.receipt.current_head == store.read_head()
    prepared = returned[0]
    assert checkpoint.to_json() == prepared.component(layout.binding.codec).to_json()
    assert prepared.manifest.fingerprint == retained.proof.observed_head.manifest_fingerprint
    assert prepared.manifest.definition.to_json() == layout.definition.to_json()
    assert checkpoint.source == layout.context.preparation.source
    assert checkpoint.plan_fingerprint == retained.plan_fingerprint == layout.context.preparation.plan_fingerprint
    assert checkpoint.coordinate_fingerprint == retained.coordinate_fingerprint == layout.context.coordinate_fingerprint
    assert checkpoint.binding == layout.initial.binding
    assert layout.configuration.periods_per_commit == 8
    assert len(layout.trajectory.periods) == 16
    expected = Experiment()
    for index in range(8):
        expected.step(index)
    assert checkpoint.experiment.to_json() == expected.export_state().to_json()
    wire = checkpoint.to_json()
    object.__setattr__(checkpoint, "plan_fingerprint", fingerprint("unpassed-caller-mutation"))
    with _acquisitions() as (fresh, _snapshots):
        standalone = layout.checkpoint(store)
        first_progress = layout.progress(store, accepted.proof.request, accepted.proof.receipt)
        retained_progress = layout.reconcile(store)
    assert fresh == ["actual-load"] * 3
    assert standalone.to_json() == wire
    assert first_progress.proof.to_json() == accepted.proof.to_json()
    assert retained_progress.proof.to_json() == retained.proof.to_json()


def test_an_original_command_won_by_a_real_other_client_executes_its_body_only_once(real_layout):
    layout, store = real_layout
    guards, competing = [], []

    def before(index, request, bindings, transition, compatibility):
        if index != 1:
            return

        def accept_original_command():
            independent = LocalStateStore.open(store.root)
            return independent.commit_transition(request, bindings, transition, compatibility=compatibility)

        with ThreadPoolExecutor(max_workers=1) as executor:
            competing.append(executor.submit(accept_original_command).result(timeout=60))

    with _actual_commits(store, before=before) as (requests, receipts), _acquisitions() as (calls, _returned):
        accepted, retained, checkpoint = layout.advance_with_checkpoint(store, lambda: guards.append("actual-period"))
    assert competing[0].status == "new"
    assert len(guards) == 8
    assert [receipt.status for receipt in receipts] == ["retained", "retained"]
    assert requests[0].to_json() == requests[1].to_json()
    assert accepted.proof.receipt.accepted_head == competing[0].accepted_head
    assert retained.proof.request.to_json() == accepted.proof.request.to_json()
    assert calls == ["actual-load"]
    assert checkpoint.experiment.next_index == accepted.frontier.position == retained.frontier.position == 8
    assert retained.proof.observed_head == store.read_head()


def test_head_advanced_before_the_real_retry_requires_a_new_complete_current_read(real_layout):
    layout, store = real_layout

    def before(index, _request, _bindings, _transition, _compatibility):
        if index == 2:
            _competing_period(layout, store, 8)

    with _actual_commits(store, before=before) as (requests, receipts), _acquisitions() as (calls, returned):
        accepted, retained, checkpoint = layout.advance_with_checkpoint(store, lambda: None)
    assert calls == ["actual-load", "actual-load"]
    assert requests[0].to_json() == requests[1].to_json()
    assert [receipt.status for receipt in receipts] == ["new", "retained"]
    assert accepted.frontier.position == 8
    assert retained.frontier.position == checkpoint.experiment.next_index == 9
    assert receipts[1].accepted_head == receipts[0].accepted_head
    assert receipts[1].current_head != receipts[1].accepted_head
    assert checkpoint.to_json() == returned[-1].component(layout.binding.codec).to_json()
    assert returned[-1].manifest.fingerprint == retained.proof.observed_head.manifest_fingerprint
    assert retained.proof.observed_head == store.read_head()


def test_head_advanced_after_the_real_retained_reply_refuses_the_old_snapshot(real_layout):
    layout, store = real_layout

    def after(index, _receipt):
        if index == 2:
            _competing_period(layout, store, 8)

    with (
        _actual_commits(store, after=after) as (_requests, receipts),
        _acquisitions() as (calls, _returned),
        pytest.raises(ValueError, match="actual current matching ACK"),
    ):
        layout.advance_with_checkpoint(store, lambda: None)
    assert [receipt.status for receipt in receipts] == ["new", "retained"]
    assert calls == ["actual-load"]
    assert layout.checkpoint(store).experiment.next_index == 9
    assert not (layout.context.artifact_directory / "coordinate-result.json").exists()


def test_actual_receipt_expiry_during_reconciliation_does_not_fabricate_retained_progress(real_layout, tmp_path):
    original, _unused = real_layout
    directory = tmp_path / "one-retained-receipt"
    directory.mkdir()
    layout = replace(original, context=replace(original.context, artifact_directory=directory))
    # A supported one-receipt window creates real bounded expiry after just one
    # competing generation. Original sixteen periods, eight-period batch,
    # byte/work/time limits and the actual SQLite profile remain unchanged.
    store = LocalStateStore.create(
        directory / "state",
        layout.definition,
        layout.context.preparation.compatibility,
        limits=StorageLimits(schema_version=1, receipt_window=1),
    )
    layout.initialize(store)

    def before(index, _request, _bindings, _transition, _compatibility):
        if index == 2:
            _competing_period(layout, store, 8)

    with (
        _actual_commits(store, before=before) as (requests, receipts),
        _acquisitions() as (calls, _returned),
        pytest.raises(ValueError, match="expired|stale"),
    ):
        layout.advance_with_checkpoint(store, lambda: None)
    assert requests[0].to_json() == requests[1].to_json()
    assert [receipt.status for receipt in receipts] == ["new"]
    assert calls == ["actual-load"]
    assert layout.checkpoint(store).experiment.next_index == 9
    current = store.read_head()
    assert current is not None
    assert current.generation == 2


def test_interrupted_actual_math_accepts_no_new_head_and_returns_no_snapshot(real_layout):
    layout, store = real_layout
    original = store.read_head()
    periods = []

    def guard():
        periods.append("actual-guard")
        if len(periods) == 2:
            raise InterruptedError("Explicit unit interruption before the second original period")

    with (
        _actual_commits(store) as (requests, receipts),
        _acquisitions() as (calls, _returned),
        pytest.raises(InterruptedError, match="Explicit unit interruption"),
    ):
        layout.advance_with_checkpoint(store, guard)
    assert len(periods) == 2
    assert len(requests) == 1
    assert receipts == []
    assert calls == []
    assert store.read_head() == original
    assert layout.checkpoint(store).experiment.next_index == 0


def test_corrupt_data_from_a_previous_return_cannot_be_hidden_by_cross_call_reuse(real_layout):
    layout, store = real_layout
    with _acquisitions() as (_calls, returned):
        _accepted, _retained, checkpoint = layout.advance_with_checkpoint(store, lambda: None)
    before = checkpoint.to_json()
    head = store.read_head()
    component = returned[-1].manifest.components[0]
    path = store.root / "blobs" / (component.fingerprint[7:] + ".json")
    content = path.read_bytes()
    path.write_bytes(content + b" ")
    try:
        with _actual_commits(store) as (requests, receipts), pytest.raises(ValueError):
            layout.advance_with_checkpoint(store, lambda: None)
        assert requests == []
        assert receipts == []
        assert checkpoint.to_json() == before
    finally:
        path.write_bytes(content)
    assert store.read_head() == head
    assert layout.checkpoint(store).to_json() == before


def test_foreign_complete_source_binding_refuses_before_any_new_or_retained_command(real_layout):
    layout, store = real_layout
    context = replace(
        layout.context,
        preparation=replace(layout.context.preparation, source=binding("foreign-source", {"rows": "different"})),
    )
    foreign = replace(layout, context=context)
    original = store.read_head()
    with _actual_commits(store) as (requests, receipts), pytest.raises(ValueError, match="collection definition"):
        foreign.advance_with_checkpoint(store, lambda: None)
    assert requests == []
    assert receipts == []
    assert store.read_head() == original
