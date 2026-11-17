"""Paired observations reuse one real acquisition without caching caller state."""

import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from typing import cast

import pytest

from coleman.core.primitives import fingerprint
from coleman.state.local import LocalStateStore
from examples.synthetic_portfolio import model
from examples.synthetic_portfolio.experiment import Experiment
from examples.synthetic_portfolio.recipe import binding, recipe
from tests.portfolio_integration.test_task_state_portfolio import ObservedStore, advance_one, make_layout
from tests.portfolio_integration.test_task_state_portfolio import real_layout as real_layout


@contextmanager
def _acquisitions(after_return=None):
    """Observe the actual unwrapped public store reader, including its result."""
    code = LocalStateStore.load.__code__
    calls, returned = [], []
    tool = next(index for index in (4, 3, 2, 5, 0, 1) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "portfolio-acquisition-test")

    def started(_code, _offset):
        calls.append("actual-load")

    def finished(_code, _offset, value):
        returned.append(value)
        if after_return is not None:
            after_return(value)

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, finished)
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN)
        yield calls, returned
    finally:
        sys.monitoring.free_tool_id(tool)


def _advance_independently(layout, store, position):
    with ThreadPoolExecutor(max_workers=1) as executor:
        receipt = executor.submit(advance_one, layout, store.root, position).result(timeout=10)
    assert receipt.status == "new"
    return receipt


def _checkpoint(layout, store, route):
    if route == "open":
        opened, checkpoint = layout.open_with_checkpoint()
        assert opened is not None
        assert opened.root == store.root
    else:
        _progress, checkpoint = layout.reconcile_with_checkpoint(store)
    assert checkpoint is not None
    return checkpoint


@pytest.mark.parametrize("position", [0, 1])
def test_open_pair_returns_one_actual_checked_acquisition_and_every_full_field(real_layout, position):
    layout, store = real_layout
    if position:
        advance_one(layout, store.root, 0)
    with _acquisitions() as (calls, returned):
        opened, checkpoint = layout.open_with_checkpoint()
    assert calls == ["actual-load"]
    assert opened is not None
    assert checkpoint is not None
    actual = returned[0]
    assert checkpoint.to_json() == actual.component(layout.binding.codec).to_json()
    assert actual.manifest.fingerprint == opened.read_head().manifest_fingerprint
    assert actual.manifest.definition.to_json() == layout.definition.to_json()
    assert checkpoint.experiment.next_index == position
    assert checkpoint.source == layout.context.preparation.source
    assert checkpoint.plan_fingerprint == layout.context.preparation.plan_fingerprint
    assert checkpoint.coordinate_fingerprint == layout.context.coordinate_fingerprint
    assert checkpoint.binding == layout.initial.binding
    expected = Experiment()
    for index in range(position):
        expected.step(index)
    assert checkpoint.experiment.to_json() == expected.export_state().to_json()
    assert layout.configuration.periods_per_commit == 8
    assert recipe().reference.version == "2"


@pytest.mark.parametrize("position", [0, 1])
def test_retained_pair_joins_one_actual_full_snapshot_to_original_sql_receipt(real_layout, position):
    layout, store = real_layout
    if position:
        advance_one(layout, store.root, 0)
    with _acquisitions() as (calls, returned):
        progress, checkpoint = layout.reconcile_with_checkpoint(store)
    assert calls == ["actual-load"]
    prepared = returned[0]
    assert checkpoint.to_json() == prepared.component(layout.binding.codec).to_json()
    assert progress.frontier.position == checkpoint.experiment.next_index == position
    assert progress.proof.receipt.status == "retained"
    assert progress.proof.receipt.accepted_head == progress.proof.receipt.current_head
    assert progress.proof.observed_head == store.read_head()
    assert progress.proof.observed_head.manifest_fingerprint == prepared.manifest.fingerprint
    assert progress.proof.collection.definition_fingerprint == prepared.manifest.definition.fingerprint
    assert progress.proof.plan_fingerprint == checkpoint.plan_fingerprint
    assert progress.proof.coordinate_fingerprint == checkpoint.coordinate_fingerprint
    assert checkpoint.source == layout.context.preparation.source
    with _acquisitions() as (fresh_calls, _fresh):
        standalone = layout.checkpoint(store)
        proof_only = layout.reconcile(store)
        opened = layout.open()
    assert fresh_calls == ["actual-load"] * 3
    assert standalone.to_json() == checkpoint.to_json()
    assert proof_only.proof.to_json() == progress.proof.to_json()
    assert opened is not None


def test_explicit_absence_and_virgin_creation_never_invent_an_acknowledged_snapshot(real_layout, tmp_path):
    layout, _store = real_layout
    virgin = replace(layout, context=replace(layout.context, artifact_directory=tmp_path / "virgin-owned"))
    with _acquisitions() as (calls, _returned):
        absent = virgin.open_with_checkpoint()
        created, checkpoint = virgin.open_with_checkpoint(create=True)
    assert absent == (None, None)
    assert created is not None
    assert checkpoint is None
    assert calls == []
    assert created.read_head() is None
    virgin.initialize(created)
    with _acquisitions() as (calls, _returned):
        _opened, initialized = virgin.open_with_checkpoint()
    assert calls == ["actual-load"]
    assert initialized is not None
    assert initialized.experiment.next_index == 0


@pytest.mark.parametrize("route", ["open", "retained"])
def test_artifact_corrupted_after_a_return_is_not_hidden_by_later_operation_cache(real_layout, route):
    layout, store = real_layout
    with _acquisitions() as (calls, returned):
        checkpoint = _checkpoint(layout, store, route)
    assert calls == ["actual-load"]
    before = checkpoint.to_json()
    selected = returned[0].manifest.components[0]
    path = store.root / "blobs" / (selected.fingerprint[7:] + ".json")
    content, head = path.read_bytes(), store.read_head()
    path.write_bytes(content + b" ")
    try:
        with _acquisitions() as (fresh_calls, _fresh), pytest.raises(ValueError):
            _checkpoint(layout, store, route)
        # Opening first checks protected artifact bytes; its early refusal
        # precedes the selected model reader. An existing store enters load.
        assert fresh_calls == ([] if route == "open" else ["actual-load"])
        with pytest.raises(ValueError):
            layout.checkpoint(store)
        assert checkpoint.to_json() == before
    finally:
        path.write_bytes(content)
    assert store.read_head() == head


@pytest.mark.parametrize("route", ["open", "retained"])
def test_mutated_unpassed_caller_snapshot_cannot_supply_the_next_acquisition(real_layout, route):
    layout, store = real_layout
    first = _checkpoint(layout, store, route)
    expected = first.to_json()
    object.__setattr__(first, "plan_fingerprint", "sha256:" + "0" * 64)
    with _acquisitions() as (calls, _returned):
        second = _checkpoint(layout, store, route)
    assert calls == ["actual-load"]
    assert second.to_json() == expected
    assert second is not first


def test_paired_open_refuses_a_real_head_advance_after_its_actual_selected_load(real_layout):
    layout, store = real_layout
    original = store.read_head()
    advanced = []

    def after_load(_prepared):
        if not advanced:
            advanced.append("advance-started")
            _advance_independently(layout, store, 0)

    with _acquisitions(after_load) as (calls, _returned), pytest.raises(ValueError, match="matching head"):
        layout.open_with_checkpoint()
    assert calls == ["actual-load"]
    assert store.read_head() != original
    assert layout.checkpoint(store).experiment.next_index == 1
    opened, current = layout.open_with_checkpoint()
    assert opened is not None
    assert current is not None
    assert current.experiment.next_index == 1


def test_real_retained_response_gap_reloads_and_pairs_the_newer_complete_state(real_layout):
    layout, store = real_layout
    original = store.read_head()
    observed = ObservedStore(store, before_receipt=lambda: _advance_independently(layout, store, 0))
    with _acquisitions() as (calls, returned):
        progress, checkpoint = layout.reconcile_with_checkpoint(cast(LocalStateStore, observed))
    assert calls == ["actual-load", "actual-load"]
    assert progress.proof.receipt.status == "retained"
    assert progress.proof.receipt.accepted_head == original
    assert progress.proof.receipt.current_head == store.read_head()
    assert progress.proof.receipt.current_head != original
    assert progress.proof.observed_head.manifest_fingerprint == returned[-1].manifest.fingerprint
    assert checkpoint.to_json() == returned[-1].component(layout.binding.codec).to_json()
    assert checkpoint.experiment.next_index == progress.frontier.position == 1
    assert layout.checkpoint(store).to_json() == checkpoint.to_json()


def test_real_advance_after_retained_response_cannot_return_old_state_with_new_head(real_layout):
    layout, store = real_layout
    original = store.read_head()
    observed = ObservedStore(store, before_head=lambda: _advance_independently(layout, store, 0))
    with _acquisitions() as (calls, _returned), pytest.raises(ValueError, match="current matching ACK"):
        layout.reconcile_with_checkpoint(cast(LocalStateStore, observed))
    assert calls == ["actual-load"]
    assert store.read_head() != original
    progress, checkpoint = layout.reconcile_with_checkpoint(store)
    assert progress.frontier.position == checkpoint.experiment.next_index == 1
    assert progress.proof.observed_head == store.read_head()


@pytest.mark.parametrize("field", ["source", "plan", "coordinate"])
def test_paired_acquisition_rejects_foreign_full_execution_bindings(real_layout, field):
    layout, store = real_layout
    context = layout.context
    if field == "source":
        context = replace(
            context,
            preparation=replace(context.preparation, source=binding("foreign-source", {"rows": "different"})),
        )
    elif field == "plan":
        context = replace(context, preparation=replace(context.preparation, plan_fingerprint=fingerprint("foreign")))
    else:
        context = replace(context, coordinate_fingerprint=fingerprint("foreign-coordinate"))
    foreign = replace(layout, context=context)
    original = store.read_head()
    with pytest.raises(ValueError):
        foreign.reconcile_with_checkpoint(store)
    with pytest.raises(ValueError):
        foreign.open_with_checkpoint()
    assert store.read_head() == original


def test_full_foreign_manifest_is_not_a_checked_snapshot_even_with_identical_owner_bindings(tmp_path):
    layout, _store = make_layout(tmp_path / "original")
    selected = replace(layout, context=replace(layout.context, artifact_directory=tmp_path / "foreign-owned"))
    selected.context.artifact_directory.mkdir()
    definition = selected.definition.model_copy(update={"reference": model.ref("foreign-complete-collection")})
    store = LocalStateStore.create(
        selected.context.artifact_directory / "state", definition, selected.context.preparation.compatibility
    )
    selected.initialize(store)
    original = store.read_head()
    with _acquisitions() as (calls, _returned), pytest.raises(ValueError, match="complete collection definition"):
        selected.open_with_checkpoint()
    assert calls == ["actual-load"]
    assert not (selected.context.artifact_directory / "owned-definition.json").exists()
    with pytest.raises(ValueError, match="complete collection definition"):
        selected.reconcile_with_checkpoint(store)
    assert store.read_head() == original
