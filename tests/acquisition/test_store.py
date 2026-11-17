"""Real acknowledged continuation, interrupted writes and concurrent delivery."""

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from coleman.acquisition.contracts import AcquisitionReceipt, AllocationState
from coleman.acquisition.engine import inspect_allocation
from coleman.artifacts import ArtifactWriter
from coleman.state.components import ComponentValue
from coleman.state.local import LocalStateStore


@pytest.fixture(scope="module")
def checkpoint_example():
    path = Path(__file__).parents[2] / "examples/acquisition/checkpoint.py"
    spec = importlib.util.spec_from_file_location("acquisition_checkpoint_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def accepted(ci_pool, checkpoint_example, tmp_path):
    example = checkpoint_example
    plan = ci_pool.make_plan()
    definition, bindings = example.layout(plan)
    compatibility = example.compatibility()
    store = LocalStateStore.create(tmp_path / "state", definition, compatibility)
    state = AllocationState(schema_version=1, plan=plan)
    request = example.command(0, None, state.fingerprint, initialize=True)
    initialized = store.initialize(
        request, (ComponentValue(bindings[0].codec, state),), bindings, compatibility=compatibility
    )
    return store, bindings, compatibility, initialized


def test_actual_store_restart_during_increment_and_elimination(ci_pool, checkpoint_example, accepted):
    store, bindings, compatibility, acknowledged = accepted
    pool = ci_pool.CIPool()
    while True:
        # Open and decode acknowledged source-bound state on every acquisition.
        store = LocalStateStore.open(store.root)
        state = store.load(bindings, compatibility=compatibility).component(bindings[0].codec)
        purchase = inspect_allocation(state).next_purchase
        if purchase is None:
            break
        prior = next(
            (row for row in reversed(state.receipts) if row.purchase.action.chosen == purchase.action.chosen), None
        )
        receipt = pool.acquire(purchase, state.plan, prior)
        request = checkpoint_example.command(len(state.receipts) + 1, acknowledged.current_head, receipt.fingerprint)
        callback_calls = []

        def transition(stage, calls=callback_calls, item=receipt):
            calls.append("accepted-only-once")
            stage.owner(bindings[0]).accept(item)

        acknowledged = store.commit_transition(request, bindings, transition, compatibility=compatibility)
        retained = store.commit_transition(request, bindings, transition, compatibility=compatibility)
        assert retained.status == "retained"
        assert callback_calls == ["accepted-only-once"]
    assert inspect_allocation(state).known_charge.value == 8
    report = inspect_allocation(state)
    assert report.winner is not None
    assert report.winner.id == "A"
    assert len(pool.reads) == 8


def test_interrupted_blob_write_leaves_previous_accepted_head_and_no_false_completion(
    ci_pool,
    checkpoint_example,
    accepted,
    monkeypatch,
):
    store, bindings, compatibility, initialized = accepted
    state = store.load(bindings, compatibility=compatibility).component(bindings[0].codec)
    receipt = ci_pool.CIPool().acquire(inspect_allocation(state).next_purchase, state.plan)
    request = checkpoint_example.command(1, initialized.current_head, receipt.fingerprint)
    writer = ArtifactWriter.write_immutable_stream

    def interrupt(self, path, chunks, **kwargs):
        def partial_then_interrupt():
            chunk = next(iter(chunks))
            yield chunk[: max(1, len(chunk) // 2)]
            raise KeyboardInterrupt("interrupted-real-acquisition-state-write")

        return writer(self, path, partial_then_interrupt(), **kwargs)

    with monkeypatch.context() as scope:
        scope.setattr(ArtifactWriter, "write_immutable_stream", interrupt)
        with pytest.raises(KeyboardInterrupt, match="interrupted-real"):
            store.commit_transition(
                request,
                bindings,
                lambda stage: stage.owner(bindings[0]).accept(receipt),
                compatibility=compatibility,
            )
    reopened = LocalStateStore.open(store.root)
    assert reopened.read_head() == initialized.current_head
    original = reopened.load(bindings, compatibility=compatibility).component(bindings[0].codec)
    assert original == state
    assert inspect_allocation(original).status == "running"
    new = reopened.commit_transition(
        request,
        bindings,
        lambda stage: stage.owner(bindings[0]).accept(receipt),
        compatibility=compatibility,
    )
    assert new.current_head.generation == 1
    assert (
        inspect_allocation(
            reopened.load(bindings, compatibility=compatibility).component(bindings[0].codec)
        ).known_charge.value
        == 1
    )


@pytest.mark.parametrize("conflicting", [False, True])
def test_actual_concurrent_writers_accept_one_charge_and_reconcile_exact_delivery(
    ci_pool,
    checkpoint_example,
    accepted,
    conflicting,
):
    store, bindings, compatibility, initialized = accepted
    state = store.load(bindings, compatibility=compatibility).component(bindings[0].codec)
    receipt = ci_pool.CIPool().acquire(inspect_allocation(state).next_purchase, state.plan)
    alternate = (
        AcquisitionReceipt.model_validate(receipt.model_copy(update={"delay_steps": 1})) if conflicting else receipt
    )
    barrier = Barrier(2)
    calls = []

    def commit(item):
        reopened = LocalStateStore.open(store.root)
        request = checkpoint_example.command(1, initialized.current_head, item.fingerprint)
        barrier.wait(timeout=10)

        def transition(stage):
            calls.append(item.fingerprint)
            stage.owner(bindings[0]).accept(item)

        try:
            return reopened.commit_transition(request, bindings, transition, compatibility=compatibility).status
        except ValueError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(commit, (receipt, alternate)))
    assert sorted(results) == (["conflict", "new"] if conflicting else ["new", "retained"])
    assert len(calls) == 1
    final = store.load(bindings, compatibility=compatibility).component(bindings[0].codec)
    assert len(final.receipts) == 1
    assert inspect_allocation(final).known_charge.value == 1
