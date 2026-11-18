"""Operation-scoped inspection of actual complete funded checkpoints and ACKs."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from coleman.core.primitives import fingerprint
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CommitIntent, CommitRequest, ExecutionCompatibility
from coleman.tasks.contracts import TaskFrontier
from coleman.tasks.protocols import CoordinateContext, PreparationContext
from examples.synthetic_portfolio import _state, model
from examples.synthetic_portfolio._state import Layout
from examples.synthetic_portfolio.experiment import Experiment
from examples.synthetic_portfolio.recipe import binding, frontier_definition
from examples.synthetic_portfolio.task_contracts import PortfolioConfiguration, PortfolioTrajectory


def make_layout(tmp_path):
    """Initialize a real store; declared test ownership adds no task approval."""
    configuration = PortfolioConfiguration(schema_version=1, strength=0.05)
    trajectory = PortfolioTrajectory(
        schema_version=1,
        reference=model.ref("ack-inspection-trajectory"),
        initial_state=Experiment().export_state(),
        rows=model.PRICE_ROWS,
    )
    implementation = binding(
        "actual-public-portfolio-test-owner",
        {"source_sha256": sha256(Path(_state.__file__).read_bytes()).hexdigest()},
    )
    compatibility = ExecutionCompatibility(
        schema_version=1,
        runtime=SelectedInstalledRuntime.capture(extras=()),
        owner_implementations=(implementation,),
    )
    context = CoordinateContext(
        preparation=PreparationContext(
            plan_fingerprint=fingerprint({"authored_test": "operation-scoped-full-state-inspection"}),
            source=binding("original-authored-test-tape", {"rows": [list(row) for row in model.PRICE_ROWS]}),
            implementation=implementation,
            compatibility=compatibility,
            job_root=tmp_path,
        ),
        coordinate_fingerprint=trajectory.fingerprint,
        reference=trajectory.reference,
        target=TaskFrontier(profile=frontier_definition(), coordinate=trajectory.reference, position=16),
        artifact_directory=tmp_path / "actual-owned-state",
    )
    layout = Layout(configuration, trajectory, context)
    store = layout.open(create=True)
    assert store is not None
    layout.initialize(store)
    return layout, store


@pytest.fixture
def real_layout(tmp_path):
    """Supply one independent actual initialized store for each inspection case."""
    return make_layout(tmp_path)


class ObservedStore:
    """Count full public loads and interleave real clients at exact boundaries."""

    def __init__(self, store, *, before_receipt=None, before_head=None):
        self.store = store
        self.loads = 0
        self.before_receipt = before_receipt
        self.before_head = before_head

    def load(self, *arguments, **keywords):
        self.loads += 1
        return self.store.load(*arguments, **keywords)

    def _receipt_hook(self):
        callback, self.before_receipt = self.before_receipt, None
        if callback is not None:
            callback()

    def initialize(self, *arguments, **keywords):
        self._receipt_hook()
        return self.store.initialize(*arguments, **keywords)

    def commit_transition(self, *arguments, **keywords):
        self._receipt_hook()
        return self.store.commit_transition(*arguments, **keywords)

    def read_head(self):
        callback, self.before_head = self.before_head, None
        if callback is not None:
            callback()
        return self.store.read_head()


def advance_one(layout, root, position):
    """Perform real one-period math on a separately opened actual store client."""
    independent = LocalStateStore.open(root)
    selected = layout.binding
    head = independent.read_head()
    assert head is not None
    request = CommitRequest(
        schema_version=1,
        expected=head,
        intent=CommitIntent(
            schema_version=1,
            generation=head.generation + 1,
            kind="transition",
            content_fingerprint=fingerprint({"original_period": position}),
        ),
    )
    return independent.commit_transition(
        request,
        (selected,),
        lambda stage: stage.owner(selected).experiment.step(position),
        compatibility=layout.context.preparation.compatibility,
    )


@pytest.mark.parametrize("position", [0, 1])
def test_reconcile_reuses_one_complete_operation_load_and_matches_full_public_progress(real_layout, position):
    layout, store = real_layout
    if position:
        advance_one(layout, store.root, 0)
    observed = ObservedStore(store)
    reconciled = layout.reconcile(cast(LocalStateStore, observed))
    assert observed.loads == 1
    assert reconciled.frontier.position == position
    assert reconciled.proof.receipt.status == "retained"
    assert reconciled.proof.receipt.accepted_head == reconciled.proof.receipt.current_head
    # The public alternative deliberately performs another complete codec read.
    full = layout.progress(store, reconciled.proof.request, reconciled.proof.receipt)
    assert full.proof.to_json() == reconciled.proof.to_json()
    assert full.frontier.model_dump(mode="json") == reconciled.frontier.model_dump(mode="json")
    assert layout.reconcile(cast(LocalStateStore, observed)).proof.to_json() == reconciled.proof.to_json()
    assert observed.loads == 2  # Reuse never escapes the individual operation.


@pytest.mark.parametrize("position", [0, 1])
def test_real_competing_advance_forces_complete_current_state_reload(real_layout, position):
    layout, store = real_layout
    if position:
        advance_one(layout, store.root, 0)
    original = store.read_head()

    def competing_advance():
        with ThreadPoolExecutor(max_workers=1) as executor:
            receipt = executor.submit(advance_one, layout, store.root, position).result()
        assert receipt.status == "new"

    observed = ObservedStore(store, before_receipt=competing_advance)
    progress = layout.reconcile(cast(LocalStateStore, observed))
    assert observed.loads == 2
    assert progress.proof.receipt.status == "retained"
    assert progress.proof.receipt.accepted_head == original
    assert progress.proof.receipt.current_head == store.read_head()
    assert progress.proof.receipt.current_head != original
    assert progress.frontier.position == position + 1
    assert layout.checkpoint(store).experiment.next_index == position + 1
    full = layout.progress(store, progress.proof.request, progress.proof.receipt)
    assert full.proof.to_json() == progress.proof.to_json()


def test_real_advance_after_retained_receipt_refuses_stale_progress_at_fresh_head(real_layout):
    layout, store = real_layout

    def advance_before_final_observation():
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(advance_one, layout, store.root, 0).result().status == "new"

    observed = ObservedStore(store, before_head=advance_before_final_observation)
    with pytest.raises(ValueError, match="actual current matching ACK"):
        layout.reconcile(cast(LocalStateStore, observed))
    assert observed.loads == 1
    assert layout.checkpoint(store).experiment.next_index == 1
    fresh = layout.reconcile(store)
    assert fresh.frontier.position == 1
    assert fresh.proof.observed_head == store.read_head()
