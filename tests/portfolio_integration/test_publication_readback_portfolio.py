"""Actual publication/readback owns one complete acquisition and no later cache."""

import shutil
import sys
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CommitIntent, CommitRequest, ExecutionCompatibility
from coleman.tasks.contracts import TaskFrontier
from coleman.tasks.protocols import CoordinateContext, PreparationContext, decode_value
from examples.synthetic_portfolio._state import Layout
from examples.synthetic_portfolio.executor import PortfolioExecutor
from examples.synthetic_portfolio.recipe import CONFIGURATION, INPUTS, binding, frontier_definition
from examples.synthetic_portfolio.sample import planning_request
from tests.portfolio_integration.test_opened_evidence_portfolio import _actual_loads


def _declared(path):
    """Prepare the real sample values without starting a worker or writing files."""
    request = planning_request()
    configuration = decode_value(CONFIGURATION, request.configuration)
    inputs = decode_value(INPUTS, request.inputs)
    coordinate = inputs.trajectories[0]
    implementation = binding(
        "actual-publication-unit-owner", {"source": sha256(Path(__file__).read_bytes()).hexdigest()}
    )
    context = PreparationContext(
        plan_fingerprint=fingerprint({"actual-unit-plan": request.model_dump(mode="json")}),
        source=inputs.binding,
        implementation=implementation,
        compatibility=ExecutionCompatibility(
            schema_version=1,
            runtime=SelectedInstalledRuntime.capture(extras=()),
            owner_implementations=(implementation,),
        ),
        job_root=path,
    )
    prepared = PortfolioExecutor().prepare(configuration, inputs, context)
    selected = CoordinateContext(
        context,
        coordinate.fingerprint,
        coordinate.reference,
        TaskFrontier(profile=frontier_definition(), coordinate=coordinate.reference, position=16),
        path / "actual-owned-coordinate",
    )
    return prepared, Layout(configuration, coordinate, selected)


@pytest.fixture(scope="module")
def complete_publication(tmp_path_factory):
    """Accept all real observations once; later copies are continuation fixtures."""
    prepared, layout = _declared(tmp_path_factory.mktemp("actual-publication-source"))
    store = layout.open(create=True)
    assert store is not None
    layout.initialize(store)
    head = store.read_head()
    assert head is not None
    selected = layout.binding
    request = CommitRequest(
        schema_version=1,
        expected=head,
        intent=CommitIntent(
            schema_version=1,
            generation=head.generation + 1,
            kind="transition",
            content_fingerprint=fingerprint({"actual-unit-observations": list(range(16))}),
        ),
    )

    def observe(stage):
        owner = stage.owner(selected)
        for index in range(16):
            owner.experiment.step(index)

    assert (
        store.commit_transition(
            request, (selected,), observe, compatibility=layout.context.preparation.compatibility
        ).status
        == "new"
    )
    return prepared, layout, layout.complete_result(store)


def _copy_evidence(complete_publication, tmp_path):
    """Reopen all inactive same-image files, never claim new observations."""
    prepared, original, result = complete_publication
    directory = tmp_path / "same-image-evidence-copy"
    shutil.copytree(original.context.artifact_directory, directory)
    layout = replace(original, context=replace(original.context, artifact_directory=directory))
    return prepared, layout, LocalStateStore.open(directory / "state"), result


@contextmanager
def _after_actual_publication(path, callback):
    """Observe the actual writer return, leaving every method unwrapped."""
    code = ArtifactWriter.write_immutable.__code__
    tool = next(index for index in (4, 3, 2, 5, 0, 1) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "actual-publication-interleaving")
    events = []

    def returned(_code, _offset, _value):
        frame = sys._getframe(1)
        if frame.f_locals["path"] == path and not events:
            events.append("actual-immutable-write-returned")
            callback()

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, returned)
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_RETURN)
        yield events
    finally:
        sys.monitoring.free_tool_id(tool)


def test_declared_publication_capability_is_lazy_pure_and_explicitly_versioned(tmp_path):
    root = tmp_path / "not-created"
    prepared, _layout = _declared(root)
    assert prepared.publication_readback is prepared
    assert prepared.profile == Reference(id="coleman-coordinate-publication-readback", version="1")
    assert not root.exists()


def test_real_handler_publishes_and_reads_one_full_acquisition_per_operation(complete_publication, tmp_path):
    prepared, layout, store, result = _copy_evidence(complete_publication, tmp_path)
    before = store.read_head()
    with _actual_loads(store.root) as loads:
        actual = prepared.publication_readback.publish_and_read(layout.trajectory, result, layout.context)
    assert loads == ["actual-selected-load"]
    path = layout.context.artifact_directory / "coordinate-result.json"
    assert path.read_text() == actual.to_json() == result.to_json()
    assert layout.resolve_result(store, actual).experiment.next_index == 16
    with _actual_loads(store.root) as loads:
        again = prepared.publication_readback.publish_and_read(layout.trajectory, result, layout.context)
    assert loads == ["actual-selected-load"]  # No earlier call validates a later one.
    assert again.to_json() == actual.to_json() and store.read_head() == before


def test_different_existing_immutable_bytes_are_preserved_and_refused(complete_publication, tmp_path):
    _prepared, layout, store, result = _copy_evidence(complete_publication, tmp_path)
    before = store.read_head()
    path = layout.context.artifact_directory / "coordinate-result.json"
    path.write_text("actual different existing bytes")
    with pytest.raises(ValueError):
        layout.publish_and_read(result)
    assert path.read_text() == "actual different existing bytes"
    assert store.read_head() == before


@pytest.mark.parametrize("changed", ["coordinate", "state_blob"])
def test_corruption_after_actual_write_cannot_return_a_checked_publication(complete_publication, tmp_path, changed):
    _prepared, layout, store, result = _copy_evidence(complete_publication, tmp_path)
    before = store.read_head()
    path = layout.context.artifact_directory / "coordinate-result.json"
    saved = []

    def corrupt():
        target = path
        if changed == "state_blob":
            target = store.root / "blobs" / (result.snapshot.component.fingerprint.removeprefix("sha256:") + ".json")
        raw = target.read_bytes()
        saved.append((target, raw))
        target.write_bytes(raw + b" ")

    try:
        with _after_actual_publication(path, corrupt) as events, pytest.raises((ValueError, OSError)):
            layout.publish_and_read(result)
        assert events == ["actual-immutable-write-returned"]
    finally:
        for target, raw in saved:
            target.write_bytes(raw)
    assert store.read_head() == before
    assert path.exists()  # Later refusal is not a rollback/completed-task claim.


def test_actual_other_client_advance_after_write_refuses_stale_publication(complete_publication, tmp_path):
    _prepared, layout, store, result = _copy_evidence(complete_publication, tmp_path)
    before = store.read_head()
    path = layout.context.artifact_directory / "coordinate-result.json"
    accepted = []

    def advance():
        independent = LocalStateStore.open(store.root)
        head = independent.read_head()
        assert head is not None
        request = CommitRequest(
            schema_version=1,
            expected=head,
            intent=CommitIntent(
                schema_version=1,
                generation=head.generation + 1,
                kind="transition",
                content_fingerprint=fingerprint({"actual-after-publication": "other-client"}),
            ),
        )
        receipt = independent.commit_transition(
            request, (layout.binding,), lambda _stage: None, compatibility=layout.context.preparation.compatibility
        )
        assert receipt.status == "new"
        accepted.append(receipt)

    with (
        _after_actual_publication(path, advance) as events,
        pytest.raises(ValueError, match="fresh current matching ACK"),
    ):
        layout.publish_and_read(result)
    assert len(accepted) == 1 and events == ["actual-immutable-write-returned"]
    assert store.read_head() == accepted[0].current_head != before
    assert path.exists()


def test_real_handler_rejects_wrong_original_context_before_artifact_effects(complete_publication, tmp_path):
    prepared, layout, store, result = _copy_evidence(complete_publication, tmp_path)
    before = store.read_head()
    context = replace(layout.context, reference=Reference(id="different-original-coordinate", version="1"))
    with pytest.raises(ValueError, match="exact admitted context"):
        prepared.publication_readback.publish_and_read(layout.trajectory, result, context)
    assert store.read_head() == before
    assert not (layout.context.artifact_directory / "coordinate-result.json").exists()
