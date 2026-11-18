"""Explicit scoped execution retains actual evidence only during its lifetime."""

import shutil
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace

import pytest

from coleman.core.primitives import fingerprint
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CommitIntent, CommitRequest
from coleman.tasks.protocols import AttemptContext, CommittedProgress, decode_value, encode_value
from examples.synthetic_portfolio.experiment import Experiment
from examples.synthetic_portfolio.recipe import PROGRESS_PROOF
from tests.portfolio_integration.test_opened_evidence_portfolio import _actual_loads
from tests.portfolio_integration.test_publication_readback_portfolio import _declared


def _attempt(context, guards):
    """Count direct-library guard calls; actual worker fences are tested installed."""
    return AttemptContext(
        context, "actual-scoped-unit-reservation", 1, time.monotonic() + 120, lambda: guards.append(1)
    )


def _concrete(progress):
    """Apply the actual public declared proof codec, without a proof stub."""
    proof = decode_value(PROGRESS_PROOF, encode_value(PROGRESS_PROOF, progress.proof))
    return CommittedProgress(progress.plan_fingerprint, progress.coordinate_fingerprint, progress.frontier, proof)


@contextmanager
def _actual_steps():
    """Observe real original period calls without wrapping the mathematical core."""
    code = Experiment.step.__code__
    tool = next(index for index in (4, 3, 2, 5, 0, 1) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "scoped-original-period-observer")
    steps = []

    def entered(_code, _offset):
        steps.append(sys._getframe(1).f_locals["index"])

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, entered)
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
        yield steps
    finally:
        sys.monitoring.free_tool_id(tool)


@pytest.fixture(scope="module")
def prefix(tmp_path_factory):
    """Accept one prefix; copies below are same-image continuation, not new data."""
    prepared, layout = _declared(tmp_path_factory.mktemp("scope-actual-prefix"))
    store = layout.open(create=True)
    assert store is not None
    layout.initialize(store)
    assert layout.advance(store, lambda: None).frontier.position == 8
    return prepared, layout


def _copy_prefix(prefix, tmp_path):
    """Copy every inactive file and reopen the same complete selected store."""
    prepared, original = prefix
    directory = tmp_path / "same-image-source-copy"
    shutil.copytree(original.context.artifact_directory, directory)
    layout = replace(original, context=replace(original.context, artifact_directory=directory))
    return prepared, layout, LocalStateStore.open(directory / "state")


def test_scope_lifetime_disposes_proxy_verifier_and_properties_without_implicit_io(tmp_path):
    root = tmp_path / "uncreated"
    prepared, layout = _declared(root)
    manager = prepared.coordinate_operation
    assert manager.profile.id == "coleman-coordinate-operation" and manager.profile.version == "1"
    with manager.open(layout.trajectory, layout.context) as scope:
        verifier = scope.progress_verifier
        assert scope is verifier
        assert scope.read_completed(layout.trajectory, layout.context) is None
        assert not root.exists()
    for operation in (
        lambda: scope.read_completed(layout.trajectory, layout.context),
        lambda: verifier.reconcile(layout.context),
        lambda: scope.state_codec,
        lambda: scope.publication_readback,
        lambda: scope.boundary,
        lambda: scope.profile,
    ):
        with pytest.raises(RuntimeError, match="closed"):
            operation()
    scope.close()  # Idempotent cleanup does not acknowledge or compensate work.
    assert not root.exists()
    with manager.open(layout.trajectory, layout.context) as another:
        assert another is not scope
        assert another.reconcile(layout.context) is None
    assert not root.exists()


def test_scoped_real_full_cycle_uses_one_selected_acquisition_per_ack_and_preserves_every_field(tmp_path):
    prepared, layout = _declared(tmp_path)
    guards, callbacks = [], []
    with (
        _actual_steps() as steps,
        _actual_loads(layout.context.artifact_directory / "state") as loads,
        prepared.coordinate_operation.open(layout.trajectory, layout.context) as scope,
    ):
        assert scope.read_completed(layout.trajectory, layout.context) is None

        def committed(progress):
            scope.progress_verifier.validate(_concrete(progress), layout.context)
            callbacks.append(progress.frontier.position)

        result = scope.advance(layout.trajectory, _attempt(layout.context, guards), committed)
        actual = scope.publish_and_read(layout.trajectory, result, layout.context)
        published = scope.read_completed(layout.trajectory, layout.context)
        assert published is not None and published.to_json() == actual.to_json()
    assert steps == list(range(16)) and callbacks == [8, 16]
    assert len(guards) == 17  # One caller guard before work and each of16 actual periods.
    assert len(loads) == 3  # Initialization, original8periodACK and terminal16periodACK.
    store = LocalStateStore.open(layout.context.artifact_directory / "state")
    full = layout.resolve_result(store, result)
    expected = Experiment()
    for index in range(16):
        expected.step(index)
    assert full.experiment.to_legacy().to_json() == expected.export_state().to_json()
    assert actual.to_json() == result.to_json()
    with (
        _actual_loads(store.root) as later_loads,
        prepared.coordinate_operation.open(layout.trajectory, layout.context) as later,
    ):
        published = later.read_completed(layout.trajectory, layout.context)
        assert published is not None and published.to_json() == result.to_json()
    assert later_loads == ["actual-selected-load"]  # A closed scope authorizes no later call.


def test_callback_response_gap_retains_ack_and_new_scope_continues_without_repeated_observations(prefix, tmp_path):
    prepared, layout, store = _copy_prefix(prefix, tmp_path)
    callbacks, guards = [], []
    with _actual_steps() as steps:
        with (
            pytest.raises(RuntimeError, match="actual outer response gap"),
            prepared.coordinate_operation.open(layout.trajectory, layout.context) as scope,
        ):

            def lost_response(progress):
                scope.validate(_concrete(progress), layout.context)
                callbacks.append(progress.frontier.position)
                raise RuntimeError("actual outer response gap")

            scope.advance(layout.trajectory, _attempt(layout.context, guards), lost_response)
        assert callbacks == [16] and steps == list(range(8, 16))
        assert layout.reconcile(store).frontier.position == 16
        with prepared.coordinate_operation.open(layout.trajectory, layout.context) as resumed:
            assert resumed.reconcile(layout.context).frontier.position == 16
            result = resumed.advance(
                layout.trajectory, _attempt(layout.context, []), lambda _: pytest.fail("repeated ACK")
            )
            resumed.publish_and_read(layout.trajectory, result, layout.context)
        assert steps == list(range(8, 16))
    with pytest.raises(RuntimeError, match="closed"):
        scope.reconcile(layout.context)
    assert layout.resolve_result(store, result).experiment.next_index == 16


def test_blob_corruption_between_actual_ack_and_scoped_verification_is_refused(prefix, tmp_path):
    prepared, layout, store = _copy_prefix(prefix, tmp_path)
    saved = []

    def changed(progress):
        path = max((store.root / "blobs").glob("*.json"), key=lambda item: item.stat().st_size)
        raw = path.read_bytes()
        saved.append((path, raw))
        path.write_bytes(raw + b" ")
        scope.validate(_concrete(progress), layout.context)

    try:
        with (
            pytest.raises((ValueError, OSError)),
            prepared.coordinate_operation.open(layout.trajectory, layout.context) as scope,
        ):
            scope.advance(layout.trajectory, _attempt(layout.context, []), changed)
    finally:
        for path, raw in saved:
            path.write_bytes(raw)
    assert layout.reconcile(store).frontier.position == 16
    assert not (layout.context.artifact_directory / "coordinate-result.json").exists()
    with pytest.raises(RuntimeError, match="closed"):
        scope.reconcile(layout.context)


def test_actual_other_client_new_head_refuses_old_proof_even_when_period_count_is_equal(prefix, tmp_path):
    prepared, layout, store = _copy_prefix(prefix, tmp_path)
    accepted = []

    def advance_head(progress):
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
                content_fingerprint=fingerprint({"actual-scoped-callback": "other-client-no-new-period"}),
            ),
        )
        receipt = independent.commit_transition(
            request, (layout.binding,), lambda _stage: None, compatibility=layout.context.preparation.compatibility
        )
        assert receipt.status == "new"
        accepted.append(receipt)
        scope.validate(_concrete(progress), layout.context)

    with (
        pytest.raises(ValueError, match="actual current ACK"),
        prepared.coordinate_operation.open(layout.trajectory, layout.context) as scope,
    ):
        scope.advance(layout.trajectory, _attempt(layout.context, []), advance_head)
    assert len(accepted) == 1 and store.read_head() == accepted[0].current_head
    assert layout.reconcile(store).frontier.position == 16
    assert not (layout.context.artifact_directory / "coordinate-result.json").exists()


def test_same_scope_refuses_cross_thread_use_before_storage_effects(tmp_path):
    root = tmp_path / "uncreated"
    prepared, layout = _declared(root)
    with prepared.coordinate_operation.open(layout.trajectory, layout.context) as scope:
        with (
            ThreadPoolExecutor(max_workers=1) as executor,
            pytest.raises(RuntimeError, match="creating thread"),
        ):
            executor.submit(scope.reconcile, layout.context).result()
        assert scope.reconcile(layout.context) is None
    assert not root.exists()


def test_separate_concurrent_scopes_do_not_repeat_actual_periods_and_both_dispose(prefix, tmp_path):
    prepared, layout, store = _copy_prefix(prefix, tmp_path)
    barrier = threading.Barrier(2)
    scopes = []

    def run():
        with prepared.coordinate_operation.open(layout.trajectory, layout.context) as scope:
            scopes.append(scope)
            barrier.wait()
            try:
                result = scope.advance(layout.trajectory, _attempt(layout.context, []), lambda _: None)
                scope.publish_and_read(layout.trajectory, result, layout.context)
                return "completed"
            except TimeoutError:
                return "busy"  # The existing actualfive-second store lock is unchanged.
            except sqlite3.OperationalError as error:
                code = getattr(error, "sqlite_errorcode", None)
                if type(code) is not int or code < 0 or code & 0xFF not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                    raise
                return "busy"
            except ValueError as error:
                if "terminal original period" not in str(error) and "acknowledged" not in str(error):
                    raise
                return "refused"

    with _actual_steps() as steps, ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run) for _ in range(2)]
        statuses = [future.result() for future in futures]
    assert "completed" in statuses
    assert sorted(steps) == list(range(8, 16))
    assert layout.reconcile(store).frontier.position == 16
    assert len(scopes) == 2 and scopes[0] is not scopes[1]
    for scope in scopes:
        with pytest.raises(RuntimeError, match="closed"):
            scope.reconcile(layout.context)
