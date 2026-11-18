"""Advance and terminal projection share only one actual operation's evidence."""

import shutil
from dataclasses import replace
from fractions import Fraction
from typing import cast

import pytest

from coleman.core.primitives import fingerprint
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CommitIntent, CommitRequest
from examples.synthetic_portfolio import model
from tests.portfolio_integration.test_task_state_portfolio import ObservedStore, make_layout


class _Observed(ObservedStore):
    """Forward real public commands once and retain their actual receipts."""

    def __init__(self, store):
        super().__init__(store)
        self.commands = []

    def commit_transition(self, request, *arguments, **keywords):
        receipt = self.store.commit_transition(request, *arguments, **keywords)
        self.commands.append((request, receipt))
        return receipt


@pytest.fixture(scope="module")
def after_eight(tmp_path_factory):
    """Keep one accepted prefix for same-image mechanical continuation cases.

    Copies below are checkpoint fixtures, never independent observations or
    physical effects. The actual original store is inactive during each copy.
    """
    layout, store = make_layout(tmp_path_factory.mktemp("advance-completion-prefix"))
    assert layout.advance(store, lambda: None).frontier.position == 8
    return layout


def _continuation(after_eight, tmp_path):
    """Copy every inactive store file and reopen its full current public reader."""
    directory = tmp_path / "same-image-prefix-copy"
    shutil.copytree(after_eight.context.artifact_directory, directory)
    layout = replace(after_eight, context=replace(after_eight.context, artifact_directory=directory))
    return layout, LocalStateStore.open(directory / "state")


def test_nonterminal_batch_delivers_one_actual_callback_without_terminal_projection(tmp_path):
    layout, store = make_layout(tmp_path)
    observed = _Observed(store)
    guard, callbacks = [], []
    accepted, result = layout.advance_with_completion(
        cast(LocalStateStore, observed), lambda: guard.append("actual-period"), callbacks.append
    )
    assert result is None and callbacks == [accepted]
    assert len(guard) == 8 and accepted.frontier.position == 8
    assert observed.loads == 1
    assert [receipt.status for _, receipt in observed.commands] == ["new"]
    assert layout.reconcile(store).frontier.position == 8


def test_terminal_result_joins_real_callback_retained_sql_and_all_fourteen_books(after_eight, tmp_path):
    layout, store = _continuation(after_eight, tmp_path)
    observed = _Observed(store)
    callbacks, guards = [], []

    def committed(progress):
        assert [receipt.status for _, receipt in observed.commands] == ["new"]
        callbacks.append(progress)

    accepted, result = layout.advance_with_completion(
        cast(LocalStateStore, observed), lambda: guards.append("actual-period"), committed
    )
    assert callbacks == [accepted] and len(guards) == 8
    assert accepted.frontier.position == 16 and result is not None
    assert observed.loads == 1  # No second complete read merely to project this same operation.
    assert [receipt.status for _, receipt in observed.commands] == ["new", "retained"]
    assert observed.commands[0][0].to_json() == observed.commands[1][0].to_json()
    assert result.proof.receipt.status == "retained"
    assert result.proof.observed_head == result.proof.receipt.accepted_head == store.read_head()
    assert result.to_json() == layout.complete_result(store).to_json()
    checkpoint = layout.resolve_result(store, result)
    assert tuple(item.name for item in result.outcomes) == model.MODES
    for index, outcome in enumerate(result.outcomes):
        receipts = tuple(period.candidates[index].settlement for period in checkpoint.experiment.periods)
        assert receipts[-1].net_value is not None
        assert outcome.terminal_net_value.fraction == receipts[-1].net_value.fraction
        for summary, field in (
            (outcome.proportional_paid, "proportional_paid"),
            (outcome.recurring_paid, "recurring_paid"),
            (outcome.executed_full_l1_notional, "turnover"),
        ):
            values = tuple(getattr(receipt, field) for receipt in receipts)
            assert all(value is not None for value in values)
            assert summary.fraction == sum(
                (Fraction(value.numerator, value.denominator) for value in values if value is not None), Fraction()
            )


def test_callback_failure_keeps_actual_ack_and_effect_without_replaying_body(after_eight, tmp_path):
    layout, store = _continuation(after_eight, tmp_path)
    observed = _Observed(store)
    effects, guards = [], []

    def failed_callback(progress):
        effects.append(progress)
        raise RuntimeError("actual callback response gap")

    with pytest.raises(RuntimeError, match="actual callback response gap"):
        layout.advance_with_completion(cast(LocalStateStore, observed), lambda: guards.append(1), failed_callback)
    assert len(effects) == 1 and effects[0].frontier.position == 16
    assert len(guards) == 8 and observed.loads == 1
    assert [receipt.status for _, receipt in observed.commands] == ["new"]
    assert layout.reconcile(store).frontier.position == 16
    assert not (layout.context.artifact_directory / "coordinate-result.json").exists()


def test_real_callback_head_advance_requires_complete_current_reload(after_eight, tmp_path):
    layout, store = _continuation(after_eight, tmp_path)
    observed = _Observed(store)
    competing = []

    def committed(progress):
        independent = LocalStateStore.open(store.root)
        head = independent.read_head()
        assert head is not None
        assert head == progress.proof.observed_head
        request = CommitRequest(
            schema_version=1,
            expected=head,
            intent=CommitIntent(
                schema_version=1,
                generation=head.generation + 1,
                kind="transition",
                content_fingerprint=fingerprint({"actual-callback": "terminal-head-advance"}),
            ),
        )
        receipt = independent.commit_transition(
            request, (layout.binding,), lambda _stage: None, compatibility=layout.context.preparation.compatibility
        )
        assert receipt.status == "new"
        competing.append(receipt)

    accepted, result = layout.advance_with_completion(cast(LocalStateStore, observed), lambda: None, committed)
    assert len(competing) == 1 and result is not None
    assert observed.loads == 2
    assert accepted.proof.observed_head != store.read_head()
    assert result.proof.observed_head == store.read_head() == competing[0].current_head
    assert result.proof.receipt.accepted_head == result.proof.observed_head
    assert result.to_json() == layout.complete_result(store).to_json()


def test_callback_corruption_cannot_turn_the_private_snapshot_into_later_authority(after_eight, tmp_path):
    layout, store = _continuation(after_eight, tmp_path)
    saved = []

    def corrupt_actual_artifact(_progress):
        # The terminal blob is the largest complete owned state. Altering it
        # preserves no checksum; the fresh actual read must refuse it.
        path = max((store.root / "blobs").glob("*.json"), key=lambda item: item.stat().st_size)
        raw = path.read_bytes()
        saved.append((path, raw))
        path.write_bytes(raw + b" ")

    try:
        with pytest.raises((ValueError, OSError)):
            layout.advance_with_completion(store, lambda: None, corrupt_actual_artifact)
    finally:
        for path, raw in saved:
            path.write_bytes(raw)
    assert layout.reconcile(store).frontier.position == 16
    assert not (layout.context.artifact_directory / "coordinate-result.json").exists()
