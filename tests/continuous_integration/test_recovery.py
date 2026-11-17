"""Actual chronological labels, complete checkpoints and SQLite process recovery."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from coleman.observations.contracts import ObservationHistory, ObservationRevision
from coleman.state.components import ComponentValue
from coleman.state.local import LocalStateStore
from tests.continuous_integration.recovery_driver import collection, command, compatibility_for, hedge_inputs

ROOT = Path(__file__).parents[2]
DRIVER = Path(__file__).with_name("recovery_driver.py")
PROCESS_SECONDS = 60


def drain(example, owner, record, value, *, at):
    owner.replay.admit_feedback((example.feedback(record, value),))
    return owner.replay.advance(eligible_through=example.instant(at), apply_at=example.instant(at))


def test_real_zero_and_negative_labels_follow_prelabel_predictions(continuous_example):
    example = continuous_example
    owner = example.ForecastOwner()
    predictions = []
    for index, value in enumerate(example.LABELS):
        record = example.emit(owner, index)
        pending = owner.host.export_state().pending[-1]
        predictions.append(pending.prediction.value.value)
        assert pending.prediction.sample_count == index
        assert pending.input.values == example.FEATURES[index]
        drain(example, owner, record, value, at=2 * index + 1.5)
    state = owner.export_state()
    assert predictions == pytest.approx((0.0, 1.0, 6 / 5, -7 / 12), rel=0, abs=1e-13)
    assert tuple(row.target.value for row in state.host_state.observations) == (2.0, 0.0, -1.0, 1.0)
    assert state.host_state.ridge.payload.gram == ((7.0, 3.0), (3.0, 4.0))
    assert state.host_state.ridge.payload.cross == (0.0, 0.0)
    assert state.state.update_count == 4
    assert state.host_state.pending == ()


def test_pending_checkpoint_retains_original_x_and_delayed_zero_label(continuous_example):
    example = continuous_example
    owner = example.ForecastOwner()
    first = example.emit(owner, 0)
    owner.replay.admit_feedback((example.feedback(first, 2.0, receipt=3, delay=1),))
    second = example.emit(owner, 1)
    owner.replay.admit_feedback((example.feedback(second, 0.0),))
    state = owner.export_state()
    restored = example.ForecastOwner()
    state_type = type(state)
    restored.restore_state(state_type.from_json(state.to_json()))
    assert restored.export_state().to_json() == state.to_json()
    assert tuple(row.input.values for row in state.host_state.pending) == ((1.0, 0.0), (1.0, 1.0))
    assert tuple(row.prediction.value.value for row in state.host_state.pending) == (0.0, 0.0)
    for actual in (owner, restored):
        updates = actual.replay.advance(eligible_through=example.instant(4), apply_at=example.instant(5))
        assert len(updates) == 2
        assert tuple(row.application.apply_at for row in updates) == (example.instant(5),) * 2
    assert restored.export_state().to_json() == owner.export_state().to_json()
    payload = owner.host.export_state().ridge.payload
    assert payload.gram == ((3.0, 1.0), (1.0, 2.0))
    assert payload.cross == (2.0, 0.0)
    assert payload.coefficients == pytest.approx((4 / 5, -2 / 5), rel=0, abs=1e-13)


def test_future_feature_correction_preserves_original_prediction_and_training_x(continuous_example):
    example = continuous_example
    owner = example.ForecastOwner()
    original = owner.replay.state.history
    assert original is not None
    source = original.revisions[0]
    correction = ObservationRevision.model_validate(
        {
            **source.model_dump(),
            "metadata": {**source.metadata.model_dump(), "revision_id": "later", "available_at": example.instant(10)},
            "received_at": example.instant(10),
            "supersedes": source.metadata.revision_id,
            "measurement": {**source.measurement.model_dump(), "value": 3.0},
        }
    )
    history = ObservationHistory.model_validate(
        {**original.model_dump(), "revisions": (*original.revisions, correction)}
    )
    earlier = owner.replay.eligible_snapshot(example.instant(0)).to_json()
    owner.replay.admit_history(history)
    assert owner.replay.eligible_snapshot(example.instant(0)).to_json() == earlier
    record = example.emit(owner, 0)
    before = owner.host.export_state().pending[0]
    drain(example, owner, record, 2.0, at=2)
    owner.replay.advance(eligible_through=example.instant(11), apply_at=example.instant(11))
    assert owner.host.export_state().observations[0].pending.model_dump_json() == before.model_dump_json()
    assert owner.host.export_state().ridge.payload.gram == ((2.0, 0.0), (0.0, 1.0))
    assert owner.host.export_state().ridge.payload.cross == (2.0, 0.0)
    assert (
        owner.replay.eligible_snapshot(example.instant(11)).fingerprint
        != owner.replay.eligible_snapshot(example.instant(0)).fingerprint
    )


def test_later_label_correction_and_exact_retries_never_reappend(continuous_example):
    example = continuous_example
    owner = example.ForecastOwner()
    record = example.emit(owner, 0)
    initial = example.feedback(record, 2.0)
    owner.replay.admit_feedback((initial, initial))
    owner.replay.advance(eligible_through=example.instant(2), apply_at=example.instant(2))
    before = owner.host.export_state().to_json()
    retained = owner.replay.emit(owner.replay.prepare(record.opportunity), execution="offline_identity_v1")
    assert retained.to_json() == record.to_json()
    late = example.feedback(record, -2.0, revision=("corrected", "initial"), receipt=3)
    owner.replay.admit_feedback((initial, late, late))
    assert owner.replay.advance(eligible_through=example.instant(4), apply_at=example.instant(4)) == ()
    assert owner.host.export_state().to_json() == before
    assert owner.replay.state.update_count == 1
    assert len(owner.replay.state.feedback) == 2
    assert owner.replay.advance(eligible_through=example.instant(4), apply_at=example.instant(4)) == ()


def test_all32_actual_predictors_and_hedge_states_checkpoint_without_reset(continuous_example, tmp_path):
    example = continuous_example
    definition, bindings, hedge_bindings = collection(example, 32, True)
    _, loss_panel = hedge_inputs(example)
    hedges = tuple(binding.new_owner() for binding in hedge_bindings)
    owners = tuple(example.ForecastOwner(index) for index in range(32))
    for owner in owners:
        record = example.emit(owner, 0)
        drain(example, owner, record, 2.0, at=1.5)
        example.emit(owner, 1)
    originals = tuple(owner.export_state() for owner in owners)
    for hedge in hedges:
        hedge.observe(loss_panel)
    hedge_originals = tuple(hedge.export_state() for hedge in hedges)
    all_bindings = (*bindings, *hedge_bindings)
    values = tuple(
        ComponentValue(binding.codec, state)
        for binding, state in zip(all_bindings, (*originals, *hedge_originals), strict=True)
    )
    compatibility = compatibility_for(example, all_bindings)
    store = LocalStateStore.create(tmp_path / "predictors32", definition, compatibility)
    receipt = store.initialize(command(0, None, kind="initialize"), values, all_bindings, compatibility=compatibility)
    loaded = LocalStateStore.open(store.root).load(all_bindings[::-1], compatibility=compatibility)
    assert tuple(loaded.component(binding.codec).to_json() for binding in bindings) == tuple(
        state.to_json() for state in originals
    )
    assert sum(state.host_state.ridge.payload.sample_count for state in originals) == 32
    assert sum(len(state.host_state.pending) for state in originals) == 32
    assert tuple(loaded.component(binding.codec).to_json() for binding in hedge_bindings) == tuple(
        state.to_json() for state in hedge_originals
    )

    def complete(stage):
        for binding in reversed(bindings):
            actual = stage.owner(binding)
            producer = actual.replay.state.decisions[-1]
            drain(example, actual, producer, 0.0, at=4)
        for binding in reversed(hedge_bindings):
            stage.owner(binding).observe(loss_panel)

    request = command(1, receipt.current_head)
    accepted = store.commit_transition(request, all_bindings, complete, compatibility=compatibility)
    current = store.load(all_bindings, compatibility=compatibility)
    for binding, old in zip(bindings, originals, strict=True):
        state = current.component(binding.codec)
        assert state.host_state.ridge.payload.sample_count == 2
        assert state.host_state.ridge.payload.coefficients == pytest.approx((4 / 5, -2 / 5), rel=0, abs=1e-13)
        assert state.host_state.pending == ()
        assert state.state.decisions == old.state.decisions
    for binding, uninterrupted in zip(hedge_bindings, hedges, strict=True):
        uninterrupted.observe(loss_panel)
        assert current.component(binding.codec).to_json() == uninterrupted.export_state().to_json()
    retry = store.commit_transition(
        request, all_bindings, lambda _: pytest.fail("retained command ran twice"), compatibility=compatibility
    )
    assert retry.status == "retained"
    assert retry.current_head == accepted.current_head


def process_command(example, root, operation, *arguments):
    return [
        sys.executable,
        "-I",
        "-B",
        str(DRIVER),
        operation,
        str(root),
        "--flow",
        example.__file__,
        "--source-root",
        str(ROOT),
        *arguments,
    ]


def invoke(example, root, operation, *arguments, expected=0):
    result = subprocess.run(
        process_command(example, root, operation, *arguments),
        cwd=root.parent,
        capture_output=True,
        text=True,
        timeout=PROCESS_SECONDS,
        check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    return json.loads(result.stdout) if result.stdout.strip() else {}


def wait_for(path, process):
    deadline = time.monotonic() + PROCESS_SECONDS
    while not path.exists():
        if process.poll() is not None:
            output, error = process.communicate()
            pytest.fail(output + error)
        if time.monotonic() >= deadline:
            pytest.fail("The actual process did not reach its bounded barrier")
        time.sleep(0.01)


def test_real_process_death_before_ack_preserves_pending_original_and_can_recover(continuous_example, tmp_path):
    example = continuous_example
    root = tmp_path / "store"
    original = invoke(example, root, "initialize")
    invoke(example, root, "transition", "--fault", "before-ack", expected=87)
    assert (tmp_path / "fault.jsonl").read_text().splitlines() == ["after_head_update"]
    recovered = invoke(example, root, "inspect")
    assert recovered == original
    accepted = invoke(example, root, "transition", "--actor", "restarted")
    assert accepted["generation"] == 1
    assert accepted["samples"] == 1
    assert accepted["pending"] == 0
    assert (tmp_path / "restarted-calls.jsonl").read_text().splitlines() == ["append"]


@pytest.mark.parametrize("divergent", [False, True])
def test_real_concurrent_commands_have_one_math_winner_and_retained_or_refused_loser(
    continuous_example, tmp_path, divergent
):
    example = continuous_example
    root = tmp_path / "store"
    original = invoke(example, root, "initialize")
    winner = loser = None
    try:
        options = ("--divergent",) if divergent else ()
        loser = subprocess.Popen(
            process_command(example, root, "transition", "--actor", "loser", "--fault", "standby", *options),
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_for(tmp_path / "loser-ready.jsonl", loser)
        winner = subprocess.Popen(
            process_command(example, root, "transition", "--actor", "winner", "--fault", "hold"),
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_for(tmp_path / "held.jsonl", winner)
        (tmp_path / "start-contender").write_text("contend using the original predecessor", encoding="utf-8")
        wait_for(tmp_path / "loser-attempt.jsonl", loser)
        assert not (tmp_path / "loser-calls.jsonl").exists()
        assert winner.poll() is None
        (tmp_path / "release").write_text("release the real writer", encoding="utf-8")
        winner_output, winner_error = winner.communicate(timeout=PROCESS_SECONDS)
        loser_output, loser_error = loser.communicate(timeout=PROCESS_SECONDS)
        assert winner.returncode == 0, winner_output + winner_error
        assert loser.returncode == 0, loser_output + loser_error
        assert json.loads(winner_output)["status"] == "new"
        result = json.loads(loser_output)
        if divergent:
            assert "conflicts" in result["rejected"]
        else:
            assert result["status"] == "retained"
        current = invoke(example, root, "inspect")
        assert current["generation"] == original["generation"] + 1
        assert current["samples"] == 1
        assert current["pending"] == 0
        assert not (tmp_path / "loser-calls.jsonl").exists()
        assert (tmp_path / "winner-calls.jsonl").read_text().splitlines() == ["append"]
    finally:
        for process in (winner, loser):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
