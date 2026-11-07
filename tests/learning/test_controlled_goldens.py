"""Versioned controlled wire compatibility with independently checked dyadics.

The wire snapshots were emitted by the SDK and reviewed as complete examples;
they are compatibility goldens, not independent mathematical solver oracles.
The expected coefficient/variation values below are hand-derived exactly.
All observations and clocks are constructed CI mechanics.
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import pytest

from coleman.core.feedback import FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.learning.control_contracts import ControlledConfig, ControlledPayload
from coleman.learning.controlled import ControlledHedge
from coleman.state.codec import read_algorithm_state

FIXTURES = Path(__file__).parents[1] / "fixtures" / "learning" / "v1"
START = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("name", "digest"),
    [
        ("controlled-law-v1.json", "bd44564a688996b5a92aee2831de29744670a6e8f712d89cbb60515673cd2a99"),
        ("controlled-pending-state-v1.json", "c599fbf30cf865ca163102cce7c30ba7e4236e86aabe869d95384900b712c814"),
        ("controlled-released-state-v1.json", "80afd061878fbe9f541e3cf453ac8c63c9519852404f34b03058b8398e1eb65e"),
    ],
)
def test_reviewed_controlled_wire_fixture_bytes_are_pinned(name, digest):
    assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest


def _learner():
    config = ControlledConfig.from_json((FIXTURES / "controlled-law-v1.json").read_bytes())
    return ControlledHedge(config, owner=Reference(id="golden-ci-controlled-owner", version="1"))


def _panel(config, losses):
    return FullPanel(
        rows=tuple(
            FeedbackRow(
                member=weight.member,
                values=(
                    OutcomeValue(
                        spec=config.loss_spec,
                        value=value,
                        status="observed",
                        origin="synthetic",
                        transform=config.transform.reference,
                    ),
                ),
            )
            for weight, value in zip(config.controls.initial_prior.weights, losses, strict=True)
        )
    )


def _read(document, learner):
    return read_algorithm_state(document, payload_type=ControlledPayload, expected=learner.binding)


def test_controlled_law_golden_is_canonical_complete_and_separate_from_dynamic_origin():
    encoded = (FIXTURES / "controlled-law-v1.json").read_text().rstrip("\n")
    law = ControlledConfig.from_json(encoded)
    assert law.to_json() == encoded
    assert law.eta == 0.0 and law.controls.target is not None
    assert law.controls.target.alpha == 1.0
    assert tuple(Fraction(item.value) for item in law.controls.initial_prior.weights) == (
        Fraction(1, 2),
        Fraction(1, 4),
        Fraction(1, 4),
    )


@pytest.mark.parametrize("alteration", ["future_version", "unknown_field"])
def test_controlled_law_reader_rejects_future_contracts_without_implicit_migration(alteration):
    raw = json.loads((FIXTURES / "controlled-law-v1.json").read_text())
    raw["schema_version" if alteration == "future_version" else "future_extension"] = 2
    encoded = json.dumps(raw)
    with pytest.raises(ValueError, match="schema_version|extra"):
        ControlledConfig.from_json(encoded)


@pytest.mark.parametrize("name", ["controlled-pending-state-v1.json", "controlled-released-state-v1.json"])
def test_bounded_controlled_state_golden_round_trip_and_actual_restore_are_byte_identical(name):
    learner = _learner()
    encoded = (FIXTURES / name).read_text().rstrip("\n")
    state = _read(encoded, learner)
    assert state.to_json() == encoded
    learner.restore_state(state)
    assert learner.export_state().to_json() == encoded
    assert learner.config.binding == state.payload.config.binding
    assert state.payload.unresolved_failure is None


def test_pending_golden_continues_to_moving_then_optimal_hold_released_golden():
    learner = _learner()
    state = _read((FIXTURES / "controlled-pending-state-v1.json").read_bytes(), learner)
    learner.restore_state(state)
    assert (learner.update_count, learner.release_count) == (1, 0)
    assert len(state.payload.pending) == 1 and state.payload.exact_path is not None
    assert all(item.atoms_decimal == "0" for item in state.payload.exact_path.used)
    moving = learner.observe(
        _panel(learner.config, (1.0, 0.0, 0.5)),
        eligible_at=START + timedelta(seconds=3),
        apply_at=START + timedelta(seconds=4),
    )
    expected = (Fraction(5, 16), Fraction(7, 16), Fraction(1, 4))
    assert moving.disposition == "released"
    assert tuple(Fraction(weight.value) for weight in learner.weights) == expected
    before_hold = learner.export_state().payload.exact_path
    assert before_hold is not None
    assert tuple(Fraction(int(item.atoms_decimal), 1 << 1074) for item in before_hold.used) == (
        Fraction(3, 16),
        Fraction(3, 16),
        Fraction(0),
    )
    # eta=0 and target alpha=1 give r=(1/4,1/2,1/4). Against the
    # new (5/16,7/16,1/4), differences are exactly +/-lambda=1/16,
    # so strict convexity makes holding the unique second optimum (nu=0).
    learner.observe(
        _panel(learner.config, (0.5, 0.5, 0.5)),
        eligible_at=START + timedelta(seconds=5),
        apply_at=START + timedelta(seconds=6),
    )
    held = learner.observe(
        _panel(learner.config, (0.5, 0.5, 0.5)),
        eligible_at=START + timedelta(seconds=7),
        apply_at=START + timedelta(seconds=8),
    )
    assert held.disposition == "released"
    assert tuple(Fraction(weight.value) for weight in learner.weights) == expected
    assert held.exact_path_before is not None and held.exact_path_before.used == before_hold.used
    assert held.movement_usage is None
    actual = learner.export_state()
    assert actual.payload.exact_path is not None and actual.payload.exact_path.used == before_hold.used
    assert (actual.payload.update_count, actual.payload.release_count, actual.payload.consumed_panels) == (4, 2, 4)
    assert actual.to_json() == (FIXTURES / "controlled-released-state-v1.json").read_text().rstrip("\n")


@pytest.mark.parametrize("target", ["outer", "payload", "receipt", "exact_path"])
@pytest.mark.parametrize("alteration", ["future_version", "unknown_field"])
def test_controlled_golden_reader_rejects_unknown_versions_or_extensions_without_migration(target, alteration):
    learner = _learner()
    raw = json.loads((FIXTURES / "controlled-released-state-v1.json").read_text())
    selected = raw
    if target == "payload":
        selected = raw["payload"]
    elif target == "receipt":
        selected = raw["payload"]["last_outcome"]
    elif target == "exact_path":
        selected = raw["payload"]["last_outcome"]["exact_path_before"]
    selected["schema_version" if alteration == "future_version" else "future_extension"] = 2
    encoded = json.dumps(raw)
    with pytest.raises(ValueError, match="schema_version|extra"):
        _read(encoded, learner)
    assert learner.update_count == learner.release_count == 0


@pytest.mark.parametrize("field", ["movement_usage", "exact_path_before"])
def test_new_receipt_context_fields_are_required_even_when_their_value_is_null(field):
    learner = _learner()
    raw = json.loads((FIXTURES / "controlled-released-state-v1.json").read_text())
    del raw["payload"]["last_outcome"][field]
    encoded = json.dumps(raw)
    with pytest.raises(ValueError, match=field):
        _read(encoded, learner)
