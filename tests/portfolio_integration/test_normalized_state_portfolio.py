"""Versioned normalization retains complete causal evidence and legacy bytes."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from coleman.state.components import _state_signature
from examples.synthetic_portfolio import model
from examples.synthetic_portfolio.experiment import Experiment
from examples.synthetic_portfolio.flow_contracts import ExperimentState, ExperimentStateV2
from examples.synthetic_portfolio.task_contracts import PortfolioCheckpoint

FIXTURES = Path(__file__).parent / "fixtures" / "normalized-state-v2"


def _json(value):
    """Supply an independent standard-library canonical byte oracle."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@pytest.fixture(scope="module")
def phases():
    """Capture real four-decision emitted and original feedback-ready phases."""
    owner = Experiment()
    for index in range(4):
        owner.step(index)
    emitted = owner.export_state(), owner.export_normalized_state()
    owner.apply_feedback(4)
    ready = owner.export_state(), owner.export_normalized_state()
    return {"emitted": emitted, "ready": ready}


def test_frozen_pre_normalization_writer_and_independent_v2_golden_remain_exact():
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    for name, expected in provenance["fixture_hashes"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == expected
    outer_bytes = (FIXTURES / "legacy-virgin-checkpoint-v1.json").read_bytes()
    legacy_bytes = (FIXTURES / "legacy-virgin-state-v1.json").read_bytes()
    normalized_bytes = (FIXTURES / "normalized-virgin-state-v2.json").read_bytes()
    outer = PortfolioCheckpoint.from_json(outer_bytes)
    legacy = ExperimentState.from_json(legacy_bytes)
    normalized = ExperimentStateV2.from_json(normalized_bytes)
    assert outer.to_json().encode() == outer_bytes
    assert outer.experiment.to_json().encode() == legacy_bytes
    assert legacy.to_json().encode() == legacy_bytes
    assert normalized.to_json().encode() == normalized_bytes
    assert ExperimentStateV2.from_legacy(legacy).to_json().encode() == normalized_bytes
    assert normalized.to_legacy().to_json().encode() == legacy_bytes
    expected = json.loads(legacy_bytes)
    del expected["observations"]
    expected["schema_version"] = 2
    assert _json(expected).encode() == normalized_bytes
    assert normalized.fingerprint == "sha256:" + hashlib.sha256(normalized_bytes).hexdigest()
    assert legacy.fingerprint != normalized.fingerprint


@pytest.mark.parametrize("phase", ["emitted-two", "ready-two"])
def test_nonvirgin_legacy_writer_goldens_preserve_all_evidence_when_normalized(phase):
    legacy_bytes = (FIXTURES / (phase + "-state-v1.json")).read_bytes()
    normalized_bytes = (FIXTURES / (phase + "-state-v2.json")).read_bytes()
    legacy = ExperimentState.from_json(legacy_bytes)
    normalized = ExperimentStateV2.from_json(normalized_bytes)
    expected_count = 2 if phase == "ready-two" else 1
    assert legacy.next_index == normalized.next_index == 2
    assert len(legacy.observations) == len(normalized.observations) == expected_count
    assert legacy.periods[1].observation is not None
    assert normalized.periods[1].observation is not None
    assert (normalized.ready_observation is not None) == (phase == "ready-two")
    assert legacy.to_json().encode() == legacy_bytes
    assert normalized.to_json().encode() == normalized_bytes
    assert ExperimentStateV2.from_legacy(legacy).to_json().encode() == normalized_bytes
    expanded = normalized.to_legacy()
    assert expanded.to_json().encode() == legacy_bytes
    assert _state_signature(expanded) == _state_signature(legacy)
    expected = json.loads(legacy_bytes)
    del expected["observations"]
    expected["schema_version"] = 2
    assert _json(expected).encode() == normalized_bytes


@pytest.mark.parametrize(
    "state_type,name",
    [(ExperimentState, "legacy-virgin-state-v1.json"), (ExperimentStateV2, "normalized-virgin-state-v2.json")],
)
@pytest.mark.parametrize("bad_version", [0, 3, True, "2", 2.0, None])
def test_versioned_readers_reject_unknown_or_noninteger_versions(state_type, name, bad_version):
    data = json.loads((FIXTURES / name).read_bytes())
    data["schema_version"] = bad_version
    with pytest.raises(ValueError):
        state_type.from_json(_json(data))


def test_v1_and_v2_are_explicit_readers_and_duplicate_field_is_not_an_alternative_source():
    legacy = (FIXTURES / "legacy-virgin-state-v1.json").read_bytes()
    normalized = (FIXTURES / "normalized-virgin-state-v2.json").read_bytes()
    with pytest.raises(ValueError):
        ExperimentState.from_json(normalized)
    with pytest.raises(ValueError):
        ExperimentStateV2.from_json(legacy)
    data = json.loads(normalized)
    data["observations"] = []
    with pytest.raises(ValueError):
        ExperimentStateV2.from_json(_json(data))
    assert "observations" in ExperimentState.model_fields
    assert "observations" not in ExperimentStateV2.model_fields


@pytest.mark.parametrize("phase", ["emitted", "ready"])
def test_complete_derived_observations_and_explicit_conversion_keep_every_original_field(phases, phase):
    legacy, normalized = phases[phase]
    before = legacy.to_json()
    assert normalized.schema_version == 2
    assert normalized.next_index == len(normalized.periods) == 4
    assert normalized.periods[0].observation is None
    expected = tuple(period.observation for period in normalized.periods[1:])
    if phase == "ready":
        assert normalized.ready_observation is not None
        assert normalized.ready_observation.index == 4
        expected += (normalized.ready_observation,)
    else:
        assert normalized.ready_observation is None
    assert all(item is not None for item in expected)
    assert tuple(item.to_json() for item in normalized.observations) == tuple(item.to_json() for item in expected)
    assert tuple(item.index for item in normalized.observations) == tuple(range(1, 4 + int(phase == "ready")))
    assert normalized.ridge.payload.sample_count == len(normalized.observations)
    assert len(normalized.observed_prices) == 4 + int(phase == "ready")
    assert "observations" not in normalized.model_dump(mode="json")
    assert normalized.to_legacy().to_json() == before
    assert ExperimentStateV2.from_legacy(legacy).to_json() == normalized.to_json()
    assert legacy.to_json() == before
    restored = ExperimentStateV2.from_json(normalized.to_json())
    assert _state_signature(restored) == _state_signature(normalized)
    assert _state_signature(restored.to_legacy()) == _state_signature(legacy)
    assert tuple(row.name for row in normalized.periods[-1].candidates) == model.MODES
    assert len(normalized.periods[-1].candidates) == 14


@pytest.mark.parametrize("phase", ["emitted", "ready"])
def test_key_permutation_changes_no_normalized_content_or_expanded_evidence(phases, phase):
    _legacy, normalized = phases[phase]

    def reverse_keys(value):
        if isinstance(value, dict):
            return {key: reverse_keys(item) for key, item in reversed(tuple(value.items()))}
        if isinstance(value, list):
            return [reverse_keys(item) for item in value]
        return value

    data = normalized.model_dump(mode="json")
    restored = ExperimentStateV2.from_json(json.dumps(reverse_keys(data), ensure_ascii=False, allow_nan=False))
    assert restored.to_json() == normalized.to_json()
    assert restored.fingerprint == normalized.fingerprint
    assert restored.to_legacy().to_json() == normalized.to_legacy().to_json()


def _malformed(normalized, change, *, mode="python"):
    data = copy.deepcopy(normalized.model_dump(mode=mode))
    if change == "missing_observation":
        data["periods"][1]["observation"] = None
    elif change == "initial_future_label":
        data["periods"][0]["observation"] = data["periods"][1]["observation"]
    elif change == "period_order":
        periods = list(data["periods"])
        periods[1], periods[2] = periods[2], periods[1]
        data["periods"] = tuple(periods)
    elif change == "duplicate_observation":
        data["periods"][2]["observation"] = data["periods"][1]["observation"]
    elif change == "pending_producer":
        data["pending"]["prediction"]["input"]["values"] = (2.0, 0.0)
    elif change == "selected_scale_producer":
        data["periods"][1]["observation"]["applications"][0]["selection"]["decision_fingerprint"] = "sha256:" + "0" * 64
    elif change == "missing_controller_call":
        observation = data["periods"][3]["observation"]
        observation["controls"] = observation["controls"][:-1]
    elif change == "stale_control_and_objective":
        index = model.MODES.index("plain")
        original = data["periods"][2]["candidates"][index]
        candidate = data["periods"][3]["candidates"][index]
        candidate["control"] = original["control"]
        candidate["control_objective"] = original["control_objective"]
    elif change == "book_path":
        data["books"][0]["state"]["cash"]["numerator"] += 1
    elif change == "missing_candidate":
        data["periods"][1]["candidates"] = data["periods"][1]["candidates"][:-1]
    else:
        raise AssertionError("Unknown independently authored corruption")
    return data


@pytest.mark.parametrize(
    "change",
    [
        "missing_observation",
        "initial_future_label",
        "period_order",
        "duplicate_observation",
        "pending_producer",
        "selected_scale_producer",
        "missing_controller_call",
        "stale_control_and_objective",
        "book_path",
        "missing_candidate",
    ],
)
def test_normalized_ingress_and_atomic_restore_reject_corrupt_complete_evidence(phases, change):
    _legacy, normalized = phases["emitted"]
    changed = _malformed(normalized, change)
    owner = Experiment()
    before = owner.export_state().to_json()
    with pytest.raises(ValueError):
        ExperimentStateV2.model_validate(changed)
    for value in (ExperimentStateV2.model_construct(**changed), normalized.model_copy(update=changed)):
        with pytest.raises(ValueError):
            ExperimentStateV2.model_validate(value)
        with pytest.raises(ValueError):
            owner.restore_normalized_state(value)
        assert owner.export_state().to_json() == before
    with pytest.raises(ValueError):
        ExperimentStateV2.from_json(_json(_malformed(normalized, change, mode="json")))


@pytest.mark.parametrize("change", ["index", "missing", "ridge_count", "prices"])
def test_feedback_ready_phase_retains_exact_frontier_and_original_mark(phases, change):
    _legacy, normalized = phases["ready"]
    data = normalized.model_dump(mode="python")
    if change == "index":
        data["ready_observation"]["index"] = 3
    elif change == "missing":
        data["ready_observation"] = None
    elif change == "ridge_count":
        data["ridge"]["payload"]["sample_count"] -= 1
    else:
        data["ready_observation"]["prices"] = data["observed_prices"][3]
    with pytest.raises(ValueError):
        ExperimentStateV2.model_validate(data)


@pytest.mark.parametrize("phase", ["emitted", "ready"])
def test_actual_full_continuation_and_duplicate_feedback_match_expanded_v1(phases, phase):
    legacy, normalized = phases[phase]
    direct, resumed = Experiment(), Experiment()
    direct.restore_state(legacy)
    resumed.restore_normalized_state(ExperimentStateV2.from_json(normalized.to_json()))
    assert resumed.export_state().to_json() == direct.export_state().to_json()
    if phase == "ready":
        before = resumed.export_normalized_state().to_json()
        first = resumed.apply_feedback(4)
        second = resumed.apply_feedback(4)
        assert first.to_json() == second.to_json() == normalized.ready_observation.to_json()
        assert resumed.export_normalized_state().to_json() == before
        assert resumed.ridge.export_state().payload.sample_count == 4
    for index in range(4, 16):
        assert resumed.step(index).to_json() == direct.step(index).to_json()
    final = resumed.export_normalized_state()
    assert final.next_index == 16
    assert len(final.observations) == final.ridge.payload.sample_count == 15
    assert final.to_legacy().to_json() == direct.export_state().to_json()
    assert resumed.step(15).to_json() == direct.step(15).to_json()
    assert resumed.export_normalized_state().to_json() == final.to_json()


def test_read_conversion_never_authorizes_historical_image_operational_resume():
    legacy = ExperimentState.from_json((FIXTURES / "legacy-virgin-state-v1.json").read_bytes())
    normalized = ExperimentStateV2.from_legacy(legacy)
    owner = Experiment()
    before = owner.export_state().to_json()
    with pytest.raises(ValueError, match="implementation"):
        owner.restore_state(legacy)
    with pytest.raises(ValueError, match="implementation"):
        owner.restore_normalized_state(normalized)
    assert owner.export_state().to_json() == before


def test_live_owner_cannot_discard_inconsistent_same_count_observations_when_normalizing(phases):
    legacy, _normalized = phases["emitted"]
    owner = Experiment()
    owner.restore_state(legacy)
    actual = owner.observations
    original_periods = tuple(period.to_json() for period in owner.periods)
    changed = actual[0].model_copy(update={"ridge_label": actual[1].ridge_label})
    owner.observations = (changed, *actual[1:])
    assert len(owner.observations) == len(actual)
    with pytest.raises(ValueError):
        owner.export_state()
    with pytest.raises(ValueError):
        owner.export_normalized_state()
    assert tuple(period.to_json() for period in owner.periods) == original_periods
    owner.observations = actual
    assert owner.export_normalized_state().to_legacy().to_json() == legacy.to_json()


@pytest.mark.parametrize("method", ["copy", "construct"])
def test_v2_copied_or_constructed_input_cannot_silently_discard_removed_observations(phases, method):
    _legacy, normalized = phases["emitted"]
    injected = {"observations": normalized.observations}
    with pytest.raises(ValueError):
        if method == "copy":
            forged = normalized.model_copy(update=injected)
        else:
            forged = ExperimentStateV2.model_construct(**(normalized.model_dump(mode="python") | injected))
        ExperimentStateV2.model_validate(forged)


def test_explicit_conversion_preserves_or_refuses_typed_duplicate_metadata(phases):
    """Canonical JSON equality alone cannot authorize loss of typed timestamp data."""
    legacy, _normalized = phases["emitted"]
    data = legacy.model_dump(mode="python")
    instant = data["observations"][0]["prices"]["at"]
    data["observations"][0]["prices"]["at"] = instant.replace(fold=1)
    original = ExperimentState.model_validate(data)
    try:
        converted = ExperimentStateV2.from_legacy(original)
    except ValueError:
        return
    assert _state_signature(converted.to_legacy()) == _state_signature(original)


def test_normalized_capture_cannot_drop_live_timestamp_fold_hidden_by_json_equality(phases):
    from coleman.problems.portfolio import PricePanel
    from examples.synthetic_portfolio.feedback import Observation

    legacy, _normalized = phases["emitted"]
    owner = Experiment()
    owner.restore_state(legacy)
    original = owner.observations[0]
    prices = PricePanel.model_validate(original.prices.model_copy(update={"at": original.prices.at.replace(fold=1)}))
    changed = Observation.model_validate(original.model_copy(update={"prices": prices}))
    assert changed.to_json() == original.to_json()
    assert changed.prices.at.fold == 1 and original.prices.at.fold == 0
    owner.observations = (changed, *owner.observations[1:])
    assert owner.export_state().observations[0].prices.at.fold == 1
    with pytest.raises(ValueError):
        owner.export_normalized_state()
    assert owner.observations[0].prices.at.fold == 1
    assert owner.periods[1].observation.prices.at.fold == 0
