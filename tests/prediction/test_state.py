"""Concrete complete state restoration and strict prefix continuation."""

import json

import pytest

from coleman.prediction import RidgePayload
from coleman.state.components import ComponentCodec
from coleman.state.contracts import AlgorithmState
from tests.prediction._support import config, features, owner, sample


def test_real_component_codec_recovers_every_statistic_and_exact_next_prediction_update():
    """Actual generic codecs restore sufficient state and the same next append."""
    original = owner()
    assert original.append(sample(original, (1.0, 0.0), 2.0)).status == "accepted"
    codec = ComponentCodec(original.binding, AlgorithmState[RidgePayload])
    encoded = codec.encode(original.export_state())
    resumed = owner()
    resumed.restore_state(codec.decode(encoded.content))
    assert resumed.export_state().to_json() == original.export_state().to_json()
    x = features(original, (1.0, 1.0))
    assert resumed.predict(x).to_json() == original.predict(x).to_json()
    next_sample = sample(original, (1.0, 1.0), 0.0)
    assert resumed.append(next_sample).to_json() == original.append(next_sample).to_json()
    assert resumed.export_state().to_json() == original.export_state().to_json()
    assert resumed.export_state().payload.update_count == 2


@pytest.mark.parametrize(
    "change",
    ["future", "coefficient", "diagonal", "symmetry", "count", "fake_untrained", "gram_capacity", "cross_capacity"],
)
def test_altered_state_refuses_before_replacing_the_actual_owner(change):
    """Readers reject forged claims instead of repairing a stored coefficient."""
    model = owner()
    assert model.append(sample(model, (1.0, 1.0), -2.0)).status == "accepted"
    before = model.export_state().to_json()
    raw = json.loads(before)
    payload = raw["payload"]
    if change == "future":
        payload["config"]["schema_version"] = 2
    elif change == "coefficient":
        payload["coefficients"][0] += 0.01
    elif change == "diagonal":
        payload["gram"][0][0] = 0.5
    elif change == "symmetry":
        payload["gram"][0][1] = 0.5
    elif change == "count":
        payload["update_count"] = 0
    elif change == "gram_capacity":
        payload["gram"][0][0] = 1e250
    elif change == "cross_capacity":
        payload["cross"][0] = 1e250
    else:
        payload["update_count"] = payload["sample_count"] = 0
    with pytest.raises(ValueError):
        AlgorithmState[RidgePayload].model_validate(raw)
    assert model.export_state().to_json() == before


def test_unknown_codec_version_refuses_even_when_the_payload_is_readable():
    """Reading known mathematics does not adopt an unknown continuation codec."""
    model = owner()
    raw = json.loads(model.export_state().to_json())
    raw["binding"]["state_codec"]["version"] = "2"
    readable = AlgorithmState[RidgePayload].model_validate(raw)
    before = model.export_state().to_json()
    with pytest.raises(ValueError, match="complete binding"):
        model.restore_state(readable)
    assert model.export_state().to_json() == before


@pytest.mark.parametrize("other", ["configuration", "owner"])
def test_restore_retains_original_configuration_and_owner_bindings(other):
    """A valid foreign snapshot cannot relabel its owner or ridge convention."""
    original = owner()
    incompatible = owner(config(regularization=2.0)) if other == "configuration" else owner(name="another-owner")
    before = original.export_state().to_json()
    incoming = incompatible.export_state()
    with pytest.raises(ValueError, match="complete binding"):
        original.restore_state(incoming)
    assert original.export_state().to_json() == before
