"""Exact virgin-state validation preserves signed zero and rejects subnormals."""

from __future__ import annotations

import json
import math
import struct

import pytest

from tests.tasks.test_reference_executor import source as source


@pytest.mark.parametrize("zero", (0.0, -0.0), ids=("positive-zero", "negative-zero"))
def test_original_virgin_wire_preserves_both_zero_signs(source, zero):
    """A zero sufficient statistic is exact; its wire retains its original sign."""
    trajectory = source.sample.inputs().trajectories[0]
    raw = trajectory.model_dump(mode="json")
    payload = raw["initial_state"]["payload"]
    for name in ("cumulative_losses", "compensations"):
        payload[name] = [zero] * len(payload[name])
    wire = json.dumps(raw)
    selected_type = type(trajectory)
    checked = selected_type.model_validate_json(wire)
    for values in (checked.initial_state.payload.cumulative_losses, checked.initial_state.payload.compensations):
        assert all(struct.pack(">d", value) == struct.pack(">d", zero) for value in values)
    assert checked.initial_state.payload.weights == trajectory.initial_state.payload.weights
    assert checked.initial_state.payload.update_count == 0


@pytest.mark.parametrize("name", ("cumulative_losses", "compensations"))
@pytest.mark.parametrize("value", (math.nextafter(0.0, 1.0), math.nextafter(0.0, -1.0)), ids=("positive", "negative"))
def test_nonzero_subnormal_cannot_become_virgin_state_via_tolerance(source, name, value):
    """Real typed decoding refuses nonzero U0 statistics before state adoption."""
    trajectory = source.sample.inputs().trajectories[0]
    raw = trajectory.model_dump(mode="json")
    raw["initial_state"]["payload"][name][0] = value
    wire = json.dumps(raw)
    selected_type = type(trajectory)
    with pytest.raises(ValueError, match="cumulative losses|compensation"):
        selected_type.model_validate_json(wire)
