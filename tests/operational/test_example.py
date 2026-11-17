"""Real shipped learning under matched off/noop/SDK instrumentation profiles."""

import math

import pytest

from coleman.telemetry.resources import ProcessResourceTracker
from coleman.telemetry.views import resource_points


@pytest.mark.parametrize("experts,panels,trajectories", ((2, 1, 1), (2, 2, 2), (8, 3, 1), (32, 2, 1)))
def test_real_hedge_math_and_complete_state_invariant_under_actual_instrumentation(
    ci_example, experts, panels, trajectories
):
    workload = ci_example.Workload(schema_version=1, experts=experts, panels=panels, trajectories=trajectories)
    outputs = [ci_example.run(workload, mode) for mode in ("off", "noop", "memory")]
    assert outputs[0].final_states == outputs[1].final_states == outputs[2].final_states
    assert outputs[0].snapshots == ()
    assert len(outputs[1].snapshots) == panels * trajectories
    assert outputs[1].snapshots == outputs[2].snapshots
    assert "coleman.lifecycle.concentration" in outputs[2].sdk_metric_names
    if experts == 2 and panels == 1:
        actual = [w.value for w in outputs[0].final_states[0].payload.weights]
        independent = 1 / (1 + math.exp(-0.5))
        assert actual == pytest.approx((independent, 1 - independent), abs=1e-15)
    if panels == 2:
        for state in outputs[0].final_states:
            values = [w.value for w in state.payload.weights]
            assert values == pytest.approx([1 / experts] * experts, abs=1e-15)
    assert all(p.value is None or p.value >= 0 for p in resource_points(ProcessResourceTracker().sample()))


def test_workload_caps_and_unknown_instrumentation_are_explicit(ci_example):
    with pytest.raises(ValueError):
        ci_example.Workload(schema_version=1, experts=33)
    with pytest.raises(ValueError):
        ci_example.Workload(schema_version=1, panels=33)
    with pytest.raises(ValueError):
        ci_example.Workload(schema_version=1, trajectories=5)
    workload = ci_example.Workload(schema_version=1)
    with pytest.raises(ValueError):
        ci_example.run(workload, "unrecognized")
