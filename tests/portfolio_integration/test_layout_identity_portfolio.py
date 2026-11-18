"""Typed metadata is not an unused state conversion or an admission witness."""

import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256

import pytest

from examples.synthetic_portfolio.experiment import Experiment
from examples.synthetic_portfolio.flow_contracts import ExperimentStateV2
from tests.portfolio_integration.test_task_state_portfolio import make_layout


@contextmanager
def _initial_conversions():
    """Observe the original converter without wrapping any selected method."""
    code = ExperimentStateV2.from_legacy.__func__.__code__
    calls = []
    tool = next(index for index in (4, 3, 2, 5, 0, 1) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "portfolio-layout-metadata-test")

    def started(_code, _offset):
        calls.append("actual-initial-conversion")

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
        yield calls
    finally:
        sys.monitoring.free_tool_id(tool)


def _digest(value):
    """Compute the declared JSON content identity with the standard library."""
    wire = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + sha256(wire.encode("utf-8")).hexdigest()


def test_metadata_keeps_complete_binding_without_materializing_unused_state(tmp_path):
    layout, store = make_layout(tmp_path)
    expected_configuration = _digest(
        {
            "configuration": _digest(json.loads(layout.configuration.to_json())),
            "trajectory": _digest(json.loads(layout.trajectory.to_json())),
            "source": layout.context.preparation.source.model_dump(mode="json"),
            "plan": layout.context.preparation.plan_fingerprint,
            "coordinate": layout.context.coordinate_fingerprint,
        }
    )
    with _initial_conversions() as calls:
        identity = layout._identity
        definition = layout.definition
        selected = layout.binding
    assert calls == []
    assert identity.binding == definition.components[0].binding == selected.codec.binding
    assert identity.binding.configuration_fingerprint == expected_configuration
    assert identity.binding.owner == layout.trajectory.reference
    assert identity.binding.implementation == layout.context.preparation.implementation
    assert identity.binding.state_codec.version == "2"
    assert identity.schema_version == 2
    assert identity.source == layout.context.preparation.source
    assert identity.plan_fingerprint == layout.context.preparation.plan_fingerprint
    assert identity.coordinate_fingerprint == layout.context.coordinate_fingerprint
    assert identity.rows == layout.trajectory.rows
    complete = layout.initial
    assert complete.model_dump(exclude={"experiment"}) == identity.model_dump(mode="python")
    assert complete.experiment.next_index == 0
    with pytest.raises(TypeError, match="exact declared concrete state type|exact declared concrete type"):
        selected.codec.encode(identity)
    assert layout.checkpoint(store).experiment.next_index == 0


def test_actual_owner_factory_still_constructs_and_restores_complete_real_math(tmp_path):
    layout, store = make_layout(tmp_path)
    with _initial_conversions() as calls:
        selected = layout.binding
        assert calls == []
        owner = selected.new_owner()
    assert calls == ["actual-initial-conversion"]
    virgin = layout.checkpoint(store)
    owner.restore_state(virgin)
    owner.experiment.step(0)
    direct = Experiment()
    direct.step(0)
    assert owner.experiment.export_state().to_json() == direct.export_state().to_json()
    snapshot = owner.export_state()
    assert snapshot.binding == virgin.binding
    assert snapshot.plan_fingerprint == virgin.plan_fingerprint
    assert snapshot.coordinate_fingerprint == virgin.coordinate_fingerprint
    assert snapshot.source == virgin.source
    assert snapshot.rows == virgin.rows
    assert selected.codec.decode(selected.codec.encode(snapshot).content).to_json() == snapshot.to_json()
    assert layout.checkpoint(store).experiment.next_index == 0  # Local math is not a durable ACK.


@pytest.mark.parametrize("change", ["plan", "coordinate", "source", "trajectory"])
def test_metadata_operations_still_reject_invalid_complete_identity(tmp_path, change):
    layout, store = make_layout(tmp_path)
    original = store.read_head()
    if change == "trajectory":
        initial = layout.trajectory.initial_state.model_copy(update={"schema_version": 3})
        bad = replace(layout, trajectory=layout.trajectory.model_copy(update={"initial_state": initial}))
    else:
        preparation = replace(layout.context.preparation)
        context = replace(layout.context, preparation=preparation)
        if change == "plan":
            object.__setattr__(preparation, "plan_fingerprint", "invalid-plan")
        elif change == "coordinate":
            object.__setattr__(context, "coordinate_fingerprint", "invalid-coordinate")
        else:
            source = preparation.source.model_copy(update={"definition_fingerprint": "invalid-source"})
            object.__setattr__(preparation, "source", source)
        bad = replace(layout, context=context)
    with _initial_conversions() as calls:
        for operation in (lambda: bad.definition, lambda: bad.binding):
            with pytest.raises(ValueError):
                operation()
    assert calls == []
    assert store.read_head() == original
    assert layout.checkpoint(store).experiment.next_index == 0
