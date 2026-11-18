"""Same-object capture joins never replace complete normalized validation."""

from __future__ import annotations

import sys
from contextlib import contextmanager

import pytest

from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.state.components import ComponentCodec, _state_signature
from coleman.state.contracts import StateBinding
from examples.synthetic_portfolio import model
from examples.synthetic_portfolio.experiment import Experiment
from examples.synthetic_portfolio.feedback import Observation
from examples.synthetic_portfolio.flow_contracts import ExperimentStateV2


@pytest.fixture(scope="module")
def normalized_prefix():
    source = Experiment()
    for index in range(2):
        source.step(index)
    return source.export_normalized_state()


@pytest.fixture
def shared_owner(normalized_prefix):
    owner = Experiment()
    owner.restore_normalized_state(normalized_prefix)
    assert owner.observations[0] is owner.periods[1].observation
    return owner


def _codec(state):
    return ComponentCodec(
        StateBinding(
            owner=model.ref("normalized-capture-test-owner"),
            implementation=state.implementation,
            state_codec=Reference(id="normalized-capture-test-state", version="2"),
            configuration_fingerprint=fingerprint({"test": "full-selected-capture-reader"}),
        ),
        ExperimentStateV2,
    )


def _math_wires(owner):
    """Observe complete real components independently of the invalid trace."""
    return (
        owner.ridge.export_state().to_json(),
        tuple((name, book.book.to_json()) for name, book in sorted(owner.books.items())),
        tuple((name, learner.export_state().to_json()) for name, learner in sorted(owner.references.items())),
        tuple((name, learner.export_state().to_json()) for name, learner in sorted(owner.controlled.items())),
    )


@contextmanager
def _observation_encodes():
    """Observe original encoder calls without wrapping it or any validator."""
    code = ComponentCodec.encode.__code__
    calls = []
    tool = next(index for index in (4, 3, 2, 5, 0, 1) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "normalized-capture-reader-test")

    def started(_code, _offset):
        if sys._getframe(1).f_locals["self"].state_type is Observation:
            calls.append("actual-observation-encode")

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
        yield calls
    finally:
        sys.monitoring.free_tool_id(tool)


def test_shared_original_capture_keeps_complete_selected_reader_without_duplicate_observation_encoding(shared_owner):
    owner = shared_owner
    before = _math_wires(owner)
    with _observation_encodes() as calls:
        snapshot = owner.export_normalized_state()
    assert calls == []
    codec = _codec(snapshot)
    encoded = codec.encode(snapshot)
    decoded = codec.decode(encoded.content)
    assert _state_signature(decoded) == _state_signature(snapshot)
    assert decoded.to_legacy().to_json() == owner.export_state().to_json()
    assert _math_wires(owner) == before


@pytest.mark.parametrize("change", ["future", "nonfinite", "producer"])
def test_same_object_join_still_fully_audits_nested_state_before_export(shared_owner, change):
    owner = shared_owner
    observation = owner.observations[0]
    assert observation is owner.periods[1].observation
    before = owner.export_normalized_state().to_json()
    math_before = _math_wires(owner)
    if change == "future":
        target, field, altered = observation, "schema_version", 3
    elif change == "nonfinite":
        target, field, altered = observation.raw[0], "value", float("nan")
    else:
        target, field, altered = observation.applications[0].selection, "outcome", model.ref("foreign-capture-outcome")
    original = getattr(target, field)
    object.__setattr__(target, field, altered)
    try:
        assert owner.observations[0] is owner.periods[1].observation
        with _observation_encodes() as calls, pytest.raises(ValueError):
            owner.export_normalized_state()
        assert calls == []
        assert _math_wires(owner) == math_before
        assert owner.observations[0] is observation
    finally:
        object.__setattr__(target, field, original)
    assert owner.export_normalized_state().to_json() == before


def test_nonidentical_signed_zero_metadata_cannot_replace_the_owned_original(shared_owner):
    owner = shared_owner
    original = owner.observations[0]
    data = original.model_dump(mode="python")
    for application in data["applications"]:
        application["selection"]["definition"]["output"]["spec"]["minimum"] = -0.0
        application["output"]["spec"]["minimum"] = -0.0
    changed = Observation.model_validate(data)
    assert changed is not original
    changed_minimum = changed.applications[0].output.spec.minimum
    original_minimum = original.applications[0].output.spec.minimum
    assert changed_minimum is not None
    assert original_minimum is not None
    assert type(changed_minimum) is float
    assert type(original_minimum) is float
    assert changed_minimum.hex() == "-0x0.0p+0"
    assert original_minimum.hex() == "0x0.0p+0"
    assert changed.to_json() != original.to_json()
    owner.observations = (changed, *owner.observations[1:])
    before = _math_wires(owner)
    try:
        with pytest.raises(ValueError):
            owner.export_normalized_state()
        assert owner.observations[0] is changed
        assert owner.periods[1].observation is original
        assert _math_wires(owner) == before
    finally:
        owner.observations = (original, *owner.observations[1:])
    assert owner.export_normalized_state().observations[0].to_json() == original.to_json()


def test_nonidentical_observation_subclass_cannot_lose_declared_fields_before_exact_codec(shared_owner):
    class ExtendedObservation(Observation):
        extra_evidence: int = 7

    owner = shared_owner
    original = owner.observations[0]
    extended = ExtendedObservation.model_validate(original.model_dump(mode="python") | {"extra_evidence": 19})
    with pytest.raises(ValueError, match="extra_evidence"):
        Observation.model_validate(extended)
    assert extended.extra_evidence == 19
    assert _state_signature(original) != _state_signature(extended)
    owner.observations = (extended, *owner.observations[1:])
    before = _math_wires(owner)
    try:
        with pytest.raises(TypeError, match="exact declared concrete type"):
            owner.export_normalized_state()
        assert owner.observations[0] is extended
        assert _math_wires(owner) == before
    finally:
        owner.observations = (original, *owner.observations[1:])
    assert owner.export_normalized_state().observations[0].to_json() == original.to_json()
