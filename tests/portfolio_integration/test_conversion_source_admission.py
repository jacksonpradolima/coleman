"""Converters admit original typed values through real public codecs."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import pytest

from coleman.state.components import ComponentCodec, _state_signature
from examples.synthetic_portfolio.flow_contracts import ExperimentState, ExperimentStateV2

FIXTURES = Path(__file__).parent / "fixtures" / "normalized-state-v2"


def _fixture(direction, phase="virgin"):
    names = (
        ("legacy-virgin-state-v1.json", "normalized-virgin-state-v2.json")
        if phase == "virgin"
        else (phase + "-state-v1.json", phase + "-state-v2.json")
    )
    legacy, normalized = ((FIXTURES / name).read_bytes() for name in names)
    if direction == "normalize":
        return ExperimentState.from_json(legacy), normalized
    return ExperimentStateV2.from_json(normalized), legacy


def _convert(direction, source):
    return ExperimentStateV2.from_legacy(source) if direction == "normalize" else source.to_legacy()


@contextmanager
def _admissions(source_type, events):
    """Observe unwrapped source validation and exact public encoder arguments."""
    encoder = ComponentCodec.encode.__code__
    validator = source_type.model_validate.__func__.__code__
    arguments = []
    tool = next(index for index in (4, 3, 2, 5, 0, 1) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "coleman-conversion-source-test")

    def started(code, _offset):
        current = sys._getframe(1).f_locals
        if code is encoder and current["self"].state_type is source_type:
            events.append("encode")
            arguments.append(current["state"])
        elif code is validator and current["cls"] is source_type:
            events.append("validate")

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
        for code in (encoder, validator):
            sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
        yield arguments
    finally:
        sys.monitoring.free_tool_id(tool)


@pytest.mark.parametrize("direction", ["normalize", "expand"])
@pytest.mark.parametrize("phase", ["virgin", "emitted-two", "ready-two"])
def test_stock_conversion_admits_original_before_projection_and_matches_independent_goldens(direction, phase):
    source, expected = _fixture(direction, phase)
    before = _state_signature(source)
    events = []
    with _admissions(type(source), events) as arguments:
        result = _convert(direction, source)
    assert len(arguments) == 1
    assert arguments[0] is source
    assert events == ["encode", "validate", "validate", "validate"]
    assert result.to_json().encode() == expected
    assert _state_signature(source) == before


@pytest.mark.parametrize("direction", ["normalize", "expand"])
@pytest.mark.parametrize("field", ["strength", "implementation"])
def test_original_typed_fields_cannot_be_normalized_before_portability_admission(direction, field):
    source, _expected = _fixture(direction)
    value = Decimal("0.05") if field == "strength" else source.implementation.model_dump(mode="python")
    copied = source.model_copy(update={field: value})
    # Ordinary model ingress can normalize these inputs; exact conversion must
    # instead check the caller's actual typed object against its selected reader.
    validated = type(source).model_validate(copied)
    assert validated.to_json() == source.to_json()
    assert _state_signature(validated) != _state_signature(copied)
    with pytest.raises(ValueError, match="preserve the exact complete declared state"):
        _convert(direction, copied)
    assert getattr(copied, field) is value


@pytest.mark.parametrize("direction", ["normalize", "expand"])
@pytest.mark.parametrize("field", ["schema_version", "implementation", "tape_fingerprint"])
def test_missing_original_headers_have_documented_refusal_without_attribute_leak(direction, field):
    source, _expected = _fixture(direction)
    data = source.model_dump(mode="python")
    del data[field]
    unchecked = type(source).model_construct(**data)
    with pytest.raises((ValueError, TypeError)):
        _convert(direction, unchecked)


@pytest.mark.parametrize("direction", ["normalize", "expand"])
@pytest.mark.parametrize("field", ["schema_version", "implementation", "tape_fingerprint"])
def test_unchecked_headers_cannot_execute_string_conversion_before_admission(direction, field):
    calls = []

    class StringBomb:
        def __str__(self):
            calls.append("unexpected-string-conversion")
            raise AssertionError("An unchecked header must not execute a conversion callback")

    source, _expected = _fixture(direction)
    unchecked = source.model_copy(update={field: StringBomb()})
    with pytest.raises((ValueError, TypeError)):
        _convert(direction, unchecked)
    assert calls == []


@pytest.mark.parametrize("direction", ["normalize", "expand"])
@pytest.mark.parametrize("version", [999, True, "2", 2.0, None])
def test_unknown_or_noninteger_original_version_never_becomes_selected_metadata(direction, version):
    source, _expected = _fixture(direction)
    unchecked = source.model_copy(update={"schema_version": version})
    with pytest.raises((ValueError, TypeError)):
        _convert(direction, unchecked)


@pytest.mark.parametrize("direction", ["normalize", "expand"])
def test_original_cyclic_graph_is_bounded_before_any_signature_projection(direction):
    source, _expected = _fixture(direction)
    cycle = []
    cycle.append(cycle)
    unchecked = source.model_copy(update={"periods": cycle})
    with pytest.raises(ValueError, match="cyclic"):
        _convert(direction, unchecked)


@pytest.mark.parametrize("direction", ["normalize", "expand"])
@pytest.mark.parametrize("custom", ["reader", "writer"])
def test_custom_public_methods_keep_prevalidation_and_the_actual_selected_callback_sequence(
    direction, custom, monkeypatch
):
    source, expected = _fixture(direction)
    source_type = type(source)
    events = []
    if custom == "reader":
        original = source_type.from_json.__func__

        def reader(cls, payload):
            events.append("reader")
            return original(cls, payload)

        monkeypatch.setattr(source_type, "from_json", classmethod(reader))
        expected_events = ["validate", "encode", "validate", "reader", "validate", "validate"]
    else:
        original = source_type.to_json

        def writer(self):
            events.append("writer")
            return original(self)

        monkeypatch.setattr(source_type, "to_json", writer)
        expected_events = [
            "validate",
            "encode",
            "validate",
            "writer",
            "validate",
            "validate",
            "validate",
            "writer",
            "validate",
        ]
    with _admissions(source_type, events) as arguments:
        result = _convert(direction, source)
    assert len(arguments) == 1
    assert arguments[0] is not source
    assert _state_signature(arguments[0]) == _state_signature(source)
    assert events == expected_events
    assert result.to_json().encode() == expected


@pytest.mark.parametrize("direction", ["normalize", "expand"])
def test_changed_target_reader_keeps_the_original_source_fallback_without_inventing_a_target_read(
    direction, monkeypatch
):
    source, expected = _fixture(direction)
    target_type = ExperimentStateV2 if direction == "normalize" else ExperimentState
    original = target_type.from_json.__func__
    events = []

    def reader(cls, payload):
        events.append("target-reader")
        return original(cls, payload)

    monkeypatch.setattr(target_type, "from_json", classmethod(reader))
    with _admissions(type(source), events) as arguments:
        result = _convert(direction, source)
    assert len(arguments) == 1
    assert arguments[0] is not source
    assert events == ["validate", "encode", "validate", "validate", "validate"]
    assert result.to_json().encode() == expected


def test_normalized_subclass_uses_the_existing_prevalidated_path():
    class ExtendedState(ExperimentStateV2):
        """A trusted same-field Python extension still retains the legacy route."""

    base, expected = _fixture("expand")
    source = ExtendedState.model_validate(base.model_dump(mode="python"))
    events = []
    with _admissions(ExtendedState, events) as arguments:
        result = source.to_legacy()
    assert len(arguments) == 1
    assert arguments[0] is not source
    assert events == ["validate", "encode", "validate", "validate", "validate"]
    assert result.to_json().encode() == expected


def test_legacy_subclass_is_not_a_new_implicitly_selected_conversion_type():
    class ExtendedState(ExperimentState):
        """The original exact-v1 source-type restriction stays explicit."""

    base, _expected = _fixture("normalize")
    source = ExtendedState.model_validate(base.model_dump(mode="python"))
    with pytest.raises(TypeError, match="exact v1 state type"):
        ExperimentStateV2.from_legacy(source)
