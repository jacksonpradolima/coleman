"""Direct built-in graph dispatch preserves independent JSON bounds.

Generated and reordered inputs are metamorphic contract checks, not independent
experimental observations. The standard library supplies every expected size.
"""

from __future__ import annotations

import json
import random
import sys
from contextlib import contextmanager
from decimal import Decimal

import pytest
from pydantic import BaseModel, Field, model_serializer

from coleman.core import _wire


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))


def _bounds(value, *, items):
    """Check inclusive bytes independently, including the immediately lower cap."""
    size = _size(value)
    _wire.guard_graph(value, maximum_bytes=size, maximum_items=items)
    _wire.guard_graph(value, maximum_bytes=size + 1, maximum_items=items)
    with pytest.raises(ValueError, match="byte budget"):
        _wire.guard_graph(value, maximum_bytes=size - 1, maximum_items=items)


@contextmanager
def _calls(*functions):
    """Count original function execution without replacing guarded methods."""
    names = {function.__code__: function.__name__ for function in functions}
    counts = dict.fromkeys(names.values(), 0)
    tool = next(index for index in (4, 3, 2, 5, 0, 1) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "coleman-graph-size-test")

    def begin(code, _offset):
        counts[names[code]] += 1

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, begin)
        for code in names:
            sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
        yield counts
    finally:
        sys.monitoring.free_tool_id(tool)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        123456789012345678901234567890,
        -123456789012345678901234567890,
        True,
        False,
        None,
        -0.0,
        "",
        'quoted"slash\\',
        "\x00\b\f\n\r\t\x1f",
        "ação/λ/雪/😀",
        "\ud7ff\ue000",
    ],
)
def test_repeated_values_and_keys_keep_independent_exact_byte_boundaries(value):
    _bounds([value, value, {"field": value}], items=6)


def test_bool_equal_integer_values_cannot_share_incorrect_numeric_sizes():
    """Python key equality must not turn true into1 or false into0."""
    _bounds([1, True, 0, False, 1, True, 0, False, -1, -1], items=11)


@pytest.mark.parametrize("kind", ["string", "integer", "negative"])
def test_repeated_and_unique_same_size_values_have_identical_budget_outcomes(kind):
    """Repetition alone preserves byte admission and occurrence charges."""
    values = list(range(1000, 1064))
    if kind == "negative":
        values = [-value for value in values]
    elif kind == "string":
        values = [f"value-{value}" for value in values]
    repeated = [values[0]] * len(values)
    assert _size(repeated) == _size(values)
    for graph in (repeated, values, list(reversed(values))):
        _bounds(graph, items=65)
        with pytest.raises(ValueError, match="item budget"):
            _wire.guard_graph(graph, maximum_bytes=_size(graph), maximum_items=64)


@pytest.mark.parametrize("case", range(24))
def test_generated_repetition_and_permutation_preserve_utf8_and_numeric_sizes(case):
    rng = random.Random(378_500 + case)
    atoms = [0, -10, 10**40, True, False, None, -0.0, "", 'a"\\\n', "é雪😀"]
    values = [rng.choice(atoms) for _ in range(32)]
    graph = [{"value": value, "again": value} for value in values]
    reordered = [{"again": value, "value": value} for value in reversed(values)]
    assert _size(graph) == _size(reordered)
    _bounds(graph, items=161)
    _bounds(reordered, items=161)


def test_repeated_aliases_still_charge_every_container_key_and_value():
    shared = {"repeated": [123456, "repeated", "repeated"]}
    graph = [shared, shared]
    _bounds(graph, items=13)
    with pytest.raises(ValueError, match="item budget"):
        _wire.guard_graph(graph, maximum_bytes=_size(graph), maximum_items=12)
    shared["repeated"].append("later mutation")
    _bounds(graph, items=15)


@pytest.mark.parametrize("container", [tuple, set, frozenset])
def test_other_builtin_containers_keep_all_byte_and_item_charges(container):
    value = container([0, 1])
    size = _size([0, 1])
    _wire.guard_graph(value, maximum_bytes=size, maximum_items=3)
    with pytest.raises(ValueError, match="byte budget"):
        _wire.guard_graph(value, maximum_bytes=size - 1, maximum_items=3)
    with pytest.raises(ValueError, match="item budget"):
        _wire.guard_graph(value, maximum_bytes=size, maximum_items=2)


def test_container_subclasses_stay_opaque_without_running_iteration_hooks():
    class Sequence(list):
        def __iter__(self):
            raise AssertionError("An opaque sequence iterator must not run")

    class Mapping(dict):
        def items(self):
            raise AssertionError("An opaque mapping iterator must not run")

    for value in (Sequence([0, 1]), Mapping(field=0)):
        with pytest.raises(TypeError, match="unsupported opaque"):
            _wire.guard_graph([0, "prefix", value], maximum_bytes=1024, maximum_items=10)


def test_real_stored_models_and_container_aliases_visit_every_occurrence():
    """Built-in dispatch keeps the actual model reader and skips serializers."""

    class Stored(BaseModel):
        payload: list[int | str]

        @model_serializer
        def refuse_serialization(self):
            raise AssertionError("Unchecked model serialization must not run")

    value = Stored(payload=[123, "repeated"])
    graph = [value, value]
    size = _size([{"payload": [123, "repeated"]}, {"payload": [123, "repeated"]}])
    with _calls(_wire.model_storage, _wire._visit_container) as calls:
        _wire.guard_graph(graph, maximum_bytes=size, maximum_items=13)
    assert calls["model_storage"] == 2
    assert calls["_visit_container"] == 5
    with pytest.raises(ValueError, match="byte budget"):
        _wire.guard_graph(graph, maximum_bytes=size - 1, maximum_items=13)
    with pytest.raises(ValueError, match="item budget"):
        _wire.guard_graph(graph, maximum_bytes=size, maximum_items=12)


def test_model_fields_hidden_from_serialization_still_require_admission():
    class Stored(BaseModel):
        payload: object = Field(exclude=True)

    value = Stored.model_construct(payload=object())
    with pytest.raises(TypeError, match="unsupported opaque"):
        _wire.guard_graph(["repeated", "repeated", value], maximum_bytes=1024, maximum_items=10)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), Decimal("NaN"), Decimal("Infinity")])
def test_previously_sized_prefix_does_not_conceal_nonfinite_values(value):
    with pytest.raises(ValueError, match="finite"):
        _wire.guard_graph([7, 7, "prefix", "prefix", value], maximum_bytes=1024, maximum_items=6)


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"])
def test_repeated_prefix_never_admits_a_lone_surrogate(surrogate):
    with pytest.raises(ValueError, match="lone surrogates"):
        _wire.guard_graph(["prefix", "prefix", surrogate], maximum_bytes=1024, maximum_items=4)


def test_supported_value_does_not_admit_equal_scalar_subclass_or_run_its_hooks():
    class Key(str):
        """Constructible mapping key still requires exact built-in admission."""

    class Text(str):
        def __hash__(self):
            raise AssertionError("Opaque string hash hook must not run")

    class Integer(int):
        def __hash__(self):
            raise AssertionError("Opaque integer hash hook must not run")

    for value in (Text("same"), Integer(7), object()):
        with pytest.raises(TypeError, match="unsupported opaque"):
            _wire.guard_graph(["same", "same", 7, 7, value], maximum_bytes=1024, maximum_items=6)
    with pytest.raises(TypeError, match="built-in strings"):
        _wire.guard_graph([{"same": 0}, {Key("same"): 0}], maximum_bytes=1024, maximum_items=7)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "Graph byte budget exceeded before model serialization"),
        ("aa", "Graph byte budget exceeded by a UTF-8/escaped string"),
        ("雪", "Graph byte budget exceeded by a UTF-8/escaped string"),
        (11, "Graph byte budget exceeded before model serialization"),
        (10**100, "Graph byte budget exceeded before model serialization"),
    ],
)
def test_repetition_at_insufficient_remaining_budget_keeps_the_original_error(value, message):
    graph = [value, value]
    with pytest.raises(ValueError) as caught:
        _wire.guard_graph(graph, maximum_bytes=_size(graph) - 1, maximum_items=3)
    assert str(caught.value) == message


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("repeated", "Graph byte budget exceeded by a UTF-8/escaped string"),
        (11, "Graph byte budget exceeded before model serialization"),
        (10**100, "Graph byte budget exceeded by an integer"),
    ],
)
def test_repeated_value_overflow_still_precedes_a_later_opaque_object(value, message):
    allowance = 4 + _size(value) + 1
    with pytest.raises(ValueError) as caught:
        _wire.guard_graph([value, value, object()], maximum_bytes=allowance, maximum_items=4)
    assert str(caught.value) == message


def test_large_repeated_scalars_keep_the_existing_public_envelope():
    """Large valid built-in scalars retain their inclusive UTF-8 byte envelope."""
    text = "雪" * 4097
    integer = 1 << 4096
    graph = [text, integer, text, integer]
    _bounds(graph, items=5)


def test_builtin_dispatch_preserves_container_depth_and_ancestor_cycles():
    nested = ["repeat", "repeat"]
    for _ in range(63):
        nested = [nested]
    _wire.guard_graph(nested, maximum_bytes=1024, maximum_items=100)
    with pytest.raises(ValueError, match="depth exceeds 64"):
        _wire.guard_graph([nested], maximum_bytes=1024, maximum_items=100)
    cycle: list[object] = ["repeat", "repeat"]
    cycle.append(cycle)
    with pytest.raises(ValueError, match="cyclic container"):
        _wire.guard_graph(cycle, maximum_bytes=1024, maximum_items=100)


def test_integer_conversion_limit_remains_fresh_between_calls():
    """A previous successful graph cannot authorize a later forbidden conversion."""
    original = sys.get_int_max_str_digits()
    value = 1 << 2200
    try:
        sys.set_int_max_str_digits(0)
        _wire.guard_graph([value, value], maximum_bytes=4096, maximum_items=3)
        sys.set_int_max_str_digits(sys.int_info.str_digits_check_threshold)
        with pytest.raises(ValueError, match="limit"):
            _wire.guard_graph([value, value], maximum_bytes=4096, maximum_items=3)
    finally:
        sys.set_int_max_str_digits(original)
