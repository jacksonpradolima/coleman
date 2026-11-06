"""Actual owned RNG continuation includes backend words and Gaussian cache."""

import json
import random
from typing import cast

import pytest

from coleman.state.random_state import PythonRandomState, capture_random, restore_random


def test_capture_is_pure_and_json_round_trip_preserves_the_actual_next_stream():
    source = random.Random(100)
    before = source.getstate()
    state = capture_random(source)
    assert source.getstate() == before
    assert len(state.words) == 624
    assert state.gauss_next is None
    decoded = PythonRandomState.from_json(state.to_json())
    target = random.Random(999)
    restore_random(target, decoded)
    expected = [625644691, 1973661107, 1953896601, 4087596401]
    assert [source.getrandbits(32) for _ in range(4)] == expected
    assert [target.getrandbits(32) for _ in range(4)] == expected
    assert target.getstate() == source.getstate()
    assert decoded.fingerprint == state.fingerprint


def test_cached_gaussian_is_restored_instead_of_becoming_another_backend_draw():
    source = random.Random(42)
    source.gauss(0.0, 1.0)
    snapshot = capture_random(source)
    assert snapshot.gauss_next is not None
    target = random.Random(123)
    restore_random(target, snapshot)
    before_words = capture_random(target).words
    before_index = capture_random(target).index
    assert target.gauss(0.0, 1.0) == snapshot.gauss_next
    after = capture_random(target)
    assert after.words == before_words
    assert after.index == before_index
    assert after.gauss_next is None
    assert target.getrandbits(32) == source.getrandbits(32)


@pytest.mark.parametrize("cached", [0.0, -0.0, -1.25, 2.5, None])
def test_explicit_cache_values_and_null_survive_canonical_round_trip(cached):
    base = capture_random(random.Random(7))
    state = PythonRandomState.model_validate(base.model_copy(update={"gauss_next": cached}))
    decoded = PythonRandomState.from_json(state.to_json())
    assert decoded.to_json() == state.to_json()


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("state_version", 2),
        ("state_version", 4),
        ("state_version", "3"),
        ("state_version", True),
        ("index", -1),
        ("index", 625),
        ("index", 1.0),
        ("index", True),
        ("words", (0,) * 623),
        ("words", (0,) * 625),
        ("words", (-1,) + (0,) * 623),
        ("words", (2**32,) + (0,) * 623),
        ("words", (False,) + (0,) * 623),
        ("words", (0.0,) + (0,) * 623),
        ("gauss_next", float("inf")),
        ("gauss_next", float("-inf")),
        ("gauss_next", float("nan")),
        ("gauss_next", True),
        ("gauss_next", "0"),
        ("codec", "python-random-mt19937-v2"),
    ],
)
def test_invalid_snapshot_rejects_before_mutating_the_actual_target(field, invalid):
    source = capture_random(random.Random(4))
    unchecked = source.model_copy(update={field: invalid})
    target = random.Random(5)
    before = target.getstate()
    with pytest.raises(ValueError):
        restore_random(target, unchecked)
    assert target.getstate() == before


def test_required_null_cache_is_not_silently_defaulted():
    payload = capture_random(random.Random(4)).model_dump(mode="json")
    del payload["gauss_next"]
    with pytest.raises(ValueError, match="gauss_next"):
        PythonRandomState.model_validate(payload)


def test_json_duplicate_and_future_tokens_reject_at_the_codec_boundary():
    payload = capture_random(random.Random(4)).to_json()
    duplicate = payload[:-1] + ',"state_version":3}'
    future = json.loads(payload)
    future["schema_version"] = 2
    encoded = json.dumps(future)
    with pytest.raises(ValueError, match="Duplicate"):
        PythonRandomState.from_json(duplicate)
    with pytest.raises(ValueError):
        PythonRandomState.from_json(encoded)


def test_subclass_backend_is_rejected_without_executing_custom_methods():
    class CustomRandom(random.Random):
        def getstate(self):
            raise AssertionError("Untrusted backend callback")

    custom = CustomRandom(1)
    state = capture_random(random.Random(2))
    with pytest.raises(TypeError, match="concrete"):
        capture_random(custom)
    with pytest.raises(TypeError, match="concrete"):
        restore_random(custom, state)


def test_wrong_snapshot_kind_rejects_without_target_mutation():
    target = random.Random(4)
    before = target.getstate()
    invalid = cast(PythonRandomState, {})
    with pytest.raises(TypeError, match="concrete"):
        restore_random(target, invalid)
    assert target.getstate() == before


def test_interleaved_owned_instances_restore_their_streams_without_sharing_state():
    left, right = random.Random(100), random.Random(101)
    left.getrandbits(32)
    right.getrandbits(32)
    left_state, right_state = capture_random(left), capture_random(right)
    resumed_left, resumed_right = random.Random(0), random.Random(0)
    restore_random(resumed_left, left_state)
    restore_random(resumed_right, right_state)
    left_expected = [left.getrandbits(32) for _ in range(9)]
    right_expected = [right.getrandbits(32) for _ in range(9)]
    actual_left, actual_right = [], []
    for _ in range(9):
        actual_right.append(resumed_right.getrandbits(32))
        actual_left.append(resumed_left.getrandbits(32))
    assert actual_left == left_expected
    assert actual_right == right_expected
    assert resumed_left.getstate() == left.getstate()
    assert resumed_right.getstate() == right.getstate()
