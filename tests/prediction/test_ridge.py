"""Actual Cholesky updates checked against independently solved eligible prefixes."""

from fractions import Fraction
from itertools import permutations
from math import isclose

import pytest

from tests.prediction._support import batch_fit, config, features, owner, sample


def _close(actual, expected):
    assert len(actual) == len(expected)
    assert all(isclose(a, float(b), rel_tol=1e-12, abs_tol=1e-13) for a, b in zip(actual, expected, strict=True))


def test_required_hand_trace_predicts_before_zero_label_changes_the_gram():
    """The second zero is training evidence, not a missing observation."""
    model = owner()
    assert model.append(sample(model, (1.0, 0.0), 2.0)).status == "accepted"
    _close(model.coefficients, (1, 0))
    before = model.export_state().to_json()
    first = model.predict(features(model, (1.0, 1.0)))
    assert first.value.value is not None
    assert isclose(first.value.value, 1.0, rel_tol=0.0, abs_tol=1e-15)
    assert first.value.spec.role == "prediction"
    assert model.export_state().to_json() == before
    assert model.append(sample(model, (1.0, 1.0), 0.0)).status == "accepted"
    state = model.export_state().payload
    assert state.gram == ((3.0, 1.0), (1.0, 2.0))
    assert state.cross == (2.0, 0.0)
    _close(state.coefficients, (Fraction(4, 5), Fraction(-2, 5)))
    result = model.predict(features(model, (2.0, 1.0)))
    assert result.value.value is not None
    assert isclose(result.value.value, 1.2, rel_tol=0.0, abs_tol=1e-14)
    assert result.sample_count == 2


@pytest.mark.parametrize(
    "rows",
    [
        (((1, 0), 2), ((1, 1), 0), ((2, 1), -3)),
        (((1, 2), -2), ((2, 4), 0), ((3, 6), 4)),
        (((0, 0), -3), ((0, 0), 0)),
        (((-1, 2), 3), ((2, -1), -3), ((1, 1), 1)),
    ],
)
def test_every_incremental_prefix_matches_an_independent_exact_batch_fit(rows):
    """Collinear and signed data retain regularization and original prefix scope."""
    model = owner()
    for end, (values, target) in enumerate(rows, start=1):
        result = model.append(sample(model, values, target))
        assert result.status == "accepted"
        assert result.sample_count == end
        assert result.update_count == end
        _close(model.coefficients, batch_fit(rows[:end]))


def test_direct_duplicate_appends_are_deliberately_nonidempotent():
    """Delivery deduplication is a host obligation rather than a hidden cache."""
    model = owner()
    original = sample(model, (1.0, 0.0), 1.0)
    first = model.append(original)
    second = model.append(original)
    assert first.status == "accepted"
    assert second.status == "accepted"
    assert second.sample_count == 2
    _close(model.coefficients, (Fraction(2, 3), 0))


@pytest.mark.parametrize(
    "values,target,code", [((1e100, 1e100), 1.0, "non_positive_pivot"), ((1e-200, 0.0), 1.0, "underflow")]
)
def test_numerical_refusal_is_typed_and_preserves_every_original_state_byte(values, target, code):
    """An SPD real problem may refuse in the declared finite normal-equation law."""
    model = owner()
    before = model.export_state().to_json()
    result = model.append(sample(model, values, target))
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == code
    assert result.before_fingerprint == result.after_fingerprint
    assert model.export_state().to_json() == before


def test_capacity_refusal_preserves_the_last_accepted_complete_state():
    """The declared finite sample cap neither evicts nor resets prior evidence."""
    model = owner(config(maximum_samples=1))
    value = sample(model, (1.0, 0.0), -1.0)
    assert model.append(value).status == "accepted"
    before = model.export_state().to_json()
    refused = model.append(value)
    assert refused.status == "failed"
    assert refused.failure is not None
    assert refused.failure.code == "capacity"
    assert model.export_state().to_json() == before


def test_prediction_overflow_is_unavailable_not_a_zero_or_state_change():
    """A finite accepted model can have an unrepresentable requested forecast."""
    model = owner(config(regularization=1e-300))
    assert model.append(sample(model, (1e-150, 0.0), 1e100)).status == "accepted"
    before = model.export_state().to_json()
    result = model.predict(features(model, (1e100, 0.0)))
    assert result.failure is not None
    assert result.failure.code == "numeric_range"
    assert result.value.status == "unavailable"
    assert result.value.value is None
    assert model.export_state().to_json() == before


def test_observation_permutations_have_the_same_batch_solution_within_the_named_envelope():
    """Order changes floating reduction bytes, not the ideal append-only objective."""
    rows = (((1, 0), 2), ((1, 1), 0), ((2, -1), -3))
    expected = batch_fit(rows)
    for order in permutations(rows):
        model = owner()
        for x, y in order:
            assert model.append(sample(model, x, y)).status == "accepted"
        _close(model.coefficients, expected)


def test_feature_permutation_retains_positions_without_silently_sorting_dimensions():
    """Renaming coordinate order permutes both statistics and coefficients."""
    original = owner()
    raw = config().model_dump(mode="python")
    raw["features"] = tuple(reversed(raw["features"]))
    permuted = owner(type(original.config).model_validate(raw))
    assert permuted.config.features[0].dimension.id == "x1"
    for x, y in (((1, 0), 2), ((1, 1), 0), ((2, -1), -3)):
        assert original.append(sample(original, x, y)).status == "accepted"
        assert permuted.append(sample(permuted, tuple(reversed(x)), y)).status == "accepted"
    _close(permuted.coefficients, tuple(reversed(original.coefficients)))
