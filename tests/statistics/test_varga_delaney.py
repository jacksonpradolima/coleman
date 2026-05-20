"""
Unit tests for the coleman evaluation metrics module.

This test suite covers the functionality of the evaluation metrics implemented
in the coleman package, specifically the NAPFDMetric and NAPFDVerdictMetric classes.
It ensures that the metrics behave as expected under various scenarios, including:

- Standard test records
- Identical durations across records
- Identical cost and results
- Empty test records
- No failure scenarios

Additionally, it tests the base EvaluationMetric class for its abstract behavior,
ensuring that derived classes implement required methods properly.

Fixtures:
- sample_records: Provides sample test records for evaluation.
- available_time: Calculates the total available time from the sample records.

Constants:
- NAPFD_FITNESS_NON_NEGATIVE: Ensures that fitness is non-negative.
- NAPFD_FITNESS_NOT_EXCEED_ONE: Ensures that fitness does not exceed 1.
- NAPFD_COST_NON_NEGATIVE: Ensures that cost is non-negative.
"""

from typing import Any, cast

import numpy as np
import polars as pl
import pytest

from coleman.statistics.vargha_delaney import reduce, vd_a, vd_a_df


def test_vd_a():
    """
    Test vd_a function for computing Vargha and Delaney's A index.
    """
    group_a = [0.8236, 0.7967, 0.9236, 0.8197, 0.7108]
    group_b = [0.8053, 0.8172, 0.8322, 0.7836, 0.8142]

    estimate, magnitude = vd_a(group_a, group_b)
    assert 0.44 <= estimate <= 0.56, "Expected negligible effect size"
    assert magnitude == "negligible", f"Expected 'negligible', got {magnitude}"


def test_vd_a_df():
    """
    Test vd_a_df function for pairwise comparisons.
    """
    data = pl.DataFrame(
        {
            "values": [1, 2, 3, 4, 5, 6],
            "group": ["A", "A", "B", "B", "C", "C"],  # Equal sizes for all groups
        }
    )

    result = vd_a_df(data, val_col="values", group_col="group")

    # Verify output structure
    assert "base" in result.columns
    assert "compared_with" in result.columns
    assert "estimate" in result.columns
    assert "magnitude" in result.columns

    # Check the number of pairwise comparisons (3 choose 2 = 3)
    assert len(result) == 3, "Expected 3 pairwise comparisons"


def test_reduce():
    """
    Test reduce function for filtering DataFrame against the best group.
    """
    data = pl.DataFrame(
        {
            "base": ["A", "A", "B", "B"],
            "compared_with": ["B", "C", "A", "C"],
            "estimate": [0.6, 0.8, 0.4, 0.3],
            "magnitude": ["medium", "large", "small", "negligible"],
        }
    )

    reduced_data = reduce(data, best="A", symbols=True)

    # Verify that only comparisons with 'A' remain
    assert all((reduced_data["base"] == "A") | (reduced_data["compared_with"] == "A")), (
        "Expected only comparisons involving 'A'"
    )
    assert "effect_size_symbol" in reduced_data.columns


def test_reduce_includes_bigstar_for_best_comparison():
    r"""Covers get_symbol() branch returning '$\\bigstar$'."""
    data = pl.DataFrame(
        {
            "base": ["A", "B"],
            "compared_with": ["A", "A"],
            "estimate": [0.5, 0.6],
            "magnitude": ["negligible", "small"],
        }
    )

    reduced_data = reduce(data, best="A", symbols=True)
    assert "$\\bigstar$" in reduced_data["effect_size_symbol"].to_list()


def test_reduce_fallback_symbol_empty_when_temp_not_matchable():
    """Cover fallback branch returning empty symbol (line 187)."""
    data = pl.DataFrame(
        {
            "base": ["A"],
            "compared_with": [None],
            "estimate": [0.5],
            "magnitude": ["negligible"],
        }
    )

    reduced_data = reduce(data, best="A", symbols=True)
    assert "" in reduced_data["effect_size_symbol"].to_list()


def test_vd_a_negligible():
    """
    Test vd_a function for negligible effect size.

    This test uses two groups, group_f and group_g, with almost equal distributions.
    The expected result is a negligible effect size with the magnitude 'negligible'.
    """
    group_f = [0.8236, 0.7967, 0.9236, 0.8197, 0.7108]
    group_g = [0.8053, 0.8172, 0.8322, 0.7836, 0.8142]
    result, magnitude = vd_a(group_g, group_f)
    assert 0.44 <= result <= 0.55, "Expected negligible effect size"
    assert magnitude == "negligible", f"Expected 'negligible', got {magnitude}"


def test_vd_a_small():
    """
    Test vd_a function for small effect size.

    This test uses two groups, group_a and group_b, where group_a slightly outperforms group_b.
    The expected result is a small effect size with the magnitude 'small'.
    """
    group_a = [
        0.4785,
        0.4639,
        0.4639,
        0.4697,
        0.4639,
        0.4746,
        0.4814,
        0.4814,
        0.4697,
        0.4814,
        0.4746,
        0.4834,
        0.4844,
        0.4492,
        0.4746,
        0.4844,
        0.4814,
        0.4639,
        0.4844,
        0.4785,
        0.4785,
        0.4570,
        0.4844,
        0.4199,
        0.4834,
        0.4785,
        0.4697,
        0.4844,
        0.4785,
        0.4639,
    ]
    group_b = [
        0.4814,
        0.4785,
        0.4492,
        0.4814,
        0.4639,
        0.4785,
        0.4746,
        0.4639,
        0.4746,
        0.4492,
        0.4746,
        0.4785,
        0.4785,
        0.4746,
        0.4697,
        0.4746,
        0.4570,
        0.4697,
        0.4785,
        0.4697,
        0.4697,
        0.4844,
        0.4570,
        0.4746,
        0.4746,
        0.4639,
        0.4570,
        0.4746,
        0.4639,
        0.4307,
    ]
    result, magnitude = vd_a(group_a, group_b)
    assert 0.55 <= result <= 0.65, "Expected small effect size"
    assert magnitude == "small", f"Expected 'small', got {magnitude}"


def test_vd_a_medium():
    """
    Test vd_a function for medium effect size.

    This test uses two groups, group_c and group_e, where group_c moderately outperforms group_e.
    The expected result is a medium effect size with the magnitude 'medium'.
    """
    group_c = [0.9108, 0.8756, 0.9003, 0.9275, 0.8778]
    group_e = [0.8664, 0.8803, 0.7817, 0.8378, 0.9306]
    result, magnitude = vd_a(group_c, group_e)
    assert 0.65 <= result <= 0.75, "Expected medium effect size"
    assert magnitude == "medium", f"Expected 'medium', got {magnitude}"


def test_vd_a_large():
    """
    Test vd_a function for large effect size.

    This test uses two groups, group_c and group_d, where group_c significantly outperforms group_d.
    The expected result is a large effect size with the magnitude 'large'.
    """
    group_c = [0.9108, 0.8756, 0.9003, 0.9275, 0.8778]
    group_d = [0.7203, 0.7700, 0.8544, 0.7947, 0.7578]
    result, magnitude = vd_a(group_c, group_d)
    assert result > 0.75, "Expected large effect size"
    assert magnitude == "large", f"Expected 'large', got {magnitude}"


def test_vd_a_invalid_input():
    """
    Test vd_a function for invalid inputs.

    This test validates that the function raises appropriate errors for invalid inputs such as:
    - An empty list
    - Non-list inputs
    """
    with pytest.raises(ValueError):
        vd_a([], [1, 2, 3])  # Empty list
    with pytest.raises(ValueError):
        vd_a([1, 2, 3], cast(Any, "invalid"))  # Non-list input


def test_vd_a_distribution_comparisons():
    """
    Test vd_a function with values sampled from uniform and normal distributions.
    Validates that the A value and magnitude are meaningful when comparing different distributions.
    """
    rng = np.random.default_rng(2)
    valid_magnitudes = ["negligible", "small", "medium", "large"]

    # Uniform distribution with distinct ranges
    a_uniform = rng.uniform(0.5, 0.75, 10)
    b_uniform = rng.uniform(0.8, 1, 10)
    estimate_uniform, magnitude_uniform = vd_a(a_uniform.tolist(), b_uniform.tolist())
    print(f"Uniform a & b: {estimate_uniform}, {magnitude_uniform}")

    # Allow A value of 0.0 if control dominates treatment
    assert 0.0 <= estimate_uniform <= 1.0, "Expected valid A value for uniform distributions"
    assert magnitude_uniform == "large", "Expected 'large' magnitude when A is 0.0 for uniform"

    # Normal distribution
    a_normal = rng.normal(62.8125, 134, 10)
    b_normal = rng.normal(10.3199, 1.124, 10)
    estimate_normal, magnitude_normal = vd_a(a_normal.tolist(), b_normal.tolist())
    print(f"Normal a & b: {estimate_normal}, {magnitude_normal}")

    # Normal distributions with high variance may not produce extreme A values
    assert 0.0 <= estimate_normal <= 1.0, "Expected valid A value for normal distributions"

    # Ensure the magnitude is valid for all cases
    assert magnitude_uniform in valid_magnitudes, f"Unexpected magnitude: {magnitude_uniform}"
    assert magnitude_normal in valid_magnitudes, f"Unexpected magnitude: {magnitude_normal}"


# ---------------------------------------------------------------------------
# Additional edge-case coverage
# ---------------------------------------------------------------------------


def test_reduce_symbols_false():
    """reduce(symbols=False) should return filtered rows without the effect_size_symbol column."""
    data = pl.DataFrame(
        {
            "base": ["A", "A", "B"],
            "compared_with": ["B", "C", "C"],
            "estimate": [0.7, 0.6, 0.4],
            "magnitude": ["medium", "small", "negligible"],
        }
    )
    result = reduce(data, best="A", symbols=False)
    assert "effect_size_symbol" not in result.columns
    assert all((result["base"] == "A") | (result["compared_with"] == "A"))


def test_reduce_best_not_in_data_returns_empty():
    """Return empty result when best value is absent from the DataFrame."""
    data = pl.DataFrame(
        {
            "base": ["A", "B"],
            "compared_with": ["B", "C"],
            "estimate": [0.6, 0.4],
            "magnitude": ["small", "negligible"],
        }
    )
    result = reduce(data, best="X", symbols=False)
    assert result.height == 0


def test_vd_a_df_raises_without_val_col():
    """vd_a_df should raise ValueError when val_col is not provided."""
    data = pl.DataFrame({"values": [1, 2, 3, 4], "group": ["A", "A", "B", "B"]})
    with pytest.raises(ValueError, match="val_col"):
        vd_a_df(data, group_col="group")


def test_vd_a_df_raises_without_group_col():
    """vd_a_df should raise ValueError when group_col is not provided."""
    data = pl.DataFrame({"values": [1, 2, 3, 4], "group": ["A", "A", "B", "B"]})
    with pytest.raises(ValueError, match="group_col"):
        vd_a_df(data, val_col="values")


def test_vd_a_df_sort_false():
    """vd_a_df with sort=False should still return valid pairwise comparisons."""
    data = pl.DataFrame({"values": [1.0, 2.0, 3.0, 4.0], "group": ["A", "A", "B", "B"]})
    result = vd_a_df(data, val_col="values", group_col="group", sort=False)
    assert result.height == 1
    assert set(result.columns) >= {"base", "compared_with", "estimate", "magnitude"}


def test_reduce_symbol_for_best_itself():
    r"""When the 'best' column matches itself, the symbol should be \bigstar."""
    data = pl.DataFrame(
        {
            "base": ["A"],
            "compared_with": ["A"],
            "estimate": [0.5],
            "magnitude": ["negligible"],
        }
    )
    result = reduce(data, best="A", symbols=True)
    assert result.height == 1
    assert result["effect_size_symbol"][0] == "$\\bigstar$"
