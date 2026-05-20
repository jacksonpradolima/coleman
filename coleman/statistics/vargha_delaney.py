"""
Effect Size Computation for Statistical Analysis.

This module implements methods for computing and analyzing effect sizes, specifically
Vargha and Delaney's A index, which measures the probability that one random observation
from a treatment group is larger than a random observation from a control group.

Functions
---------
vd_a
    Computes the Vargha and Delaney A index for two groups.
vd_a_df
    Computes pairwise A index comparisons for multiple groups in a Polars DataFrame.
reduce
    Filters and annotates effect sizes for comparisons against a specified best group.

References
----------
.. [1] A. Vargha and H. D. Delaney, "A critique and improvement of the CL common
   language effect size statistics of McGraw and Wong," Journal of Educational and
   Behavioral Statistics, vol. 25, no. 2, pp. 101-132, 2000.
.. [2] Hess and Kromrey, 2004, for thresholds of effect size magnitudes.
"""

import itertools as it
from bisect import bisect_left

import numpy as np
import polars as pl
import scipy.stats as ss


def vd_a(treatment: list[float], control: list[float]) -> tuple[float, str]:
    """Compute Vargha and Delaney A index.

    The formula to compute A has been transformed to minimize accuracy errors.

    Parameters
    ----------
    treatment : list of float
        A numeric list representing the treatment group.
    control : list of float
        A numeric list representing the control group.

    Returns
    -------
    estimate : float
        The A index value estimate.
    magnitude : str
        The effect size magnitude (negligible, small, medium, or large).

    Raises
    ------
    ValueError
        If treatment and control lists have different lengths.

    References
    ----------
    .. [1] A. Vargha and H. D. Delaney. "A critique and improvement of the CL
       common language effect size statistics of McGraw and Wong." Journal of
       Educational and Behavioral Statistics, 25(2):101-132, 2000.
    """
    m = len(treatment)
    n = len(control)

    if m != n:
        raise ValueError(f"treatment and control must have the same length, got {m} and {n}")

    r = ss.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    # estimate = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    estimate = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_a = (estimate - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_a))]

    return estimate, magnitude


def vd_a_df(
    data: pl.DataFrame,
    val_col: str | None = None,
    group_col: str | None = None,
    sort: bool = True,
) -> pl.DataFrame:
    """Compute pairwise Vargha and Delaney A index for groups in a DataFrame.

    Parameters
    ----------
    data : polars.DataFrame
        A polars DataFrame with two-dimensional data. Each pair of groups
        must have the same number of observations (equal-length groups).
    val_col : str, optional
        Name of the column that contains values.
    group_col : str, optional
        Name of the column that contains group names.
    sort : bool, optional
        Whether to sort the DataFrame by group_col. Default is True.

    Returns
    -------
    polars.DataFrame
        DataFrame of effect sizes with columns: 'base', 'compared_with',
        'estimate', and 'magnitude'.
    """
    x = data.clone()

    if val_col is None or group_col is None:
        raise ValueError("Both val_col and group_col must be provided")

    if sort:
        x = x.with_columns([pl.col(group_col).cast(pl.Categorical)]).sort([group_col, val_col])

    groups = x[group_col].unique()
    groups_list = groups.to_list()

    # Pairwise combinations
    g1, g2 = np.array(list(it.combinations(np.arange(len(groups_list)), 2))).T

    # Compute effect size for each combination
    ef = np.array(
        [
            vd_a(
                list(x.filter(pl.col(group_col) == groups_list[i])[val_col].to_list()),
                list(x.filter(pl.col(group_col) == groups_list[j])[val_col].to_list()),
            )
            for i, j in zip(g1, g2, strict=False)
        ]
    )

    groups_array = groups.to_numpy()
    return pl.DataFrame(
        {"base": groups_array[g1], "compared_with": groups_array[g2], "estimate": ef[:, 0], "magnitude": ef[:, 1]}
    )


def reduce(df: pl.DataFrame, best: str, symbols: bool = True) -> pl.DataFrame:
    """Reduce a DataFrame of effect sizes to compare only against the best.

    Parameters
    ----------
    df : polars.DataFrame
        A DataFrame of effect sizes.
    best : str
        The name of the best sample, for instance, algorithm AAA.
    symbols : bool, optional
        Whether to use LaTeX symbols to represent the magnitudes.
        Default is True.

    Returns
    -------
    polars.DataFrame
        Filtered DataFrame with an additional 'effect_size_symbol' column.
    """
    # Effect Size
    magnitude = ["negligible", "small", "medium", "large"]
    symbol_map = ["$\\blacktriangledown$", "$\\triangledown$", "$\\vartriangle$", "$\\blacktriangle$"]

    # Get only the effect size magnitudes related with the best
    df = df.filter((pl.col("base") == best) | (pl.col("compared_with") == best))

    if not symbols:
        return df

    # Create a new column to compare against the other policies
    df = df.with_columns(
        [pl.when(pl.col("base") == best).then(pl.col("compared_with")).otherwise(pl.col("base")).alias("temp")]
    )

    # Get magnitude symbol (in latex) for each comparison
    # The best has the bigstar symbol
    def get_symbol(row):
        """Return the LaTeX effect-size symbol for one comparison row."""
        temp = row["temp"]
        if temp == best:
            return "$\\bigstar$"
        # Find the magnitude for this temp value
        mag_row = df.filter(pl.col("temp") == temp)
        if mag_row.height > 0:
            mag = mag_row["magnitude"][0]
            return symbol_map[magnitude.index(mag)]
        return ""

    # Apply symbol mapping
    effect_symbols = [get_symbol(row) for row in df.to_dicts()]

    df = df.with_columns([pl.Series("effect_size_symbol", effect_symbols)])

    return df.drop("temp")
