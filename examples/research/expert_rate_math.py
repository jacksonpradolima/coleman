"""Independent ideal arithmetic for the public expert-rate entry experiment.

This module imports no COLEMAN learner, framework, transport or numerical
helper. It consumes the same once-converted binary64 loss observations, then
uses Decimal80 recurrences and exact Fraction scores. Related authored paths
and repeated physical execution do not become independent observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from math import isfinite
from typing import cast

# Exact kinds prevent bool/subclass conversion hooks at this scientific boundary.
# pylint: disable=unidiomatic-typecheck

type Vector = tuple[float, float, float]
type ExactVector = tuple[Decimal, Decimal, Decimal]


@dataclass(frozen=True)
class OracleTrace:
    """Keep independent before/after ideal coefficients, not executed actions."""

    before: tuple[ExactVector, ...]
    after: tuple[ExactVector, ...]


def observations(rows: tuple[Vector, ...]) -> tuple[Vector, ...]:
    """Validate the finite three-expert observation profile without conversion.

    Parameters
    ----------
    rows : tuple of tuple of float
        At most64 complete finite binary64 loss vectors in dimensionless[0,1].

    Returns
    -------
    tuple
        The original immutable observations, without clipping or normalization.

    Raises
    ------
    ValueError
        Shape, Python type, finiteness or bounds are outside the frozen profile.
    """
    if type(rows) is not tuple or not 1 <= len(rows) <= 64:
        raise ValueError("The entry oracle requires one through64 complete panels")
    if any(type(row) is not tuple or len(row) != 3 for row in rows):
        raise ValueError("The entry oracle requires three complete coordinates")
    if any(type(x) is not float or not isfinite(x) or not 0 <= x <= 1 for row in rows for x in row):
        raise ValueError("The entry oracle requires finite observed binary64 losses in[0,1]")
    return rows


def _exponential(prior: ExactVector, losses: ExactVector, rate: Decimal) -> ExactVector:
    """Use scalar Decimal exponentials, independently of the production kernel."""
    pivot = min(losses)
    raw = tuple(p * (-rate * (loss - pivot)).exp() for p, loss in zip(prior, losses, strict=True))
    total = sum(raw)
    return raw[0] / total, raw[1] / total, raw[2] / total


def _share_decimal(current: ExactVector, alpha: float) -> ExactVector:
    """Use the explicit independent three-by-three excluding-origin transition."""
    fraction = Decimal.from_float(alpha)
    matrix = tuple(tuple(1 - fraction if i == j else fraction / 2 for j in range(3)) for i in range(3))
    return cast(ExactVector, tuple(sum(matrix[i][j] * current[j] for j in range(3)) for i in range(3)))


def _rate(value: float | None, *, positive: bool = False) -> bool:
    """Admit only builtin binary64 rates within this bounded oracle envelope."""
    return type(value) is float and isfinite(value) and (0 < value <= 4 if positive else 0 <= value <= 4)


def _parameters(profile, eta, multiplier, alpha) -> None:
    """Refuse unsupported profile/parameter combinations before arithmetic."""
    if profile not in {"hedge-constant", "hedge-anytime-cumulative", "fixed-share-excluding-origin"}:
        raise ValueError("Unknown reference profile")
    if profile == "hedge-anytime-cumulative":
        if eta is not None or alpha is not None or not _rate(multiplier, positive=True):
            raise ValueError("Anytime oracle requires only its positive finite multiplier")
    elif not _rate(eta) or multiplier is not None:
        raise ValueError("Constant oracle requires a finite nonnegative rate")
    if (profile == "fixed-share-excluding-origin") != (alpha is not None) or (
        alpha is not None and (not _rate(alpha) or alpha > 1)
    ):
        raise ValueError("Only excluding-origin share admits alpha in[0,1]")


def reference_trace(
    profile: str,
    rows: tuple[Vector, ...],
    *,
    eta: float | None = None,
    multiplier: float | None = None,
    alpha: float | None = None,
) -> OracleTrace:
    """Compute independent constant/anytime/excluding-origin reference traces.

    Parameters
    ----------
    profile : str
        Exactly one of the three existing public reference profile names.
    rows : tuple of tuple of float
        The same complete dimensionless binary64 observations as the learner.
    eta : float or None, optional
        Builtin float constant rate in[0,4], including the frozen study grid.
    multiplier : float or None, optional
        Builtin float c in(0,4] for eta_t=c*sqrt(log(3)/t), only for anytime Hedge.
    alpha : float or None, optional
        Builtin float excluding-origin share in[0,1], only for Fixed Share.

    Returns
    -------
    OracleTrace
        Decimal80 ideal coefficients before/after each accepted observation.

    Raises
    ------
    ValueError
        Unsupported profile, observations or incompatible rate/share parameters.

    Notes
    -----
    The uniform ideal prior is exactly1/3. This is a separately labeled ideal
    calculation; actual emitted binary64 coefficients retain their residual.
    The anytime recurrence uses the CURRENT next rate on cumulative losses,
    never historical-rate products. No production update/helper is called.
    Decimal80 agreement is qualified for this three-member/64-panel envelope
    and frozen bounded rates; arbitrary large-rate arithmetic is unsupported.
    """
    observations(rows)
    _parameters(profile, eta, multiplier, alpha)
    before, after = [], []
    with localcontext() as context:
        context.prec = 80
        prior: ExactVector = (Decimal(1) / 3, Decimal(1) / 3, Decimal(1) / 3)
        current = prior
        totals: ExactVector = (Decimal(0), Decimal(0), Decimal(0))
        for count, row in enumerate(rows, start=1):
            before.append(current)
            loss = cast(ExactVector, tuple(Decimal.from_float(x) for x in row))
            totals = cast(ExactVector, tuple(x + y for x, y in zip(totals, loss, strict=True)))
            if profile == "hedge-anytime-cumulative":
                assert multiplier is not None
                rate = Decimal.from_float(multiplier) * (Decimal(3).ln() / (count + 1)).sqrt()
                current = _exponential(prior, totals, rate)
            else:
                assert eta is not None
                current = _exponential(current, loss, Decimal.from_float(eta))
            if profile == "fixed-share-excluding-origin":
                assert alpha is not None
                current = _share_decimal(current, alpha)
            after.append(current)
    return OracleTrace(tuple(before), tuple(after))


def periodic_trace(rows: tuple[Vector, ...], window: int) -> OracleTrace:
    """Recompute periodic coefficients from eligible source slices only.

    Parameters
    ----------
    rows : tuple of tuple of float
        Complete once-converted binary64 losses, at most64.
    window : int
        Exactly8,16,32or64 trailing observations, with an eight-panel refit.

    Returns
    -------
    OracleTrace
        Ideal coefficients from independent full-prefix recomputation.

    Raises
    ------
    ValueError
        Invalid observations or an undeclared trailing-window setting.

    Notes
    -----
    Unlike an incremental host baseline, every expected vector is recomputed
    from original prefix indexes. Fraction compares exact binary64 values.
    Uniform ties are retained and no future observation enters a refit.
    """
    observations(rows)
    if type(window) is not int or window not in (8, 16, 32, 64):
        raise ValueError("The periodic oracle requires a frozen trailing window")

    def at(consumed: int) -> ExactVector:
        cutoff = consumed - consumed % 8
        if cutoff == 0:
            return Decimal(1) / 3, Decimal(1) / 3, Decimal(1) / 3
        totals = tuple(
            sum((Fraction(row[i]) for row in rows[max(0, cutoff - window) : cutoff]), Fraction()) for i in range(3)
        )
        best = min(totals)
        ties = sum(value == best for value in totals)
        return cast(ExactVector, tuple(Decimal(1) / ties if value == best else Decimal(0) for value in totals))

    with localcontext() as context:
        context.prec = 80
        return OracleTrace(tuple(at(t) for t in range(len(rows))), tuple(at(t + 1) for t in range(len(rows))))


def score(weights: tuple[Vector, ...], rows: tuple[Vector, ...]) -> Fraction:
    """Sum actual emitted-weight dot products exactly, retaining simplex residual.

    Parameters
    ----------
    weights, rows : tuple of tuple of float
        Matched three-coordinate binary64 weights BEFORE each observed panel.

    Returns
    -------
    fractions.Fraction
        Exact sum of stored binary64 products, in cumulative loss units1.

    Raises
    ------
    ValueError
        Coordinate shape, cardinality, type or finite input is invalid.
    """
    observations(rows)
    observations(weights)
    if len(weights) != len(rows):
        raise ValueError("Scores require one original emitted vector per observation")
    return sum(
        (
            Fraction(p) * Fraction(loss)
            for vector, row in zip(weights, rows, strict=True)
            for p, loss in zip(vector, row, strict=True)
        ),
        Fraction(),
    )


def comparator_losses(rows: tuple[Vector, ...]) -> tuple[Fraction, Fraction]:
    """Enumerate exact best fixed and zero/one-switch retrospective comparators.

    Parameters
    ----------
    rows : tuple of tuple of float
        Complete matched binary64 observations, at most64.

    Returns
    -------
    tuple of fractions.Fraction
        Best fixed loss and best loss with at most one switch, both unit1.

    Raises
    ------
    ValueError
        Observations do not satisfy the declared finite three-expert profile.

    Notes
    -----
    Exhaustive splits include zero and T; these are oracle diagnostics, never
    deployable policies. Exact stored binary64 values determine their totals.
    """
    observations(rows)
    prefixes = tuple(
        tuple(sum((Fraction(row[i]) for row in rows[:t]), Fraction()) for i in range(3)) for t in range(len(rows) + 1)
    )
    fixed = min(prefixes[-1])
    switched = min(
        prefixes[t][i] + prefixes[-1][j] - prefixes[t][j]
        for t in range(len(rows) + 1)
        for i in range(3)
        for j in range(3)
    )
    return fixed, switched
