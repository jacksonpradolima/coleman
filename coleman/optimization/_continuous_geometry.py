"""Compile declared coefficient geometry to finite exact halfspaces."""

from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from coleman.optimization.continuous_contracts import ContinuousSet


class Halfspace(NamedTuple):
    """Keep one original rational normal and inclusive upper bound."""

    normal: tuple[Fraction, ...]
    bound: Fraction


class Geometry(NamedTuple):
    """Retain canonical inequalities and optional sum-one equality."""

    inequalities: tuple[Halfspace, ...]
    equality: Halfspace | None


def dot(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> Fraction:
    """Evaluate an exact small-coordinate inner product.

    Parameters
    ----------
    left, right : tuple of Fraction
        Equal-length vectors of at most4 original rational coordinates.

    Returns
    -------
    Fraction
        Exact product sum without intermediate binary64 rounding.
    """
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction())


def _extra_constraints(declared: ContinuousSet) -> tuple[Halfspace, ...]:
    """Expand exact net/cash and optional L1 sign inequalities."""
    count = len(declared.variables)
    one = (Fraction(1),) * count
    minus = (Fraction(-1),) * count
    rows = [
        Halfspace(one, Fraction(declared.net.upper)),
        Halfspace(minus, -Fraction(declared.net.lower)),
        Halfspace(one, 1 - Fraction(declared.cash.lower)),
        Halfspace(minus, Fraction(declared.cash.upper) - 1),
    ]
    if declared.gross_limit is not None:
        rows.extend(
            Halfspace(tuple(Fraction(value) for value in signs), Fraction(declared.gross_limit))
            for signs in product((-1, 1), repeat=count)
        )
    if declared.leverage_limit is not None:
        rows.extend(
            Halfspace(tuple(Fraction(value - cash) for value in signs), Fraction(declared.leverage_limit) - cash)
            for signs in product((-1, 1), repeat=count)
            for cash in (-1, 1)
        )
    return tuple(rows)


def linear_geometry(declared: ContinuousSet) -> Geometry:
    """Compile a compact declared set without silently weakening a limit.

    Parameters
    ----------
    declared : ContinuousSet
        Validated one-to-four-coordinate geometry with finite original boxes.

    Returns
    -------
    Geometry
        At most60 inequalities, no greater than one sum-one equality. Exact
        duplicate rows and extra inequalities implied by original boxes are
        removed deterministically; original box constraints remain present.

    Notes
    -----
    L1 sign expansion is exact, not an estimate of exposure or financial risk.
    Constraint order is coordinate upper/lower, net/cash, gross, then leverage.
    """
    count = len(declared.variables)
    rows: list[Halfspace] = []
    for index, variable in enumerate(declared.variables):
        normal = tuple(Fraction(int(index == j)) for j in range(count))
        rows.append(Halfspace(normal, Fraction(variable.upper)))
        rows.append(Halfspace(tuple(-value for value in normal), -Fraction(variable.lower)))
    for row in _extra_constraints(declared):
        maximum = sum(
            (
                value * Fraction(variable.upper if value >= 0 else variable.lower)
                for value, variable in zip(row.normal, declared.variables, strict=True)
            ),
            Fraction(),
        )
        if maximum > row.bound:
            rows.append(row)
    equality = Halfspace((Fraction(1),) * count, Fraction(1)) if declared.profile == "simplex-v1" else None
    return Geometry(tuple(dict.fromkeys(rows)), equality)


def primal_residual(point: tuple[Fraction, ...], geometry: Geometry) -> Fraction:
    """Return the largest original constraint violation, including sum-one.

    Parameters
    ----------
    point : tuple of Fraction
        Original vector whose feasibility is being checked.
    geometry : Geometry
        Exact compiled inequalities and optional sum-one equality.

    Returns
    -------
    Fraction
        Nonnegative largest violation;0 means exact feasibility.
    """
    violations = tuple(max(Fraction(), dot(row.normal, point) - row.bound) for row in geometry.inequalities)
    equality = geometry.equality
    residual = Fraction() if equality is None else abs(dot(equality.normal, point) - equality.bound)
    return max((residual, *violations))
