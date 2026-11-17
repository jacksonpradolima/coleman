"""Bounded exact small linear algebra, without numerical jitter or inversion."""

from fractions import Fraction
from typing import TYPE_CHECKING

from coleman.objectives.arithmetic import checked_fraction
from coleman.optimization._continuous_geometry import dot

if TYPE_CHECKING:
    from coleman.optimization.continuous_contracts import QuadraticObjective


class ArithmeticLimitError(ArithmeticError):
    """Refuse derived rational work outside its declared finite envelope."""

    def __init__(self, reason: str, *, faces_examined: int = 0) -> None:
        """Retain the deterministic failure and actual attempted-face count.

        Parameters
        ----------
        reason : str
            Finite internal failure category; no caller callback is performed.
        faces_examined : int
            Actual active systems attempted before failure, default0.
        """
        super().__init__(reason)
        self.faces_examined = faces_examined


def bounded(value: Fraction, maximum_bits: int) -> Fraction:
    """Require bounded exact numerator and denominator, returning the input.

    Parameters
    ----------
    value : Fraction
        Exact operation result, never a callback-bearing scalar.
    maximum_bits : int
        Declared per-integer cap from the complete solve request.

    Returns
    -------
    Fraction
        Original result if both integer sizes fit.

    Raises
    ------
    ArithmeticLimitError
        Numerator or denominator exceeds the selected work envelope.
    """
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > maximum_bits:
        raise ArithmeticLimitError("derived_rational_bit_limit")
    return value


def emit(values: tuple[Fraction, ...]) -> tuple[float, ...]:
    """Round once to finite binary64, refusing nonzero underflow/overflow.

    Parameters
    ----------
    values : tuple of Fraction
        Bounded exact intermediates in original coordinate order.

    Returns
    -------
    tuple of float
        One nearest finite rounding per coordinate; exact0 becomes+0.

    Raises
    ------
    ArithmeticLimitError
        Any nonzero coordinate underflows or a coordinate overflows.
    """
    try:
        return tuple(checked_fraction(value) for value in values)
    except ValueError as error:
        raise ArithmeticLimitError("binary64_emission_failure") from error


def solve_linear(
    matrix: tuple[tuple[Fraction, ...], ...], vector: tuple[Fraction, ...], maximum_bits: int
) -> tuple[Fraction, ...] | None:
    """Solve a finite square system by deterministic exact Gauss-Jordan steps.

    Parameters
    ----------
    matrix, vector : tuple
        Matching square rational system, at most8 by8 for this solver.
    maximum_bits : int
        Per-derived-integer work cap, not a convergence tolerance.

    Returns
    -------
    tuple or None
        Unique exact solution, or null for a singular active-face system.

    Raises
    ------
    ArithmeticLimitError
        An exact intermediate exceeds the declared integer envelope.
    """
    size = len(vector)
    rows = [
        [bounded(item, maximum_bits) for item in row] + [bounded(value, maximum_bits)]
        for row, value in zip(matrix, vector, strict=True)
    ]
    for column in range(size):
        pivot = next((index for index in range(column, size) if rows[index][column]), None)
        if pivot is None:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [bounded(value / divisor, maximum_bits) for value in rows[column]]
        _eliminate(rows, column, maximum_bits)
    return tuple(row[-1] for row in rows)


def _eliminate(rows: list[list[Fraction]], column: int, maximum_bits: int) -> None:
    """Remove one pivot column without changing deterministic row order."""
    for index, row in enumerate(rows):
        if index != column:
            factor = row[column]
            rows[index] = [bounded(a - factor * b, maximum_bits) for a, b in zip(row, rows[column], strict=True)]


def positive_definite(matrix: tuple[tuple[Fraction, ...], ...], maximum_bits: int) -> bool:
    """Check symmetric positive definiteness by exact leading Schur pivots.

    Parameters
    ----------
    matrix : tuple
        Already admitted symmetric rational matrix, dimension at most4.
    maximum_bits : int
        Declared intermediate cap; no small-pivot tolerance is substituted.

    Returns
    -------
    bool
        True only when every exact Schur pivot is strictly positive.

    Raises
    ------
    ArithmeticLimitError
        A derived rational intermediate exceeds the declared bit cap.
    """
    rows = [list(row) for row in matrix]
    for index, row in enumerate(rows):
        pivot = row[index]
        if pivot <= 0:
            return False
        for following in range(index + 1, len(rows)):
            for column in range(index + 1, len(rows)):
                rows[following][column] = bounded(
                    rows[following][column] - rows[following][index] * row[column] / pivot, maximum_bits
                )
    return True


def gradient(
    objective: QuadraticObjective, point: tuple[Fraction, ...], maximum_bits: int = 32768
) -> tuple[Fraction, ...]:
    """Evaluate the exact analytic gradient of the declared coupled quadratic.

    Parameters
    ----------
    objective : QuadraticObjective
        Complete symmetric matrix and linear dimensionless score law.
    point : tuple of Fraction
        Original matching allocation coordinate order.
    maximum_bits : int
        Maximum numerator/denominator bits in each resulting coordinate.

    Returns
    -------
    tuple of Fraction
        Q times point plus the declared linear term, without finite differences.
    """
    return tuple(
        bounded(dot(tuple(Fraction(value) for value in row), point) + Fraction(linear), maximum_bits)
        for row, linear in zip(objective.matrix, objective.linear, strict=True)
    )


def objective_value(objective: QuadraticObjective, point: tuple[Fraction, ...], maximum_bits: int = 32768) -> Fraction:
    """Evaluate the original scalar quadratic score with its declared offset.

    Parameters
    ----------
    objective : QuadraticObjective
        Original half-quadratic score, including its finite constant.
    point : tuple of Fraction
        Matching original dimensionless allocation coordinates.
    maximum_bits : int
        Selected numerator/denominator bound on the resulting scalar.

    Returns
    -------
    Fraction
        Exact score of the supplied point, not a realized execution charge.
    """
    products = tuple(dot(tuple(Fraction(value) for value in row), point) for row in objective.matrix)
    return bounded(
        dot(point, products) / 2
        + dot(tuple(Fraction(x) for x in objective.linear), point)
        + Fraction(objective.constant),
        maximum_bits,
    )
