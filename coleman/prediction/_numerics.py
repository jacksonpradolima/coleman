"""Finite normal-equation arithmetic for the named ridge reference only."""

from math import fsum, isfinite, sqrt

Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]


class NumericalError(ArithmeticError):
    """Keep a named numerical refusal separate from invalid input content."""

    def __init__(self, code: str, message: str) -> None:
        """Retain a finite-profile refusal without touching mathematical state.

        Parameters
        ----------
        code : str
            Closed numerical reason selected by the internal arithmetic.
        message : str
            Bounded diagnostic selected by this implementation.
        """
        super().__init__(message)
        self.code = code


def product(left: float, right: float) -> float:
    """Multiply admitted binary64 values without silently erasing nonzero mass.

    Parameters
    ----------
    left, right : float
        Already validated finite values in the declared numerical profile.

    Returns
    -------
    float
        Rounded binary64 product; a genuine zero remains zero.

    Raises
    ------
    NumericalError
        Nonfinite product or nonzero operands underflowing to zero.
    """
    result = left * right
    if not isfinite(result):
        raise NumericalError("numeric_range", "Ridge multiplication is outside finite binary64")
    if left and right and not result:
        raise NumericalError("underflow", "Ridge multiplication erases a nonzero value")
    return result


def total(values) -> float:
    """Use the fixed fsum reduction with explicit range refusal.

    Parameters
    ----------
    values : iterable of float
        Finite bounded internal terms; an empty reduction is exactly zero.

    Returns
    -------
    float
        Finite fsum result in the unchanged source order.

    Raises
    ------
    NumericalError
        Reduction overflow or a nonfinite result. No state changes occur.
    """
    try:
        result = fsum(values)
    except (OverflowError, ValueError) as error:
        raise NumericalError("numeric_range", "Ridge sum is outside finite binary64") from error
    if not isfinite(result):
        raise NumericalError("numeric_range", "Ridge sum is outside finite binary64")
    return result


def _divide(numerator: float, denominator: float) -> float:
    """Refuse nonfinite division and nonzero-to-zero quotient underflow."""
    result = numerator / denominator
    if not isfinite(result):
        raise NumericalError("numeric_range", "Ridge quotient is outside finite binary64")
    if numerator and not result:
        raise NumericalError("underflow", "Ridge division erases a nonzero value")
    return result


def _factor(gram: Matrix) -> list[list[float]]:
    """Compute unpivoted SPD Cholesky without inversion, repair or jitter."""
    size = len(gram)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            residual = total(
                (gram[row][column], -total(product(lower[row][k], lower[column][k]) for k in range(column)))
            )
            if row == column:
                if residual <= 0:
                    raise NumericalError("non_positive_pivot", "Ridge Cholesky requires a positive finite pivot")
                lower[row][column] = sqrt(residual)
            else:
                lower[row][column] = _divide(residual, lower[column][column])
    return lower


def _residual(gram: Matrix, cross: Vector, coefficients: Vector) -> None:
    """Admit a scaled normal-equation residual, not a coefficient error theorem."""
    for row, wanted in zip(gram, cross, strict=True):
        values = tuple(product(a, b) for a, b in zip(row, coefficients, strict=True))
        error = abs(total((*values, -wanted)))
        scale = max(1.0, abs(wanted), total(abs(value) for value in values))
        if error > 1e-12 + 1e-10 * scale:
            raise NumericalError("residual", "Ridge normal-equation residual exceeds its declared envelope")


def solve(gram: Matrix, cross: Vector) -> Vector:
    """Solve admitted SPD normal equations with checked Cholesky arithmetic.

    Parameters
    ----------
    gram : tuple of tuple of float
        Finite symmetric A in the fixed ordered feature basis, at most16square.
    cross : tuple of float
        Finite b in the same basis and target units.

    Returns
    -------
    tuple of float
        Checked solution. No inverse, pseudoinverse or diagonal jitter is used.

    Raises
    ------
    NumericalError
        Nonpositive pivot, underflow, nonfinite intermediate or excessive scaled
        residual. Residual acceptance is not a coefficient-error guarantee.
    """
    lower = _factor(gram)
    size = len(cross)
    intermediate = [0.0] * size
    for row in range(size):
        rest = total(product(lower[row][k], intermediate[k]) for k in range(row))
        intermediate[row] = _divide(total((cross[row], -rest)), lower[row][row])
    answer = [0.0] * size
    for row in reversed(range(size)):
        rest = total(product(lower[k][row], answer[k]) for k in range(row + 1, size))
        answer[row] = _divide(total((intermediate[row], -rest)), lower[row][row])
    result = tuple(answer)
    _residual(gram, cross, result)
    return result


def advance(gram: Matrix, cross: Vector, features: Vector, target: float) -> tuple[Matrix, Vector, Vector]:
    """Prepare one rank-one append without changing the supplied state.

    Parameters
    ----------
    gram, cross : tuple
        Original immutable finite sufficient statistics.
    features : tuple of float
        Same-schema dimensionless vector in its original order.
    target : float
        One finite observed label in the declared target units.

    Returns
    -------
    tuple
        New A, b and checked coefficients. Counts are owned by the caller.

    Raises
    ------
    NumericalError
        Product, reduction or solve refusal. No partial successor is assigned.
    """
    updated = tuple(
        tuple(total((value, product(features[row], features[column]))) for column, value in enumerate(old))
        for row, old in enumerate(gram)
    )
    rhs = tuple(total((value, product(feature, target))) for value, feature in zip(cross, features, strict=True))
    return updated, rhs, solve(updated, rhs)
