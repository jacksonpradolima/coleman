"""Bounded rational intermediates with explicit guarded binary64 output."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Literal

ArithmeticProfile = Literal["binary64-rational-v1"]
PROFILE: ArithmeticProfile = "binary64-rational-v1"


def checked_fraction(value: Fraction) -> float:
    """Project an exact rational operation result to finite binary64 once.

    Parameters
    ----------
    value : Fraction
        Exact intermediate from a bounded operation on validated stored values.

    Returns
    -------
    float
        Binary64-rounded finite result; an exact rational zero becomes +0.0.

    Raises
    ------
    ValueError
        Overflow or nonzero underflow prevents representing the operation result.
    TypeError
        Input is not a built-in Fraction; conversion callbacks are not invoked.

    Notes
    -----
    ``binary64-rational-v1`` does not inherit ambient Decimal precision or perform
    a sequence of unchecked float intermediates. This is mathematical projection,
    not exact financial accounting. It performs no rounding-to-unit, repair or
    I/O. Callers preserve signed input zero separately when identity requires it.
    """
    # pylint: disable-next=unidiomatic-typecheck
    if type(value) is not Fraction:
        raise TypeError("Objective intermediates require a built-in Fraction")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("Objective arithmetic overflowed binary64") from exc
    if not math.isfinite(result) or (value != 0 and result == 0):
        raise ValueError("Objective arithmetic is nonfinite or underflows binary64")
    return result
