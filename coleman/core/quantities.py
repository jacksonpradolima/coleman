"""Exact nonnegative decimal quantities shared by optional domain adapters.

The established quantity limits, canonical strings and integer arithmetic are
unchanged. No solver, persistence or numerical backend is imported here.
"""

from __future__ import annotations

from decimal import MAX_EMAX, MIN_EMIN, ROUND_HALF_EVEN, Context, Decimal, localcontext
from typing import Annotated, Any, cast

from pydantic import Field, ValidatorFunctionWrapHandler, field_serializer, field_validator

from coleman.core.base import Immutable
from coleman.core.units import UnitSemantics

SCALE = 10**12


def ticks(value: Decimal) -> int:
    """Convert a validated finite decimal to exact signed 10^-12 ticks.

    Parameters
    ----------
    value : decimal.Decimal
        Already finite input; negative values are supported by this arithmetic
        helper even though ``Quantity`` permits only nonnegative values.

    Returns
    -------
    int
        Exact integer ticks, without rounding or ambient-context dependence.

    Raises
    ------
    ValueError, OverflowError
        Nonfinite input or more than twelve effective decimal places.
    """
    numerator, denominator = value.as_integer_ratio()
    scaled, remainder = divmod(numerator * SCALE, denominator)
    if remainder:
        raise ValueError("Quantity exceeds twelve decimal places")
    return scaled


def decimal_value(value: int, *, places: int = 12) -> Decimal:
    """Convert scaled integers exactly at 12/24/48 places, without Decimal-context rounding.

    Default precision preserves existing coverage artifacts. Product moments use
    24/48 places; unsupported precision raises ValueError rather than rounding.

    Parameters
    ----------
    value : int
        Signed integer count at the requested scale.
    places : int, optional
        Exactly 12, 24 or 48 fractional places; boolean values reject.

    Returns
    -------
    decimal.Decimal
        Exact decimal without mutating the caller's Decimal context.

    Raises
    ------
    ValueError
        The precision is unsupported.
    """
    if isinstance(places, bool) or not isinstance(places, int) or places not in {12, 24, 48}:
        raise ValueError("Supported exact decimal scales are 12, 24 and 48 places")
    whole, fraction = divmod(abs(value), 10**places)
    sign = "-" if value < 0 else ""
    return Decimal(f"{sign}{whole}.{fraction:0{places}d}")


def decimal_text(value: Decimal) -> str:
    """Return context-independent finite decimal spelling with unsigned zero.

    Parameters
    ----------
    value : decimal.Decimal
        Already validated finite value at a caller-bounded precision/range.

    Returns
    -------
    str
        Fixed-point spelling without fractional trailing zeros. This low-level
        formatter does not impose ``Quantity`` limits or validate nonfinites.
    """
    if not value:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def validate_decimal_places(value: Decimal, *, places: int = 12) -> Decimal:
    """Enforce effective fractional precision without rounding or expanding compact exponents.

    Decimal-context underflow must not make a nonzero out-of-contract value
    appear to have zero fractional places. Trailing coefficient zeros do not
    add precision; signed zero remains valid. Finite Decimal input is returned
    unchanged. Invalid precision/nonfinite values raise ValueError, with no
    powers, rational denominators, file access or context mutation.

    Parameters
    ----------
    value : decimal.Decimal
        Original finite decimal; its sign and value are retained.
    places : int, optional
        Nonnegative integer fractional precision limit, default 12.

    Returns
    -------
    decimal.Decimal
        The unchanged input when its effective precision is within the limit.

    Raises
    ------
    ValueError
        Invalid precision, nonfinite input or excessive fractional precision.
    """
    # Preserve the established integer-only contract, excluding bool and subclasses.
    if type(places) is not int or places < 0:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Decimal places must be a nonnegative integer")
    if not value.is_finite():
        raise ValueError("Decimal quantity must be finite")
    if not value:
        return value
    parts = value.as_tuple()
    trailing = 0
    for digit in reversed(parts.digits):
        if digit:
            break
        trailing += 1
    if int(parts.exponent) + trailing < -places:
        raise ValueError(f"Quantity exceeds {places} decimal places")
    return value


def validate_decimal_schema(value: Any, handler: ValidatorFunctionWrapHandler) -> Decimal:
    """Run declared Decimal field constraints in a deterministic local context.

    The 110-digit validation context covers current input/output fields and uses
    the backend's widest exponent range. Ambient precision, traps and flags are
    never modified. Exact place checks still validate the original returned
    Decimal: context normalization cannot certify fractional precision.
    Pydantic validation errors propagate; this helper performs no I/O.

    Parameters
    ----------
    value : object
        Original field input for the supplied Pydantic validator.
    handler : pydantic.ValidatorFunctionWrapHandler
        Validator enforcing the selected field's declared Decimal constraints.

    Returns
    -------
    decimal.Decimal
        Validated field value. The caller still checks effective places.

    Raises
    ------
    pydantic.ValidationError
        The input fails the selected field's constraints.
    """
    context = Context(prec=110, Emin=MIN_EMIN, Emax=MAX_EMAX, rounding=ROUND_HALF_EVEN, traps=[])
    with localcontext(context):
        return cast(Decimal, handler(value))


class Quantity(Immutable):
    """Nonnegative finite decimal quantity with explicit units and meaning.

    At most twelve fractional places and 37 significant/total decimal digits are
    accepted, with value <=10^24. JSON uses canonical decimal strings, including
    zero as "0"; null/unavailable and negative quantities are invalid here.
    Input costs, budgets and outcome weights have the tighter 10^12 cap below.
    """

    value: Annotated[Decimal, Field(ge=0, le=Decimal("1e24"), max_digits=37, decimal_places=12)]
    semantics: UnitSemantics

    @field_validator("value", mode="wrap")
    @classmethod
    def _zero(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Decimal:
        """Enforce the field's declared precision without context rounding, then canonicalize zero."""
        value = validate_decimal_schema(value, handler)
        # Pydantic exposes this mapping through a descriptor that Pylint cannot infer.
        limits = (
            getattr(item, "decimal_places", None)
            for item in cls.model_fields["value"].metadata  # pylint: disable=unsubscriptable-object
        )
        places = next((limit for limit in limits if isinstance(limit, int)), 12)
        validate_decimal_places(value, places=places)
        return value if value else Decimal(0)

    @field_serializer("value")
    def _decimal(self, value: Decimal) -> str:
        """Canonicalize equivalent decimal spellings without context-dependent rounding."""
        return decimal_text(value)
