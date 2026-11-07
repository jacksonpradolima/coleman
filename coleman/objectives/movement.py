"""Pure exact emitted-coordinate arithmetic below the control declarations.

No control declaration is imported. The public accounting DTOs, exact joins
and transitions live in controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import BeforeValidator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference
from coleman.core.numeric import SIMPLEX_TOLERANCE

_ATOM_EXPONENT = 1074
_ATOM_DENOMINATOR = 1 << _ATOM_EXPONENT
_MAX_ATOM_DIGITS = 632


def _same_members(expected: tuple[Reference, ...], actual: tuple[Reference, ...]) -> None:
    """Require complete canonical expert identity and semantic-version equality."""
    if expected != actual:
        raise ValueError("Controls require the same complete expert IDs and versions")


def _intersection_reason(
    intervals: tuple[tuple[Fraction, Fraction], ...], limits: tuple[tuple[Fraction, Fraction], ...]
) -> Literal["nonempty", "empty_coordinate", "simplex_sum", "period_budget"]:
    """Classify the computed stored-rational box/simplex intersection."""
    if any(lower > upper for lower, upper in limits):
        return "period_budget"
    if any(lower > upper for lower, upper in intervals):
        return "empty_coordinate"
    lower_sum, upper_sum = _endpoint_sums(intervals)
    if lower_sum > 1 + SIMPLEX_TOLERANCE or upper_sum < 1 - SIMPLEX_TOLERANCE:
        return "simplex_sum"
    return "nonempty"


def _endpoint_sums(intervals: tuple[tuple[Fraction, Fraction], ...]) -> tuple[Fraction, Fraction]:
    """Sum original rational effective endpoints, independent of input order."""
    return (
        sum((lower for lower, _ in intervals), Fraction()),
        sum((upper for _, upper in intervals), Fraction()),
    )


def _exact_intersection(intervals: tuple[tuple[Fraction, Fraction], ...]) -> bool:
    """Distinguish original exact-real feasibility from numerical admission."""
    lower_sum, upper_sum = _endpoint_sums(intervals)
    return all(lower <= upper for lower, upper in intervals) and lower_sum <= 1 <= upper_sum


def _canonical_atoms(value: object) -> str:
    """Admit bounded canonical ASCII counts before integer conversion."""
    # pylint: disable-next=unidiomatic-typecheck
    if type(value) is not str:
        raise ValueError("Exact movement atoms require a built-in decimal string")
    if not 1 <= len(value) <= _MAX_ATOM_DIGITS or not value.isascii() or not value.isdecimal():
        raise ValueError("Exact movement atoms require one through 632 ASCII digits")
    if len(value) > 1 and value[0] == "0":
        raise ValueError("Exact movement atoms forbid leading zeroes")
    return value


def _binary64_atoms(value: float) -> int:
    """Convert a validated nonnegative binary64 value to its exact grid count."""
    numerator, denominator = value.as_integer_ratio()
    return numerator << (_ATOM_EXPONENT - (denominator.bit_length() - 1))


def _exact_path_remaining(
    current: tuple[float, ...], start: tuple[float, ...], used: tuple[str, ...], allowances: tuple[float, ...]
) -> tuple[Fraction, ...]:
    """Derive exact remaining amounts after the caller's complete identity joins."""
    remaining = []
    for value, initial, count, allowance in zip(current, start, used, allowances, strict=True):
        atoms = int(count)
        if atoms < abs(_binary64_atoms(value) - _binary64_atoms(initial)):
            raise ValueError("Exact accumulated usage is below known net displacement")
        remaining.append(Fraction(_binary64_atoms(allowance) - atoms, _ATOM_DENOMINATOR))
    return tuple(remaining)


@dataclass(frozen=True, slots=True)
class ExactExpertInterval:
    """Expose one ephemeral original rational coefficient interval.

    Attributes
    ----------
    member : Reference
        Exact canonical coefficient identity and version.
    lower, upper : Fraction
        Original effective endpoints, without binary64 display projection.
    """

    member: Reference
    lower: Fraction
    upper: Fraction


@dataclass(frozen=True, slots=True)
class ExactMixtureIntervals[Evidence: CoreContract]:
    """Retain exact solver geometry separately from legacy numerical admission.

    Attributes
    ----------
    profile, inputs_fingerprint : str
        Exact-interval convention and complete original-input content identity.
    intervals : tuple of ExactExpertInterval
        Canonical effective intervals in original rational arithmetic.
    status : {'feasible', 'infeasible'}
        Exact sum-one and coordinate feasibility, with zero relaxation.
    reason : str
        Nonempty, period budget, empty coordinate or exact simplex-sum reason.
    legacy_intersection : MixtureIntersection or None
        Unchanged v1 admission evidence when legacy usage is selected; null for
        exact-path accounting, which has no equivalent v1 usage certificate.

    Notes
    -----
    Fractions are ephemeral in-process solver inputs, not a new JSON wire.
    Original definitions retain signed-zero identity in the fingerprint.
    """

    profile: str
    inputs_fingerprint: str
    intervals: tuple[ExactExpertInterval, ...]
    status: Literal["feasible", "infeasible"]
    reason: Literal["nonempty", "empty_coordinate", "simplex_sum", "period_budget"]
    legacy_intersection: Evidence | None


class ExactCoefficientVariation(CoreImmutable):
    """Store one exact accumulated emitted-coordinate movement count.

    Attributes
    ----------
    member : Reference
        Exact coefficient identity and version.
    atoms_decimal : str
        Canonical nonnegative ASCII integer, at most 632 digits. One atom is
        exactly 2**-1074 coefficient fractions. Signs and leading zeros reject.

    Notes
    -----
    This is variation of committed binary64 coefficients, not a fee, a half-L1
    convention, or an authenticated history. No float rounding is performed.
    """

    member: Reference
    atoms_decimal: Annotated[str, BeforeValidator(_canonical_atoms)]
