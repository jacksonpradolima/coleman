"""Exact decimal quantities and finite weighted-coverage input contracts."""

from __future__ import annotations

from decimal import MAX_EMAX, MIN_EMIN, ROUND_HALF_EVEN, Context, Decimal, localcontext
from typing import Annotated, Any, Literal, Self, cast

from pydantic import (
    Field,
    ValidatorFunctionWrapHandler,
    field_serializer,
    field_validator,
    model_validator,
)

from coleman.core.base import Immutable as Immutable
from coleman.core.units import UnitSemantics as UnitSemantics
from coleman.results.domain import Fingerprint, Identifier
from coleman.results.versioned import VersionedContract

SCALE = 10**12
INPUT_MAXIMUM = Decimal("1000000000000")


def ticks(value: Decimal) -> int:
    """Convert a validated finite decimal to exact signed 10^-12 ticks, without rounding."""
    numerator, denominator = value.as_integer_ratio()
    scaled, remainder = divmod(numerator * SCALE, denominator)
    if remainder:
        raise ValueError("Quantity exceeds twelve decimal places")
    return scaled


def decimal_value(value: int, *, places: int = 12) -> Decimal:
    """Convert scaled integers exactly at 12/24/48 places, without Decimal-context rounding.

    Default precision preserves existing coverage artifacts. Product moments use
    24/48 places; unsupported precision raises ValueError rather than rounding.
    """
    if isinstance(places, bool) or not isinstance(places, int) or places not in {12, 24, 48}:
        raise ValueError("Supported exact decimal scales are 12, 24 and 48 places")
    whole, fraction = divmod(abs(value), 10**places)
    sign = "-" if value < 0 else ""
    return Decimal(f"{sign}{whole}.{fraction:0{places}d}")


def decimal_text(value: Decimal) -> str:
    """Return context-independent finite decimal spelling with canonical unsigned zero."""
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
    """
    if type(places) is not int or places < 0:
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


def unique_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    """Sort finite identity declarations, rejecting duplicates rather than dropping them."""
    if len(values) != len(set(values)):
        raise ValueError("Duplicate identifier")
    return tuple(sorted(values))


class CoverageCandidate(Immutable):
    """One indivisible candidate with nonnegative cost and overlapping covered outcomes."""

    candidate_id: Identifier
    cost: Quantity
    covers: Annotated[tuple[Identifier, ...], Field(max_length=10000)]

    @field_validator("covers")
    @classmethod
    def _covers(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Canonicalize memberships; duplicate outcome IDs indicate invalid input."""
        return unique_ids(values)

    @model_validator(mode="after")
    def _cost(self) -> Self:
        """Cap a single modeled cost before scaled integer arithmetic."""
        if self.cost.value > INPUT_MAXIMUM:
            raise ValueError("Candidate cost exceeds 10^12 units")
        return self


class CoverageOutcome(Immutable):
    """One outcome and its nonnegative additive weight, counted once when covered."""

    outcome_id: Identifier
    weight: Quantity

    @model_validator(mode="after")
    def _weight(self) -> Self:
        """Bound a single coverage weight; it is not a calibrated probability."""
        if self.weight.value > INPUT_MAXIMUM:
            raise ValueError("Outcome weight exceeds 10^12 units")
        return self


class CoverageConstraints(Immutable):
    """Inclusive budget, mandatory IDs and optional inclusive cardinality ceiling.

    Budget zero permits genuinely zero-cost candidates. A mandatory set exceeding
    budget/cardinality is a valid infeasible problem, not exhausted computation.
    Negative budgets are invalid inputs. Costs must share budget units/meaning.
    """

    budget: Quantity
    required_ids: Annotated[tuple[Identifier, ...], Field(max_length=10000)] = ()
    maximum_selected: Annotated[int, Field(strict=True, ge=0, le=10000)] | None = None

    @field_validator("required_ids")
    @classmethod
    def _required(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Canonicalize mandatory identities without silently repairing duplicates."""
        return unique_ids(values)

    @model_validator(mode="after")
    def _budget(self) -> Self:
        """Bound modeled budget to 10^12 declared units."""
        if self.budget.value > INPUT_MAXIMUM:
            raise ValueError("Budget exceeds 10^12 units")
        return self


class CoverageProblem(VersionedContract):
    """Complete declared universe and an explicit, optionally restricted search scope.

    Objective is the sum of weights in the union of selected coverage sets. It
    has no ranking/reward/probability interpretation. All values use exact decimal
    semantics with zero rounding tolerance. A restricted optimum makes no claim
    about excluded candidates. Completeness of an externally declared universe
    and validity/availability of modeled costs/coverage remain producer duties.
    """

    data_fingerprint: Fingerprint
    objective_semantics: UnitSemantics
    candidates: Annotated[tuple[CoverageCandidate, ...], Field(max_length=10000)]
    outcomes: Annotated[tuple[CoverageOutcome, ...], Field(max_length=10000)]
    constraints: CoverageConstraints
    search_ids: Annotated[tuple[Identifier, ...], Field(max_length=10000)] | None = None

    @field_validator("candidates", "outcomes")
    @classmethod
    def _ordered(cls, values):
        """Sort each typed universe and reject duplicate candidate/outcome identities."""
        field = "candidate_id" if values and isinstance(values[0], CoverageCandidate) else "outcome_id"
        keys = [getattr(value, field) for value in values]
        unique_ids(tuple(keys))
        return tuple(sorted(values, key=lambda value: getattr(value, field)))

    @field_validator("search_ids")
    @classmethod
    def _search(cls, values: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """Canonicalize an explicit subset; None resolves to the complete universe."""
        return None if values is None else unique_ids(values)

    @model_validator(mode="after")
    def _references(self) -> Self:
        """Validate memberships, constraint identities and common quantity semantics."""
        candidate_ids = {candidate.candidate_id for candidate in self.candidates}
        outcome_ids = {outcome.outcome_id for outcome in self.outcomes}
        if self.search_ids is None:
            object.__setattr__(self, "search_ids", tuple(sorted(candidate_ids)))
        if not set(self.search_ids or ()).issubset(candidate_ids):
            raise ValueError("Unknown search candidate")
        if not set(self.constraints.required_ids).issubset(candidate_ids):
            raise ValueError("Unknown mandatory candidate")
        for candidate in self.candidates:
            if not set(candidate.covers).issubset(outcome_ids):
                raise ValueError("Unknown covered outcome")
            if candidate.cost.semantics != self.constraints.budget.semantics:
                raise ValueError("Incompatible cost/budget units or meaning")
        if any(outcome.weight.semantics != self.objective_semantics for outcome in self.outcomes):
            raise ValueError("Incompatible objective weights")
        return self

    @property
    def search_scope(self) -> Literal["full", "restricted"]:
        """Declare full/restricted scope relative only to this complete supplied universe."""
        return "full" if len(self.search_ids or ()) == len(self.candidates) else "restricted"

    def scoped_candidates(self) -> tuple[CoverageCandidate, ...]:
        """Return candidates in stable bit/prefix order for selection and certificates."""
        ids = set(self.search_ids or ())
        return tuple(candidate for candidate in self.candidates if candidate.candidate_id in ids)
