"""Exact decimal quantities and finite weighted-coverage input contracts."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from coleman.core.base import Immutable as Immutable
from coleman.core.quantities import (
    SCALE as SCALE,
)
from coleman.core.quantities import (
    Quantity as Quantity,
)
from coleman.core.quantities import (
    decimal_text as decimal_text,
)
from coleman.core.quantities import (
    decimal_value as decimal_value,
)
from coleman.core.quantities import (
    ticks as ticks,
)
from coleman.core.quantities import (
    validate_decimal_places as validate_decimal_places,
)
from coleman.core.quantities import (
    validate_decimal_schema as validate_decimal_schema,
)
from coleman.core.units import UnitSemantics as UnitSemantics
from coleman.results.domain import Fingerprint, Identifier
from coleman.results.versioned import VersionedContract

INPUT_MAXIMUM = Decimal("1000000000000")


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
