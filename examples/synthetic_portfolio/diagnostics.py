"""Exact controller-objective diagnostics, separate from any executed paid fee."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import model_validator

from coleman.learning.control_contracts import ControlOutcome
from coleman.objectives.controls import L1MovementPenalty
from coleman.problems.portfolio import Exact
from coleman.problems.portfolio.contracts import PortfolioContract


class ControlObjective(PortfolioContract):
    """Separate emitted-point mathematical objective terms from any paid charge.

    Notes
    -----
    Proximity and objective/penalty terms have squared coefficient-fraction units.
    L1 movement and strength each have coefficient-fraction units. A projection
    has no penalty term; lambda0 under the L1 profile has an actual zero term.
    Held/empty receipts performed no solve. Failed attempts have unavailable
    objective values, rather than accepting the old vector as a solved output.
    """

    status: Literal["evaluated", "not_applicable", "unavailable"]
    proximity: Exact | None
    coefficient_l1: Exact | None
    strength: Exact | None
    weighted_penalty: Exact | None
    total: Exact | None
    objective_unit: Literal["coefficient-fraction-squared"] = "coefficient-fraction-squared"
    movement_unit: Literal["coefficient-fraction"] = "coefficient-fraction"

    @model_validator(mode="after")
    def _terms(self) -> Self:
        """Keep unavailable terms null and evaluated objective arithmetic exact."""
        terms = (self.proximity, self.coefficient_l1, self.strength, self.weighted_penalty, self.total)
        if self.status != "evaluated":
            if any(term is not None for term in terms):
                raise ValueError("Unevaluated control objective cannot invent numerical terms")
            return self
        self._evaluated(terms)
        return self

    def _evaluated(self, terms: tuple[Exact | None, ...]) -> None:
        """Validate the complete observed nonnegative base objective quantities."""
        if self.proximity is None or self.coefficient_l1 is None or self.total is None:
            raise ValueError("Evaluated control objective requires complete exact base terms")
        if any(term is not None and term.numerator < 0 for term in terms):
            raise ValueError("Control proximity, movement, strength and penalty cannot be negative")
        if (self.strength is None) != (self.weighted_penalty is None):
            raise ValueError("Control penalty applicability must agree with its declared strength")
        self._penalty()

    def _penalty(self) -> None:
        """Reconcile penalty multiplication and full objective using exact ratios."""
        assert self.coefficient_l1 is not None and self.proximity is not None and self.total is not None
        if (
            self.strength is not None
            and self.weighted_penalty is not None
            and self.weighted_penalty.fraction != self.strength.fraction * self.coefficient_l1.fraction
        ):
            raise ValueError("Control weighted penalty differs from declared strength times full L1")
        penalty = Fraction() if self.weighted_penalty is None else self.weighted_penalty.fraction
        if self.total.fraction != self.proximity.fraction + penalty:
            raise ValueError("Control total objective differs from its exact separate terms")


def _control_objective(control: ControlOutcome | None) -> ControlObjective | None:
    """Project exact emitted-point diagnostics from one actual control receipt."""
    if control is None:
        return None
    if control.disposition != "released":
        return ControlObjective(
            schema_version=1,
            status="not_applicable" if control.disposition in {"held", "empty"} else "unavailable",
            proximity=None,
            coefficient_l1=None,
            strength=None,
            weighted_penalty=None,
            total=None,
        )
    return _released_objective(control)


def _released_objective(control: ControlOutcome) -> ControlObjective:
    """Compute a released point's separate exact distance and movement terms."""
    assert control.proposal is not None
    current = tuple(Fraction(item.value) for item in control.weights)
    proposal = tuple(Fraction(item.value) for item in control.proposal)
    old = tuple(Fraction(item.value) for item in control.origin.weights)
    proximity = (
        sum(((value - proposed) ** 2 for value, proposed in zip(current, proposal, strict=True)), Fraction()) / 2
    )
    movement = sum((abs(value - previous) for value, previous in zip(current, old, strict=True)), Fraction())
    strength = Fraction(control.output.strength) if isinstance(control.output, L1MovementPenalty) else None
    penalty = None if strength is None else strength * movement
    return ControlObjective(
        schema_version=1,
        status="evaluated",
        proximity=Exact.of(proximity),
        coefficient_l1=Exact.of(movement),
        strength=None if strength is None else Exact.of(strength),
        weighted_penalty=None if penalty is None else Exact.of(penalty),
        total=Exact.of(proximity + (Fraction() if penalty is None else penalty)),
    )
