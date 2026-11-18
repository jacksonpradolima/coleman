"""Complete causal producers, loss applications and actual controller calls."""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated, Self

from pydantic import Field, model_validator

from coleman.core.numeric import Scalar
from coleman.learning.control_contracts import ControlOutcome
from coleman.objectives.volatility import ScaleSelection, VolatilityApplication
from coleman.prediction import RidgePrediction
from coleman.problems.portfolio import Exact, PricePanel
from coleman.problems.portfolio.contracts import PortfolioContract

from . import model


class Pending(PortfolioContract):
    """Freeze original pre-label forecast and per-recipe loss-scale selections."""

    index: Annotated[int, Field(strict=True, ge=0, le=127)]
    prediction: RidgePrediction
    selections: Annotated[tuple[ScaleSelection, ...], Field(min_length=3, max_length=3)]
    recipe_opening_values: Annotated[tuple[Exact, ...], Field(min_length=3, max_length=3)]

    @model_validator(mode="after")
    def _original(self) -> Self:
        """Retain every indexed original scale and a defined funded denominator."""
        if self.prediction.sample_count != self.index:
            raise ValueError("Pending forecast must retain its original pre-label counter")
        if tuple(selection.member for selection in self.selections) != model.RECIPES:
            raise ValueError("Pending scales must retain every original recipe/version")
        if any(value.numerator <= 0 for value in self.recipe_opening_values):
            raise ValueError("Signed wealth return requires a positive original funded denominator")
        return self


class ControlReceipt(PortfolioContract):
    """Retain one actual controller call, rather than a stale last outcome."""

    name: str
    outcome: ControlOutcome


class Observation(PortfolioContract):
    """Retain original signed recipe returns, frozen scales and eligible labels."""

    index: Annotated[int, Field(strict=True, ge=1, le=15)]
    prices: PricePanel
    raw: Annotated[tuple[Scalar, ...], Field(min_length=3, max_length=3)]
    applications: Annotated[tuple[VolatilityApplication, ...], Field(min_length=3, max_length=3)]
    ridge_label: Scalar
    controls: Annotated[tuple[ControlReceipt, ...], Field(max_length=10)]

    @model_validator(mode="after")
    def _joins(self) -> Self:
        """Bind complete original raw/schema/member facts and actual call clocks."""
        if self.prices.at != model.START + timedelta(days=self.index):
            raise ValueError("Observed source prices belong to another original period")
        if self.ridge_label.spec != model.ridge_config().target or self.ridge_label.status != "observed":
            raise ValueError("Original ridge label requires its exact signed observed schema")
        for member, (raw, application) in enumerate(zip(self.raw, self.applications, strict=True)):
            if raw.spec != model.RAW.spec or raw.status != "observed" or application.raw != raw:
                raise ValueError("Original recipe raw return and loss application disagree")
            if (
                application.selection.member != model.RECIPES[member]
                or application.selection.definition != model.TRANSFORM
            ):
                raise ValueError("Original loss application belongs to another recipe or transform")
            if application.selection.outcome != model.ref("outcome-" + str(self.index) + "-" + str(member)):
                raise ValueError("Original recipe observation belongs to another producer outcome")
        expected = model.CONTROLLED_MODES if all(row.output.status == "observed" for row in self.applications) else ()
        if self.index % 2 == 0:
            expected += ("anchored-bounded-interval-l1", "composed-online")
        if tuple(row.name for row in self.controls) != expected:
            raise ValueError("Observation must retain only its actual eligible-panel/interval controller calls")
        return self


def _original_observation(
    observation: Observation,
    producer: Pending,
    marked_values: tuple[Exact | None, ...],
    previous_prices: PricePanel,
) -> None:
    """Join original selected scales and raw labels to retained actual books."""
    for application, selected, opening, marked, raw in zip(
        observation.applications,
        producer.selections,
        producer.recipe_opening_values,
        marked_values,
        observation.raw,
        strict=True,
    ):
        if application.selection.model_dump_json() != selected.model_dump_json():
            raise ValueError("Applied loss scale differs from its retained original producing selection")
        if marked is None or raw.value != float(marked.fraction / opening.fraction - 1):
            raise ValueError("Raw recipe return differs from its actual original funded path")
    ratios = tuple(
        current.value.fraction / previous.value.fraction
        for previous, current in zip(previous_prices.quotes, observation.prices.quotes, strict=True)
        if current.value is not None and previous.value is not None
    )
    if len(ratios) != 2 or observation.ridge_label.value != float(ratios[0] - ratios[1]):
        raise ValueError("Ridge label differs from its original observed exogenous price change")
