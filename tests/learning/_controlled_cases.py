"""Public constructed bounded-loss controller fixtures, without production mocks."""

from typing import Literal

from coleman.core.actions import Weight
from coleman.core.feedback import FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.learning.control_contracts import ControlledConfig, ControlledLimits
from coleman.objectives.bindings import DefinitionLineage
from coleman.objectives.controls import (
    CoefficientOrigin,
    ExpertBound,
    ExpertBounds,
    IntervalCadence,
    L1MovementPenalty,
    MixtureControls,
    MovementAllowance,
    MovementLimit,
    ProjectionOutput,
    RoundCadence,
)
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS, DeclaredTargetShare
from tests.learning.test_reference import LOG_TWO
from tests.learning.test_reference import configuration as reference_configuration


def ref(name: str) -> Reference:
    return Reference(id=name, version="1")


def configuration(
    *,
    prior: tuple[float, ...] = (0.5, 0.25, 0.25),
    eta: float = LOG_TWO,
    strength: float | None = 0.05,
    target: tuple[float, tuple[float, ...]] | None = None,
    lower: tuple[float, ...] | None = None,
    upper: tuple[float, ...] | None = None,
    movement: tuple[float, ...] | None = None,
    measure: Literal["net_displacement", "accumulated_absolute"] = "accumulated_absolute",
    scope: Literal["per_release", "per_interval"] = "per_release",
    cadence: int | Reference = 1,
    limits: ControlledLimits | None = None,
) -> ControlledConfig:
    source = reference_configuration(prior=tuple(1 / len(prior) for _ in prior))
    initial = type(source.prior).model_validate(
        source.prior.model_dump()
        | {
            "weights": tuple(
                Weight(member=item.member, value=value) for item, value in zip(source.prior.weights, prior, strict=True)
            )
        }
    )
    lineage = DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)
    origin = CoefficientOrigin(
        schema_version=1,
        reference=ref("initial-ci-coefficients"),
        lineage=lineage,
        weights=initial.weights,
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )
    target_definition = None
    if target is not None:
        alpha, weights = target
        target_definition = DeclaredTargetShare(
            schema_version=1,
            reference=ref("declared-ci-target"),
            lineage=lineage,
            weights=tuple(
                Weight(member=item.member, value=value) for item, value in zip(initial.weights, weights, strict=True)
            ),
            profile="binary64-simplex-v1",
            semantics=COEFFICIENT_SEMANTICS,
            alpha=alpha,
        )
    bounds = ExpertBounds(
        schema_version=1,
        reference=ref("ci-expert-boxes"),
        lineage=lineage,
        semantics=COEFFICIENT_SEMANTICS,
        bounds=tuple(
            ExpertBound(
                member=item.member,
                lower=0.0 if lower is None else lower[index],
                upper=1.0 if upper is None else upper[index],
            )
            for index, item in enumerate(initial.weights)
        ),
    )
    allowance = (
        None
        if movement is None
        else MovementAllowance(
            schema_version=1,
            reference=ref("ci-coefficient-movement"),
            lineage=lineage,
            semantics=COEFFICIENT_SEMANTICS,
            scope=scope,
            measure=measure,
            limits=tuple(
                MovementLimit(member=item.member, maximum=value)
                for item, value in zip(initial.weights, movement, strict=True)
            ),
        )
    )
    output = (
        ProjectionOutput()
        if strength is None
        else L1MovementPenalty(
            schema_version=1,
            reference=ref("declared-ci-l1-control"),
            lineage=lineage,
            strength=strength,
            origin=origin,
            semantics=COEFFICIENT_SEMANTICS,
        )
    )
    controls = MixtureControls(
        schema_version=1,
        reference=ref("ci-mixture-controls"),
        lineage=lineage,
        initial_prior=initial,
        target=target_definition,
        bounds=bounds,
        movement=allowance,
        cadence=RoundCadence(panels=cadence) if isinstance(cadence, int) else IntervalCadence(calendar=cadence),
        output=output,
    )
    return ControlledConfig(
        schema_version=1,
        reference=ref("constructed-controlled-ci-law"),
        lineage=lineage,
        profile="controlled-hedge-projection" if strength is None else "controlled-hedge-l1-movement",
        controls=controls,
        loss_spec=source.loss_spec,
        transform=source.transform,
        eta=eta,
        period_clock=ref("declared-movement-periods") if allowance is not None and scope == "per_interval" else None,
        limits=ControlledLimits() if limits is None else limits,
    )


def panel(config: ControlledConfig, losses: tuple[float | None, ...]) -> FullPanel:
    return FullPanel(
        rows=tuple(
            FeedbackRow(
                member=item.member,
                values=(
                    OutcomeValue(
                        spec=config.loss_spec,
                        value=value,
                        status="unavailable" if value is None else "observed",
                        origin="synthetic",
                        transform=config.transform.reference,
                    ),
                ),
            )
            for item, value in zip(config.controls.initial_prior.weights, losses, strict=True)
        )
    )


def values(learner):
    return tuple(item.value for item in learner.weights)
