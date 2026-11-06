"""Keep exact synthetic accounting separate from two explicit learner surrogates.

Run this standalone file with the v5 integration source's built wheel. All
credits, historical values and clocks are authored synthetic evidence. No market
data, adaptive learner, trading execution or performance claim is involved.
"""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from coleman.core.feedback import OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionLineage, QuantityMeaning
from coleman.objectives.transforms import AffineLoss, apply_loss, loss_outcome
from coleman.objectives.volatility import (
    FitSample,
    FitSnapshot,
    RollingScaleSpec,
    SignedVolatilitySpec,
    apply_signed_volatility,
    clipping_statistics,
    fit_rolling_scale,
    select_scale,
)


def _reference(name: str) -> Reference:
    """Use explicit versions for synthetic source, horizon and outcome identities."""
    return Reference(id=name, version="1")


def main() -> None:
    """Fit only historical values, apply declared losses and print retained facts.

    Raises
    ------
    ValueError
        An expected complete synthetic result cannot be produced.

    Notes
    -----
    Accounting uses exact Decimal values; conversion into a mathematical Scalar
    is an explicit binary64 projection. The frozen scale is selected before the
    producing decision. No durable checkpoint, clock read, file write, learner
    update or financial recommendation is made. The only side effect is stdout.
    """
    start = datetime(2026, 1, 1, tzinfo=UTC)
    horizon = _reference("one-day-resource-credit")
    lineage = DefinitionLineage(
        source=_reference("authored-resource-credit-table"),
        source_fingerprint=fingerprint({"history": [-1, 0, 1], "gross": "-2", "paid_cost": "0.5"}),
        declared_at=start,
        supersedes=None,
    )
    raw = QuantityMeaning(
        spec=ScalarSpec(
            dimension=_reference("net-credit"),
            semantics=UnitSemantics(unit="credit", meaning="synthetic-net-credit-per-day-v1"),
            role="objective",
        ),
        interpretation="raw_outcome",
        horizon=horizon,
    )
    loss = QuantityMeaning(
        spec=ScalarSpec(
            dimension=_reference("scaled-net-loss"),
            semantics=UnitSemantics(unit="1", meaning="signed-volatility-scaled-net-loss-v1"),
            role="loss",
            minimum=0.0,
            maximum=1.0,
        ),
        interpretation="normalized_loss",
        horizon=horizon,
    )
    scale_definition = RollingScaleSpec(
        schema_version=1,
        reference=_reference("three-observation-sample-sd"),
        lineage=lineage,
        quantity=raw,
        scale_spec=ScalarSpec(
            dimension=_reference("historical-net-credit-sd"),
            semantics=UnitSemantics(unit="credit", meaning="historical-net-credit-sample-sd-v1"),
            role="diagnostic",
            minimum=0.0,
        ),
        window=3,
        scope="shared",
        maximum_age_seconds=604800.0,
    )
    snapshot = FitSnapshot(
        schema_version=1,
        source=_reference("historical-credit-snapshot"),
        quantity=raw,
        member=None,
        training_cutoff=start + timedelta(days=3),
        samples=tuple(
            FitSample(
                observation=_reference("historical-observation-" + str(index)),
                value=Scalar(spec=raw.spec, value=value, status="observed"),
                timing=FeedbackTiming(
                    event_at=start + timedelta(days=index),
                    available_at=start + timedelta(days=index, hours=1),
                    mature_at=start + timedelta(days=index, hours=1),
                    received_at=start + timedelta(days=index, hours=2),
                ),
            )
            for index, value in enumerate((-1.0, 0.0, 1.0))
        ),
    )
    scale = fit_rolling_scale(scale_definition, snapshot, frozen_at=start + timedelta(days=3, hours=1))
    definition = SignedVolatilitySpec(
        schema_version=1,
        reference=_reference("signed-net-loss"),
        lineage=lineage,
        input=raw,
        output=loss,
        scale_definition=scale_definition.binding,
        c=2.0,
    )
    timing = DecisionTiming(
        decision_at=start + timedelta(days=4),
        information_cutoff=start + timedelta(days=4),
        target_start=start + timedelta(days=5),
        target_end=start + timedelta(days=6),
        eligible_input_fingerprint=snapshot.fingerprint,
        mode="future",
    )
    selection = select_scale(
        definition,
        scale,
        decision_fingerprint=fingerprint({"task": "synthetic-resource-allocation", "decision": "period-4"}),
        timing=timing,
        outcome=_reference("allocated-net-result"),
    )
    gross, paid_cost = Decimal("-2"), Decimal("0.5")
    net = gross - paid_cost
    projected_net = Scalar(spec=raw.spec, value=net, status="observed")
    scaled = apply_signed_volatility(selection, projected_net)
    affine_loss_quantity = QuantityMeaning(
        spec=ScalarSpec(
            dimension=_reference("assumed-range-net-loss"),
            semantics=UnitSemantics(unit="1", meaning="assumed-range-net-loss-v1"),
            role="loss",
            minimum=0.0,
            maximum=1.0,
        ),
        interpretation="normalized_loss",
        horizon=horizon,
    )
    affine_definition = AffineLoss(
        schema_version=1,
        reference=_reference("assumed-range-net-loss"),
        lineage=lineage,
        input=raw,
        output=affine_loss_quantity,
        lower=-5.0,
        upper=5.0,
        direction="maximize",
        outside="clip",
        bound_basis="modeling_assumption",
    )
    affine = apply_loss(affine_definition, projected_net)
    origin = loss_outcome(
        affine,
        OutcomeValue(spec=raw.spec, value=net, status="observed", origin="synthetic", transform=None),
    )
    if affine.output is None:
        raise ValueError("The assumed-range fixture must produce an explicit loss")
    applications = tuple(
        apply_signed_volatility(
            select_scale(
                definition,
                scale,
                decision_fingerprint=fingerprint({"synthetic-case": index}),
                timing=timing,
                outcome=_reference("synthetic-outcome-" + str(index)),
            ),
            Scalar(spec=raw.spec, value=value, status="observed"),
        )
        for index, value in enumerate((-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0))
    )
    missing = apply_signed_volatility(
        select_scale(
            definition,
            scale,
            decision_fingerprint=fingerprint({"synthetic-case": "unknown"}),
            timing=timing,
            outcome=_reference("synthetic-unknown-outcome"),
        ),
        Scalar(spec=raw.spec, value=None, status="unknown"),
    )
    stats = clipping_statistics(
        (*applications, applications[0], missing), split=_reference("public-development-fixture")
    )
    print(
        json.dumps(
            {
                "origin": origin.origin,
                "gross_credit": str(gross),
                "paid_cost_credit": str(paid_cost),
                "net_credit": str(net),
                "affine_loss": affine.output.value,
                "historical_scale_credit": scale.scale.value,
                "volatility_loss": scaled.output.value,
                "volatility_clipping": scaled.clipping,
                "net_losses": [application.output.value for application in applications],
                "clipping_sides": [application.clipping for application in applications],
                "counts": {
                    "computed": stats.computed,
                    "unknown": stats.unknown,
                    "lower": stats.lower,
                    "upper": stats.upper,
                },
                "clipping_denominator": stats.computed,
                "scale_fingerprint": scale.fingerprint,
            }
        )
    )


if __name__ == "__main__":
    main()
