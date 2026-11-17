"""A real flat, zero-fee no-skill case keeps unavailable scales and fixed books."""

from datetime import timedelta
from fractions import Fraction

from coleman.core.numeric import Scalar
from coleman.core.primitives import fingerprint
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.learning.controlled import ControlledHedge
from coleman.objectives.volatility import (
    FitSample,
    FitSnapshot,
    apply_signed_volatility,
    fit_rolling_scale,
    select_scale,
)
from coleman.problems.portfolio import Fees, execute, mark
from examples.synthetic_portfolio import model


def test_authored_no_skill_zero_scale_retains_fixed_baseline_without_fake_loss():
    # This independent bounded case explicitly declares zero paid fees. All
    # marks stay100, so each actual funded recipe has exactly zero return. It
    # is a mathematical edge case, not market evidence or a free-cost default.
    free = Fees.model_validate(model.fees(0).model_dump() | {"proportional": {"numerator": 0, "denominator": 1}})
    raw_paths = []
    terminal = []
    for index, weights in enumerate(
        ((Fraction(4, 5), Fraction(1, 5)), (Fraction(1, 5), Fraction(4, 5)), (Fraction(1, 2), Fraction(1, 2)))
    ):
        receipt = execute(
            model.initial_book("zero-scale-" + str(index)),
            model.price_panel(0, (100, 100)),
            model.target("zero-scale-" + str(index), weights),
            free,
            request=model.ref("zero-open-" + str(index)),
        )
        opening = receipt.net_value.fraction
        raw = tuple(
            Scalar(
                spec=model.RAW.spec,
                status="observed",
                value=float(mark(receipt.after, model.price_panel(day, (100, 100))).value.fraction / opening - 1),
            )
            for day in (1, 2)
        )
        assert tuple(row.value for row in raw) == (0.0, 0.0)
        terminal.append(mark(receipt.after, model.price_panel(3, (100, 100))).value.fraction)
        raw_paths.append(raw)
    learner = ControlledHedge(model.controlled_config("penalized"), owner=model.ref("zero-scale-owner"))
    before = learner.export_state().to_json()
    at = model.START + timedelta(days=2)
    outputs = []
    for member, raw in zip(model.RECIPES, raw_paths, strict=True):
        fitted = fit_rolling_scale(
            model.SCALE,
            FitSnapshot(
                schema_version=1,
                source=model.ref("actual-flat-returns"),
                quantity=model.RAW,
                member=member,
                training_cutoff=at,
                samples=tuple(
                    FitSample(
                        observation=model.ref("flat-" + str(index)),
                        value=value,
                        timing=FeedbackTiming(
                            event_at=model.START + timedelta(days=index),
                            available_at=model.START + timedelta(days=index),
                            mature_at=model.START + timedelta(days=index),
                            received_at=model.START + timedelta(days=index),
                        ),
                    )
                    for index, value in enumerate(raw, start=1)
                ),
            ),
            frozen_at=at,
        )
        selection = select_scale(
            model.TRANSFORM,
            fitted,
            member=member,
            outcome=model.ref("flat-next"),
            decision_fingerprint=fingerprint({"flat": member.id}),
            timing=DecisionTiming(
                decision_at=at,
                information_cutoff=at,
                target_start=at,
                target_end=at + timedelta(days=1),
                eligible_input_fingerprint=fingerprint({"flat": 100}),
                mode="future",
            ),
        )
        application = apply_signed_volatility(selection, Scalar(spec=model.RAW.spec, status="observed", value=0.0))
        assert fitted.status == application.reason == "zero_scale"
        assert fitted.scale.status == "observed" and fitted.scale.value == 0.0
        assert application.output.status == "unavailable" and application.output.value is None
        outputs.append(application.output)
    assert not all(row.status == "observed" for row in outputs)
    # No complete observed panel exists, hence no learner update or penalty is
    # delivered. The fixed baseline ties every no-skill recipe at100credits.
    assert learner.export_state().to_json() == before
    assert terminal == [Fraction(100)] * 3
    assert learner.export_state().payload.update_count == 0
