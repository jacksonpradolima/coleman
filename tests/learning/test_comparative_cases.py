"""Independent authored CI comparisons keep loss, movement and resources apart."""

import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from coleman.learning.contracts import LearningUpdate, ReferenceConfig
from coleman.learning.control_contracts import ControlOutcome
from coleman.learning.controlled import ControlledHedge
from coleman.learning.reference import FixedShare, Hedge
from tests.learning._controlled_cases import configuration, panel, ref, values
from tests.learning.test_independent_math import _scalar_exponential
from tests.optimization._independent_oracles import effective_intervals, face_minimum, full_movement

FIXTURE = Path(__file__).parents[1] / "fixtures/learning/v1/controlled-comparisons.json"
ETA = math.log(3)


def _data():
    result = json.loads(FIXTURE.read_text())
    assert result["schema_version"] == 1
    return result


def _reference(base, *, fixed_share=False):
    return ReferenceConfig(
        schema_version=1,
        reference=ref("comparison-reference"),
        lineage=base.lineage,
        profile="fixed-share-excluding-origin" if fixed_share else "hedge-constant",
        prior=base.controls.initial_prior,
        loss_spec=base.loss_spec,
        transform=base.transform,
        eta=base.eta,
        alpha=0.1 if fixed_share else None,
    )


def _actual(name):
    settings = _data()["methods"][name]
    target = (0.25, (0.75, 0.25)) if name == "anchored" else None
    lower, upper = ((0.25, 0.25), (0.75, 0.75)) if name == "bounded" else (None, None)
    movement = (0.125, 0.125) if name == "bounded" else None
    cadence = settings["accepted_panels_per_release"] if name == "cadenced" else 1
    base = configuration(
        prior=(0.5, 0.5),
        eta=ETA,
        strength=0.125 if name == "penalized" else None,
        target=target,
        lower=lower,
        upper=upper,
        movement=movement,
        cadence=cadence,
    )
    if name == "plain":
        return Hedge(_reference(base), owner=ref("actual-plain")), base
    return ControlledHedge(base, owner=ref("actual-" + name)), base


def _oracle(name, losses):
    """Use independent Decimal exponentials and Fraction face minimization."""
    current = (0.5, 0.5)
    before, after, pending = [], [], []
    releases = 0
    for observation in losses:
        before.append(current)
        if name != "fixed":
            pending.append(observation)
            if name != "cadenced" or len(pending) == 2:
                sums = tuple(float(sum((Fraction(row[i]) for row in pending), Fraction())) for i in range(2))
                q = tuple(float(x) for x in _scalar_exponential(current, sums, ETA))
                if name == "anchored":
                    q = tuple(
                        float(Fraction(3, 4) * Fraction(x) + Fraction(1, 4) * a)
                        for x, a in zip(q, (Fraction(3, 4), Fraction(1, 4)), strict=True)
                    )
                if name == "plain":
                    current = q
                else:
                    lower, upper = (Fraction(),) * 2, (Fraction(1),) * 2
                    if name == "bounded":
                        lower, upper = effective_intervals(
                            (Fraction(1, 4),) * 2,
                            (Fraction(3, 4),) * 2,
                            tuple(Fraction(x) for x in current),
                            allowances=(Fraction(1, 8),) * 2,
                        )
                    strength = Fraction(1, 8) if name == "penalized" else Fraction()
                    found = face_minimum(
                        tuple(Fraction(x) for x in q), tuple(Fraction(x) for x in current), lower, upper, strength
                    )
                    assert found is not None
                    current = tuple(float(x) for x in found.weights)
                pending = []
                releases += 1
        after.append(current)
    return tuple(before), tuple(after), releases


def _run(name, losses):
    if name == "fixed":
        before = after = ((0.5, 0.5),) * len(losses)
        releases = 0
    else:
        learner, base = _actual(name)
        before, after = [], []
        for loss in losses:
            before.append(values(learner))
            outcome = learner.observe(panel(base, loss))
            if name == "plain":
                assert isinstance(outcome, LearningUpdate)
                assert outcome.status == "accepted"
            else:
                assert isinstance(outcome, ControlOutcome)
                assert outcome.disposition in {"held", "released"}
            after.append(values(learner))
        releases = learner.update_count if isinstance(learner, Hedge) else learner.release_count
    gross = sum(
        (
            Fraction(coefficient) * Fraction(loss)
            for weights, row in zip(before, losses, strict=True)
            for coefficient, loss in zip(weights, row, strict=True)
        ),
        Fraction(),
    )
    previous = (Fraction(1, 2), Fraction(1, 2))
    movement = Fraction()
    for emitted in after:
        vector = tuple(Fraction(x) for x in emitted)
        movement += full_movement(vector, previous)
        previous = vector
    model = _data()["external_resource_model"]
    proxy_seconds = Fraction(model["seconds_per_normalized_mixture_loss"]) * gross
    reconfiguration_seconds = Fraction(model["seconds_per_full_coefficient_L1_emission"]) * movement
    return {
        "before": tuple(before),
        "after": tuple(after),
        "releases": releases,
        "gross_loss": gross,
        "coefficient_L1": movement,
        "modeled_proxy_seconds": proxy_seconds,
        "modeled_reconfiguration_seconds": reconfiguration_seconds,
        "modeled_total_seconds": proxy_seconds + reconfiguration_seconds,
    }


@pytest.mark.parametrize("stream", ["reversal", "no_skill", "cost_dominated"])
@pytest.mark.parametrize("name", ["fixed", "plain", "anchored", "bounded", "cadenced", "penalized"])
def test_actual_six_routes_match_independent_authored_stream_recurrence(name, stream):
    data = _data()
    losses = tuple(tuple(float(Fraction(value)) for value in row) for row in data["streams"][stream])
    expected_before, expected_after, releases = _oracle(name, losses)
    actual = _run(name, losses)
    assert actual["releases"] == releases
    for reported, expected in zip(actual["before"], expected_before, strict=True):
        assert reported == pytest.approx(expected, abs=2e-14, rel=2e-13)
    for reported, expected in zip(actual["after"], expected_after, strict=True):
        assert reported == pytest.approx(expected, abs=2e-14, rel=2e-13)
    if name == "bounded":
        assert all(0.25 <= value <= 0.75 for weights in actual["after"] for value in weights)
        assert all(
            abs(new - old) <= 0.125
            for before, after in zip(actual["before"], actual["after"], strict=True)
            for old, new in zip(before, after, strict=True)
        )
    if name == "cadenced":
        assert all(actual["before"][i] == actual["after"][i] for i in range(0, len(losses), 2))
    assert data["external_resource_model"]["origin"] == "modeled_counterfactual"
    assert actual["modeled_reconfiguration_seconds"] == 100 * actual["coefficient_L1"]


def test_authored_cost_dominated_case_reverses_gross_ranking_without_an_adaptive_win_requirement():
    data = _data()
    losses = tuple(tuple(float(Fraction(x)) for x in row) for row in data["streams"]["cost_dominated"])
    plain, fixed = _run("plain", losses), _run("fixed", losses)
    hand = data["hand_substitutions"]["cost_dominated_plain"]
    assert float(plain["gross_loss"]) == pytest.approx(float(Fraction(hand["gross_normalized_loss"])), abs=2e-15)
    assert float(plain["coefficient_L1"]) == pytest.approx(float(Fraction(hand["full_coefficient_L1"])), abs=2e-15)
    assert float(plain["modeled_total_seconds"]) == pytest.approx(
        float(Fraction(hand["modeled_total_seconds"])), abs=2e-13
    )
    assert fixed["gross_loss"] == Fraction(hand["fixed_gross_normalized_loss"])
    assert fixed["modeled_total_seconds"] == Fraction(hand["fixed_total_seconds"])
    assert plain["gross_loss"] < fixed["gross_loss"]
    assert plain["modeled_total_seconds"] > fixed["modeled_total_seconds"]
    assert fixed["coefficient_L1"] == 0


def test_authored_reversal_shows_a_held_gate_can_lag_without_claiming_it_always_loses():
    data = _data()
    losses = tuple(tuple(float(Fraction(x)) for x in row) for row in data["streams"]["reversal"])
    plain, cadenced = _run("plain", losses), _run("cadenced", losses)
    hand = data["hand_substitutions"]["reversal_first_coefficients"]
    assert tuple(row[0] for row in plain["before"]) == pytest.approx(
        tuple(float(Fraction(x)) for x in hand["plain_before_panels"]), abs=2e-15
    )
    assert tuple(row[0] for row in cadenced["before"]) == pytest.approx(
        tuple(float(Fraction(x)) for x in hand["cadenced_before_panels"]), abs=2e-15
    )
    assert cadenced["before"][-1][0] > plain["before"][-1][0]


def test_no_skill_declared_target_changes_weights_without_creating_predictive_evidence():
    data = _data()
    losses = tuple(tuple(float(Fraction(x)) for x in row) for row in data["streams"]["no_skill"])
    hand = data["hand_substitutions"]["no_skill_anchored"]
    anchored, fixed = _run("anchored", losses), _run("fixed", losses)
    assert anchored["after"][-1][0] == float(Fraction(hand["last_first_coefficient"]))
    assert anchored["gross_loss"] == fixed["gross_loss"] == 2
    assert anchored["coefficient_L1"] == Fraction(hand["full_coefficient_L1"])
    assert anchored["modeled_total_seconds"] > fixed["modeled_total_seconds"]


def test_actual_declared_target_and_excluding_origin_shares_are_distinct():
    base = configuration(prior=(0.7, 0.3), eta=0.0, strength=None, target=(0.1, (0.4, 0.6)))
    anchored = ControlledHedge(base, owner=ref("target-convention"))
    excluding = FixedShare(_reference(base, fixed_share=True), owner=ref("excluding-convention"))
    assert anchored.observe(panel(base, (0.0, 0.0))).disposition == "released"
    assert excluding.observe(panel(base, (0.0, 0.0))).status == "accepted"
    assert values(anchored) == pytest.approx((0.67, 0.33), abs=3e-16)
    assert values(excluding) == pytest.approx((0.66, 0.34), abs=3e-16)
    assert values(anchored) != values(excluding)
