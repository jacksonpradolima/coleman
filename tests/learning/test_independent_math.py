"""Independent scalar/transition-matrix arithmetic for fixed-catalogue learning."""

import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from coleman.core.actions import Weight
from coleman.core.feedback import FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.learning._math import excluding_origin_share, exp_update
from coleman.learning.contracts import LearningNumericalError, ReferenceConfig
from coleman.learning.reference import AnytimeHedge, FixedShare, Hedge
from coleman.objectives.preferences import InitialPrior

FIXTURES = Path(__file__).parents[1] / "fixtures" / "learning" / "v1"
DECIMAL_PRECISION = 100


def _decimal(value):
    """Interpret stored binary64 exactly before high-precision arithmetic."""
    return Decimal.from_float(value) if isinstance(value, float) else Decimal(value)


def _scalar_exponential(prior, losses, eta) -> tuple[Decimal, ...]:
    """Normalize independently evaluated 100-digit scalar exponentials.

    A common minimum loss is removed algebraically before multiplication. This
    allows equal huge finite losses to retain prior ratios rather than erasing
    their differences by subtracting two rounded huge log values.
    """
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.Emax = 999999999999999999
        context.Emin = -999999999999999999
        values = tuple(_decimal(value) for value in losses)
        baseline = min(values)
        unnormalized = tuple(
            _decimal(p) * (-_decimal(eta) * (loss - baseline)).exp() for p, loss in zip(prior, values, strict=True)
        )
        total = sum(unnormalized, Decimal())
        return tuple(value / total for value in unnormalized)


def _matrix_share(exponential, alpha) -> tuple[Decimal, ...]:
    """Multiply by an explicit transition matrix; do not use one-minus-q."""
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        size = len(exponential)
        if size == 1:
            return (Decimal(1),)
        share = _decimal(alpha)
        matrix = tuple(tuple(1 - share if i == j else share / (size - 1) for j in range(size)) for i in range(size))
        return tuple(sum((exponential[i] * matrix[i][j] for i in range(size)), Decimal()) for j in range(size))


def _anytime(prior, panels, multiplier=2) -> tuple[Decimal, ...]:
    """Use the current rate on all past losses, independent of prior rates."""
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        if len(prior) == 1:
            return (Decimal(1),)
        accumulated = tuple(sum((_decimal(panel[i]) for panel in panels), Decimal()) for i in range(len(prior)))
        eta = _decimal(multiplier) * (Decimal(len(prior)).ln() / (len(panels) + 1)).sqrt()
        return _scalar_exponential(prior, accumulated, eta)


def test_hand_zero_eta_excluding_origin_transition_trace():
    document = json.loads((FIXTURES / "reference-hand-traces.json").read_text())
    assert document["schema_version"] == 1
    case = document["fixed_share_zero_eta"]
    expected = tuple(tuple(Fraction(x) for x in row) for row in case["weights_after_each_update"])
    current = tuple(Decimal(Fraction(x).numerator) / Decimal(Fraction(x).denominator) for x in case["prior"])
    for target in expected:
        current = _matrix_share(current, 0.5)
        assert tuple(Fraction(x) for x in current) == target


def test_anytime_current_cumulative_loss_exposes_wrong_historical_rate_update():
    panels = ((0.0, 1.0), (1.0, 0.0))
    correct = _anytime((0.5, 0.5), panels)
    assert correct == (Decimal("0.5"),) * 2
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        eta1 = 2 * Decimal(2).ln().sqrt()
        eta2 = 2 * (Decimal(2).ln() / 2).sqrt()
        wrong = _scalar_exponential((0.5, 0.5), (eta2, eta1), 1)
    assert wrong[0] > Decimal("0.6")
    assert wrong[1] < Decimal("0.4")


def test_equal_histories_preserve_hedge_prior_ratios_but_not_excluding_origin_share():
    prior = (0.5, 0.25, 0.25)
    exponential = _scalar_exponential(prior, (1.0, 1.0, 1.0), 1e308)
    assert exponential == (Decimal("0.5"), Decimal("0.25"), Decimal("0.25"))
    assert exponential[0] / exponential[1] == 2
    shared = _matrix_share(exponential, 0.5)
    assert shared[0] / shared[1] != 2
    assert shared == (Decimal("0.375"), Decimal("0.3125"), Decimal("0.3125"))


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_transition_endpoints_and_single_expert_are_independent_of_loss(alpha):
    assert _matrix_share((Decimal(1),), alpha) == (Decimal(1),)
    two = (Decimal("0.75"), Decimal("0.25"))
    assert _matrix_share(two, alpha) == (two if alpha == 0 else tuple(reversed(two)))


def test_tiny_other_mass_survives_matrix_share_when_largest_binary64_q_rounds_to_one():
    q = _scalar_exponential((0.5, 0.5), (0.0, 1.0), 40.0)
    assert float(q[0]) == 1.0
    assert float(q[1]) > 0
    expected = _matrix_share(q, 1.0)
    assert expected == tuple(reversed(q))
    assert expected[0] > Decimal("4e-18")
    assert 1.0 - float(q[0]) == 0.0


def test_ideal_prior_normalization_is_not_the_tolerated_raw_binary64_sum():
    raw = (0.5, 0.25, math.nextafter(0.25, 0.0))
    total = sum((Fraction.from_float(x) for x in raw), Fraction())
    assert total == 1 - Fraction(1, 2**55)
    ideal = tuple(Fraction.from_float(x) / total for x in raw)
    assert sum(ideal) == 1
    assert ideal != tuple(Fraction.from_float(x) for x in raw)
    expected = _scalar_exponential(raw, (0.0, 0.0, 0.0), 1.0)
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        assert expected == tuple(Decimal(x.numerator) / Decimal(x.denominator) for x in ideal)


def _config(profile="hedge-constant", prior=(0.5, 0.25, 0.25), eta=0.25, alpha=None):
    """Use the authored public wire as configuration vocabulary, not a factory oracle."""
    document = json.loads((FIXTURES / "hedge-config-wire.json").read_text())
    document["profile"] = profile
    document["eta"] = None if profile == "hedge-anytime-cumulative" else eta
    document["alpha"] = alpha
    document["anytime_multiplier"] = 2.0 if profile == "hedge-anytime-cumulative" else None
    document["prior"]["weights"] = [
        {"member": {"id": f"expert-{i}", "version": "v1"}, "value": value} for i, value in enumerate(prior)
    ]
    return ReferenceConfig.model_validate(document)


def _panel(config, losses):
    return FullPanel(
        rows=tuple(
            FeedbackRow(
                member=weight.member,
                values=(
                    OutcomeValue(
                        spec=config.loss_spec,
                        value=float(loss),
                        status="observed",
                        origin="synthetic",
                        transform=config.transform.reference,
                    ),
                ),
            )
            for weight, loss in zip(config.prior.weights, losses, strict=True)
        )
    )


def test_authored_reference_configuration_wire_round_trip():
    payload = (FIXTURES / "hedge-config-wire.json").read_text().rstrip("\n")
    config = ReferenceConfig.from_json(payload)
    assert config.to_json() == payload
    assert ReferenceConfig.from_json(config.to_json()) == config
    assert config.prior_normalization == "ideal-raw-mass-emitted-fsum-v1"


@pytest.mark.parametrize("version", [None, 0, 2, "1", True])
def test_reference_wire_requires_known_strict_schema_version(version):
    document = json.loads((FIXTURES / "hedge-config-wire.json").read_text())
    if version is None:
        document.pop("schema_version")
    else:
        document["schema_version"] = version
    payload = json.dumps(document)
    with pytest.raises(ValueError, match="schema_version"):
        ReferenceConfig.from_json(payload)


def test_raw_prior_fingerprint_retains_signed_zero_parameter_identity():
    plus = _config(eta=0.0)
    minus = _config(eta=-0.0)
    assert plus.eta == minus.eta == 0
    assert plus.fingerprint != minus.fingerprint
    assert plus.prior == minus.prior


def test_reference_zero_prior_rejection_does_not_weaken_controlled_zero_contract():
    config = _config()
    prior = InitialPrior.model_validate(
        {
            **config.prior.model_dump(),
            "weights": tuple(
                Weight(member=weight.member, value=value)
                for weight, value in zip(config.prior.weights, (0.5, 0.5, 0.0), strict=True)
            ),
        }
    )
    assert prior.weights[2].value == 0.0
    document = {**config.model_dump(), "prior": prior}
    with pytest.raises(ValueError, match="strictly positive"):
        ReferenceConfig.model_validate(document)


def _weights(values):
    return tuple(
        Weight(member=Reference(id=f"expert-{i}", version="v1"), value=value) for i, value in enumerate(values)
    )


@pytest.mark.parametrize(
    ("prior", "losses", "eta"),
    [
        ((0.5, 0.25, 0.25), (0.0, 1.0, 1.0), math.log(2)),
        ((0.5, 0.25, 0.25), (1.0, 1.0, 1.0), 1e308),
        ((0.25, 0.25, 0.25, 0.25), (0.0, 0.25, 0.5, 1.0), 4.0),
        ((0.5, 0.5), (0.0, 1.0), 40.0),
        ((0.5, 0.5), (1.0, 0.0), 0.0),
        ((1.0,), (1.0,), 1e308),
    ],
)
def test_actual_exponential_primitive_matches_high_precision_scalar(prior, losses, eta):
    expected = _scalar_exponential(prior, losses, eta)
    result = exp_update(_weights(prior), losses, eta)
    assert tuple(weight.value for weight in result) == pytest.approx(
        tuple(float(x) for x in expected), rel=4e-14, abs=1e-16
    )
    if eta == math.log(2):
        assert tuple(weight.value for weight in result) == pytest.approx((2 / 3, 1 / 6, 1 / 6), abs=2e-16)


@pytest.mark.parametrize("alpha", [0.0, 0.125, 0.5, 1.0])
@pytest.mark.parametrize("prior", [(1.0,), (0.75, 0.25), (0.5, 0.25, 0.25), (0.25,) * 4])
def test_actual_excluding_origin_primitive_matches_explicit_transition_matrix(prior, alpha):
    expected = _matrix_share(tuple(_decimal(value) for value in prior), alpha)
    result = excluding_origin_share(_weights(prior), alpha)
    assert tuple(weight.value for weight in result) == pytest.approx(tuple(float(x) for x in expected), abs=2e-16)


def test_actual_share_retains_other_mass_when_largest_q_rounds_to_one():
    q = exp_update(_weights((0.5, 0.5)), (0.0, 1.0), 40.0)
    assert q[0].value == 1
    assert q[1].value > 0
    result = excluding_origin_share(q, 1.0)
    assert result[0].value == q[1].value
    assert result[1].value == 1
    assert result[0].value > 4e-18


def test_actual_exponential_equal_loss_offset_invariance_and_structural_zero():
    prior = _weights((0.0, 0.25, 0.75))
    first = exp_update(prior, (0.0, 0.25, 0.5), 2.0)
    offset = exp_update(prior, (0.5, 0.75, 1.0), 2.0)
    assert first == offset
    assert first[0].value == 0
    # The sharing reference is a distinct operation; it can restore zero mass.
    shared = excluding_origin_share(first, 0.125)
    assert shared[0].value > 0


def test_actual_positive_underflow_is_a_refusal_rather_than_new_structural_zero():
    prior = _weights((0.5, 0.5))
    losses = (0.0, 1.0)
    expected = _scalar_exponential((0.5, 0.5), losses, 1000.0)
    assert expected[1] > 0
    assert float(expected[1]) == 0
    with pytest.raises(LearningNumericalError, match="underflows"):
        exp_update(prior, losses, 1000.0)
    assert tuple(weight.value for weight in prior) == (0.5, 0.5)


@pytest.mark.parametrize(
    ("method", "profile", "alpha"),
    [
        (Hedge, "hedge-constant", None),
        (AnytimeHedge, "hedge-anytime-cumulative", None),
        (FixedShare, "fixed-share-excluding-origin", 0.125),
    ],
)
def test_actual_reference_trajectory_matches_independent_scalar_and_matrix_arithmetic(method, profile, alpha):
    config = _config(profile, eta=0.375, alpha=alpha)
    owner = Reference(id="independent-math-owner", version="v1")
    learner = method(config, owner=owner)
    resumed = method(config, owner=owner)
    raw_prior = tuple(weight.value for weight in config.prior.weights)
    initial = tuple(float(value) for value in _scalar_exponential(raw_prior, (0.0,) * 3, 0.0))
    oracle = initial
    panels = []
    for step in range(96):
        # Authored bounded non-dyadic/tiny observations stress compensated
        # cumulative loss without making seeds or rounds independent units.
        losses = tuple((0.0, 0.1, 0.3, 2**-54, 1.0)[(step + 2 * i) % 5] for i in range(3))
        panels.append(losses)
        if profile == "hedge-anytime-cumulative":
            expected = _anytime(initial, panels)
        else:
            expected = _scalar_exponential(oracle, losses, 0.375)
            if alpha is not None:
                expected = _matrix_share(expected, alpha)
        oracle = tuple(float(value) for value in expected)
        result = learner.observe(_panel(config, losses))
        assert result.status == "accepted"
        assert result.update_count == step + 1
        assert tuple(weight.value for weight in learner.weights) == pytest.approx(oracle, rel=4e-13, abs=2e-14)
        assert all(math.isfinite(weight.value) and weight.value > 0 for weight in learner.weights)
        assert abs(sum((Fraction.from_float(weight.value) for weight in learner.weights), Fraction()) - 1) <= Fraction(
            1, 2**49
        )
        if step == 16:
            snapshot = learner.export_state()
            recovered = type(snapshot).from_json(snapshot.to_json())
            resumed.restore_state(recovered)
        elif step > 16:
            restored_result = resumed.observe(_panel(config, losses))
            assert restored_result == result
    assert learner.update_count == resumed.update_count == 96
    assert resumed.export_state().to_json() == learner.export_state().to_json()
    assert any(value != 0 for value in learner.export_state().payload.compensations)


def test_actual_anytime_reversal_returns_to_prior_and_rejects_historical_rate_result():
    config = _config("hedge-anytime-cumulative", prior=(0.5, 0.5))
    learner = AnytimeHedge(config, owner=Reference(id="anytime-reversal", version="v1"))
    assert learner.observe(_panel(config, (0.0, 1.0))).status == "accepted"
    assert learner.weights[0].value > 0.5
    assert learner.observe(_panel(config, (1.0, 0.0))).status == "accepted"
    assert tuple(weight.value for weight in learner.weights) == (0.5, 0.5)
    assert learner.export_state().payload.cumulative_losses == (1.0, 1.0)


def test_actual_fixed_share_zero_eta_reproduces_authored_exact_trace():
    config = _config("fixed-share-excluding-origin", eta=0.0, alpha=0.5)
    learner = FixedShare(config, owner=Reference(id="zero-eta-share", version="v1"))
    document = json.loads((FIXTURES / "reference-hand-traces.json").read_text())
    for row in document["fixed_share_zero_eta"]["weights_after_each_update"]:
        expected = tuple(Fraction(value) for value in row)
        assert learner.observe(_panel(config, (0.0, 0.5, 1.0))).status == "accepted"
        assert tuple(Fraction.from_float(weight.value) for weight in learner.weights) == expected


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_actual_fixed_share_endpoints_match_matrix_over_repeated_changes(alpha):
    config = _config("fixed-share-excluding-origin", prior=(0.75, 0.25), eta=0.5, alpha=alpha)
    learner = FixedShare(config, owner=Reference(id="endpoint-share", version="v1"))
    oracle = (0.75, 0.25)
    for losses in ((0.0, 1.0), (1.0, 0.0), (0.25, 0.75), (0.5, 0.5)):
        oracle = tuple(float(value) for value in _matrix_share(_scalar_exponential(oracle, losses, 0.5), alpha))
        assert learner.observe(_panel(config, losses)).status == "accepted"
        assert tuple(weight.value for weight in learner.weights) == pytest.approx(oracle, abs=5e-16)


def test_actual_zero_share_reference_reduces_exactly_to_constant_hedge():
    hedge_config = _config(eta=0.375)
    share_config = _config("fixed-share-excluding-origin", eta=0.375, alpha=0.0)
    owner = Reference(id="alpha-zero-parity", version="v1")
    hedge, share = Hedge(hedge_config, owner=owner), FixedShare(share_config, owner=owner)
    for losses in ((0.0, 0.1, 1.0), (1.0, 0.3, 0.0), (0.25, 0.25, 0.25)):
        assert hedge.observe(_panel(hedge_config, losses)).status == "accepted"
        assert share.observe(_panel(share_config, losses)).status == "accepted"
        assert hedge.weights == share.weights
        assert hedge.update_count == share.update_count
    assert hedge.binding != share.binding


@pytest.mark.parametrize(
    ("method", "profile", "alpha"),
    [
        (Hedge, "hedge-constant", None),
        (AnytimeHedge, "hedge-anytime-cumulative", None),
        (FixedShare, "fixed-share-excluding-origin", 1.0),
    ],
)
def test_actual_single_expert_is_absorbing_for_all_reference_methods(method, profile, alpha):
    config = _config(profile, prior=(1.0,), eta=1e308, alpha=alpha)
    learner = method(config, owner=Reference(id="singleton", version="v1"))
    for loss in (0.0, 1.0, 0.3):
        result = learner.observe(_panel(config, (loss,)))
        assert result.status == "accepted"
        assert result.weights[0].value == 1.0
    assert learner.update_count == 3


@pytest.mark.parametrize(
    "method,profile,alpha", [(Hedge, "hedge-constant", None), (FixedShare, "fixed-share-excluding-origin", 1.0)]
)
def test_actual_reference_underflow_preserves_all_sufficient_state(method, profile, alpha):
    config = _config(profile, prior=(0.5, 0.5), eta=1000.0, alpha=alpha)
    learner = method(config, owner=Reference(id="underflow-owner", version="v1"))
    before = learner.export_state().to_json()
    result = learner.observe(_panel(config, (0.0, 1.0)))
    assert result.status == "numerical_failure"
    assert result.failure is not None
    assert result.failure.code == "positive_underflow"
    assert learner.update_count == 0
    assert tuple(weight.value for weight in learner.weights) == (0.5, 0.5)
    assert learner.export_state().to_json() == before


def test_actual_reference_zero_and_unavailable_loss_have_distinct_dispositions():
    config = _config(prior=(0.5, 0.5))
    learner = Hedge(config, owner=Reference(id="missingness", version="v1"))
    zero = _panel(config, (0.0, 0.0))
    assert learner.observe(zero).status == "accepted"
    before = learner.export_state().to_json()
    unavailable = OutcomeValue(
        spec=config.loss_spec,
        value=None,
        status="unavailable",
        origin="synthetic",
        transform=config.transform.reference,
    )
    panel = FullPanel(rows=(FeedbackRow(member=learner.members[0], values=(unavailable,)), zero.rows[1]))
    with pytest.raises(ValueError, match="observed loss; null is not zero"):
        learner.observe(panel)
    assert learner.export_state().to_json() == before
    assert learner.update_count == 1
