"""Actual pure reference recurrences, complete state and mathematical endpoints."""

import math
from fractions import Fraction

import pytest

from coleman.core.actions import Weight
from coleman.core.feedback import FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.units import UnitSemantics
from coleman.learning._math import excluding_origin_share, exp_update
from coleman.learning.contracts import ReferenceConfig, ReferencePayload
from coleman.learning.reference import AnytimeHedge, FixedShare, Hedge
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS, InitialPrior
from coleman.state.contracts import AlgorithmState

LOG_TWO = math.log(2)


def ref(name, version="1"):
    return Reference(id=name, version=version)


def configuration(profile="hedge-constant", *, prior=(0.5, 0.25, 0.25), eta=LOG_TWO, alpha=0.1, **changes):
    lineage = DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)
    initial = InitialPrior(
        schema_version=1,
        reference=ref("public-positive-prior"),
        lineage=lineage,
        weights=tuple(Weight(member=ref(chr(65 + index)), value=value) for index, value in enumerate(prior)),
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )
    values = dict(
        schema_version=1,
        reference=ref("public-" + profile),
        lineage=lineage,
        profile=profile,
        prior=initial,
        loss_spec=ScalarSpec(
            dimension=ref("bounded-loss"),
            semantics=UnitSemantics(unit="1", meaning="public-normalized-loss"),
            role="loss",
            minimum=0,
            maximum=1,
        ),
        transform=DefinitionBinding(reference=ref("identity-loss"), definition_fingerprint=fingerprint("identity-v1")),
        eta=None if profile == "hedge-anytime-cumulative" else eta,
        alpha=alpha if profile == "fixed-share-excluding-origin" else None,
    )
    return ReferenceConfig.model_validate(values | changes)


def panel(config, losses):
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
            for item, value in zip(config.prior.weights, losses, strict=True)
        )
    )


def values(learner):
    return tuple(item.value for item in learner.weights)


def test_constant_hedge_has_independent_hand_substitution_and_no_hidden_decision_clock():
    config = configuration()
    learner = Hedge(config, owner=ref("one-owner"))
    assert values(learner) == (0.5, 0.25, 0.25)
    first = learner.observe(panel(config, (0.0, 1.0, 1.0)))
    assert first.status == "accepted"
    assert values(learner) == pytest.approx((2 / 3, 1 / 6, 1 / 6), abs=2e-16)
    assert learner.update_count == 1
    assert learner.members == tuple(item.member for item in config.prior.weights)
    assert learner.binding.implementation.reference == ref("coleman-hedge-constant")
    assert learner.loss_spec == config.loss_spec
    assert learner.transform == config.transform
    before = learner.export_state().to_json()
    assert learner.weights == first.weights
    assert learner.export_state().to_json() == before
    assert learner.update_count == 1


def test_anytime_recomputes_current_rate_on_cumulative_loss_instead_of_historical_products():
    config = configuration("hedge-anytime-cumulative", prior=(0.75, 0.25))
    learner = AnytimeHedge(config, owner=ref("anytime-owner"))
    assert config.anytime_multiplier == 2
    rate2 = 2 * math.sqrt(math.log(2) / 2)
    numerator = 0.25 * math.exp(-rate2)
    first = learner.observe(panel(config, (0.0, 1.0)))
    assert first.status == "accepted"
    assert values(learner) == pytest.approx((0.75 / (0.75 + numerator), numerator / (0.75 + numerator)), abs=2e-16)
    second = learner.observe(panel(config, (1.0, 0.0)))
    assert second.status == "accepted"
    assert values(learner) == (0.75, 0.25)
    wrong_ratio = 3 * math.exp(rate2 - 2 * math.sqrt(math.log(2) / 3))
    assert abs(values(learner)[0] / values(learner)[1] - wrong_ratio) > 0.1
    assert learner.export_state().payload.cumulative_losses == (1.0, 1.0)


@pytest.mark.parametrize("alpha", [0.0, 0.1, 1.0])
def test_fixed_share_excluding_origin_follows_independent_transition_matrix(alpha):
    config = configuration("fixed-share-excluding-origin", eta=0.0, alpha=alpha)
    learner = FixedShare(config, owner=ref("fixed-owner"))
    initial = values(learner)
    expected = tuple(math.fsum(initial[j] * (1 - alpha if i == j else alpha / 2) for j in range(3)) for i in range(3))
    result = learner.observe(panel(config, (0.0, 0.0, 0.0)))
    assert result.status == "accepted"
    assert values(learner) == pytest.approx(expected, abs=2e-16)
    assert learner.binding.implementation.reference == ref("coleman-fixed-share-excluding-origin")


def test_alpha_zero_recovers_actual_constant_hedge_and_alpha_one_swaps_two_experts():
    hedge = Hedge(configuration(), owner=ref("hedge"))
    shared = FixedShare(configuration("fixed-share-excluding-origin", alpha=0.0), owner=ref("share"))
    for loss in ((0.0, 1.0, 0.5), (1.0, 0.0, 0.25), (0.25, 0.25, 0.25)):
        assert hedge.observe(panel(hedge.config, loss)).status == "accepted"
        assert shared.observe(panel(shared.config, loss)).status == "accepted"
        assert values(shared) == values(hedge)
    config = configuration("fixed-share-excluding-origin", prior=(0.75, 0.25), eta=0.0, alpha=1.0)
    swapped = FixedShare(config, owner=ref("swap"))
    assert swapped.observe(panel(config, (0.2, 0.7))).status == "accepted"
    assert values(swapped) == (0.25, 0.75)


@pytest.mark.parametrize(
    "profile,constructor",
    [
        ("hedge-constant", Hedge),
        ("hedge-anytime-cumulative", AnytimeHedge),
        ("fixed-share-excluding-origin", FixedShare),
    ],
)
def test_one_expert_is_exactly_absorbing_and_does_not_divide_by_n_minus_one(profile, constructor):
    config = configuration(profile, prior=(1.0,), alpha=1.0, eta=1e308)
    learner = constructor(config, owner=ref("singleton"))
    for loss in (0.0, 1.0, 0.25):
        assert learner.observe(panel(config, (loss,))).status == "accepted"
        assert values(learner) == (1.0,)
    assert learner.update_count == 3


def test_equal_huge_losses_preserve_nonuniform_prior_without_eta_loss_cancellation():
    config = configuration(prior=(0.75, 0.25), eta=1e308)
    learner = Hedge(config, owner=ref("huge"))
    assert learner.observe(panel(config, (1.0, 1.0))).status == "accepted"
    assert values(learner) == (0.75, 0.25)
    actual = exp_update(learner.weights, (1e308, 1e308), 1e308)
    assert tuple(item.value for item in actual) == (0.75, 0.25)


def test_exp_update_accepts_controlled_sum_loss_and_real_structural_zeros():
    weights = (
        Weight(member=ref("A"), value=0),
        Weight(member=ref("B"), value=0.25),
        Weight(member=ref("C"), value=0.75),
    )
    actual = exp_update(weights, (2.0, 4.0, 4.0), 1e308)
    assert tuple(item.value for item in actual) == (0.0, 0.25, 0.75)
    changed = exp_update(weights, (99.0, 2.0, 3.0), math.log(2))
    assert tuple(item.value for item in changed) == pytest.approx((0.0, 0.4, 0.6), abs=2e-16)


def test_excluding_origin_complements_keep_tiny_mass_beside_rounded_one():
    weights = (
        Weight(member=ref("A"), value=1),
        Weight(member=ref("B"), value=1e-300),
        Weight(member=ref("C"), value=2e-300),
    )
    actual = excluding_origin_share(weights, 1.0)
    assert actual[0].value == pytest.approx(1.5e-300, rel=1e-15, abs=0)
    assert actual[0].value > 0
    assert actual[1].value == 0.5
    assert actual[2].value == 0.5


def test_positive_underflow_is_typed_and_preserves_the_entire_reference_state():
    config = configuration(prior=(0.5, 0.5), eta=1000.0)
    learner = Hedge(config, owner=ref("underflow"))
    before = learner.export_state().to_json()
    update = learner.observe(panel(config, (1.0, 0.0)))
    assert update.status == "numerical_failure"
    assert update.failure is not None
    assert update.failure.code == "positive_underflow"
    assert update.update_count == 0
    assert learner.export_state().to_json() == before
    assert learner.observe(panel(config, (0.0, 0.0))).status == "accepted"


@pytest.mark.parametrize(
    "profile,constructor",
    [
        ("hedge-constant", Hedge),
        ("hedge-anytime-cumulative", AnytimeHedge),
        ("fixed-share-excluding-origin", FixedShare),
    ],
)
def test_complete_state_roundtrip_continues_identically_and_counts_raw_repeated_panels(profile, constructor):
    config = configuration(profile)
    first = constructor(config, owner=ref("retained"))
    for loss in ((0.0, 1.0, 0.5), (0.25, 0.0, 0.75)):
        assert first.observe(panel(config, loss)).status == "accepted"
    state = first.export_state()
    decoded = AlgorithmState[ReferencePayload].from_json(state.to_json())
    second = constructor(config, owner=ref("retained"))
    second.restore_state(decoded)
    for loss in ((0.5, 0.25, 0.0), (0.5, 0.25, 0.0)):
        offered = panel(config, loss)
        assert first.observe(offered).status == "accepted"
        assert second.update(offered).status == "accepted"
        assert first.export_state().to_json() == second.export_state().to_json()
    assert second.update_count == 4
    assert set(vars(second)) == {"_state"}


def test_anytime_compensations_survive_restore_and_match_independent_exact_loss_sums():
    config = configuration("hedge-anytime-cumulative", prior=(0.5, 0.5))
    learner = AnytimeHedge(config, owner=ref("compensated"))
    exact = [Fraction(), Fraction()]
    for index in range(80):
        loss = (1.0, 2**-54) if index % 2 == 0 else (2**-54, 1.0)
        exact = [total + Fraction(value) for total, value in zip(exact, loss, strict=True)]
        assert learner.observe(panel(config, loss)).status == "accepted"
    state = learner.export_state()
    assert any(state.payload.compensations)
    for total, correction, expected in zip(
        state.payload.cumulative_losses, state.payload.compensations, exact, strict=True
    ):
        assert math.fsum((total, correction)) == float(expected)
    restored = AnytimeHedge(config, owner=ref("compensated"))
    restored.restore_state(AlgorithmState[ReferencePayload].from_json(state.to_json()))
    assert restored.observe(panel(config, (0.25, 0.5))).status == "accepted"
    assert learner.observe(panel(config, (0.25, 0.5))).status == "accepted"
    assert restored.export_state().to_json() == learner.export_state().to_json()


def test_prior_law_retains_raw_residual_and_emitted_numerical_profile_separately():
    config = configuration(prior=(1 / 3, 1 / 3, 1 / 3))
    raw_mass = sum((Fraction(item.value) for item in config.prior.weights), Fraction())
    assert raw_mass != 1
    assert config.prior_normalization == "ideal-raw-mass-emitted-fsum-v1"
    learner = Hedge(config, owner=ref("thirds"))
    assert learner.config.prior.weights == config.prior.weights
    assert values(learner) == (1 / 3, 1 / 3, 1 / 3)
    assert learner.observe(panel(config, (0.5, 0.5, 0.5))).status == "accepted"
    assert values(learner) == (1 / 3, 1 / 3, 1 / 3)


@pytest.mark.parametrize(
    "profile,constructor",
    [
        ("hedge-constant", Hedge),
        ("hedge-anytime-cumulative", AnytimeHedge),
        ("fixed-share-excluding-origin", FixedShare),
    ],
)
def test_common_bounded_loss_offsets_preserve_coefficients_under_the_declared_reference_law(profile, constructor):
    config = configuration(profile)
    original = constructor(config, owner=ref("original"))
    shifted = constructor(config, owner=ref("shifted"))
    for loss in ((0.0, 0.25, 0.5), (0.5, 0.0, 0.25)):
        offset = tuple(value + 0.25 for value in loss)
        assert original.observe(panel(config, loss)).status == "accepted"
        assert shifted.observe(panel(config, offset)).status == "accepted"
        assert values(original) == values(shifted)


def test_expert_permutation_preserves_member_bound_exponential_and_share_coefficients():
    weights = configuration().prior.weights
    losses = (0.0, 0.25, 1.0)
    original = excluding_origin_share(exp_update(weights, losses, 2.0), 0.2)
    order = (2, 0, 1)
    permuted_weights = tuple(weights[index] for index in order)
    permuted_losses = tuple(losses[index] for index in order)
    permuted = excluding_origin_share(exp_update(permuted_weights, permuted_losses, 2.0), 0.2)
    actual = {item.member: item.value for item in permuted}
    for item in original:
        assert actual[item.member] == pytest.approx(item.value, abs=4e-16)


def test_changing_and_resetting_one_actual_owner_preserves_another_complete_state():
    config = configuration()
    first = Hedge(config, owner=ref("first"))
    second = Hedge(config, owner=ref("second"))
    initial = first.export_state()
    unchanged = second.export_state().to_json()
    assert first.observe(panel(config, (0.0, 1.0, 1.0))).status == "accepted"
    assert second.export_state().to_json() == unchanged
    first.restore_state(initial)
    assert second.export_state().to_json() == unchanged
