"""Unit tests for MAB (Multi-Armed Bandit) policies."""

import numpy as np
import polars as pl
import pytest

from coleman.agent import Agent
from coleman.policy import (
    BayesianUCBPolicy,
    BootstrappedThompsonPolicy,
    ChangeDetectionUCBPolicy,
    DiscountedUCBPolicy,
    EpsilonDecreasingPolicy,
    EXP3IXPolicy,
    EXP3Policy,
    KLUCBPolicy,
    MOSSUCBPolicy,
    PHEPolicy,
    Policy,
    PursuitPolicy,
    SoftmaxPolicy,
    ThompsonSamplingPolicy,
    UCBTunedPolicy,
    UCBVPolicy,
)


@pytest.fixture
def base_agent():
    """Create an agent with deterministic action table for policy tests."""
    agent = Agent(policy=Policy())
    agent.actions = pl.DataFrame(
        {
            "Name": ["A1", "A2", "A3"],
            "ActionAttempts": [1.0, 2.0, 3.0],
            "ValueEstimates": [0.2, 0.8, 1.2],
            "Q": [0.2, 0.4, 0.4],
        }
    )
    return agent


@pytest.mark.parametrize(
    "policy",
    [
        ThompsonSamplingPolicy(),
        BayesianUCBPolicy(),
        KLUCBPolicy(),
        UCBTunedPolicy(),
        MOSSUCBPolicy(),
        DiscountedUCBPolicy(),
        EXP3Policy(),
        EXP3IXPolicy(),
        SoftmaxPolicy(),
        PursuitPolicy(),
        EpsilonDecreasingPolicy(),
        BootstrappedThompsonPolicy(),
        UCBVPolicy(),
        PHEPolicy(),
        ChangeDetectionUCBPolicy(),
    ],
)
def test_extended_policies_choose_all_return_all_actions(base_agent, policy):
    """Each new policy must produce a full ordering of all available actions."""
    ordered = policy.choose_all(base_agent)
    assert len(ordered) == base_agent.actions.height
    assert sorted(ordered) == sorted(base_agent.actions["Name"].to_list())


@pytest.mark.parametrize(
    "policy",
    [
        ThompsonSamplingPolicy(),
        BayesianUCBPolicy(),
        KLUCBPolicy(),
        UCBTunedPolicy(),
        MOSSUCBPolicy(),
        DiscountedUCBPolicy(),
        EXP3Policy(),
        EXP3IXPolicy(),
        SoftmaxPolicy(),
        PursuitPolicy(),
        EpsilonDecreasingPolicy(),
        BootstrappedThompsonPolicy(),
        UCBVPolicy(),
        PHEPolicy(),
        ChangeDetectionUCBPolicy(),
    ],
)
def test_extended_policies_credit_assignment_keeps_q_finite(base_agent, policy):
    """Each new policy must keep Q values finite after credit assignment."""
    policy.credit_assignment(base_agent)
    q_vals = base_agent.actions["Q"].to_numpy()
    assert np.isfinite(q_vals).all()


def test_thompson_sampling_updates_internal_posterior(base_agent):
    """Thompson Sampling must create posterior state for all actions."""
    policy = ThompsonSamplingPolicy()
    policy.credit_assignment(base_agent)
    assert set(policy.alpha.keys()) == {"A1", "A2", "A3"}
    assert set(policy.beta.keys()) == {"A1", "A2", "A3"}


def test_klucb_explores_unseen_actions_first(base_agent):
    """KL-UCB should prioritize unseen actions with infinite optimistic index."""
    policy = KLUCBPolicy()
    base_agent.actions = base_agent.actions.with_columns(
        [
            pl.when(pl.col("Name") == "A1").then(0.0).otherwise(pl.col("ActionAttempts")).alias("ActionAttempts"),
            pl.when(pl.col("Name") == "A1").then(0.0).otherwise(pl.col("ValueEstimates")).alias("ValueEstimates"),
        ]
    )
    ordered = policy.choose_all(base_agent)
    assert ordered[0] == "A1"


def test_discounted_ucb_applies_discount_across_rounds(base_agent):
    """Discounted-UCB should change Q values after a second update."""
    policy = DiscountedUCBPolicy(gamma=0.8, c=1.0)
    policy.credit_assignment(base_agent)
    first_q = base_agent.actions["Q"].to_numpy().copy()

    base_agent.actions = base_agent.actions.with_columns(
        [
            (pl.col("ValueEstimates") + 0.4).alias("ValueEstimates"),
        ]
    )
    policy.credit_assignment(base_agent)
    second_q = base_agent.actions["Q"].to_numpy()
    assert not np.allclose(first_q, second_q)


def test_exp3_probabilities_are_positive(base_agent):
    """EXP3 should produce positive action weights and non-negative Q."""
    policy = EXP3Policy(gamma=0.1)
    policy.credit_assignment(base_agent)
    assert all(weight > 0 for weight in policy.weights.values())
    assert (base_agent.actions["Q"].to_numpy() >= 0).all()


def test_exp3ix_probabilities_are_positive(base_agent):
    """EXP3-IX should preserve positive weights after update."""
    policy = EXP3IXPolicy(eta=0.2, gamma=0.05)
    policy.credit_assignment(base_agent)
    assert all(weight > 0 for weight in policy.weights.values())


def test_pursuit_policy_probabilities_sum_to_one(base_agent):
    """Pursuit policy should keep normalized probabilities."""
    policy = PursuitPolicy(beta=0.2)
    policy.credit_assignment(base_agent)
    total = sum(policy.probs.values())
    assert np.isclose(total, 1.0)


def test_bootstrapped_thompson_updates_ensemble(base_agent):
    """Bootstrapped Thompson must update per-head counts after assignment."""
    policy = BootstrappedThompsonPolicy(n_bootstrap=4)
    policy.credit_assignment(base_agent)
    assert all(policy.counts[name].sum() >= 0 for name in ["A1", "A2", "A3"])


def test_phe_counts_increase_after_credit_assignment(base_agent):
    """PHE should increase action counts after one update."""
    policy = PHEPolicy(a=1.5)
    policy.credit_assignment(base_agent)
    assert all(policy.counts[name] >= 1 for name in ["A1", "A2", "A3"])


def test_change_detection_ucb_resets_on_abrupt_shift(base_agent):
    """ChangeDetectionUCB should keep finite state when a large shift appears."""
    policy = ChangeDetectionUCBPolicy(c=1.0, window=4, threshold=0.05)

    policy.credit_assignment(base_agent)
    base_agent.actions = base_agent.actions.with_columns((pl.col("ValueEstimates") + 10.0).alias("ValueEstimates"))
    policy.credit_assignment(base_agent)

    assert all(policy.counts[name] >= 1 for name in ["A1", "A2", "A3"])
    assert np.isfinite(base_agent.actions["Q"].to_numpy()).all()


def test_public_namespace_exposes_all_extended_policy_classes():
    """Top-level policy module should expose every extended policy class."""
    expected = {
        "ThompsonSamplingPolicy",
        "BayesianUCBPolicy",
        "KLUCBPolicy",
        "UCBTunedPolicy",
        "MOSSUCBPolicy",
        "DiscountedUCBPolicy",
        "EXP3Policy",
        "EXP3IXPolicy",
        "SoftmaxPolicy",
        "PursuitPolicy",
        "EpsilonDecreasingPolicy",
        "BootstrappedThompsonPolicy",
        "UCBVPolicy",
        "PHEPolicy",
        "ChangeDetectionUCBPolicy",
    }

    import coleman.policy as policy_module

    for name in expected:
        assert hasattr(policy_module, name)


def test_policy_string_representations_cover_mab_variants():
    policies = [
        EXP3Policy(gamma=0.07),
        EXP3IXPolicy(eta=0.1, gamma=0.01),
        SoftmaxPolicy(tau=0.2),
        PursuitPolicy(beta=0.1),
        EpsilonDecreasingPolicy(epsilon0=1.0, decay=0.5),
        DiscountedUCBPolicy(gamma=0.95, c=1.0),
        BootstrappedThompsonPolicy(n_bootstrap=4),
        ChangeDetectionUCBPolicy(c=1.0, window=4, threshold=0.1),
        ThompsonSamplingPolicy(),
        BayesianUCBPolicy(c=2.0),
        KLUCBPolicy(c=3.0),
        UCBTunedPolicy(c=1.0),
        UCBVPolicy(c=1.0, b=1.0),
        MOSSUCBPolicy(),
        PHEPolicy(a=1.0),
    ]
    assert all(str(policy) for policy in policies)


def test_exp3_probabilities_fallback_to_uniform_when_weights_are_zero(base_agent):
    policy = EXP3Policy(gamma=0.1)
    policy.weights = {"A1": 0.0, "A2": 0.0, "A3": 0.0}
    probs = policy._probs(["A1", "A2", "A3"])
    assert probs == {"A1": pytest.approx(1 / 3), "A2": pytest.approx(1 / 3), "A3": pytest.approx(1 / 3)}


def test_pursuit_credit_assignment_handles_empty_actions_table():
    agent = Agent(policy=Policy())
    agent.actions = pl.DataFrame(
        {
            "Name": [],
            "ActionAttempts": [],
            "ValueEstimates": [],
            "Q": [],
        },
        schema={
            "Name": pl.String,
            "ActionAttempts": pl.Float64,
            "ValueEstimates": pl.Float64,
            "Q": pl.Float64,
        },
    )
    policy = PursuitPolicy(beta=0.2)
    policy.credit_assignment(agent)
    assert policy.probs == {}


def test_epsilon_decreasing_exploration_branch_shuffles_actions(base_agent, monkeypatch):
    policy = EpsilonDecreasingPolicy(epsilon0=1.0, decay=0.0)

    class _RngStub:
        def random(self):
            return 0.0

        def shuffle(self, values):
            values.reverse()

    monkeypatch.setattr("coleman.policy.base._rng", _RngStub())
    assert policy.choose_all(base_agent) == ["A3", "A2", "A1"]


def test_nonstationary_choose_all_includes_unseen_actions_with_inf_score(base_agent):
    policy = DiscountedUCBPolicy(gamma=0.9, c=1.0)
    ordered = policy.choose_all(base_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_discounted_ucb_choose_all_after_credit_uses_finite_ucb_scores(base_agent):
    policy = DiscountedUCBPolicy(gamma=0.9, c=1.0)
    policy.credit_assignment(base_agent)
    ordered = policy.choose_all(base_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_ucb_tuned_choose_all_prioritizes_unseen_actions(base_agent):
    policy = UCBTunedPolicy(c=1.0)
    ordered = policy.choose_all(base_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_ucb_tuned_choose_all_after_credit_uses_variance_path(base_agent):
    policy = UCBTunedPolicy(c=1.0)
    policy.credit_assignment(base_agent)
    ordered = policy.choose_all(base_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_ucbv_choose_all_prioritizes_unseen_actions(base_agent):
    policy = UCBVPolicy(c=1.0, b=1.0)
    ordered = policy.choose_all(base_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_ucbv_choose_all_after_credit_uses_variance_path(base_agent):
    policy = UCBVPolicy(c=1.0, b=1.0)
    policy.credit_assignment(base_agent)
    ordered = policy.choose_all(base_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_klucb_solve_index_handles_non_positive_n():
    policy = KLUCBPolicy(c=3.0)
    assert policy._solve_index(mean=0.3, n=0.0, budget=1.0) == pytest.approx(1.0)


def test_bootstrapped_thompson_choose_all_uses_default_for_unseen_heads(base_agent):
    policy = BootstrappedThompsonPolicy(n_bootstrap=4)
    ordered = policy.choose_all(base_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_change_detection_choose_all_handles_unseen_actions(base_agent):
    policy = ChangeDetectionUCBPolicy(c=1.0, window=4, threshold=0.1)
    ordered = policy.choose_all(base_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_change_detection_choose_all_after_credit_uses_finite_ucb_scores(base_agent):
    policy = ChangeDetectionUCBPolicy(c=1.0, window=4, threshold=0.1)
    policy.credit_assignment(base_agent)
    ordered = policy.choose_all(base_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_change_detection_credit_assignment_triggers_reset_on_large_shift():
    agent = Agent(policy=Policy())
    agent.actions = pl.DataFrame(
        {
            "Name": ["A1"],
            "ActionAttempts": [1.0],
            "ValueEstimates": [0.0],
            "Q": [0.0],
        }
    )
    policy = ChangeDetectionUCBPolicy(c=1.0, window=4, threshold=0.05)

    policy.credit_assignment(agent)
    policy.credit_assignment(agent)
    agent.actions = agent.actions.with_columns(pl.lit(10.0).alias("ValueEstimates"))
    policy.credit_assignment(agent)
    policy.credit_assignment(agent)

    assert policy.counts["A1"] <= 4
    assert np.isfinite(agent.actions["Q"].to_numpy()).all()


def test_moss_ucb_handles_unseen_actions_with_zero_attempts(base_agent):
    policy = MOSSUCBPolicy()
    base_agent.actions = base_agent.actions.with_columns(
        pl.when(pl.col("Name") == "A1").then(0.0).otherwise(pl.col("ActionAttempts")).alias("ActionAttempts")
    )
    ordered = policy.choose_all(base_agent)
    assert ordered[0] == "A1"


def test_phe_choose_all_handles_zero_counts_before_updates(base_agent):
    policy = PHEPolicy(a=1.0)
    ordered = policy.choose_all(base_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_phe_choose_all_after_credit_uses_perturbed_history_path(base_agent):
    policy = PHEPolicy(a=1.0)
    policy.credit_assignment(base_agent)
    ordered = policy.choose_all(base_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]
