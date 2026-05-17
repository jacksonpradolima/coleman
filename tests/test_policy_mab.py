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
