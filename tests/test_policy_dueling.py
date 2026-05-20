"""Unit tests for dueling and ranking bandit policies."""

import numpy as np
import polars as pl
import pytest

from coleman.agent import Agent
from coleman.policy import (
    DuelingUCBPolicy,
    PairwiseThompsonRankingPolicy,
    Policy,
)


def _build_agent() -> Agent:
    agent = Agent(policy=Policy())
    agent.actions = pl.DataFrame(
        {
            "Name": ["A1", "A2", "A3", "A4"],
            "ActionAttempts": [0.0, 2.0, 3.0, 1.0],
            "ValueEstimates": [0.0, 0.8, 1.2, 0.3],
            "Q": [0.0, 0.4, 0.4, 0.3],
        }
    )
    return agent


def test_dueling_ucb_updates_pairwise_state_and_q():
    agent = _build_agent()
    policy = DuelingUCBPolicy(c=1.0)
    agent.last_prioritization = ["A3", "A2", "A4", "A1"]

    policy.credit_assignment(agent)
    ordered = policy.choose_all(agent)

    assert "DuelingUCB" in str(policy)
    assert len(policy._duels) > 0
    assert len(ordered) == agent.actions.height
    assert np.isfinite(agent.actions["Q"].to_numpy()).all()


def test_dueling_pair_and_copeland_empty_history_branch():
    policy = DuelingUCBPolicy(c=1.0)
    assert policy._pair("Z", "A") == ("A", "Z")

    # With no duel history, n <= 0 branch in Copeland scoring is used.
    scores = policy._copeland_scores(["B", "A"])
    assert set(scores.keys()) == {"A", "B"}
    assert all(v >= 0.0 for v in scores.values())


def test_pairwise_thompson_ranking_keeps_full_order_and_finite_q():
    agent = _build_agent()
    policy = PairwiseThompsonRankingPolicy()

    ordered = policy.choose_all(agent)
    agent.last_prioritization = ordered
    policy.credit_assignment(agent)

    assert len(ordered) == agent.actions.height
    assert sorted(ordered) == sorted(agent.actions["Name"].to_list())
    assert np.isfinite(agent.actions["Q"].to_numpy()).all()


def test_dueling_validations_and_pairwise_reverse_branch():
    with pytest.raises(ValueError, match="Exploration parameter c must be positive"):
        DuelingUCBPolicy(c=0)

    # Force reverse lexicographic branch in _sample_pref and beta update path.
    policy = PairwiseThompsonRankingPolicy(alpha_prior=2.0, beta_prior=2.0)
    assert "PairwiseThompsonRanking" in str(policy)
    pref = policy._sample_pref("B", "A", policy._alpha, policy._beta)
    assert 0.0 <= pref <= 1.0

    agent = Agent(policy=Policy())
    agent.actions = pl.DataFrame(
        {
            "Name": ["B", "A"],
            "ActionAttempts": [1.0, 1.0],
            "ValueEstimates": [0.4, 0.5],
            "Q": [0.0, 0.0],
        }
    )
    agent.last_prioritization = ["B", "A"]
    policy.credit_assignment(agent)
    key = policy._pair("B", "A")
    assert policy._beta[key] > 2.0


def test_public_namespace_exposes_dueling_policies():
    expected = {"DuelingUCBPolicy", "PairwiseThompsonRankingPolicy"}

    import coleman.policy as policy_module

    for name in expected:
        assert hasattr(policy_module, name)
