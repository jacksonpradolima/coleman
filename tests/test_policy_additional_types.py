"""Unit tests for newly added greedy/contextual/UCB policy groups."""

import numpy as np
import polars as pl
import pytest

from coleman.agent import Agent, ContextualAgent, RewardSlidingWindowAgent, SlidingWindowContextualAgent
from coleman.policy import (
    ContextualEpsilonGreedyPolicy,
    DecayEpsilonGreedyPolicy,
    LinTSPolicy,
    OptimisticGreedyPolicy,
    Policy,
    SlidingWindowUCBPolicy,
    SWContextualEpsilonGreedyPolicy,
    SWLinTSPolicy,
    UCB2Policy,
)


@pytest.fixture
def simple_agent():
    """Create a base agent with fixed action table."""
    agent = Agent(policy=Policy())
    agent.actions = pl.DataFrame(
        {
            "Name": ["A1", "A2", "A3"],
            "ActionAttempts": [0.0, 1.0, 2.0],
            "ValueEstimates": [0.0, 0.6, 0.8],
            "Q": [0.0, 0.6, 0.4],
        }
    )
    return agent


@pytest.fixture
def contextual_agent():
    """Create a contextual agent with simple 2D features."""

    def reward_fn(action, context):  # pylint: disable=unused-argument
        return 1.0

    agent = ContextualAgent(policy=Policy(), reward_function=reward_fn)
    agent.context_features = pl.DataFrame(
        {
            "Name": ["A1", "A2", "A3"],
            "feat1": [0.1, 0.7, 0.4],
            "feat2": [0.8, 0.2, 0.5],
        }
    )
    agent.features = ["feat1", "feat2"]
    agent.actions = pl.DataFrame(
        {
            "Name": ["A1", "A2", "A3"],
            "ActionAttempts": [1.0, 1.0, 1.0],
            "ValueEstimates": [0.3, 0.9, 0.5],
            "Q": [0.0, 0.0, 0.0],
        }
    )
    return agent


@pytest.fixture
def sw_contextual_agent():
    """Create a sliding-window contextual agent with history."""

    def reward_fn(action, context):  # pylint: disable=unused-argument
        return 1.0

    agent = SlidingWindowContextualAgent(policy=Policy(), reward_function=reward_fn, window_size=4)
    agent.context_features = pl.DataFrame(
        {
            "Name": ["A1", "A2", "A3"],
            "feat1": [0.2, 0.6, 0.3],
            "feat2": [0.9, 0.1, 0.4],
        }
    )
    agent.features = ["feat1", "feat2"]
    agent.actions = pl.DataFrame(
        {
            "Name": ["A1", "A2", "A3"],
            "ActionAttempts": [1.0, 1.0, 1.0],
            "ValueEstimates": [0.2, 0.6, 0.4],
            "Q": [0.0, 0.0, 0.0],
        }
    )
    agent.history = pl.DataFrame(
        {
            "Name": ["A1", "A2", "A1", "A3", "A2"],
            "ActionAttempts": [1.0, 1.0, 1.0, 1.0, 1.0],
            "ValueEstimates": [0.2, 0.5, 0.1, 0.6, 0.4],
            "Q": [0.2, 0.5, 0.1, 0.6, 0.4],
            "T": [1, 2, 3, 4, 5],
        }
    )
    agent.t = 6
    return agent


@pytest.fixture
def sw_reward_agent():
    """Create a sliding-window reward agent with history for SW-UCB tests."""

    def reward_fn(action, context):  # pylint: disable=unused-argument
        return 1.0

    agent = RewardSlidingWindowAgent(policy=Policy(), reward_function=reward_fn, window_size=3)
    agent.actions = pl.DataFrame(
        {
            "Name": ["A1", "A2", "A3"],
            "ActionAttempts": [1.0, 2.0, 3.0],
            "ValueEstimates": [0.5, 0.3, 0.8],
            "Q": [0.0, 0.0, 0.0],
        }
    )
    agent.history = pl.DataFrame(
        {
            "Name": ["A1", "A2", "A3", "A1", "A2", "A3"],
            "ActionAttempts": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ValueEstimates": [0.2, 0.1, 0.3, 0.4, 0.3, 0.5],
            "Q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "T": [1, 2, 3, 4, 5, 6],
        }
    )
    return agent


def test_decay_epsilon_greedy_returns_all_actions(simple_agent):
    """DecayEpsilonGreedy should return a full action ranking."""
    policy = DecayEpsilonGreedyPolicy(epsilon0=1.0, decay=0.7, min_epsilon=0.05)
    ordered = policy.choose_all(simple_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]
    assert "DecayEpsilonGreedy" in str(policy)


def test_optimistic_greedy_assigns_q0_to_unseen_action(simple_agent):
    """OptimisticGreedy should keep optimistic value for never-attempted actions."""
    policy = OptimisticGreedyPolicy(optimistic_q=2.0)
    policy.credit_assignment(simple_agent)
    a1_q = simple_agent.actions.filter(pl.col("Name") == "A1")["Q"][0]
    assert np.isclose(a1_q, 2.0)
    assert "OptimisticGreedy" in str(policy)


def test_ucb2_credit_assignment_keeps_q_finite(simple_agent):
    """UCB2 should produce finite Q values after confidence update."""
    policy = UCB2Policy(c=1.0, alpha=0.3)
    policy.credit_assignment(simple_agent)
    assert np.isfinite(simple_agent.actions["Q"].to_numpy()).all()


def test_ucb2_validation_and_string_repr():
    """UCB2 should validate alpha and expose a stable string representation."""
    with pytest.raises(ValueError, match="alpha must be positive"):
        UCB2Policy(c=1.0, alpha=0.0)

    assert "UCB2" in str(UCB2Policy(c=1.0, alpha=0.5))


def test_sliding_window_ucb_uses_recent_history(sw_reward_agent):
    """SlidingWindowUCB should update Q using recent window observations."""
    policy = SlidingWindowUCBPolicy(c=1.0)
    policy.credit_assignment(sw_reward_agent)
    assert np.isfinite(sw_reward_agent.actions["Q"].to_numpy()).all()


def test_sliding_window_ucb_fallback_paths_and_string(simple_agent):
    """SW-UCB should fallback to base behavior for non-sliding agents and empty history."""
    policy = SlidingWindowUCBPolicy(c=1.0)
    assert "SWUCB" in str(policy)

    # Non sliding-window agent branch.
    policy.credit_assignment(simple_agent)
    assert np.isfinite(simple_agent.actions["Q"].to_numpy()).all()

    # Empty-history branch for sliding-window reward agent.
    agent = RewardSlidingWindowAgent(policy=Policy(), reward_function=lambda *_: 1.0, window_size=3)
    agent.actions = simple_agent.actions.clone()
    agent.history = pl.DataFrame(
        {"Name": [], "ActionAttempts": [], "ValueEstimates": [], "Q": [], "T": []},
        schema={
            "Name": pl.Utf8,
            "ActionAttempts": pl.Float64,
            "ValueEstimates": pl.Float64,
            "Q": pl.Float64,
            "T": pl.Int64,
        },
    )
    policy.credit_assignment(agent)
    assert np.isfinite(agent.actions["Q"].to_numpy()).all()


def test_lints_choose_all_returns_all_actions(contextual_agent):
    """LinTS should rank all contextual actions."""
    policy = LinTSPolicy(alpha=0.4)
    policy.update_actions(contextual_agent, ["A1", "A2", "A3"])
    ordered = policy.choose_all(contextual_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_lints_and_sw_lints_string_repr():
    """LinTS variants should expose stable string representations."""
    assert "LinTS" in str(LinTSPolicy(alpha=0.2))
    assert "SWLinTS" in str(SWLinTSPolicy(alpha=0.2))


def test_contextual_epsilon_greedy_returns_all_actions(contextual_agent):
    """ContextualEpsilonGreedy should rank all contextual actions."""
    policy = ContextualEpsilonGreedyPolicy(epsilon=0.2)
    policy.update_actions(contextual_agent, ["A1", "A2", "A3"])
    ordered = policy.choose_all(contextual_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_contextual_epsilon_greedy_random_branch_and_string(contextual_agent):
    """Contextual epsilon-greedy should support pure exploration branch."""
    policy = ContextualEpsilonGreedyPolicy(epsilon=1.0)
    policy.update_actions(contextual_agent, ["A1", "A2", "A3"])
    ordered = policy.choose_all(contextual_agent)
    assert "ContextualEpsilonGreedy" in str(policy)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_sw_lints_requires_sliding_contextual_agent(contextual_agent):
    """SWLinTS must reject non-sliding contextual agents."""
    policy = SWLinTSPolicy(alpha=0.4)
    policy.update_actions(contextual_agent, ["A1", "A2", "A3"])
    with pytest.raises(TypeError, match="SlidingWindowContextualAgent"):
        policy.choose_all(contextual_agent)


def test_sw_lints_choose_all_with_sliding_agent(sw_contextual_agent):
    """SWLinTS should rank all actions for sliding-window contextual agent."""
    policy = SWLinTSPolicy(alpha=0.4)
    policy.update_actions(sw_contextual_agent, ["A1", "A2", "A3"])
    ordered = policy.choose_all(sw_contextual_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_sw_contextual_epsilon_greedy_choose_all(sw_contextual_agent):
    """SWContextualEpsilonGreedy should rank all actions with recency-aware scoring."""
    policy = SWContextualEpsilonGreedyPolicy(epsilon=0.1)
    policy.update_actions(sw_contextual_agent, ["A1", "A2", "A3"])
    ordered = policy.choose_all(sw_contextual_agent)
    assert sorted(ordered) == ["A1", "A2", "A3"]


def test_sw_contextual_epsilon_greedy_random_and_type_guard(sw_contextual_agent, contextual_agent):
    """SW contextual epsilon-greedy should support type guard and exploration branch."""
    policy = SWContextualEpsilonGreedyPolicy(epsilon=1.0)
    policy.update_actions(sw_contextual_agent, ["A1", "A2", "A3"])
    ordered = policy.choose_all(sw_contextual_agent)
    assert "SWContextualEpsilonGreedy" in str(policy)
    assert sorted(ordered) == ["A1", "A2", "A3"]

    policy.update_actions(contextual_agent, ["A1", "A2", "A3"])
    with pytest.raises(TypeError, match="SlidingWindowContextualAgent"):
        policy.choose_all(contextual_agent)
