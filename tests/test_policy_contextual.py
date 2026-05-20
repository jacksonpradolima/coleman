"""Unit tests for contextual bandit policies."""

import numpy as np
import polars as pl
import pytest

from coleman.agent import Agent
from coleman.exceptions import QException
from coleman.policy import (
    ContextualEpsilonGreedyPolicy,
    Policy,
    SWContextualEpsilonGreedyPolicy,
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


def test_contextual_epsilon_invalid_score_shape_raises_qexception():
    from coleman.agent import ContextualAgent
    from coleman.reward import TimeRankReward

    # Create a ContextualAgent instead of base Agent
    reward_func = TimeRankReward()
    policy = ContextualEpsilonGreedyPolicy(epsilon=0.0)
    agent = ContextualAgent(policy, reward_func)
    agent.add_action("A1")
    agent.context_features = pl.DataFrame({"Name": ["A1"], "f1": [1.0], "f2": [0.5]})
    agent.features = ["f1", "f2"]

    policy.update_actions(agent, ["A1"])
    policy.context["b"]["A1"] = np.zeros((2, 2))

    with pytest.raises(QException, match="invalid score shape"):
        policy.choose_all(agent)


def test_sw_contextual_epsilon_invalid_score_shape_raises_qexception():
    from coleman.agent import SlidingWindowContextualAgent

    sw_agent = SlidingWindowContextualAgent(policy=Policy(), reward_function=lambda *_: 1.0, window_size=2)
    sw_agent.context_features = pl.DataFrame({"Name": ["A1"], "f1": [1.0], "f2": [0.5]})
    sw_agent.features = ["f1", "f2"]
    sw_agent.history = pl.DataFrame(
        {"Name": ["A1"], "ActionAttempts": [1.0], "ValueEstimates": [0.2], "Q": [0.2], "T": [1]}
    )
    sw_agent.t = 3

    policy = SWContextualEpsilonGreedyPolicy(epsilon=0.0)
    policy.update_actions(sw_agent, ["A1"])
    policy.context["b"]["A1"] = np.zeros((2, 2))

    with pytest.raises(QException, match="invalid score shape"):
        policy.choose_all(sw_agent)


def test_public_namespace_exposes_contextual_policies():
    expected = {"ContextualEpsilonGreedyPolicy", "SWContextualEpsilonGreedyPolicy"}

    import coleman.policy as policy_module

    for name in expected:
        assert hasattr(policy_module, name)
