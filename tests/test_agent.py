"""
coleman.agent.tests
-----------------------

This module contains unit and performance tests for the `Agent` module in the Coleman framework.

Tests ensure the functionality and efficiency of the following classes:
- `Agent`: Base class for agents, including action management, selection, and reward observation.
- `RewardAgent`: Extends `Agent` to incorporate reward-based learning.
- `ContextualAgent`: Extends `RewardAgent` with context-aware decision-making.
- `RewardSlidingWindowAgent`: Combines reward-based learning with a sliding window for historical data.
- `SlidingWindowContextualAgent`: Combines the sliding window mechanism with context-aware decision-making.

Features Tested
---------------
- Initialization, action management, and reward observation methods for all agent types.
- Contextual updates and historical data management for contextual and sliding window agents.
- Performance benchmarks for scaling with a large number of test cases.

Performance Benchmark Results
-----------------------------
- `Agent` execution time reduced by ~69% from 92s to 28s in large-scale experiments.
- Significant improvements in reward-based decision-making speed.
"""

import random
from unittest.mock import MagicMock

import polars as pl
import pytest

from coleman.agent import (
    Agent,
    ContextualAgent,
    RewardAgent,
    RewardSlidingWindowAgent,
    SlidingWindowContextualAgent,
)
from coleman.bandit import Bandit
from coleman.evaluation import EvaluationMetric


@pytest.fixture
def mock_bandit():
    """
    Provides a mock Bandit instance.
    """
    bandit = MagicMock(spec=Bandit)
    bandit.get_arms.return_value = ["Test1", "Test2", "Test3"]
    return bandit


@pytest.fixture
def mock_policy():
    """
    Provides a mock policy for the agents.
    """
    policy = MagicMock()
    policy.choose_all.return_value = ["Test1", "Test2", "Test3"]
    policy.credit_assignment = MagicMock()
    return policy


@pytest.fixture
def mock_evaluation_metric():
    """
    Provides a mock evaluation metric for testing.
    """
    metric = MagicMock(spec=EvaluationMetric)
    metric.detection_ranks = [1, 2]
    metric.scheduled_testcases = ["Test1", "Test2", "Test3"]
    return metric


def test_agent_initialization(mock_policy, mock_bandit):
    """
    Test the initialization of the base Agent class.
    """
    agent = Agent(mock_policy, mock_bandit)
    assert agent.policy == mock_policy
    assert agent.bandit == mock_bandit
    assert agent.t == 0
    assert isinstance(agent.actions, pl.DataFrame)


def test_agent_add_action():
    """
    Test adding an action to the agent.
    """
    agent = Agent(MagicMock())
    agent.add_action("Test1")
    assert "Test1" in agent.actions["Name"].to_list()


def test_agent_update_actions():
    """
    Test updating the action set of the agent.
    """
    agent = Agent(MagicMock())
    agent.update_actions(["Test1", "Test2"])
    assert all(name in agent.actions["Name"].to_list() for name in ["Test1", "Test2"])


def test_agent_choose(mock_policy, mock_bandit):
    """
    Test the action selection logic of the agent.
    """
    agent = Agent(mock_policy, mock_bandit)
    agent.update_actions(["Test1", "Test2", "Test3"])
    chosen = agent.choose()
    if agent.t == 0:
        # Check if the initial random choice contains all actions
        assert set(chosen) == set(["Test1", "Test2", "Test3"])
    else:
        assert chosen == mock_policy.choose_all.return_value


def test_agent_choose_delegates_after_first_turn(mock_policy, mock_bandit):
    """Cover Agent.choose() for t > 0 and null Q filling."""
    mock_policy.choose_all.return_value = ["Test3", "Test2", "Test1"]
    agent = Agent(mock_policy, mock_bandit)
    agent.update_actions(["Test1", "Test2", "Test3"])
    agent.t = 1
    agent.actions = agent.actions.with_columns(pl.lit(None).alias("Q"))

    chosen = agent.choose()

    assert chosen == ["Test3", "Test2", "Test1"]
    mock_policy.choose_all.assert_called_once_with(agent)
    assert agent.actions["Q"].null_count() == 0


def test_reward_agent_observe(mock_policy, mock_bandit, mock_evaluation_metric):
    """
    Test the observe method of RewardAgent.
    """
    reward_function = MagicMock()
    reward_function.evaluate.return_value = [1.0, 0.5, 0.0]

    agent = RewardAgent(mock_policy, reward_function)
    agent.update_bandit(mock_bandit)
    agent.update_actions(["Test1", "Test2", "Test3"])
    agent.choose()
    agent.observe(mock_evaluation_metric)

    assert agent.last_reward == [1.0, 0.5, 0.0]


def test_contextual_agent_context_update():
    """
    Test the context update functionality of ContextualAgent.
    """
    agent = ContextualAgent(MagicMock(), MagicMock())
    context_features = pl.DataFrame({"Name": ["tc1"], "feature1": [1], "feature2": [2]})
    agent.update_context(context_features)
    assert agent.context_features.equals(context_features)


def test_reward_sliding_window_agent_observe(mock_policy, mock_bandit, mock_evaluation_metric):
    """
    Test the observe method of RewardSlidingWindowAgent.
    """
    reward_function = MagicMock()
    reward_function.evaluate.return_value = [1.0, 0.5, 0.0]

    agent = RewardSlidingWindowAgent(mock_policy, reward_function, window_size=2)
    agent.update_bandit(mock_bandit)
    agent.choose()
    agent.observe(mock_evaluation_metric)

    assert len(agent.history) > 0
    assert "T" in agent.history.columns


def test_reward_sliding_window_agent_observe_empty_prioritization(mock_policy, mock_bandit, mock_evaluation_metric):
    """Cover the empty-prioritization branch in RewardSlidingWindowAgent.observe()."""
    reward_function = MagicMock()
    reward_function.evaluate.return_value = []

    agent = RewardSlidingWindowAgent(mock_policy, reward_function, window_size=2)
    agent.update_bandit(mock_bandit)
    agent.update_actions(["Test1", "Test2"])
    agent.last_prioritization = []

    agent.observe(mock_evaluation_metric)

    assert agent.last_reward == []
    assert agent.t == 1
    assert agent.actions["ValueEstimates"].to_list() == [0.0, 0.0]


def test_sliding_window_contextual_agent_history_truncation():
    """
    Test the history truncation logic of SlidingWindowContextualAgent.
    """
    agent = SlidingWindowContextualAgent(MagicMock(), MagicMock(), window_size=2)
    agent.history = pl.DataFrame(
        {
            "Name": ["Test1", "Test2", "Test3"],
            "ActionAttempts": [1.0, 1.0, 1.0],
            "ValueEstimates": [0.1, 0.2, 0.3],
            "Q": [0.1, 0.2, 0.3],
            "T": [1, 2, 3],
        }
    )

    agent.update_history()
    assert len(agent.history) == 2
    assert agent.history["T"].min() == 2


def test_reward_sliding_window_agent_update_history_empty_history(mock_policy):
    """Cover RewardSlidingWindowAgent.update_history() with no rows."""
    reward_function = MagicMock()
    agent = RewardSlidingWindowAgent(mock_policy, reward_function, window_size=2)

    agent.update_history()

    assert agent.history.is_empty()


def test_sliding_window_contextual_agent_observe_empty_prioritization(mock_evaluation_metric):
    """Cover the empty-prioritization branch in SlidingWindowContextualAgent.observe()."""
    policy = MagicMock()
    policy.choose_all.return_value = []
    policy.credit_assignment = MagicMock()
    policy.update_actions = MagicMock()

    reward_function = MagicMock()
    reward_function.evaluate.return_value = []

    agent = SlidingWindowContextualAgent(policy, reward_function, window_size=5)
    agent.update_actions(["A", "B"])
    agent.last_prioritization = []

    agent.observe(mock_evaluation_metric)

    assert agent.last_reward == []
    assert agent.t == 1
    assert agent.actions["ValueEstimates"].to_list() == [0.0, 0.0]


def test_sliding_window_contextual_agent_update_history_empty_history():
    """Cover SlidingWindowContextualAgent.update_history() with no rows."""
    agent = SlidingWindowContextualAgent(MagicMock(), MagicMock(), window_size=2)

    agent.update_history()

    assert agent.history.is_empty()


@pytest.mark.benchmark(group="agent")
@pytest.mark.parametrize("action_count", [100, 1000, 10000])
def test_agent_performance(benchmark, action_count):
    """
    Benchmark the performance of the Agent class with a large number of actions.
    """
    agent = Agent(MagicMock())
    actions = [f"Test{i}" for i in range(action_count)]

    def update_actions():
        agent.update_actions(actions)

    benchmark(update_actions)
    assert len(agent.actions) == action_count


@pytest.mark.benchmark(group="agent")
@pytest.mark.parametrize("action_count", [100, 1000, 10000])
def test_reward_agent_performance(benchmark, mock_policy, action_count):
    """
    Benchmark the performance of the RewardAgent class with a large number of actions.
    """
    reward_function = MagicMock()
    reward_function.evaluate.return_value = [random.random() for _ in range(action_count)]

    agent = RewardAgent(mock_policy, reward_function)
    actions = [f"Test{i}" for i in range(action_count)]

    agent.update_actions(actions)
    agent.choose()

    def observe():
        agent.observe(MagicMock())

    benchmark(observe)
    assert len(agent.actions) == action_count


def test_agent_observe_direct(mock_policy, mock_bandit):
    """Test Agent.observe() with a direct reward list (base class path)."""
    agent = Agent(mock_policy, mock_bandit)
    agent.update_actions(["Test1", "Test2", "Test3"])
    agent.choose()
    t_before = agent.t
    agent.observe([1.0, 0.5, 0.0])
    assert agent.t == t_before + 1
    assert "ValueEstimates" in agent.actions.columns


def test_agent_observe_empty_reward_list_increments_time(mock_policy, mock_bandit):
    """Cover the early-return branch in Agent.observe()."""
    agent = Agent(mock_policy, mock_bandit)
    agent.update_actions(["Test1", "Test2"])
    agent.last_prioritization = ["Test1", "Test2"]
    t_before = agent.t

    agent.observe([])

    assert agent.t == t_before + 1
    assert agent.actions["ValueEstimates"].to_list() == [0.0, 0.0]


def test_agent_observe_keeps_estimate_when_reward_missing(mock_policy, mock_bandit):
    """Cover observe() branch where an action has no mapped reward (lines 200-201)."""
    agent = Agent(mock_policy, mock_bandit)
    agent.update_actions(["A", "B", "C"])
    agent.last_prioritization = ["A", "B", "C"]
    agent.actions = agent.actions.with_columns([pl.Series("ActionAttempts", [1.0, 1.0, 1.0])])
    before = agent.actions["ValueEstimates"].to_list()

    # Missing reward for action C -> observed_reward is None for that row.
    agent.observe([1.0, 0.0])

    after = agent.actions["ValueEstimates"].to_list()
    assert after[2] == before[2]


def test_contextual_agent_str():
    """Test ContextualAgent.__str__ returns the policy string."""
    policy = MagicMock()
    policy.__str__ = MagicMock(return_value="MockPolicy")
    agent = ContextualAgent(policy, MagicMock())
    assert str(agent) == "MockPolicy"


def test_contextual_agent_choose():
    """Test ContextualAgent.choose() delegates to policy.choose_all."""
    policy = MagicMock()
    policy.choose_all.return_value = ["A", "B", "C"]
    agent = ContextualAgent(policy, MagicMock())
    agent.update_actions(["A", "B", "C"])
    result = agent.choose()
    assert result == ["A", "B", "C"]
    policy.choose_all.assert_called_once_with(agent)


def test_contextual_agent_update_actions():
    """Test ContextualAgent.update_actions filters and adds actions."""
    policy = MagicMock()
    policy.update_actions = MagicMock()
    agent = ContextualAgent(policy, MagicMock())
    agent.update_actions(["A", "B"])
    assert "A" in agent.actions["Name"].to_list()
    assert "B" in agent.actions["Name"].to_list()
    agent.update_actions(["B", "C"])
    assert "A" not in agent.actions["Name"].to_list()
    assert "C" in agent.actions["Name"].to_list()


def test_contextual_agent_update_bandit():
    """Test ContextualAgent.update_bandit sets the bandit and updates actions."""
    policy = MagicMock()
    policy.update_actions = MagicMock()
    bandit = MagicMock()
    bandit.get_arms.return_value = ["TC1", "TC2"]
    agent = ContextualAgent(policy, MagicMock())
    agent.update_bandit(bandit)
    assert agent.bandit is bandit
    assert "TC1" in agent.actions["Name"].to_list()


def test_contextual_agent_update_features():
    """Test ContextualAgent.update_features stores the features list."""
    agent = ContextualAgent(MagicMock(), MagicMock())
    agent.update_features(["feat1", "feat2"])
    assert agent.features == ["feat1", "feat2"]


def test_reward_sliding_window_agent_history_truncation(mock_policy, mock_bandit, mock_evaluation_metric):
    """Test RewardSlidingWindowAgent.update_history prunes old T entries beyond window."""
    reward_function = MagicMock()
    reward_function.evaluate.return_value = [1.0, 0.5, 0.0]

    agent = RewardSlidingWindowAgent(mock_policy, reward_function, window_size=1)
    agent.update_bandit(mock_bandit)

    # Observe twice to force truncation (window_size=1 means only 1 unique T kept)
    agent.choose()
    agent.observe(mock_evaluation_metric)
    agent.choose()
    agent.observe(mock_evaluation_metric)

    # Only the latest T should remain after truncation
    unique_t = agent.history["T"].unique().to_list()
    assert len(unique_t) <= 1


def test_sliding_window_contextual_agent_str():
    """Test SlidingWindowContextualAgent.__str__ includes window size."""
    policy = MagicMock()
    policy.__str__ = MagicMock(return_value="SWLinUCB (Alpha=0.5")
    agent = SlidingWindowContextualAgent(policy, MagicMock(), window_size=7)
    result = str(agent)
    assert "7" in result


def test_sliding_window_contextual_agent_observe(mock_evaluation_metric):
    """Test SlidingWindowContextualAgent.observe updates value estimates via reward function."""
    policy = MagicMock()
    policy.choose_all.return_value = ["A", "B"]
    policy.credit_assignment = MagicMock()
    policy.update_actions = MagicMock()

    reward_function = MagicMock()
    reward_function.evaluate.return_value = [1.0, 0.0]

    agent = SlidingWindowContextualAgent(policy, reward_function, window_size=5)
    agent.update_actions(["A", "B"])
    agent.choose()
    agent.observe(mock_evaluation_metric)

    assert agent.t == 1
    assert len(agent.history) > 0
