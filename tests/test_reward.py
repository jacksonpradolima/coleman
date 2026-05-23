"""
Unit tests for reward functions in the coleman.reward module.

This module provides unit tests for the TimeRankReward and RNFailReward classes,
which are part of the multi-armed bandit framework for test case prioritization.
The tests cover the following aspects:

- Correctness of reward function outputs for varying scenarios.
- Handling of edge cases such as no detections.
- Proper representation and naming of reward classes
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from coleman.evaluation import EvaluationMetric
from coleman.reward import (
    APFDcReward,
    DiscountedFailureReward,
    ReciprocalRankReward,
    RNFailReward,
    TimeRankReward,
    TopKRNFailReward,
)


@pytest.fixture
def mock_evaluation_metric():
    """
    Provides a mock evaluation metric for testing reward functions.
    """
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = [1, 3, 5]  # Failures detected at these ranks
    mock_metric.scheduled_testcases = ["Test1", "Test2", "Test3", "Test4", "Test5"]
    mock_metric.detected_failures = True
    return mock_metric


@pytest.fixture
def mock_empty_evaluation_metric():
    """
    Provides an empty mock evaluation metric for edge case testing.
    """
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = []
    mock_metric.scheduled_testcases = ["Test1", "Test2", "Test3", "Test4", "Test5"]
    mock_metric.detected_failures = False
    return mock_metric


@pytest.fixture
def sample_prioritization():
    """
    Provides a sample prioritization list.
    """
    return ["Test1", "Test2", "Test3", "Test4", "Test5"]


def test_time_rank_reward_name():
    """
    Test the string representation and name of TimeRankReward.
    """
    reward = TimeRankReward()
    assert str(reward) == "Time-ranked Reward"
    assert reward.get_name() == "timerank"


def test_time_rank_reward_evaluation(mock_evaluation_metric, sample_prioritization):
    """
    Test the evaluation method of TimeRankReward.
    """
    reward = TimeRankReward()
    results = reward.evaluate(mock_evaluation_metric, sample_prioritization)
    print("Scheduled Testcases:", mock_evaluation_metric.scheduled_testcases)
    print("Results:", results)
    # Updated expected results to reflect the cumulative logic
    assert np.allclose(results, [1.0, 0.3333333333, 1.0, 0.6666666667, 1.0])


@pytest.mark.parametrize(
    "detection_ranks, expected",
    [
        ([1, 2], [1.0, 1.0, 1.0, 1.0, 1.0]),  # Updated expectation for cumulative logic
        ([3], [0.0, 0.0, 1.0, 1.0, 1.0]),  # Adjusted expectation
        ([], [0.0, 0.0, 0.0, 0.0, 0.0]),  # No detections result in zero rewards
    ],
)
def test_time_rank_reward_varied_detections(detection_ranks, expected, sample_prioritization):
    """
    Test TimeRankReward with varied detection ranks.
    """
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = detection_ranks
    mock_metric.scheduled_testcases = ["Test1", "Test2", "Test3", "Test4", "Test5"]
    reward = TimeRankReward()
    results = reward.evaluate(mock_metric, sample_prioritization)
    assert np.allclose(results, expected)


def test_rn_fail_reward_name():
    """
    Test the string representation and name of RNFailReward.
    """
    reward = RNFailReward()
    assert str(reward) == "Reward Based on Failures"
    assert reward.get_name() == "RNFail"


def test_rn_fail_reward_evaluation(mock_evaluation_metric, sample_prioritization):
    """
    Test the evaluation method of RNFailReward.
    """
    reward = RNFailReward()
    results = reward.evaluate(mock_evaluation_metric, sample_prioritization)
    assert np.allclose(results, [1.0, 0.0, 1.0, 0.0, 1.0])


def test_rn_fail_reward_no_failures(mock_empty_evaluation_metric, sample_prioritization):
    """
    Test RNFailReward evaluation with no failures.
    """
    reward = RNFailReward()
    results = reward.evaluate(mock_empty_evaluation_metric, sample_prioritization)
    assert np.allclose(results, [0.0] * len(sample_prioritization))


def test_topk_rn_fail_reward_precision_style_behavior():
    """TopKRNFailReward should encode failure rate over the first k tests."""
    reward = TopKRNFailReward(top_k=6)
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = [1, 2, 3, 4, 5, 6]
    prioritization = [f"Test{i}" for i in range(1, 11)]

    values = reward.evaluate(mock_metric, prioritization)
    assert np.allclose(values[:6], [1 / 6] * 6)
    assert np.allclose(values[6:], [0.0] * 4)
    assert sum(values) == pytest.approx(1.0)


def test_topk_rn_fail_reward_can_use_time_budget_cap():
    """Budget mode should cap k by the number of scheduled tests."""
    reward = TopKRNFailReward(top_k=4, use_time_budget=True)
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = [1, 2, 3]
    mock_metric.scheduled_testcases = ["Test1", "Test2"]
    prioritization = [f"Test{i}" for i in range(1, 6)]

    values = reward.evaluate(mock_metric, prioritization)
    assert np.allclose(values, [0.5, 0.5, 0.0, 0.0, 0.0])


def test_topk_rn_fail_reward_default_mode_ignores_budget_cap():
    """Legacy mode should continue to use only top-k regardless of scheduled count."""
    reward = TopKRNFailReward(top_k=4, use_time_budget=False)
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = [1, 2, 3]
    mock_metric.scheduled_testcases = ["Test1", "Test2"]
    prioritization = [f"Test{i}" for i in range(1, 6)]

    values = reward.evaluate(mock_metric, prioritization)
    assert np.allclose(values, [0.25, 0.25, 0.25, 0.0, 0.0])


def test_discounted_failure_reward_values():
    """DiscountedFailureReward should prioritize early failures."""
    reward = DiscountedFailureReward()
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = [1, 3, 5]
    prioritization = [f"Test{i}" for i in range(1, 6)]

    values = reward.evaluate(mock_metric, prioritization)
    expected = [1.0, 0.0, 1 / np.log2(4), 0.0, 1 / np.log2(6)]
    assert np.allclose(values, expected)


def test_apfdc_reward_matches_cost_contributions():
    """APFDcReward should distribute the cost-aware score across failing ranks."""
    reward_metric = MagicMock(spec=EvaluationMetric)
    reward_metric.detection_ranks = [1, 3]
    reward_metric.detected_failures = 2
    reward_metric.undetected_failures = 1
    reward_metric.testcase_costs = [2.0, 1.0, 3.0]
    prioritization = ["Test1", "Test2", "Test3"]

    reward = APFDcReward()
    values = reward.evaluate(reward_metric, prioritization)

    total_cost = sum(reward_metric.testcase_costs)
    expected_first = ((2.0 + 1.0 + 3.0) - 0.5 * 2.0) / (total_cost * 3)
    expected_third = (3.0 - 0.5 * 3.0) / (total_cost * 3)
    assert np.allclose(values, [expected_first, 0.0, expected_third])
    assert sum(values) == pytest.approx(expected_first + expected_third)


def test_apfdc_reward_handles_empty_prioritization():
    reward_metric = MagicMock(spec=EvaluationMetric)
    reward_metric.detection_ranks = [1]
    reward_metric.detected_failures = 1
    reward_metric.undetected_failures = 0
    reward_metric.testcase_costs = [1.0]

    reward = APFDcReward()
    assert reward.evaluate(reward_metric, []) == []


def test_apfdc_reward_returns_zeros_when_costs_missing_or_invalid():
    reward = APFDcReward()
    prioritization = ["Test1", "Test2"]

    no_costs_metric = MagicMock(spec=EvaluationMetric)
    no_costs_metric.detection_ranks = [1]
    no_costs_metric.detected_failures = 1
    no_costs_metric.undetected_failures = 0
    no_costs_metric.testcase_costs = None
    assert reward.evaluate(no_costs_metric, prioritization) == [0.0, 0.0]

    zero_cost_metric = MagicMock(spec=EvaluationMetric)
    zero_cost_metric.detection_ranks = [1]
    zero_cost_metric.detected_failures = 1
    zero_cost_metric.undetected_failures = 0
    zero_cost_metric.testcase_costs = [0.0, 0.0]
    assert reward.evaluate(zero_cost_metric, prioritization) == [0.0, 0.0]


def test_discounted_failure_reward_handles_empty_inputs():
    reward = DiscountedFailureReward()
    empty_metric = MagicMock(spec=EvaluationMetric)
    empty_metric.detection_ranks = []
    assert reward.evaluate(empty_metric, []) == []
    assert reward.evaluate(empty_metric, ["Test1", "Test2"]) == [0.0, 0.0]


def test_reciprocal_rank_reward_handles_empty_inputs():
    reward = ReciprocalRankReward()
    empty_metric = MagicMock(spec=EvaluationMetric)
    empty_metric.detection_ranks = []
    assert reward.evaluate(empty_metric, []) == []
    assert reward.evaluate(empty_metric, ["Test1", "Test2"]) == [0.0, 0.0]


def test_topk_rn_fail_reward_rejects_non_positive_k():
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        TopKRNFailReward(top_k=0)


def test_topk_rn_fail_reward_handles_empty_prioritization_and_budget_cap():
    reward = TopKRNFailReward(top_k=4, use_time_budget=True)
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = [1, 2]
    mock_metric.scheduled_testcases = []
    assert reward.evaluate(mock_metric, []) == []

    prioritization = ["Test1", "Test2"]
    values = reward.evaluate(mock_metric, prioritization)
    assert values == [0.0, 0.0]


@pytest.mark.benchmark(group="reward")
@pytest.mark.parametrize("num_testcases", [100, 1000, 10000])
def test_time_rank_reward_performance(benchmark, num_testcases):
    """
    Performance test for TimeRankReward evaluation method.
    This test evaluates the performance of the reward function for larger datasets.
    """
    # Mocking a large dataset
    detection_ranks = list(range(1, num_testcases + 1, 2))  # Failures at odd indices
    scheduled_testcases = [f"Test{i}" for i in range(1, num_testcases + 1)]
    sample_prioritization = scheduled_testcases  # Assume prioritization is the same as scheduling

    # Mocking EvaluationMetric
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = detection_ranks
    mock_metric.scheduled_testcases = scheduled_testcases

    reward = TimeRankReward()

    # Benchmark the evaluation method
    def run_evaluation():
        reward.evaluate(mock_metric, sample_prioritization)

    benchmark(run_evaluation)


@pytest.mark.benchmark(group="reward")
@pytest.mark.parametrize("num_testcases", [100, 1000, 10000])
def test_rn_fail_reward_performance(benchmark, num_testcases):
    """
    Performance test for RNFailReward evaluation method.
    This test evaluates the performance of the reward function for larger datasets.
    """
    # Mocking a large dataset
    detection_ranks = list(range(1, num_testcases + 1, 2))  # Failures at odd indices
    scheduled_testcases = [f"Test{i}" for i in range(1, num_testcases + 1)]
    sample_prioritization = scheduled_testcases  # Assume prioritization is the same as scheduling

    # Mocking EvaluationMetric
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.detection_ranks = detection_ranks
    mock_metric.scheduled_testcases = scheduled_testcases
    mock_metric.detected_failures = True

    reward = RNFailReward()

    # Benchmark the evaluation method
    def run_evaluation():
        reward.evaluate(mock_metric, sample_prioritization)

    benchmark(run_evaluation)
