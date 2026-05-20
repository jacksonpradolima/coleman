"""
coleman/tests/test_bandit.py

Unit and performance tests for the Bandit classes in the coleman package. This module tests the functionality of
the `Bandit`, `DynamicBandit`, and `EvaluationMetricBandit` classes, including:

- Initialization and management of arms (test cases).
- Handling of duplicate arms and updates to priorities.
- Dynamic updates of arms in `DynamicBandit`.
- Integration with evaluation metrics in `EvaluationMetricBandit`.
- Validation of actions and error handling for invalid input.
- Performance benchmarking for core operations like adding arms and updating priorities.

The test suite uses mock classes (`MockBandit` and
`MockDynamicBandit`) to simulate behavior for the abstract base classes.

Fixtures:
- `sample_arms`: Provides a sample set of test cases with attributes such as `Name`, `Duration`, and `Verdict`.
- `mock_evaluation_metric`: Creates a mock evaluation metric for testing `EvaluationMetricBandit`.

Coverage:
- Initialization and reset of bandits.
- Adding and retrieving arms.
- Updating priorities based on actions, including scenarios with duplicate arms.
- Integration with evaluation metrics for test prioritization.
- Error handling for invalid actions and empty action lists.

Performance Tests:
- Adding a large number of arms to `Bandit`.
- Updating priorities for large action sets.
- Pulling actions with `EvaluationMetricBandit` for test suites of varying sizes
"""

from unittest.mock import MagicMock

import polars as pl
import pytest

from coleman.bandit import Bandit, DynamicBandit, EvaluationMetricBandit
from coleman.evaluation import EvaluationMetric


class MockBandit(Bandit):
    """
    A mock implementation of the abstract Bandit class for testing purposes.
    """

    def pull(self, action):
        """Mock implementation of the abstract pull method."""


class MockDynamicBandit(DynamicBandit):
    """
    A mock implementation of the abstract DynamicBandit class for testing
    """

    def pull(self, action):
        """Mock implementation of the abstract pull method."""


@pytest.fixture
def sample_arms():
    """
    Provides a sample set of test cases for the bandit.
    """
    return [
        {
            "Name": "Test1",
            "Duration": 10,
            "CalcPrio": 0,
            "LastRun": "2023-01-01 10:00",
            "NumRan": 3,
            "NumErrors": 1,
            "Verdict": 1,
            "LastResults": [1, 0, 0],
        },
        {
            "Name": "Test2",
            "Duration": 20,
            "CalcPrio": 0,
            "LastRun": "2023-01-02 11:00",
            "NumRan": 2,
            "NumErrors": 2,
            "Verdict": 0,
            "LastResults": [0, 0],
        },
        {
            "Name": "Test3",
            "Duration": 15,
            "CalcPrio": 0,
            "LastRun": "2023-01-03 12:00",
            "NumRan": 4,
            "NumErrors": 0,
            "Verdict": 1,
            "LastResults": [1, 1, 1, 0],
        },
    ]


@pytest.fixture
def mock_evaluation_metric():
    """
    Provides a mock evaluation metric.
    """
    mock_metric = MagicMock(spec=EvaluationMetric)
    mock_metric.evaluate = MagicMock()
    mock_metric.__str__.return_value = "MockEvaluationMetric"
    return mock_metric


def test_evaluation_metric_bandit_str(sample_arms, mock_evaluation_metric):
    """
    Test the string representation of EvaluationMetricBandit.
    """
    bandit = EvaluationMetricBandit(sample_arms, mock_evaluation_metric)
    assert str(bandit) == "MockEvaluationMetric"


def test_bandit_initialization(sample_arms):
    """
    Test that the Bandit initializes correctly with the provided arms.
    """
    bandit = MockBandit(sample_arms)
    assert isinstance(bandit.arms, pl.DataFrame)
    assert len(bandit.arms) == len(sample_arms)
    assert set(bandit.arms.columns) == set(bandit.tc_fieldnames)


def test_bandit_reset(sample_arms):
    """
    Test that reset clears all arms.
    """
    bandit = MockBandit(sample_arms)
    bandit.reset()
    assert bandit.arms.height == 0


def test_bandit_add_arms(sample_arms):
    """
    Test that add_arms correctly adds arms to the Bandit.
    """
    bandit = MockBandit([])
    bandit.add_arms(sample_arms)
    assert len(bandit.arms) == len(sample_arms)
    assert set(bandit.arms["Name"]) == {"Test1", "Test2", "Test3"}


def test_bandit_get_arms(sample_arms):
    """
    Test that get_arms returns the correct list of arm names.
    """
    bandit = MockBandit(sample_arms)
    arms = bandit.get_arms()
    assert arms == ["Test1", "Test2", "Test3"]


def test_bandit_update_priority(sample_arms):
    """
    Test that update_priority correctly updates priorities based on the action order.
    """
    bandit = MockBandit(sample_arms)
    action = ["Test3", "Test1", "Test2"]
    bandit.update_priority(action)
    assert bandit.arms.filter(pl.col("Name") == "Test3")["CalcPrio"][0] == 1
    assert bandit.arms.filter(pl.col("Name") == "Test1")["CalcPrio"][0] == 2
    assert bandit.arms.filter(pl.col("Name") == "Test2")["CalcPrio"][0] == 3


def test_bandit_update_priority_empty_action_sets_zero(sample_arms):
    """Empty action list should set CalcPrio=0 for all arms."""
    bandit = MockBandit(sample_arms)
    bandit.update_priority([])
    assert bandit.arms["CalcPrio"].to_list() == [0, 0, 0]


def test_dynamic_bandit_update_arms(sample_arms):
    """
    Test that DynamicBandit updates arms correctly.
    """
    bandit = MockDynamicBandit(sample_arms)
    new_arms = [
        {
            "Name": "Test4",
            "Duration": 30,
            "CalcPrio": 0,
            "LastRun": "2023-01-04 13:00",
            "NumRan": 1,
            "NumErrors": 3,
            "Verdict": 1,
            "LastResults": [1],
        },
    ]
    bandit.update_arms(new_arms)
    assert len(bandit.arms) == 1
    assert bandit.arms["Name"][0] == "Test4"


def test_evaluation_metric_bandit_initialization(sample_arms, mock_evaluation_metric):
    """
    Test that EvaluationMetricBandit initializes with arms and an evaluation metric.
    """
    bandit = EvaluationMetricBandit(sample_arms, mock_evaluation_metric)
    assert bandit.evaluation_metric == mock_evaluation_metric


def test_evaluation_metric_bandit_pull(sample_arms, mock_evaluation_metric):
    """
    Test that pull evaluates the prioritized test suite.
    """
    bandit = EvaluationMetricBandit(sample_arms, mock_evaluation_metric)
    action = ["Test2", "Test1", "Test3"]
    result = bandit.pull(action)
    mock_evaluation_metric.evaluate.assert_called_once()
    assert isinstance(mock_evaluation_metric.evaluate.call_args.args[0], pl.DataFrame)
    assert result == mock_evaluation_metric


def test_evaluation_metric_bandit_sort_order(sample_arms, mock_evaluation_metric):
    """
    Test that pull sorts arms by priority before evaluation.
    """
    bandit = EvaluationMetricBandit(sample_arms, mock_evaluation_metric)
    action = ["Test3", "Test1", "Test2"]
    bandit.pull(action)
    sorted_names = bandit.arms["Name"].to_list()
    assert sorted_names == ["Test3", "Test1", "Test2"]


def test_bandit_allow_duplicate_arms(sample_arms):
    """
    Test that duplicate arms are allowed in the Bandit.
    """
    bandit = MockBandit(sample_arms)
    bandit.add_arms(sample_arms)  # Add the same arms again
    assert len(bandit.arms) == 2 * len(sample_arms), "Duplicate arms should be allowed."


def test_bandit_update_priority_with_duplicates():
    """
    Test that update_priority correctly updates priorities for duplicate arms.
    """
    sample_arms = [
        {
            "Name": "Test1",
            "Duration": 10,
            "CalcPrio": 0,
            "LastRun": "2023-01-01 10:00",
            "NumRan": 3,
            "NumErrors": 1,
            "Verdict": 1,
            "LastResults": [1, 0, 0],
        },
        {
            "Name": "Test1",
            "Duration": 20,
            "CalcPrio": 0,
            "LastRun": "2023-01-01 11:00",
            "NumRan": 2,
            "NumErrors": 2,
            "Verdict": 0,
            "LastResults": [0, 0],
        },
    ]
    bandit = MockBandit(sample_arms)
    action = ["Test1"]
    bandit.update_priority(action)
    assert bandit.arms.filter(pl.col("Name") == "Test1")["CalcPrio"][0] > 0, (
        "All duplicates of 'Test1' should have updated priorities."
    )


def test_dynamic_bandit_reset_after_update(sample_arms):
    """
    Test that DynamicBandit resets arms correctly after an update.
    """
    bandit = MockDynamicBandit(sample_arms)
    bandit.update_arms([])
    assert bandit.arms.height == 0


def test_evaluation_metric_bandit_with_empty_action(sample_arms, mock_evaluation_metric):
    """
    Test that EvaluationMetricBandit handles an empty action gracefully.
    """
    bandit = EvaluationMetricBandit(sample_arms, mock_evaluation_metric)
    with pytest.raises(ValueError, match="Action list cannot be empty"):
        bandit.pull([])


def test_bandit_abstract_method():
    """
    Test that the Bandit class cannot be instantiated due to its abstract pull method.
    """
    with pytest.raises(TypeError) as excinfo:
        Bandit([])  # pylint: disable=abstract-class-instantiated
    exception_message = str(excinfo.value)
    assert "Can't instantiate abstract class Bandit" in exception_message
    # Python 3.12+ uses "abstract method pull" while older versions use "abstract method 'pull'"
    assert "abstract method pull" in exception_message or "abstract method 'pull'" in exception_message


def test_bandit_subclass_without_pull():
    """
    Test that a subclass of Bandit without implementing the pull method cannot be instantiated.
    """

    class IncompleteBandit(Bandit):  # pylint: disable=abstract-class-instantiated
        """Incomplete bandit subclass without pull method for testing."""

    with pytest.raises(TypeError) as excinfo:
        IncompleteBandit([])  # pylint: disable=abstract-class-instantiated

    exception_message = str(excinfo.value)
    assert "Can't instantiate abstract class IncompleteBandit" in exception_message
    # Python 3.12+ uses "abstract method pull" while older versions use "abstract method 'pull'"
    assert "abstract method pull" in exception_message or "abstract method 'pull'" in exception_message


def test_bandit_subclass_with_pull():
    """
    Test that a subclass of Bandit with the pull method implemented can be instantiated.
    """

    class CompleteBandit(Bandit):
        """Complete bandit subclass with pull method for testing."""

        def pull(self, action):
            return action

    bandit = CompleteBandit([])
    assert bandit.pull(["Test1", "Test2"]) == ["Test1", "Test2"]


@pytest.mark.benchmark(group="bandit")
def test_bandit_add_arms_performance(benchmark, sample_arms):
    """
    Benchmark adding a large number of arms to a Bandit.
    """
    large_arms = sample_arms * 1000  # Create a large dataset
    bandit = MockBandit([])

    def add_arms():
        bandit.add_arms(large_arms)

    benchmark(add_arms)


@pytest.mark.benchmark(group="bandit")
def test_bandit_update_priority_performance(benchmark, sample_arms):
    """
    Benchmark updating priorities for a large set of arms.
    """
    large_arms = sample_arms * 1000  # Create a large dataset
    bandit = MockBandit(large_arms)
    action = [arm["Name"] for arm in large_arms]  # Generate an action list with all arm names

    def update_priority():
        bandit.update_priority(action)

    benchmark(update_priority)


@pytest.mark.benchmark(group="bandit")
def test_evaluation_metric_bandit_pull_performance(benchmark, sample_arms, mock_evaluation_metric):
    """
    Benchmark the pull operation in EvaluationMetricBandit for large action sets.
    """
    large_arms = sample_arms * 1000  # Create a large dataset
    bandit = EvaluationMetricBandit(large_arms, mock_evaluation_metric)
    action = [arm["Name"] for arm in large_arms]  # Generate an action list with all arm names

    def pull_action():
        bandit.pull(action)

    benchmark(pull_action)
