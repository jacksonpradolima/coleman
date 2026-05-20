"""Time-rank based reward function."""

import numpy as np

from coleman.evaluation import EvaluationMetric

from .base import Reward


class TimeRankReward(Reward):
    """Time-ranked Reward (TimeRank).

    This reward function explicitly includes the order of test cases and rewards
    each test case based on its rank in the test schedule and whether it failed.
    As a good schedule executes failing test cases early, every passed test case
    reduces the schedule's quality if it precedes a failing test case. Each test
    case is rewarded by the total number of failed test cases; for failed test
    cases it is the same as reward function 'RNFailReward'. For passed test
    cases, the reward is further decreased by the number of failed test cases
    ranked after the passed test case to penalize scheduling passing test cases
    early.
    """

    def __str__(self):
        """Return a string representation of the reward function.

        Returns
        -------
        str
            The reward function name.
        """
        return "Time-ranked Reward"

    def get_name(self):
        """Return the identifier of the reward function.

        Returns
        -------
        str
            The reward function identifier.
        """
        return "timerank"

    def evaluate(self, reward: EvaluationMetric, last_prioritization: list):
        """Evaluate rewards based on the prioritization rank of test cases.

        Parameters
        ----------
        reward : EvaluationMetric
            The evaluation metric containing detection ranks and scheduled test cases.
        last_prioritization : list of str
            The list of test case names in the prioritization order.

        Returns
        -------
        list of float
            A list of rewards for each test case in the prioritization.
        """
        num_failing_tests = len(reward.detection_ranks)
        if num_failing_tests == 0:
            return [0.0] * len(last_prioritization)

        failing_indices = np.array(reward.detection_ranks) - 1

        scheduled_testcases = reward.scheduled_testcases

        rewards = np.zeros(len(scheduled_testcases))
        rewards[failing_indices] = 1
        rewards = np.cumsum(rewards)
        rewards[failing_indices] = num_failing_tests

        normalized_reward_by_testcase = dict(
            zip(scheduled_testcases, (rewards / num_failing_tests).tolist(), strict=False)
        )
        normalized_rewards = [normalized_reward_by_testcase.get(test_case, 0.0) for test_case in last_prioritization]

        return normalized_rewards
