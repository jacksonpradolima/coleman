"""Reward based on failures (RNFail)."""

from coleman.evaluation import EvaluationMetric

from .base import Reward


class RNFailReward(Reward):
    """Reward Based on Failures (RNFail).

    This reward function is based on the number of failures associated with
    test cases t' in T': 1 if t' failed; 0 otherwise.
    """

    def __str__(self):
        """Return a string representation of the reward function.

        Returns
        -------
        str
            The reward function name.
        """
        return "Reward Based on Failures"

    def get_name(self):
        """Return the identifier of the reward function.

        Returns
        -------
        str
            The reward function identifier.
        """
        return "RNFail"

    def evaluate(self, reward: EvaluationMetric, last_prioritization: list[str]):
        """Evaluate rewards based on failures.

        Parameters
        ----------
        reward : EvaluationMetric
            Evaluation metric containing detection ranks and scheduled test cases.
        last_prioritization : list of str
            Test case names in prioritization order.

        Returns
        -------
        list of float
            List of rewards for each test case in the prioritization.
        """
        if not reward.detection_ranks:
            return [0.0] * len(last_prioritization)

        failing_indices = set(reward.detection_ranks)

        rewards = [1.0 if i + 1 in failing_indices else 0.0 for i, tc in enumerate(last_prioritization)]
        return rewards
