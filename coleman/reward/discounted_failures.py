"""Discounted rank reward for failures.

This reward follows the same intuition as Discounted Cumulative Gain (DCG):
earlier relevant items receive higher gain with logarithmic discount.

References
----------
- Jarvelin, K.; Kekalainen, J. (2002). Cumulated gain-based evaluation of IR
  techniques. ACM Transactions on Information Systems, 20(4), 422-446.
"""

import math

from coleman.evaluation import EvaluationMetric

from .base import Reward


class DiscountedFailureReward(Reward):
    r"""Reward failures with logarithmic discount by rank.

    For each failing test at rank ``r`` (1-indexed), the reward is:

    .. math:: gain(r) = 1 / \log_2(r + 1)

    Non-failing tests receive ``0``.
    """

    def __str__(self):
        """Return a string representation of the reward function."""
        return "Discounted Failure Reward"

    def get_name(self):
        """Return the identifier of the reward function."""
        return "DiscountedFailure"

    def evaluate(self, reward: EvaluationMetric, last_prioritization: list[str]):
        """Evaluate discounted rewards for failing positions.

        Parameters
        ----------
        reward : EvaluationMetric
            Evaluation metric containing detection ranks.
        last_prioritization : list of str
            Test case names in prioritization order.

        Returns
        -------
        list of float
            Discounted reward values aligned with ``last_prioritization``.
        """
        n = len(last_prioritization)
        if n == 0 or not reward.detection_ranks:
            return [0.0] * n

        rank_to_gain = {rank: 1.0 / math.log2(rank + 1.0) for rank in reward.detection_ranks if rank > 0}
        return [rank_to_gain.get(i + 1, 0.0) for i in range(n)]
