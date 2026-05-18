"""APFDc-based reward function.

This reward turns the classic cost-aware APFDc score into a per-test reward
signal, assigning each failing test its contribution to the total APFDc value.

References
----------
- Elbaum, S.; Malishevsky, A. G.; Rothermel, G. (2002). Test case prioritization:
  a family of empirical studies. IEEE TSE.
- Rothermel, G.; Untch, R. H.; Chu, C.; Harrold, M. J. (2001). Prioritizing test
  cases for regression testing. IEEE TSE.
"""

from coleman.evaluation import EvaluationMetric

from .base import Reward


class APFDcReward(Reward):
    """Reward failing tests by their APFDc contribution.

    The reward for each failing test at rank ``r`` is its APFDc contribution,
    normalized by total execution cost and the total number of failing tests.
    Non-failing tests receive ``0``.
    """

    def __str__(self):
        """Return a string representation of the reward function."""
        return "APFDc Reward"

    def get_name(self):
        """Return the identifier of the reward function."""
        return "APFDc"

    def evaluate(self, reward: EvaluationMetric, last_prioritization: list[str]):
        """Evaluate APFDc-style rewards using stored test execution costs.

        Parameters
        ----------
        reward : EvaluationMetric
            Evaluation metric containing detection ranks and testcase costs.
        last_prioritization : list of str
            Test case names in prioritization order.

        Returns
        -------
        list of float
            Reward values aligned with ``last_prioritization``.
        """
        n = len(last_prioritization)
        if n == 0:
            return []

        costs = getattr(reward, "testcase_costs", None)
        if not costs or not reward.detection_ranks:
            return [0.0] * n

        total_cost = float(sum(costs))
        total_failures = int(getattr(reward, "detected_failures", 0) + getattr(reward, "undetected_failures", 0))
        if total_cost <= 0 or total_failures <= 0:
            return [0.0] * n

        suffix_costs = [0.0] * n
        running = 0.0
        for idx in range(n - 1, -1, -1):
            running += float(costs[idx])
            suffix_costs[idx] = running

        values = [0.0] * n
        normalization = total_cost * total_failures
        for rank in reward.detection_ranks:
            idx = rank - 1
            if 0 <= idx < n:
                contribution = suffix_costs[idx] - 0.5 * float(costs[idx])
                values[idx] = contribution / normalization
        return values
