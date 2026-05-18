"""Top-k binary failure reward function.

This is a prefix-constrained binary signal equivalent to precision@k in a
single suite when interpreted as failure relevance.

References
----------
- Manning, C. D.; Raghavan, P.; Schutze, H. (2008). Introduction to
    Information Retrieval. Cambridge University Press.
"""

from coleman.evaluation import EvaluationMetric

from .base import Reward


class TopKRNFailReward(Reward):
    """Top-k binary failure reward.

    Considers only whether a test failed (binary signal) inside the first
    ``top_k`` positions. Each failing test within top-k receives ``1 / k_eff``,
    where ``k_eff = min(top_k, len(last_prioritization))``. The reward sum over
    the selected prefix is therefore the failure percentage in top-k.

    When ``use_time_budget`` is enabled, the prefix is also capped by the
    number of tests scheduled by the metric under the active time budget.
    """

    def __init__(self, top_k: int = 6, use_time_budget: bool = False):
        """Initialize the reward with a top-k cutoff."""
        if top_k <= 0:
            msg = "top_k must be a positive integer"
            raise ValueError(msg)
        self.top_k = top_k
        self.use_time_budget = use_time_budget

    def __str__(self):
        """Return a string representation of the reward function."""
        return f"Top-k RNFail Reward (k={self.top_k})"

    def get_name(self):
        """Return the identifier of the reward function."""
        return "TopKRNFail"

    def evaluate(self, reward: EvaluationMetric, last_prioritization: list[str]):
        """Evaluate top-k binary rewards.

        Parameters
        ----------
        reward : EvaluationMetric
            Evaluation metric containing detection ranks.
        last_prioritization : list of str
            Test case names in prioritization order.

        Returns
        -------
        list of float
            Reward vector aligned to ``last_prioritization``.
        """
        n = len(last_prioritization)
        if n == 0:
            return []

        k_eff = min(self.top_k, n)
        if self.use_time_budget:
            scheduled_count = len(getattr(reward, "scheduled_testcases", []))
            k_eff = min(k_eff, scheduled_count)

        if k_eff <= 0:
            return [0.0] * n

        failing_indices = set(reward.detection_ranks)
        reward_per_failure = 1.0 / k_eff

        values = [0.0] * n
        for idx in range(k_eff):
            if idx + 1 in failing_indices:
                values[idx] = reward_per_failure
        return values
