"""Reciprocal-rank based reward function.

References
----------
- Cormack, G. V.; Clarke, C. L. A.; Buettcher, S. (2009). Reciprocal rank
    fusion outperforms condorcet and individual rank learning methods. SIGIR.
- Manning, C. D.; Raghavan, P.; Schutze, H. (2008). Introduction to
    Information Retrieval. Cambridge University Press.
"""

from coleman.evaluation import EvaluationMetric

from .base import Reward


class ReciprocalRankReward(Reward):
    """Reciprocal-Rank reward.

    Rewards failing tests by the inverse of their rank, following a classic
    information-retrieval signal that strongly favors earlier detections.
    """

    def __str__(self):
        """Return a string representation of the reward function."""
        return "Reciprocal-rank Reward"

    def get_name(self):
        """Return the identifier of the reward function."""
        return "ReciprocalRank"

    def evaluate(self, reward: EvaluationMetric, last_prioritization: list[str]):
        """Evaluate rewards based on reciprocal failing ranks.

        Parameters
        ----------
        reward : EvaluationMetric
            Evaluation metric containing detection ranks.
        last_prioritization : list of str
            Test case names in prioritization order.

        Returns
        -------
        list of float
            Reciprocal-rank reward values aligned with ``last_prioritization``.
        """
        if not reward.detection_ranks:
            return [0.0] * len(last_prioritization)

        reciprocal_by_rank = {rank: 1.0 / rank for rank in reward.detection_ranks if rank > 0}
        return [reciprocal_by_rank.get(i + 1, 0.0) for i in range(len(last_prioritization))]
