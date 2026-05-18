"""Top-k evaluation metrics for binary fault detection.

Implemented metrics are standard in information retrieval and frequently
adopted in test case prioritization studies.

References
----------
- Manning, C. D.; Raghavan, P.; Schutze, H. (2008). Introduction to
    Information Retrieval. Cambridge University Press.
- Baeza-Yates, R.; Ribeiro-Neto, B. (2011). Modern Information Retrieval:
    The Concepts and Technology behind Search (2nd ed.). Addison-Wesley.
"""

import numpy as np

from .base import EvaluationMetric


class _TopKVerdictMetric(EvaluationMetric):
    """Shared top-k logic for verdict-based metrics."""

    metric_name = "TopK"

    def __init__(self, top_k: int = 6, use_time_budget: bool = False):
        """Initialize a top-k metric.

        Parameters
        ----------
        top_k : int
            Prefix size considered by the metric.
        use_time_budget : bool
            If True, only tests whose cumulative duration is within
            ``available_time`` are eligible in the top-k prefix.
        """
        if top_k <= 0:
            msg = "top_k must be a positive integer"
            raise ValueError(msg)
        super().__init__()
        self.top_k = top_k
        self.use_time_budget = use_time_budget

    def __str__(self):
        """Return a string representation of the metric."""
        return self.metric_name

    def _compute_fitness(self, detected_failures: int, selected_count: int, total_failures: int) -> float:
        """Compute the metric-specific fitness score."""
        raise NotImplementedError

    def evaluate(self, test_suite):
        """Evaluate a test suite using the first eligible ``top_k`` test cases."""
        self.reset()
        suite_df = self._as_suite_frame(test_suite, "Verdict")

        if suite_df.is_empty():
            self.set_default_metrics()
            return

        suite_durations = np.asarray(suite_df["Duration"].to_numpy(), dtype=np.float64)
        if self.use_time_budget:
            cum_duration = np.cumsum(suite_durations)
            budget_count = int(np.count_nonzero(cum_duration <= float(self.available_time)))
            selected_count = min(self.top_k, budget_count)
        else:
            selected_count = min(self.top_k, suite_df.height)

        top_df = suite_df.head(selected_count)

        names = top_df["Name"].to_list()
        durations = np.asarray(top_df["Duration"].to_numpy(), dtype=np.float64)
        verdicts = np.asarray(top_df["Verdict"].to_numpy(), dtype=np.float64)
        all_verdicts = np.asarray(suite_df["Verdict"].to_numpy(), dtype=np.float64)

        failing_idx = np.flatnonzero(verdicts > 0)

        self.scheduled_testcases = names
        self.unscheduled_testcases = suite_df["Name"].slice(selected_count).to_list()
        self.detection_ranks = (failing_idx + 1).astype(np.int64).tolist()
        self.detection_ranks_failures = verdicts[failing_idx].tolist()
        self.detection_ranks_time = durations[failing_idx].tolist()

        self.detected_failures = int(failing_idx.size)
        total_failures = int(all_verdicts.sum())
        self.undetected_failures = max(total_failures - self.detected_failures, 0)

        if failing_idx.size > 0:
            self.ttf = int(failing_idx[0] + 1)
            self.ttf_duration = float(np.cumsum(durations)[failing_idx[0]])
        else:
            self.ttf = -1
            self.ttf_duration = float(durations.sum())

        if total_failures == 0:
            self.recall = 1.0
        else:
            self.recall = self.detected_failures / total_failures

        self.fitness = self._compute_fitness(self.detected_failures, selected_count, total_failures)
        self.avg_precision = self.fitness
        self.cost = self.fitness


class PrecisionAtKMetric(_TopKVerdictMetric):
    """Precision@k for test-failure detection.

    Defined as the fraction of failing tests among the first ``k`` selected
    tests. Example: with ``k=6`` and 6 failures in the first 6 positions,
    Precision@k is ``1.0`` (100%).
    """

    metric_name = "PrecisionAtK"

    def _compute_fitness(self, detected_failures: int, selected_count: int, total_failures: int) -> float:
        """Compute precision@k."""
        del total_failures
        if selected_count == 0:
            return 0.0
        return detected_failures / selected_count


class RecallAtKMetric(_TopKVerdictMetric):
    """Recall@k for test-failure detection.

    Defined as the fraction of all failing tests that were found within the
    first ``k`` selected tests.
    """

    metric_name = "RecallAtK"

    def _compute_fitness(self, detected_failures: int, selected_count: int, total_failures: int) -> float:
        """Compute recall@k."""
        del selected_count
        if total_failures == 0:
            return 1.0
        return detected_failures / total_failures


class AveragePrecisionAtKMetric(_TopKVerdictMetric):
    r"""Average Precision at k (AP@k) for failure detection.

    AP@k averages precision values observed at each failing rank up to ``k``:

    .. math:: AP@k = \frac{1}{\\min(F, k)} \\sum_{i=1}^{k} P(i) \\cdot rel(i)

    where ``rel(i)`` is 1 when rank ``i`` is failing, 0 otherwise.
    """

    metric_name = "AveragePrecisionAtK"

    def _compute_fitness(self, detected_failures: int, selected_count: int, total_failures: int) -> float:
        """Compute AP@k."""
        del detected_failures

        if selected_count == 0:
            return 0.0

        if total_failures == 0:
            return 1.0

        if not self.detection_ranks:
            return 0.0

        ap_sum = 0.0
        for rank in self.detection_ranks:
            precision_at_rank = sum(1 for r in self.detection_ranks if r <= rank) / rank
            ap_sum += precision_at_rank

        normalizer = min(total_failures, selected_count)
        return ap_sum / normalizer if normalizer > 0 else 0.0


class ReciprocalRankAtKMetric(_TopKVerdictMetric):
    r"""Reciprocal Rank at k (RR@k).

    Returns the reciprocal of the first failing rank within the top-k prefix:

    .. math:: RR@k = \begin{cases}
        1/r_1, & \text{if a failure appears in top-k at rank } r_1 \\
        0, & \text{otherwise}
    \end{cases}

    For one prioritized suite, RR@k is equivalent to MRR@k.
    """

    metric_name = "ReciprocalRankAtK"

    def _compute_fitness(self, detected_failures: int, selected_count: int, total_failures: int) -> float:
        """Compute reciprocal rank at k."""
        del detected_failures, selected_count, total_failures
        if not self.detection_ranks:
            return 0.0
        first_rank = min(self.detection_ranks)
        return 1.0 / first_rank


class NDCGAtKMetric(_TopKVerdictMetric):
    r"""Normalized Discounted Cumulative Gain at k (nDCG@k).

    With binary relevance (fail = 1, pass = 0), nDCG@k is:

    .. math:: nDCG@k = \frac{\sum_{i=1}^{k} rel(i)/\log_2(i+1)}{\sum_{i=1}^{m} 1/\log_2(i+1)}

    where ``m = min(F, k)`` and ``F`` is the total number of failing tests.
    """

    metric_name = "NDCGAtK"

    def _compute_fitness(self, detected_failures: int, selected_count: int, total_failures: int) -> float:
        """Compute nDCG@k with binary relevance."""
        del detected_failures

        if selected_count == 0:
            return 0.0

        if total_failures == 0:
            return 1.0

        if not self.detection_ranks:
            return 0.0

        dcg = sum(1.0 / np.log2(rank + 1.0) for rank in self.detection_ranks)

        ideal_hits = min(total_failures, selected_count)
        idcg = sum(1.0 / np.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))

        if idcg == 0.0:
            return 0.0
        return dcg / idcg
