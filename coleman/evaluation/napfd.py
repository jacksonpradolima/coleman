"""NAPFD evaluation metrics."""

import numpy as np

from .base import EvaluationMetric


class NAPFDMetric(EvaluationMetric):
    """Normalized Average Percentage of Faults Detected (NAPFD) Metric.

    Based on error counts.
    """

    def __str__(self):
        """Return a string representation of the metric.

        Returns
        -------
        str
            The metric name.
        """
        return "NAPFD"

    def evaluate(self, test_suite):
        """Evaluate the test suite using the NAPFD metric.

        Parameters
        ----------
        test_suite : list of dict
            Test suite to evaluate.
        """
        self.reset()
        costs, total_failure_count, total_failed_tests = self.process_test_suite(test_suite, "NumErrors")

        if total_failure_count > 0:
            self.compute_metrics(costs, total_failure_count, total_failed_tests, len(test_suite))
        else:
            self.set_default_metrics()

    def compute_metrics(self, costs, total_failure_count, total_failed_tests, no_testcases):
        """Compute NAPFD and APFDc metrics.

        Parameters
        ----------
        costs : list
            A list containing the costs (e.g., execution time) for each test case.
        total_failure_count : int
            Total number of failures detected across all test cases.
        total_failed_tests : int
            Total number of test cases that failed.
        no_testcases : int
            Total number of test cases in the test suite.

        Notes
        -----
        This method updates the instance's attributes directly and does not
        return any value.
        """
        self.ttf = self.detection_ranks[0] if self.detection_ranks else 0
        self.recall = sum(self.detection_ranks_failures) / total_failure_count
        self.avg_precision = _avg_precision_from_detection_ranks(self.detection_ranks)

        p = self.recall if self.undetected_failures > 0 else 1

        # NAPFD
        self.fitness = p - self.detected_failures / (total_failure_count * no_testcases) + p / (2 * no_testcases)

        # APFDc
        self.cost = sum(sum(costs[i - 1 :]) - 0.5 * costs[i - 1] for i in self.detection_ranks) / (
            sum(costs) * total_failed_tests
        )


class NAPFDVerdictMetric(EvaluationMetric):
    """Normalized Average Percentage of Faults Detected (NAPFD) Metric based on Verdict."""

    def __str__(self):
        """Return a string representation of the metric.

        Returns
        -------
        str
            The metric name.
        """
        return "NAPFDVerdict"

    def evaluate(self, test_suite):
        """Evaluate the test suite using the NAPFD Verdict metric.

        Parameters
        ----------
        test_suite : list of dict
            Test suite to evaluate.
        """
        self.reset()
        costs, total_failure_count, _ = self.process_test_suite(test_suite, "Verdict")

        assert self.undetected_failures + self.detected_failures == total_failure_count

        if total_failure_count > 0:
            self.compute_metrics(costs, total_failure_count, len(test_suite))
        else:
            self.set_default_metrics()

    def compute_metrics(self, costs, total_failure_count, no_testcases):
        """Compute NAPFD and APFDc metrics based on test verdicts.

        Parameters
        ----------
        costs : list
            A list containing the costs (e.g., execution time) for each test case.
        total_failure_count : int
            Total number of test cases that failed.
        no_testcases : int
            Total number of test cases in the test suite.

        Notes
        -----
        This method updates the instance's attributes directly and does not
        return any value.
        """
        self.ttf = self.detection_ranks[0] if self.detection_ranks else 0
        self.recall = self.detected_failures / total_failure_count
        self.avg_precision = _avg_precision_from_detection_ranks(self.detection_ranks)

        p = self.recall if self.undetected_failures > 0 else 1

        # NAPFD
        self.fitness = p - sum(self.detection_ranks) / (total_failure_count * no_testcases) + p / (2 * no_testcases)

        # APFDc
        self.cost = sum(sum(costs[i - 1 :]) - 0.5 * costs[i - 1] for i in self.detection_ranks) / (
            sum(costs) * total_failure_count
        )


def _avg_precision_from_detection_ranks(detection_ranks: list[int]) -> float:
    """Compute average precision from 1-based failure detection ranks."""
    if not detection_ranks:
        return 0.0

    ranks = np.asarray(detection_ranks, dtype=np.float64)
    positives = np.arange(1, ranks.size + 1, dtype=np.float64)
    return float(np.mean(positives / ranks))
