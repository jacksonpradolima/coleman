"""APFDc (cost-aware) evaluation metric."""

from .base import EvaluationMetric


class APFDcMetric(EvaluationMetric):
    """APFDc (Average Percentage of Faults Detected cost-aware) Metric.

    Extends NAPFD with explicit exposure of cost-aware fault detection scoring.
    """

    def __init__(self):
        """Initialize the APFDcMetric."""
        super().__init__()
        self.testcase_costs = []

    def __str__(self):
        """Return a string representation of the metric.

        Returns
        -------
        str
            The metric name.
        """
        return "APFDc"

    def evaluate(self, test_suite):
        """Evaluate the test suite using the APFDc metric.

        Parameters
        ----------
        test_suite : list of dict
            Test suite to evaluate.
        """
        self.reset()
        self.testcase_costs = []

        costs, total_failure_count, total_failed_tests = self.process_test_suite(test_suite, "NumErrors")
        self.testcase_costs = [item["Duration"] for item in test_suite]

        if total_failure_count > 0:
            self.compute_metrics(costs, total_failure_count, total_failed_tests, len(test_suite))
        else:
            self.set_default_metrics()

    def compute_metrics(self, costs, total_failure_count, total_failed_tests, no_testcases):
        """Compute APFDc metric (cost-aware faults detected).

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
        self.recall = sum(self.detection_ranks_failures) / total_failure_count if total_failure_count > 0 else 1.0

        # APFDc: cost-weighted fault detection
        if self.detection_ranks and sum(costs) > 0 and total_failed_tests > 0:
            self.cost = sum(sum(costs[i - 1 :]) - 0.5 * costs[i - 1] for i in self.detection_ranks) / (
                sum(costs) * total_failed_tests
            )
        else:
            self.cost = 0.0

        self.fitness = self.cost
        self.avg_precision = self.cost
