"""Base evaluation metric class."""

import numpy as np
import polars as pl


class EvaluationMetric:
    """Base class for evaluation metrics.

    Attributes
    ----------
    available_time : float
        The time available for test execution.
    scheduled_testcases : list
        Test cases that were scheduled for execution.
    unscheduled_testcases : list
        Test cases that were not scheduled.
    detection_ranks : list
        Ranks at which failures were detected.
    detection_ranks_time : list
        Durations of failure-detecting test cases.
    detection_ranks_failures : list
        Failure counts at each detection rank.
    ttf : int
        Time to Fail (rank value).
    ttf_duration : float
        Time spent until the first test case fail.
    fitness : float
        APFD or NAPFD value.
    cost : float
        APFDc value.
    detected_failures : int
        Number of detected failures.
    undetected_failures : int
        Number of undetected failures.
    recall : float
        Recall metric value.
    avg_precision : float
        Average precision metric value.
    """

    def __init__(self):
        """Initialize the EvaluationMetric."""
        self.available_time = 0
        self.reset()

    def update_available_time(self, available_time: float):
        """Update the available time for the metric.

        Parameters
        ----------
        available_time : float
            Time available for the metric.
        """
        self.available_time = available_time

    def reset(self):
        """Reset all attributes to their default values."""
        self.scheduled_testcases = []
        self.unscheduled_testcases = []
        self.detection_ranks = []
        self.detection_ranks_time = []
        self.detection_ranks_failures = []
        self.ttf = self.ttf_duration = 0  # Time to Fail (rank value)
        self.fitness = 0  # APFD or NAPFD value
        self.cost = 0  # APFDc (to compute at same time, for instance, with NAPFD)
        self.detected_failures = self.undetected_failures = 0
        self.recall = self.avg_precision = 0

    def _as_suite_frame(self, test_suite, error_key: str) -> pl.DataFrame:
        """Normalize test-suite input to a Polars DataFrame.

        Supports both list-of-dicts and ``polars.DataFrame`` inputs so metric
        implementations can use a single vectorized code path.
        """
        suite_df = pl.DataFrame(test_suite)

        if suite_df.is_empty() and suite_df.width == 0:
            return pl.DataFrame({"Name": [], "Duration": [], "Verdict": [], error_key: []})

        required_columns = {"Name", "Duration", "Verdict", error_key}
        missing = required_columns.difference(set(suite_df.columns))
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise KeyError(f"Missing required test-suite columns: {missing_str}")

        return suite_df

    def process_test_suite(self, test_suite, error_key):
        """Process the test suite and return the costs and total failure count.

        Parameters
        ----------
        test_suite : list of dict
            Test suite to process.
        error_key : str
            Key to determine the error in the test suite.

        Returns
        -------
        costs : list
            List of durations for each test case.
        total_failure_count : int
            Total number of failures detected.
        total_failed_tests : int
            Total number of test cases that failed.
        """
        suite_df = self._as_suite_frame(test_suite, error_key)
        if suite_df.is_empty():
            return [], 0, 0

        names = suite_df["Name"].to_list()
        durations = np.asarray(suite_df["Duration"].to_numpy(), dtype=np.float64)
        errors = np.asarray(suite_df[error_key].to_numpy(), dtype=np.float64)
        verdicts = np.asarray(suite_df["Verdict"].to_numpy(), dtype=np.float64)

        cum_duration = np.cumsum(durations)
        scheduled_mask = cum_duration <= float(self.available_time)
        unscheduled_mask = ~scheduled_mask
        scheduled_rank = np.cumsum(scheduled_mask.astype(np.int64))
        detected_mask = scheduled_mask & (errors > 0)

        scheduled_idx = np.flatnonzero(scheduled_mask)
        unscheduled_idx = np.flatnonzero(unscheduled_mask)
        detected_idx = np.flatnonzero(detected_mask)

        self.scheduled_testcases = [names[i] for i in scheduled_idx]
        self.unscheduled_testcases = [names[i] for i in unscheduled_idx]
        self.detection_ranks = scheduled_rank[detected_idx].astype(np.int64).tolist()
        self.detection_ranks_failures = errors[detected_idx].tolist()
        self.detection_ranks_time = durations[detected_idx].tolist()

        if detected_idx.size > 0:
            self.ttf_duration = float(cum_duration[detected_idx[0]])
        else:
            self.ttf_duration = float(cum_duration[-1])

        total_failure_count = int(errors.sum())
        total_failed_tests = int(verdicts.sum())
        self.undetected_failures = int(errors[unscheduled_mask].sum())

        if error_key == "Verdict":
            self.detected_failures = int(detected_idx.size)
        else:
            self.detected_failures = int((errors[detected_mask] * scheduled_rank[detected_mask]).sum())

        costs = durations.tolist()
        return costs, total_failure_count, total_failed_tests

    def evaluate(self, test_suite):
        """Evaluate the test suite.

        This is an abstract method and must be implemented in child classes.

        Parameters
        ----------
        test_suite : list of dict
            Test suite to evaluate.

        Raises
        ------
        NotImplementedError
            If not implemented in a child class.
        """
        raise NotImplementedError("This method must be overridden in child classes")

    def set_default_metrics(self):
        """Set the default values for NAPFD and APFDc metrics.

        This method is called when there are no detected failures in the test
        suite, ensuring that the metric attributes are appropriately
        initialized.

        Notes
        -----
        This method updates the instance's attributes directly and does not
        return any value.
        """
        self.ttf = -1
        self.recall = self.avg_precision = 1
        self.fitness = self.cost = 1
