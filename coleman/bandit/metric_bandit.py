"""EvaluationMetricBandit - a dynamic bandit backed by an evaluation metric."""

from coleman.evaluation import EvaluationMetric

from .dynamic import DynamicBandit


class EvaluationMetricBandit(DynamicBandit):
    """A Dynamic Bandit that provides feedback based on an evaluation metric.

    Parameters
    ----------
    arms : list of dict
        The arms of the bandit (test case records).
    evaluation_metric : EvaluationMetric
        The evaluation metric used to provide feedback.

    Attributes
    ----------
    evaluation_metric : EvaluationMetric
        The evaluation metric instance.
    """

    def __init__(self, arms: list[dict], evaluation_metric: EvaluationMetric):
        """Initialize the EvaluationMetricBandit.

        Parameters
        ----------
        arms : list of dict
            The arms of the bandit (test case records).
        evaluation_metric : EvaluationMetric
            The evaluation metric used to provide feedback.
        """
        super().__init__(arms)
        self.evaluation_metric = evaluation_metric

    def __str__(self):
        """Return a string representation of the bandit.

        Returns
        -------
        str
            String representation of the evaluation metric.
        """
        return str(self.evaluation_metric)

    def pull(self, action):
        """Submit prioritized test set for evaluation and get new measurements.

        Parameters
        ----------
        action : list of str
            The prioritized test suite list.

        Returns
        -------
        EvaluationMetric
            The result ("state") of an evaluation by the evaluation metric.

        Raises
        ------
        ValueError
            If the action list is empty.
        """
        if not action:
            raise ValueError("Action list cannot be empty")

        super().update_priority(action)

        self.arms = self.arms.sort("CalcPrio", descending=False)

        self.evaluation_metric.evaluate(self.arms)

        return self.evaluation_metric
