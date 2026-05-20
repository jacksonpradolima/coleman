"""Abstract base class for reward functions."""

from abc import ABC, abstractmethod

from coleman.evaluation import EvaluationMetric


class Reward(ABC):
    """Abstract base class for reward functions.

    A reward function is used by the agent in the observe method to evaluate
    bandit results and return a reward.
    """

    def get_name(self):
        """Retrieve the name or identifier of the reward function.

        Returns
        -------
        str
            The name or identifier of the reward function.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, reward: EvaluationMetric, last_prioritization: list[str]):
        """Evaluate a bandit result and return a reward.

        Parameters
        ----------
        reward : EvaluationMetric
            The evaluation metric result.
        last_prioritization : list of str
            The last prioritized test suite list.

        Returns
        -------
        list of float
            The computed rewards for each test case.
        """
        raise NotImplementedError
