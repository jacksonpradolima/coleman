"""Epsilon-greedy policy variants."""

import polars as pl

from coleman.agent import Agent

from .. import base as _policy_base
from ..base import Policy


class EpsilonGreedyPolicy(Policy):
    """Epsilon-Greedy policy for action selection.

    Chooses a random action with probability epsilon and takes the best
    apparent approach with probability 1-epsilon. If multiple actions are tied
    for best choice, then a random action from that subset is selected.

    Parameters
    ----------
    epsilon : float
        Probability of choosing a random action.

    Attributes
    ----------
    epsilon : float
        Probability of choosing a random action.
    """

    def __init__(self, epsilon):
        """Initialize the EpsilonGreedyPolicy.

        Parameters
        ----------
        epsilon : float
            Probability of choosing a random action.
        """
        self.epsilon = epsilon

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name with epsilon value.
        """
        return f"\u03b5-greedy (\u03b5={self.epsilon})"

    def choose_all(self, agent: Agent):
        """Choose all actions using the epsilon-greedy strategy.

        Each action is independently flagged for random exploration with
        probability *epsilon*. Exploration actions are placed first in a
        random order; the remaining actions are sorted by Q value descending
        (exploitation).

        Parameters
        ----------
        agent : Agent
            The agent whose actions are to be prioritized.

        Returns
        -------
        list of str
            List of action names ordered by the epsilon-greedy strategy.
        """
        actions = agent.actions.clone()
        n = len(actions)
        rand_vals = _policy_base._rng.random(n)
        actions = actions.with_columns([pl.Series("rand_val", rand_vals)])
        actions = actions.with_columns(
            [
                (pl.col("rand_val") < self.epsilon).alias("is_random"),
                pl.when(pl.col("rand_val") < self.epsilon)
                .then(pl.col("rand_val"))
                .otherwise(pl.col("Q"))
                .alias("sort_key"),
            ]
        )
        actions = actions.sort(["is_random", "sort_key"], descending=[True, True])

        return actions["Name"].to_list()


class GreedyPolicy(EpsilonGreedyPolicy):
    """Greedy policy that always takes the best apparent action.

    Ties are broken by random selection. This is a special case of
    EpsilonGreedy where epsilon = 0 (always exploit).
    """

    def __init__(self):
        """Initialize the GreedyPolicy with epsilon = 0."""
        super().__init__(0)

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name.
        """
        return "Greedy"
