"""Decaying epsilon-greedy policy."""

import polars as pl

from coleman.agent import Agent

from .. import base as _policy_base
from ..base import Policy


class DecayEpsilonGreedyPolicy(Policy):
    """Epsilon-greedy with polynomial decay schedule.

    References
    ----------
    .. [1] Sutton, R. S.; Barto, A. G. "Reinforcement Learning: An
       Introduction." MIT Press, 2018.
    """

    def __init__(self, epsilon0: float = 1.0, decay: float = 0.5, min_epsilon: float = 0.01):
        """Initialize decay schedule hyperparameters."""
        self.epsilon0 = epsilon0
        self.decay = decay
        self.min_epsilon = min_epsilon

    def __str__(self):
        """Return a string representation of the policy."""
        return f"DecayEpsilonGreedy (Epsilon0={self.epsilon0}, Decay={self.decay}, Min={self.min_epsilon})"

    def choose_all(self, agent: Agent):
        """Choose actions by decaying epsilon-greedy ordering."""
        epsilon_t = max(self.min_epsilon, self.epsilon0 / ((max(agent.t, 0) + 1.0) ** self.decay))

        actions = agent.actions.clone()
        n = len(actions)
        rand_vals = _policy_base._rng.random(n)
        actions = actions.with_columns([pl.Series("rand_val", rand_vals)])
        actions = actions.with_columns(
            [
                (pl.col("rand_val") < epsilon_t).alias("is_random"),
                pl.when(pl.col("rand_val") < epsilon_t)
                .then(pl.col("rand_val"))
                .otherwise(pl.col("Q"))
                .alias("sort_key"),
            ]
        )
        actions = actions.sort(["is_random", "sort_key"], descending=[True, True])

        return actions["Name"].to_list()
