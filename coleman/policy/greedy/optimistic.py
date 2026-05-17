"""Optimistic greedy policy."""

import polars as pl

from coleman.agent import Agent

from .epsilon_greedy import GreedyPolicy


class OptimisticGreedyPolicy(GreedyPolicy):
    """Greedy with optimistic initialization for unseen actions.

    References
    ----------
    .. [1] Sutton, R. S.; Barto, A. G. "Reinforcement Learning: An
       Introduction." MIT Press, 2018.
    """

    def __init__(self, optimistic_q: float = 1.0):
        """Initialize optimistic Q value for never-attempted actions."""
        super().__init__()
        self.optimistic_q = optimistic_q

    def __str__(self):
        """Return a string representation of the policy."""
        return f"OptimisticGreedy (Q0={self.optimistic_q})"

    def credit_assignment(self, agent: Agent):
        """Assign empirical means and keep optimistic scores for unseen actions."""
        super().credit_assignment(agent)
        agent.actions = agent.actions.with_columns(
            pl.when(pl.col("ActionAttempts") <= 0)
            .then(pl.lit(float(self.optimistic_q)))
            .otherwise(pl.col("Q"))
            .alias("Q")
        )
