"""Sliding-window UCB policy."""

import math

import polars as pl

from coleman.agent import Agent, RewardSlidingWindowAgent

from .policies import UCBPolicyBase


class SlidingWindowUCBPolicy(UCBPolicyBase):
    """Sliding-window UCB policy.

    References
    ----------
    .. [1] Garivier, A.; Moulines, E. "On Upper-Confidence Bound Policies for
       Non-Stationary Bandit Problems." ALT, 2011.
    """

    def __str__(self):
        """Return a string representation of the policy."""
        return f"SWUCB (C={self.c})"

    def credit_assignment(self, agent: Agent):
        """Assign credit using only the most recent sliding-window data."""
        if not isinstance(agent, RewardSlidingWindowAgent):
            super().credit_assignment(agent)
            return

        if agent.history.height == 0:
            super().credit_assignment(agent)
            return

        max_t_value = agent.history["T"].max()
        max_t = int(max_t_value) if isinstance(max_t_value, int | float) else 0
        min_t = max(max_t - agent.window_size + 1, 0)
        window_history = agent.history.filter(pl.col("T") >= min_t)

        grouped = window_history.group_by("Name").agg(
            [
                pl.col("ValueEstimates").sum().alias("sum_rewards"),
                pl.col("T").count().cast(pl.Float64).alias("sw_attempts"),
            ]
        )

        total_attempts = float(grouped["sw_attempts"].sum() or 0.0)
        log_total = math.log(max(total_attempts, 1.0))

        grouped = grouped.with_columns(
            [
                (pl.col("sum_rewards") / pl.col("sw_attempts")).alias("Q"),
                pl.when(pl.col("sw_attempts") > 0)
                .then((pl.lit(log_total) / pl.col("sw_attempts")).sqrt())
                .otherwise(0.0)
                .alias("_bonus"),
            ]
        ).with_columns((pl.col("Q") + self.c * pl.col("_bonus")).alias("Q"))

        agent.actions = (
            agent.actions.drop("Q")
            .join(grouped.select(["Name", "Q"]), on="Name", how="left")
            .with_columns(pl.col("Q").fill_null(0.0))
        )
