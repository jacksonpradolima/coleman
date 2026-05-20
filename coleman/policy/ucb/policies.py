"""UCB policy variants."""

import numpy as np
import polars as pl

from coleman.agent import Agent

from ..base import Policy


class UCBPolicyBase(Policy):
    """Base class for Upper Confidence Bound (UCB) policies.

    Parameters
    ----------
    c : float
        Exploration parameter controlling the width of the confidence bound.

    Attributes
    ----------
    c : float
        Exploration parameter.
    """

    def __init__(self, c: float):
        """Initialize the UCBPolicyBase.

        Parameters
        ----------
        c : float
            Exploration parameter controlling the width of the confidence bound.
        """
        if c <= 0:
            raise ValueError(f"Exploration parameter c must be positive, got {c!r}")
        self.c = c

    def choose_all(self, agent: Agent) -> list:
        """Choose all actions sorted by Q value in descending order.

        Parameters
        ----------
        agent : Agent
            The agent whose actions are to be prioritized.

        Returns
        -------
        list of str
            List of action names sorted by Q value.
        """
        return agent.actions.sort("Q", descending=True)["Name"].to_list()


class UCB1Policy(UCBPolicyBase):
    """Upper Confidence Bound algorithm (UCB1).

    Applies an exploration factor to the expected value of each arm which can
    influence a greedy selection strategy to more intelligently explore less
    confident options.
    """

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name with C value.
        """
        return f"UCB1 (C={self.c})"

    def credit_assignment(self, agent: Agent):
        """Assign credit using the UCB1 formula.

        Parameters
        ----------
        agent : Agent
            The agent for which credit assignment is to be performed.
        """
        super().credit_assignment(agent)

        log_t = np.log1p(agent.t)
        agent.actions = agent.actions.with_columns(
            pl.when(pl.col("ActionAttempts") > 0)
            .then((pl.lit(log_t) / pl.col("ActionAttempts")).pow(1 / self.c))
            .otherwise(0.0)
            .alias("_exploration")
        )
        agent.actions = agent.actions.with_columns((pl.col("Q") + pl.col("_exploration")).alias("Q")).drop(
            "_exploration"
        )


class UCBPolicy(UCBPolicyBase):
    """Upper Confidence Bound algorithm (UCB) with scaling factor."""

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name with C value.
        """
        return f"UCB (C={self.c})"

    def credit_assignment(self, agent: Agent):
        """Assign credit using the UCB formula with scaling factor.

        Parameters
        ----------
        agent : Agent
            The agent for which credit assignment is to be performed.
        """
        super().credit_assignment(agent)

        sum_attempts = float(agent.actions["ActionAttempts"].sum() or 0.0)
        log_sum_attempts = float(np.log1p(sum_attempts))

        agent.actions = agent.actions.with_columns(
            pl.when(pl.col("ActionAttempts") > 0)
            .then(((2 * log_sum_attempts) / pl.col("ActionAttempts")).pow(0.5))
            .otherwise(0.0)
            .alias("_exploration")
        )
        agent.actions = agent.actions.with_columns((pl.col("Q") + self.c * pl.col("_exploration")).alias("Q")).drop(
            "_exploration"
        )
