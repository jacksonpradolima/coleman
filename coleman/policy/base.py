"""Base Policy class."""

import os

import numpy as np
import polars as pl

from coleman.agent import Agent

# Shared RNG used by policy implementations.
#
# Default is deterministic for reproducibility/static-analysis compliance, but
# callers can override globally with COLEMAN_POLICY_SEED or at runtime via the
# experiment execution.seed option (runner reseeds this generator per run).
_rng = np.random.default_rng(int(os.getenv("COLEMAN_POLICY_SEED", "0")))


class Policy:
    """A policy prescribes an action to be taken based on the memory of an agent."""

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name.
        """
        return "Untreated"

    def choose_all(self, agent: Agent):
        """Return all actions in their default (untreated) order.

        Parameters
        ----------
        agent : Agent
            The agent whose actions are to be returned.

        Returns
        -------
        list of str
            List of action names.
        """
        return agent.actions["Name"].to_list()

    def credit_assignment(self, agent):
        """Assign credit to actions based on their outcomes.

        The credit assignment method calculates the value estimates for each
        action based on the rewards observed. The specific implementation of
        how credit is assigned depends on the policy in use.

        Parameters
        ----------
        agent : Agent
            The agent for which credit assignment is to be performed.

        Notes
        -----
        This is a base method and should be overridden in derived classes to
        provide specific credit assignment logic. The method modifies the
        agent's state, updating the value estimates for each action based on
        the outcomes observed.
        """
        agent.actions = agent.actions.with_columns(
            pl.when(pl.col("ActionAttempts") > 0)
            .then(pl.col("ValueEstimates") / pl.col("ActionAttempts"))
            .otherwise(0.0)
            .alias("Q")
        )
