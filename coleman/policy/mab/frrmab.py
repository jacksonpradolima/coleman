"""FRRMAB policy."""

import math

import polars as pl

from coleman.agent import HISTORY_SCHEMA, Agent, RewardSlidingWindowAgent

from ..base import Policy


class FRRMABPolicy(Policy):
    """Fitness-Rate-Rank based Multi-Armed Bandit (FRRMAB) policy.

    Parameters
    ----------
    c : float
        Exploration parameter.
    decayed_factor : float, optional
        Decay factor for ranking. Default is 1.

    Attributes
    ----------
    c : float
        Exploration parameter.
    decayed_factor : float
        Decay factor for ranking.
    history : polars.DataFrame
        History of actions and their outcomes.
    """

    def __init__(self, c, decayed_factor=1):
        """Initialize the FRRMABPolicy.

        Parameters
        ----------
        c : float
            Exploration parameter.
        decayed_factor : float, optional
            Decay factor for ranking. Default is 1.
        """
        self.c = c
        self.decayed_factor = decayed_factor
        self.history = pl.DataFrame(schema=HISTORY_SCHEMA)
        self._history_names: set[str] = set()
        self._ordered_names: list[str] = []

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name with C and D values.
        """
        return f"FRRMAB (C={self.c}, D={self.decayed_factor})"

    def choose_all(self, agent: Agent):
        """Choose all actions based on Q values from the FRRMAB history.

        Parameters
        ----------
        agent : Agent
            The agent whose actions are to be prioritized.

        Returns
        -------
        list of str
            List of action names sorted by Q value.
        """
        if (not self._history_names) and (self.history.height > 0):
            self._history_names = set(self.history["Name"].to_list())

        agent_names_cached = getattr(agent, "_action_names", None)
        if (not agent_names_cached) or (len(agent_names_cached) != agent.actions.height):
            agent_names = set(agent.actions["Name"].to_list())
        else:
            agent_names = set(agent_names_cached)
        new_names = agent_names - self._history_names

        if new_names:
            new_entries = pl.DataFrame(
                {
                    "Name": list(new_names),
                    "ActionAttempts": [0.0] * len(new_names),
                    "ValueEstimates": [0.0] * len(new_names),
                    "Q": [0.0] * len(new_names),
                    "T": [0] * len(new_names),
                },
                schema=HISTORY_SCHEMA,
            )
            self.history = pl.concat([self.history, new_entries], how="vertical")
            self._history_names |= new_names
            self.history = self.history.sort("Q", descending=True)
            self._ordered_names = self.history["Name"].to_list()

        if not self._ordered_names and self.history.height > 0:
            self._ordered_names = self.history["Name"].to_list()

        return list(self._ordered_names)

    def credit_assignment(self, agent: RewardSlidingWindowAgent):
        """Assign credit using the Fitness-Rate-Rank method.

        Parameters
        ----------
        agent : RewardSlidingWindowAgent
            The agent for which credit assignment is to be performed.
        """
        self.history = agent.history.group_by("Name").agg(
            [
                pl.col("ActionAttempts").sum(),
                pl.col("ValueEstimates").sum(),
                pl.col("T").count().cast(HISTORY_SCHEMA["T"]).alias("T"),
            ]
        )

        self.history = self.history.sort("ValueEstimates", descending=True)
        self.history = self.history.with_row_index("rank", offset=1).with_columns(
            [(pl.col("ValueEstimates") * (pl.lit(float(self.decayed_factor)).pow(pl.col("rank")))).alias("_decay")]
        )

        decay_total = float(self.history["_decay"].sum()) if self.history.height > 0 else 0.0
        selected_times_total = float(self.history["T"].sum()) if self.history.height > 0 else 0.0
        log_selected_times = math.log1p(selected_times_total) if selected_times_total > 0 else 0.0

        self.history = self.history.with_columns(
            [
                (
                    pl.when(pl.lit(abs(decay_total) <= 1e-12))
                    .then(0.0)
                    .otherwise(pl.col("_decay") / pl.lit(decay_total))
                ).alias("frr"),
                (
                    pl.when(pl.col("T") > 0)
                    .then((pl.lit(2.0 * log_selected_times) / pl.col("T")).sqrt())
                    .otherwise(0.0)
                ).alias("exploration"),
            ]
        )

        self.history = (
            self.history.with_columns((pl.col("frr") + self.c * pl.col("exploration")).alias("Q"))
            .drop(["rank", "_decay", "frr", "exploration"])
            .sort("Q", descending=True)
            .select(list(HISTORY_SCHEMA.keys()))
        )
        self._history_names = set(self.history["Name"].to_list())
        self._ordered_names = self.history["Name"].to_list()
