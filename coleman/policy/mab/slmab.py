"""SlMAB policy."""

import polars as pl

from coleman.agent import HISTORY_SCHEMA, Agent, RewardSlidingWindowAgent

from ..base import Policy


class SlMABPolicy(Policy):
    """Sliding Multi-Armed Bandit policy.

    Attributes
    ----------
    history : polars.DataFrame
        History of actions and their outcomes.
    """

    def __init__(self):
        """Initialize the SlMABPolicy."""
        self.history = pl.DataFrame(schema=HISTORY_SCHEMA)

    def __str__(self):
        """Return a string representation of the policy.

        The closing parenthesis is intentionally omitted so the agent can
        append the window size when constructing the full label.

        Returns
        -------
        str
            The policy name without closing parenthesis.
        """
        return "SlMAB ("

    def choose_all(self, agent: Agent):
        """Choose all actions based on Q values from the SlMAB history.

        Parameters
        ----------
        agent : Agent
            The agent whose actions are to be prioritized.

        Returns
        -------
        list of str
            List of action names sorted by Q value.
        """
        existing_names = set(self.history["Name"].to_list())
        agent_names = set(agent.actions["Name"].to_list())
        new_names = agent_names - existing_names

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

        return self.history.sort("Q", descending=True)["Name"].to_list()

    def credit_assignment(self, agent: RewardSlidingWindowAgent):
        """Assign credit using the SlMAB method.

        Parameters
        ----------
        agent : RewardSlidingWindowAgent
            The agent for which credit assignment is to be performed.
        """
        super().credit_assignment(agent)

        self.history = agent.history.group_by(["Name"]).agg(
            [pl.col("T").count().cast(HISTORY_SCHEMA["T"]).alias("T"), pl.col("T").max().alias("Ti")]
        )

        agent_data = agent.actions.select(["Name", "Q"]).rename({"Q": "action_Q"})
        self.history = self.history.join(agent_data, on="Name", how="left")
        self.history = self.history.rename({"action_Q": "Q"})

        if "R" in agent.actions.columns:
            agent_r = agent.actions.select(["Name", "R"]).rename({"R": "action_R"})
            self.history = self.history.join(agent_r, on="Name", how="left")
            self.history = self.history.rename({"action_R": "R"})
        else:
            self.history = self.history.with_columns([pl.lit(0.0).alias("R")])

        self.history = self.history.with_columns([(pl.lit(agent.t) - pl.col("Ti")).alias("DiffSelection")])

        self.history = self.history.with_columns(
            [
                (
                    pl.col("T")
                    * ((agent.window_size / (agent.window_size + pl.col("DiffSelection"))) + (1.0 / (pl.col("T") + 1)))
                ).alias("T")
            ]
        )

        self.history = self.history.with_columns(
            [
                (
                    pl.col("Q")
                    * (
                        (agent.window_size / (agent.window_size + pl.col("DiffSelection")))
                        + pl.col("R") * (1.0 / (pl.col("T") + 1))
                    )
                ).alias("Q")
            ]
        )
