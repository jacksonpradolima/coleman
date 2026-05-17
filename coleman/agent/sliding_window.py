"""Sliding-window agent variants."""

import polars as pl

from coleman.evaluation import EvaluationMetric

from .contextual import ContextualAgent
from .reward_agent import RewardAgent
from .schemas import HISTORY_SCHEMA


class RewardSlidingWindowAgent(RewardAgent):
    """An agent that learns using a sliding window and a reward function.

    Parameters
    ----------
    policy : object
        The policy used by the agent to choose an action.
    reward_function : object
        The reward function used by the agent.
    window_size : int
        The size of the sliding window.

    Attributes
    ----------
    window_size : int
        The size of the sliding window.
    history : polars.DataFrame
        History of actions taken by the agent.
    """

    def __init__(self, policy, reward_function, window_size, seed: int | None = None):
        """Initialize the RewardSlidingWindowAgent.

        Parameters
        ----------
        policy : object
            The policy used by the agent to choose an action.
        reward_function : object
            The reward function used by the agent.
        window_size : int
            The size of the sliding window.
        seed : int, optional
            Seed forwarded to :class:`RewardAgent` for reproducible initial shuffle.
        """
        super().__init__(policy, reward_function, seed=seed)
        self.window_size = window_size

        self.history = pl.DataFrame(schema=HISTORY_SCHEMA)

    def __str__(self):
        """Return a string representation of the sliding window agent.

        Returns
        -------
        str
            String representation including policy and window size.
        """
        return f"{str(self.policy)}, SW={self.window_size})"

    def observe(self, reward: EvaluationMetric):
        """Observe the reward and update value estimates using the sliding window.

        Parameters
        ----------
        reward : EvaluationMetric
            The reward (result) obtained by the evaluation metric.
        """
        self.update_action_attempts()

        self.last_reward = self.reward_function.evaluate(reward, self.last_prioritization)
        n = len(self.last_prioritization)
        if n > 0:
            rewards_df = pl.DataFrame(
                {
                    "Name": self.last_prioritization[:n],
                    "_reward": list(self.last_reward[:n]),
                }
            )
            self.actions = (
                self.actions.join(rewards_df, on="Name", how="left")
                .with_columns(pl.col("_reward").fill_null(0.0).alias("ValueEstimates"))
                .drop("_reward")
            )
        else:
            self.actions = self.actions.with_columns(pl.lit(0.0).alias("ValueEstimates"))

        self.t += 1
        self.update_history()
        self.policy.credit_assignment(self)

    def update_history(self):
        """Update the agent's history of actions and outcomes.

        Adds the current action and its outcome to the agent's history.
        If the length of the history exceeds the window size, the oldest
        entries are removed to maintain the specified window size.
        """
        new_rows = self.actions.with_columns(pl.lit(self.t, dtype=pl.Int64).alias("T"))
        self.history = pl.concat([self.history, new_rows], how="vertical")

        if self.history.height > 0:
            max_t_value = self.history["T"].max()
            max_t = int(max_t_value) if isinstance(max_t_value, int | float) else 0
        else:
            max_t = 0
        if max_t > self.window_size:
            min_t = max_t - self.window_size
            self.history = self.history.filter(pl.col("T") > min_t)


class SlidingWindowContextualAgent(ContextualAgent):
    """An agent that learns using a reward function, contextual information, and a sliding window.

    Combines contextual decision-making with a sliding window mechanism.

    Parameters
    ----------
    policy : object
        The policy used by the agent to choose an action.
    reward_function : object
        The reward function used by the agent.
    window_size : int
        The size of the sliding window.

    Attributes
    ----------
    window_size : int
        The size of the sliding window.
    context_features : object or None
        The features of the current context.
    features : object or None
        The features used for decision-making.
    history : polars.DataFrame
        History of actions taken by the agent.
    """

    def __init__(self, policy, reward_function, window_size, seed: int | None = None):
        """Initialize the SlidingWindowContextualAgent.

        Parameters
        ----------
        policy : object
            The policy used by the agent to choose an action.
        reward_function : object
            The reward function used by the agent.
        window_size : int
            The size of the sliding window.
        seed : int, optional
            Seed forwarded to :class:`ContextualAgent` for reproducible initial shuffle.
        """
        super().__init__(policy, reward_function, seed=seed)

        self.window_size = window_size

        self.context_features: pl.DataFrame = pl.DataFrame()
        self.features: list[str] = []

        self.history = pl.DataFrame(schema=HISTORY_SCHEMA)

    def __str__(self):
        """Return a string representation of the sliding window contextual agent.

        Returns
        -------
        str
            String representation including policy and window size.
        """
        return f"{str(self.policy)}, SW={self.window_size})"

    def observe(self, reward: EvaluationMetric):
        """Observe the reward and update value estimates using the sliding window.

        Parameters
        ----------
        reward : EvaluationMetric
            The reward (result) obtained by the evaluation metric.
        """
        self.update_action_attempts()

        self.last_reward = self.reward_function.evaluate(reward, self.last_prioritization)
        n = len(self.last_prioritization)
        if n > 0:
            rewards_df = pl.DataFrame(
                {
                    "Name": self.last_prioritization[:n],
                    "_reward": list(self.last_reward[:n]),
                }
            )
            self.actions = (
                self.actions.join(rewards_df, on="Name", how="left")
                .with_columns(pl.col("_reward").fill_null(0.0).alias("ValueEstimates"))
                .drop("_reward")
            )
        else:
            self.actions = self.actions.with_columns(pl.lit(0.0).alias("ValueEstimates"))

        self.t += 1
        self.update_history()
        self.policy.credit_assignment(self)

    def update_history(self):
        """Update the agent's history of actions and outcomes.

        Adds the current action and its outcome to the agent's history.
        If the length of the history exceeds the window size, the oldest
        entries are removed to maintain the specified window size.
        """
        new_rows = self.actions.with_columns(pl.lit(self.t, dtype=pl.Int64).alias("T"))
        self.history = pl.concat([self.history, new_rows], how="vertical")

        if self.history.height > 0:
            max_t_value = self.history["T"].max()
            max_t = int(max_t_value) if isinstance(max_t_value, int | float) else 0
        else:
            max_t = 0
        if max_t > self.window_size:
            min_t = max_t - self.window_size
            self.history = self.history.filter(pl.col("T") > min_t)
