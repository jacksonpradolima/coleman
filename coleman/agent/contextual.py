"""ContextualAgent - agent with contextual information."""

import polars as pl

from .reward_agent import RewardAgent
from .schemas import ACTIONS_SCHEMA


class ContextualAgent(RewardAgent):
    """An agent that learns using a reward function and contextual information.

    The contextual information can be chosen by the user to guide
    decision-making.

    Parameters
    ----------
    policy : object
        The policy used by the agent to choose an action.
    reward_function : object
        The reward function used by the agent to evaluate outcomes.

    Attributes
    ----------
    context_features : object or None
        The features of the current context.
    features : object or None
        The features used for decision-making.
    """

    def __init__(self, policy, reward_function, seed: int | None = None):
        """Initialize the ContextualAgent.

        Parameters
        ----------
        policy : object
            The policy used by the agent to choose an action.
        reward_function : object
            The reward function used by the agent to evaluate outcomes.
        seed : int, optional
            Seed forwarded to :class:`RewardAgent` for reproducible initial shuffle.
        """
        super().__init__(policy, reward_function, seed=seed)

        self.context_features: pl.DataFrame = pl.DataFrame()
        self.features: list[str] = []

    def __str__(self):
        """Return a string representation of the contextual agent.

        Returns
        -------
        str
            String representation of the agent's policy.
        """
        return f"{str(self.policy)}"

    def choose(self) -> list[str]:
        """Choose an action using the agent's policy.

        An action is the prioritized test suite.

        Returns
        -------
        list of str
            List of test cases in ascending order of priority.
        """
        self.last_prioritization = self.policy.choose_all(self)
        return self.last_prioritization

    def update_actions(self, actions):
        """Update the set of available actions based on the current context.

        This method adjusts the agent's possible actions based on the current
        context. In some situations, the available actions might change based on
        the state of the environment or other contextual information. This method
        ensures that the agent always has an up-to-date set of actions to choose
        from.

        Parameters
        ----------
        actions : list of str
            List of available action names.
        """
        self.actions = self.actions.filter(pl.col("Name").is_in(actions))

        new_actions = [action for action in actions if action not in self.actions["Name"].to_list()]

        self.policy.update_actions(self, new_actions)

        if new_actions:
            new_actions_df = pl.DataFrame(
                {
                    "Name": new_actions,
                    "ValueEstimates": [0.0] * len(new_actions),
                    "ActionAttempts": [0.0] * len(new_actions),
                    "Q": [0.0] * len(new_actions),
                },
                schema=ACTIONS_SCHEMA,
            )
            self.actions = pl.concat([self.actions, new_actions_df], how="vertical")

    def update_bandit(self, bandit):
        """Update the internal bandit instance used by the agent.

        This method updates the agent's internal bandit to the provided
        instance. This can be useful when the agent needs to adapt to changes
        in the environment or when the bandit's state changes over time.

        Parameters
        ----------
        bandit : Bandit
            The new bandit instance to be used by the agent.
        """
        self.bandit = bandit
        self.update_actions(bandit.get_arms())

    def update_context(self, context_features: pl.DataFrame):
        """Update the agent's current context information.

        The context provides additional information that can help the agent in
        making decisions. This might include external factors or environmental
        states that could influence the agent's strategy.

        Parameters
        ----------
        context_features : object
            A collection or dataframe containing the contextual information.
        """
        self.context_features = context_features

    def update_features(self, features: list[str]):
        """Update the features used by the agent for decision-making.

        Features represent specific characteristics or properties of data that
        the agent uses to make its decisions.

        Parameters
        ----------
        features : list
            A list or collection of features.
        """
        self.features = features
