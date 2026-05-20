"""LinUCB and SWLinUCB contextual bandit policies."""

import numpy as np
import polars as pl

from coleman.agent import Agent, ContextualAgent, SlidingWindowContextualAgent
from coleman.exceptions import QException

from ..base import Policy


class LinUCBPolicy(Policy):
    """LinUCB with Disjoint Linear Models.

    Parameters
    ----------
    alpha : float, optional
        The constant that determines the width of the upper confidence bound.
        Default is 0.5.

    Attributes
    ----------
    alpha : float
        The exploration parameter.
    context : dict
        Dictionary containing A matrices, their inverses, and b vectors for
        each action.
    context_features : object or None
        Current context features.
    features : object or None
        Feature names.

    References
    ----------
    .. [1] Lihong Li, et al. "A Contextual-Bandit Approach to Personalized
       News Article Recommendation." In Proceedings of the 19th
       International Conference on World Wide Web (WWW), 2010.
    """

    def __init__(self, alpha=0.5):
        """Initialize LinUCBPolicy.

        Parameters
        ----------
        alpha : float, optional
            The constant that determines the width of the upper confidence
            bound. Default is 0.5.
        """
        self.alpha = alpha

        self.context = {
            "A": {},
            "A_inv": {},
            "b": {},
        }

        self.context_features: pl.DataFrame = pl.DataFrame()
        self.features: list[str] = []

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name with alpha value.
        """
        return f"LinUCB (Alpha={self.alpha})"

    def add_action(self, action_id):
        """Add an action to the policy's context.

        Parameters
        ----------
        action_id : str
            The identifier of the action to add.
        """
        context_dimension = len(self.features)
        a = np.identity(context_dimension)

        self.context["A"][action_id] = a
        self.context["A_inv"][action_id] = np.linalg.inv(a)
        self.context["b"][action_id] = np.zeros((context_dimension, 1))

    def update_actions(self, agent: ContextualAgent, new_actions):
        """Update actions based on the agent's context.

        Parameters
        ----------
        agent : ContextualAgent
            The contextual agent providing context information.
        new_actions : list of str
            List of new action identifiers to add.
        """
        self.context_features = agent.context_features.sort("Name")
        self.features = agent.features

        for a in new_actions:
            self.add_action(a)

    def choose_all(self, agent: Agent):
        """Choose all actions based on the LinUCB policy.

        Parameters
        ----------
        agent : Agent
            The agent whose actions are to be prioritized.

        Returns
        -------
        list of str
            List of action names sorted by Q value in descending order.

        Raises
        ------
        QException
            If Q computation results in unexpected shape.
        """
        features = self.context_features.select(self.features).to_numpy()
        actions = self.context_features["Name"].to_list()

        q_values = []
        for a, x in zip(actions, features, strict=False):
            x_i = x.reshape(-1, 1)

            a_inv = self.context["A_inv"][a]
            theta_a = a_inv.dot(self.context["b"][a])

            p_t = theta_a.T.dot(x_i) + self.alpha * np.sqrt(x_i.T.dot(a_inv).dot(x_i))

            if p_t.size > 1:
                raise QException(f"[LinUCB] p_t is more than 1: shape={p_t.shape}, value={p_t}")

            q_values.append((a, p_t[0, 0]))

        return [action for action, _ in sorted(q_values, key=lambda x: x[1], reverse=True)]

    def credit_assignment(self, agent):
        """Assign credit based on the agent's actions and rewards.

        Parameters
        ----------
        agent : Agent
            The agent for which credit assignment is to be performed.
        """
        assert len(set(agent.actions["Name"].to_list()) - set(self.context_features["Name"].to_list())) == 0

        actions = agent.actions.clone()
        actions = actions.sort("Name")

        features = self.context_features.select(self.features).to_numpy()
        actions_data = actions.select(["Name", "ValueEstimates"]).to_numpy()
        for a, x in zip(actions_data, features, strict=False):
            x_i = x.reshape(-1, 1)
            act = a[0]
            reward = a[1]  # ValueEstimates

            self.context["A"][act] += x_i.dot(x_i.T)
            self.context["A_inv"][act] = np.linalg.inv(self.context["A"][act])
            self.context["b"][act] += reward * x_i


class SWLinUCBPolicy(LinUCBPolicy):
    """LinUCB with Disjoint Linear Models and Sliding Window.

    References
    ----------
    .. [1] Nicolas Gutowski, Tassadit Amghar, Olivier Camp, and Fabien Chhel.
       "Global Versus Individual Accuracy in Contextual Multi-Armed Bandit."
       In Proceedings of the 34th ACM/SIGAPP Symposium on Applied Computing
       (SAC '19), April 8-12, 2019, Limassol, Cyprus. ACM, 8 pages.
    """

    def __str__(self):
        """Return a string representation of the policy.

        Returns
        -------
        str
            The policy name with alpha value.
        """
        return f"SWLinUCB (Alpha={self.alpha}"

    def choose_all(self, agent: Agent):
        """Choose all actions based on the sliding window policy.

        Parameters
        ----------
        agent : Agent
            The agent whose actions are to be prioritized.

        Returns
        -------
        list of str
            List of action names sorted by Q value in descending order.

        Raises
        ------
        QException
            If Q computation results in unexpected shape.
        """
        if not isinstance(agent, SlidingWindowContextualAgent):
            raise TypeError("SWLinUCBPolicy requires a SlidingWindowContextualAgent")

        features = self.context_features.select(self.features).to_numpy()
        actions = self.context_features["Name"].to_list()

        history_names = set(agent.history["Name"].unique().to_list())
        history_counts = agent.history["Name"].value_counts().to_dicts()
        history_counts_dict = {item["Name"]: item["count"] for item in history_counts}

        q_values = []
        for a, x in zip(actions, features, strict=False):
            x_i = x.reshape(-1, 1)

            a_inv = self.context["A_inv"][a]
            theta_a = a_inv.dot(self.context["b"][a])
            q = theta_a.T.dot(x_i) + self.alpha * np.sqrt(x_i.T.dot(a_inv).dot(x_i))

            occ = 0
            if agent.t > agent.window_size and a in history_names:
                occ = history_counts_dict.get(a, 0)

            q *= 1 - occ / agent.window_size

            if q.size > 1:
                raise QException(f"[SWLinUCB] Q computation resulted in unexpected shape: {q.shape}")

            q_values.append((a, q[0, 0]))

        return [action for action, _ in sorted(q_values, key=lambda x: x[1], reverse=True)]
