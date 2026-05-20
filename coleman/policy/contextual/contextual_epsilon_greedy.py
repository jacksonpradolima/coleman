"""Contextual epsilon-greedy policies, including sliding-window variant."""

from coleman.agent import Agent, SlidingWindowContextualAgent
from coleman.exceptions import QException

from .. import base as _policy_base
from .linucb import LinUCBPolicy


class ContextualEpsilonGreedyPolicy(LinUCBPolicy):
    """Contextual epsilon-greedy based on linear reward estimates.

    References
    ----------
    .. [1] Langford, J.; Zhang, T. "The Epoch-Greedy Algorithm for
       Multi-armed Bandits with Side Information." NIPS, 2007.
    """

    def __init__(self, epsilon: float = 0.1):
        """Initialize contextual epsilon-greedy policy.

        Parameters
        ----------
        epsilon : float, optional
            Exploration probability.
        """
        super().__init__(alpha=0.0)
        self.epsilon = epsilon

    def __str__(self):
        """Return a string representation of the policy."""
        return f"ContextualEpsilonGreedy (Epsilon={self.epsilon})"

    def choose_all(self, agent: Agent):
        """Choose all actions with epsilon-randomized contextual scoring."""
        features = self.context_features.select(self.features).to_numpy()
        actions = self.context_features["Name"].to_list()

        scores: list[tuple[str, float]] = []
        for a, x in zip(actions, features, strict=False):
            x_i = x.reshape(-1, 1)
            a_inv = self.context["A_inv"][a]
            theta_a = a_inv.dot(self.context["b"][a])
            p_t = theta_a.T.dot(x_i)
            if p_t.size > 1:
                raise QException(f"[ContextualEpsilonGreedy] invalid score shape: {p_t.shape}")
            scores.append((a, float(p_t[0, 0])))

        if _policy_base._rng.random() < self.epsilon:
            shuffled = [a for a, _ in scores]
            _policy_base._rng.shuffle(shuffled)
            return shuffled

        return [action for action, _ in sorted(scores, key=lambda x: x[1], reverse=True)]


class SWContextualEpsilonGreedyPolicy(ContextualEpsilonGreedyPolicy):
    """Sliding-window contextual epsilon-greedy policy.

    References
    ----------
    .. [1] Besbes, O.; Gur, Y.; Zeevi, A. "Stochastic Multi-Armed-Bandit
       Problem with Non-stationary Rewards." NeurIPS, 2014.
    """

    def __str__(self):
        """Return a string representation of the policy."""
        return f"SWContextualEpsilonGreedy (Epsilon={self.epsilon})"

    def choose_all(self, agent: Agent):
        """Choose all actions with contextual epsilon-greedy and window discount."""
        if not isinstance(agent, SlidingWindowContextualAgent):
            raise TypeError("SWContextualEpsilonGreedyPolicy requires a SlidingWindowContextualAgent")

        features = self.context_features.select(self.features).to_numpy()
        actions = self.context_features["Name"].to_list()
        history_names = set(agent.history["Name"].unique().to_list())
        history_counts = agent.history["Name"].value_counts().to_dicts()
        history_counts_dict = {item["Name"]: item["count"] for item in history_counts}

        scores: list[tuple[str, float]] = []
        for a, x in zip(actions, features, strict=False):
            x_i = x.reshape(-1, 1)
            a_inv = self.context["A_inv"][a]
            theta_a = a_inv.dot(self.context["b"][a])
            p_t = theta_a.T.dot(x_i)
            if p_t.size > 1:
                raise QException(f"[SWContextualEpsilonGreedy] invalid score shape: {p_t.shape}")

            score = float(p_t[0, 0])
            occ = 0
            if agent.t > agent.window_size and a in history_names:
                occ = history_counts_dict.get(a, 0)
            score *= 1 - occ / agent.window_size
            scores.append((a, score))

        if _policy_base._rng.random() < self.epsilon:
            shuffled = [a for a, _ in scores]
            _policy_base._rng.shuffle(shuffled)
            return shuffled

        return [action for action, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
