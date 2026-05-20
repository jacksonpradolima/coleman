"""Linear Thompson Sampling contextual policies."""

from coleman.agent import Agent, SlidingWindowContextualAgent

from .. import base as _policy_base
from .linucb import LinUCBPolicy


class LinTSPolicy(LinUCBPolicy):
    """Linear Thompson Sampling for contextual bandits.

    References
    ----------
    .. [1] Agrawal, S.; Goyal, N. "Thompson Sampling for Contextual Bandits
       with Linear Payoffs." ICML, 2013.
    """

    def __init__(self, alpha: float = 0.5):
        """Initialize the LinTS exploration scale."""
        super().__init__(alpha=alpha)

    def __str__(self):
        """Return a string representation of the policy."""
        return f"LinTS (Alpha={self.alpha})"

    def choose_all(self, agent: Agent):
        """Choose all actions by Thompson samples from linear posteriors."""
        features = self.context_features.select(self.features).to_numpy()
        actions = self.context_features["Name"].to_list()

        q_values = []
        for a, x in zip(actions, features, strict=False):
            x_i = x.reshape(-1, 1)
            a_inv = self.context["A_inv"][a]
            theta_hat = a_inv.dot(self.context["b"][a]).reshape(-1)

            theta_sample = _policy_base._rng.multivariate_normal(theta_hat, (self.alpha**2) * a_inv)
            p_t = float(theta_sample.dot(x_i.reshape(-1)))
            q_values.append((a, p_t))

        return [action for action, _ in sorted(q_values, key=lambda x: x[1], reverse=True)]


class SWLinTSPolicy(LinTSPolicy):
    """Sliding-window Linear Thompson Sampling.

    References
    ----------
    .. [1] Russac, Y.; et al. "Weighted Linear Bandits for Non-Stationary
       Environments." NeurIPS, 2019.
    """

    def __str__(self):
        """Return a string representation of the policy."""
        return f"SWLinTS (Alpha={self.alpha})"

    def choose_all(self, agent: Agent):
        """Choose all actions with LinTS and sliding-window frequency discount."""
        if not isinstance(agent, SlidingWindowContextualAgent):
            raise TypeError("SWLinTSPolicy requires a SlidingWindowContextualAgent")

        features = self.context_features.select(self.features).to_numpy()
        actions = self.context_features["Name"].to_list()

        history_names = set(agent.history["Name"].unique().to_list())
        history_counts = agent.history["Name"].value_counts().to_dicts()
        history_counts_dict = {item["Name"]: item["count"] for item in history_counts}

        q_values = []
        for a, x in zip(actions, features, strict=False):
            x_i = x.reshape(-1, 1)
            a_inv = self.context["A_inv"][a]
            theta_hat = a_inv.dot(self.context["b"][a]).reshape(-1)
            theta_sample = _policy_base._rng.multivariate_normal(theta_hat, (self.alpha**2) * a_inv)

            q = float(theta_sample.dot(x_i.reshape(-1)))
            occ = 0
            if agent.t > agent.window_size and a in history_names:
                occ = history_counts_dict.get(a, 0)
            q *= 1 - occ / agent.window_size
            q_values.append((a, q))

        return [action for action, _ in sorted(q_values, key=lambda x: x[1], reverse=True)]
