"""Adversarial MAB policies."""

from __future__ import annotations

import math

import polars as pl

from coleman.agent import Agent

from ..base import Policy
from .shared import DeltaRewardMixin, action_names


class EXP3Policy(Policy, DeltaRewardMixin):
    """EXP3 for adversarial bandits.

    References
    ----------
    .. [1] Auer, P.; Cesa-Bianchi, N.; Freund, Y.; Schapire, R.
       "The Nonstochastic Multiarmed Bandit Problem." SIAM J. Computing, 2002.
    """

    def __init__(self, gamma: float = 0.07):
        """Initialize EXP3 exploration weight and gamma parameter."""
        DeltaRewardMixin.__init__(self)
        self.gamma = gamma
        self.weights: dict[str, float] = {}

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"EXP3 (Gamma={self.gamma})"

    def _ensure_actions(self, agent: Agent):
        """Initialize weights for any new actions."""
        for name in action_names(agent):
            self.weights.setdefault(name, 1.0)

    def _probs(self, names: list[str]) -> dict[str, float]:
        """Compute the current mixed EXP3 probability distribution over actions."""
        k = len(names)
        total_w = sum(self.weights[name] for name in names)
        if total_w <= 0:
            return {name: 1.0 / k for name in names}
        return {name: (1.0 - self.gamma) * (self.weights[name] / total_w) + self.gamma / k for name in names}

    def choose_all(self, agent: Agent) -> list[str]:
        """Select and rank actions by EXP3 probabilities."""
        self._ensure_actions(agent)
        names = action_names(agent)
        probs = self._probs(names)
        return [name for name, _ in sorted(probs.items(), key=lambda item: item[1], reverse=True)]

    def credit_assignment(self, agent: Agent) -> None:
        """Update weights based on adversarial rewards."""
        self._ensure_actions(agent)
        names = action_names(agent)
        probs = self._probs(names)
        rewards = self.extract_step_rewards(agent)
        k = max(len(names), 1)

        for name in names:
            est = rewards.get(name, 0.0) / max(probs[name], 1e-12)
            self.weights[name] *= math.exp((self.gamma * est) / k)

        q = [probs[name] for name in names]
        agent.actions = agent.actions.with_columns(pl.Series("Q", q))


class EXP3IXPolicy(EXP3Policy):
    """EXP3-IX with implicit exploration.

    References
    ----------
    .. [1] Neu, G. "First-order regret bounds for combinatorial semi-bandits."
       COLT, 2015.
    """

    def __init__(self, eta: float = 0.1, gamma: float = 0.01):
        """Initialize EXP3-IX learning rate and exploration parameters."""
        super().__init__(gamma=gamma)
        self.eta = eta

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"EXP3IX (Eta={self.eta}, Gamma={self.gamma})"

    def credit_assignment(self, agent: Agent) -> None:
        """Update weights using implicit exploration."""
        self._ensure_actions(agent)
        names = action_names(agent)
        probs = self._probs(names)
        rewards = self.extract_step_rewards(agent)

        for name in names:
            est = rewards.get(name, 0.0) / max(probs[name] + self.gamma, 1e-12)
            self.weights[name] *= math.exp(self.eta * est)

        q = [probs[name] for name in names]
        agent.actions = agent.actions.with_columns(pl.Series("Q", q))
