"""Exploration-focused MAB policies."""

from __future__ import annotations

import numpy as np
import polars as pl

from coleman.agent import Agent

from .. import base as _policy_base
from ..base import Policy
from .shared import action_names


class SoftmaxPolicy(Policy):
    """Boltzmann/Softmax exploration policy.

    References
    ----------
    .. [1] Sutton, R. S.; Barto, A. G. "Reinforcement Learning: An
       Introduction." MIT Press, 2018.
    """

    def __init__(self, tau: float = 0.2):
        """Initialize Softmax temperature parameter."""
        self.tau = tau

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"Softmax (Tau={self.tau})"

    def choose_all(self, agent: Agent) -> list[str]:
        """Select and rank actions using softmax probabilities."""
        q = np.array(agent.actions["Q"].fill_null(0.0).to_numpy(), dtype=float)
        logits = q / max(self.tau, 1e-6)
        logits -= float(np.max(logits)) if logits.size > 0 else 0.0
        probs = np.exp(logits)
        probs_sum = float(np.sum(probs))
        probs = np.ones_like(probs) / max(len(probs), 1) if probs_sum <= 0 else probs / probs_sum

        names = action_names(agent)
        return [name for _, name in sorted(zip(probs.tolist(), names, strict=False), reverse=True)]

    def credit_assignment(self, agent: Agent) -> None:
        """Update value estimates through parent policy."""
        super().credit_assignment(agent)


class PursuitPolicy(Policy):
    """Pursuit method that tracks best arm and shifts selection probabilities.

    References
    ----------
    .. [1] Thathachar, M. A. L.; Sastry, P. S. "A New Approach to the Design
       of Reinforcement Schemes for Learning Automata." IEEE Trans. SMC, 1985.
    """

    def __init__(self, beta: float = 0.1):
        """Initialize Pursuit beta update rate."""
        self.beta = beta
        self.probs: dict[str, float] = {}

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"Pursuit (Beta={self.beta})"

    def _ensure_actions(self, agent: Agent):
        """Initialize uniform probabilities for any new actions."""
        names = action_names(agent)
        k = max(len(names), 1)
        for name in names:
            self.probs.setdefault(name, 1.0 / k)

    def choose_all(self, agent: Agent) -> list[str]:
        """Select and rank actions by pursuit probabilities."""
        self._ensure_actions(agent)
        names = action_names(agent)
        pairs = ((n, self.probs.get(n, 0.0)) for n in names)
        return [name for name, _ in sorted(pairs, key=lambda item: item[1], reverse=True)]

    def credit_assignment(self, agent: Agent) -> None:
        """Update pursuit probabilities based on best action."""
        super().credit_assignment(agent)
        self._ensure_actions(agent)
        names = action_names(agent)
        if not names:
            return

        best_name = max(names, key=lambda n: float(agent.actions.filter(pl.col("Name") == n)["Q"][0]))
        k = len(names)
        for name in names:
            target = 1.0 if name == best_name else 0.0
            self.probs[name] = (1.0 - self.beta) * self.probs.get(name, 1.0 / k) + self.beta * target

        total = sum(self.probs[n] for n in names)
        if total > 0:
            for name in names:
                self.probs[name] /= total

        q = [self.probs[name] for name in names]
        agent.actions = agent.actions.with_columns(pl.Series("Q", q))


class EpsilonDecreasingPolicy(Policy):
    """Epsilon-greedy with decreasing exploration schedule.

    References
    ----------
    .. [1] Sutton, R. S.; Barto, A. G. "Reinforcement Learning: An
       Introduction." MIT Press, 2018.
    """

    def __init__(self, epsilon0: float = 1.0, decay: float = 0.5):
        """Initialize epsilon schedule parameters."""
        self.epsilon0 = epsilon0
        self.decay = decay

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"EpsilonDecreasing (Epsilon0={self.epsilon0}, Decay={self.decay})"

    def choose_all(self, agent: Agent) -> list[str]:
        """Select actions using epsilon-greedy with decreasing exploration."""
        epsilon_t = self.epsilon0 / ((max(agent.t, 0) + 1.0) ** self.decay)
        names = action_names(agent)
        if _policy_base._rng.random() < epsilon_t:
            shuffled = names.copy()
            _policy_base._rng.shuffle(shuffled)
            return shuffled
        return agent.actions.sort("Q", descending=True)["Name"].to_list()

    def credit_assignment(self, agent: Agent) -> None:
        """Update value estimates through parent policy."""
        super().credit_assignment(agent)
