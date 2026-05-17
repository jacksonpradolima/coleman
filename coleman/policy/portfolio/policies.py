"""Portfolio meta-policies for online policy selection."""

from __future__ import annotations

import math
from collections import deque

from coleman.agent import Agent

from ..base import Policy
from ..mab.shared import bounded_reward


class PortfolioUCBPolicy(Policy):
    """UCB portfolio selector over a set of candidate policies.

    What it is
    ----------
    A meta-policy that treats each candidate policy as an arm in a portfolio
    bandit. It selects which policy to run online based on recent performance,
    balancing exploitation of strong policies and exploration of alternatives.

    Parameters
    ----------
    policies : list[Policy]
        Candidate policies that can be selected online.
    c : float, optional
        Exploration strength for portfolio-level UCB.
    window_size : int, optional
        Number of recent rewards used to score each candidate policy.

    References
    ----------
    .. [1] Auer, P.; Cesa-Bianchi, N.; Fischer, P. "Finite-time Analysis of
        the Multiarmed Bandit Problem." Machine Learning, 2002.
    .. [2] Fialho, Á.; Da Costa, L.; Schoenauer, M.; Sebag, M. "Analyzing
        Bandit-Based Adaptive Operator Selection Mechanisms." Annals of
        Mathematics and Artificial Intelligence, 2010.
    """

    def __init__(self, policies: list[Policy], c: float = 1.0, window_size: int = 20):
        """Initialize the portfolio with candidate policies and UCB parameters."""
        if not policies:
            raise ValueError("PortfolioUCBPolicy requires at least one candidate policy")
        if c <= 0:
            raise ValueError(f"Exploration parameter c must be positive, got {c!r}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size!r}")

        self.policies = list(policies)
        self.c = c
        self.window_size = window_size
        self._policy_rewards = [deque(maxlen=window_size) for _ in self.policies]
        self._policy_uses = [0 for _ in self.policies]
        self._active_index = 0
        self._last_value_total: float | None = None

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"PortfolioUCB (N={len(self.policies)}, C={self.c}, W={self.window_size})"

    @property
    def active_policy(self) -> Policy:
        """Return the currently selected policy."""
        return self.policies[self._active_index]

    def _policy_score(self, index: int, total_uses: int) -> float:
        """Compute the UCB score for a candidate policy by its index."""
        uses = self._policy_uses[index]
        if uses <= 0:
            return float("inf")

        rewards = self._policy_rewards[index]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        bonus = self.c * math.sqrt(math.log(total_uses + 1.0) / uses)
        return mean_reward + bonus

    def _select_policy(self) -> int:
        """Select the policy index with the highest UCB score."""
        total_uses = sum(self._policy_uses)
        scores = [self._policy_score(i, total_uses) for i in range(len(self.policies))]
        return max(range(len(self.policies)), key=lambda i: scores[i])

    def choose_all(self, agent: Agent) -> list[str]:
        """Select a policy and delegate action selection to it."""
        self._active_index = self._select_policy()
        return self.active_policy.choose_all(agent)

    def credit_assignment(self, agent: Agent) -> None:
        """Update the active policy and track portfolio reward."""
        self.active_policy.credit_assignment(agent)

        current_total = float(agent.actions["ValueEstimates"].sum() or 0.0)
        observed = 0.0 if self._last_value_total is None else bounded_reward(current_total - self._last_value_total)

        self._policy_rewards[self._active_index].append(observed)
        self._policy_uses[self._active_index] += 1
        self._last_value_total = current_total
