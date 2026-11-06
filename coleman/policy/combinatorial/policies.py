"""Combinatorial bandit policies.

These preserved CI adaptations emit a full ordering. Their subset-size splice
``ordered[:k] + ordered[k:]`` is order-neutral; it neither selects an executed
subset nor establishes combinatorial semi-bandit feedback or regret guarantees.
"""

from __future__ import annotations

import math

import polars as pl

from coleman.agent import Agent

from .. import base as _policy_base
from ..base import Policy
from ..mab.shared import DeltaRewardMixin, action_names, bounded_reward, safe_mean


class CombinatorialUCBPolicy(Policy):
    """Preserved full UCB ranking with an order-neutral subset-size parameter.

    What it is
    ----------
    Ranks every current test using native weighted attempts and cumulative
    credit. Changing subset_size alone leaves the complete order unchanged.
    The cited combinatorial action/feedback guarantees are not claimed.

    Parameters
    ----------
    subset_size : int, optional
        Positive count used by the order-neutral legacy splice.
    c : float, optional
        Exploration strength for the UCB bonus.

    References
    ----------
    .. [1] Chen, W.; Wang, Y.; Yuan, Y. "Combinatorial Multi-Armed Bandit:
       General Framework and Applications." ICML, 2013.
    .. [2] Kveton, B.; Wen, Z.; Ashkan, A.; Szepesvári, C. "Tight Regret
       Bounds for Stochastic Combinatorial Semi-Bandits." AISTATS, 2015.
    """

    def __init__(self, subset_size: int = 5, c: float = 1.0):
        """Initialize combinatorial UCB parameters."""
        if subset_size <= 0:
            raise ValueError(f"subset_size must be positive, got {subset_size!r}")
        if c <= 0:
            raise ValueError(f"Exploration parameter c must be positive, got {c!r}")
        self.subset_size = subset_size
        self.c = c

    def __str__(self):
        """Return a concise human-readable policy description."""
        return f"CombinatorialUCB (K={self.subset_size}, C={self.c})"

    def _order_with_subset(self, scores: dict[str, float]) -> list[str]:
        """Sort by score and move the top-k subset to the front."""
        ordered = [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
        k = min(self.subset_size, len(ordered))
        return ordered[:k] + ordered[k:]

    def choose_all(self, agent: Agent):
        """Rank all actions using UCB and keep the best subset at the head."""
        total_n = float(agent.actions["ActionAttempts"].sum() or 0.0) + 1.0
        log_total = math.log(total_n)
        scores: dict[str, float] = {}

        for row in agent.actions.select(["Name", "ActionAttempts", "ValueEstimates"]).iter_rows(named=True):
            name = str(row["Name"])
            attempts = float(row["ActionAttempts"])
            mean = safe_mean(float(row["ValueEstimates"]), attempts)
            if attempts <= 0:
                scores[name] = float("inf")
            else:
                scores[name] = mean + self.c * math.sqrt(log_total / attempts)

        return self._order_with_subset(scores)

    def credit_assignment(self, agent: Agent):
        """Apply the default reward update for compatibility with base policy behavior."""
        super().credit_assignment(agent)


class CombinatorialThompsonPolicy(Policy, DeltaRewardMixin):
    """Preserved sampled-score full ranking using per-name Beta state.

    What it is
    ----------
    Samples a score per current name, sorts every name and applies an
    order-neutral subset splice. The emitted action is a complete ranking,
    not a sampled subset; native zero credit is not an observed failure/pass.
    The cited combinatorial semi-bandit guarantees are not claimed.

    Parameters
    ----------
    subset_size : int, optional
        Positive count used by the order-neutral legacy splice.
    alpha_prior : float, optional
        Prior alpha parameter for each action.
    beta_prior : float, optional
        Prior beta parameter for each action.

    References
    ----------
    .. [1] Agrawal, S.; Goyal, N. "Analysis of Thompson Sampling for the
       Multi-armed Bandit Problem." COLT, 2012.
    .. [2] Wang, S.; Chen, W. "Thompson Sampling for Combinatorial Semi-
       Bandits." ICML, 2018.
    """

    def __init__(self, subset_size: int = 5, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """Initialize combinatorial Thompson Sampling parameters."""
        if subset_size <= 0:
            raise ValueError(f"subset_size must be positive, got {subset_size!r}")
        DeltaRewardMixin.__init__(self)
        self.subset_size = subset_size
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.alpha: dict[str, float] = {}
        self.beta: dict[str, float] = {}

    def __str__(self):
        """Return a concise human-readable policy description."""
        return f"CombinatorialThompson (K={self.subset_size}, Alpha={self.alpha_prior}, Beta={self.beta_prior})"

    def _ensure_actions(self, agent: Agent):
        """Initialize Beta priors for any new actions."""
        for name in action_names(agent):
            self.alpha.setdefault(name, float(self.alpha_prior))
            self.beta.setdefault(name, float(self.beta_prior))

    def _order_with_subset(self, scores: dict[str, float]) -> list[str]:
        """Sort by score and move the top-k subset to the front."""
        ordered = [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
        k = min(self.subset_size, len(ordered))
        return ordered[:k] + ordered[k:]

    def choose_all(self, agent: Agent):
        """Sample Thompson scores for every action and return subset-first ordering."""
        self._ensure_actions(agent)
        samples = {
            name: float(_policy_base._rng.beta(self.alpha[name], self.beta[name])) for name in action_names(agent)
        }
        return self._order_with_subset(samples)

    def credit_assignment(self, agent: Agent):
        """Update Beta posterior parameters from bounded step rewards."""
        self._ensure_actions(agent)
        rewards = self.extract_step_rewards(agent)

        for name, reward in rewards.items():
            bounded = bounded_reward(reward)
            self.alpha[name] += bounded
            self.beta[name] += 1.0 - bounded

        means = [self.alpha[name] / (self.alpha[name] + self.beta[name]) for name in action_names(agent)]
        agent.actions = agent.actions.with_columns(pl.Series("Q", means))
