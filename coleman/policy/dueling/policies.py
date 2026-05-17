"""Dueling and ranking bandit policies.

The policies in this module update action preferences from pairwise outcomes
induced by the last produced ranking.
"""

from __future__ import annotations

import math
from collections import defaultdict

import polars as pl

from coleman.agent import Agent

from .. import base as _policy_base
from ..base import Policy
from ..mab.shared import action_names


class DuelingUCBPolicy(Policy):
    """Pairwise-UCB ranking with Copeland-style scores.

    What it is
    ----------
    A dueling/ranking bandit policy that learns preferences through pairwise
    comparisons between actions, then induces a global ranking from these
    preferences using Copeland-like aggregation.

    Parameters
    ----------
    c : float, optional
        Exploration parameter for optimistic pairwise preference estimates.

    References
    ----------
    .. [1] Yue, Y.; Joachims, T. "Beat the Mean Bandit." ICML, 2011.
    .. [2] Zoghi, M.; Karnin, Z.; Whiteson, S.; de Rijke, M.; Munos, R.
       "Copeland Dueling Bandits." NeurIPS, 2015.
    """

    def __init__(self, c: float = 1.0):
        """Initialize DuelingUCB exploration parameter and pairwise counters."""
        if c <= 0:
            raise ValueError(f"Exploration parameter c must be positive, got {c!r}")
        self.c = c
        self._wins_first: dict[tuple[str, str], float] = defaultdict(float)
        self._duels: dict[tuple[str, str], float] = defaultdict(float)

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"DuelingUCB (C={self.c})"

    @staticmethod
    def _pair(a: str, b: str) -> tuple[str, str]:
        """Return a canonical (sorted) pair key for two action names."""
        if a <= b:
            return (a, b)
        return (b, a)

    def _copeland_scores(self, names: list[str]) -> dict[str, float]:
        """Compute Copeland scores from current win/duel statistics."""
        total_duels = float(sum(self._duels.values())) + 1.0
        log_total = math.log(total_duels)
        scores = {name: 0.0 for name in names}

        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                key = self._pair(left, right)
                n = self._duels.get(key, 0.0)
                wins_first = self._wins_first.get(key, 0.0)
                wins_left = wins_first if left == key[0] else (n - wins_first)
                if n <= 0.0:
                    pref_left = 0.5
                else:
                    mean_left = wins_left / n
                    bonus = self.c * math.sqrt(log_total / (n + 1.0))
                    pref_left = min(1.0, max(0.0, mean_left + bonus))

                if pref_left >= 0.5:
                    scores[left] += 1.0
                if (1.0 - pref_left) >= 0.5:
                    scores[right] += 1.0

        return scores

    def choose_all(self, agent: Agent) -> list[str]:
        """Select and rank all actions using Copeland scores."""
        names = action_names(agent)
        scores = self._copeland_scores(names)
        return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

    def credit_assignment(self, agent: Agent) -> None:
        """Update preference estimates from the ranking."""
        ranking = list(agent.last_prioritization)
        for i, winner in enumerate(ranking):
            for loser in ranking[i + 1 :]:
                key = self._pair(winner, loser)
                self._duels[key] += 1.0
                if winner == key[0]:
                    self._wins_first[key] += 1.0

        names = action_names(agent)
        scores = self._copeland_scores(names)
        q = [scores[name] for name in names]
        agent.actions = agent.actions.with_columns(pl.Series("Q", q))


class PairwiseThompsonRankingPolicy(Policy):
    """Pairwise Thompson Sampling over duels for ranking.

    What it is
    ----------
    A ranking-oriented bandit policy that models pairwise preferences with
    Beta posteriors and samples them online to produce a total action ordering.

    Uses a Beta posterior per unordered action pair and samples preferences
    to build a ranking at each decision step.

    References
    ----------
    .. [1] Thompson, W. R. "On the likelihood that one unknown probability
        exceeds another in view of the evidence of two samples." Biometrika,
        1933.
    .. [2] Wu, H.; Liu, X.; Yue, Y. "Double Thompson Sampling for Dueling
        Bandits." NeurIPS, 2016.
    """

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """Initialize Beta priors for each pairwise action combination."""
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self._alpha: dict[tuple[str, str], float] = defaultdict(lambda: float(alpha_prior))
        self._beta: dict[tuple[str, str], float] = defaultdict(lambda: float(beta_prior))

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"PairwiseThompsonRanking (Alpha={self.alpha_prior}, Beta={self.beta_prior})"

    @staticmethod
    def _pair(a: str, b: str) -> tuple[str, str]:
        """Return a canonical (sorted) pair key for two action names."""
        if a <= b:
            return (a, b)
        return (b, a)

    @staticmethod
    def _sample_pref(
        left: str,
        right: str,
        alpha: dict[tuple[str, str], float],
        beta: dict[tuple[str, str], float],
    ) -> float:
        """Sample a preference probability for left over right from the Beta posterior."""
        key = PairwiseThompsonRankingPolicy._pair(left, right)
        sample = float(_policy_base._rng.beta(alpha[key], beta[key]))
        if left <= right:
            return sample
        return 1.0 - sample

    def choose_all(self, agent: Agent) -> list[str]:
        """Sample and rank all actions using Thompson sampling."""
        names = action_names(agent)
        sampled_scores = {name: 0.0 for name in names}

        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                pref_left = self._sample_pref(left, right, self._alpha, self._beta)
                if pref_left >= 0.5:
                    sampled_scores[left] += 1.0
                else:
                    sampled_scores[right] += 1.0

        return [name for name, _ in sorted(sampled_scores.items(), key=lambda item: item[1], reverse=True)]

    def credit_assignment(self, agent: Agent) -> None:
        """Update Beta posteriors from the ranking."""
        ranking = list(agent.last_prioritization)
        for i, winner in enumerate(ranking):
            for loser in ranking[i + 1 :]:
                key = self._pair(winner, loser)
                if winner <= loser:
                    self._alpha[key] += 1.0
                else:
                    self._beta[key] += 1.0

        names = action_names(agent)
        mean_scores = {name: 0.0 for name in names}
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                key = self._pair(left, right)
                p_left = self._alpha[key] / (self._alpha[key] + self._beta[key])
                mean_scores[left] += p_left
                mean_scores[right] += 1.0 - p_left

        q = [mean_scores[name] for name in names]
        agent.actions = agent.actions.with_columns(pl.Series("Q", q))
