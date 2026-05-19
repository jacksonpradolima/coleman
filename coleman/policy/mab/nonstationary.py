"""Non-stationary and bootstrap-based MAB policies."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import polars as pl

from coleman.agent import Agent

from .. import base as _policy_base
from ..base import Policy
from .shared import DeltaRewardMixin, action_names


class DiscountedUCBPolicy(Policy, DeltaRewardMixin):
    """Discounted-UCB for non-stationary rewards.

    References
    ----------
    .. [1] Garivier, A.; Moulines, E. "On Upper-Confidence Bound Policies for
       Non-Stationary Bandit Problems." ALT, 2011.
    """

    def __init__(self, gamma: float = 0.95, c: float = 1.0):
        """Initialize discount factor and exploration constant."""
        DeltaRewardMixin.__init__(self)
        self.gamma = gamma
        self.c = c
        self.n_disc: dict[str, float] = {}
        self.r_disc: dict[str, float] = {}

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"DiscountedUCB (Gamma={self.gamma}, C={self.c})"

    def _ensure_actions(self, agent: Agent):
        """Initialize discounted statistics for any new actions."""
        for name in action_names(agent):
            self.n_disc.setdefault(name, 0.0)
            self.r_disc.setdefault(name, 0.0)

    def choose_all(self, agent: Agent) -> list[str]:
        """Select actions using discounted UCB scores."""
        self._ensure_actions(agent)
        total_n = sum(self.n_disc.values()) + 1.0
        log_total = math.log(total_n)
        scores: dict[str, float] = {}
        for name in action_names(agent):
            n = self.n_disc[name]
            if n <= 1e-12:
                scores[name] = float("inf")
                continue
            mean = self.r_disc[name] / n
            bonus = math.sqrt(log_total / n)
            scores[name] = mean + self.c * bonus

        return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

    def credit_assignment(self, agent: Agent) -> None:
        """Update discounted statistics."""
        self._ensure_actions(agent)
        rewards = self.extract_step_rewards(agent)

        for name in self.n_disc:
            self.n_disc[name] *= self.gamma
            self.r_disc[name] *= self.gamma

        for name, reward in rewards.items():
            self.n_disc[name] += 1.0
            self.r_disc[name] += reward

        q = [self.r_disc[name] / self.n_disc[name] if self.n_disc[name] > 0 else 0.0 for name in action_names(agent)]
        agent.actions = agent.actions.with_columns(pl.Series("Q", q))


class BootstrappedThompsonPolicy(Policy, DeltaRewardMixin):
    """Bootstrapped Thompson Sampling using Poisson-resampled heads.

    References
    ----------
    .. [1] Osband, I.; Van Roy, B.; Wen, Z. "Generalization and Exploration
       via Randomized Value Functions." ICML, 2016.
    """

    def __init__(self, n_bootstrap: int = 8):
        """Initialize the number of bootstrap heads."""
        DeltaRewardMixin.__init__(self)
        self.n_bootstrap = n_bootstrap
        self.sums: dict[str, np.ndarray] = {}
        self.counts: dict[str, np.ndarray] = {}

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"BootstrappedThompson (B={self.n_bootstrap})"

    def _ensure_actions(self, agent: Agent):
        """Initialize zero-filled bootstrap arrays for any new actions."""
        for name in action_names(agent):
            if name not in self.sums:
                self.sums[name] = np.zeros(self.n_bootstrap, dtype=float)
                self.counts[name] = np.zeros(self.n_bootstrap, dtype=float)

    def choose_all(self, agent: Agent) -> list[str]:
        """Select actions using bootstrap heads."""
        self._ensure_actions(agent)
        head = int(_policy_base._rng.integers(0, self.n_bootstrap))
        scores: dict[str, float] = {}
        for name in action_names(agent):
            c = self.counts[name][head]
            scores[name] = self.sums[name][head] / c if c > 0 else 0.5
        return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

    def credit_assignment(self, agent: Agent) -> None:
        """Update bootstrap sums and counts."""
        self._ensure_actions(agent)
        rewards = self.extract_step_rewards(agent)
        for name, reward in rewards.items():
            k = _policy_base._rng.poisson(1.0, size=self.n_bootstrap).astype(float)
            self.sums[name] += k * reward
            self.counts[name] += k

        q = []
        for name in action_names(agent):
            c = self.counts[name].sum()
            q.append(float(self.sums[name].sum() / c) if c > 0 else 0.5)
        agent.actions = agent.actions.with_columns(pl.Series("Q", q))


class ChangeDetectionUCBPolicy(Policy, DeltaRewardMixin):
    """UCB with simple change detection and per-arm reset.

    References
    ----------
    .. [1] Liu, F.; Lee, J.; Shroff, N. "A Change-Detection based Framework
       for Piecewise-stationary Multi-Armed Bandit Problem." AAAI, 2018.
    .. [2] Garivier, A.; Moulines, E. "On Upper-Confidence Bound Policies for
       Non-Stationary Bandit Problems." ALT, 2011.
    """

    def __init__(self, c: float = 1.0, window: int = 20, threshold: float = 0.25):
        """Initialize UCB constant, change-detection window and threshold."""
        DeltaRewardMixin.__init__(self)
        self.c = c
        self.window = window
        self.threshold = threshold
        self.recent: dict[str, deque[float]] = {}
        self.counts: dict[str, int] = {}
        self.sums: dict[str, float] = {}

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"ChangeDetectionUCB (C={self.c}, W={self.window}, Th={self.threshold})"

    def _ensure_actions(self, agent: Agent):
        """Initialize per-arm history, counts and sums for any new actions."""
        for name in action_names(agent):
            self.recent.setdefault(name, deque(maxlen=self.window))
            self.counts.setdefault(name, 0)
            self.sums.setdefault(name, 0.0)

    def choose_all(self, agent: Agent) -> list[str]:
        """Select actions using UCB with change detection."""
        self._ensure_actions(agent)
        t = max(sum(self.counts.values()), 1)
        log_t = math.log(t + 1.0)
        scores: dict[str, float] = {}
        for name in action_names(agent):
            n = self.counts[name]
            if n <= 0:
                scores[name] = float("inf")
                continue
            mean = self.sums[name] / n
            scores[name] = mean + self.c * math.sqrt(log_t / n)
        return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

    def credit_assignment(self, agent: Agent) -> None:
        """Update statistics with change detection."""
        self._ensure_actions(agent)
        rewards = self.extract_step_rewards(agent)
        for name, reward in rewards.items():
            hist = self.recent[name]
            hist.append(reward)
            self.counts[name] += 1
            self.sums[name] += reward

            if len(hist) >= self.window:
                arr = np.array(hist, dtype=float)
                mid = len(arr) // 2
                before = float(arr[:mid].mean())
                after = float(arr[mid:].mean())
                if abs(after - before) > self.threshold:
                    self.counts[name] = len(arr[mid:])
                    self.sums[name] = float(arr[mid:].sum())
                    self.recent[name] = deque(arr[mid:].tolist(), maxlen=self.window)

        q = [self.sums[name] / self.counts[name] if self.counts[name] > 0 else 0.0 for name in action_names(agent)]
        agent.actions = agent.actions.with_columns(pl.Series("Q", q))
