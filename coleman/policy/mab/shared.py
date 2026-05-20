"""Shared helpers for extended MAB policies."""

from __future__ import annotations

import math

import numpy as np

from coleman.agent import Agent


def safe_mean(value_estimate: float, attempts: float) -> float:
    """Return empirical mean reward guarding against zero attempts."""
    if attempts <= 0:
        return 0.0
    return float(value_estimate) / float(attempts)


def bounded_reward(value: float) -> float:
    """Map arbitrary rewards into [0, 1] for probability-based updates."""
    if not np.isfinite(value):
        return 0.0
    if 0.0 <= value <= 1.0:
        return float(value)
    return float(0.5 * (math.tanh(value) + 1.0))


def action_names(agent: Agent) -> list[str]:
    """Return action names from agent state."""
    return agent.actions["Name"].to_list()


class DeltaRewardMixin:
    """Mixin that derives per-step rewards from cumulative value estimates."""

    def __init__(self):
        """Initialize internal state for reward deltas."""
        self._last_value_estimate: dict[str, float] = {}

    def extract_step_rewards(self, agent: Agent) -> dict[str, float]:
        """Compute reward increments for the current step from cumulative values."""
        updates: dict[str, float] = {}
        for row in agent.actions.select(["Name", "ValueEstimates"]).iter_rows(named=True):
            name = str(row["Name"])
            current = float(row["ValueEstimates"])
            previous = self._last_value_estimate.get(name, 0.0)
            updates[name] = bounded_reward(current - previous)
            self._last_value_estimate[name] = current
        return updates
