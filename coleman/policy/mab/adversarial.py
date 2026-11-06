"""Adversarial MAB policies."""

from __future__ import annotations

import math

import polars as pl

from coleman.agent import Agent

from ..base import Policy
from .shared import DeltaRewardMixin, action_names


class EXP3Policy(Policy, DeltaRewardMixin):
    """Preserved EXP3-inspired native CI ranking adaptation.

    Notes
    -----
    ``choose_all`` sorts internal mixed scores without sampling an action.
    Credit assignment updates every current name using the native agent's
    delta credit, including its bounded transform and unscheduled-zero meaning.
    The scores are not executed-ranking propensities. The sampled-action
    estimator and regret guarantees in the reference are not claimed here.

    References
    ----------
    .. [1] Auer, P.; Cesa-Bianchi, N.; Freund, Y.; Schapire, R.
       "The Nonstochastic Multiarmed Bandit Problem." SIAM J. Computing, 2002.
    """

    def __init__(self, gamma: float = 0.07):
        """Initialize the legacy mixed-score coefficient without validation.

        Parameters
        ----------
        gamma : float, optional
            Dimensionless mixture/update coefficient, default 0.07. This legacy
            constructor does not enforce finite values or the interval [0, 1].

        Notes
        -----
        Creates empty weights/delta-credit state; performs no RNG draw or I/O.
        It does not define a generic bounded loss or sampling contract.
        """
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
        """Compute legacy mixed ranking scores, not emitted-action propensities."""
        k = len(names)
        total_w = sum(self.weights[name] for name in names)
        if total_w <= 0:
            return {name: 1.0 / k for name in names}
        return {name: (1.0 - self.gamma) * (self.weights[name] / total_w) + self.gamma / k for name in names}

    def choose_all(self, agent: Agent) -> list[str]:
        """Sort current names by internal mixed scores without sampling.

        Parameters
        ----------
        agent : Agent
            Native current action table; names and insertion-order ties survive.

        Returns
        -------
        list of str
            Full descending score order, not a sampled action or propensity.

        Notes
        -----
        Initializes missing policy weights to one. No RNG draw or I/O occurs;
        the base agent may perform a separate initial shuffle before this path.
        """
        self._ensure_actions(agent)
        names = action_names(agent)
        probs = self._probs(names)
        return [name for name, _ in sorted(probs.items(), key=lambda item: item[1], reverse=True)]

    def credit_assignment(self, agent: Agent) -> None:
        """Apply the preserved all-name positive native-credit update.

        Parameters
        ----------
        agent : Agent
            Native post-observe action table with cumulative or window credits.

        Raises
        ------
        OverflowError
            The unbounded exponential update exceeds the floating-point range.

        Notes
        -----
        Mutates weights, delta-credit cache and the agent's dimensionless Q
        column. Does not observe raw test outcomes or save state. Legacy finite
        credit in [0, 1] is unchanged; other finite deltas are tanh-projected and
        non-finite credit maps to zero. This is not a sampled-action estimator.
        """
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
    """Preserved EXP3IX-named positive-gain native ranking adaptation.

    Notes
    -----
    Inherits sorted full-ranking selection. Every current name receives a
    positive exponential update using native delta credit divided by its
    internal mixed score plus gamma. This is not the sampled-loss estimator
    of the reference; no implicit-exploration regret bound is claimed.

    References
    ----------
    .. [1] Neu, G. "Explore no more: Improved high-probability regret bounds
       for non-stochastic bandits." NeurIPS, 2015.
    """

    def __init__(self, eta: float = 0.1, gamma: float = 0.01):
        """Initialize the preserved positive-gain update parameters.

        Parameters
        ----------
        eta : float, optional
            Dimensionless exponential gain scale, default 0.1.
        gamma : float, optional
            Dimensionless mixed-score and denominator coefficient, default 0.01.

        Notes
        -----
        Legacy construction validates neither parameter. Creates fresh local
        weights/delta state without sampling or I/O; does not define a loss.
        """
        super().__init__(gamma=gamma)
        self.eta = eta

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"EXP3IX (Eta={self.eta}, Gamma={self.gamma})"

    def credit_assignment(self, agent: Agent) -> None:
        """Apply the preserved positive-gain denominator-offset update.

        Parameters
        ----------
        agent : Agent
            Native post-observe action table; no sampled-loss receipt is used.

        Raises
        ------
        OverflowError
            The unbounded positive exponential update exceeds binary64 range.

        Notes
        -----
        Mutates all current-name weights, the delta-credit cache and Q. Uses
        ``max(score + gamma, 1e-12)`` and the same native bounded-credit transform
        as EXP3Policy. Performs no I/O and claims no EXP3-IX loss guarantee.
        """
        self._ensure_actions(agent)
        names = action_names(agent)
        probs = self._probs(names)
        rewards = self.extract_step_rewards(agent)

        for name in names:
            est = rewards.get(name, 0.0) / max(probs[name] + self.gamma, 1e-12)
            self.weights[name] *= math.exp(self.eta * est)

        q = [probs[name] for name in names]
        agent.actions = agent.actions.with_columns(pl.Series("Q", q))
