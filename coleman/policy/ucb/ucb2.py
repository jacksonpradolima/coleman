"""UCB2 policy."""

import math

import polars as pl

from coleman.agent import Agent

from .policies import UCBPolicyBase


class UCB2Policy(UCBPolicyBase):
    """UCB2 policy.

    References
    ----------
    .. [1] Auer, P.; Cesa-Bianchi, N.; Fischer, P. "Finite-time Analysis of
       the Multiarmed Bandit Problem." Machine Learning, 2002.
    """

    def __init__(self, c: float = 1.0, alpha: float = 0.2):
        """Initialize UCB2 hyperparameters."""
        super().__init__(c=c)
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha!r}")
        self.alpha = alpha

    def __str__(self):
        """Return a string representation of the policy."""
        return f"UCB2 (C={self.c}, Alpha={self.alpha})"

    def credit_assignment(self, agent: Agent):
        """Assign credit using UCB2 confidence schedule."""
        super().credit_assignment(agent)

        total_attempts = float(agent.actions["ActionAttempts"].sum() or 0.0)
        t = max(total_attempts, 1.0)

        def _ucb2_bonus(attempts: float) -> float:
            """Compute the UCB2 exploration bonus for a given attempt count."""
            n = max(float(attempts), 1.0)
            r = math.floor(math.log(n, 1.0 + self.alpha)) if n > 1 else 0
            tau_r = (1.0 + self.alpha) ** r
            num = (1.0 + self.alpha) * math.log(max(math.e * t / tau_r, 1.0))
            return math.sqrt(max(num / (2.0 * tau_r), 0.0))

        bonuses = [_ucb2_bonus(v) for v in agent.actions["ActionAttempts"].to_list()]
        agent.actions = agent.actions.with_columns(pl.Series("_bonus", bonuses))
        agent.actions = agent.actions.with_columns((pl.col("Q") + self.c * pl.col("_bonus")).alias("Q")).drop("_bonus")
