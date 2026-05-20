"""Stochastic MAB policies."""

from __future__ import annotations

import math

import polars as pl

from coleman.agent import Agent

from .. import base as _policy_base
from ..base import Policy
from .shared import DeltaRewardMixin, action_names, bounded_reward, safe_mean


class ThompsonSamplingPolicy(Policy, DeltaRewardMixin):
    """Thompson Sampling with Beta-Bernoulli posterior.

    References
    ----------
    .. [1] Thompson, W. R. "On the likelihood that one unknown probability
       exceeds another in view of the evidence of two samples." Biometrika,
       1933.
    .. [2] Agrawal, S.; Goyal, N. "Analysis of Thompson Sampling for the
       Multi-armed Bandit Problem." COLT, 2012.
    """

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """Initialize Beta priors for each arm."""
        DeltaRewardMixin.__init__(self)
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.alpha: dict[str, float] = {}
        self.beta: dict[str, float] = {}

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return "ThompsonSampling"

    def _ensure_actions(self, agent: Agent):
        """Initialize Beta priors for any new actions."""
        for name in action_names(agent):
            self.alpha.setdefault(name, float(self.alpha_prior))
            self.beta.setdefault(name, float(self.beta_prior))

    def choose_all(self, agent: Agent) -> list[str]:
        """Sample from Beta posteriors and rank actions."""
        self._ensure_actions(agent)
        samples = {
            name: float(_policy_base._rng.beta(self.alpha[name], self.beta[name])) for name in action_names(agent)
        }
        return [name for name, _ in sorted(samples.items(), key=lambda item: item[1], reverse=True)]

    def credit_assignment(self, agent: Agent) -> None:
        """Update Beta priors based on observed rewards."""
        self._ensure_actions(agent)
        rewards = self.extract_step_rewards(agent)
        for name, reward in rewards.items():
            self.alpha[name] += reward
            self.beta[name] += 1.0 - reward

        means = [self.alpha[name] / (self.alpha[name] + self.beta[name]) for name in action_names(agent)]
        agent.actions = agent.actions.with_columns(pl.Series("Q", means))


class BayesianUCBPolicy(ThompsonSamplingPolicy):
    """Bayesian UCB with Beta posterior mean and standard deviation bonus.

    References
    ----------
    .. [1] Kaufmann, E.; Korda, N.; Munos, R. "Thompson Sampling: an
       asymptotically optimal finite-time analysis." ALT, 2012.
    """

    def __init__(self, c: float = 2.0, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """Initialize Bayesian UCB confidence scaling and Beta priors."""
        super().__init__(alpha_prior=alpha_prior, beta_prior=beta_prior)
        self.c = c

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"BayesianUCB (C={self.c})"

    def choose_all(self, agent: Agent) -> list[str]:
        """Select actions using Bayesian UCB with Beta posterior statistics."""
        self._ensure_actions(agent)
        scores: dict[str, float] = {}
        for name in action_names(agent):
            a = self.alpha[name]
            b = self.beta[name]
            total = a + b
            mean = a / total
            var = (a * b) / ((total * total) * (total + 1.0))
            scores[name] = mean + self.c * math.sqrt(max(var, 0.0))
        return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


class KLUCBPolicy(Policy):
    """KL-UCB for Bernoulli rewards.

    References
    ----------
    .. [1] Garivier, A.; Cappé, O. "The KL-UCB Algorithm for Bounded
       Stochastic Bandits and Beyond." COLT, 2011.
    """

    def __init__(self, c: float = 3.0):
        """Initialize the KL-UCB exploration constant."""
        self.c = c

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"KLUCB (C={self.c})"

    @staticmethod
    def _kl_bernoulli(p: float, q: float) -> float:
        """Compute the KL divergence between two Bernoulli distributions."""
        p = min(max(p, 1e-12), 1 - 1e-12)
        q = min(max(q, 1e-12), 1 - 1e-12)
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    def _solve_index(self, mean: float, n: float, budget: float) -> float:
        """Binary-search for the KL-UCB upper confidence index."""
        if n <= 0:
            return 1.0
        low = min(max(mean, 0.0), 1.0)
        high = 1.0
        for _ in range(32):
            mid = 0.5 * (low + high)
            if n * self._kl_bernoulli(mean, mid) <= budget:
                low = mid
            else:
                high = mid
        return low

    def choose_all(self, agent: Agent) -> list[str]:
        """Select actions using KL-UCB index."""
        t = max(float(agent.t), 1.0)
        log_term = math.log(t) + self.c * math.log(max(math.log(t), 1.0))

        scores: dict[str, float] = {}
        for row in agent.actions.select(["Name", "ActionAttempts", "ValueEstimates"]).iter_rows(named=True):
            name = str(row["Name"])
            n = float(row["ActionAttempts"])
            mean = bounded_reward(safe_mean(float(row["ValueEstimates"]), n))
            scores[name] = self._solve_index(mean, n, log_term)

        return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

    def credit_assignment(self, agent: Agent) -> None:
        """Update value estimates through parent policy."""
        super().credit_assignment(agent)


class UCBTunedPolicy(Policy, DeltaRewardMixin):
    """UCB-Tuned with empirical variance correction.

    References
    ----------
    .. [1] Auer, P.; Cesa-Bianchi, N.; Fischer, P. "Finite-time Analysis of
       the Multiarmed Bandit Problem." Machine Learning, 2002.
    """

    def __init__(self, c: float = 1.0):
        """Initialize UCB-Tuned exploration constant and Welford accumulators."""
        DeltaRewardMixin.__init__(self)
        self.c = c
        self.counts: dict[str, int] = {}
        self.means: dict[str, float] = {}
        self.m2: dict[str, float] = {}

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"UCBTuned (C={self.c})"

    def _ensure_actions(self, agent: Agent):
        """Initialize Welford accumulators for any new actions."""
        for name in action_names(agent):
            self.counts.setdefault(name, 0)
            self.means.setdefault(name, 0.0)
            self.m2.setdefault(name, 0.0)

    def choose_all(self, agent: Agent) -> list[str]:
        """Select actions using UCB-Tuned with variance correction."""
        self._ensure_actions(agent)
        t = max(sum(self.counts.values()), 1)
        log_t = math.log(t + 1.0)

        scores: dict[str, float] = {}
        for name in action_names(agent):
            n = self.counts[name]
            if n <= 0:
                scores[name] = float("inf")
                continue
            mean = self.means[name]
            variance = self.m2[name] / n
            v = variance + math.sqrt(2.0 * log_t / n)
            bonus = math.sqrt((log_t / n) * min(0.25, v))
            scores[name] = mean + self.c * bonus

        return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

    def credit_assignment(self, agent: Agent) -> None:
        """Update means and variance estimates using Welford's algorithm."""
        self._ensure_actions(agent)
        rewards = self.extract_step_rewards(agent)
        for name, reward in rewards.items():
            n1 = self.counts[name]
            n2 = n1 + 1
            delta = reward - self.means[name]
            mean = self.means[name] + delta / n2
            delta2 = reward - mean
            self.m2[name] += delta * delta2
            self.means[name] = mean
            self.counts[name] = n2

        q = [self.means.get(name, 0.0) for name in action_names(agent)]
        agent.actions = agent.actions.with_columns(pl.Series("Q", q))


class MOSSUCBPolicy(Policy):
    """Minimax Optimal Strategy in the Stochastic case (MOSS).

    References
    ----------
    .. [1] Audibert, J.-Y.; Bubeck, S. "Minimax Policies for Adversarial and
       Stochastic Bandits." COLT, 2009.
    """

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return "MOSSUCB"

    def choose_all(self, agent: Agent) -> list[str]:
        """Select actions using MOSS minimax optimal strategy."""
        k = max(agent.actions.height, 1)
        total_n = float(agent.actions["ActionAttempts"].sum() or 0.0) + 1.0
        scores: dict[str, float] = {}

        for row in agent.actions.select(["Name", "ActionAttempts", "ValueEstimates"]).iter_rows(named=True):
            name = str(row["Name"])
            n = float(row["ActionAttempts"])
            mean = safe_mean(float(row["ValueEstimates"]), n)
            if n <= 0:
                scores[name] = float("inf")
                continue
            ratio = max(total_n / (k * n), 1.0)
            bonus = math.sqrt(max(math.log(ratio), 0.0) / n)
            scores[name] = mean + bonus

        return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

    def credit_assignment(self, agent: Agent) -> None:
        """Update value estimates through parent policy."""
        super().credit_assignment(agent)


class UCBVPolicy(UCBTunedPolicy):
    """UCB-V variance-aware confidence bound.

    References
    ----------
    .. [1] Audibert, J.-Y.; Munos, R.; Szepesvári, C. "Exploration-
       Exploitation Tradeoff using Variance Estimates in Multi-Armed Bandits."
       Theoretical Computer Science, 2009.
    """

    def __init__(self, c: float = 1.0, b: float = 1.0):
        """Initialize UCB-V confidence and range parameters."""
        super().__init__(c=c)
        self.b = b

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"UCBV (C={self.c}, B={self.b})"

    def choose_all(self, agent: Agent) -> list[str]:
        """Select actions using UCB-V variance-aware bounds."""
        self._ensure_actions(agent)
        t = max(sum(self.counts.values()), 1)
        log_t = math.log(t + 1.0)

        scores: dict[str, float] = {}
        for name in action_names(agent):
            n = self.counts[name]
            if n <= 0:
                scores[name] = float("inf")
                continue
            mean = self.means[name]
            variance = self.m2[name] / n
            bonus = self.c * math.sqrt((2.0 * variance * log_t) / n) + (3.0 * self.b * log_t) / n
            scores[name] = mean + bonus

        return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


class PHEPolicy(Policy, DeltaRewardMixin):
    """Perturbed-History Exploration (PHE).

    References
    ----------
    .. [1] Kveton, B.; et al. "Perturbed-History Exploration in Stochastic
       Multi-Armed Bandits." IJCAI, 2019.
    """

    def __init__(self, a: float = 1.0):
        """Initialize PHE perturbation multiplier."""
        DeltaRewardMixin.__init__(self)
        self.a = a
        self.successes: dict[str, float] = {}
        self.counts: dict[str, float] = {}

    def __str__(self) -> str:
        """Return string representation of the policy."""
        return f"PHE (A={self.a})"

    def _ensure_actions(self, agent: Agent):
        """Initialize success and count accumulators for any new actions."""
        for name in action_names(agent):
            self.successes.setdefault(name, 0.0)
            self.counts.setdefault(name, 0.0)

    def choose_all(self, agent: Agent) -> list[str]:
        """Select actions using perturbed history exploration."""
        self._ensure_actions(agent)
        scores: dict[str, float] = {}
        for name in action_names(agent):
            n = self.counts[name]
            if n <= 0:
                scores[name] = float("inf")
                continue
            m = max(int(round(self.a * n)), 1)
            pseudo = float(_policy_base._rng.binomial(m, 0.5))
            scores[name] = (self.successes[name] + pseudo) / (n + m)
        return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

    def credit_assignment(self, agent: Agent) -> None:
        """Update success and count statistics."""
        self._ensure_actions(agent)
        rewards = self.extract_step_rewards(agent)
        for name, reward in rewards.items():
            self.successes[name] += reward
            self.counts[name] += 1.0

        q = [self.successes[name] / self.counts[name] if self.counts[name] > 0 else 0.0 for name in action_names(agent)]
        agent.actions = agent.actions.with_columns(pl.Series("Q", q))
