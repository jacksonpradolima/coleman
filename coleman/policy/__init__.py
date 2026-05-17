"""Policies for multi-armed bandit and contextual bandit action selection.

This module provides a collection of policies that are designed to operate
with multi-armed bandits and contextual bandits. Each policy dictates how an
agent will select its actions based on prior knowledge, current context, or
exploration strategies.

Classes
-------
Policy
    Basic policy class that prescribes actions based on the memory of an agent.
EpsilonGreedyPolicy
    Chooses either the best apparent action or a random one based on a probability epsilon.
GreedyPolicy
    Always chooses the best apparent action.
RandomPolicy
    Always chooses a random action.
UCBPolicyBase
    Base class for Upper Confidence Bound policies.
UCB1Policy
    Implementation of the UCB1 algorithm.
UCBPolicy
    A variation of the UCB algorithm with a scaling factor.
FRRMABPolicy
    Fitness-Rate-Rank based Multi-Armed Bandit policy.
SlMABPolicy
    Sliding window-based Multi-Armed Bandit policy.
LinUCBPolicy
    Contextual bandit policy using linear upper confidence bounds.
SWLinUCBPolicy
    Variation of LinUCBPolicy using a sliding window approach.
CombinatorialUCBPolicy
    Combinatorial policy that prioritizes a top-k subset via UCB scores.
CombinatorialThompsonPolicy
    Combinatorial policy that prioritizes a top-k subset via Thompson sampling.
DuelingUCBPolicy
    Dueling/ranking policy based on pairwise UCB-style preference estimates.
PairwiseThompsonRankingPolicy
    Dueling/ranking policy based on pairwise Thompson preference sampling.
PortfolioUCBPolicy
    Meta-policy that selects among candidate policies online using UCB.

Notes
-----
- UCB (Upper Confidence Bound) policies are designed to balance exploration and exploitation by
  considering both the estimated reward of an action and the uncertainty around that reward.
- EpsilonGreedy and its variations (Greedy, Random) are simpler strategies that either exploit
  the best-known action or explore random actions based on a fixed probability.
- LinUCB and SWLinUCB are contextual bandits. They choose actions not just based on past rewards,
  but also considering the current context. SWLinUCB adds a sliding window mechanism to LinUCB,
  giving more weight to recent actions.
- Combinatorial policies prioritize a strong subset first (top-k) while preserving
    compatibility with full-ranking execution.
- Dueling/ranking policies learn pairwise preferences and derive a global priority order.
- Portfolio meta-policies select which policy to use online based on recent performance.

References
----------
.. [1] Lihong Li, et al. "A Contextual-Bandit Approach to Personalized News Article
   Recommendation." In Proceedings of the 19th International Conference on World Wide
   Web (WWW), 2010.
.. [2] Nicolas Gutowski, Tassadit Amghar, Olivier Camp, and Fabien Chhel. "Global Versus
   Individual Accuracy in Contextual Multi-Armed Bandit." In Proceedings of the 34th
   ACM/SIGAPP Symposium on Applied Computing (SAC '19), April 8-12, 2019, Limassol, Cyprus.
.. [3] Chen, W.; Wang, Y.; Yuan, Y. "Combinatorial Multi-Armed Bandit: General
    Framework and Applications." ICML, 2013.
.. [4] Yue, Y.; Joachims, T. "Beat the Mean Bandit." ICML, 2011.
.. [5] Zoghi, M.; Karnin, Z.; Whiteson, S.; de Rijke, M.; Munos, R.
    "Copeland Dueling Bandits." NeurIPS, 2015.
.. [6] Auer, P.; Cesa-Bianchi, N.; Fischer, P. "Finite-time Analysis of the
    Multiarmed Bandit Problem." Machine Learning, 2002.
"""

import sys
from types import ModuleType

from . import base as _policy_base
from .base import Policy
from .combinatorial import CombinatorialThompsonPolicy, CombinatorialUCBPolicy
from .contextual import (
    ContextualEpsilonGreedyPolicy,
    LinTSPolicy,
    LinUCBPolicy,
    SWContextualEpsilonGreedyPolicy,
    SWLinTSPolicy,
    SWLinUCBPolicy,
)
from .dueling import DuelingUCBPolicy, PairwiseThompsonRankingPolicy
from .greedy import DecayEpsilonGreedyPolicy, EpsilonGreedyPolicy, GreedyPolicy, OptimisticGreedyPolicy
from .mab import (
    BayesianUCBPolicy,
    BootstrappedThompsonPolicy,
    ChangeDetectionUCBPolicy,
    DiscountedUCBPolicy,
    EpsilonDecreasingPolicy,
    EXP3IXPolicy,
    EXP3Policy,
    FRRMABPolicy,
    KLUCBPolicy,
    MOSSUCBPolicy,
    PHEPolicy,
    PursuitPolicy,
    SlMABPolicy,
    SoftmaxPolicy,
    ThompsonSamplingPolicy,
    UCBTunedPolicy,
    UCBVPolicy,
)
from .portfolio import PortfolioUCBPolicy
from .random import RandomPolicy
from .ucb import SlidingWindowUCBPolicy, UCB1Policy, UCB2Policy, UCBPolicy, UCBPolicyBase


class _PolicyModule(ModuleType):
    """Module proxy that keeps ``coleman.policy._rng`` in sync with ``base._rng``."""

    def __getattr__(self, name):
        if name == "_rng":
            return _policy_base._rng
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name == "_rng":
            _policy_base._rng = value
            return
        super().__setattr__(name, value)


class _RNGProxy:
    """Proxy object that forwards RNG access to ``coleman.policy.base._rng``."""

    def __getattr__(self, name):
        return getattr(_policy_base._rng, name)

    def __repr__(self):
        return repr(_policy_base._rng)


# Make assignments like `coleman.policy._rng = ...` update the shared RNG used
# by policy implementations (`coleman.policy.base._rng`).
sys.modules[__name__].__class__ = _PolicyModule
_rng = _RNGProxy()

__all__ = [
    "_rng",
    "Policy",
    "EpsilonGreedyPolicy",
    "GreedyPolicy",
    "DecayEpsilonGreedyPolicy",
    "OptimisticGreedyPolicy",
    "RandomPolicy",
    "UCBPolicyBase",
    "UCB1Policy",
    "UCBPolicy",
    "UCB2Policy",
    "SlidingWindowUCBPolicy",
    "FRRMABPolicy",
    "SlMABPolicy",
    "ThompsonSamplingPolicy",
    "BayesianUCBPolicy",
    "KLUCBPolicy",
    "UCBTunedPolicy",
    "MOSSUCBPolicy",
    "DiscountedUCBPolicy",
    "EXP3Policy",
    "EXP3IXPolicy",
    "SoftmaxPolicy",
    "PursuitPolicy",
    "EpsilonDecreasingPolicy",
    "BootstrappedThompsonPolicy",
    "UCBVPolicy",
    "PHEPolicy",
    "ChangeDetectionUCBPolicy",
    "LinUCBPolicy",
    "SWLinUCBPolicy",
    "LinTSPolicy",
    "ContextualEpsilonGreedyPolicy",
    "SWLinTSPolicy",
    "SWContextualEpsilonGreedyPolicy",
    "CombinatorialUCBPolicy",
    "CombinatorialThompsonPolicy",
    "DuelingUCBPolicy",
    "PairwiseThompsonRankingPolicy",
    "PortfolioUCBPolicy",
]
