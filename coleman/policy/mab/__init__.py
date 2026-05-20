"""Sliding-window Multi-Armed Bandit (MAB) policies."""

from .adversarial import EXP3IXPolicy, EXP3Policy
from .exploration import EpsilonDecreasingPolicy, PursuitPolicy, SoftmaxPolicy
from .frrmab import FRRMABPolicy
from .nonstationary import BootstrappedThompsonPolicy, ChangeDetectionUCBPolicy, DiscountedUCBPolicy
from .slmab import SlMABPolicy
from .stochastic import (
    BayesianUCBPolicy,
    KLUCBPolicy,
    MOSSUCBPolicy,
    PHEPolicy,
    ThompsonSamplingPolicy,
    UCBTunedPolicy,
    UCBVPolicy,
)

__all__ = [
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
]
