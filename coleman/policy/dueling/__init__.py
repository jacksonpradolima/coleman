"""Dueling and ranking bandit policies based on pairwise preferences."""

from .policies import DuelingUCBPolicy, PairwiseThompsonRankingPolicy

__all__ = ["DuelingUCBPolicy", "PairwiseThompsonRankingPolicy"]
