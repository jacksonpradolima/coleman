"""
coleman.bandit - Multi-Armed Bandit Models for Test Case Prioritization.

This module provides implementations of different multi-armed bandit (MAB) models to be used in
the context of software testing. Specifically, it provides a general Bandit class, a dynamic
version that allows for the addition and removal of arms, and an extension that incorporates
evaluation metrics to provide feedback on the selected arms.

Classes
-------
Bandit
    Represents the classic multi-armed bandit model.
DynamicBandit
    An extension of the basic Bandit that supports dynamic management of its arms.
EvaluationMetricBandit
    A dynamic bandit that provides feedback based on a given evaluation metric.

Notes
-----
The module facilitates the use of MAB models for test case prioritization in software testing
scenarios. By integrating feedback from evaluation metrics, it allows for adaptive
prioritization strategies that can react to changes in the testing environment or system
under test.
"""

from .base import BANDIT_SCHEMA, Bandit
from .dynamic import DynamicBandit
from .metric_bandit import EvaluationMetricBandit

__all__ = ["BANDIT_SCHEMA", "Bandit", "DynamicBandit", "EvaluationMetricBandit"]
