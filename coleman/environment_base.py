"""Abstract environment contract for extensible environment implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractEnvironment(ABC):
    """Base class describing the minimum environment API."""

    @abstractmethod
    def reset(self) -> None:
        """Reset state for a new simulation run."""

    @abstractmethod
    def run_single(
        self,
        experiment: int,
        trials: int = 100,
        bandit_type: Any = None,
        restore: bool = True,
        forecast_next: bool = True,
    ) -> None:
        """Execute one experiment iteration.

        Parameters
        ----------
        experiment : int
            Experiment identifier.
        trials : int, optional
            Maximum number of trials to execute.
        bandit_type : Any, optional
            Bandit implementation class.
        restore : bool, optional
            Whether to attempt restore from checkpoints.
        forecast_next : bool, optional
            Whether to compute the next prioritization forecast after the
            final evaluated build.
        """

    @abstractmethod
    def run(
        self,
        experiments: int = 1,
        trials: int = 100,
        bandit_type: Any = None,
        restore: bool = True,
        forecast_next: bool = True,
    ) -> None:
        """Execute one or more experiments.

        Parameters
        ----------
        experiments : int, optional
            Number of experiments to execute.
        trials : int, optional
            Maximum number of trials per experiment.
        bandit_type : Any, optional
            Bandit implementation class.
        restore : bool, optional
            Whether to attempt restore from checkpoints.
        forecast_next : bool, optional
            Whether to compute the next prioritization forecast after each
            experiment.
        """
