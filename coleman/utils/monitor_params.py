"""
coleman.utils.monitor_params - Data Structure for MonitorCollector Parameters.

This module defines the ``CollectParams`` dataclass, which serves as a container for the
parameters required by the ``MonitorCollector.collect`` method. By grouping these parameters
into a single object, it improves method readability, reduces complexity, and provides a
structured approach to managing data during experiment execution.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from coleman.evaluation import EvaluationMetric


class ScenarioLoaderLike(Protocol):
    """Minimum interface needed by MonitorCollector during collection."""

    name: str
    avail_time_ratio: float


# Backward-compatible alias
ScenarioProviderLike = ScenarioLoaderLike


@dataclass
class CollectParams:
    """Dataclass encapsulating parameters required for collecting experiment data.

    Parameters
    ----------
    scenario_provider : ScenarioLoaderLike
        Any object satisfying the ``ScenarioLoaderLike`` protocol (must
        expose ``name: str`` and ``avail_time_ratio: float``).
    available_time : float
        The time available for scheduling or execution.
    experiment : int
        The experiment identifier.
    t : int
        The specific step or part of the experiment being analyzed.
    policy : str
        The name of the policy being evaluated.
    reward_function : str
        The reward function used by the agent to observe the environment.
    metric : EvaluationMetric
        The evaluation metric containing experiment results.
    total_build_duration : int
        The total duration of the build process.
    prioritization_time : int
        The time spent on prioritizing the test cases.
    rewards : float
        The average reward from the prioritized test set.
    prioritization_order : list
        The order of prioritized test cases.
    execution_id : str or None
        Unique identifier for the execution instance.
    worker_id : str or None
        Logical worker identifier used in parallel runs.
    parallel_mode : str or None
        Execution mode label, for example ``sequential`` or ``process``.
    process_memory_rss_mib : float or None
        Current process resident memory in MiB.
    process_memory_peak_rss_mib : float or None
        Peak resident memory seen by the process in MiB.
    process_cpu_utilization_percent : float or None
        CPU utilization percentage measured for the latest interval.
    process_cpu_time_seconds : float or None
        Total process CPU time since the experiment started.
    wall_time_seconds : float or None
        Wall-clock time since the experiment started.
    variant : str or None
        The variant name (for HCS systems).  ``None`` for non-variant runs.
    """

    scenario_provider: ScenarioLoaderLike
    available_time: float
    experiment: int
    t: int
    policy: str
    reward_function: str
    metric: EvaluationMetric
    total_build_duration: int
    prioritization_time: int
    rewards: float
    prioritization_order: list[Any]
    execution_id: str | None = None
    worker_id: str | None = None
    parallel_mode: str | None = None
    process_memory_rss_mib: float | None = None
    process_memory_peak_rss_mib: float | None = None
    process_cpu_utilization_percent: float | None = None
    process_cpu_time_seconds: float | None = None
    wall_time_seconds: float | None = None
    variant: str | None = None
