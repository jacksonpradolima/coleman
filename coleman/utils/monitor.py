"""
coleman.utils.monitor - Monitor Utilities.

This module provides tools for monitoring and collecting data during experiments related to
the Coleman framework. The primary functionality revolves around the `MonitorCollector`
class, which facilitates data collection during an experiment and writes results to
a configured ``ResultsSink``.

Classes
-------
MonitorCollector
    Collects data during an experiment and writes to a ResultsSink (NullSink by default;
    pass a ``ParquetSink`` or other sink explicitly to persist results).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coleman.results.sink_base import NullSink, ResultsSink
from coleman.utils.monitor_params import CollectParams

if TYPE_CHECKING:
    from typing import Any


class MonitorCollector:
    """Collect data during an experiment and write to a ResultsSink.

    When ``sink`` is ``None``, a ``NullSink`` (no-op) is used so data is
    silently discarded.  Pass a ``ParquetSink`` (or any other
    ``ResultsSink``) explicitly to persist results.

    Parameters
    ----------
    sink : ResultsSink or None
        The results sink to write to.  When ``None`` a ``NullSink`` is used.

    Attributes
    ----------
    sink : ResultsSink
        The active results sink.
    rows_collected : int
        Total number of rows collected since last ``clear()``.

    Notes
    -----
    Schema columns written to the sink:

    - ``scenario``: Experiment name (system under test).
    - ``experiment``: Experiment number.
    - ``step``: Part number (Build) from scenario that is being analyzed.
    - ``policy``: Policy name that is evaluating a part of the scenario.
    - ``reward_function``: Reward function used by the agent to observe the environment.
    - ``sched_time``: Percentage of time available (i.e., 50 % of total for the Build).
    - ``sched_time_duration``: The time in number obtained from percentage.
    - ``total_build_duration``: Build Duration.
    - ``prioritization_time``: Prioritization Time.
    - ``detected``: Failures detected.
    - ``missed``: Failures missed.
    - ``tests_ran``: Number of tests executed.
    - ``tests_not_ran``: Number of tests not executed.
    - ``ttf``: Rank of the Time to Fail (Order of the first test case which failed).
    - ``ttf_duration``: Time spent until the first test case fail.
    - ``time_reduction``: Time Reduction (Total Time for the Build - ttf_duration).
    - ``fitness``: Evaluation metric result (example, NAPFD).
    - ``cost``: Evaluation metric that considers cost, for instance, APFDc.
    - ``rewards``: AVG Reward from the prioritized test set.
    - ``avg_precision``: 1 - We found all failures, 123 - We did not find all failures.
    - ``prioritization_order``: Prioritized test set (stored as hash + optional top-k in Parquet).
    - ``variant``: Variant name for HCS systems (``None`` for non-variant runs).
    """

    def __init__(self, sink: ResultsSink | None = None) -> None:
        """Initialize the MonitorCollector with a results sink.

        Parameters
        ----------
        sink : ResultsSink or None
            Target sink.  Defaults to ``NullSink()`` when ``None``.
        """
        self.sink: ResultsSink = sink if sink is not None else NullSink()
        self.rows_collected: int = 0

    def collect(self, params: CollectParams) -> None:
        """Collect the feedback of an analysis and write to the sink.

        Parameters
        ----------
        params : CollectParams
            CollectParams object containing all input data.
        """
        row: dict[str, Any] = {
            "scenario": params.scenario_provider.name,
            "experiment": params.experiment,
            "step": params.t,
            "execution_id": params.execution_id,
            "worker_id": params.worker_id,
            "parallel_mode": params.parallel_mode,
            "policy": params.policy,
            "reward_function": params.reward_function,
            "sched_time": params.scenario_provider.avail_time_ratio,
            "sched_time_duration": params.available_time,
            "total_build_duration": params.total_build_duration,
            "prioritization_time": params.prioritization_time,
            "process_memory_rss_mib": params.process_memory_rss_mib,
            "process_memory_peak_rss_mib": params.process_memory_peak_rss_mib,
            "process_cpu_utilization_percent": params.process_cpu_utilization_percent,
            "process_cpu_time_seconds": params.process_cpu_time_seconds,
            "wall_time_seconds": params.wall_time_seconds,
            "detected": params.metric.detected_failures,
            "missed": params.metric.undetected_failures,
            "tests_ran": len(params.metric.scheduled_testcases),
            "tests_not_ran": len(params.metric.unscheduled_testcases),
            "ttf": params.metric.ttf,
            "ttf_duration": params.metric.ttf_duration,
            "time_reduction": params.total_build_duration - params.metric.ttf_duration,
            "fitness": params.metric.fitness,
            "cost": params.metric.cost,
            "rewards": params.rewards,
            "avg_precision": params.metric.avg_precision,
            "prioritization_order": params.prioritization_order,
            "variant": params.variant,
        }

        self.sink.write_row(row)
        self.rows_collected += 1

    def flush(self) -> None:
        """Force-flush any buffered data to the underlying sink."""
        self.sink.flush()

    def close(self) -> None:
        """Flush and release sink resources."""
        self.sink.close()

    def clear(self) -> None:
        """Reset the row counter (sink data is already persisted)."""
        self.rows_collected = 0
