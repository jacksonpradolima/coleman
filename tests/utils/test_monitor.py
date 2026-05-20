"""
Test cases for the MonitorCollector utility in the coleman package.

These tests cover functionality including data collection via the ResultsSink
and the new Parquet-based architecture.
"""

from unittest.mock import MagicMock

import pyarrow.parquet as pq
import pytest

from coleman.evaluation import EvaluationMetric
from coleman.results.parquet_sink import ParquetSink
from coleman.results.sink_base import NullSink
from coleman.utils.monitor import MonitorCollector
from coleman.utils.monitor_params import CollectParams


@pytest.fixture
def mock_metric():
    """Create a mock for the EvaluationMetric class."""
    metric = MagicMock(spec=EvaluationMetric)
    metric.detected_failures = 5
    metric.undetected_failures = 3
    metric.scheduled_testcases = ["test1", "test2", "test3"]
    metric.unscheduled_testcases = ["test4", "test5"]
    metric.ttf = 2
    metric.ttf_duration = 10
    metric.fitness = 0.8
    metric.cost = 0.6
    metric.avg_precision = 0.75
    return metric


@pytest.fixture
def mock_scenario_provider():
    """Create a mock for a scenario provider."""
    scenario_provider = MagicMock()
    scenario_provider.name = "TestScenario"
    scenario_provider.avail_time_ratio = 0.5
    return scenario_provider


@pytest.fixture
def null_monitor():
    """Create a MonitorCollector backed by a NullSink."""
    return MonitorCollector(sink=NullSink())


def _make_params(mock_scenario_provider, mock_metric, **overrides):
    """Build a CollectParams with sensible defaults."""
    params = CollectParams(
        scenario_provider=mock_scenario_provider,
        available_time=50,
        experiment=1,
        t=1,
        policy="TestPolicy",
        reward_function="TestReward",
        metric=mock_metric,
        total_build_duration=100,
        prioritization_time=10,
        rewards=0.9,
        prioritization_order=["test1", "test2"],
    )
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


def test_default_monitor_uses_null_sink():
    """MonitorCollector with no sink defaults to NullSink."""
    mc = MonitorCollector()
    assert isinstance(mc.sink, NullSink)


def test_collect_increments_counter(null_monitor, mock_scenario_provider, mock_metric):
    """Each collect() call increments rows_collected."""
    params = _make_params(mock_scenario_provider, mock_metric)
    null_monitor.collect(params)
    assert null_monitor.rows_collected == 1
    null_monitor.collect(params)
    assert null_monitor.rows_collected == 2


def test_clear_resets_counter(null_monitor, mock_scenario_provider, mock_metric):
    """clear() resets the rows_collected counter."""
    params = _make_params(mock_scenario_provider, mock_metric)
    null_monitor.collect(params)
    null_monitor.clear()
    assert null_monitor.rows_collected == 0


def test_collect_writes_to_parquet(tmp_path, mock_scenario_provider, mock_metric):
    """Collected data appears in Parquet files after flush."""
    sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
    mc = MonitorCollector(sink=sink)

    for i in range(5):
        params = _make_params(mock_scenario_provider, mock_metric, t=i)
        mc.collect(params)

    mc.flush()

    parquet_files = list(tmp_path.rglob("*.parquet"))
    assert len(parquet_files) >= 1
    total = sum(pq.read_table(f).num_rows for f in parquet_files)
    assert total == 5


def test_collect_auto_flushes_at_batch_size(tmp_path, mock_scenario_provider, mock_metric):
    """ParquetSink auto-flushes when batch_size is reached."""
    sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=5)
    mc = MonitorCollector(sink=sink)

    for i in range(6):
        params = _make_params(mock_scenario_provider, mock_metric, t=i)
        mc.collect(params)

    # At least one batch should have been flushed
    parquet_files = list(tmp_path.rglob("*.parquet"))
    assert len(parquet_files) >= 1


def test_flush_and_close(tmp_path, mock_scenario_provider, mock_metric):
    """flush() and close() persist buffered data."""
    sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=1000)
    mc = MonitorCollector(sink=sink)

    params = _make_params(mock_scenario_provider, mock_metric)
    mc.collect(params)
    mc.close()

    parquet_files = list(tmp_path.rglob("*.parquet"))
    assert len(parquet_files) >= 1


def test_null_sink_collect_does_not_fail(mock_scenario_provider, mock_metric):
    """NullSink-backed monitor silently discards data."""
    mc = MonitorCollector(sink=NullSink())
    for i in range(100):
        params = _make_params(mock_scenario_provider, mock_metric, t=i)
        mc.collect(params)
    mc.flush()
    mc.close()
    assert mc.rows_collected == 100


@pytest.mark.benchmark(group="monitor_collector")
@pytest.mark.parametrize("num_records", [1000, 10_000, 100_000])
def test_collect_performance(benchmark, num_records):
    """Benchmark collect() throughput with a NullSink."""
    mock_metric = MagicMock()
    mock_metric.detected_failures = 5
    mock_metric.undetected_failures = 3
    mock_metric.scheduled_testcases = [1, 2, 3]
    mock_metric.unscheduled_testcases = [4, 5]
    mock_metric.ttf = 1
    mock_metric.ttf_duration = 10
    mock_metric.fitness = 0.9
    mock_metric.cost = 0.8
    mock_metric.avg_precision = 0.85

    mock_scenario_provider = MagicMock()
    mock_scenario_provider.name = "BenchmarkScenario"
    mock_scenario_provider.avail_time_ratio = 0.5

    def add_records():
        collector = MonitorCollector(sink=NullSink())
        for i in range(num_records):
            params = CollectParams(
                scenario_provider=mock_scenario_provider,
                available_time=0.5,
                experiment=1,
                t=i,
                policy="PolicyBenchmark",
                reward_function="RewardBenchmark",
                metric=mock_metric,
                total_build_duration=100,
                prioritization_time=10,
                rewards=0.95,
                prioritization_order=[1, 2, 3],
            )
            collector.collect(params)
        return collector

    collector = benchmark(add_records)
    assert collector.rows_collected == num_records
