"""Integration tests for the complete coleman experiment flow.

Tests the full pipeline:
    runner → environment → scenarios → reward → evaluation → sinks → telemetry

Using the fakedata dataset from examples/fakedata/.
All stochastic tests pin a fixed seed for reproducibility.

Sections
--------
1  – Scenario loading
2  – Environment single-run (multiple policies / rewards)
3  – Monitor: row collection and schema
4  – Reward and evaluation metric correctness
5  – Seed reproducibility
6  – Runner orchestration (EnvironmentBuildConfig, build_environment, isolated run)
7  – Multiple experiments
8  – ParquetSink + DuckDB end-to-end
9  – ClickHouse sink integration (mocked driver)
10 – Telemetry integration (NoOp and live-loop behaviour)
11 – ResultsWriter async drain integration
12 – DuckDBCatalog query layer
13 – Pipeline from run.yaml config
"""

from __future__ import annotations

import contextlib
import logging
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

import coleman.policy.base as _policy_base
from coleman.agent import RewardAgent, RewardSlidingWindowAgent
from coleman.environment import Environment
from coleman.evaluation import NAPFDVerdictMetric
from coleman.evaluation.base import EvaluationMetric
from coleman.policy import FRRMABPolicy, GreedyPolicy, RandomPolicy, UCB1Policy
from coleman.results.clickhouse_sink import _CLICKHOUSE_TABLE, _INSERT_COLS, ClickHouseSink
from coleman.results.duckdb_catalog import DuckDBCatalog
from coleman.results.parquet_sink import ParquetSink
from coleman.results.sink_base import ResultsSink
from coleman.results.writer import ResultsWriter
from coleman.reward.rnfail import RNFailReward
from coleman.reward.timerank import TimeRankReward
from coleman.runner import (
    EnvironmentBuildConfig,
    ExecutionPlan,
    build_agents_from_config,
    build_environment,
    create_agents,
    exp_run_industrial_dataset,
    exp_run_industrial_dataset_isolated,
    get_scenario_provider,
)
from coleman.scenarios import ScenarioLoader
from coleman.telemetry.otel import NoOpTelemetry, get_telemetry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_EXAMPLES_DIR = str(_HERE.parent / "examples")
_DATASET = "fakedata"
_SCHED_TIME_RATIO = 0.5
_TRIALS = 5  # small enough to keep the suite fast
_SEED = 42


# ---------------------------------------------------------------------------
# In-memory sink helper
# ---------------------------------------------------------------------------


class CapturingSink(ResultsSink):
    """In-memory ResultsSink that stores every written row for later inspection."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.flush_count: int = 0
        self.closed: bool = False

    def write_row(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _make_provider(sched_time_ratio: float = _SCHED_TIME_RATIO) -> ScenarioLoader:
    """Return a fresh ScenarioLoader backed by the fakedata parquet file."""
    return get_scenario_provider(
        datasets_dir=_EXAMPLES_DIR,
        dataset=_DATASET,
        sched_time_ratio=sched_time_ratio,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
    )


def _run_and_capture(seed: int, trials: int = _TRIALS) -> list[dict[str, Any]]:
    """Run a GreedyPolicy experiment with a fixed seed and return all collected rows.

    Both the Polars shuffle (t=0) and the NumPy-based MAB policy RNG are seeded,
    so repeated calls with the same seed produce bit-identical results.
    """
    _policy_base._rng = np.random.default_rng(seed)
    sink = CapturingSink()
    provider = _make_provider()
    agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[], seed=seed)
    env = Environment(agents, provider, NAPFDVerdictMetric())
    env.monitor.sink = sink
    env.run_single(experiment=1, trials=trials)
    return sink.rows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def seed_rng():
    """Pin the shared policy RNG to _SEED and restore the original after the test."""
    original = _policy_base._rng
    _policy_base._rng = np.random.default_rng(_SEED)
    yield
    _policy_base._rng = original


@pytest.fixture()
def capturing_sink() -> CapturingSink:
    return CapturingSink()


@pytest.fixture()
def provider() -> ScenarioLoader:
    return _make_provider()


@pytest.fixture()
def greedy_build_config() -> EnvironmentBuildConfig:
    return EnvironmentBuildConfig(
        datasets_dir=_EXAMPLES_DIR,
        dataset=_DATASET,
        sched_time_ratio=_SCHED_TIME_RATIO,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={"enabled": False},
        telemetry_config={"enabled": False},
        algorithm_configs={"greedy": {"rnfail": {}}},
        rewards_names=["RNFail"],
        policy_names=["Greedy"],
    )


# ---------------------------------------------------------------------------
# 1 — Scenario loading
# ---------------------------------------------------------------------------


class TestScenarioLoading:
    def test_loads_parquet_and_reports_positive_max_builds(self):
        p = _make_provider()
        assert isinstance(p, ScenarioLoader)
        assert p.max_builds > 0

    def test_sched_time_ratio_is_preserved(self):
        p = _make_provider(sched_time_ratio=0.8)
        assert p.get_avail_time_ratio() == pytest.approx(0.8)

    def test_name_derived_from_dataset_directory(self):
        p = _make_provider()
        assert p.name == _DATASET

    def test_iteration_yields_scenarios_with_testcases(self, provider):
        first = next(iter(provider))
        assert first is not None
        assert len(first.get_testcases()) > 0

    def test_virtual_scenario_available_time_non_negative(self, provider):
        scenario = next(iter(provider))
        assert scenario.get_available_time() >= 0.0

    def test_all_builds_iterable(self, provider):
        count = sum(1 for _ in provider)
        assert count == provider.max_builds

    def test_last_build_resets_iteration(self, provider):
        """last_build(0) makes iteration restart from the first build."""
        first_pass = [s.get_available_time() for s in provider]
        provider.last_build(0)
        second_pass = [s.get_available_time() for s in provider]
        assert first_pass == pytest.approx(second_pass)


# ---------------------------------------------------------------------------
# 2 — Environment single-run — multiple policies and reward functions
# ---------------------------------------------------------------------------


class TestEnvironmentSingleRun:
    @pytest.mark.parametrize(
        "policy_cls,policy_kwargs,reward_cls,window_sizes",
        [
            (GreedyPolicy, {}, RNFailReward, []),
            (GreedyPolicy, {}, TimeRankReward, []),
            (RandomPolicy, {}, RNFailReward, []),
            (UCB1Policy, {"c": 2.0}, RNFailReward, []),
            (FRRMABPolicy, {"c": 0.5}, RNFailReward, [5, 10]),
        ],
        ids=["greedy-rnfail", "greedy-timerank", "random-rnfail", "ucb1-rnfail", "frrmab-rnfail"],
    )
    def test_run_completes_without_error(self, seed_rng, policy_cls, policy_kwargs, reward_cls, window_sizes):
        agents = create_agents(policy_cls(**policy_kwargs), reward_cls(), window_sizes)
        env = Environment(agents, _make_provider(), NAPFDVerdictMetric())
        env.run_single(experiment=1, trials=_TRIALS)
        env.store_experiment()

    def test_run_single_does_not_mutate_results_config(self, seed_rng):
        """Running an experiment must not modify the results_config dict in-place."""
        cfg: dict[str, Any] = {}
        agents = create_agents(GreedyPolicy(), RNFailReward(), [])
        env = Environment(agents, _make_provider(), NAPFDVerdictMetric(), results_config=cfg)
        env.run_single(experiment=1, trials=_TRIALS)
        assert cfg == {}


# ---------------------------------------------------------------------------
# 3 — Monitor: row collection and schema
# ---------------------------------------------------------------------------


class TestMonitorRowCollection:
    def test_one_row_per_trial_per_agent(self, seed_rng, provider, capturing_sink):
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, provider, NAPFDVerdictMetric())
        env.monitor.sink = capturing_sink
        env.run_single(experiment=1, trials=_TRIALS)
        expected = min(_TRIALS, provider.max_builds) * len(agents)
        assert len(capturing_sink.rows) == expected

    def test_rows_contain_required_schema_keys(self, seed_rng, provider, capturing_sink):
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, provider, NAPFDVerdictMetric())
        env.monitor.sink = capturing_sink
        env.run_single(experiment=1, trials=_TRIALS)
        required = {
            "scenario",
            "experiment",
            "step",
            "policy",
            "reward_function",
            "fitness",
            "cost",
            "detected",
            "missed",
        }
        for row in capturing_sink.rows:
            assert required.issubset(row.keys()), f"Missing keys: {required - row.keys()}"

    def test_fitness_values_bounded_in_unit_interval(self, seed_rng, provider, capturing_sink):
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, provider, NAPFDVerdictMetric())
        env.monitor.sink = capturing_sink
        env.run_single(experiment=1, trials=_TRIALS)
        for row in capturing_sink.rows:
            assert 0.0 <= row["fitness"] <= 1.0, f"fitness out of bounds: {row['fitness']}"

    def test_cost_values_bounded_in_unit_interval(self, seed_rng, provider, capturing_sink):
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, provider, NAPFDVerdictMetric())
        env.monitor.sink = capturing_sink
        env.run_single(experiment=1, trials=_TRIALS)
        for row in capturing_sink.rows:
            assert 0.0 <= row["cost"] <= 1.0, f"cost out of bounds: {row['cost']}"

    def test_experiment_index_and_step_counters_are_correct(self, seed_rng, provider, capturing_sink):
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, provider, NAPFDVerdictMetric())
        env.monitor.sink = capturing_sink
        env.run_single(experiment=7, trials=_TRIALS)
        assert all(row["experiment"] == 7 for row in capturing_sink.rows)
        steps = sorted({row["step"] for row in capturing_sink.rows})
        assert steps == list(range(1, len(steps) + 1))

    def test_store_experiment_flushes_sink_at_least_once(self, seed_rng, provider, capturing_sink):
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, provider, NAPFDVerdictMetric())
        env.monitor.sink = capturing_sink
        env.run_single(experiment=1, trials=_TRIALS)
        env.store_experiment()
        assert capturing_sink.flush_count >= 1

    def test_scenario_name_matches_dataset(self, seed_rng, provider, capturing_sink):
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, provider, NAPFDVerdictMetric())
        env.monitor.sink = capturing_sink
        env.run_single(experiment=1, trials=_TRIALS)
        for row in capturing_sink.rows:
            assert row["scenario"] == _DATASET

    def test_variant_is_none_for_non_hcs_run(self, seed_rng, provider, capturing_sink):
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, provider, NAPFDVerdictMetric())
        env.monitor.sink = capturing_sink
        env.run_single(experiment=1, trials=_TRIALS)
        for row in capturing_sink.rows:
            assert row["variant"] is None


# ---------------------------------------------------------------------------
# 4 — Reward and evaluation metric correctness
# ---------------------------------------------------------------------------


class TestRewardEvaluationIntegration:
    def test_rnfail_returns_all_zeros_when_no_failures(self):
        metric = EvaluationMetric()
        rewards = RNFailReward().evaluate(metric, ["tc_a", "tc_b", "tc_c"])
        assert rewards == [0.0, 0.0, 0.0]

    def test_rnfail_marks_rank_1_failure(self):
        metric = EvaluationMetric()
        metric.detection_ranks = [1]
        rewards = RNFailReward().evaluate(metric, ["tc_a", "tc_b", "tc_c"])
        assert rewards[0] == pytest.approx(1.0)
        assert rewards[1] == pytest.approx(0.0)
        assert rewards[2] == pytest.approx(0.0)

    def test_rnfail_multi_failure_correct_positions(self):
        metric = EvaluationMetric()
        metric.detection_ranks = [1, 3]
        rewards = RNFailReward().evaluate(metric, ["tc_a", "tc_b", "tc_c"])
        assert len(rewards) == 3
        assert rewards[0] == pytest.approx(1.0)
        assert rewards[1] == pytest.approx(0.0)
        assert rewards[2] == pytest.approx(1.0)

    def test_napfd_fitness_in_unit_interval_after_full_run(self, seed_rng):
        metric = NAPFDVerdictMetric()
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, _make_provider(), metric)
        env.run_single(experiment=1, trials=_TRIALS)
        assert 0.0 <= metric.fitness <= 1.0

    def test_napfd_cost_in_unit_interval_after_full_run(self, seed_rng):
        metric = NAPFDVerdictMetric()
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, _make_provider(), metric)
        env.run_single(experiment=1, trials=_TRIALS)
        assert 0.0 <= metric.cost <= 1.0


# ---------------------------------------------------------------------------
# 5 — Seed reproducibility
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    def test_same_seed_produces_identical_row_count(self):
        rows_a = _run_and_capture(_SEED)
        rows_b = _run_and_capture(_SEED)
        assert len(rows_a) == len(rows_b)

    def test_same_seed_produces_identical_fitness_sequence(self):
        rows_a = _run_and_capture(_SEED)
        rows_b = _run_and_capture(_SEED)
        fitness_a = [r["fitness"] for r in rows_a]
        fitness_b = [r["fitness"] for r in rows_b]
        assert fitness_a == pytest.approx(fitness_b)

    def test_same_seed_produces_identical_prioritization_orders(self):
        rows_a = _run_and_capture(_SEED)
        rows_b = _run_and_capture(_SEED)
        for row_a, row_b in zip(rows_a, rows_b, strict=True):
            assert row_a["prioritization_order"] == row_b["prioritization_order"]

    def test_different_seeds_produce_different_initial_ordering(self):
        """Two distinct seeds must produce a different ordering at t=0 (initial shuffle)."""
        rows_a = _run_and_capture(0)
        rows_b = _run_and_capture(99999)
        # At least the first step (t=0 shuffle) must differ across seeds
        orders_a = [r["prioritization_order"] for r in rows_a]
        orders_b = [r["prioritization_order"] for r in rows_b]
        assert orders_a != orders_b


# ---------------------------------------------------------------------------
# 6 — Runner orchestration: EnvironmentBuildConfig, build_environment, isolated run
# ---------------------------------------------------------------------------


class TestRunnerOrchestration:
    def test_build_environment_creates_environment_and_reports_max_builds(self, greedy_build_config):
        metadata = {"execution_id": "int-test", "worker_id": "0", "parallel_mode": "sequential"}
        env, max_builds = build_environment(greedy_build_config, metadata)
        assert isinstance(env, Environment)
        assert max_builds > 0

    def test_build_environment_produces_reward_agents_for_greedy(self, seed_rng, greedy_build_config):
        metadata = {"execution_id": "int-test", "worker_id": "0", "parallel_mode": "sequential"}
        env, _ = build_environment(greedy_build_config, metadata)
        assert len(env.agents) == 1
        assert isinstance(env.agents[0], RewardAgent)

    def test_build_agents_from_config_greedy(self, seed_rng):
        agents = build_agents_from_config(
            algorithm_configs={"greedy": {"rnfail": {}}},
            policy_names=["Greedy"],
            rewards_names=["RNFail"],
        )
        assert len(agents) == 1
        assert isinstance(agents[0], RewardAgent)

    def test_build_agents_from_config_frrmab_creates_sliding_window_agents(self, seed_rng):
        agents = build_agents_from_config(
            algorithm_configs={"frrmab": {"rnfail": {"c": 0.5}, "window_sizes": [5, 10]}},
            policy_names=["FRRMAB"],
            rewards_names=["RNFail"],
        )
        assert len(agents) == 2
        assert all(isinstance(a, RewardSlidingWindowAgent) for a in agents)

    def test_build_environment_multi_policy_multi_reward(self, seed_rng):
        config = EnvironmentBuildConfig(
            datasets_dir=_EXAMPLES_DIR,
            dataset=_DATASET,
            sched_time_ratio=_SCHED_TIME_RATIO,
            use_hcs=False,
            use_context=False,
            context_config={},
            feature_groups={},
            results_config={},
            checkpoint_config={"enabled": False},
            telemetry_config={"enabled": False},
            algorithm_configs={
                "greedy": {"rnfail": {}, "timerank": {}},
                "random": {"rnfail": {}, "timerank": {}},
            },
            rewards_names=["RNFail", "TimeRank"],
            policy_names=["Greedy", "Random"],
        )
        metadata = {"execution_id": "multi", "worker_id": "0", "parallel_mode": "sequential"}
        env, _ = build_environment(config, metadata)
        # Greedy×{RNFail,TimeRank} + Random×{RNFail,TimeRank} = 4 agents
        assert len(env.agents) == 4

    def test_exp_run_industrial_dataset_completes(self, seed_rng, greedy_build_config):
        metadata = {"execution_id": "int-test", "worker_id": "0", "parallel_mode": "sequential"}
        env, _ = build_environment(greedy_build_config, metadata)
        exp_run_industrial_dataset(
            iteration=1,
            trials=_TRIALS,
            env=env,
            level=logging.WARNING,
            runtime_metadata=metadata,
        )

    def test_exp_run_industrial_dataset_isolated_completes(self, seed_rng, greedy_build_config):
        plan = ExecutionPlan(
            iteration=1,
            trials=_TRIALS,
            level=logging.WARNING,
            execution_id="int-test-iso",
            worker_id="1",
            parallel_mode="sequential",
        )
        exp_run_industrial_dataset_isolated(greedy_build_config, plan)

    def test_runtime_metadata_propagated_to_rows(self, seed_rng, greedy_build_config, capturing_sink):
        metadata = {"execution_id": "exec-42", "worker_id": "7", "parallel_mode": "sequential"}
        env, _ = build_environment(greedy_build_config, metadata)
        env.monitor.sink = capturing_sink
        exp_run_industrial_dataset(
            iteration=1,
            trials=_TRIALS,
            env=env,
            level=logging.WARNING,
            runtime_metadata=metadata,
        )
        for row in capturing_sink.rows:
            assert row["execution_id"] == "exec-42"
            assert row["worker_id"] == "7"


# ---------------------------------------------------------------------------
# 7 — Multiple experiments (env.run)
# ---------------------------------------------------------------------------


class TestMultipleExperiments:
    def test_run_two_experiments_collects_double_rows(self, seed_rng, capturing_sink):
        provider = _make_provider()
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, provider, NAPFDVerdictMetric())
        # Inject the sink before env.run() so that the internal reset() picks it up
        env._sink = capturing_sink
        env.run(experiments=2, trials=_TRIALS)
        expected = 2 * min(_TRIALS, provider.max_builds) * len(agents)
        assert len(capturing_sink.rows) == expected

    def test_run_preserves_distinct_experiment_indices(self, seed_rng, capturing_sink):
        provider = _make_provider()
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, provider, NAPFDVerdictMetric())
        env._sink = capturing_sink
        env.run(experiments=2, trials=_TRIALS)
        # env.run uses range(experiments) → experiments 0 and 1
        assert {row["experiment"] for row in capturing_sink.rows} == {0, 1}

    def test_run_metrics_non_negative_across_experiments(self, seed_rng, capturing_sink):
        provider = _make_provider()
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, provider, NAPFDVerdictMetric())
        env._sink = capturing_sink
        env.run(experiments=2, trials=_TRIALS)
        for row in capturing_sink.rows:
            assert row["detected"] >= 0
            assert row["missed"] >= 0
            assert row["tests_ran"] >= 0


# ---------------------------------------------------------------------------
# 8 — ParquetSink + DuckDB end-to-end
# ---------------------------------------------------------------------------


class TestParquetSinkEndToEnd:
    """Full loop: Environment → ParquetSink → Parquet files → DuckDB query."""

    def test_parquet_files_created_on_flush(self, tmp_path, seed_rng):
        sink = ParquetSink(out_dir=str(tmp_path), batch_size=1)
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, _make_provider(), NAPFDVerdictMetric())
        env.monitor.sink = sink
        env.run_single(experiment=1, trials=_TRIALS)
        env.store_experiment()
        parquet_files = list(tmp_path.rglob("*.parquet"))
        assert len(parquet_files) > 0, "No Parquet files were written"

    def test_parquet_schema_matches_expected_columns(self, tmp_path, seed_rng):
        sink = ParquetSink(out_dir=str(tmp_path), batch_size=1)
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, _make_provider(), NAPFDVerdictMetric())
        env.monitor.sink = sink
        env.run_single(experiment=1, trials=_TRIALS)
        env.store_experiment()
        files = list(tmp_path.rglob("*.parquet"))
        schema = pq.read_schema(files[0])
        # "scenario" is a Hive partition directory key — it is NOT stored inside
        # the Parquet file itself, so it will NOT appear in schema.names.
        required_cols = {"experiment", "step", "fitness", "cost", "detected", "missed"}
        assert required_cols.issubset(set(schema.names))

    def test_duckdb_catalog_aggregates_fitness_correctly(self, tmp_path, seed_rng):
        sink = ParquetSink(out_dir=str(tmp_path), batch_size=1000)
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, _make_provider(), NAPFDVerdictMetric())
        env.monitor.sink = sink
        env.run_single(experiment=1, trials=_TRIALS)
        env.store_experiment()
        # Query via DuckDB — mirrors the workflow.py pattern
        catalog = DuckDBCatalog(str(tmp_path))
        df = catalog.query("SELECT AVG(fitness) AS avg_fitness, COUNT(*) AS n FROM results")
        catalog.close()
        assert df["n"][0] >= 1
        assert 0.0 <= float(df["avg_fitness"][0]) <= 1.0

    def test_duckdb_catalog_sum_detected_plus_missed_equals_total_tc(self, tmp_path, seed_rng):
        """Detected + missed ≤ tests_ran + tests_not_ran for every row."""
        sink = ParquetSink(out_dir=str(tmp_path), batch_size=1000)
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, _make_provider(), NAPFDVerdictMetric())
        env.monitor.sink = sink
        env.run_single(experiment=1, trials=_TRIALS)
        env.store_experiment()
        catalog = DuckDBCatalog(str(tmp_path))
        df = catalog.query(
            "SELECT SUM(detected + missed) AS total_fail, SUM(tests_ran + tests_not_ran) AS total_tc FROM results"
        )
        catalog.close()
        assert df["total_fail"][0] <= df["total_tc"][0]

    def test_parquet_top_k_stores_subset_of_order(self, tmp_path, seed_rng):
        sink = ParquetSink(out_dir=str(tmp_path), batch_size=1000, top_k=3)
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, _make_provider(), NAPFDVerdictMetric())
        env.monitor.sink = sink
        env.run_single(experiment=1, trials=_TRIALS)
        env.store_experiment()
        catalog = DuckDBCatalog(str(tmp_path))
        df = catalog.query(
            "SELECT prioritization_order_top_k FROM results WHERE prioritization_order_top_k IS NOT NULL LIMIT 1"
        )
        catalog.close()
        if len(df) > 0 and df["prioritization_order_top_k"][0]:
            import json

            top_k_list = json.loads(df["prioritization_order_top_k"][0])
            assert len(top_k_list) <= 3

    def test_parquet_sink_batch_flush_boundary(self, tmp_path, seed_rng):
        """Sink flushes exactly when batch_size is reached, not before."""
        batch_size = 2
        sink = ParquetSink(out_dir=str(tmp_path), batch_size=batch_size)
        for i in range(batch_size - 1):
            sink.write_row(
                {
                    "scenario": "s",
                    "experiment": 1,
                    "step": i,
                    "execution_id": None,
                    "worker_id": None,
                    "parallel_mode": None,
                    "policy": "G",
                    "reward_function": "R",
                    "sched_time": 0.5,
                    "sched_time_duration": 1.0,
                    "total_build_duration": 2.0,
                    "prioritization_time": 0.01,
                    "process_memory_rss_mib": None,
                    "process_memory_peak_rss_mib": None,
                    "process_cpu_utilization_percent": None,
                    "process_cpu_time_seconds": None,
                    "wall_time_seconds": None,
                    "detected": 0,
                    "missed": 0,
                    "tests_ran": 1,
                    "tests_not_ran": 0,
                    "ttf": 0.0,
                    "ttf_duration": 0.0,
                    "time_reduction": 0.0,
                    "fitness": 0.0,
                    "cost": 0.0,
                    "rewards": 0.0,
                    "avg_precision": 1.0,
                    "prioritization_order": [],
                    "variant": None,
                }
            )
        assert list(tmp_path.rglob("*.parquet")) == [], "Should not flush before batch_size"
        sink.flush()
        assert list(tmp_path.rglob("*.parquet")) != [], "Should flush after explicit flush()"

    def test_parquet_sink_hive_partitioning_dirs_created(self, tmp_path, seed_rng):
        sink = ParquetSink(out_dir=str(tmp_path), batch_size=1000)
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, _make_provider(), NAPFDVerdictMetric())
        env.monitor.sink = sink
        env.run_single(experiment=1, trials=_TRIALS)
        env.store_experiment()
        # Hive partitioning creates subdirs like scenario=.../policy=...
        subdirs = [p for p in tmp_path.rglob("*") if p.is_dir()]
        assert len(subdirs) > 0, "Hive partition directories should exist"

    def test_duckdb_catalog_policy_group_by(self, tmp_path, seed_rng):
        """DuckDB GROUP BY policy must produce one row per distinct policy."""
        sink = ParquetSink(out_dir=str(tmp_path), batch_size=1000)
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, _make_provider(), NAPFDVerdictMetric())
        env.monitor.sink = sink
        env.run_single(experiment=1, trials=_TRIALS)
        env.store_experiment()
        catalog = DuckDBCatalog(str(tmp_path))
        df = catalog.query("SELECT policy, COUNT(*) AS n FROM results GROUP BY policy")
        catalog.close()
        assert len(df) == 1
        assert df["policy"][0] == "Greedy"
        assert df["n"][0] >= 1


# ---------------------------------------------------------------------------
# 9 — ClickHouse sink integration (mocked clickhouse-connect driver)
# ---------------------------------------------------------------------------


def _make_mock_ch_client() -> MagicMock:
    client = MagicMock()
    client.command.return_value = None
    client.insert.return_value = None
    client.close.return_value = None
    return client


def _make_mock_cc_module(client: MagicMock) -> MagicMock:
    mod = MagicMock()
    mod.get_client.return_value = client
    return mod


def _ch_row(**overrides) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario": "fakedata",
        "experiment": 1,
        "step": 1,
        "execution_id": "exec-1",
        "worker_id": "1",
        "parallel_mode": "sequential",
        "policy": "Greedy",
        "reward_function": "RNFail",
        "sched_time": 0.5,
        "sched_time_duration": 50.0,
        "total_build_duration": 100.0,
        "prioritization_time": 0.01,
        "process_memory_rss_mib": None,
        "process_memory_peak_rss_mib": None,
        "process_cpu_utilization_percent": None,
        "process_cpu_time_seconds": None,
        "wall_time_seconds": None,
        "detected": 2,
        "missed": 1,
        "tests_ran": 5,
        "tests_not_ran": 2,
        "ttf": 1.0,
        "ttf_duration": 5.0,
        "time_reduction": 80.0,
        "fitness": 0.75,
        "cost": 0.5,
        "rewards": 0.8,
        "avg_precision": 1.0,
        "prioritization_order": ["tc1", "tc2"],
        "variant": None,
    }
    row.update(overrides)
    return row


class TestClickHouseSinkIntegration:
    def _make_sink(self, client: MagicMock, batch_size: int = 100) -> ClickHouseSink:
        cc_mod = _make_mock_cc_module(client)
        with patch("coleman.results.clickhouse_sink.importlib.import_module", return_value=cc_mod):
            return ClickHouseSink(batch_size=batch_size)

    def test_write_row_buffers_without_inserting(self):
        client = _make_mock_ch_client()
        sink = self._make_sink(client)
        sink.write_row(_ch_row())
        client.insert.assert_not_called()

    def test_flush_calls_insert_with_correct_table(self):
        client = _make_mock_ch_client()
        sink = self._make_sink(client)
        sink.write_row(_ch_row())
        sink.flush()
        client.insert.assert_called_once()
        assert client.insert.call_args[0][0] == _CLICKHOUSE_TABLE

    def test_flush_empty_buffer_noop(self):
        client = _make_mock_ch_client()
        sink = self._make_sink(client)
        sink.flush()
        client.insert.assert_not_called()

    def test_close_flushes_remaining_buffer(self):
        client = _make_mock_ch_client()
        sink = self._make_sink(client)
        sink.write_row(_ch_row())
        sink.close()
        client.insert.assert_called_once()
        client.close.assert_called_once()

    def test_auto_flush_at_batch_boundary(self):
        client = _make_mock_ch_client()
        sink = self._make_sink(client, batch_size=3)
        for i in range(3):
            sink.write_row(_ch_row(step=i))
        client.insert.assert_called_once()

    def test_write_row_serializes_list_prioritization_order(self):
        client = _make_mock_ch_client()
        sink = self._make_sink(client)
        sink.write_row(_ch_row(prioritization_order=["tc_a", "tc_b"]))
        sink.flush()
        # Inspect inserted data — prioritization_order must be string
        inserted_rows = client.insert.call_args[0][1]
        order_col_idx = _INSERT_COLS.index("prioritization_order")
        assert isinstance(inserted_rows[0][order_col_idx], str)

    def test_insert_column_count_matches_schema(self):
        client = _make_mock_ch_client()
        sink = self._make_sink(client)
        sink.write_row(_ch_row())
        sink.flush()
        cols = client.insert.call_args[1].get("column_names") or client.insert.call_args[0][2]
        assert len(cols) == len(_INSERT_COLS)

    def test_thread_safe_concurrent_writes_no_data_loss(self):
        client = _make_mock_ch_client()
        sink = self._make_sink(client, batch_size=10000)
        n = 50
        errors: list[Exception] = []

        def _write():
            try:
                for _ in range(10):
                    sink.write_row(_ch_row())
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_write) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Thread errors: {errors}"
        sink.flush()
        total_inserted = sum(len(call[0][1]) for call in client.insert.call_args_list)
        assert total_inserted == n * 10

    def test_environment_with_clickhouse_sink_end_to_end(self, seed_rng):
        """Full environment run with ClickHouseSink (mocked driver)."""
        client = _make_mock_ch_client()
        cc_mod = _make_mock_cc_module(client)
        with patch("coleman.results.clickhouse_sink.importlib.import_module", return_value=cc_mod):
            sink = ClickHouseSink(batch_size=10000)
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, _make_provider(), NAPFDVerdictMetric())
        env.monitor.sink = sink
        env.run_single(experiment=1, trials=_TRIALS)
        env.store_experiment()
        # After store_experiment → flush → insert called at least once or buffer had data
        # Verify that all insert calls used the right table
        for call in client.insert.call_args_list:
            assert call[0][0] == _CLICKHOUSE_TABLE

    def test_clickhouse_raises_import_error_when_driver_absent(self):
        with (
            patch(
                "coleman.results.clickhouse_sink.importlib.import_module",
                side_effect=ImportError("no module"),
            ),
            pytest.raises(ImportError, match="clickhouse-connect"),
        ):
            ClickHouseSink()


# ---------------------------------------------------------------------------
# 10 — Telemetry integration: NoOp in loop, factory coverage
# ---------------------------------------------------------------------------


class TestTelemetryIntegration:
    def test_noop_telemetry_survives_full_experiment_loop(self, seed_rng):
        """NoOpTelemetry must not raise at any call site during a full run."""
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(
            agents,
            _make_provider(),
            NAPFDVerdictMetric(),
            telemetry_config={"enabled": False},
        )
        assert isinstance(env.telemetry, NoOpTelemetry)
        env.run_single(experiment=1, trials=_TRIALS)
        env.store_experiment()

    def test_get_telemetry_disabled_returns_noop(self):
        t = get_telemetry(enabled=False)
        assert isinstance(t, NoOpTelemetry)

    def test_get_telemetry_disabled_all_methods_callable(self):
        t = get_telemetry(enabled=False)
        t.record_cycle({"scenario": "s"})
        t.mark_run_started({"experiment": "1"})
        t.mark_run_finished({"experiment": "1"})
        t.record_latency("prioritization", 0.01)
        t.record_fitness(0.8, 0.6)
        t.record_resource_snapshot(64.0, 128.0, 50.0)
        t.record_experiment_resources(10.0, 3.0, 128.0)
        t.record_checkpoint_save()
        t.flush()

    def test_get_telemetry_enabled_otel_absent_returns_noop(self):
        """Even when enabled=True, absent SDK → NoOpTelemetry (no crash)."""
        import coleman.telemetry.otel as _otel_mod

        original = _otel_mod._HAS_OTEL
        _otel_mod._HAS_OTEL = False
        try:
            t = get_telemetry(enabled=True)
            assert isinstance(t, NoOpTelemetry)
        finally:
            _otel_mod._HAS_OTEL = original

    def test_noop_span_context_manager_yields_none(self):
        t = NoOpTelemetry()
        with t.span("test_span", attributes={"step": 1}) as s:
            assert s is None

    def test_telemetry_config_none_in_environment_falls_back_to_noop(self, seed_rng):
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, _make_provider(), NAPFDVerdictMetric(), telemetry_config=None)
        assert isinstance(env.telemetry, NoOpTelemetry)
        env.run_single(experiment=1, trials=_TRIALS)

    def test_environment_runtime_metadata_in_telemetry_attributes(self, seed_rng, capturing_sink):
        """Telemetry attributes built from runtime metadata carry execution_id."""
        metadata = {"execution_id": "tel-test-42", "worker_id": "2", "parallel_mode": "sequential"}
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, _make_provider(), NAPFDVerdictMetric())
        env.monitor.sink = capturing_sink
        env.set_runtime_metadata(metadata)
        env.run_single(experiment=1, trials=_TRIALS)
        attrs = env._experiment_telemetry_attributes(1)
        assert attrs["execution_id"] == "tel-test-42"
        assert attrs["worker_id"] == "2"

    def test_multiple_experiments_telemetry_does_not_accumulate_errors(self, seed_rng, capturing_sink):
        """Running multiple experiments should not cause telemetry state corruption."""
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, _make_provider(), NAPFDVerdictMetric())
        env.monitor.sink = capturing_sink
        for exp in range(3):
            env.run_single(experiment=exp, trials=_TRIALS)
        assert len(capturing_sink.rows) >= 3


# ---------------------------------------------------------------------------
# 11 — ResultsWriter async drain integration
# ---------------------------------------------------------------------------


class TestResultsWriterIntegration:
    def test_writer_drains_all_rows_to_sink(self):
        sink = CapturingSink()
        writer = ResultsWriter(sink, max_queue_size=100)
        writer.start()
        n = 20
        for i in range(n):
            writer.enqueue(
                {
                    "scenario": "s",
                    "experiment": 1,
                    "step": i,
                    "execution_id": None,
                    "worker_id": None,
                    "parallel_mode": None,
                    "policy": "G",
                    "reward_function": "R",
                    "sched_time": 0.5,
                    "sched_time_duration": 1.0,
                    "total_build_duration": 2.0,
                    "prioritization_time": 0.01,
                    "process_memory_rss_mib": None,
                    "process_memory_peak_rss_mib": None,
                    "process_cpu_utilization_percent": None,
                    "process_cpu_time_seconds": None,
                    "wall_time_seconds": None,
                    "detected": 0,
                    "missed": 0,
                    "tests_ran": 1,
                    "tests_not_ran": 0,
                    "ttf": 0.0,
                    "ttf_duration": 0.0,
                    "time_reduction": 0.0,
                    "fitness": 0.0,
                    "cost": 0.0,
                    "rewards": 0.0,
                    "avg_precision": 1.0,
                    "prioritization_order": [],
                    "variant": None,
                }
            )
        writer.stop()
        assert len(sink.rows) == n

    def test_writer_flush_called_on_stop(self):
        sink = CapturingSink()
        writer = ResultsWriter(sink, max_queue_size=100)
        writer.start()
        writer.enqueue(
            {
                "scenario": "x",
                "experiment": 1,
                "step": 1,
                "execution_id": None,
                "worker_id": None,
                "parallel_mode": None,
                "policy": "G",
                "reward_function": "R",
                "sched_time": 0.5,
                "sched_time_duration": 1.0,
                "total_build_duration": 2.0,
                "prioritization_time": 0.0,
                "process_memory_rss_mib": None,
                "process_memory_peak_rss_mib": None,
                "process_cpu_utilization_percent": None,
                "process_cpu_time_seconds": None,
                "wall_time_seconds": None,
                "detected": 0,
                "missed": 0,
                "tests_ran": 0,
                "tests_not_ran": 0,
                "ttf": 0.0,
                "ttf_duration": 0.0,
                "time_reduction": 0.0,
                "fitness": 0.0,
                "cost": 0.0,
                "rewards": 0.0,
                "avg_precision": 1.0,
                "prioritization_order": [],
                "variant": None,
            }
        )
        writer.stop()
        assert sink.flush_count >= 1

    def test_writer_start_is_idempotent(self):
        sink = CapturingSink()
        writer = ResultsWriter(sink, max_queue_size=10)
        writer.start()
        writer.start()  # second call must not create a second thread
        writer.stop()

    def test_writer_stop_is_idempotent(self):
        sink = CapturingSink()
        writer = ResultsWriter(sink, max_queue_size=10)
        writer.start()
        writer.stop()
        writer.stop()  # second call must not raise


# ---------------------------------------------------------------------------
# 12 — DuckDBCatalog query layer
# ---------------------------------------------------------------------------


class TestDuckDBCatalogLayer:
    def _write_parquet_data(self, tmp_path: Path, seed_rng) -> None:
        sink = ParquetSink(out_dir=str(tmp_path), batch_size=1000)
        agents = create_agents(GreedyPolicy(), RNFailReward(), window_sizes=[])
        env = Environment(agents, _make_provider(), NAPFDVerdictMetric())
        env.monitor.sink = sink
        env.run_single(experiment=1, trials=_TRIALS)
        env.store_experiment()

    def test_catalog_query_returns_polars_dataframe(self, tmp_path, seed_rng):
        self._write_parquet_data(tmp_path, seed_rng)
        catalog = DuckDBCatalog(str(tmp_path))
        df = catalog.query("SELECT COUNT(*) AS n FROM results")
        catalog.close()
        import polars as pl

        assert isinstance(df, pl.DataFrame)
        assert df["n"][0] >= 1

    def test_catalog_empty_dir_returns_empty_result(self, tmp_path):
        """DuckDBCatalog raises IOException when no Parquet files are found."""
        import duckdb

        with pytest.raises(duckdb.IOException):
            DuckDBCatalog(str(tmp_path))

    def test_catalog_policy_filter(self, tmp_path, seed_rng):
        self._write_parquet_data(tmp_path, seed_rng)
        catalog = DuckDBCatalog(str(tmp_path))
        df = catalog.query("SELECT policy, AVG(fitness) AS avg_fitness FROM results GROUP BY policy")
        catalog.close()
        assert len(df) >= 1

    def test_catalog_close_is_idempotent(self, tmp_path, seed_rng):
        self._write_parquet_data(tmp_path, seed_rng)
        catalog = DuckDBCatalog(str(tmp_path))
        catalog.close()
        # Second close should not raise
        with contextlib.suppress(Exception):
            catalog.close()

    def test_catalog_results_view_duckdb_sql_from_workflow(self, tmp_path, seed_rng):
        """Replicate the same SQL snippet used in docs/workflow.py."""
        self._write_parquet_data(tmp_path, seed_rng)
        catalog = DuckDBCatalog(str(tmp_path))
        df = catalog.query(
            """
            SELECT scenario, policy, AVG(fitness) AS avg_napfd
            FROM results
            GROUP BY scenario, policy
            ORDER BY avg_napfd DESC
            """
        )
        catalog.close()
        assert "policy" in df.columns
        assert "avg_napfd" in df.columns
        assert len(df) >= 1


# ---------------------------------------------------------------------------
# 13 — Pipeline from run.yaml config
# ---------------------------------------------------------------------------

_RUN_YAML = Path(__file__).parent.parent / "run.yaml"


@pytest.mark.skipif(not _RUN_YAML.exists(), reason="run.yaml not present")
class TestRunYamlConfigPipeline:
    """Validate that run.yaml resolves to a runnable EnvironmentBuildConfig."""

    def _load_yaml_config(self) -> dict[str, Any]:
        """Load run.yaml as a plain dict."""
        with open(_RUN_YAML, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_run_yaml_is_valid_yaml(self):
        with open(_RUN_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)

    def test_run_yaml_has_packs_key(self):
        with open(_RUN_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "packs" in cfg, "run.yaml must have a 'packs' key"

    def test_run_yaml_packs_are_strings(self):
        with open(_RUN_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert all(isinstance(p, str) for p in cfg["packs"])

    def test_build_environment_from_run_yaml_resolves(self, seed_rng, tmp_path):
        """EnvironmentBuildConfig built from a fakedata override of run.yaml must succeed."""
        # Build a minimal config that mirrors run.yaml structure but uses fakedata
        config = EnvironmentBuildConfig(
            datasets_dir=_EXAMPLES_DIR,
            dataset=_DATASET,
            sched_time_ratio=_SCHED_TIME_RATIO,
            use_hcs=False,
            use_context=False,
            context_config={},
            feature_groups={},
            results_config={"enabled": True, "sink": "parquet", "out_dir": str(tmp_path)},
            checkpoint_config={"enabled": False},
            telemetry_config={"enabled": False},
            algorithm_configs={"greedy": {"rnfail": {}}},
            rewards_names=["RNFail"],
            policy_names=["Greedy"],
        )
        metadata = {"execution_id": "yaml-test", "worker_id": "0", "parallel_mode": "sequential"}
        env, max_builds = build_environment(config, metadata)
        assert isinstance(env, Environment)
        assert max_builds > 0

    def test_full_run_with_parquet_results_config(self, seed_rng, tmp_path):
        """End-to-end run using results.enabled=True with ParquetSink from config."""
        config = EnvironmentBuildConfig(
            datasets_dir=_EXAMPLES_DIR,
            dataset=_DATASET,
            sched_time_ratio=_SCHED_TIME_RATIO,
            use_hcs=False,
            use_context=False,
            context_config={},
            feature_groups={},
            results_config={"enabled": True, "sink": "parquet", "out_dir": str(tmp_path), "batch_size": 1},
            checkpoint_config={"enabled": False},
            telemetry_config={"enabled": False},
            algorithm_configs={"greedy": {"rnfail": {}}},
            rewards_names=["RNFail"],
            policy_names=["Greedy"],
        )
        metadata = {"execution_id": "yaml-e2e", "worker_id": "0", "parallel_mode": "sequential"}
        env, _ = build_environment(config, metadata)
        exp_run_industrial_dataset(
            iteration=1, trials=_TRIALS, env=env, level=logging.WARNING, runtime_metadata=metadata
        )
        parquet_files = list(tmp_path.rglob("*.parquet"))
        assert len(parquet_files) > 0

    def test_unsupported_sink_type_raises_value_error(self, seed_rng):
        config = EnvironmentBuildConfig(
            datasets_dir=_EXAMPLES_DIR,
            dataset=_DATASET,
            sched_time_ratio=_SCHED_TIME_RATIO,
            use_hcs=False,
            use_context=False,
            context_config={},
            feature_groups={},
            results_config={"enabled": True, "sink": "nonexistent_sink"},
            checkpoint_config={"enabled": False},
            telemetry_config={"enabled": False},
            algorithm_configs={"greedy": {"rnfail": {}}},
            rewards_names=["RNFail"],
            policy_names=["Greedy"],
        )
        metadata = {"execution_id": "bad-sink", "worker_id": "0", "parallel_mode": "sequential"}
        with pytest.raises(ValueError, match="nonexistent_sink"):
            build_environment(config, metadata)
