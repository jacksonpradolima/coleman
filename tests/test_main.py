"""
Unit and integration tests for the experiment runner logic.

This module contains unit and integration tests to verify the functionality
of various core functions present in the ``coleman.runner`` module. The
tests cover a wide range of behaviors, including logging setup, managing
agents, loading classes dynamically, working with experiments, and scenario
providers.

Purpose:
- Ensure the robustness, reliability, and correct behavior of the runner
  module's core logic.
- Validate the integrations with other components like agents, policies,
  and scenario providers.

Tested Functionalities:
1. **Logging**:
   - `create_logger`: Verifies correct logger creation and configuration.

2. **Experiment Execution**:
   - `exp_run_industrial_dataset`: Tests running experiments with mock environments.

3. **Dynamic Class Loading**:
   - `load_class_from_module`: Ensures classes are correctly loaded dynamically from modules.

4. **Agent Creation**:
   - `create_agents`: Verifies the creation of different agent types
     (e.g., `RewardSlidingWindowAgent`, `ContextualAgent`) based on the
     provided policies and window sizes.

5. **Scenario Setup**:
   - `get_scenario_provider`: Tests the initialization of different scenario providers.

6. **End-to-End Workflow**:
   - Ensures integration of various components with mock objects to validate behavior across larger workflows.
"""

import logging
import warnings
from typing import Any, cast
from unittest.mock import Mock, patch

import polars as pl
import pytest

import coleman.policy
from coleman.agent import ContextualAgent, RewardSlidingWindowAgent
from coleman.budget import BudgetMode
from coleman.environment import Environment
from coleman.policy import FRRMABPolicy, SWLinUCBPolicy
from coleman.runner import (
    EnvironmentBuildConfig,
    ExecutionPlan,
    _effective_parallel_pool_size,
    _resolve_budget_points,
    build_agents_from_config,
    build_runtime_metadata,
    create_agents,
    create_logger,
    exp_run_industrial_dataset,
    exp_run_industrial_dataset_isolated,
    get_scenario_provider,
    load_class_from_module,
    run_parallel_executions,
)
from coleman.scenarios import ScenarioLoader

# ------------------------
# Unit Tests
# ------------------------


def test_create_logger():
    """Test that a logger is created with the correct level and handler."""
    # Cleanup global handlers to isolate the logger for this test
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Create a logger
    logger = create_logger(level=logging.DEBUG)

    # Ensure the logger's level is set correctly
    assert logger.level == logging.DEBUG

    # Verify there's exactly one handler attached
    assert len(logger.handlers) == 1

    # Cleanup: Remove all handlers from the logger to avoid reuse issues
    logger.handlers.clear()


@patch("coleman.runner.create_logger")
@patch("coleman.runner.Environment")
def test_exp_run_industrial_dataset(mock_environment, mock_create_logger, tmpdir):
    """Test that a single experiment run executes the expected environment methods."""
    mock_env = mock_environment.return_value
    mock_logger = mock_create_logger.return_value
    mock_env.scenario_provider = Mock()
    mock_env.logger = mock_logger

    exp_run_industrial_dataset(
        iteration=1,
        trials=10,
        env=cast(Environment, mock_env),
        level=20,
    )

    mock_env.set_runtime_metadata.assert_called_once()
    mock_env.run_single.assert_called_once()
    mock_env.store_experiment.assert_called_once()


def test_build_runtime_metadata_is_unique_per_execution():
    """Execution metadata should uniquely identify independent runs."""
    metadata_a = build_runtime_metadata("fakedata", 1, "process", BudgetMode.RATIO, 0.5)
    metadata_b = build_runtime_metadata("fakedata", 1, "process", BudgetMode.RATIO, 0.5)

    assert metadata_a["worker_id"] == "1"
    assert metadata_a["parallel_mode"] == "process"
    assert metadata_a["execution_id"] != metadata_b["execution_id"]


def test_build_runtime_metadata_includes_budget_identifiers():
    metadata = build_runtime_metadata(
        "fakedata",
        1,
        "process",
        budget_mode=BudgetMode.FIXED_TIME,
        budget_value=30.0,
    )

    assert metadata["budget_mode"] == "fixed_time"
    assert metadata["budget_value"] == "30"
    assert "bm=fixed_time" in metadata["execution_id"]


def test_resolve_budget_points_uses_explicit_budget_model():
    points = _resolve_budget_points({"budget": {"mode": "ratio", "values": [0.2, 0.6]}})
    assert points == [(BudgetMode.RATIO, 0.2), (BudgetMode.RATIO, 0.6)]


def test_resolve_budget_points_subset_size_mode():
    experiment = {"budget": {"mode": "subset_size", "values": [10]}}
    points = _resolve_budget_points(experiment)
    assert points == [(BudgetMode.SUBSET_SIZE, 10.0)]


def test_effective_parallel_pool_size_keeps_parallel_when_not_profiled():
    """Without profiler instrumentation, pool size should be preserved."""
    with patch.dict("os.environ", {}, clear=True):
        assert _effective_parallel_pool_size(4) == 4


def test_effective_parallel_pool_size_forces_sequential_under_scalene():
    """Scalene instrumentation should force sequential execution for stability."""
    with patch.dict("os.environ", {"SCALENE_ALLOCATION_SAMPLING_WINDOW": "1024"}, clear=True):
        assert _effective_parallel_pool_size(4) == 1


def test_effective_parallel_pool_size_allows_scalene_parallel_override():
    """Users may opt out of Scalene fallback via execution config."""
    with patch.dict("os.environ", {"SCALENE_ALLOCATION_SAMPLING_WINDOW": "1024"}, clear=True):
        assert _effective_parallel_pool_size(4, force_sequential_under_scalene=False) == 4


def test_run_parallel_executions_dispatches_unique_execution_plans():
    """Parallel dispatch should preserve per-worker isolation metadata."""
    captured: dict[str, Any] = {}

    class FakeAsyncResult:
        def get(self, timeout=None):  # noqa: ARG002
            return None

    class FakePool:
        def __init__(self, size, maxtasksperchild=None):
            captured["pool_size"] = size
            if maxtasksperchild is not None:
                captured["maxtasksperchild"] = maxtasksperchild

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

        def starmap_async(self, func, args_list):
            captured["args"] = list(args_list)
            return FakeAsyncResult()

        def terminate(self):
            return None

        def join(self):
            return None

    class FakeContext:
        def pool(self, size, maxtasksperchild=None):
            return FakePool(size, maxtasksperchild=maxtasksperchild)

        Pool = pool  # noqa: N815

    build_config = EnvironmentBuildConfig(
        datasets_dir="examples",
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={},
        telemetry_config={},
        algorithm_configs={},
        rewards_names=["RNFail"],
        policy_names=["Random"],
    )
    plans = [
        ExecutionPlan(
            iteration=1,
            trials=10,
            level=20,
            execution_id="exec-1",
            worker_id="1",
            parallel_mode="process",
        ),
        ExecutionPlan(
            iteration=2,
            trials=10,
            level=20,
            execution_id="exec-2",
            worker_id="2",
            parallel_mode="process",
        ),
    ]

    with patch("coleman.runner.get_context", return_value=FakeContext()):
        run_parallel_executions(2, build_config, plans)

    assert captured["pool_size"] == 2
    assert captured["maxtasksperchild"] == 1
    assert len(captured["args"]) == 2

    dispatched_first_plan = captured["args"][0][1]
    dispatched_second_plan = captured["args"][1][1]

    assert isinstance(dispatched_first_plan, ExecutionPlan)
    assert isinstance(dispatched_second_plan, ExecutionPlan)
    assert dispatched_first_plan.execution_id == "exec-1"
    assert dispatched_second_plan.execution_id == "exec-2"
    assert dispatched_first_plan.execution_id != dispatched_second_plan.execution_id


def test_run_parallel_executions_keyboard_interrupt_terminates_pool():
    """Cover runner KeyboardInterrupt handling path (lines 404-410)."""
    captured: dict[str, Any] = {"terminated": False, "joined": False}

    class FakeAsyncResult:
        def get(self, timeout=None):  # noqa: ARG002
            raise KeyboardInterrupt

    class FakePool:
        def __init__(self, size, maxtasksperchild=None):  # noqa: ARG002
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

        def starmap_async(self, func, args_list):  # noqa: ARG002
            return FakeAsyncResult()

        def terminate(self):
            captured["terminated"] = True

        def join(self):
            captured["joined"] = True

    class FakeContext:
        def pool(self, size, maxtasksperchild=None):
            return FakePool(size, maxtasksperchild=maxtasksperchild)

        Pool = pool  # noqa: N815

    build_config = EnvironmentBuildConfig(
        datasets_dir="examples",
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={},
        telemetry_config={},
        algorithm_configs={},
        rewards_names=["RNFail"],
        policy_names=["Random"],
    )
    plans = [ExecutionPlan(iteration=1, trials=1, level=20, execution_id="e", worker_id="1", parallel_mode="p")]

    with patch("coleman.runner.get_context", return_value=FakeContext()), pytest.raises(SystemExit) as exc:
        run_parallel_executions(1, build_config, plans)

    assert exc.value.code == 130
    assert captured["terminated"] is True
    assert captured["joined"] is True


def test_run_parallel_executions_retries_on_timeout_then_completes():
    """Cover TimeoutError continue branch in run_parallel_executions (line 405)."""
    from multiprocessing import TimeoutError as MPTimeoutError

    class FakeAsyncResult:
        def __init__(self):
            self.calls = 0

        def get(self, timeout=None):  # noqa: ARG002
            self.calls += 1
            if self.calls == 1:
                raise MPTimeoutError
            return None

    class FakePool:
        def __init__(self, size, maxtasksperchild=None):  # noqa: ARG002
            self.result = FakeAsyncResult()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

        def starmap_async(self, func, args_list):  # noqa: ARG002
            return self.result

        def terminate(self):
            return None

        def join(self):
            return None

    class FakeContext:
        def pool(self, size, maxtasksperchild=None):
            return FakePool(size, maxtasksperchild=maxtasksperchild)

        Pool = pool  # noqa: N815

    build_config = EnvironmentBuildConfig(
        datasets_dir="examples",
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={},
        telemetry_config={},
        algorithm_configs={},
        rewards_names=["RNFail"],
        policy_names=["Random"],
    )
    plans = [ExecutionPlan(iteration=1, trials=1, level=20, execution_id="e", worker_id="1", parallel_mode="p")]

    with patch("coleman.runner.get_context", return_value=FakeContext()):
        run_parallel_executions(1, build_config, plans)


def test_run_experiment_sets_seeds_when_config_seed_present(tmp_path):
    """Cover seed assignment branch in run_experiment (runner lines 499-500)."""
    from coleman.runner import run_experiment

    cfg = {
        "execution": {"seed": 7, "independent_executions": 1, "parallel_pool_size": 1, "verbose": False},
        "experiment": {
            "budget": {"mode": "ratio", "values": [0.1]},
            "datasets_dir": "examples",
            "datasets": ["fakedata"],
            "rewards": ["RNFail"],
            "policies": ["Random"],
        },
        "results": {"enabled": False, "out_dir": str(tmp_path)},
    }

    with (
        patch("coleman.runner.get_scenario_provider"),
        patch("coleman.runner.build_agents_from_config", return_value=[]),
        patch("coleman.runner.Environment"),
        patch("coleman.runner.exp_run_industrial_dataset"),
        patch("coleman.runner.run_parallel_executions"),
    ):
        run_experiment(cfg)


def test_load_class_from_module_valid():
    """Test that a valid class is loaded correctly from a module."""
    policy_class = load_class_from_module(coleman.policy, "FRRMABPolicy")
    assert policy_class.__name__ == "FRRMABPolicy"


def test_load_class_from_module_invalid():
    """Test that a ValueError is raised when an invalid class name is provided."""
    with pytest.raises(ValueError, match="Class 'InvalidPolicy' not found"):
        load_class_from_module(coleman.policy, "InvalidPolicy")


def test_create_agents_frrmab():
    """Test that create_agents returns RewardSlidingWindowAgent instances for FRRMABPolicy."""
    policy = Mock(spec=FRRMABPolicy)
    reward_fun = Mock()
    window_sizes = [5, 10]

    agents = create_agents(policy, reward_fun, window_sizes)
    assert len(agents) == 2
    assert all(isinstance(agent, RewardSlidingWindowAgent) for agent in agents)


def test_create_agents_swlinucb():
    """Test that create_agents returns ContextualAgent instances for SWLinUCBPolicy."""
    policy = Mock(spec=SWLinUCBPolicy)
    reward_fun = Mock()
    window_sizes = [5, 10]

    agents = create_agents(policy, reward_fun, window_sizes)
    assert len(agents) == 2
    assert all(isinstance(agent, ContextualAgent) for agent in agents)


def test_get_scenario_provider_basic():
    """Test that get_scenario_provider returns a ScenarioLoader instance."""
    datasets_dir = "examples"
    dataset = "fakedata"
    sched_time_ratio = 0.5
    use_hcs = False
    use_context = False
    context_config = {}
    feature_groups = {}

    scenario_provider = get_scenario_provider(
        datasets_dir, dataset, sched_time_ratio, use_hcs, use_context, context_config, feature_groups
    )

    assert isinstance(scenario_provider, ScenarioLoader)


def test_get_scenario_provider_prefers_parquet(tmp_path):
    """Test that get_scenario_provider selects Parquet when both formats are available."""
    datasets_dir = tmp_path / "datasets"
    dataset_dir = datasets_dir / "toy"
    dataset_dir.mkdir(parents=True)

    csv_df = pl.DataFrame(
        {
            "Name": ["TC_FROM_CSV"],
            "Duration": [1.0],
            "CalcPrio": [0],
            "LastRun": ["2023-01-01"],
            "Verdict": [1],
            "BuildId": [1],
        }
    )
    parquet_df = csv_df.with_columns(pl.lit("TC_FROM_PARQUET").alias("Name"))
    csv_df.write_csv(dataset_dir / "features-engineered.csv", separator=";")
    parquet_df.write_parquet(dataset_dir / "features-engineered.parquet")

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        scenario_provider = get_scenario_provider(
            str(datasets_dir),
            "toy",
            0.5,
            use_hcs=False,
            use_context=False,
            context_config={},
            feature_groups={},
        )
        first_scenario = next(scenario_provider)

    assert not any(issubclass(w.category, DeprecationWarning) for w in recorded_warnings)
    assert len(first_scenario.get_testcases()) == 1
    assert first_scenario.get_testcases()[0]["Name"] == "TC_FROM_PARQUET"


def test_get_scenario_provider_falls_back_to_csv_when_parquet_missing(tmp_path):
    """If Parquet is absent, provider should use CSV fallback path."""
    datasets_dir = tmp_path / "datasets"
    dataset_dir = datasets_dir / "toycsv"
    dataset_dir.mkdir(parents=True)

    csv_df = pl.DataFrame(
        {
            "Name": ["TC_ONLY_CSV"],
            "Duration": [1.0],
            "CalcPrio": [0],
            "LastRun": ["2023-01-01"],
            "Verdict": [1],
            "BuildId": [1],
        }
    )
    csv_df.write_csv(dataset_dir / "features-engineered.csv", separator=";")

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        scenario_provider = get_scenario_provider(
            str(datasets_dir),
            "toycsv",
            0.5,
            use_hcs=False,
            use_context=False,
            context_config={},
            feature_groups={},
        )
        first_scenario = next(scenario_provider)

    assert len(first_scenario.get_testcases()) == 1
    assert first_scenario.get_testcases()[0]["Name"] == "TC_ONLY_CSV"
    assert any(issubclass(w.category, DeprecationWarning) for w in recorded_warnings)


def test_build_agents_from_config_creates_windowed_agents():
    """build_agents_from_config should create one agent per FRRMAB window size."""
    agents = build_agents_from_config(
        algorithm_configs={"frrmab": {"window_sizes": [3, 7], "rnfail": {"c": 0.3, "decayed_factor": 0.9}}},
        policy_names=["FRRMAB"],
        rewards_names=["RNFail"],
    )

    assert len(agents) == 2
    assert all(isinstance(agent, RewardSlidingWindowAgent) for agent in agents)


# ------------------------
# Integration Tests
# ------------------------


@patch("coleman.runner.create_logger")
@patch("coleman.runner.Environment")
def test_end_to_end_execution(mock_environment, mock_create_logger, tmpdir):
    """Test end-to-end execution using mocked environment and logger."""
    mock_env = mock_environment.return_value
    mock_env.scenario_provider = Mock()
    mock_env.max_builds = 10
    mock_env.logger = mock_create_logger.return_value

    independent_executions = 3
    parameters = [(i + 1, 10, cast(Environment, mock_env), 20) for i in range(independent_executions)]

    for iteration, trials, env, level in parameters:
        exp_run_industrial_dataset(iteration, trials, env, level)

    assert mock_env.run_single.call_count == independent_executions
    assert mock_env.store_experiment.call_count == independent_executions


# ------------------------
# create_agents: remaining policy branches
# ------------------------


def test_create_agents_linucb():
    """LinUCBPolicy should produce a single ContextualAgent (no window sizes)."""
    from coleman.agent import ContextualAgent
    from coleman.policy import LinUCBPolicy
    from coleman.runner import create_agents

    policy = Mock(spec=LinUCBPolicy)
    reward_fun = Mock()

    agents = create_agents(policy, reward_fun, [])
    assert len(agents) == 1
    assert isinstance(agents[0], ContextualAgent)


def test_create_agents_default_policy():
    """Unknown (non-MAB) policies should produce a single RewardAgent."""
    from coleman.agent import RewardAgent
    from coleman.runner import create_agents

    policy = Mock()
    reward_fun = Mock()

    agents = create_agents(policy, reward_fun, [])
    assert len(agents) == 1
    assert isinstance(agents[0], RewardAgent)


# ------------------------
# get_scenario_provider: error branches
# ------------------------


def test_get_scenario_provider_hcs_and_context_raises():
    """use_hcs=True and use_context=True must raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        get_scenario_provider(
            datasets_dir="examples",
            dataset="fakedata",
            sched_time_ratio=0.5,
            use_hcs=True,
            use_context=True,
            context_config={"previous_build": ["Duration"]},
            feature_groups={"feature_group_name": "g", "feature_group_values": ["Duration"]},
        )


def test_get_scenario_provider_context_invalid_feature_group_values():
    """Non-list feature_group_values must raise TypeError."""
    with pytest.raises(TypeError, match="feature_group_values"):
        get_scenario_provider(
            datasets_dir="examples",
            dataset="alibaba@druid",
            sched_time_ratio=0.5,
            use_hcs=False,
            use_context=True,
            context_config={"previous_build": ["Duration"]},
            feature_groups={"feature_group_name": "g", "feature_group_values": "not_a_list"},
        )


def test_get_scenario_provider_context_invalid_previous_build():
    """Non-list previous_build values must raise TypeError."""
    with pytest.raises(TypeError, match="previous_build"):
        get_scenario_provider(
            datasets_dir="examples",
            dataset="alibaba@druid",
            sched_time_ratio=0.5,
            use_hcs=False,
            use_context=True,
            context_config={"previous_build": "not_a_list"},
            feature_groups={"feature_group_name": "g", "feature_group_values": ["Duration"]},
        )


# ------------------------
# build_agents_from_config
# ------------------------


def test_build_agents_from_config_basic():
    """build_agents_from_config returns one agent per policy×reward combination."""
    from coleman.runner import build_agents_from_config

    agents = build_agents_from_config(
        algorithm_configs={},
        policy_names=["Random"],
        rewards_names=["RNFail"],
    )
    assert len(agents) == 1


def test_build_agents_from_config_multiple_policies():
    """Multiple policies × rewards × window sizes expand correctly."""
    from coleman.runner import build_agents_from_config

    agents = build_agents_from_config(
        algorithm_configs={"frrmab": {"rnfail": {"c": 0.5}, "window_sizes": [5, 10]}},
        policy_names=["FRRMAB"],
        rewards_names=["RNFail"],
    )
    # FRRMAB × 1 reward × 2 window sizes = 2 agents
    assert len(agents) == 2


# ------------------------
# build_environment
# ------------------------


def test_build_environment_returns_environment_and_trial_count():
    """build_environment should construct a valid Environment and report max_builds."""
    from coleman.environment import Environment
    from coleman.runner import EnvironmentBuildConfig, build_environment

    config = EnvironmentBuildConfig(
        datasets_dir="examples",
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={},
        telemetry_config={},
        algorithm_configs={},
        rewards_names=["RNFail"],
        policy_names=["Random"],
    )
    runtime_metadata = {"execution_id": "test-exec", "worker_id": "1", "parallel_mode": "sequential"}
    env, max_builds = build_environment(config, runtime_metadata)

    assert isinstance(env, Environment)
    assert max_builds > 0


# ------------------------
# exp_run_industrial_dataset_isolated
# ------------------------


@patch("coleman.runner.build_environment")
@patch("coleman.runner.exp_run_industrial_dataset")
def test_exp_run_industrial_dataset_isolated(mock_run, mock_build_env):
    """exp_run_industrial_dataset_isolated should call build_environment then run."""
    from coleman.runner import EnvironmentBuildConfig, ExecutionPlan, exp_run_industrial_dataset_isolated

    mock_env = Mock()
    mock_build_env.return_value = (mock_env, 5)

    config = EnvironmentBuildConfig(
        datasets_dir="examples",
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={},
        telemetry_config={},
        algorithm_configs={},
        rewards_names=["RNFail"],
        policy_names=["Random"],
    )
    plan = ExecutionPlan(
        iteration=1,
        trials=5,
        level=20,
        execution_id="exec-1",
        worker_id="1",
        parallel_mode="sequential",
    )

    exp_run_industrial_dataset_isolated(config, plan)

    mock_build_env.assert_called_once()
    mock_run.assert_called_once()


# ------------------------
# _is_scalene_active
# ------------------------


def test_is_scalene_active_false_by_default():
    from unittest.mock import patch

    from coleman.runner import _is_scalene_active

    with patch.dict("os.environ", {}, clear=True):
        assert not _is_scalene_active()


def test_is_scalene_active_true_with_scalene_env():
    from unittest.mock import patch

    from coleman.runner import _is_scalene_active

    with patch.dict("os.environ", {"SCALENE_ALLOCATION_SAMPLING_WINDOW": "1024"}, clear=True):
        assert _is_scalene_active()


# ------------------------
# _dispatch_executions
# ------------------------


@patch("coleman.runner.run_parallel_executions")
@patch("coleman.runner.exp_run_industrial_dataset_isolated")
def test_dispatch_executions_sequential(mock_isolated, mock_parallel):
    """Pool size of 1 should run sequentially, not via process pool."""
    from coleman.runner import EnvironmentBuildConfig, ExecutionPlan, _dispatch_executions

    config = EnvironmentBuildConfig(
        datasets_dir="examples",
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={},
        telemetry_config={},
        algorithm_configs={},
        rewards_names=["RNFail"],
        policy_names=["Random"],
    )
    plans = [
        ExecutionPlan(iteration=1, trials=5, level=20, execution_id="e1", worker_id="1", parallel_mode="sequential")
    ]
    _dispatch_executions(1, config, plans)
    mock_isolated.assert_called_once_with(config, plans[0])
    mock_parallel.assert_not_called()


@patch("coleman.runner.run_parallel_executions")
@patch("coleman.runner.exp_run_industrial_dataset_isolated")
def test_dispatch_executions_parallel(mock_isolated, mock_parallel):
    """Pool size > 1 should delegate to run_parallel_executions."""
    from coleman.runner import EnvironmentBuildConfig, ExecutionPlan, _dispatch_executions

    config = EnvironmentBuildConfig(
        datasets_dir="examples",
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={},
        telemetry_config={},
        algorithm_configs={},
        rewards_names=["RNFail"],
        policy_names=["Random"],
    )
    plans = [
        ExecutionPlan(
            iteration=i + 1, trials=5, level=20, execution_id=f"e{i}", worker_id=str(i), parallel_mode="process"
        )
        for i in range(3)
    ]
    _dispatch_executions(3, config, plans)
    mock_parallel.assert_called_once_with(3, config, plans)
    mock_isolated.assert_not_called()


def test_run_experiment_dispatches_coordinator_hooks():
    from coleman.runner import run_experiment

    hook = Mock()
    scenario = Mock()
    scenario.max_builds = 3

    cfg = {
        "_run_id": "rid-123",
        "execution": {
            "seed": 7,
            "independent_executions": 1,
            "parallel_pool_size": 1,
            "verbose": False,
        },
        "experiment": {
            "budget": {"mode": "ratio", "values": [0.1]},
            "datasets_dir": "examples",
            "datasets": ["fakedata"],
            "rewards": ["RNFail"],
            "policies": ["Random"],
        },
        "hooks": {
            "fail_fast": True,
            "plugins": ["tests.support.hook_plugins.RecordingHook"],
        },
        "extensions": {"my_domain": {"k": "v"}},
        "results": {"enabled": False},
    }

    with (
        patch("coleman.runner.get_scenario_provider", return_value=scenario),
        patch("coleman.runner.build_agents_from_config", return_value=[]),
        patch("coleman.runner._dispatch_executions"),
        patch("coleman.runner.load_hook_plugins", return_value=[hook]),
    ):
        run_experiment(cfg)

    hook.on_run_start.assert_called_once()
    hook.on_dataset_start.assert_called_once()
    hook.on_dataset_end.assert_called_once()
    hook.on_run_end.assert_called_once()


def test_run_experiment_with_extension_skips_core_scenario_loading():
    from coleman.runner import RunnerExtension, run_experiment_with_extension

    extension = RunnerExtension(
        build_environment_fn=lambda config, runtime_metadata, seed: (Mock(), 5),  # noqa: ARG005
    )
    cfg = {
        "_run_id": "rid-123",
        "execution": {
            "seed": 7,
            "independent_executions": 1,
            "parallel_pool_size": 1,
            "verbose": False,
        },
        "experiment": {
            "budget": {"mode": "ratio", "values": [0.1]},
            "datasets_dir": "examples",
            "datasets": ["fakedata"],
            "rewards": ["RNFail"],
            "policies": ["Random"],
        },
        "results": {"enabled": False},
    }

    with (
        patch(
            "coleman.runner.get_scenario_provider", side_effect=AssertionError("must not be called")
        ) as scenario_patch,
        patch("coleman.runner.build_agents_from_config", return_value=[]),
        patch("coleman.runner._dispatch_executions") as dispatch_patch,
        patch("coleman.runner.load_hook_plugins", return_value=[]),
    ):
        run_experiment_with_extension(cfg, extension)

    scenario_patch.assert_not_called()
    dispatch_patch.assert_called_once()
    plans = dispatch_patch.call_args.args[2]
    assert plans
    assert all(plan.trials == 0 for plan in plans)


def test_exp_run_isolated_dispatches_execution_hooks():
    hook = Mock()
    mock_env = Mock()

    config = EnvironmentBuildConfig(
        datasets_dir="examples",
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={},
        telemetry_config={},
        algorithm_configs={},
        rewards_names=["RNFail"],
        policy_names=["Random"],
        run_id="rid-999",
        extensions={"my_domain": {"a": 1}},
        hook_plugin_paths=["tests.support.hook_plugins.RecordingHook"],
        hook_fail_fast=True,
    )
    plan = ExecutionPlan(
        iteration=1,
        trials=3,
        level=20,
        execution_id="exec-1",
        worker_id="1",
        parallel_mode="sequential",
    )

    with (
        patch("coleman.runner.load_hook_plugins", return_value=[hook]),
        patch("coleman.runner.build_environment", return_value=(mock_env, 3)),
        patch("coleman.runner.exp_run_industrial_dataset"),
    ):
        exp_run_industrial_dataset_isolated(config, plan)

    hook.on_execution_start.assert_called_once()
    hook.on_execution_end.assert_called_once()


def test_exp_run_isolated_prefers_build_environment_trial_count():
    hook = Mock()
    mock_env = Mock()

    config = EnvironmentBuildConfig(
        datasets_dir="examples",
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={},
        telemetry_config={},
        algorithm_configs={},
        rewards_names=["RNFail"],
        policy_names=["Random"],
        run_id="rid-999",
        extensions={},
        hook_plugin_paths=["tests.support.hook_plugins.RecordingHook"],
        hook_fail_fast=True,
    )
    plan = ExecutionPlan(
        iteration=1,
        trials=3,
        level=20,
        execution_id="exec-1",
        worker_id="1",
        parallel_mode="sequential",
    )

    with (
        patch("coleman.runner.load_hook_plugins", return_value=[hook]),
        patch("coleman.runner.build_environment", return_value=(mock_env, 7)),
        patch("coleman.runner.exp_run_industrial_dataset") as run_patch,
    ):
        exp_run_industrial_dataset_isolated(config, plan)

    run_patch.assert_called_once()
    assert run_patch.call_args.args[1] == 7


def test_exp_run_isolated_dispatches_error_when_build_environment_fails():
    hook = Mock()

    config = EnvironmentBuildConfig(
        datasets_dir="examples",
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={},
        telemetry_config={},
        algorithm_configs={},
        rewards_names=["RNFail"],
        policy_names=["Random"],
        run_id="rid-999",
        extensions={},
        hook_plugin_paths=["tests.support.hook_plugins.RecordingHook"],
        hook_fail_fast=True,
    )
    plan = ExecutionPlan(
        iteration=1,
        trials=3,
        level=20,
        execution_id="exec-1",
        worker_id="1",
        parallel_mode="sequential",
    )

    with (
        patch("coleman.runner.load_hook_plugins", return_value=[hook]),
        patch("coleman.runner.build_environment", side_effect=RuntimeError("bad env")),
        pytest.raises(RuntimeError, match="bad env"),
    ):
        exp_run_industrial_dataset_isolated(config, plan)

    hook.on_execution_start.assert_called_once()
    hook.on_error.assert_called_once()


def test_exp_run_isolated_dispatches_error_when_execution_start_hook_fails():
    hook = Mock()
    hook.on_execution_start.side_effect = RuntimeError("start boom")

    config = EnvironmentBuildConfig(
        datasets_dir="examples",
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={},
        telemetry_config={},
        algorithm_configs={},
        rewards_names=["RNFail"],
        policy_names=["Random"],
        run_id="rid-999",
        extensions={},
        hook_plugin_paths=["tests.support.hook_plugins.RecordingHook"],
        hook_fail_fast=True,
    )
    plan = ExecutionPlan(
        iteration=1,
        trials=3,
        level=20,
        execution_id="exec-1",
        worker_id="1",
        parallel_mode="sequential",
    )

    with (
        patch("coleman.runner.load_hook_plugins", return_value=[hook]),
        patch("coleman.runner.build_environment"),
        pytest.raises(RuntimeError, match="start boom"),
    ):
        exp_run_industrial_dataset_isolated(config, plan)

    hook.on_error.assert_called_once()
    error_context = hook.on_error.call_args.args[0]
    assert error_context.execution_id == "exec-1"


def test_run_experiment_dispatches_error_with_dataset_context():
    from coleman.runner import run_experiment

    hook = Mock()
    scenario = Mock()
    scenario.max_builds = 3

    cfg = {
        "_run_id": "rid-123",
        "execution": {
            "seed": 7,
            "independent_executions": 1,
            "parallel_pool_size": 1,
            "verbose": False,
        },
        "experiment": {
            "budget": {"mode": "ratio", "values": [0.1]},
            "datasets_dir": "examples",
            "datasets": ["fakedata"],
            "rewards": ["RNFail"],
            "policies": ["Random"],
        },
        "hooks": {
            "fail_fast": True,
            "plugins": ["tests.support.hook_plugins.RecordingHook"],
        },
        "results": {"enabled": False},
    }

    with (
        patch("coleman.runner.get_scenario_provider", return_value=scenario),
        patch("coleman.runner.build_agents_from_config", return_value=[]),
        patch("coleman.runner.load_hook_plugins", return_value=[hook]),
        patch("coleman.runner._dispatch_executions", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        run_experiment(cfg)

    hook.on_error.assert_called_once()
    error_context = hook.on_error.call_args.args[0]
    assert error_context.dataset_id == "fakedata"
