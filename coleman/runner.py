"""
coleman.runner - Experiment orchestration utilities.

Provides the core functions for building agents, environments, and
running experiments — both sequentially and in parallel.  These were
previously in the top-level ``main.py`` and are now importable as a
library module.

Functions
---------
create_logger
    Create a multiprocessing-safe logger.
load_class_from_module
    Dynamically load a class from a module.
create_agents
    Build agent instances from a policy / reward / window-size triple.
get_scenario_provider
    Return the appropriate scenario provider for the given dataset config.
build_agents_from_config
    Build all agents from algorithm config dicts.
build_environment
    Create a fresh ``Environment`` for one execution.
build_runtime_metadata
    Build stable execution metadata for telemetry and results.
exp_run_industrial_dataset
    Execute a single experiment run.
exp_run_industrial_dataset_isolated
    Execute one run by constructing an isolated ``Environment`` in the worker.
run_parallel_executions
    Run worker executions with responsive Ctrl+C handling.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from multiprocessing import TimeoutError, get_context
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

import coleman.policy
import coleman.policy.base
import coleman.reward
from coleman.agent import (
    ContextualAgent,
    RewardAgent,
    RewardSlidingWindowAgent,
    SlidingWindowContextualAgent,
)
from coleman.environment import Environment
from coleman.evaluation import NAPFDVerdictMetric
from coleman.hooks import (
    DatasetResult,
    ExecutionResult,
    HookContext,
    RunResult,
    dispatch_hook_event,
    load_hook_plugins,
)
from coleman.policy import FRRMABPolicy, LinUCBPolicy, SWLinUCBPolicy
from coleman.scenarios import (
    ContextScenarioLoader,
    HCSScenarioLoader,
    ScenarioLoader,
)
from coleman.spec import RunSpec
from coleman.spec.selection import resolve_requested_names


@dataclass(frozen=True)
class ExecutionPlan:
    """Serializable worker plan for one independent execution."""

    iteration: int
    trials: int
    level: int
    execution_id: str
    worker_id: str
    parallel_mode: str
    seed: int | None = None


@dataclass(frozen=True)
class EnvironmentBuildConfig:
    """Serializable configuration required to build an isolated Environment."""

    datasets_dir: str
    dataset: str
    sched_time_ratio: float
    use_hcs: bool
    use_context: bool
    context_config: dict[str, Any]
    feature_groups: dict[str, Any]
    results_config: dict[str, Any]
    checkpoint_config: dict[str, Any]
    telemetry_config: dict[str, Any]
    algorithm_configs: dict[str, Any]
    rewards_names: list[str]
    policy_names: list[str]
    seed: int | None = None
    run_id: str | None = None
    extensions: dict[str, Any] = field(default_factory=dict)
    hook_plugin_paths: list[str] = field(default_factory=list)
    hook_fail_fast: bool = True
    extension: RunnerExtension | None = None
    resolved_spec: RunSpec | None = None


@dataclass(frozen=True)
class RunnerExtension:
    """Typed extension points for core runner orchestration."""

    build_environment_fn: Callable[[EnvironmentBuildConfig, dict[str, str], int | None], tuple[Any, int]]
    build_metric_fn: Callable[[RunSpec, str, float], Any] | None = None
    post_execution_fn: Callable[[HookContext, Any], None] | None = None


@dataclass(frozen=True)
class AgentBuildIssue:
    """Validation issue produced during agent build normalization."""

    code: str
    message: str
    policy_name: str
    reward_name: str | None = None
    hint: str | None = None


# taken from https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python
def create_logger(level):
    """Create and configure a logger for multiprocessing-safe logging.

    Parameters
    ----------
    level : int
        The logging level (e.g., ``logging.DEBUG``, ``logging.INFO``).

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # this bit will make sure you won't have duplicated messages in the output
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def load_class_from_module(module, class_name: str):
    """Dynamically load a class from a given module.

    Parameters
    ----------
    module : module
        The Python module from which to load the class.
    class_name : str
        The name of the class to be loaded.

    Returns
    -------
    type
        The loaded class.

    Raises
    ------
    ValueError
        If the class is not found in the provided module.
    """
    if hasattr(module, class_name):
        return getattr(module, class_name)
    raise ValueError(f"Class '{class_name}' not found in {module.__name__}!")


def create_agents(policy, rew_fun, window_sizes, seed: int | None = None):
    """Create agent instances based on the policy type.

    Parameters
    ----------
    policy : object
        The policy instance.
    rew_fun : object
        The reward function instance.
    window_sizes : list
        List of window sizes (only relevant for policies that use Sliding
        Window such as FRRMABPolicy).
    seed : int, optional
        Seed for the agents' internal RNG, which controls the random initial
        shuffle at ``t=0``.  When provided, repeated runs with the same seed
        and scenario produce identical prioritization sequences.

    Returns
    -------
    list
        A list of agent instances.
    """
    if isinstance(policy, FRRMABPolicy):
        return [RewardSlidingWindowAgent(policy, rew_fun, w, seed=seed) for w in window_sizes]

    if isinstance(policy, SWLinUCBPolicy):
        return [SlidingWindowContextualAgent(policy, rew_fun, w, seed=seed) for w in window_sizes]

    if isinstance(policy, LinUCBPolicy):
        return [ContextualAgent(policy, rew_fun, seed=seed)]

    return [RewardAgent(policy, rew_fun, seed=seed)]


def get_scenario_provider(  # pylint: disable=too-many-positional-arguments
    datasets_dir: str,
    dataset: str,
    sched_time_ratio: float,
    use_hcs: bool,
    use_context: bool,
    context_config: dict[str, Any],
    feature_groups: dict[str, Any],
) -> ScenarioLoader | HCSScenarioLoader | ContextScenarioLoader:
    """Return the appropriate scenario loader based on the given configuration.

    The function selects the scenario loader based on whether the
    HCS (Highly-Configurable System) configuration is used. It constructs the
    appropriate paths for the dataset files and initializes the scenario
    loader with these paths.

    Parameters
    ----------
    datasets_dir : str
        The directory where datasets are stored.
    dataset : str
        The specific dataset to be used.
    sched_time_ratio : float
        The ratio of scheduled time to be used in the scenario.
    use_hcs : bool
        If True, returns an ``HCSScenarioLoader`` instance.
    use_context : bool
        If True, returns a ``ContextScenarioLoader`` instance.
    context_config : dict
        Configuration for contextual information.
    feature_groups : dict
        Feature group configuration.

    Returns
    -------
    ScenarioLoader or HCSScenarioLoader or ContextScenarioLoader
        An instance of the scenario loader based on the given configuration.
    """

    def _prefer_parquet(base_path_without_ext: str) -> str:
        """Return the Parquet path if it exists, else fall back to CSV."""
        parquet_path = f"{base_path_without_ext}.parquet"
        csv_path = f"{base_path_without_ext}.csv"
        if os.path.exists(parquet_path):
            return parquet_path
        return csv_path

    base_tcfile = _prefer_parquet(f"{datasets_dir}/{dataset}/features-engineered")

    if use_hcs and not use_context:
        variants_file = _prefer_parquet(f"{datasets_dir}/{dataset}/data-variants")
        return HCSScenarioLoader(base_tcfile, variants_file, sched_time_ratio)

    if use_hcs and use_context:
        raise NotImplementedError

    if use_context:
        contextual_tcfile = _prefer_parquet(f"{datasets_dir}/{dataset}/features-engineered-contextual")
        feature_group_name = str(feature_groups["feature_group_name"])

        feature_group_values_raw = feature_groups["feature_group_values"]
        previous_build_raw = context_config["previous_build"]

        if not isinstance(feature_group_values_raw, list) or not all(
            isinstance(value, str) for value in feature_group_values_raw
        ):
            raise TypeError("feature_group_values must be a list[str]")

        if not isinstance(previous_build_raw, list) or not all(isinstance(value, str) for value in previous_build_raw):
            raise TypeError("previous_build must be a list[str]")

        return ContextScenarioLoader(
            contextual_tcfile,
            feature_group_name,
            feature_group_values_raw,
            previous_build_raw,
            sched_time_ratio,
        )

    return ScenarioLoader(base_tcfile, sched_time_ratio)


def build_agents_from_config(
    algorithm_configs: dict[str, Any],
    policy_names: list[str],
    rewards_names: list[str],
    seed: int | None = None,
) -> list[RewardAgent | RewardSlidingWindowAgent | ContextualAgent | SlidingWindowContextualAgent]:
    """Build all agents from config values in a process-local way."""
    policies = {
        policy_name: {
            reward_name: load_class_from_module(coleman.policy, policy_name + "Policy")(
                **algorithm_configs.get(policy_name.lower(), {}).get(reward_name.lower(), {})
            )
            for reward_name in rewards_names
        }
        for policy_name in policy_names
    }

    return [
        agent
        for policy_name, reward_policies in policies.items()
        for reward_name, policy in reward_policies.items()
        for agent in create_agents(
            policy,
            load_class_from_module(coleman.reward, reward_name + "Reward")(),
            algorithm_configs.get(policy_name.lower(), {}).get("window_sizes", []),
            seed=seed,
        )
    ]


def build_runtime_metadata(
    dataset: str,
    sched_time_ratio: float,
    iteration: int,
    parallel_mode: str,
) -> dict[str, str]:
    """Build stable execution metadata for telemetry and persisted results."""
    execution_id = f"{dataset}|tr={sched_time_ratio:.2f}|exp={iteration}|{uuid4().hex[:8]}"
    return {
        "execution_id": execution_id,
        "worker_id": str(iteration),
        "parallel_mode": parallel_mode,
    }


def build_environment(
    build_config: EnvironmentBuildConfig,
    runtime_metadata: dict[str, str],
    agent_seed: int | None = None,
) -> tuple[Environment, int]:
    """Create a fresh environment for one execution."""
    effective_agent_seed = build_config.seed if agent_seed is None else agent_seed

    if build_config.extension is not None and build_config.extension.build_environment_fn is not None:
        return build_config.extension.build_environment_fn(build_config, runtime_metadata, effective_agent_seed)

    agents = build_agents_from_config(
        build_config.algorithm_configs,
        build_config.policy_names,
        build_config.rewards_names,
        seed=effective_agent_seed,
    )
    scenario = get_scenario_provider(
        build_config.datasets_dir,
        build_config.dataset,
        build_config.sched_time_ratio,
        build_config.use_hcs,
        build_config.use_context,
        build_config.context_config,
        build_config.feature_groups,
    )
    env = Environment(
        agents,
        scenario,
        _resolve_metric(build_config),
        results_config=build_config.results_config,
        checkpoint_config=build_config.checkpoint_config,
        telemetry_config=build_config.telemetry_config,
        runtime_metadata=runtime_metadata,
    )
    return env, scenario.max_builds


def _resolve_metric(build_config: EnvironmentBuildConfig) -> Any:
    """Resolve execution metric from extension callback or default metric."""
    extension = build_config.extension
    if extension is not None and extension.build_metric_fn is not None and build_config.resolved_spec is not None:
        return extension.build_metric_fn(
            build_config.resolved_spec,
            build_config.dataset,
            build_config.sched_time_ratio,
        )
    return NAPFDVerdictMetric()


def exp_run_industrial_dataset(
    iteration: int,
    trials: int,
    env: Environment,
    level: int,
    runtime_metadata: dict[str, str] | None = None,
) -> None:
    """Execute a single run of the industrial dataset experiment.

    Parameters
    ----------
    iteration : int
        The current iteration of the experiment.
    trials : int
        The total number of trials to be executed.
    env : Environment
        An instance of the environment where the experiment is run.
    level : int
        The logging level.
    runtime_metadata : dict[str, str] or None
        Execution-scoped metadata attached to telemetry and persisted results.
    """
    # Initialize logging for worker processes without mutating Environment state.
    create_logger(level)
    env.set_runtime_metadata(runtime_metadata)
    env.run_single(iteration, trials)
    env.store_experiment()


def exp_run_industrial_dataset_isolated(build_config: EnvironmentBuildConfig, plan: ExecutionPlan) -> None:
    """Execute one run by constructing an isolated Environment inside the worker process."""
    execution_hooks = load_hook_plugins(build_config.hook_plugin_paths)

    if plan.seed is not None:
        coleman.policy.base._rng = np.random.default_rng(plan.seed)
        pl.set_random_seed(plan.seed)

    runtime_metadata = {
        "execution_id": plan.execution_id,
        "worker_id": plan.worker_id,
        "parallel_mode": plan.parallel_mode,
    }

    hook_context = HookContext(
        run_id=build_config.run_id,
        dataset_id=build_config.dataset,
        execution_id=plan.execution_id,
        worker_id=plan.worker_id,
        parallel_mode=plan.parallel_mode,
        iteration=plan.iteration,
        trials=plan.trials,
        sched_time_ratio=build_config.sched_time_ratio,
        extensions=build_config.extensions,
    )

    started_at = time.time()
    try:
        dispatch_hook_event(
            execution_hooks,
            "on_execution_start",
            hook_context,
            fail_fast=build_config.hook_fail_fast,
        )
        env, _ = build_environment(build_config, runtime_metadata, agent_seed=plan.seed)
        exp_run_industrial_dataset(plan.iteration, plan.trials, env, plan.level, runtime_metadata)
        if build_config.extension is not None and build_config.extension.post_execution_fn is not None:
            build_config.extension.post_execution_fn(hook_context, env)
        dispatch_hook_event(
            execution_hooks,
            "on_execution_end",
            hook_context,
            ExecutionResult(status="ok", duration_seconds=time.time() - started_at),
            fail_fast=build_config.hook_fail_fast,
        )
    except Exception as exc:  # noqa: BLE001
        dispatch_hook_event(
            execution_hooks,
            "on_error",
            hook_context,
            exc,
            fail_fast=build_config.hook_fail_fast,
        )
        raise


def run_parallel_executions(
    parallel_pool_size: int,
    build_config: EnvironmentBuildConfig,
    execution_plans: list[ExecutionPlan],
) -> None:
    """Run worker executions with responsive Ctrl+C handling.

    Parameters
    ----------
    parallel_pool_size : int
        Number of worker processes.
    build_config : EnvironmentBuildConfig
        Serializable configuration used to build isolated environments.
    execution_plans : list[ExecutionPlan]
        Task parameters for worker execution.
    """
    # Recycle workers after one task to avoid intermittent queue-unpickling
    # corruption seen with spawn pools on Python 3.14 under heavy I/O.
    with get_context("spawn").Pool(parallel_pool_size, maxtasksperchild=1) as pool:
        async_result = pool.starmap_async(
            exp_run_industrial_dataset_isolated,
            [(build_config, execution_plan) for execution_plan in execution_plans],
        )

        try:
            while True:
                try:
                    async_result.get(timeout=1)
                    break
                except TimeoutError:
                    continue
        except KeyboardInterrupt:
            logging.warning("Interrupt received. Terminating worker pool...")
            pool.terminate()
            pool.join()
            raise SystemExit(130) from None


def _dispatch_executions(
    parallel_pool_size: int,
    build_config: EnvironmentBuildConfig,
    execution_plans: list[ExecutionPlan],
) -> None:
    """Run execution plans in parallel or sequentially.

    Extracted to keep ``run_experiment`` within the cognitive-complexity
    budget (≤ 15).

    Parameters
    ----------
    parallel_pool_size : int
        When > 1, plans are dispatched to a process pool; otherwise they
        run sequentially in the current process.
    build_config : EnvironmentBuildConfig
        Serializable configuration used to build isolated environments.
    execution_plans : list[ExecutionPlan]
        Task parameters for each independent execution.
    """
    if parallel_pool_size > 1:
        run_parallel_executions(parallel_pool_size, build_config, execution_plans)
    else:
        for execution_plan in execution_plans:
            exp_run_industrial_dataset_isolated(build_config, execution_plan)


def _is_scalene_active() -> bool:
    """Return True when running under Scalene profiler instrumentation."""
    return any(key.startswith("SCALENE_") for key in os.environ)


def _effective_parallel_pool_size(
    parallel_pool_size: int,
    *,
    force_sequential_under_scalene: bool = True,
) -> int:
    """Return the safe pool size for the current runtime.

    Scalene + Python 3.14 can intermittently corrupt multiprocessing spawn
    queues, causing ``_pickle.UnpicklingError`` in worker bootstrap. By
    default we force sequential execution for profiling stability.
    """
    if parallel_pool_size > 1 and _is_scalene_active() and force_sequential_under_scalene:
        logging.warning(
            "Scalene profiling detected; forcing sequential execution "
            "(parallel_pool_size=1) to avoid multiprocessing instability "
            "and incomplete per-thread tracking."
        )
        return 1
    return parallel_pool_size


def effective_parallel_pool_size(
    configured_pool_size: int,
    *,
    force_sequential_under_scalene: bool = True,
) -> int:
    """Public helper mirroring runner Scalene-safe pool-size behavior."""
    return _effective_parallel_pool_size(
        configured_pool_size,
        force_sequential_under_scalene=force_sequential_under_scalene,
    )


def _available_policy_names() -> list[str]:
    """Return canonical policy names exported by ``coleman.policy``."""
    names = [name[:-6] for name in coleman.policy.__all__ if name.endswith("Policy") and name != "Policy"]
    return sorted(set(names), key=str.lower)


def _available_reward_names() -> list[str]:
    """Return canonical reward names exported by ``coleman.reward``."""
    names = [name[:-6] for name in coleman.reward.__all__ if name.endswith("Reward") and name != "Reward"]
    return sorted(set(names), key=str.lower)


def normalize_and_validate_agent_build(
    *,
    algorithm_configs: dict[str, Any],
    policy_names: list[str],
    rewards_names: list[str],
    strict: bool = True,
) -> tuple[dict[str, Any], list[AgentBuildIssue]]:
    """Normalize policy configs and validate compatibility before execution."""
    normalized = deepcopy(algorithm_configs)
    issues: list[AgentBuildIssue] = []

    available_policy_names = _available_policy_names()
    available_reward_names = _available_reward_names()

    policy_lookup = {name.lower(): name for name in available_policy_names}
    reward_lookup = {name.lower(): name for name in available_reward_names}

    for policy_name in policy_names:
        if policy_name.lower() not in policy_lookup:
            issues.append(
                AgentBuildIssue(
                    code="unknown_policy",
                    message=f"Unknown policy {policy_name!r}.",
                    policy_name=policy_name,
                    hint="Use one of the canonical policy names exported by coleman.policy.",
                )
            )
            continue

        cfg_key = policy_name.lower()
        policy_cfg = normalized.get(cfg_key, {})
        if not isinstance(policy_cfg, dict):
            issues.append(
                AgentBuildIssue(
                    code="invalid_policy_config",
                    message=f"Algorithm config for policy {policy_name!r} must be a dictionary.",
                    policy_name=policy_name,
                )
            )
            continue

        if policy_name in {"FRRMAB", "SWLinUCB"}:
            window_sizes = policy_cfg.get("window_sizes", [])
            if not isinstance(window_sizes, list) or len(window_sizes) == 0:
                issues.append(
                    AgentBuildIssue(
                        code="missing_window_sizes",
                        message=f"Policy {policy_name!r} requires a non-empty window_sizes list.",
                        policy_name=policy_name,
                        hint=f"Define algorithm.{cfg_key}.window_sizes: [5, 10] (example).",
                    )
                )

        if policy_name == "PortfolioUCB":
            for reward_name in rewards_names:
                if reward_name.lower() not in reward_lookup:
                    issues.append(
                        AgentBuildIssue(
                            code="unknown_reward",
                            message=f"Unknown reward {reward_name!r}.",
                            policy_name=policy_name,
                            reward_name=reward_name,
                            hint="Use one of the canonical reward names exported by coleman.reward.",
                        )
                    )
                    continue

                reward_key = reward_name.lower()
                reward_cfg = policy_cfg.get(reward_key, {})
                if not isinstance(reward_cfg, dict):
                    issues.append(
                        AgentBuildIssue(
                            code="invalid_reward_policy_config",
                            message=(
                                f"Config for policy {policy_name!r} and reward {reward_name!r} must be a dictionary."
                            ),
                            policy_name=policy_name,
                            reward_name=reward_name,
                        )
                    )
                    continue

                candidate_policies_raw = reward_cfg.get("policies")
                if not isinstance(candidate_policies_raw, list):
                    issues.append(
                        AgentBuildIssue(
                            code="portfolio_missing_policies",
                            message=(f"Policy {policy_name!r} requires '{reward_key}.policies' as a non-empty list."),
                            policy_name=policy_name,
                            reward_name=reward_name,
                            hint=(
                                f"Define algorithm.portfolioucb.{reward_key}.policies with canonical "
                                "policy names, e.g. ['UCB1', 'Random']."
                            ),
                        )
                    )
                    continue

                if not candidate_policies_raw:
                    issues.append(
                        AgentBuildIssue(
                            code="portfolio_empty_policies",
                            message=f"Policy {policy_name!r} received an empty policies list.",
                            policy_name=policy_name,
                            reward_name=reward_name,
                        )
                    )
                    continue

                if all(isinstance(item, str) for item in candidate_policies_raw):
                    resolved_candidates, unknown_candidates = resolve_requested_names(
                        candidate_policies_raw,
                        available_policy_names,
                    )
                    for unknown_name in unknown_candidates:
                        issues.append(
                            AgentBuildIssue(
                                code="unknown_portfolio_policy",
                                message=(
                                    f"Unknown nested portfolio policy {unknown_name!r} for reward {reward_name!r}."
                                ),
                                policy_name=policy_name,
                                reward_name=reward_name,
                            )
                        )

                    if any(candidate == "PortfolioUCB" for candidate in resolved_candidates):
                        issues.append(
                            AgentBuildIssue(
                                code="portfolio_recursive_reference",
                                message="PortfolioUCB cannot include itself as a nested candidate policy.",
                                policy_name=policy_name,
                                reward_name=reward_name,
                            )
                        )
                        continue

                    if not unknown_candidates and resolved_candidates:
                        candidate_instances = []
                        for candidate in resolved_candidates:
                            params = normalized.get(candidate.lower(), {}).get(reward_key, {})
                            if not isinstance(params, dict):
                                params = {}
                            try:
                                candidate_instances.append(
                                    load_class_from_module(coleman.policy, candidate + "Policy")(**params)
                                )
                            except Exception as exc:  # noqa: BLE001
                                issues.append(
                                    AgentBuildIssue(
                                        code="portfolio_policy_init_error",
                                        message=(
                                            f"Failed to initialize nested policy {candidate!r} "
                                            f"for reward {reward_name!r}: {exc}"
                                        ),
                                        policy_name=policy_name,
                                        reward_name=reward_name,
                                        hint=(
                                            f"Provide required parameters under algorithm.{candidate.lower()}.{reward_key}."  # noqa: E501
                                        ),
                                    )
                                )
                                continue
                        reward_cfg = dict(reward_cfg)
                        reward_cfg["policies"] = candidate_instances
                        policy_cfg[reward_key] = reward_cfg
                        normalized[cfg_key] = policy_cfg

    if strict and issues:
        lines = [
            "Invalid agent/policy configuration detected:",
            *["- " + issue.message + (f" Hint: {issue.hint}" if issue.hint else "") for issue in issues],
        ]
        raise ValueError("\n".join(lines))

    return normalized, issues


def _run_experiment_impl(spec_dict: dict[str, Any], extension: RunnerExtension | None = None) -> None:
    """Run a full experiment from a resolved spec dictionary.

    This is the bridge between the new YAML/pack-based config system and
    the existing experiment execution engine.

    Parameters
    ----------
    spec_dict : dict[str, Any]
        A resolved ``RunSpec`` as a plain dictionary (e.g., from
        ``spec.model_dump()``).
    """
    from pathlib import Path

    execution = spec_dict.get("execution", {})
    experiment = spec_dict.get("experiment", {})
    algorithm_configs = spec_dict.get("algorithm", {})
    hcs_config = spec_dict.get("hcs_configuration", {})
    contextual_info = spec_dict.get("contextual_information", {})
    results_config = spec_dict.get("results", {})
    checkpoint_config = spec_dict.get("checkpoint", {})
    telemetry_config = spec_dict.get("telemetry", {})
    hooks_config = spec_dict.get("hooks", {})
    extensions = spec_dict.get("extensions", {})
    run_id = spec_dict.get("_run_id")
    resolved_payload = dict(spec_dict)
    resolved_payload.pop("_run_id", None)
    resolved_spec = RunSpec.model_validate(resolved_payload)

    parallel_pool_size = execution.get("parallel_pool_size", 10)
    independent_executions = execution.get("independent_executions", 10)
    seed = execution.get("seed")
    verbose = execution.get("verbose", False)
    force_sequential_under_scalene = execution.get("force_sequential_under_scalene", True)
    hook_plugin_paths = hooks_config.get("plugins", [])
    hook_fail_fast = hooks_config.get("fail_fast", True)

    coordinator_hooks = load_hook_plugins(hook_plugin_paths)
    run_context = HookContext(run_id=run_id, extensions=extensions)

    # Apply seed to both RNGs for full reproducibility:
    # - numpy RNG is used by policy modules (RandomPolicy, EpsilonGreedyPolicy)
    # - polars RNG is used by Agent.choose() at t=0 via Series.shuffle()
    if seed is not None:
        coleman.policy.base._rng = np.random.default_rng(seed)
        pl.set_random_seed(seed)

    sched_time_ratio = experiment.get("scheduled_time_ratio", [0.1, 0.5, 0.8])
    datasets_dir = experiment.get("datasets_dir", "examples")
    datasets = experiment.get("datasets", [])
    experiment_dir = experiment.get("experiment_dir", "results/experiments/")
    rewards_names = experiment.get("rewards", ["RNFail", "TimeRank"])
    policy_names = experiment.get("policies", ["Random"])

    available_policy_names = _available_policy_names()
    available_reward_names = _available_reward_names()

    policy_names, unknown_policy_names = resolve_requested_names(policy_names, available_policy_names)
    rewards_names, unknown_reward_names = resolve_requested_names(rewards_names, available_reward_names)

    if unknown_policy_names:
        warnings.warn(
            f"Ignoring unknown policy names: {', '.join(unknown_policy_names)}",
            stacklevel=2,
        )
    if unknown_reward_names:
        warnings.warn(
            f"Ignoring unknown reward names: {', '.join(unknown_reward_names)}",
            stacklevel=2,
        )

    if not policy_names:
        msg = "No valid policy names resolved from experiment.policies."
        raise ValueError(msg)
    if not rewards_names:
        msg = "No valid reward names resolved from experiment.rewards."
        raise ValueError(msg)

    normalized_algorithm_configs, _ = normalize_and_validate_agent_build(
        algorithm_configs=algorithm_configs,
        policy_names=policy_names,
        rewards_names=rewards_names,
        strict=True,
    )

    use_hcs = hcs_config.get("wts_strategy", False)

    context_config = contextual_info.get("config", {})
    feature_groups = contextual_info.get("feature_group", {})

    agents = build_agents_from_config(normalized_algorithm_configs, policy_names, rewards_names, seed=seed)

    has_sliding_window_contextual_agent = any(isinstance(agent, SlidingWindowContextualAgent) for agent in agents)
    has_contextual_agent = any(isinstance(agent, ContextualAgent) for agent in agents)
    use_context = has_contextual_agent or has_sliding_window_contextual_agent

    level = logging.DEBUG if verbose else logging.INFO
    create_logger(level)

    effective_parallel_pool_size = _effective_parallel_pool_size(
        parallel_pool_size,
        force_sequential_under_scalene=force_sequential_under_scalene,
    )

    run_started_at = time.time()
    dispatch_hook_event(coordinator_hooks, "on_run_start", run_context, fail_fast=hook_fail_fast)
    current_error_context = run_context

    try:
        for tr in sched_time_ratio:
            experiment_directory = f"{experiment_dir}time_ratio_{int(tr * 100)}/"
            Path(experiment_directory).mkdir(parents=True, exist_ok=True)

            for dataset in datasets:
                dataset_context = HookContext(
                    run_id=run_id,
                    dataset_id=dataset,
                    parallel_mode="process" if effective_parallel_pool_size > 1 else "sequential",
                    sched_time_ratio=tr,
                    extensions=extensions,
                )
                current_error_context = dataset_context
                dispatch_hook_event(coordinator_hooks, "on_dataset_start", dataset_context, fail_fast=hook_fail_fast)
                dataset_started_at = time.time()

                scenario = get_scenario_provider(
                    datasets_dir, dataset, tr, use_hcs, use_context, context_config, feature_groups
                )
                trials = scenario.max_builds

                build_config = EnvironmentBuildConfig(
                    datasets_dir=datasets_dir,
                    dataset=dataset,
                    sched_time_ratio=tr,
                    use_hcs=use_hcs,
                    use_context=use_context,
                    context_config=context_config,
                    feature_groups=feature_groups,
                    results_config=results_config,
                    checkpoint_config=checkpoint_config,
                    telemetry_config=telemetry_config,
                    algorithm_configs=normalized_algorithm_configs,
                    rewards_names=rewards_names,
                    policy_names=policy_names,
                    seed=seed,
                    run_id=run_id,
                    extensions=extensions,
                    hook_plugin_paths=hook_plugin_paths,
                    hook_fail_fast=hook_fail_fast,
                    extension=extension,
                    resolved_spec=resolved_spec,
                )

                logging.info(
                    "Starting dataset=%s time_ratio=%.2f executions=%s agents=%s trials=%s",
                    dataset,
                    tr,
                    independent_executions,
                    len(agents),
                    trials,
                )

                parallel_mode = "process" if effective_parallel_pool_size > 1 else "sequential"
                execution_plans = [
                    ExecutionPlan(
                        iteration=i + 1,
                        trials=trials,
                        level=level,
                        execution_id=build_runtime_metadata(dataset, tr, i + 1, parallel_mode)["execution_id"],
                        worker_id=str(i + 1),
                        parallel_mode=parallel_mode,
                        seed=None if seed is None else seed + i,
                    )
                    for i in range(independent_executions)
                ]

                start = time.time()
                _dispatch_executions(effective_parallel_pool_size, build_config, execution_plans)
                end = time.time()
                logging.info("Time spent running the experiments: %s\n\n", end - start)

                dispatch_hook_event(
                    coordinator_hooks,
                    "on_dataset_end",
                    dataset_context,
                    DatasetResult(
                        status="ok",
                        executions=independent_executions,
                        duration_seconds=time.time() - dataset_started_at,
                    ),
                    fail_fast=hook_fail_fast,
                )

        dispatch_hook_event(
            coordinator_hooks,
            "on_run_end",
            run_context,
            RunResult(status="ok", datasets=len(datasets), duration_seconds=time.time() - run_started_at),
            fail_fast=hook_fail_fast,
        )
    except Exception as exc:  # noqa: BLE001
        dispatch_hook_event(
            coordinator_hooks,
            "on_error",
            current_error_context,
            exc,
            fail_fast=hook_fail_fast,
        )
        raise


def run_experiment(spec_dict: dict[str, Any]) -> None:
    """Run a full experiment from a resolved spec dictionary."""
    _run_experiment_impl(spec_dict, extension=None)


def run_experiment_with_extension(spec_dict: dict[str, Any], extension: RunnerExtension) -> None:
    """Run experiment preserving orchestration while delegating extension callbacks."""
    _run_experiment_impl(spec_dict, extension=extension)
