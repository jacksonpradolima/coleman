"""Stage-bounded adapter to the existing TCPCI runner, agents and native recovery."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from pathlib import Path

import polars as pl

from coleman.api import run_with_extension
from coleman.application.workflow_checkpoint import WorkflowCheckpointStore
from coleman.bandit import EvaluationMetricBandit
from coleman.environment import Environment
from coleman.evaluation import NAPFDVerdictMetric
from coleman.results.domain import fingerprint
from coleman.runner import EnvironmentBuildConfig, RunnerExtension, build_agents_from_config
from coleman.scenarios.loaders import ScenarioLoader
from coleman.spec import RunSpec
from coleman.workflows.contracts import ExperimentPlan, Trial, WorkflowError


class _WorkflowEnvironment(Environment):
    """Restrict native iteration to a declared prefix and report real checkpoint commits."""

    progress: Callable[[int], None]

    def run_single(self, experiment, trials=100, bandit_type=EvaluationMetricBandit, restore=True, forecast_next=False):
        """Delegate native recovery/learning unchanged, without an out-of-stage forecast."""
        super().run_single(experiment, trials, bandit_type, restore, forecast_next=False)

    def save_periodically(self, restore, t, experiment, bandit, interval=None):
        """Publish progress only after the actual native result/checkpoint commit succeeds."""
        super().save_periodically(restore, t, experiment, bandit, interval)
        self.progress(t)


class _WorkflowScenario(ScenarioLoader):
    """Filter stage membership before native eager/cache outcome materialization."""

    def __init__(self, config: EnvironmentBuildConfig, allowed: tuple[int, ...]):
        """Keep original byte/path identity while exposing only allowed native build rows."""
        self._allowed = allowed
        path = Path(config.datasets_dir) / config.dataset / "features-engineered.parquet"
        super().__init__(str(path), config.sched_time_ratio, config.budget_mode, config.budget_value)

    def _read_testcases(self, tcfile: str) -> pl.LazyFrame:
        """Push filtering before eager build maps, outcome delivery, iteration and forecasts."""
        return super()._read_testcases(tcfile).filter(pl.col("BuildId").is_in(self._allowed))

    def get(self):
        """Use a declared binary64 fsum for modeled source work, independent of Polars reduction threads.

        This workflow-only projection leaves ordinary native loader behavior and
        stored historical artifacts unchanged. Actual testcase costs still cross
        the native binary32 boundary; their rounding is not a physical measurement.
        """
        scenario = super().get()
        if scenario is not None:
            assert self._current_build_df is not None
            self.total_build_duration = math.fsum(self._current_build_df["Duration"].to_list())
            scenario.total_build_duration = self.total_build_duration
            scenario.available_time = self.compute_available_time(
                self.total_build_duration, self._current_build_df.height
            )
        return scenario


def _build(
    config: EnvironmentBuildConfig,
    metadata: dict[str, str],
    seed: int | None,
    *,
    steps: int,
    allowed: tuple[int, ...],
    attempt_id: str,
    binding: str,
    minimum_step: int,
    progress: Callable[[int], None],
):
    """Build actual native components; only the execution frontier is stage bounded."""
    metadata["execution_id"] = attempt_id
    scenario = _WorkflowScenario(config, allowed)
    agents = build_agents_from_config(config.algorithm_configs, config.policy_names, config.rewards_names, seed=seed)
    env = _WorkflowEnvironment(
        agents,
        scenario,
        NAPFDVerdictMetric(),
        results_config=config.results_config,
        checkpoint_config=config.checkpoint_config,
        telemetry_config=config.telemetry_config,
        runtime_metadata=metadata,
    )
    env.progress = progress
    env.checkpoint_store = WorkflowCheckpointStore(
        str(config.checkpoint_config["base_dir"]), binding, minimum_step=minimum_step
    )
    return env, steps


def execute_native(
    root: Path,
    plan: ExperimentPlan,
    trial: Trial,
    steps: int,
    progress: Callable[[int], None],
    *,
    attempt_id: str,
    minimum_step: int = 0,
) -> Path:
    """Run/continue one real logical coordinate with a stable native configuration.

    Parameters
    ----------
    root : pathlib.Path
        Job-owned artifact root containing verified registered input bytes.
    plan : ExperimentPlan
        Immutable complete native specifications and compatibility binding.
    trial : Trial
        Saved policy/seed/hyperparameter coordinate.
    steps : int
        Required prefix frontier, in processed native build ordinals. Continuation
        changes this frontier only; the RunSpec/data/checkpoint identity stays fixed.
    progress : callable
        Trusted host callback after a real durable per-build checkpoint commit.
        Callback exceptions propagate; they cannot certify an uncommitted result.
    attempt_id : str
        Host-issued physical reservation ID, supplied outside RunSpec so native
        logical/checkpoint identity remains stable through continuation.
    minimum_step : int
        Per-trial native build ordinal already acknowledged by durable metadata.
        Older restored state is rejected before RNG or native output recovery.

    Returns
    -------
    pathlib.Path
        Actual native run directory. Its existence alone never means completion.

    Raises
    ------
    ValueError, OSError
        Invalid frontier, native compatibility/recovery or artifact failure.

    Notes
    -----
    Uses the existing runner, Random/UCB agents, RNFail, Parquet and checkpoint
    engine. No subject test command is launched. Only host-owned input/output paths
    replace the saved abstract paths; native run-ID normalization remains unchanged.
    """
    if not 1 <= steps <= len(plan.dataset.build_ids):
        raise WorkflowError("Invalid native workflow frontier")
    if type(minimum_step) is not int or not 0 <= minimum_step <= steps:
        raise WorkflowError("Acknowledged native frontier exceeds the requested stage")
    payload = RunSpec.model_validate_json(trial.resolved_spec_json).model_dump()
    payload["experiment"]["datasets_dir"] = str(root / "inputs")
    payload["results"]["out_dir"] = str(root / "native" / trial.trial_id[7:])
    spec = RunSpec.model_validate(payload)
    result = run_with_extension(
        spec,
        RunnerExtension(
            build_environment_fn=partial(
                _build,
                steps=steps,
                allowed=plan.dataset.build_ids[:steps],
                progress=progress,
                attempt_id=attempt_id,
                binding=fingerprint([plan.fingerprint, trial.trial_id]),
                minimum_step=minimum_step,
            )
        ),
    )
    if result.artifacts_dir is None:
        raise WorkflowError("Native execution did not identify its artifacts")
    return Path(result.artifacts_dir)
