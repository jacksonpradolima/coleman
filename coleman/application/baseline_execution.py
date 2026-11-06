"""Frozen baseline cells through the existing RunSpec extension/environment/Parquet/checkpoint path."""

from __future__ import annotations

import hashlib
import io
import math
import re
from functools import partial
from pathlib import Path
from typing import Annotated, cast

import polars as pl
from pydantic import Field, TypeAdapter

from coleman.application.baseline_agents import BaselineAgent
from coleman.application.baseline_lock import baseline_lock
from coleman.application.baseline_observations import BaselineMonitor
from coleman.application.baseline_resources import BaselineResourceGuard
from coleman.bandit import EvaluationMetricBandit
from coleman.budget import BudgetMode
from coleman.corpus.baseline_contracts import BaselinePlan
from coleman.corpus.baseline_results import BaselineCoordinate, baseline_cell_id
from coleman.corpus.materialization import validate_source_storage
from coleman.environment import Environment
from coleman.evaluation import NAPFDVerdictMetric
from coleman.results.domain import Fingerprint, Identifier
from coleman.results.versioned import VersionedContract
from coleman.runner import EnvironmentBuildConfig, RunnerExtension
from coleman.scenarios import ScenarioLoader, VirtualScenario
from coleman.spec.selected_runtime import capture_native_runtime


class NativeBaselineBinding(VersionedContract):
    """Typed local application-to-native binding; only this boundary contains a host file location.

    The domain plan itself has no filesystem/CLI/MCP dependency. Actual plan,
    runtime and compiled Parquet bytes are reverified before native execution.
    Unknown binding versions fail closed; arbitrary executable/plugin paths are
    not accepted. This is trusted local orchestration, not a remote path API.
    """

    plan_file: Identifier
    plan_fingerprint: Fingerprint
    runtime_fingerprint: Fingerprint
    coordinate: BaselineCoordinate
    input_sha256: Fingerprint
    input_size_bytes: Annotated[int, Field(strict=True, ge=1, le=64 * 1024**2)]


def read_baseline_plan(path: Path) -> BaselinePlan:
    """Read at most 16 MiB of a strict v1 plan; no network, repair or legacy migration occurs."""
    validate_source_storage(path)
    with path.open("rb") as stream:
        payload = stream.read(16 * 1024**2 + 1)
    if len(payload) > 16 * 1024**2:
        raise ValueError("Baseline frozen plan exceeds 16 MiB")
    return BaselinePlan.from_json(payload)


def baseline_dataset_name(plan: BaselinePlan) -> str:
    """Retain a safe source label or use its full interpreted digest, never interpolate a host path from source IDs."""
    name = plan.source.source_id
    return name if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}", name) else "source-" + plan.source.fingerprint[7:]


def baseline_native_frame(plan: BaselinePlan) -> pl.DataFrame:
    """Project a complete observed source into the unchanged native input schema.

    Ordinals map explicitly to source members in the frozen plan. Unknown original
    clocks stay null in that plan; native LastRun is an unused empty compatibility
    field, never a fabricated source date. Current costs/outcomes are scoring
    truth and are excluded from attainable decision views. Unknown data rejects
    rather than being filled with zero. No foreign source is fetched or admitted.
    """
    rows = []
    for build in plan.source.builds:
        for test in build.tests:
            if test.duration.status != "observed" or test.failure.status != "observed":
                raise ValueError("Native baseline projection requires complete observed source truth")
            rows.append(
                {
                    "BuildId": build.step,
                    "Name": test.test_id,
                    "Duration": cast(float, test.duration.value),
                    "Verdict": int(cast(float, test.failure.value)),
                    "CalcPrio": 0,
                    "LastRun": "",
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "BuildId": pl.Int64,
            "Name": pl.String,
            "Duration": pl.Float64,
            "Verdict": pl.Int8,
            "CalcPrio": pl.Int32,
            "LastRun": pl.String,
        },
    )


def _verify_native_input(path: Path, binding: NativeBaselineBinding, plan: BaselinePlan) -> None:
    """Verify compiled bytes and full semantic row correspondence independently of caller-supplied hashes."""
    validate_source_storage(path)
    if path.stat().st_size != binding.input_size_bytes:
        raise ValueError("Baseline compiled input size changed")
    expected = baseline_input_bytes(plan)
    if (
        len(expected) != binding.input_size_bytes
        or "sha256:" + hashlib.sha256(expected).hexdigest() != binding.input_sha256
    ):
        raise ValueError("Baseline compiled input does not encode its complete frozen source interpretation")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = binding.input_size_bytes
        while block := stream.read(1024**2):
            remaining -= len(block)
            if remaining < 0:
                raise ValueError("Baseline compiled input grew beyond its frozen size")
            digest.update(block)
    if "sha256:" + digest.hexdigest() != binding.input_sha256:
        raise ValueError("Baseline compiled input digest changed")


def baseline_input_bytes(plan: BaselinePlan) -> bytes:
    """Encode the typed native projection deterministically in memory using the recorded Polars runtime.

    The worker independently re-encodes its frozen source before accepting actual
    compiled bytes. It never decodes an arbitrary caller-supplied Parquet merely
    because the caller supplied a matching hash. Cross-runtime codec identity is
    not assumed; a changed recorded runtime gets a separate native execution.
    """
    stream = io.BytesIO()
    baseline_native_frame(plan).write_parquet(stream, compression="zstd")
    payload = stream.getvalue()
    if len(payload) > 64 * 1024**2:
        raise ValueError("Baseline compiled native input exceeds 64 MiB")
    return payload


class _BaselineScenarioLoader(ScenarioLoader):
    """Retain native rows/checkpoint position with a stable frozen-source binary64 budget.

    Polars parallel sums can vary by one or more ulps with reduction order. V1
    matrix source work uses ``math.fsum`` in canonical source order instead.
    This changes only this qualified adapter; legacy native loaders keep their
    existing interpretation. No source outcome enters attainable decision views.
    """

    def __init__(self, path: str, ratio: float, budget: float, plan: BaselinePlan):
        """Cache full source work in seconds; eligibility has already required complete finite durations."""
        super().__init__(path, ratio, BudgetMode.RATIO, budget)
        self._source_work = tuple(
            math.fsum(test.duration.value for test in build.tests if test.duration.value is not None)
            for build in plan.source.builds
        )

    def get(self) -> VirtualScenario | None:
        """Normalize source work and its ratio budget before native bandit construction, including resume."""
        scenario = super().get()
        if scenario is not None:
            self.total_build_duration = self._source_work[int(self.current_build) - 1]
            scenario.total_build_duration = self.total_build_duration
            scenario.available_time = self.compute_available_time(
                self.total_build_duration, scenario.get_testcases_df().height
            )
        return scenario


class _BaselineEnvironment(Environment):
    """Reuse actual native orchestration with serialized attempts and no undeclared final forecast."""

    baseline_root: Path
    baseline_guard: BaselineResourceGuard
    baseline_plan: BaselinePlan
    baseline_binding: NativeBaselineBinding
    baseline_run_id: str

    def run_single(self, experiment, trials=100, bandit_type=EvaluationMetricBandit, restore=True, forecast_next=False):
        """Hold the child-owned lock through ordinary native result/checkpoint publication.

        The worker lock survives a killed parent while this worker is alive. A
        subsequent parent cannot race its active native execution. Resource checks
        are cooperative at decision/publication boundaries; all original native
        recovery, flush and checkpoint logic remains in Environment.run_single.
        """
        if forecast_next:
            raise ValueError("A frozen baseline matrix cannot invent an out-of-source forecast")
        with baseline_lock(self.baseline_root / "native-execution.lock"):
            self.baseline_guard.check()
            super().run_single(experiment, trials, bandit_type, restore, forecast_next=False)
            self.baseline_guard.check()
            from coleman.application.baseline_storage import publish_native_baseline_completion

            publish_native_baseline_completion(
                self.baseline_root,
                self.baseline_plan,
                self.baseline_binding.coordinate,
                self.baseline_binding.fingerprint,
                self.baseline_binding.runtime_fingerprint,
                str(self.scenario_provider),
                self.baseline_guard,
            )

    def run_prioritization(self, agent, bandit, bandit_duration, experiment, t, virtual_scenario):
        """Check common resources before delegating the complete native decision unchanged."""
        self.baseline_guard.check()
        return super().run_prioritization(agent, bandit, bandit_duration, experiment, t, virtual_scenario)


def _binding_plan(config: EnvironmentBuildConfig, agent_seed: int | None) -> tuple[NativeBaselineBinding, BaselinePlan]:
    """Validate typed binding, declared seed/cell and actual installed implementation/runtime identity."""
    binding = NativeBaselineBinding.from_json(config.extensions["baseline_matrix_v1"])
    plan = read_baseline_plan(Path(binding.plan_file))
    cell = binding.coordinate
    if plan.fingerprint != binding.plan_fingerprint or cell.cell_id != baseline_cell_id(
        plan, cell.method, cell.seed, cell.budget
    ):
        raise ValueError("Baseline native binding differs from its frozen plan/cell")
    if agent_seed != cell.seed or capture_native_runtime().fingerprint != binding.runtime_fingerprint:
        raise ValueError("Baseline native seed or installed runtime differs from its frozen binding")
    if (
        config.dataset != baseline_dataset_name(plan)
        or config.budget_mode != BudgetMode.RATIO
        or config.budget_value != float(cell.budget.value)
        or config.use_hcs
        or config.use_context
        or config.hook_plugin_paths
    ):
        raise ValueError("Baseline native source/budget/topology differs from its qualified matrix projection")
    return binding, plan


def build_baseline_environment(
    config: EnvironmentBuildConfig, metadata: dict[str, str], agent_seed: int | None, *, attempt_id: str | None = None
):
    """Create existing native components for one typed cell; optionally label process-local execution metadata.

    The matrix driver supplies its physical invocation ID outside RunSpec so
    checkpoint/logical identity stays stable. Updating this worker-local metadata
    binds native operational rows and explicit raw attempt IDs to that durable
    invocation event; it does not reseed or create another statistical trial.
    """
    binding, plan = _binding_plan(config, agent_seed)
    if config.run_id is None or config.resolved_spec is None:
        raise ValueError("Baseline native execution requires resolved run/specification provenance")
    if (
        config.resolved_spec.execution.independent_executions != 1
        or config.resolved_spec.execution.parallel_pool_size != 2
        or metadata.get("parallel_mode") != "process"
        or config.telemetry_config.get("enabled", False)
    ):
        raise ValueError("Baseline native execution requires one fresh process task and disabled optional telemetry")
    if not config.results_config.get("enabled") or config.results_config.get("sink") != "parquet":
        raise ValueError("Qualified baseline matrix execution requires the existing Parquet sink")
    if not config.checkpoint_config.get("enabled") or config.checkpoint_config.get("interval") != 1:
        raise ValueError("Qualified baseline matrix execution requires per-build native checkpoints")
    path = Path(config.datasets_dir) / config.dataset / "features-engineered.parquet"
    _verify_native_input(path, binding, plan)
    root = Path(config.results_config["out_dir"]).parent
    if root.name != config.run_id:
        raise ValueError("Baseline native artifact directory does not match its resolved run identity")
    guard = BaselineResourceGuard(plan.limits, root)
    if attempt_id is not None:
        metadata["execution_id"] = TypeAdapter(Identifier).validate_python(attempt_id)
    source = _BaselineScenarioLoader(str(path), config.sched_time_ratio, float(binding.coordinate.budget.value), plan)
    env = _BaselineEnvironment(
        [BaselineAgent(plan, binding.coordinate.method, binding.coordinate.seed)],
        source,
        NAPFDVerdictMetric(),
        results_config=config.results_config,
        checkpoint_config=config.checkpoint_config,
        telemetry_config=config.telemetry_config,
        runtime_metadata=metadata,
    )
    env.baseline_root, env.baseline_guard = root, guard
    env.baseline_plan, env.baseline_binding, env.baseline_run_id = plan, binding, config.run_id
    env.monitor = BaselineMonitor(
        env,
        plan,
        binding.coordinate,
        config.run_id,
        binding.fingerprint,
        root,
        guard,
    )
    return env, source.max_builds


BASELINE_EXTENSION = RunnerExtension(build_environment_fn=build_baseline_environment)


def baseline_extension(attempt_id: str) -> RunnerExtension:
    """Bind one physical invocation outside logical RunSpec identity, reusing serializable native callbacks."""
    validated = TypeAdapter(Identifier).validate_python(attempt_id)
    return RunnerExtension(build_environment_fn=partial(build_baseline_environment, attempt_id=validated))
