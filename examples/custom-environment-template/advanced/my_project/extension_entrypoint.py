"""Example public entrypoint using Coleman RunnerExtension API."""

from __future__ import annotations

from coleman.api import load_spec, run_with_extension
from coleman.environment import Environment
from coleman.evaluation import NAPFDVerdictMetric
from coleman.runner import RunnerExtension, build_agents_from_config, get_scenario_provider


def _build_environment(config, runtime_metadata, agent_seed):
    agents = build_agents_from_config(
        config.algorithm_configs,
        config.policy_names,
        config.rewards_names,
        seed=agent_seed,
    )
    scenario = get_scenario_provider(
        config.datasets_dir,
        config.dataset,
        config.sched_time_ratio,
        config.use_hcs,
        config.use_context,
        config.context_config,
        config.feature_groups,
    )
    env = Environment(
        agents,
        scenario,
        NAPFDVerdictMetric(),
        results_config=config.results_config,
        checkpoint_config=config.checkpoint_config,
        telemetry_config=config.telemetry_config,
        runtime_metadata=runtime_metadata,
    )
    return env, scenario.max_builds


def main() -> None:
    spec = load_spec("run.yaml")
    extension = RunnerExtension(build_environment_fn=_build_environment)
    result = run_with_extension(spec, extension)
    print(f"run_id: {result.run_id}")  # noqa: T201


if __name__ == "__main__":
    main()
