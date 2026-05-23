"""Environment for agent-bandit interactions in Coleman.

This module contains the `Environment` class which simulates the agent's interactions and
collects results during the reinforcement learning process. The environment provides a
platform where agents learn from interactions with the bandit to make better decisions
over time.

The module also contains the following features:

- Mechanism to reset the environment and agent's memories.
- Ability to run a single experiment or multiple experiments.
- Support for both simple and contextual agents.
- Periodic saving of experiments via the checkpoint store.
- Facilities for creating and storing results obtained during experiments.
- Support for scenarios with variants, commonly found in Heterogeneous Computing Systems (HCS).
- Helper methods for loading and saving experiment states for recovery purposes.

Classes
-------
Environment
    Represents the learning environment where agents interact with bandits.
"""

import logging
import time
import warnings
from typing import Any

import numpy as np
import polars as pl

from coleman.agent import ContextualAgent, SlidingWindowContextualAgent
from coleman.bandit import EvaluationMetricBandit
from coleman.checkpoint.checkpoint_store import CheckpointStore, NullCheckpointStore
from coleman.checkpoint.state import CheckpointPayload
from coleman.environment_base import AbstractEnvironment
from coleman.results.parquet_sink import ParquetSink
from coleman.results.sink_base import NullSink, ResultsSink
from coleman.scenarios import HCSScenarioLoader, VirtualHCSScenario
from coleman.telemetry.otel import get_telemetry
from coleman.telemetry.resources import ProcessResourceTracker, ResourceSnapshot
from coleman.utils.monitor import MonitorCollector
from coleman.utils.monitor_params import CollectParams

logger = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=FutureWarning)


class Environment(AbstractEnvironment):
    """The environment class that simulates the agent's interactions and collects results.

    Parameters
    ----------
    agents : list
        List of agent instances participating in the simulation.
    scenario_provider : object
        The scenario provider supplying test case data.
    evaluation_metric : EvaluationMetric
        The evaluation metric used to assess prioritization performance.
    results_config : dict or None
        Configuration for the results sink (from the ``results`` section of the spec).
        When ``None``, a ``NullSink`` is used.
    checkpoint_config : dict or None
        Configuration for checkpoints (from the ``checkpoint`` section of the spec).
        When ``None``, a ``NullCheckpointStore`` is used.
    telemetry_config : dict or None
        Configuration for telemetry (from the ``telemetry`` section of the spec).
        When ``None``, a ``NoOpTelemetry`` is used.

    Attributes
    ----------
    agents : list
        List of agent instances.
    scenario_provider : object
        The scenario provider instance.
    evaluation_metric : EvaluationMetric
        The evaluation metric instance.
    monitor : MonitorCollector
        Monitor for collecting feedback during the process.
    variant_monitors : dict
        Dictionary of monitors for each variant.
    checkpoint_store : CheckpointStore
        Store for saving/loading experiment checkpoints.
    telemetry : Telemetry or NoOpTelemetry
        Telemetry facade for metrics and traces.
    """

    def __init__(  # noqa: PLR0913
        self,
        agents,
        scenario_provider,
        evaluation_metric,
        results_config: dict | None = None,
        checkpoint_config: dict | None = None,
        telemetry_config: dict | None = None,
        runtime_metadata: dict[str, Any] | None = None,
    ):
        """Initialize the Environment.

        Parameters
        ----------
        agents : list
            List of agent instances participating in the simulation.
        scenario_provider : object
            The scenario provider supplying test case data.
        evaluation_metric : EvaluationMetric
            The evaluation metric used to assess prioritization performance.
        results_config : dict or None
            Results configuration dict.
        checkpoint_config : dict or None
            Checkpoint configuration dict.
        telemetry_config : dict or None
            Telemetry configuration dict.
        """
        self.agents = agents
        self.scenario_provider = scenario_provider
        self.evaluation_metric = evaluation_metric
        self.runtime_metadata: dict[str, str] = {}
        self.set_runtime_metadata(runtime_metadata)
        self._resource_tracker: ProcessResourceTracker | None = None
        self._last_resource_snapshot: ResourceSnapshot | None = None

        # Build the results sink from config
        self._results_config = results_config or {}
        self._sink = self._build_sink(self._results_config)

        # Build the checkpoint store from config
        ckpt_cfg = checkpoint_config or {}
        if ckpt_cfg.get("enabled", False):
            from coleman.checkpoint.checkpoint_store import LocalCheckpointStore

            self.checkpoint_store: CheckpointStore = LocalCheckpointStore(
                base_dir=ckpt_cfg.get("base_dir", "checkpoints"),
            )
            self._checkpoint_interval: int = ckpt_cfg.get("interval", 50000)
        else:
            self.checkpoint_store = NullCheckpointStore()
            self._checkpoint_interval = 50000

        # Build the telemetry facade from config
        tel_cfg = telemetry_config or {}
        self.telemetry = get_telemetry(
            enabled=tel_cfg.get("enabled", False),
            service_name=tel_cfg.get("service_name", "coleman"),
            otlp_endpoint=tel_cfg.get("otlp_endpoint", "http://localhost:4318"),
            export_interval_millis=tel_cfg.get("export_interval_millis", 5000),
            resource_attributes=tel_cfg.get("resource_attributes"),
        )

        self.monitor: MonitorCollector = MonitorCollector(sink=self._sink)
        self.variant_monitors: dict[str, MonitorCollector] = {}
        self.forecast_prioritizations: list[dict[str, Any]] = []
        self.reset()

    @staticmethod
    def _build_sink(results_config: dict) -> ResultsSink:
        """Create a ResultsSink from the ``[results]`` config section.

        Parameters
        ----------
        results_config : dict
            Config dict with keys ``enabled``, ``sink``, ``out_dir``, etc.

        Returns
        -------
        ResultsSink
        """
        if not results_config.get("enabled", False):
            return NullSink()

        sink_type = results_config.get("sink", "parquet")
        if sink_type == "parquet":
            top_k_raw = results_config.get("top_k_prioritization", 0)
            top_k = top_k_raw if top_k_raw and top_k_raw > 0 else None
            return ParquetSink(
                out_dir=results_config.get("out_dir", "./runs"),
                batch_size=results_config.get("batch_size", 1000),
                top_k=top_k,
            )
        if sink_type == "duckdb":
            from coleman.results.duckdb_sink import DuckDBSink

            duckdb_cfg = results_config.get("duckdb", {})
            if not isinstance(duckdb_cfg, dict):
                msg = "results.duckdb must be a dictionary of DuckDBSink keyword arguments"
                raise TypeError(msg)

            top_k_raw = results_config.get("top_k_prioritization", 0)
            top_k = top_k_raw if top_k_raw and top_k_raw > 0 else None
            return DuckDBSink(
                out_dir=results_config.get("out_dir", "./runs"),
                batch_size=results_config.get("batch_size", 1000),
                top_k=top_k,
                **duckdb_cfg,
            )
        if sink_type == "clickhouse":
            from coleman.results.clickhouse_sink import ClickHouseSink

            clickhouse_cfg = results_config.get("clickhouse", {})
            if not isinstance(clickhouse_cfg, dict):
                msg = "results.clickhouse must be a dictionary of ClickHouseSink keyword arguments"
                raise TypeError(msg)
            return ClickHouseSink(**clickhouse_cfg)
        msg = f"Unsupported results sink type: {sink_type!r}. Valid options are 'parquet', 'duckdb' or 'clickhouse'."
        raise ValueError(msg)

    def reset(self):
        """Reset the environment for a new simulation."""
        self.reset_agents_memory()
        # Create monitors backed by the configured sink
        self.monitor = MonitorCollector(sink=self._sink)
        self.variant_monitors = {}

        if isinstance(self.scenario_provider, HCSScenarioLoader) and self.scenario_provider.get_total_variants() > 0:
            for variant in self.scenario_provider.get_all_variants():
                self.variant_monitors[variant] = MonitorCollector(sink=self._sink)

    def reset_agents_memory(self):
        """Reset all agents' memory to an initial state."""
        for agent in self.agents:
            agent.reset()

    def set_runtime_metadata(self, runtime_metadata: dict[str, Any] | None = None) -> None:
        """Set execution-scoped metadata used for telemetry and persisted results."""
        default_metadata = {
            "execution_id": "unknown",
            "worker_id": "0",
            "parallel_mode": "sequential",
        }
        incoming_metadata = runtime_metadata or {}
        self.runtime_metadata = {key: str(value) for key, value in default_metadata.items()}
        self.runtime_metadata.update({key: str(value) for key, value in incoming_metadata.items() if value is not None})

    def _current_runtime_metadata(self) -> dict[str, str]:
        """Return execution metadata attached to the current environment run."""
        return dict(self.runtime_metadata)

    def _experiment_telemetry_attributes(self, experiment: int) -> dict[str, Any]:
        """Build low-cardinality telemetry attributes for one experiment run."""
        attributes = {
            "scenario": str(self.scenario_provider),
            "experiment": str(experiment),
            "time_ratio": f"{self.scenario_provider.get_avail_time_ratio():.2f}",
        }
        attributes.update(self._current_runtime_metadata())
        return attributes

    def _agent_telemetry_attributes(self, experiment: int, policy: str, reward_function: str) -> dict[str, Any]:
        """Build telemetry attributes for one agent execution within an experiment."""
        attrs = self._experiment_telemetry_attributes(experiment)
        attrs.update(
            {
                "policy": policy,
                "reward_function": reward_function,
            }
        )
        return attrs

    def _sample_resources(self) -> ResourceSnapshot:
        """Capture one process resource sample, creating the tracker lazily when necessary."""
        if self._resource_tracker is None:
            self._resource_tracker = ProcessResourceTracker()

        snapshot = self._resource_tracker.sample()
        self._last_resource_snapshot = snapshot
        return snapshot

    def _try_restore_checkpoint(self, experiment):
        """Attempt to restore experiment state from the checkpoint store.

        Parameters
        ----------
        experiment : int
            Current experiment number.

        Returns
        -------
        tuple[int, object | None]
            A ``(restore_step, bandit)`` pair.  When no checkpoint exists the
            defaults ``(1, None)`` are returned.
        """
        run_id = str(self.scenario_provider)
        payload = self.checkpoint_store.load(run_id, experiment)
        if payload is None:
            return 1, None

        self.agents = payload.agents if payload.agents is not None else self.agents
        self.scenario_provider.last_build(payload.step)
        return payload.step + 1, payload.bandit

    def run_single(
        self,
        experiment,
        trials=100,
        bandit_type=EvaluationMetricBandit,
        restore=True,
        forecast_next=True,
    ):
        """Execute a single simulation experiment.

        Parameters
        ----------
        experiment : int
            Current experiment number.
        trials : int, optional
            The max number of scenarios that will be analyzed. Default is 100.
        bandit_type : type, optional
            The bandit class to use. Default is ``EvaluationMetricBandit``.
        restore : bool, optional
            Restore the experiment if it fails (e.g., energy down). Default is True.
        forecast_next : bool, optional
            Compute and store the next prioritization after the final evaluated
            build without running evaluation/reward updates. Default is True.
        """
        # The agent need to learn from the beginning for each experiment
        self.reset_agents_memory()

        # create a bandit (initially is None to know that is the first bandit
        bandit = None

        # restore to step
        restore_step = 1
        last_step = restore_step - 1
        last_virtual_scenario = None
        last_bandit_duration = 0.0
        experiment_attrs = self._experiment_telemetry_attributes(experiment)
        self._resource_tracker = ProcessResourceTracker()
        self._last_resource_snapshot = None
        self.forecast_prioritizations = []

        # Each independent execution must restart the scenario iteration from the beginning
        # unless a checkpoint explicitly restores to a later step.
        self.scenario_provider.last_build(0)

        self.telemetry.mark_run_started(experiment_attrs)

        try:
            if restore:
                restore_step, bandit = self._try_restore_checkpoint(experiment)

            logger.info(
                "Starting experiment=%s scenario=%s from_step=%s total_trials=%s",
                experiment,
                self.scenario_provider,
                restore_step,
                trials,
            )

            # Test Budget percentage
            avail_time_ratio = self.scenario_provider.get_avail_time_ratio()

            # For each "virtual scenario" I must analyse it and evaluate it
            for t, virtual_scenario in enumerate(self.scenario_provider, start=restore_step):
                # The max number of scenarios that will be analyzed
                if t > trials:
                    break

                # Time Budget
                available_time = virtual_scenario.get_available_time()
                self.evaluation_metric.update_available_time(available_time)

                # Compute time
                start_bandit = time.time()

                if bandit is None:
                    bandit = bandit_type(virtual_scenario.get_testcases(), self.evaluation_metric)
                else:
                    # Update bandit (the arms = TC available) for all agents
                    bandit.update_arms(virtual_scenario.get_testcases())

                end_bandit = time.time()

                bandit_duration = end_bandit - start_bandit
                last_bandit_duration = bandit_duration
                self.telemetry.record_latency("bandit_update", bandit_duration, attributes=experiment_attrs)

                # we can analyse the same "moment/scenario t" for "i agents"
                for _, agent in enumerate(self.agents):
                    # Update again because the variants
                    # each variant has its own test budget size, that is, different from the whole system
                    self.evaluation_metric.update_available_time(available_time)

                    action, end, exp_name, start = self.run_prioritization(
                        agent, bandit, bandit_duration, experiment, t, virtual_scenario
                    )

                    # If we are working with HCS scenario and there are variants
                    if isinstance(virtual_scenario, VirtualHCSScenario) and len(virtual_scenario.get_variants()) > 0:
                        self.run_prioritization_hcs(
                            agent,
                            action,
                            avail_time_ratio,
                            bandit_duration,
                            end,
                            exp_name,
                            experiment,
                            start,
                            t,
                            virtual_scenario,
                        )

                self.telemetry.record_cycle(experiment_attrs)
                self.save_periodically(restore, t, experiment, bandit)
                last_step = t
                last_virtual_scenario = virtual_scenario

            if forecast_next:
                self._run_forecast_prioritization(
                    experiment=experiment,
                    next_step=last_step + 1,
                    bandit=bandit,
                    virtual_scenario=last_virtual_scenario,
                    bandit_duration=last_bandit_duration,
                )

            logger.info("Finished experiment=%s scenario=%s", experiment, self.scenario_provider)
        finally:
            run_snapshot = self._sample_resources()
            self.telemetry.record_experiment_resources(
                run_snapshot.wall_time_seconds,
                run_snapshot.cpu_time_seconds,
                run_snapshot.peak_rss_mib,
                attributes=experiment_attrs,
            )
            self.telemetry.mark_run_finished(experiment_attrs)
            self.telemetry.flush()

    def _run_forecast_prioritization(self, experiment, next_step, bandit, virtual_scenario, bandit_duration):
        """Compute the next prioritization forecast without evaluating outcomes."""
        if bandit is None or virtual_scenario is None:
            return

        for agent in self.agents:
            if type(agent) in [ContextualAgent, SlidingWindowContextualAgent]:
                agent.update_context(virtual_scenario.get_context_features())
                agent.update_features(virtual_scenario.get_features())
                policy_name = f"{str(agent)}_{virtual_scenario.get_feature_group()}"
            else:
                policy_name = str(agent)

            reward_function_name = str(agent.get_reward_function())
            telemetry_attrs = self._agent_telemetry_attributes(experiment, policy_name, reward_function_name)

            agent.update_bandit(bandit)
            start = time.time()
            action = agent.choose()
            end = time.time()

            prioritization_time = (end - start) + bandit_duration
            self.telemetry.record_latency("forecast_prioritization", prioritization_time, attributes=telemetry_attrs)

            forecast = {
                "experiment": experiment,
                "step": next_step,
                "policy": policy_name,
                "reward_function": reward_function_name,
                "prioritization_order": action,
            }
            self.forecast_prioritizations.append(forecast)
            logger.info(
                "Forecast experiment=%s step=%s policy=%s reward=%s prioritization=%s",
                experiment,
                next_step,
                policy_name,
                reward_function_name,
                action,
            )

    def run_prioritization(  # pylint: disable=too-many-positional-arguments
        self, agent, bandit, bandit_duration, experiment, t, virtual_scenario
    ):
        """Run the prioritization process for a given agent and scenario.

        Parameters
        ----------
        agent : Agent
            The agent that is being used for the prioritization.
        bandit : Bandit
            The bandit mechanism used for choosing actions.
        bandit_duration : float
            Time taken by the bandit process.
        experiment : int
            The current experiment number.
        t : int
            The current step or iteration of the simulation.
        virtual_scenario : VirtualScenario
            The virtual scenario being considered.

        Returns
        -------
        tuple
            A tuple containing the chosen action, the ending time, the
            experiment name, and the starting time.
        """
        # MAB or CMAB
        if type(agent) in [ContextualAgent, SlidingWindowContextualAgent]:
            # For a Contextual agent, we first update the current context information
            agent.update_context(virtual_scenario.get_context_features())
            agent.update_features(virtual_scenario.get_features())

            exp_name = f"{str(agent)}_{virtual_scenario.get_feature_group()}"
        else:
            exp_name = str(agent)

        reward_function_name = str(agent.get_reward_function())
        telemetry_attrs = self._agent_telemetry_attributes(experiment, exp_name, reward_function_name)

        # Update the bandit inside the agent.
        # This loop also update the actions available for policy choose
        agent.update_bandit(bandit)

        # Compute time
        start = time.time()

        # Choose action (Prioritized Test Suite List) from agent (from current Q estimate)
        action = agent.choose()
        after_choose = time.time()

        # Pick up reward from bandit for chosen action
        # Submit prioritized test cases for evaluation step the environment and get new measurements
        start_evaluation = time.time()
        metric = agent.bandit.pull(action)

        # Update Q action-value estimates (Reward Functions)
        agent.observe(metric)

        # Compute end time
        end = time.time()

        prioritization_time = (after_choose - start) + bandit_duration
        evaluation_time = end - start_evaluation
        resource_snapshot = self._sample_resources()
        self.telemetry.record_latency("prioritization", prioritization_time, attributes=telemetry_attrs)
        self.telemetry.record_latency("evaluation", evaluation_time, attributes=telemetry_attrs)
        self.telemetry.record_fitness(metric.fitness, metric.cost, attributes=telemetry_attrs)
        self.telemetry.record_resource_snapshot(
            resource_snapshot.current_rss_mib,
            resource_snapshot.peak_rss_mib,
            resource_snapshot.cpu_utilization_percent,
            attributes=telemetry_attrs,
        )

        logger.debug(
            "Exp: %s - Ep: %s - Name: %s (%s) - NAPFD/APFDc: %.4f/%.4f",
            experiment,
            t,
            exp_name,
            reward_function_name,
            metric.fitness,
            metric.cost,
        )

        # Collect the data during the experiment
        params = CollectParams(
            scenario_provider=self.scenario_provider,
            available_time=virtual_scenario.get_available_time(),
            experiment=experiment,
            t=t,
            policy=exp_name,
            reward_function=reward_function_name,
            metric=metric,
            total_build_duration=self.scenario_provider.total_build_duration,
            prioritization_time=prioritization_time,
            rewards=np.mean(agent.last_reward),
            prioritization_order=action,
            execution_id=self.runtime_metadata.get("execution_id"),
            worker_id=self.runtime_metadata.get("worker_id"),
            parallel_mode=self.runtime_metadata.get("parallel_mode"),
            process_memory_rss_mib=resource_snapshot.current_rss_mib,
            process_memory_peak_rss_mib=resource_snapshot.peak_rss_mib,
            process_cpu_utilization_percent=resource_snapshot.cpu_utilization_percent,
            process_cpu_time_seconds=resource_snapshot.cpu_time_seconds,
            wall_time_seconds=resource_snapshot.wall_time_seconds,
        )

        self.monitor.collect(params)

        return action, end, exp_name, start

    def run_prioritization_hcs(  # pylint: disable=too-many-positional-arguments
        self, agent, action, avail_time_ratio, bandit_duration, end, exp_name, experiment, start, t, virtual_scenario
    ):
        """Run the prioritization process for a given agent and HCS scenario.

        Parameters
        ----------
        agent : Agent
            The agent that is being used for the prioritization.
        action : list of str
            The chosen action by the agent.
        avail_time_ratio : float
            The available time ratio for the experiment.
        bandit_duration : float
            Time taken by the bandit process.
        end : float
            The ending time of the process.
        exp_name : str
            The name of the experiment.
        experiment : int
            The current experiment number.
        start : float
            The starting time of the process.
        t : int
            The current step or iteration of the simulation.
        virtual_scenario : VirtualHCSScenario
            The virtual HCS scenario being considered.
        """
        # Get the variants that exist in the current commit
        variants = virtual_scenario.get_variants()
        resource_snapshot = self._last_resource_snapshot

        # For each variant I will evaluate the impact of the main prioritization
        for variant in variants["Variant"].unique():
            # Get the data from current variant
            df = variants.filter(pl.col("Variant") == variant)

            # Order by the test cases according to the main prioritization
            action_map = {name: idx + 1 for idx, name in enumerate(action)}
            df = df.with_columns([pl.col("Name").replace_strict(action_map, default=0).alias("CalcPrio")])
            df = df.sort("CalcPrio")

            total_build_duration = df["Duration"].sum()
            total_time = total_build_duration * avail_time_ratio

            # Update the available time concerning the variant build duration
            self.evaluation_metric.update_available_time(total_time)

            # Submit prioritized test cases for evaluation step and get new measurements
            self.evaluation_metric.evaluate(df)

            # Save the information (collect the data)
            params = CollectParams(
                scenario_provider=self.scenario_provider,
                available_time=total_time,
                experiment=experiment,
                t=t,
                policy=exp_name,
                reward_function=str(agent.get_reward_function()),
                metric=self.evaluation_metric,
                total_build_duration=total_build_duration,
                prioritization_time=(end - start) + bandit_duration,
                rewards=0,
                prioritization_order=df["Name"].to_list(),
                execution_id=self.runtime_metadata.get("execution_id"),
                worker_id=self.runtime_metadata.get("worker_id"),
                parallel_mode=self.runtime_metadata.get("parallel_mode"),
                process_memory_rss_mib=resource_snapshot.current_rss_mib if resource_snapshot else None,
                process_memory_peak_rss_mib=resource_snapshot.peak_rss_mib if resource_snapshot else None,
                process_cpu_utilization_percent=resource_snapshot.cpu_utilization_percent
                if resource_snapshot
                else None,  # noqa: E501
                process_cpu_time_seconds=resource_snapshot.cpu_time_seconds if resource_snapshot else None,
                wall_time_seconds=resource_snapshot.wall_time_seconds if resource_snapshot else None,
                variant=variant,
            )

            self.variant_monitors[variant].collect(params)

    def save_periodically(  # pylint: disable=too-many-positional-arguments
        self, restore, t, experiment, bandit, interval=None
    ):
        """Save the experiment periodically based on a predefined interval.

        Parameters
        ----------
        restore : bool
            Flag to indicate if the experiment should be restored.
        t : int
            The current step or iteration of the simulation.
        experiment : int
            The current experiment number.
        bandit : Bandit
            The current bandit being used in the simulation.
        interval : int or None, optional
            The interval at which the experiment should be saved.
            When ``None``, uses the configured checkpoint interval.
        """
        save_interval = interval if interval is not None else self._checkpoint_interval
        # Save experiment each X builds
        if restore and t % save_interval == 0:
            self.save_experiment(experiment, t, bandit)

    def run(
        self,
        experiments=1,
        trials=100,
        bandit_type=EvaluationMetricBandit,
        restore=True,
        forecast_next=True,
    ):
        """Execute a simulation over multiple experiments.

        Parameters
        ----------
        experiments : int, optional
            Number of experiments. Default is 1.
        trials : int, optional
            The max number of scenarios that will be analyzed. Default is 100.
        bandit_type : type, optional
            The bandit class to use. Default is ``EvaluationMetricBandit``.
        restore : bool, optional
            Restore the experiment if it fails. Default is True.
        forecast_next : bool, optional
            Compute and store the next prioritization forecast after each
            experiment. Default is True.
        """
        self.reset()

        for exp in range(experiments):
            self.run_single(exp, trials, bandit_type, restore, forecast_next)

    def store_experiment(self):
        """Flush and persist the results collected during the experiment."""
        self.monitor.flush()
        for variant_monitor in self.variant_monitors.values():
            variant_monitor.flush()
        self.telemetry.flush()

    def load_experiment(self, experiment):
        """Load a backup of the experiment from the checkpoint store.

        Parameters
        ----------
        experiment : int
            The current experiment number.

        Returns
        -------
        CheckpointPayload or None
            The loaded checkpoint payload, or ``None`` if not found.
        """
        run_id = str(self.scenario_provider)
        return self.checkpoint_store.load(run_id, experiment)

    def save_experiment(self, experiment, t, bandit):
        """Save a checkpoint for the experiment.

        Parameters
        ----------
        experiment : int
            The current experiment number.
        t : int
            The current step or iteration of the simulation.
        bandit : Bandit
            The current bandit being used in the simulation.

        Raises
        ------
        Exception
            If there is an error saving the experiment.
        """
        try:
            self.monitor.flush()
            for variant_monitor in self.variant_monitors.values():
                variant_monitor.flush()
            payload = CheckpointPayload(
                run_id=str(self.scenario_provider),
                experiment=experiment,
                step=t,
                agents=self.agents,
                bandit=bandit,
            )
            self.checkpoint_store.save(payload)
            self.telemetry.record_checkpoint_save(self._experiment_telemetry_attributes(experiment))
        except Exception:
            logger.exception("Error saving the experiment")
            raise
