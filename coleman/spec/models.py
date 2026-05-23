"""
Pydantic v2 models for the Coleman run specification.

Each model corresponds to a section in the configuration, providing
type-safe defaults and validation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExecutionSpec(BaseModel):
    """Execution-level settings.

    Parameters
    ----------
    parallel_pool_size : int
        Number of parallel worker processes.
    independent_executions : int
        How many independent repetitions to run.
    seed : int | None
        Random seed for this execution (``None`` = use default/global behavior).
    verbose : bool
        Enable verbose logging.
    force_sequential_under_scalene : bool
        When ``True`` and Scalene is active, force ``parallel_pool_size=1``.
        This improves profiling stability and avoids multiprocessing issues
        observed on Python 3.14.
    """

    model_config = ConfigDict(extra="forbid")

    parallel_pool_size: int = 10
    independent_executions: int = 10
    seed: int | None = None
    verbose: bool = False
    force_sequential_under_scalene: bool = True


class ExperimentSpec(BaseModel):
    """Experiment-level settings.

    Parameters
    ----------
    scheduled_time_ratio : list[float]
        Time-budget ratios to evaluate.
    datasets_dir : str
        Root directory containing dataset files.
    datasets : list[str]
        Dataset identifiers (``org@project`` format).
    experiment_dir : str
        Directory where experiment artefacts are stored.
    rewards : list[str]
        Reward function names.
    policies : list[str]
        Bandit policy names.
    """

    model_config = ConfigDict(extra="forbid")

    scheduled_time_ratio: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.8])
    datasets_dir: str = "examples"
    datasets: list[str] = Field(default_factory=lambda: ["alibaba@druid"])
    experiment_dir: str = "results/experiments/"
    rewards: list[str] = Field(default_factory=lambda: ["RNFail", "TimeRank"])
    policies: list[str] = Field(
        default_factory=lambda: ["Random"],
    )


class AlgorithmSpec(BaseModel, extra="allow"):
    """Per-algorithm hyper-parameter block.

    Accepts arbitrary nested keys so that any algorithm can store its
    own parameter map (e.g. ``frrmab.window_sizes``, ``ucb.timerank.c``).
    """


class HCSConfigurationSpec(BaseModel):
    """HCS-specific flags.

    Parameters
    ----------
    wts_strategy : bool
        Whether to use the WTS strategy.
    """

    model_config = ConfigDict(extra="forbid")

    wts_strategy: bool = False


class ContextualFeatureGroupSpec(BaseModel):
    """Feature group configuration.

    Parameters
    ----------
    feature_group_name : str
        Name of the feature group.
    feature_group_values : list[str]
        Column names in the feature group.
    """

    model_config = ConfigDict(extra="forbid")

    feature_group_name: str = "time_execution"
    feature_group_values: list[str] = Field(default_factory=lambda: ["Duration", "NumErrors"])


class ContextualConfigSpec(BaseModel):
    """Contextual information configuration.

    Parameters
    ----------
    previous_build : list[str]
        Column names from the previous build.
    """

    model_config = ConfigDict(extra="forbid")

    previous_build: list[str] = Field(default_factory=lambda: ["Duration", "NumRan", "NumErrors"])


class ContextualInformationSpec(BaseModel):
    """Contextual information settings.

    Parameters
    ----------
    config : ContextualConfigSpec
        Previous build column configuration.
    feature_group : ContextualFeatureGroupSpec
        Feature group configuration.
    """

    model_config = ConfigDict(extra="forbid")

    config: ContextualConfigSpec = Field(default_factory=ContextualConfigSpec)
    feature_group: ContextualFeatureGroupSpec = Field(default_factory=ContextualFeatureGroupSpec)


class ResultsSpec(BaseModel):
    """Results sink settings.

    Parameters
    ----------
    enabled : bool
        Whether result persistence is active.
    sink : str
        Sink backend (``"parquet"`` or ``"clickhouse"``).
    out_dir : str
        Output directory for result files.
    batch_size : int
        Rows per write batch.
    top_k_prioritization : int
        Top-k value for prioritisation metrics (0 = disabled).
    clickhouse : dict[str, Any]
        Optional ClickHouse sink kwargs (e.g., ``host``, ``port``, ``secure``).
    duckdb : dict[str, Any]
        Optional DuckDB sink kwargs (e.g., ``file_count``, ``base_name``, ``shard_key``).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    sink: str = "parquet"
    out_dir: str = "./runs"
    batch_size: int = 1000
    top_k_prioritization: int = 0
    clickhouse: dict[str, Any] = Field(default_factory=dict)
    duckdb: dict[str, Any] = Field(default_factory=dict)


class CheckpointSpec(BaseModel):
    """Checkpoint persistence settings.

    Parameters
    ----------
    enabled : bool
        Whether checkpointing is active.
    interval : int
        Steps between checkpoints.
    base_dir : str
        Directory for checkpoint files.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    interval: int = 50000
    base_dir: str = "checkpoints"


class TelemetrySpec(BaseModel):
    """OpenTelemetry export settings.

    Parameters
    ----------
    enabled : bool
        Whether telemetry export is active.
    otlp_endpoint : str
        OTLP collector endpoint.
    service_name : str
        Reported service name.
    export_interval_millis : int
        Export interval in milliseconds.
    resource_attributes : dict[str, str]
        Extra OTel resource attributes (e.g. ``execution_id``, ``run_id``).
        These are attached to the OTel Resource, not to individual metric
        labels, avoiding high-cardinality issues.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    otlp_endpoint: str = "http://localhost:4318"
    service_name: str = "coleman"
    export_interval_millis: int = 5000
    resource_attributes: dict[str, str] = Field(default_factory=dict)


class HooksSpec(BaseModel):
    """Runner lifecycle hook configuration.

    Parameters
    ----------
    fail_fast : bool
        If ``True``, stop run execution on hook failures.
        If ``False``, hook failures are logged and execution continues.
    plugins : list[str]
        Dotted paths to hook plugins (class or function).
    """

    model_config = ConfigDict(extra="forbid")

    fail_fast: bool = True
    plugins: list[str] = Field(default_factory=list)


class RunSpec(BaseModel):
    """Top-level experiment specification.

    Composes every sub-spec into a single validated object that fully
    describes one experiment run.  When serialised to canonical JSON the
    result is deterministic and can be hashed to produce a stable
    ``run_id``.

    Parameters
    ----------
    execution : ExecutionSpec
        Execution-level settings.
    experiment : ExperimentSpec
        Experiment-level settings.
    algorithm : AlgorithmSpec
        Per-algorithm hyper-parameters (free-form, extra keys allowed).
    hcs_configuration : HCSConfigurationSpec
        HCS-specific flags.
    contextual_information : ContextualInformationSpec
        Contextual feature configuration.
    results : ResultsSpec
        Results sink settings.
    checkpoint : CheckpointSpec
        Checkpoint persistence settings.
    telemetry : TelemetrySpec
        Telemetry export settings.
    hooks : HooksSpec
        Optional lifecycle hook plugin configuration.
    extensions : dict[str, Any]
        Namespaced custom config passthrough for domain workflows.
    """

    model_config = ConfigDict(extra="forbid")

    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    experiment: ExperimentSpec = Field(default_factory=ExperimentSpec)
    algorithm: AlgorithmSpec = Field(default_factory=AlgorithmSpec)
    hcs_configuration: HCSConfigurationSpec = Field(default_factory=HCSConfigurationSpec)
    contextual_information: ContextualInformationSpec = Field(default_factory=ContextualInformationSpec)
    results: ResultsSpec = Field(default_factory=ResultsSpec)
    checkpoint: CheckpointSpec = Field(default_factory=CheckpointSpec)
    telemetry: TelemetrySpec = Field(default_factory=TelemetrySpec)
    hooks: HooksSpec = Field(default_factory=HooksSpec)
    extensions: dict[str, Any] = Field(default_factory=dict)
