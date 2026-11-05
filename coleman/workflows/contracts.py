"""Versioned, transport-independent declarations for bounded native CI workflows."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.budget import BudgetMode
from coleman.evaluation.protected import SplitDecision, TemporalObservation, TemporalPlan
from coleman.results.domain import Fingerprint, Finite
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable
from coleman.spec.models import BudgetSpec
from coleman.spec.runtime_identity import InstalledRuntime

WorkflowId = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]
Policy = Literal["Random", "UCB"]
Objective = Literal["napfd", "apfdc"]
MAX_WORKFLOW_ARTIFACT_ENTRIES = 32768


class WorkflowError(ValueError):
    """Safe host-independent workflow explanation; no source path or worker traceback.

    Raised only with package-defined bounded messages at planning/application
    boundaries. Transports may expose this explanation alongside a stable error
    classification. Arbitrary storage/native exceptions must not use this class.
    """


class WorkflowContract(VersionedContract):
    """Strict v1 workflow namespace; application-v1 schemas/lifetimes stay unchanged.

    Round trips reject duplicate/unknown keys, missing/future versions and
    non-finite quantities. Canonical serialization and fingerprints are inherited
    from the existing domain contract; these objects perform no I/O.
    """

    contract: Literal["coleman-workflows-v1"] = "coleman-workflows-v1"


class ScheduleBudget(Immutable):
    """Native scheduling budgets, distinct from tuning work or observed runtime.

    Parameters
    ----------
    mode : BudgetMode
        Ratio, fixed seconds or a maximum selected-test count.
    unit : {'1', 's', 'count'}
        Must match the mode; units are never inferred from a numeric value.
    values : tuple[float, ...]
        Unique finite native budgets, validated with the existing BudgetSpec.
        Zero is rejected by that native contract rather than silently replaced.
    """

    mode: BudgetMode = BudgetMode.RATIO
    unit: Literal["1", "s", "count"] = "1"
    values: Annotated[tuple[Finite, ...], Field(min_length=1, max_length=16)] = (0.5,)

    @model_validator(mode="after")
    def _native(self) -> Self:
        """Reject incompatible units/duplicates and canonicalize budget declaration order."""
        units = {BudgetMode.RATIO: "1", BudgetMode.FIXED_TIME: "s", BudgetMode.SUBSET_SIZE: "count"}
        if self.unit != units[self.mode] or len(set(self.values)) != len(self.values):
            raise ValueError("Scheduling budget requires matching units and unique values")
        BudgetSpec(mode=self.mode, values=list(self.values))
        return self


class WorkflowLimits(Immutable):
    """Frozen job caps; configurations, trials and physical attempts remain distinct.

    Parameters
    ----------
    max_configurations : int
        Maximum resolved policy/hyperparameter/budget combinations before seeds.
    max_trials : int
        Maximum logical configuration/seed pairs.
    max_build_work : int
        Worst-case native build work including the declared maximum attempts.
    max_attempts : int
        Maximum physical invocations per logical trial, including resume.
        Also bounds job-worker starts; protected tuning/evaluation requires at
        least two starts. A rejected restoration retains its acknowledged frontier.
    attempt_seconds : float
        Positive observed-wall-time allowance per worker invocation, in seconds.
    max_memory_bytes : int
        Sampled worker RSS ceiling in bytes; not a hostile-code sandbox.
    max_output_bytes : int
        Total reserved job bytes, including its input copy, native checkpoints
        and results. Input alone must leave space for produced artifacts.
    """

    max_configurations: Annotated[int, Field(strict=True, ge=1, le=64)] = 12
    max_trials: Annotated[int, Field(strict=True, ge=1, le=128)] = 32
    max_build_work: Annotated[int, Field(strict=True, ge=1, le=100000)] = 100000
    max_attempts: Annotated[int, Field(strict=True, ge=1, le=8)] = 4
    attempt_seconds: Annotated[Finite, Field(gt=0, le=3600)] = 300.0
    max_memory_bytes: Annotated[int, Field(strict=True, ge=67108864, le=2147483648)] = 1073741824
    max_output_bytes: Annotated[int, Field(strict=True, ge=65536, le=67108864)] = 16777216


class ProtectedSettings(Immutable):
    """Predeclared bounded-mean family protocol, independent of observed outcomes.

    Parameters
    ----------
    minimum_units : int
        Required independent repository families; seeds never increase this count.
    practical_effect : float
        Nonnegative dimensionless improvement threshold for NAPFD/APFDc.
    alpha : float
        Family-wise error allowance, strictly between zero and one.

    Notes
    -----
    Known metric bounds are [0, 1]; normality is not assumed. This integration
    initially exercises constructed mechanisms, never grants empirical admission.
    """

    minimum_units: Annotated[int, Field(strict=True, ge=2, le=4096)] = 20
    practical_effect: Annotated[Finite, Field(ge=0, le=1)] = 0.01
    alpha: Annotated[Finite, Field(gt=0, lt=1)] = 0.05


class PlanningRequest(WorkflowContract):
    """Typed, bounded enumeration request; no paths, plugins or arbitrary RunSpec.

    Parameters
    ----------
    dataset_id : str
        Stable ID discovered from trusted host registration.
    policies : tuple[{'Random', 'UCB'}, ...]
        Random is the required baseline. Reordering does not change a grid plan.
    ucb_exploration : tuple[float, ...]
        Unique positive finite RNFail exploration coefficients, UCB only.
    budget : ScheduleBudget
        Native scheduling modes/units; independent of the job resource caps.
    seeds : tuple[int, ...]
        Explicit RNG repetitions; never independent scientific observations.
    mode : {'grid', 'zip'}
        Existing sweep semantics for variable UCB/budget dimensions. Zip pairs
        retain their declared correspondence; constants are held fixed.
    objective : {'napfd', 'apfdc'}
        Legacy native quality column to maximize, both dimensionless; APFDc is
        not seconds, runtime or money. Ties prefer Random, then smaller UCB c,
        then canonical configuration identity.
    evaluation : {'descriptive', 'protected'}
        Full-history descriptive replay or a separate eligible temporal workflow.
    limits : WorkflowLimits
        Explicit finite caps, checked before materializing sweep coordinates.
    protected : ProtectedSettings
        Frozen uncertainty, practical threshold and independent-unit requirements.
    """

    dataset_id: WorkflowId
    policies: Annotated[tuple[Policy, ...], Field(min_length=1, max_length=2)] = ("Random", "UCB")
    ucb_exploration: Annotated[
        tuple[Annotated[Finite, Field(gt=0, le=1000)], ...], Field(min_length=1, max_length=32)
    ] = (0.5, 1.0, 2.0)
    budget: ScheduleBudget = ScheduleBudget()
    seeds: Annotated[tuple[Seed, ...], Field(min_length=1, max_length=64)] = (42, 43)
    mode: Literal["grid", "zip"] = "grid"
    objective: Objective = "napfd"
    evaluation: Literal["descriptive", "protected"] = "descriptive"
    limits: WorkflowLimits = WorkflowLimits()
    protected: ProtectedSettings = ProtectedSettings()

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        """Reject duplicate or irrelevant coordinates rather than dropping declarations."""
        for values in (self.policies, self.ucb_exploration, self.seeds, self.budget.values):
            if len(values) != len(set(values)):
                raise ValueError("Workflow dimensions must contain unique coordinates")
        if "Random" not in self.policies:
            raise ValueError("A workflow requires its Random baseline")
        if "UCB" not in self.policies and self.ucb_exploration != (0.5, 1.0, 2.0):
            raise ValueError("UCB axes are irrelevant without UCB")
        return self


class BuildMetadata(Immutable):
    """Host-declared source clocks/groups; absent clocks never become invented dates.

    Parameters
    ----------
    build_id : int
        Exact native BuildId associated with this declaration.
    observation : TemporalObservation
        Original or explicitly authored event, availability and leakage-group IDs.
    unit_id : str
        Reviewed repository/family inference unit, not a seed or worker ID.
    costs_known_at_decision : bool
        Host declaration that scheduling costs were available before ranking.
    """

    build_id: Annotated[int, Field(strict=True)]
    observation: TemporalObservation
    unit_id: WorkflowId
    costs_known_at_decision: Annotated[bool, Field(strict=True)] = False


class DatasetManifest(WorkflowContract):
    """Exact local native input binding and explicit evidence scope, without paths.

    Parameters
    ----------
    dataset_id : str
        Public registry ID; exact bytes and metadata, not this label, bind identity.
    sha256 : str
        Identified SHA-256 of the original native Parquet bytes.
    size_bytes : int
        Encoded source size, bounded before decoding.
    build_ids : tuple[int, ...]
        Complete sorted unique native builds, retaining original IDs.
    rows : int
        Total native test observations, not independent evidence units.
    maximum_tests : int
        Largest per-build test count, distinct from tuning-trial caps.
    source_fingerprint, projection_fingerprint : str
        Host-verified source/projection identities retained for audit.
    evidence : {'constructed_mechanism', 'descriptive_source'}
        Qualification scope. Registration never grants scientific admission.
    assumptions : tuple[str, ...]
        Explicit projection/chronology/feedback declarations.
    metadata : tuple[BuildMetadata, ...]
        Optional complete temporal/group declarations; missing stays missing.
    temporal_plan : TemporalPlan or None
        Host-approved split, required for a protected planning request.
    """

    dataset_id: WorkflowId
    sha256: Fingerprint
    size_bytes: Annotated[int, Field(strict=True, ge=1, le=67108864)]
    build_ids: Annotated[tuple[Annotated[int, Field(strict=True)], ...], Field(min_length=1, max_length=512)]
    rows: Annotated[int, Field(strict=True, ge=1, le=1000000)]
    maximum_tests: Annotated[int, Field(strict=True, ge=1, le=10000)]
    source_fingerprint: Fingerprint
    projection_fingerprint: Fingerprint
    duration_unit: Literal["s"] = "s"
    verdict_semantics: Literal["binary-failing-test-v1"] = "binary-failing-test-v1"
    evidence: Literal["constructed_mechanism", "descriptive_source"]
    assumptions: Annotated[
        tuple[Annotated[str, Field(strict=True, min_length=1, max_length=512)], ...], Field(max_length=16)
    ]
    metadata: Annotated[tuple[BuildMetadata, ...], Field(max_length=512)] = ()
    temporal_plan: TemporalPlan | None = None

    def split_by_build(self) -> tuple[SplitDecision, ...]:
        """Return source temporal decisions joined to the immutable native build sequence.

        Returns
        -------
        tuple[SplitDecision, ...]
            One decision per metadata/build ID, in exactly build_ids order. The
            existing splitter sorts by observation/revision identity; that ordering
            never determines native learning ordinals or protected membership.

        Raises
        ------
        ValueError
            Missing temporal declarations or invalid observation/group identity.

        Notes
        -----
        Joins by both observation_id and revision_id, retaining distinct revisions.
        Reads only declared metadata; no outcomes, files, runtime state or source
        data are loaded or changed. Serialization and native build IDs are unchanged.
        """
        if self.temporal_plan is None or not self.metadata:
            raise WorkflowError("Temporal build decisions require complete registered metadata")
        decisions = self.temporal_plan.split(tuple(item.observation for item in self.metadata))
        by_revision = {(item.observation.observation_id, item.observation.revision_id): item for item in decisions}
        return tuple(
            by_revision[item.observation.observation_id, item.observation.revision_id] for item in self.metadata
        )

    @model_validator(mode="after")
    def _membership(self) -> Self:
        """Require complete, unambiguous membership when temporal declarations are supplied."""
        if self.build_ids != tuple(sorted(set(self.build_ids))):
            raise ValueError("Dataset build IDs must be sorted and unique")
        if self.metadata and tuple(item.build_id for item in self.metadata) != self.build_ids:
            raise ValueError("Temporal metadata must bind every native build exactly once")
        if self.temporal_plan is not None:
            if not self.metadata:
                raise ValueError("A temporal plan requires complete build metadata")
            self.temporal_plan.split(tuple(item.observation for item in self.metadata))
        return self


class Trial(Immutable):
    """One canonical configuration/seed coordinate; no host locations or outcomes.

    Parameters
    ----------
    trial_id, configuration_id : str
        Content-derived workflow IDs; distinct from native configuration run_id.
    policy : {'Random', 'UCB'}
        Actual native policy name.
    exploration : float or None
        Positive UCB coefficient; None is not applicable to Random.
    budget_value : float
        Scheduling quantity in the plan's explicit budget mode/unit.
    seed : int
        One native independent execution's RNG seed, not an inference unit.
    resolved_spec_json : str
        Complete canonical RunSpec using normalized owned-path placeholders.
        Execution only substitutes host-owned paths; no policy is reconstructed.
    """

    trial_id: Fingerprint
    configuration_id: Fingerprint
    policy: Policy
    exploration: Annotated[Finite, Field(gt=0, le=1000)] | None
    budget_value: Finite
    seed: Seed
    resolved_spec_json: Annotated[str, Field(strict=True, max_length=16384)]

    @model_validator(mode="after")
    def _parameter(self) -> Self:
        """Separate Random's inapplicable parameter from an observed zero coefficient."""
        if (self.policy == "UCB") != (self.exploration is not None):
            raise ValueError("Only UCB has an exploration coefficient")
        return self


class ExperimentPlan(WorkflowContract):
    """Immutable executable resolved plan, independent of CLI/MCP/conversation.

    Parameters
    ----------
    request : PlanningRequest
        Canonical enumeration/defaults and explicit resource caps.
    dataset : DatasetManifest
        Complete exact-byte and evidence binding.
    runtime : InstalledRuntime
        Existing conservative SDK/base-runtime checkpoint compatibility identity.
    trials : tuple[Trial, ...]
        Exact expected coordinates; validated against existing sweep expansion.
    defaults_origin : str
        Versioned source of documented discovered defaults, not guessed values.
    native_threads : int
        Actual host Polars pool size; resumption rejects a different thread profile.
        The pure planner defaults to an explicit one-thread hypothetical profile;
        the production application always captures its actual installed pool size.
    """

    request: PlanningRequest
    dataset: DatasetManifest
    runtime: InstalledRuntime
    trials: Annotated[tuple[Trial, ...], Field(min_length=1, max_length=128)]
    defaults_origin: Literal["coleman-workflow-defaults-v1"] = "coleman-workflow-defaults-v1"
    native_threads: Annotated[int, Field(strict=True, ge=1, le=1024)] = 1

    @model_validator(mode="after")
    def _resolved(self) -> Self:
        """Recompute the bounded coordinate contract before accepting a saved plan."""
        from coleman.workflows.planning import resolve_trials

        if self.request.dataset_id != self.dataset.dataset_id or self.trials != resolve_trials(
            self.request, self.dataset
        ):
            raise ValueError("Saved plan differs from its declared dataset/search space")
        return self
