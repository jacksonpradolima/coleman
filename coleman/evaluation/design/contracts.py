"""Immutable experiment designs and inference requests, independently of I/O."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from coleman.core.base import CoreImmutable
from coleman.core.identity import Reference, ordered_references
from coleman.core.numeric import Binary64, Scalar
from coleman.core.primitives import Fingerprint, Identifier
from coleman.core.reading import bounded_json_bytes
from coleman.core.splits import TemporalPlan
from coleman.core.units import UnitSemantics
from coleman.evaluation.design._wire import DesignContract, guard_graph, model_storage, wire_size
from coleman.evaluation.design.limits import DesignCounts, ProtocolLimits
from coleman.evaluation.design.source_contracts import (
    SourceMember,
    SourcePartitionManifest,
    SourceScope,
    SourceSnapshot,
)
from coleman.evaluation.protected.bounded_inference import BoundedMeanProtocol
from coleman.evaluation.protected.contracts import ComparisonHypothesis, EvaluationProtocol, FrozenCandidate
from coleman.objectives.bindings import DefinitionBinding, QuantityMeaning
from coleman.results.domain import RESERVED_METRICS

Seed = Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]
InferenceMethod = Literal["paired_bounded_mean_hoeffding_holm_v1", "paired_t_holm_bonferroni_ci_v1"]


class StrategyBindings(CoreImmutable):
    """Freeze the treatment law while allowing its declared causal state updates.

    Attributes
    ----------
    implementation, initialization, update_law, refit_law : DefinitionBinding
        Complete method/version/initial-prior and update/refit definitions.
    transform_law, eligibility, accounting, stopping : DefinitionBinding
        Frozen transformations/state updates, past-information eligibility,
        executed accounting and stopping rules. Bind explicit no-op laws when
        appropriate rather than inventing an operation from a method name.
    initial_state_fingerprint : str or None
        Actual selected initial state identity, or explicit unknown/absent.
    update_scope : {'none', 'within_unit', 'across_units'}
        Scope of the fixed law. Across-unit adaptation is representable but
        unsupported by the first fixed-independent-unit inference profile.
    semantic_context : DefinitionBinding or None
        Supplied complete semantic-context identity, independently absent when
        a legacy producer has no such envelope.

    Notes
    -----
    Definitions identify content, not authentication or scientific efficacy.
    No learning, lookup, state capture, refit, transform or I/O is performed.
    """

    implementation: DefinitionBinding
    initialization: DefinitionBinding
    update_law: DefinitionBinding
    refit_law: DefinitionBinding
    transform_law: DefinitionBinding
    eligibility: DefinitionBinding
    accounting: DefinitionBinding
    stopping: DefinitionBinding
    initial_state_fingerprint: Fingerprint | None
    update_scope: Literal["none", "within_unit", "across_units"]
    semantic_context: DefinitionBinding | None = None


class CandidatePlan(CoreImmutable):
    """Hold a base resolved configuration, separate from its later projection.

    Attributes
    ----------
    candidate : FrozenCandidate
        BASE configuration only. Projected protected configuration later binds
        this base plus design/selection fingerprints, avoiding a content cycle.
    role : str
        Candidate, deployable fixed/equal-weight/periodically refit baseline or
        retrospective oracle. An oracle role never implies deployability.
    strategy : StrategyBindings
        Exact frozen law and initialization rather than future realized weights.

    Notes
    -----
    JSON is retained only at the existing opaque FrozenCandidate configuration
    boundary. The new envelope bounds its complete UTF-8 bytes and depth; the
    legacy reader/defaults remain unchanged. No candidate is executed here.
    """

    candidate: FrozenCandidate
    role: Literal[
        "candidate", "fixed_baseline", "equal_weight_baseline", "periodic_refit_baseline", "retrospective_oracle"
    ]
    strategy: StrategyBindings

    @model_validator(mode="before")
    @classmethod
    def _configuration_ingress(cls, value: Any) -> Any:
        """Bound nested legacy candidate JSON before its canonical parser runs."""
        fields = model_storage(value) if isinstance(value, BaseModel) else value
        if isinstance(fields, dict):
            candidate = fields.get("candidate")
            if isinstance(candidate, BaseModel):
                candidate = model_storage(candidate)
            if isinstance(candidate, dict) and isinstance(candidate.get("configuration_json"), (str, bytes)):
                bounded_json_bytes(candidate["configuration_json"], 16 * 1024)
        return value

    @model_validator(mode="after")
    def _configuration(self) -> Self:
        """Require a finite bounded base object before protected projection."""
        bounded_json_bytes(self.candidate.configuration_json, 16 * 1024)
        return self


class MetricPlan(CoreImmutable):
    """Freeze one measured quantity, its scientific estimand and distinct role.

    Attributes
    ----------
    name : str
        Explicit adapter measurement name. Reserved native output names reject;
        native fitness/cost columns require their own documented projection.
    quantity : QuantityMeaning
        Full dimension, physical units, meaning, inclusive target bounds and
        horizon. Bounded learner loss cannot supply bounds for a currency target.
    direction : {'minimize', 'maximize', 'none'}
        Raw measured objective direction, without inferring probabilities.
    role : str
        Raw outcome, learner surrogate, executed objective, feasibility, physical
        cost, uncertainty or diagnostic. Equal units do not merge these roles.
    estimand : DefinitionBinding
        Complete cumulative/terminal/prediction/precision target definition.

    Notes
    -----
    The first panel profile aggregates its primary metric only; physical group
    costs are separate ledger evidence, not silently averaged over result cells.
    """

    name: Identifier
    quantity: QuantityMeaning
    direction: Literal["minimize", "maximize", "none"]
    role: Literal[
        "raw_outcome", "surrogate", "executed_objective", "feasibility", "physical_cost", "uncertainty", "diagnostic"
    ]
    estimand: DefinitionBinding

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        """Prevent a new generic measurement from impersonating native columns."""
        if value in RESERVED_METRICS:
            raise ValueError("Generic metric names cannot impersonate reserved native outputs")
        return value


class Threshold(CoreImmutable):
    """Declare a practical improvement in the primary raw metric's units.

    Attributes
    ----------
    value : float
        Finite nonnegative binary64 difference. Observed zero is meaningful.
    semantics : UnitSemantics
        Exact raw primary units/meaning. Raw target bounds do not constrain a
        difference threshold; no currency or unit conversion is performed.
    """

    value: Annotated[Binary64, Field(ge=0)]
    semantics: UnitSemantics


class InferenceRequest(CoreImmutable):
    """Represent supported and unsupported requests without inventing inference.

    Attributes
    ----------
    method : str
        Existing versioned bounded-paired or paired-t route, never a new estimator.
    dependence : str
        Declared relationship between final units, not time steps within a unit.
    assignment, analysis : str
        Complete paired target versus adaptive partial assignment, and fixed
        final versus optional-stopping analysis. Unsupported states are retained.
    normal_differences_assumed : bool
        Explicit legacy t assumption, not inferred by testing normality or seeds.
    practical_effect : Threshold
        Frozen improvement threshold with exact primary units/meaning.
    alpha, minimum_units : float, int
        Full-family error level and independent-unit requirement.
    independent_unit : str
        Declared scientific meaning of one unit; no automatic independence claim.
    independence_basis, bounds_basis : DefinitionBinding, DefinitionBinding or None
        Supplied evidence/assumption definitions; null bounds remain unknown.

    Notes
    -----
    Unknown/unbounded target ranges preclude the bounded route. The existing t
    route separately accepts finite observations under independent normal
    differences and its optional analytics capability. No capability is loaded
    or checked by this pure request model.
    """

    method: InferenceMethod
    dependence: Literal["independent", "serial", "clustered", "unknown"]
    assignment: Literal["complete_paired_units", "adaptive_partial"]
    analysis: Literal["fixed_final", "optional_stopping"]
    normal_differences_assumed: Annotated[bool, Field(strict=True)] = False
    practical_effect: Threshold
    alpha: Annotated[Binary64, Field(gt=0, lt=1)] = 0.05
    minimum_units: Annotated[int, Field(strict=True, ge=2)] = 20
    independent_unit: Identifier
    independence_basis: DefinitionBinding
    bounds_basis: DefinitionBinding | None


class InferenceQualification(DesignContract):
    """Return typed method support without data loading or efficacy conclusions.

    Attributes
    ----------
    status : {'supported', 'unsupported'}
        Compatibility of requested assumptions with the existing inference route.
    protocol : BoundedMeanProtocol or EvaluationProtocol or None
        Actual supported legacy protocol, or explicit null when unsupported.
    reasons : tuple of str
        Required distinct unsupported reasons; empty for supported requests.

    Notes
    -----
    Supported is not sufficient evidence, source admission or confirmatory
    eligibility. It does not capture runtime capabilities or activate a policy.
    """

    status: Literal["supported", "unsupported"]
    protocol: Annotated[BoundedMeanProtocol | EvaluationProtocol, Field(discriminator="method")] | None
    reasons: Annotated[tuple[Identifier, ...], Field(max_length=32)]

    @model_validator(mode="after")
    def _support(self) -> Self:
        """Keep support status, actual protocol and reasons mutually consistent."""
        supported = self.status == "supported"
        if supported != (self.protocol is not None) or supported != (not self.reasons):
            raise ValueError("Inference support requires a protocol without reasons; unsupported requires reasons only")
        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("Inference reasons must be distinct")
        return self


class BudgetPlan(CoreImmutable):
    """Declare finite evidence/attempt budgets without assuming a job per cell.

    Attributes
    ----------
    maximum_trials, maximum_physical_attempts, maximum_output_references : int
        Global complete-experiment capacities for all attempted configurations,
        real grouped executions/retries and output references across development
        and protected stages combined. One process may emit many cells; imports
        invent none. Separate ledgers never duplicate the available budget.
    allowances : tuple of Scalar
        Typed planned compute/resource/cost limits, distinct from actual measured
        cost records. Unknown allowances retain null/status, never fabricated zero.
    stopping : DefinitionBinding
        Frozen overall stopping/budget interpretation before protected access.
    """

    maximum_trials: Annotated[int, Field(strict=True, ge=0, le=256)] = 32
    maximum_physical_attempts: Annotated[int, Field(strict=True, ge=0, le=4096)] = 32
    maximum_output_references: Annotated[int, Field(strict=True, ge=0, le=16384)] = 512
    allowances: Annotated[tuple[Scalar, ...], Field(max_length=32)] = ()
    stopping: DefinitionBinding

    @field_validator("allowances")
    @classmethod
    def _allowances(cls, values: tuple[Scalar, ...]) -> tuple[Scalar, ...]:
        """Keep one explicitly typed planned resource per declared dimension."""
        ordered_references(tuple(item.spec.dimension for item in values))
        if any(item.spec.role != "resource" for item in values):
            raise ValueError("Planned allowances require resource-role quantities")
        return tuple(sorted(values, key=lambda item: item.spec.dimension.id))


class ExperimentDesign(DesignContract):
    """Freeze a complete generic experiment before protected outcome access.

    Attributes
    ----------
    reference, scope, snapshot : Reference, SourceScope, SourceSnapshot
        Experiment identity, stable upstream and exact selected source input.
    source_plan, comparison_plan : TemporalPlan
        Historical scientific membership and actual derived-result availability.
        Only comparison.test_end may extend; all other fields remain equal.
    members, candidates, seeds, metrics, hypotheses : tuple
        Complete bounded source catalogue, BASE candidates, repetition coordinates,
        quantity/estimand declarations and full comparison family.
    primary_metric : str
        Exactly one declared metric for the first complete-panel inference route.
    inference, budget, limits : InferenceRequest, BudgetPlan, ProtocolLimits
        Frozen assumptions, physical/evidence limits and coupled wire capacities.
    evidence_basis : {'constructed_mechanism', 'empirical_protected', 'unqualified'}
        Explicit caller-supplied scientific scope. Constructed mechanics are not
        automatically empirical evidence or an unseen source.
    source_admission_fingerprint : str or None
        Supplied independent assessment reference, required for an empirical
        declaration. A reference alone authenticates or admits nothing.
    aggregation : str
        Named equal-seed then equal-member mean, never pooled raw-cell inference.

    Notes
    -----
    Pure design/identity only: no source access, fitting, runner, clock capture,
    artifact store, transport or scientific admission. Selection separately binds
    this design; protected projection then binds both without a digest cycle.
    """

    reference: Reference
    scope: SourceScope
    snapshot: SourceSnapshot
    source_plan: TemporalPlan
    comparison_plan: TemporalPlan
    members: Annotated[tuple[SourceMember, ...], Field(min_length=1, max_length=4096)]
    candidates: Annotated[tuple[CandidatePlan, ...], Field(min_length=2, max_length=64)]
    seeds: Annotated[tuple[Seed, ...], Field(min_length=1, max_length=64)]
    metrics: Annotated[tuple[MetricPlan, ...], Field(min_length=1, max_length=32)]
    primary_metric: Identifier
    hypotheses: Annotated[tuple[ComparisonHypothesis, ...], Field(min_length=1, max_length=64)]
    inference: InferenceRequest
    budget: BudgetPlan
    limits: ProtocolLimits
    evidence_basis: Literal["constructed_mechanism", "empirical_protected", "unqualified"] = "unqualified"
    source_admission_fingerprint: Fingerprint | None = None
    aggregation: Literal["equal-seed-member-unit-mean-binary64-v1"] = "equal-seed-member-unit-mean-binary64-v1"

    @field_validator("seeds")
    @classmethod
    def _seeds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        """Reject duplicate repetition coordinates before sorting their set."""
        if len(values) != len(set(values)):
            raise ValueError("Repetition seed coordinates must be distinct")
        return tuple(sorted(values))

    @model_validator(mode="after")
    def _design(self) -> Self:
        """Validate full identities, chronology and products before enumeration."""
        guard_graph(self, maximum_bytes=self.limits.max_envelope_bytes, maximum_items=self.limits.max_graph_items)
        if self.evidence_basis == "empirical_protected" and self.source_admission_fingerprint is None:
            raise ValueError("Empirical declaration requires its supplied independent source assessment reference")
        self._chronology()
        manifest = self.partition_manifest
        object.__setattr__(self, "members", manifest.members)
        self._catalogues()
        self._family()
        self.limits.require_counts(self.counts)
        self.limits.reserve(self.raw_cells, self.aggregate_inputs)
        if wire_size(self) + 1 > min(self.limits.max_design_bytes, self.limits.max_metadata_bytes):
            raise ValueError("Actual encoded design exceeds its declared metadata/root capacity")
        return self

    def _chronology(self) -> None:
        """Keep complete source interpretation fixed when analysis time extends."""
        if (
            self.source_plan.model_dump(mode="json", exclude={"test_end"})
            != self.comparison_plan.model_dump(mode="json", exclude={"test_end"})
            or self.comparison_plan.test_end < self.source_plan.test_end
        ):
            raise ValueError("Only comparison test_end may extend the complete original source plan")

    def _catalogues(self) -> None:
        """Require unique names and exact primary raw threshold semantics."""
        candidates = [item.candidate.candidate_id for item in self.candidates]
        metrics = [item.name for item in self.metrics]
        if len(candidates) != len(set(candidates)) or len(metrics) != len(set(metrics)):
            raise ValueError("Candidate and metric declarations require distinct identities")
        if self.primary_metric not in metrics:
            raise ValueError("Primary metric is not in the complete metric catalogue")
        if self.inference.practical_effect.semantics != self.metric(self.primary_metric).quantity.spec.semantics:
            raise ValueError("Practical threshold units/meaning differ from the primary raw quantity")
        if any(
            len(item.candidate.configuration_json.encode("utf-8")) > self.limits.max_configuration_bytes
            for item in self.candidates
        ):
            raise ValueError("Actual encoded base configuration exceeds its declared capacity")
        object.__setattr__(
            self, "candidates", tuple(sorted(self.candidates, key=lambda item: item.candidate.candidate_id))
        )
        object.__setattr__(self, "metrics", tuple(sorted(self.metrics, key=lambda item: item.name)))

    def _family(self) -> None:
        """Freeze every unique hypothesis/pair without dropping unfavorable tasks."""
        identities = [item.hypothesis_id for item in self.hypotheses]
        pairs = [frozenset((item.candidate_id, item.baseline_id)) for item in self.hypotheses]
        known = {item.candidate.candidate_id for item in self.candidates}
        if len(identities) != len(set(identities)) or len(pairs) != len(set(pairs)):
            raise ValueError("Comparison family requires distinct hypotheses and unordered pairs")
        if any(item.candidate_id not in known or item.baseline_id not in known for item in self.hypotheses):
            raise ValueError("Comparison family references an undeclared final candidate")
        object.__setattr__(self, "hypotheses", tuple(sorted(self.hypotheses, key=lambda item: item.hypothesis_id)))

    @property
    def partition_manifest(self) -> SourcePartitionManifest:
        """Return the complete pure source binding for persisted narrow claims."""
        return SourcePartitionManifest(
            schema_version=1,
            scope=self.scope,
            snapshot=self.snapshot,
            source_plan=self.source_plan,
            members=self.members,
        )

    @property
    def protected_members(self) -> tuple[SourceMember, ...]:
        """Return eligible fixed source-test members, without widening the source."""
        return self.partition_manifest.protected_members

    @property
    def unit_ids(self) -> tuple[str, ...]:
        """Return declared protected independent-unit IDs, never seed IDs."""
        return tuple(sorted({item.unit_id for item in self.protected_members if item.unit_id is not None}))

    @property
    def counts(self) -> DesignCounts:
        """Return bounded actual catalogue products and declared ledger capacities."""
        return DesignCounts(
            candidates=len(self.candidates),
            source_members=len(self.members),
            protected_members=len(self.protected_members),
            units=len(self.unit_ids),
            seeds=len(self.seeds),
            hypotheses=len(self.hypotheses),
            metrics=len(self.metrics),
            trials=self.budget.maximum_trials,
            physical_attempts=self.budget.maximum_physical_attempts,
            output_references=self.budget.maximum_output_references,
        )

    @property
    def raw_cells(self) -> int:
        """Return the actual complete candidate/member/seed product before enumeration."""
        return len(self.candidates) * len(self.protected_members) * len(self.seeds)

    @property
    def aggregate_inputs(self) -> int:
        """Return one derived input per candidate and declared protected unit."""
        return len(self.candidates) * len(self.unit_ids)

    def candidate(self, candidate_id: str) -> CandidatePlan:
        """Query a frozen base candidate without changing its configuration.

        Parameters
        ----------
        candidate_id : str
            Exact named candidate, independent of declaration order.

        Returns
        -------
        CandidatePlan
            Original declared base configuration and law.

        Raises
        ------
        KeyError
            Identity was not declared. No lookup or I/O occurs.
        """
        for item in self.candidates:
            if item.candidate.candidate_id == candidate_id:
                return item
        raise KeyError(candidate_id)

    def metric(self, name: str) -> MetricPlan:
        """Query an exact declared metric without unit conversion.

        Parameters
        ----------
        name : str
            Exact measurement name.

        Returns
        -------
        MetricPlan
            Complete quantity, role and estimand.

        Raises
        ------
        KeyError
            Name was not declared. No source or capability is loaded.
        """
        for item in self.metrics:
            if item.name == name:
                return item
        raise KeyError(name)
