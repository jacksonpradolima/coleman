"""Pure structural negotiation for typed decisions and feedback, without execution."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, TypeAdapter, field_validator, model_validator

from coleman.core.actions import ActionKind, ActionSpace, ChoiceAction, selected_members, space_members
from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.decisions import Decision, ExecutedDecision, StateOwner
from coleman.core.feedback import (
    AbsentFeedback,
    AggregateFeedback,
    ChosenFeedback,
    ComponentFeedback,
    Feedback,
    FeedbackBatch,
    FeedbackKind,
    FeedbackRow,
    FullPanel,
    Origin,
    OutcomeValue,
)
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.primitives import MAX_DIMENSIONS, Identifier
from coleman.core.sampling import SamplingRecord
from coleman.core.temporal import Timestamp

TimingMode = Literal["untimed", "future", "ongoing"]
SamplingRequirement = Literal["not_required", "exact", "qualified_approximation"]
SamplingStatus = Literal["exact", "qualified_approximation", "unavailable", "not_required"]
_ALL_SAMPLING_STATUSES: frozenset[SamplingStatus] = frozenset(
    {"exact", "qualified_approximation", "unavailable", "not_required"}
)


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    """Canonicalize declared support without silently dropping duplicates."""
    if len(values) != len(set(values)):
        raise ValueError("Capability declarations must not repeat supported values")
    return tuple(sorted(values))


def _schemas(values: tuple[ScalarSpec, ...]) -> tuple[ScalarSpec, ...]:
    """Retain one schema version per dimension ID in canonical order."""
    names = [value.dimension.id for value in values]
    if len(names) != len(set(names)):
        raise ValueError("Capability dimension IDs and versions must be unique")
    return tuple(sorted(values, key=lambda value: value.dimension.id))


def _input_schemas(values: tuple[ScalarSpec, ...]) -> tuple[ScalarSpec, ...]:
    """Limit direct input requirements to supplied features and predictions."""
    if any(value.role not in {"feature", "prediction"} for value in values):
        raise ValueError("Decision input schemas require feature or prediction roles")
    return _schemas(values)


class ProblemCapabilities(CoreContract):
    """Declare a bounded problem's actual action and feedback support.

    Attributes
    ----------
    space : ActionSpace
        Typed feasible-action declaration; an arbitrary feasible-set reference
        does not prove a solver's mathematics.
    feedback_kind : str
        Full panel, chosen, component, aggregate or absent coverage.
    values : tuple of ScalarSpec
        At most 64 advertised outcome schemas, including exact units and bounds.
    inputs : tuple of ScalarSpec, optional
        Global feature/prediction input schemas. Member context never implicitly
        satisfies these global requirements.
    origins : tuple of str
        Nonempty distinct permitted origins, not a source-authentication claim.
    timing_mode : {'untimed', 'future', 'ongoing'}
        Actual temporal contract, independent of feedback coverage.
    delayed_feedback : bool
        Whether feedback may span additional logical decision rounds.
    sampling_statuses : tuple of str
        Nonempty canonical set of all actual selection-evidence statuses this
        problem may supply. One possible exact status is insufficient when
        unavailable or approximate evidence can also occur.
    receipt_status : {'known', 'unknown'}
        Whether a known actual receipt is guaranteed to be supplied. Unknown
        confers no receipt guarantee and never means immediate delivery.
    state_owners : tuple of {'host', 'learner', 'stateless'}
        Nonempty supported algorithm-state ownership modes.

    Raises
    ------
    ValueError
        Invalid version, empty/duplicate support, schemas or coerced booleans.

    Notes
    -----
    Pure declarations require no registration, dataset, service or storage.
    They do not qualify scientific efficacy, arbitrary feasibility or a theorem.
    Sampling and receipt guarantees are caller attestations, not authentication;
    concrete decisions and feedback must still be checked before use.
    """

    space: ActionSpace
    feedback_kind: FeedbackKind
    values: Annotated[tuple[ScalarSpec, ...], Field(max_length=MAX_DIMENSIONS)]
    inputs: Annotated[tuple[ScalarSpec, ...], Field(max_length=MAX_DIMENSIONS)] = ()
    origins: Annotated[tuple[Origin, ...], Field(min_length=1, max_length=3)]
    timing_mode: TimingMode
    delayed_feedback: Annotated[bool, Field(strict=True)]
    sampling_statuses: Annotated[tuple[SamplingStatus, ...], Field(min_length=1, max_length=4)]
    receipt_status: Literal["known", "unknown"]
    state_owners: Annotated[tuple[StateOwner, ...], Field(min_length=1, max_length=3)]

    @field_validator("values")
    @classmethod
    def _values(cls, values: tuple[ScalarSpec, ...]) -> tuple[ScalarSpec, ...]:
        """Reject conflicting outcome schemas before canonical ordering."""
        return _schemas(values)

    @field_validator("inputs")
    @classmethod
    def _inputs(cls, values: tuple[ScalarSpec, ...]) -> tuple[ScalarSpec, ...]:
        """Retain uniquely identified global feature/prediction schemas."""
        return _input_schemas(values)

    @field_validator("origins", "state_owners", "sampling_statuses")
    @classmethod
    def _support(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Normalize nonempty support sets without deduplication."""
        return _unique(values)


class Capability(CoreContract):
    """Describe one implementation's structural direct-use requirements.

    Attributes
    ----------
    implementation : Reference
        Exact implementation and semantic version; names do not confer support.
    action_kinds, feedback_kinds : tuple of str
        Nonempty bounded support declarations. Coverage is not dimensionality.
    required_values : tuple of ScalarSpec
        Required feedback dimensions, roles, semantics and admissible bounds.
        Empty is valid for an admitted absent/stateless notification.
    required_inputs : tuple of ScalarSpec, optional
        Required global inputs in ``DecisionRequest.inputs`` only. Member input
        rows remain separate declared context, not an implicit alternative route.
    origins, timing_modes : tuple of str
        Nonempty supported outcome origins and temporal modes.
    supports_delayed_feedback : bool
        Supports additional logical feedback rounds; ordinary within-round
        outcome waiting is not that capability.
    requires_known_receipt : bool
        Requires a batch with actual known receipt metadata.
    requires_observed : bool
        Required input/outcome coordinates must contain observed finite values.
        Unused diagnostics retain their own missingness states.
    state_owner : {'host', 'learner', 'stateless'}
        One declared owner of algorithm state; no durable runtime is inferred.
    requires_sampling : bool
        Requires selection-probability evidence, not necessarily a randomized
        mechanism. Deterministic mass one can meet an exact requirement.
    sampling_requirement : {'not_required', 'exact', 'qualified_approximation'}
        ``requires_sampling`` is true exactly for the latter two. Exact accepts
        only exact evidence; qualified approximation also admits exact evidence.

    Raises
    ------
    ValueError
        Invalid support/schema/version, coerced boolean or contradictory sampling
        requirement. No support declaration is inferred or repaired.

    Notes
    -----
    Profiles are host attestations checked structurally. They do not verify an
    implementation's mathematics, source honesty or expected transform policy.
    Objective/transform qualification remains a separate concern.
    """

    implementation: Reference
    action_kinds: Annotated[tuple[ActionKind, ...], Field(min_length=1, max_length=7)]
    feedback_kinds: Annotated[tuple[FeedbackKind, ...], Field(min_length=1, max_length=5)]
    required_values: Annotated[tuple[ScalarSpec, ...], Field(max_length=MAX_DIMENSIONS)]
    required_inputs: Annotated[tuple[ScalarSpec, ...], Field(max_length=MAX_DIMENSIONS)] = ()
    origins: Annotated[tuple[Origin, ...], Field(min_length=1, max_length=3)]
    timing_modes: Annotated[tuple[TimingMode, ...], Field(min_length=1, max_length=3)]
    supports_delayed_feedback: Annotated[bool, Field(strict=True)]
    requires_known_receipt: Annotated[bool, Field(strict=True)]
    requires_observed: Annotated[bool, Field(strict=True)]
    state_owner: StateOwner
    requires_sampling: Annotated[bool, Field(strict=True)]
    sampling_requirement: SamplingRequirement

    @field_validator("required_values")
    @classmethod
    def _values(cls, values: tuple[ScalarSpec, ...]) -> tuple[ScalarSpec, ...]:
        """Canonicalize required outcome schemas with explicit versions."""
        return _schemas(values)

    @field_validator("required_inputs")
    @classmethod
    def _inputs(cls, values: tuple[ScalarSpec, ...]) -> tuple[ScalarSpec, ...]:
        """Validate global input requirements independently of member context."""
        return _input_schemas(values)

    @field_validator("action_kinds", "feedback_kinds", "origins", "timing_modes")
    @classmethod
    def _support(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Keep each advertised support set unique and canonical."""
        return _unique(values)

    @model_validator(mode="after")
    def _sampling(self) -> Self:
        """Reject contradictory selection-evidence declarations."""
        if self.requires_sampling != (self.sampling_requirement != "not_required"):
            raise ValueError("Sampling evidence flag and requirement disagree")
        return self


class Incompatibility(CoreImmutable):
    """Retain one stable structural rejection without executing user code.

    Attributes
    ----------
    code : str
        Stable incompatibility category, such as ``value_schema`` or ``timing``.
    detail : str
        Explanation referencing declared identities, without ambient host data.
    """

    code: Identifier
    detail: Identifier


class CompatibilityResult(CoreImmutable):
    """Return explicit structural acceptance or nonempty incompatibility reasons.

    Attributes
    ----------
    compatible : bool
        True exactly when there are no incompatibility reasons.
    reasons : tuple of Incompatibility
        At most 256 unique reasons in deterministic code/detail order.

    Raises
    ------
    ValueError
        Contradictory flag/reasons, duplicate reasons or malformed fields.

    Notes
    -----
    Compatibility does not certify a theorem, efficacy or executable effects.
    No algorithm, registry, filesystem or transport is consulted.
    """

    compatible: Annotated[bool, Field(strict=True)]
    reasons: Annotated[tuple[Incompatibility, ...], Field(max_length=256)]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Require deterministic distinct reasons and an honest result flag."""
        if self.compatible != (not self.reasons):
            raise ValueError("Compatibility flag and reasons disagree")
        keys = [(reason.code, reason.detail) for reason in self.reasons]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate incompatibility reason")
        object.__setattr__(
            self, "reasons", tuple(sorted(self.reasons, key=lambda reason: (reason.code, reason.detail)))
        )
        return self


def _schema_matches(provided: ScalarSpec, required: ScalarSpec) -> bool:
    """Check explicit schemas and interval containment without unit conversion."""
    if (provided.dimension, provided.semantics, provided.role) != (
        required.dimension,
        required.semantics,
        required.role,
    ):
        return False
    lower = required.minimum is None or (provided.minimum is not None and provided.minimum >= required.minimum)
    upper = required.maximum is None or (provided.maximum is not None and provided.maximum <= required.maximum)
    return lower and upper


def _schema_reasons(
    provided: tuple[ScalarSpec, ...], required: tuple[ScalarSpec, ...], code: str
) -> list[Incompatibility]:
    """Compare each required dimension while retaining unrelated extra schemas."""
    by_id = {value.dimension.id: value for value in provided}
    return [
        Incompatibility(
            code=code, detail=f"Missing or incompatible dimension {value.dimension.id}@{value.dimension.version}"
        )
        for value in required
        if value.dimension.id not in by_id or not _schema_matches(by_id[value.dimension.id], value)
    ]


def _support_reasons(problem: ProblemCapabilities, capability: Capability) -> list[Incompatibility]:
    """Check distinct action, coverage and origin support declarations."""
    checks = (
        (problem.space.kind in capability.action_kinds, "action_kind", "Action shape is unsupported"),
        (problem.feedback_kind in capability.feedback_kinds, "feedback_kind", "Feedback coverage is unsupported"),
        (set(problem.origins) <= set(capability.origins), "origin", "Outcome origin is unsupported"),
        (problem.timing_mode in capability.timing_modes, "timing", "Temporal mode is unsupported"),
        (
            not problem.delayed_feedback or capability.supports_delayed_feedback,
            "delay",
            "Delayed feedback is unsupported",
        ),
        (capability.state_owner in problem.state_owners, "state_owner", "Algorithm state ownership is incompatible"),
        (
            set(problem.sampling_statuses) <= _accepted_sampling_statuses(capability),
            "sampling",
            "Offered selection evidence does not always satisfy the required status",
        ),
        (
            not capability.requires_known_receipt or problem.receipt_status == "known",
            "receipt",
            "A known feedback receipt is not guaranteed",
        ),
    )
    return [Incompatibility(code=code, detail=detail) for accepted, code, detail in checks if not accepted]


def _coverage_shape(kind: FeedbackKind, action_kind: str) -> bool:
    """Reject coverage shapes that cannot refer to the declared action family."""
    if kind == "chosen":
        return action_kind == "choice"
    if kind == "component":
        return action_kind in {"subset", "slate", "acquisition"}
    return not (kind == "full_panel" and action_kind == "vector")


def validate_composition(problem: ProblemCapabilities, capability: Capability) -> CompatibilityResult:
    """Check declared structural compatibility before joining problem and method.

    Parameters
    ----------
    problem : ProblemCapabilities
        Actual action/feedback, input/outcome schemas, origins and temporal mode.
    capability : Capability
        One implementation's supported modes and required schemas/bounds.

    Returns
    -------
    CompatibilityResult
        Deterministic explicit reasons or compatible true. Every problem interval
        must be contained in its corresponding required admissible interval.

    Raises
    ------
    ValueError
        Malformed or unchecked model content; ordinary incompatibility is returned.

    Notes
    -----
    Requires no learner, runtime, plugin, dataset or global registry. Converts no
    units or objectives and grants no scientific or algorithmic guarantee.
    Exact/approximate selection-mass offers on mixture, vector or acquisition
    shapes reject even when the consumer does not require sampling evidence.
    """
    problem = ProblemCapabilities.model_validate(problem)
    capability = Capability.model_validate(capability)
    reasons = _support_reasons(problem, capability)
    if not _coverage_shape(problem.feedback_kind, problem.space.kind):
        reasons.append(Incompatibility(code="coverage_shape", detail="Coverage cannot describe this action shape"))
    offered_mass = any(status in {"exact", "qualified_approximation"} for status in problem.sampling_statuses)
    if problem.space.kind in {"mixture", "vector", "acquisition"} and (capability.requires_sampling or offered_mass):
        reasons.append(Incompatibility(code="sampling", detail="Action shape has no admitted finite selection mass"))
    if problem.feedback_kind != "absent":
        reasons.extend(_schema_reasons(problem.values, capability.required_values, "value_schema"))
    reasons.extend(_schema_reasons(problem.inputs, capability.required_inputs, "input_schema"))
    return CompatibilityResult(compatible=not reasons, reasons=tuple(reasons))


def _accepted_sampling_statuses(capability: Capability) -> frozenset[SamplingStatus]:
    """Share declared and concrete evidence requirements without weakening either."""
    if capability.sampling_requirement == "exact":
        return frozenset({"exact"})
    if capability.sampling_requirement == "qualified_approximation":
        return frozenset({"exact", "qualified_approximation"})
    return _ALL_SAMPLING_STATUSES


def _sampling(record: SamplingRecord, capability: Capability) -> None:
    """Require actual bound selection evidence without inferring randomness."""
    if record.status not in _accepted_sampling_statuses(capability):
        raise ValueError("Required sampling evidence is unavailable or insufficiently exact")


def _required_values(values: tuple[Scalar, ...], required: tuple[ScalarSpec, ...], observed: bool) -> None:
    """Check required dimensions and observedness, preserving unused diagnostics."""
    reasons = _schema_reasons(tuple(value.spec for value in values), required, "value_schema")
    if reasons:
        raise ValueError(reasons[0].detail)
    required_ids = {value.dimension.id for value in required}
    if observed and any(value.spec.dimension.id in required_ids and value.status != "observed" for value in values):
        raise ValueError("Required dimensions must contain observed values")


def _request_requirements(decision: Decision, capability: Capability) -> None:
    """Check request context without confusing proposed and executed actions."""
    if decision.implementation != capability.implementation or decision.state_owner != capability.state_owner:
        raise ValueError("Decision implementation/version or state owner is incompatible")
    mode = "untimed" if decision.request.timing is None else decision.request.timing.mode
    if mode not in capability.timing_modes:
        raise ValueError("Decision temporal mode is unsupported")
    _required_values(decision.request.inputs, capability.required_inputs, capability.requires_observed)


def validate_decision(decision: Decision, capability: Capability) -> None:
    """Validate a real typed decision against its declared direct-use capability.

    Parameters
    ----------
    decision : Decision
        Concrete request, action, implementation, sampling and state ownership.
    capability : Capability
        Supported shape, global input schema, timing and selection evidence.

    Raises
    ------
    ValueError
        Implementation/version, action shape, ownership, required global inputs,
        observation state, timing or sampling evidence is incompatible.

    Notes
    -----
    Member context cannot satisfy required global inputs. Revalidates unchecked
    content, performs no algorithm call and creates no evidence or durable state.
    """
    decision = Decision.model_validate(decision)
    capability = Capability.model_validate(capability)
    _request_requirements(decision, capability)
    if decision.action.space.kind not in capability.action_kinds:
        raise ValueError("Decision action shape is unsupported")
    _sampling(decision.sampling, capability)


def _feedback_rows(executed: ExecutedDecision, feedback: Feedback) -> tuple[FeedbackRow, ...]:
    """Validate actual selected member coverage before reading row outcomes."""
    if isinstance(feedback, FullPanel):
        expected = space_members(executed.action.space)
        if tuple(row.member for row in feedback.rows) != expected:
            raise ValueError("Full panel must match exact active members and versions")
        return feedback.rows
    if isinstance(feedback, ChosenFeedback):
        if not isinstance(executed.action, ChoiceAction) or feedback.row.member != executed.action.chosen:
            raise ValueError("Chosen feedback must match the actual selected choice and version")
        return (feedback.row,)
    if isinstance(feedback, ComponentFeedback):
        selected = {(member.id, member.version) for member in selected_members(executed.action)}
        observed = {(row.member.id, row.member.version) for row in feedback.rows}
        if not observed <= selected or (feedback.completeness == "complete" and observed != selected):
            raise ValueError("Component feedback must cover declared actual selected members and completeness")
        return feedback.rows
    return ()


def _outcomes(values: tuple[OutcomeValue, ...], capability: Capability) -> None:
    """Validate required quantity schemas and every explicitly retained origin."""
    _required_values(values, capability.required_values, capability.requires_observed)
    if any(value.origin not in capability.origins for value in values):
        raise ValueError("Feedback contains an unsupported outcome origin")


def _transforms(rows: tuple[FeedbackRow, ...], required: tuple[ScalarSpec, ...]) -> None:
    """Keep one required dimension's transform identity consistent across rows."""
    wanted = {value.dimension.id for value in required}
    known: dict[str, Reference | None] = {}
    for row in rows:
        for value in row.values:
            name = value.spec.dimension.id
            if name in wanted:
                previous = known.setdefault(name, value.transform)
                if previous != value.transform:
                    raise ValueError("Required outcome dimensions cannot mix transform identities or versions")


def _timing(
    executed: ExecutedDecision, batch: FeedbackBatch | None, capability: Capability, apply_at: Timestamp | None
) -> None:
    """Check explicit update time against known feedback clocks without inventing receipt."""
    timed = executed.proposal.request.timing is not None
    timing = None if batch is None else batch.timing
    if batch is not None and batch.delay_rounds > 0 and not capability.supports_delayed_feedback:
        raise ValueError("Additional logical feedback rounds are unsupported")
    if timed and timing is None:
        raise ValueError("Timed feedback requires an explicit batch timing envelope")
    if capability.requires_known_receipt and (timing is None or timing.received_at is None):
        raise ValueError("This capability requires a known feedback receipt")
    if timing is not None:
        if apply_at is None:
            raise ValueError("Timed feedback requires explicit apply_at")
        instant = TypeAdapter(Timestamp).validate_python(apply_at)
        clocks = [timing.event_at, timing.available_at, timing.mature_at]
        if timing.received_at is not None:
            clocks.append(timing.received_at)
        if instant < max(clocks):
            raise ValueError("Feedback cannot apply before known availability, maturity or receipt")


def validate_feedback(
    executed: ExecutedDecision,
    feedback_or_batch: Feedback | FeedbackBatch,
    capability: Capability,
    *,
    apply_at: Timestamp | None = None,
) -> None:
    """Validate actual feedback coverage and eligibility without applying an update.

    Parameters
    ----------
    executed : ExecutedDecision
        Actual immutable action, proposal and its preserved/recomputed/invalidated
        selection evidence. Batch identity must bind this complete content.
    feedback_or_batch : Feedback or FeedbackBatch
        Typed direct feedback or a versioned temporal/revision envelope. Plain
        feedback is admitted only for untimed modes without a receipt requirement.
    capability : Capability
        Exact consumer implementation/profile; required dimensions and intervals
        are checked independently from observed member coverage.
    apply_at : Timestamp or None, optional
        Explicit zoned update instant when timing is supplied/required. Must not
        precede any known event, availability, maturity or receipt time.

    Raises
    ------
    ValueError
        Bad executed binding, unsupported coverage, incorrect actual members,
        schema/units/bounds/origin, missing required values/receipt, timing,
        unsupported delay, incompatible transforms or invalidated sampling.
    TypeError
        Feedback is not one of the declared typed variants or a FeedbackBatch.

    Notes
    -----
    Admitted absence supplies no zero and performs no adaptive update. Required
    transforms must agree across rows, but this function executes none and
    certifies no expected-transform policy, source lineage or causal inference.
    It performs no I/O, registry lookup, replay ordering, deduplication or commit.
    """
    executed = ExecutedDecision.model_validate(executed)
    capability = Capability.model_validate(capability)
    _request_requirements(executed.proposal, capability)
    if executed.action.space.kind not in capability.action_kinds:
        raise ValueError("Executed action shape is unsupported")
    batch = feedback_or_batch if isinstance(feedback_or_batch, FeedbackBatch) else None
    if batch is not None:
        batch = FeedbackBatch.model_validate(batch)
        if batch.decision_fingerprint != executed.fingerprint:
            raise ValueError("Feedback batch does not bind this executed decision content")
        feedback = batch.feedback
    elif isinstance(
        feedback_or_batch, FullPanel | ChosenFeedback | ComponentFeedback | AggregateFeedback | AbsentFeedback
    ):
        feedback = type(feedback_or_batch).model_validate(feedback_or_batch)
    else:
        raise TypeError("Feedback validation requires a typed feedback variant or batch")
    if feedback.kind not in capability.feedback_kinds:
        raise ValueError("Feedback coverage is unsupported")
    if not _coverage_shape(feedback.kind, executed.action.space.kind):
        raise ValueError("Feedback coverage cannot describe this actual action shape")
    _sampling(executed.sampling, capability)
    _timing(executed, batch, capability, apply_at)
    rows = _feedback_rows(executed, feedback)
    for row in rows:
        _outcomes(row.values, capability)
    _transforms(rows, capability.required_values)
    if isinstance(feedback, AggregateFeedback):
        _outcomes(feedback.values, capability)
