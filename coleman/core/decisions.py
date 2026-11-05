"""Separate supplied signals, proposals and actual actions without orchestration."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.actions import Action, ActionSpace, space_members
from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference, ordered_references
from coleman.core.numeric import Scalar
from coleman.core.primitives import MAX_DIMENSIONS, MAX_MEMBERS, Identifier, canonical_json, fingerprint
from coleman.core.sampling import SamplingRecord, validate_sampling
from coleman.core.temporal import Timestamp
from coleman.core.timing import DecisionTiming

StateOwner = Literal["host", "learner", "stateless"]


def _input_values(values: tuple[Scalar, ...]) -> tuple[Scalar, ...]:
    """Canonicalize explicit input dimensions without imputing missing values."""
    ordered_references(tuple(value.spec.dimension for value in values))
    if any(value.spec.role not in {"feature", "prediction"} for value in values):
        raise ValueError("Decision inputs must declare feature or prediction roles")
    return tuple(sorted(values, key=lambda value: value.spec.dimension.id))


class MemberInputs(CoreImmutable):
    """Attach explicit contextual dimensions to one feasible member.

    Attributes
    ----------
    member : Reference
        Exact versioned member, checked against the request universe.
    values : tuple of Scalar
        Unique feature/prediction dimensions with units, meaning and explicit
        missingness. Canonicalized by dimension ID, at most 64.

    Notes
    -----
    A member's fields do not satisfy global required-input declarations. Missing
    rows are not implicit zero features. No calibration or learning occurs.
    """

    member: Reference
    values: Annotated[tuple[Scalar, ...], Field(max_length=MAX_DIMENSIONS)]

    @model_validator(mode="after")
    def _values(self) -> Self:
        """Validate and canonicalize declared member context without imputation."""
        object.__setattr__(self, "values", _input_values(self.values))
        return self


class SuppliedSignal(CoreImmutable):
    """Retain an external expert/signal output separately from learner output.

    Attributes
    ----------
    source : Reference
        Stable producer/expert identity and exact semantic version.
    transform : Reference or None
        Required explicit optional transform declaration, without executing it.
    values : tuple of Scalar
        Unique feature/prediction dimensions with units and missingness.
    available_at : datetime or None
        Required nullable zoned availability. Timed requests require a known
        instant no later than their information cutoff.

    Notes
    -----
    No implicit prediction, probability calibration or source authentication is
    inferred. Unknown availability is retained for untimed problems only.
    """

    source: Reference
    transform: Reference | None
    values: Annotated[tuple[Scalar, ...], Field(max_length=MAX_DIMENSIONS)]
    available_at: Timestamp | None

    @model_validator(mode="after")
    def _values(self) -> Self:
        """Keep signal input roles explicit and dimensions uniquely versioned."""
        object.__setattr__(self, "values", _input_values(self.values))
        return self


class DecisionRequest(CoreContract):
    """Versioned domain-neutral input to a direct decision call.

    Attributes
    ----------
    schema_version : int
        Required version 1. Unknown versions/fields reject without migration.
    task : Reference
        Stable logical task identity and semantic version.
    decision_id : str
        Caller-owned logical opportunity ID, stable across delivery attempts.
    round : int
        Explicit nonnegative logical round, not a timestamp or independent seed.
    space : ActionSpace
        Feasible action declaration with exact versions.
    objective : Reference
        Versioned objective meaning, not an executed reward transform.
    inputs : tuple of Scalar, optional
        Global feature/prediction inputs, unique by dimension ID.
    member_inputs : tuple of MemberInputs, optional
        Explicit feasible-member contexts; no implicit complete panel.
    signals : tuple of SuppliedSignal, optional
        Unique external sources, separate from the learner's action.
    timing : DecisionTiming or None, optional
        Explicit information/target clocks, or genuinely untimed.

    Notes
    -----
    Set-like fields are canonically ordered. Only task+decision_id define
    logical identity; the content fingerprint additionally covers all inputs.
    Construction performs no I/O, scheduling, learning or external action.
    """

    task: Reference
    decision_id: Identifier
    round: Annotated[int, Field(strict=True, ge=0)]
    space: ActionSpace
    objective: Reference
    inputs: Annotated[tuple[Scalar, ...], Field(max_length=MAX_DIMENSIONS)] = ()
    member_inputs: Annotated[tuple[MemberInputs, ...], Field(max_length=MAX_MEMBERS)] = ()
    signals: Annotated[tuple[SuppliedSignal, ...], Field(max_length=MAX_MEMBERS)] = ()
    timing: DecisionTiming | None = None

    @model_validator(mode="after")
    def _context(self) -> Self:
        """Check active member versions, unique sources and eligible signal time."""
        object.__setattr__(self, "inputs", _input_values(self.inputs))
        ordered_references(tuple(row.member for row in self.member_inputs))
        if not set(row.member for row in self.member_inputs).issubset(space_members(self.space)):
            raise ValueError("Member inputs must reference exact feasible members")
        ordered_references(tuple(signal.source for signal in self.signals))
        self._signal_times()
        object.__setattr__(self, "member_inputs", tuple(sorted(self.member_inputs, key=lambda row: row.member.id)))
        object.__setattr__(self, "signals", tuple(sorted(self.signals, key=lambda signal: signal.source.id)))
        return self

    def _signal_times(self) -> None:
        """Reject unknown or post-cutoff signal availability in timed requests."""
        if self.timing is None:
            return
        if any(
            signal.available_at is None or signal.available_at > self.timing.information_cutoff
            for signal in self.signals
        ):
            raise ValueError("Timed supplied signals require known availability at or before the cutoff")

    @property
    def logical_identity(self) -> str:
        """Return stable task/opportunity identity independent of physical retries.

        Returns
        -------
        str
            SHA-256 identity over versioned task and logical decision ID.

        Notes
        -----
        Does not claim content equality, global registration or authenticity.
        Full records revalidate when serialized; no state or I/O is involved.
        """
        return fingerprint({"task": self.task.model_dump(mode="json"), "decision_id": self.decision_id})


class Decision(CoreContract):
    """Versioned learner/solver proposal, distinct from the action actually used.

    Attributes
    ----------
    schema_version : int
        Required version 1.
    request : DecisionRequest
        Complete input and stable logical identity.
    implementation : Reference
        Versioned implementation producing the proposal.
    action : Action
        Proposed action, retaining the exact request space.
    sampling : SamplingRecord
        Evidence bound to the proposal's action content, not its ranks/weights.
    state_owner : {'host', 'learner', 'stateless'}
        Explicit state policy; no state is created or persisted here.

    Raises
    ------
    ValueError
        Proposal universe differs or sampling content/interpretation is invalid.
    """

    request: DecisionRequest
    implementation: Reference
    action: Action
    sampling: SamplingRecord
    state_owner: StateOwner

    @model_validator(mode="after")
    def _proposal(self) -> Self:
        """Require the requested universe and action-bound sampling declaration."""
        proposed = canonical_json(self.action.space.model_dump(mode="json"))
        requested = canonical_json(self.request.space.model_dump(mode="json"))
        if proposed != requested:
            raise ValueError("Proposal must retain the exact requested action space")
        validate_sampling(self.action, self.sampling)
        return self

    @property
    def logical_identity(self) -> str:
        """Return the request's logical identity without creating an attempt ID.

        Returns
        -------
        str
            Stable versioned task/opportunity digest.
        """
        return self.request.logical_identity


class ExecutedDecision(CoreContract):
    """Bind actual action content and sampling disposition to a complete proposal.

    Attributes
    ----------
    schema_version : int
        Required version 1.
    proposal : Decision
        Exact originating proposal; never just a potentially ambiguous ID.
    action : Action
        Action actually used by the host. Explicit mapping may change its
        universe/shape; downstream capability checks use this actual action.
    mapping : {'unchanged', 'filtered', 'projected', 'overridden'}
        Required declaration of host-side processing, not executed here.
        Unchanged requires identical action and sampling content.
    sampling_disposition : {'preserved', 'recomputed', 'invalidated'}
        Preserved requires identical action and sampling content; recomputed
        binds new evidence; invalidated requires explicit unavailable evidence.
    sampling : SamplingRecord
        Sampling evidence bound to the actual action's content.

    Notes
    -----
    This is an observation/envelope, not an execution engine or exactly-once
    delivery receipt. Mapping declarations are checked structurally, not proved.
    """

    proposal: Decision
    action: Action
    mapping: Literal["unchanged", "filtered", "projected", "overridden"]
    sampling_disposition: Literal["preserved", "recomputed", "invalidated"]
    sampling: SamplingRecord

    @model_validator(mode="after")
    def _execution(self) -> Self:
        """Check actual action binding and explicit preservation/invalidation."""
        same_action = self.action.fingerprint == self.proposal.action.fingerprint
        if self.mapping == "unchanged":
            self._preserved(same_action)
            if self.sampling_disposition != "preserved":
                raise ValueError("Unchanged execution must preserve its sampling disposition")
        validate_sampling(self.action, self.sampling)
        if self.sampling_disposition == "preserved":
            self._preserved(same_action)
        if self.sampling_disposition == "invalidated" and self.sampling.status != "unavailable":
            raise ValueError("Invalidated sampling must explicitly be unavailable")
        return self

    def _preserved(self, same_action: bool) -> None:
        """Require exact serialized sampling bytes and unchanged action content."""
        old = canonical_json(self.proposal.sampling.model_dump(mode="json"))
        new = canonical_json(self.sampling.model_dump(mode="json"))
        if not same_action or old != new:
            raise ValueError("Preserved sampling requires identical action and sampling bytes")

    @property
    def logical_identity(self) -> str:
        """Return the originating task/opportunity identity.

        Returns
        -------
        str
            Stable identity independent of delivery attempts and mapping output.
        """
        return self.proposal.logical_identity


def deduplicate_decisions(values: tuple[Decision, ...]) -> tuple[Decision, ...]:
    """Deduplicate bounded exact proposal retries and reject logical conflicts.

    Parameters
    ----------
    values : tuple of Decision
        At most 4096 attempts. Each supported writer revalidates complete data.

    Returns
    -------
    tuple of Decision
        One proposal per logical task/opportunity, sorted by logical identity.

    Raises
    ------
    ValueError
        Invalid shape/count or same logical identity with different canonical
        content, including signed-zero distinctions.

    Notes
    -----
    Pure bounded in-memory reconciliation. It does not persist delivery keys,
    make learner updates idempotent, choose revisions or promise exactly-once.
    """
    if not isinstance(values, tuple) or len(values) > MAX_MEMBERS:
        raise ValueError("Decision retries require a tuple of at most 4096 attempts")
    found: dict[str, tuple[str, Decision]] = {}
    for value in values:
        checked = Decision.model_validate(value)
        key, payload = checked.logical_identity, checked.to_json()
        if key in found and found[key][0] != payload:
            raise ValueError("Conflicting decision content for one logical opportunity")
        found[key] = (payload, checked)
    return tuple(found[key][1] for key in sorted(found))
