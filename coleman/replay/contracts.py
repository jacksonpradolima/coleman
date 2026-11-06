"""Immutable causal replay records without host execution or storage access."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.capabilities import Capability, validate_decision
from coleman.core.decisions import DecisionRequest, ExecutedDecision
from coleman.core.feedback import (
    AbsentFeedback,
    AggregateFeedback,
    ChosenFeedback,
    ComponentFeedback,
    Feedback,
    FeedbackBatch,
    Origin,
    OutcomeValue,
)
from coleman.core.identity import Reference, ordered_references
from coleman.core.primitives import MAX_MEMBERS, Fingerprint, fingerprint
from coleman.core.temporal import Timestamp
from coleman.objectives.bindings import DefinitionBinding
from coleman.objectives.context import SemanticContext, SemanticRequest
from coleman.observations.contracts import ObservationHistory, ObservationSnapshot

MAX_REPLAY_RECORDS = MAX_MEMBERS
MAX_REPLAY_EVENTS = MAX_REPLAY_RECORDS
MAX_REPLAY_VALUES = MAX_MEMBERS * 64
Count = Annotated[int, Field(strict=True, ge=0)]


def _wire(value: CoreImmutable | CoreContract) -> str:
    """Compare complete finite canonical content, including signed zero."""
    return value.model_dump_json()


def outcome_fingerprint(value: OutcomeValue) -> str:
    """Identify one validated outcome without inferring its source or meaning.

    Parameters
    ----------
    value : OutcomeValue
        One typed value, including its schema, null state, origin and transform.

    Returns
    -------
    str
        COLEMAN JSON v1 SHA-256; signed zero remains content-significant.

    Raises
    ------
    ValueError, TypeError
        Invalid or unchecked value content cannot be encoded.

    Notes
    -----
    Performs no source lookup, transform, model execution or I/O.
    """
    checked = OutcomeValue.model_validate(value)
    return fingerprint(json.loads(checked.model_dump_json()))


def outcome_coordinates(feedback: Feedback) -> tuple[tuple[Reference | None, OutcomeValue], ...]:
    """Enumerate only the values explicitly supplied by typed feedback.

    Parameters
    ----------
    feedback : Feedback
        Full, chosen, component, aggregate or absent payload.

    Returns
    -------
    tuple
        Member/value pairs; aggregate members are null and absence is empty.

    Notes
    -----
    Reveals no absent alternatives and never fans aggregate values into members.
    Exact execution coverage is validated separately by the core capability API.
    """
    if isinstance(feedback, AbsentFeedback):
        return ()
    if isinstance(feedback, AggregateFeedback):
        return tuple((None, value) for value in feedback.values)
    rows = (feedback.row,) if isinstance(feedback, ChosenFeedback) else feedback.rows
    return tuple((row.member, value) for row in rows for value in row.values)


class ReplayParent(CoreImmutable):
    """Bind a child to an unchanged complete parent checkpoint and causal prefix.

    Attributes
    ----------
    lineage : Reference
        Parent lineage identity/version, distinct from the active child.
    state_fingerprint, prefix_fingerprint : str
        Complete parent checkpoint state and its causal-prefix content digests.
    next_sequence : int
        First child sequence; all preceding causal records remain unchanged.
    host_fingerprint : str
        Complete host snapshot digest at the parent boundary, including RNG.
    ancestors : tuple of Reference
        Complete ordered root-to-parent lineage identities, ending in lineage.

    Notes
    -----
    Digests authenticate nothing. The session verifies a complete typed matched
    continuation before forking; no checkpoint store or migration is invoked.
    """

    lineage: Reference
    state_fingerprint: Fingerprint
    prefix_fingerprint: Fingerprint
    next_sequence: Count
    host_fingerprint: Fingerprint
    ancestors: Annotated[tuple[Reference, ...], Field(min_length=1, max_length=MAX_MEMBERS)]

    @model_validator(mode="after")
    def _ancestry(self) -> Self:
        """Retain complete lineage names even when the parent emitted no records."""
        names = tuple(value.model_dump_json() for value in self.ancestors)
        if len(set(names)) != len(names) or self.ancestors[-1] != self.lineage:
            raise ValueError("Replay parent ancestry must be unique and end in its exact lineage")
        return self


class ReplayPolicy(CoreContract):
    """Declare a bounded closed-drain profile without transferring a theorem.

    Attributes
    ----------
    stream, lineage : Reference
        Logical task stream and active execution lineage, with explicit versions.
    parent : ReplayParent or None
        Matched parent checkpoint, or explicit null for an independent origin.
    source : Reference
        ObservationHistory source_id/source_version, not future-content digest.
    capability : Capability
        Exact host implementation requirements, schemas and state ownership.
    delivery : {'synchronous', 'delayed'}
        Explicit schedule support; delayed requires the capability's support.
    specialists : bool
        Explicit host attestation of changing active-member support.
    correction : {'preserve_live_v1'}, optional
        Consume each logical feedback round once; later corrections audit only.
    ordering : {'closed-drain-v1'}, optional
        Actual drain calls are processing inputs; no partition invariance claim.

    Raises
    ------
    ValueError
        A child repeats its parent lineage or delayed support is contradictory.

    Notes
    -----
    Bounds are fixed by this codec. No algorithm, clock, registry or I/O occurs.
    """

    stream: Reference
    lineage: Reference
    parent: ReplayParent | None
    source: Reference
    capability: Capability
    delivery: Literal["synchronous", "delayed"]
    specialists: Annotated[bool, Field(strict=True)]
    correction: Literal["preserve_live_v1"] = "preserve_live_v1"
    ordering: Literal["closed-drain-v1"] = "closed-drain-v1"

    @model_validator(mode="after")
    def _profile(self) -> Self:
        """Reject declarations incompatible with their selected capability."""
        if self.parent is not None and self.lineage in self.parent.ancestors:
            raise ValueError("Child replay lineage must differ from every retained ancestor")
        if self.delivery == "delayed" and not self.capability.supports_delayed_feedback:
            raise ValueError("Delayed replay requires explicit delayed capability support")
        return self


class ReplayTarget(CoreImmutable):
    """Retain target, horizon and dependence identities without independence claims.

    Attributes
    ----------
    reference, horizon : Reference
        Exact logical target and measurement horizon versions.
    groups : tuple of Reference
        Canonical bounded dependence groups; empty asserts no known group.

    Notes
    -----
    Target interval belongs to producing DecisionTiming, never inferred here.
    """

    reference: Reference
    horizon: Reference
    groups: Annotated[tuple[Reference, ...], Field(max_length=MAX_MEMBERS)]

    @model_validator(mode="after")
    def _groups(self) -> Self:
        """Keep group versions unique without fabricating independence."""
        object.__setattr__(self, "groups", ordered_references(self.groups))
        return self


class ReplayOpportunity(CoreContract):
    """Bind an unchanged core request to optional full producing semantics.

    Attributes
    ----------
    request : DecisionRequest
        Timed request; round counts emitted opportunities, never updates.
    context : SemanticContext or None
        Required nullable semantic contract; null confers no verified context.
    target : ReplayTarget
        Exact target, horizon and dependence identities for its one panel round.

    Raises
    ------
    ValueError
        Untimed input, mismatched producing context, frozen scale or horizon.

    Notes
    -----
    No features are derived and no learner or source is called at construction.
    """

    request: DecisionRequest
    context: SemanticContext | None
    target: ReplayTarget

    @model_validator(mode="after")
    def _producing(self) -> Self:
        """Retain exact original clocks and complete optional transform content."""
        if self.request.timing is None:
            raise ValueError("Causal replay requires explicit producing decision timing")
        if self.context is not None:
            SemanticRequest(schema_version=1, request=self.request, context=self.context)
            if any(term.quantity.horizon != self.target.horizon for term in self.context.objective.terms):
                raise ValueError("Producing objective horizons must match this explicit replay target")
        return self


class PreparedDecision(CoreContract):
    """Retain only a selected eligible view and a session-bound opportunity.

    Attributes
    ----------
    opportunity : ReplayOpportunity
        Validated request, optional context and target.
    snapshot : ObservationSnapshot
        Actual as_of output; no future history or hidden outcome oracle.
    state_fingerprint, host_state_fingerprint : str
        Expected session metadata and complete host state at preparation time.

    Raises
    ------
    ValueError
        Snapshot cutoff or eligible input digest differs from the request.

    Notes
    -----
    A snapshot alone attests supplied completeness; the session performs as_of.
    Matching fingerprints do not prove arbitrary input feature derivation.
    """

    opportunity: ReplayOpportunity
    snapshot: ObservationSnapshot
    state_fingerprint: Fingerprint
    host_state_fingerprint: Fingerprint

    @model_validator(mode="after")
    def _view(self) -> Self:
        """Bind the exact selected view to the original information cutoff."""
        timing = self.opportunity.request.timing
        if timing is None or self.snapshot.information_cutoff != timing.information_cutoff:
            raise ValueError("Prepared view cutoff differs from the producing request")
        if self.snapshot.fingerprint != timing.eligible_input_fingerprint:
            raise ValueError("Prepared eligible view fingerprint differs from the request")
        return self


class ValueOriginEvidence(CoreImmutable):
    """Attach a model or generator binding to exactly one supplied value.

    Attributes
    ----------
    member : Reference or None
        Exact member/version, or null only for whole-action aggregate feedback.
    dimension : Reference
        Exact value dimension/version, independent of its numerical units.
    value_fingerprint : str
        Digest of the complete OutcomeValue including null state and signed zero.
    origin : {'realized', 'modeled_counterfactual', 'synthetic'}
        Must equal the actual value origin; never promoted during replay.
    producer : DefinitionBinding or None
        Model/generator version and content digest for modeled/synthetic values;
        explicit null for realized observations, whose source remains separate.

    Raises
    ------
    ValueError
        Non-realized values lack their producer, or realized values claim one.

    Notes
    -----
    Content binding is not source authentication or a model-quality claim.
    """

    member: Reference | None
    dimension: Reference
    value_fingerprint: Fingerprint
    origin: Origin
    producer: DefinitionBinding | None

    @model_validator(mode="after")
    def _producer(self) -> Self:
        """Preserve explicit realized versus model/generator provenance."""
        if (self.origin == "realized") != (self.producer is None):
            raise ValueError("Exactly non-realized values require a model or generator binding")
        return self


class ReplayFeedback(CoreContract):
    """Retain one immutable feedback revision without acceptance or application.

    Attributes
    ----------
    lineage : Reference
        PRODUCING decision lineage, including an inherited parent in a child run.
    batch : FeedbackBatch
        Unchanged core v1 executed binding, revision, values and clocks.
    target : ReplayTarget
        Exact producing target/horizon/groups; no cross-target joining by name.
    origins : tuple of ValueOriginEvidence
        Exactly one annotation per supplied coordinate; no hidden alternatives.

    Raises
    ------
    ValueError
        Missing clocks/horizon or conflicting/missing/extraneous origin facts.

    Notes
    -----
    Unknown receipt and required nulls remain valid pending facts. Graph ancestry
    and actual execution coverage are checked by replay's graph helpers.
    """

    lineage: Reference
    batch: FeedbackBatch
    target: ReplayTarget
    origins: Annotated[tuple[ValueOriginEvidence, ...], Field(max_length=MAX_REPLAY_VALUES)]

    @model_validator(mode="after")
    def _evidence(self) -> Self:
        """Require complete exact per-value evidence without executing a model."""
        if self.batch.timing is None or self.batch.horizon != self.target.horizon:
            raise ValueError("Replay feedback requires explicit timing and its exact target horizon")
        from coleman.replay.revisions import _checked_origins  # pylint: disable=import-outside-toplevel

        object.__setattr__(self, "origins", _checked_origins(self))
        return self

    @property
    def logical_key(self) -> tuple[str, str, str]:
        """Return producing lineage/execution/target identity, excluding revision.

        Returns
        -------
        tuple of str
            Canonical exact references and the complete executed fingerprint.

        Notes
        -----
        feedback_id aliases cannot create a second round for this key.
        """
        return _wire(self.lineage), self.batch.decision_fingerprint, _wire(self.target.reference)


class ReplayApplication(CoreContract):
    """Supply an actual update clock to the existing generic Updater protocol.

    Attributes
    ----------
    lineage : Reference
        ACTIVE session lineage, possibly different from the producing lineage.
    executed : ExecutedDecision
        Exact actual action and proposal matched by feedback.batch.
    feedback : ReplayFeedback
        Selected revision only; no other alternatives or future revision values.
    eligible_at, apply_at : datetime
        Latest required known clock and actual supplied processing instant.
    decision_count_before, update_count_before : int
        Separate successful decision and completed-update counts before this call.

    Raises
    ------
    ValueError
        Bad binding, unknown receipt, partial/absent payload or inconsistent clocks.

    Notes
    -----
    Capability validation still checks required observed dimensions and units.
    Construction does not call a host or infer apply_at from receipt time.
    """

    lineage: Reference
    executed: ExecutedDecision
    feedback: ReplayFeedback
    eligible_at: Timestamp
    apply_at: Timestamp
    decision_count_before: Count
    update_count_before: Count

    @model_validator(mode="after")
    def _application(self) -> Self:
        """Reject unchecked update envelopes before a host receives them."""
        from coleman.replay.revisions import (  # pylint: disable=import-outside-toplevel
            _application_clocks,
            _application_counts,
        )

        _application_clocks(self)
        _application_counts(self)
        payload = self.feedback.batch.feedback
        if isinstance(payload, AbsentFeedback):
            raise ValueError("Absent feedback cannot complete an update")
        if isinstance(payload, ComponentFeedback) and payload.completeness != "complete":
            raise ValueError("Partial component feedback cannot complete an update")
        return self


class ReplayDecisionRecord(CoreContract):
    """Retain one successful simulated decision without rewriting its lineage.

    Attributes
    ----------
    lineage : Reference
        Producing lineage, unchanged when inherited by a child checkpoint.
    opportunity : ReplayOpportunity
        Exact producing request, optional semantic context and target.
    executed : ExecutedDecision
        Explicitly simulated actual action, retaining the producing proposal.
    snapshot : ObservationSnapshot
        Eligible source view with exactly the request's cutoff and input digest.
    decision_index, update_count_before : int
        Zero-based successful decision position and preceding completed updates.
    accepted_at : datetime
        Producing decision_at; not an information frontier.
    host_before, host_after : str
        Complete host snapshot digests across the actual decide call.
    sequence : int
        Total causal acceptance order, resolving equal timestamps.
    execution : {'offline_identity_v1'}, optional
        Explicit simulated unchanged action, never a physical-effect receipt.

    Raises
    ------
    ValueError
        Inconsistent request, view, counts or implied physical execution.
    """

    lineage: Reference
    opportunity: ReplayOpportunity
    executed: ExecutedDecision
    snapshot: ObservationSnapshot
    decision_index: Count
    update_count_before: Count
    accepted_at: Timestamp
    host_before: Fingerprint
    host_after: Fingerprint
    sequence: Count
    execution: Literal["offline_identity_v1"] = "offline_identity_v1"

    @model_validator(mode="after")
    def _record(self) -> Self:
        """Keep producing context and explicit offline action mapping exact."""
        request = self.opportunity.request
        if request.to_json() != self.executed.proposal.request.to_json():
            raise ValueError("Emitted decision changed the producing request")
        if request.round != self.decision_index or self.update_count_before > self.decision_index:
            raise ValueError("Decision record counters disagree with its logical opportunity")
        timing = request.timing
        if timing is None or timing.decision_at != self.accepted_at:
            raise ValueError("Decision acceptance differs from the producing decision instant")
        if (
            self.snapshot.information_cutoff != timing.information_cutoff
            or self.snapshot.fingerprint != timing.eligible_input_fingerprint
        ):
            raise ValueError("Decision record differs from its exact eligible source snapshot")
        if self.executed.mapping != "unchanged" or self.executed.sampling_disposition != "preserved":
            raise ValueError("Offline identity execution requires an explicitly unchanged action")
        return self

    @property
    def logical_identity(self) -> str:
        """Return lineage-scoped stable opportunity identity without I/O.

        Returns
        -------
        str
            Digest of producing lineage and core task/opportunity identity.
        """
        return fingerprint(
            {"lineage": self.lineage.model_dump(mode="json"), "decision": self.executed.logical_identity}
        )


class ReplayUpdateRecord(CoreContract):
    """Retain one completed host update and its actual application envelope.

    Attributes
    ----------
    lineage : Reference
        Active session lineage, equal to application's active lineage.
    application : ReplayApplication
        Exact selected feedback, clocks, producing action and preceding counters.
    host_before, host_after : str
        Complete actual host state digests across the update call.
    sequence : int
        Total causal acceptance order, not statistical independence.
    """

    lineage: Reference
    application: ReplayApplication
    host_before: Fingerprint
    host_after: Fingerprint
    sequence: Count

    @model_validator(mode="after")
    def _lineage(self) -> Self:
        """Keep the active update lineage distinct from producer provenance."""
        if self.lineage != self.application.lineage:
            raise ValueError("Update record and application active lineages disagree")
        return self


class ReplayDrain(CoreContract):
    """Record an explicit closed-drain schedule, including an empty drain.

    Attributes
    ----------
    lineage : Reference
        Active processing lineage.
    eligible_through, apply_at : datetime
        Independent information and actual processing frontiers; eligible<=apply.
    decision_count, update_count_before, update_count_after : int
        Successful decisions and completed rounds around this explicit drain.
    host_before, host_after : str
        Complete host state at the start/end of this possibly multi-update drain.
    sequence : int
        Acceptance position after its update records and before any next decision.

    Notes
    -----
    One drain at30 may differ from drains at10 then30 after a correction at20.
    The recorded schedule makes no historical continuous-processing claim.
    """

    lineage: Reference
    eligible_through: Timestamp
    apply_at: Timestamp
    decision_count: Count
    update_count_before: Count
    update_count_after: Count
    host_before: Fingerprint
    host_after: Fingerprint
    sequence: Count

    @model_validator(mode="after")
    def _frontiers(self) -> Self:
        """Keep clocks and completed-round counts independent and ordered."""
        if self.eligible_through > self.apply_at:
            raise ValueError("Drain information frontier cannot follow actual processing time")
        if not self.update_count_before <= self.update_count_after <= self.decision_count:
            raise ValueError("Drain completed update counts are inconsistent")
        if self.update_count_before == self.update_count_after and self.host_before != self.host_after:
            raise ValueError("An empty drain cannot claim a changed host state")
        return self


def _prefix(stream: Reference, records: tuple[ReplayDecisionRecord | ReplayUpdateRecord | ReplayDrain, ...]) -> str:
    """Identify causal records without future inventory or a child header."""
    return fingerprint(
        {
            "profile": "replay-prefix-v1",
            "stream": stream.model_dump(mode="json"),
            "records": [record.model_dump(mode="json") for record in sorted(records, key=lambda item: item.sequence)],
        }
    )


class ReplayState(CoreContract):
    """Retain bounded immutable delivery metadata separately from host state.

    Attributes
    ----------
    policy : ReplayPolicy
        Current stream, active lineage, capability and closed-drain profile.
    history : ObservationHistory or None
        Full admitted source inventory, or explicit null before source admission.
    state_codec : Reference
        Explicit host snapshot semantic/codec identity; never dynamically imported.
    host_fingerprint : str
        Expected complete actual host state at this metadata frontier.
    decisions, feedback, updates, drains : tuple
        Bounded typed immutable records; feedback includes pending/audit revisions.
    eligible_through, apply_at : datetime or None, optional
        Independent explicit frontiers; null together before the first drain.

    Raises
    ------
    ValueError
        Invalid source, graph, counters, joins, host chain, frontier or bounds.

    Notes
    -----
    Full content changes on future admission. The causal-prefix fingerprint
    excludes inventory; this model performs no host calls, restore or I/O.
    """

    policy: ReplayPolicy
    history: ObservationHistory | None
    state_codec: Reference
    host_fingerprint: Fingerprint
    decisions: Annotated[tuple[ReplayDecisionRecord, ...], Field(max_length=MAX_REPLAY_RECORDS)] = ()
    feedback: Annotated[tuple[ReplayFeedback, ...], Field(max_length=MAX_REPLAY_RECORDS)] = ()
    updates: Annotated[tuple[ReplayUpdateRecord, ...], Field(max_length=MAX_REPLAY_RECORDS)] = ()
    drains: Annotated[tuple[ReplayDrain, ...], Field(max_length=MAX_REPLAY_RECORDS)] = ()
    eligible_through: Timestamp | None = None
    apply_at: Timestamp | None = None

    @model_validator(mode="after")
    def _state(self) -> Self:
        """Check complete retained evidence before accepting a continuation."""
        _state_inventory(self)
        _state_trace(self)
        # Graph helpers import only contract types lazily, never an engine/store.
        from coleman.replay.revisions import (  # pylint: disable=import-outside-toplevel
            _state_applications,
            validate_feedback_graph,
        )

        checked = validate_feedback_graph(self.feedback, self.decisions, self.policy.capability)
        if tuple(value.to_json() for value in checked) != tuple(value.to_json() for value in self.feedback):
            raise ValueError("State feedback graph must be in canonical logical/revision order")
        _state_applications(self)
        return self

    @property
    def decision_count(self) -> int:
        """Return the successful decision count, derived from retained records.

        Returns
        -------
        int
            Emitted logical opportunities, including inherited parent records.
        """
        return len(self.decisions)

    @property
    def update_count(self) -> int:
        """Return completed accepted rounds, excluding retries and pending facts.

        Returns
        -------
        int
            Number of successful retained host update records.
        """
        return len(self.updates)

    @property
    def next_sequence(self) -> int:
        """Return the next causal acceptance coordinate without changing state.

        Returns
        -------
        int
            Zero for an empty trace, otherwise its greatest sequence plus one.
        """
        records = (*self.decisions, *self.updates, *self.drains)
        return max((value.sequence for value in records), default=-1) + 1

    @property
    def causal_prefix_fingerprint(self) -> str:
        """Identify only emitted causal records, excluding future source inventory.

        Returns
        -------
        str
            Digest over stream and ordered decisions/updates/explicit drains.

        Notes
        -----
        The full state's fingerprint intentionally also covers pending revisions.
        """
        return _prefix(self.policy.stream, (*self.decisions, *self.updates, *self.drains))


def _state_inventory(state: ReplayState) -> None:
    """Validate bounded inventory and recompute previously selected source views."""
    _inventory_limits(state)
    if state.history is None:
        if state.decisions:
            raise ValueError("Emitted decisions require their complete retained source history")
        return
    _history_identity(state.history, state.policy.source)
    for decision in state.decisions:
        _decision_source(state, decision, state.history)


def _inventory_limits(state: ReplayState) -> None:
    """Count all retained records and supplied coordinates without silent eviction."""
    count = len(state.decisions) + len(state.feedback) + len(state.updates) + len(state.drains)
    if count > MAX_REPLAY_RECORDS:
        raise ValueError("Replay inventory exceeds 4096 retained records")
    values = sum(len(outcome_coordinates(item.batch.feedback)) for item in state.feedback)
    if values > MAX_REPLAY_VALUES:
        raise ValueError("Replay inventory exceeds its bounded supplied-value budget")


def _history_identity(history: ObservationHistory, source: Reference) -> None:
    """Require the original provider/schema identity rather than a future digest."""
    if (history.source_id, history.source_version) != (source.id, source.version):
        raise ValueError("Replay source identity/version differs from the history")


def _decision_source(state: ReplayState, decision: ReplayDecisionRecord, history: ObservationHistory) -> None:
    """Verify one source view and actual decision against its retained contract."""
    from coleman.replay.revisions import validate_opportunity  # pylint: disable=import-outside-toplevel

    first = state.decisions[0].opportunity if state.decisions else None
    validate_opportunity(state.policy, first, decision.opportunity)
    timing = decision.opportunity.request.timing
    if timing is None or history.as_of(timing.information_cutoff).to_json() != decision.snapshot.to_json():
        raise ValueError("Retained source history changes an old eligible decision view")
    validate_decision(decision.executed.proposal, state.policy.capability)


def _state_trace(state: ReplayState) -> None:
    """Reconcile exact causal order, counts, host transitions and lineage boundary."""
    records = sorted((*state.decisions, *state.updates, *state.drains), key=lambda item: item.sequence)
    _record_order(state, records)
    _trace_counts(state, records)
    _trace_frontiers(state)
    _trace_drains(state, records)
    _lineage_records(state, records)


def _record_order(state: ReplayState, records: list[ReplayDecisionRecord | ReplayUpdateRecord | ReplayDrain]) -> None:
    """Require one contiguous causal sequence and canonical per-family order."""
    if [item.sequence for item in records] != list(range(len(records))):
        raise ValueError("Replay causal sequences must be unique, contiguous and zero-based")
    if len({item.logical_identity for item in state.decisions}) != len(state.decisions):
        raise ValueError("Replay retains more than one decision at a logical opportunity")
    for values in (state.decisions, state.updates, state.drains):
        if [item.sequence for item in values] != sorted(item.sequence for item in values):
            raise ValueError("Replay record collections must retain their causal sequence order")


def _lineage_records(
    state: ReplayState, records: list[ReplayDecisionRecord | ReplayUpdateRecord | ReplayDrain]
) -> None:
    """Check newly active records independently from an unchanged parent prefix."""
    parent = state.policy.parent
    boundary = 0 if parent is None else parent.next_sequence
    if any(item.lineage != state.policy.lineage for item in records if item.sequence >= boundary):
        raise ValueError("New causal records must retain the active replay lineage")
    if parent is not None:
        _parent_prefix(state, records, parent)


def _parent_prefix(
    state: ReplayState, records: list[ReplayDecisionRecord | ReplayUpdateRecord | ReplayDrain], parent: ReplayParent
) -> None:
    """Preserve a complete prefix rather than accepting a mixed generation."""
    old = tuple(item for item in records if item.sequence < parent.next_sequence)
    if len(old) != parent.next_sequence or _prefix(state.policy.stream, old) != parent.prefix_fingerprint:
        raise ValueError("Child replay does not retain its exact complete parent prefix")
    if any(item.lineage not in parent.ancestors for item in old):
        raise ValueError("Retained parent prefix contains an undeclared ancestor lineage")
    if old:
        _parent_end(old[-1], parent)
    else:
        host = state.host_fingerprint if not records else records[0].host_before
        if host != parent.host_fingerprint:
            raise ValueError("Child replay origin differs from its exact parent host snapshot")


def _parent_end(record: ReplayDecisionRecord | ReplayUpdateRecord | ReplayDrain, parent: ReplayParent) -> None:
    """Require a closed complete parent host boundary."""
    if record.host_after != parent.host_fingerprint:
        raise ValueError("Child replay parent host binding differs from its retained prefix")
    if isinstance(record, ReplayUpdateRecord):
        raise ValueError("Parent checkpoint cannot end inside an unclosed drain")


@dataclass
class _TraceCursor:
    """Keep local reconstruction facts; never crosses a public model boundary."""

    decision_count: int = 0
    update_count: int = 0
    host: str | None = None
    producers: dict[str, ReplayDecisionRecord] = field(default_factory=dict)
    applied: set[tuple[str, str, str]] = field(default_factory=set)


def _trace_counts(state: ReplayState, records: list[ReplayDecisionRecord | ReplayUpdateRecord | ReplayDrain]) -> None:
    """Reconstruct counters from genuine causal record order without host execution."""
    cursor = _TraceCursor()
    for item in records:
        if isinstance(item, ReplayDrain):
            _cursor_drain(cursor, item)
        else:
            _cursor_host(cursor, item)
            if isinstance(item, ReplayDecisionRecord):
                if state.policy.delivery == "synchronous" and cursor.update_count != cursor.decision_count:
                    raise ValueError("Synchronous replay cannot emit while a previous round remains pending")
                _cursor_decision(cursor, item)
            else:
                _cursor_update(cursor, item)
    if cursor.host is not None and cursor.host != state.host_fingerprint:
        raise ValueError("Replay expected host state differs from the last causal transition")


def _cursor_host(cursor: _TraceCursor, item: ReplayDecisionRecord | ReplayUpdateRecord) -> None:
    """Verify the complete host-state link before advancing local reconstruction."""
    if cursor.host is not None and item.host_before != cursor.host:
        raise ValueError("Replay host state chain is disconnected")
    cursor.host = item.host_after


def _cursor_decision(cursor: _TraceCursor, item: ReplayDecisionRecord) -> None:
    """Advance only the independently reconstructed successful decision count."""
    if item.decision_index != cursor.decision_count or item.update_count_before != cursor.update_count:
        raise ValueError("Decision counts differ from the retained causal prefix")
    cursor.producers[item.executed.fingerprint + _wire(item.lineage)] = item
    cursor.decision_count += 1


def _cursor_update(cursor: _TraceCursor, item: ReplayUpdateRecord) -> None:
    """Match one producing execution and consume its logical feedback only once."""
    application = item.application
    feedback = application.feedback
    producer = cursor.producers.get(application.executed.fingerprint + _wire(feedback.lineage))
    if producer is None or _wire(producer.opportunity.target) != _wire(feedback.target):
        raise ValueError("Update has no exact preceding producing execution/target")
    if (
        application.decision_count_before != cursor.decision_count
        or application.update_count_before != cursor.update_count
    ):
        raise ValueError("Update counts differ from the retained causal prefix")
    if feedback.logical_key in cursor.applied:
        raise ValueError("Logical replay feedback was applied more than once")
    cursor.applied.add(feedback.logical_key)
    cursor.update_count += 1


def _cursor_drain(cursor: _TraceCursor, item: ReplayDrain) -> None:
    """Match a drain's final frontier after its already counted updates."""
    if item.decision_count != cursor.decision_count or item.update_count_after != cursor.update_count:
        raise ValueError("Drain counts differ from the retained causal prefix")
    if cursor.host is not None and item.host_after != cursor.host:
        raise ValueError("Drain final host state differs from the retained causal prefix")
    cursor.host = item.host_after


def _trace_frontiers(state: ReplayState) -> None:
    """Validate explicit closed-drain clocks rather than inferring from receipts."""
    previous: ReplayDrain | None = None
    for item in state.drains:
        if previous is not None:
            _monotone_drains(previous, item)
        previous = item
    if previous is None:
        if state.eligible_through is not None or state.apply_at is not None:
            raise ValueError("Replay frontiers require an explicit retained drain")
    elif state.eligible_through != previous.eligible_through or state.apply_at != previous.apply_at:
        raise ValueError("Replay frontiers differ from the last explicit drain")


def _monotone_drains(previous: ReplayDrain, current: ReplayDrain) -> None:
    """Keep information and processing coordinates independently monotonic."""
    if current.eligible_through < previous.eligible_through or current.apply_at < previous.apply_at:
        raise ValueError("Replay information and processing frontiers must advance independently")


def _trace_drains(state: ReplayState, records: list[ReplayDecisionRecord | ReplayUpdateRecord | ReplayDrain]) -> None:
    """Require explicit closed update blocks and producing decision boundaries."""
    pending: list[ReplayUpdateRecord] = []
    last: ReplayDrain | None = None
    clock: Timestamp | None = None
    for item in records:
        if isinstance(item, ReplayUpdateRecord):
            pending.append(item)
        elif isinstance(item, ReplayDrain):
            _drain_clock(item, clock)
            _closed_drain(state, item, pending)
            pending = []
            last = item
            clock = item.apply_at
        else:
            _decision_boundary(item, last, bool(pending))
            clock = item.accepted_at
    if pending:
        raise ValueError("Replay state cannot acknowledge an unclosed update block")


def _drain_clock(drain: ReplayDrain, clock: Timestamp | None) -> None:
    """Reject processing before a previously emitted decision instant."""
    if clock is not None and drain.apply_at < clock:
        raise ValueError("Drain processing cannot precede an already emitted decision")


def _decision_boundary(record: ReplayDecisionRecord, drain: ReplayDrain | None, has_pending: bool) -> None:
    """Bind decision state to its completed pre-decision information drain."""
    timing = record.opportunity.request.timing
    if has_pending or drain is None or timing is None:
        raise ValueError("Every decision requires its explicit completed pre-decision drain")
    if drain.eligible_through != timing.information_cutoff or drain.apply_at != record.accepted_at:
        raise ValueError("Decision state frontier differs from its information/processing boundary")


def _closed_drain(state: ReplayState, drain: ReplayDrain, updates: list[ReplayUpdateRecord]) -> None:
    """Check one explicit processing cohort against its actual update order."""
    from coleman.replay.revisions import _complete_drain  # pylint: disable=import-outside-toplevel

    _complete_drain(state, drain, updates)
    if drain.update_count_after - drain.update_count_before != len(updates):
        raise ValueError("Drain completed count differs from its actual update cohort")
    if not updates:
        return
    if drain.host_before != updates[0].host_before or drain.host_after != updates[-1].host_after:
        raise ValueError("Drain boundary snapshots differ from its actual updates")
    _update_drain_order(drain, updates)


def _update_drain_order(drain: ReplayDrain, updates: list[ReplayUpdateRecord]) -> None:
    """Order eligibility clocks first, with deterministic stable producer ties."""
    order = []
    for update in updates:
        application = update.application
        if application.apply_at != drain.apply_at or application.eligible_at > drain.eligible_through:
            raise ValueError("Accepted update is outside its explicit information/processing drain")
        order.append(
            (application.eligible_at, application.executed.proposal.request.round, application.feedback.logical_key)
        )
    if order != sorted(order):
        raise ValueError("Drain updates must order eligible time first, then producer and logical key")


class ReplayContinuation[S: CoreContract](CoreContract):
    """Compose one bounded replay frontier and a complete typed host snapshot.

    Attributes
    ----------
    state : ReplayState
        Immutable delivery/lineage frontier, including the expected host digest.
    host_state : S
        Explicit concrete versioned snapshot, including RNG/pending where used.

    Raises
    ------
    ValueError
        The snapshot digest does not match this exact replay frontier.

    Notes
    -----
    Select the concrete type with ReplayContinuation[HostState] or the bounded
    codec's explicit state_type. No dynamic class lookup, pickle, host call,
    persistence acknowledgement or arbitrary recovery guarantee occurs.
    """

    state: ReplayState
    host_state: S

    @model_validator(mode="after")
    def _snapshot(self) -> Self:
        """Require the complete host snapshot matching this metadata frontier."""
        arguments = type(self).__pydantic_generic_metadata__["args"]
        if not arguments or arguments[0] is CoreContract:
            raise ValueError("Continuation requires an explicit concrete host snapshot type")
        if self.host_state.fingerprint != self.state.host_fingerprint:
            raise ValueError("Continuation host snapshot differs from the expected replay state")
        return self
