"""Bounded feedback ancestry and source-visible selection without host calls."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from coleman.core.capabilities import Capability, ProblemCapabilities, validate_composition, validate_feedback
from coleman.core.decisions import ExecutedDecision
from coleman.core.feedback import (
    AbsentFeedback,
    ComponentFeedback,
    FeedbackBatch,
    FeedbackKind,
    OutcomeValue,
)
from coleman.core.identity import Reference
from coleman.core.primitives import canonical_json
from coleman.core.temporal import Timestamp

if TYPE_CHECKING:
    from coleman.replay.contracts import (
        ReplayApplication,
        ReplayDecisionRecord,
        ReplayDrain,
        ReplayFeedback,
        ReplayOpportunity,
        ReplayPolicy,
        ReplayState,
        ReplayUpdateRecord,
        ValueOriginEvidence,
    )

FeedbackValueReason = Literal["absent", "partial_components", "missing_required_value", "no_observed_value"]


def feedback_key(value: ReplayFeedback) -> tuple[str, str, str]:
    """Return one producing-lineage/execution/target logical round key.

    Parameters
    ----------
    value : ReplayFeedback
        Typed immutable revision; feedback ID is not a second round namespace.

    Returns
    -------
    tuple of str
        Exact producing lineage, executed fingerprint and target reference.

    Notes
    -----
    Performs no source access, update, acceptance or registry lookup.
    """
    return value.logical_key


def feedback_chains(values: tuple[ReplayFeedback, ...]) -> tuple[tuple[ReplayFeedback, ...], ...]:
    """Deduplicate exact retries and validate complete bounded revision graphs.

    Parameters
    ----------
    values : tuple of ReplayFeedback
        At most 4096 attempted revisions including duplicates. A newly supplied
        child requires its parent in this tuple, usually retained plus incoming.

    Returns
    -------
    tuple of tuple of ReplayFeedback
        Chains sorted by exact logical round key, each ordered root to successor.
        Equal clocks are ordered by ancestry, never lexical revision strings.

    Raises
    ------
    ValueError, TypeError
        Invalid count/content, conflicting bytes, aliases, missing parent, fork,
        cycle or changed logical/schema/origin/model semantics.

    Notes
    -----
    No event is applied. First receipts may be out of parent order; source
    correction availability cannot precede its direct predecessor.
    """
    # Runtime import avoids a contract/helper initialization cycle.
    from coleman.replay.contracts import MAX_REPLAY_EVENTS, ReplayFeedback  # pylint: disable=import-outside-toplevel

    if not isinstance(values, tuple) or len(values) > MAX_REPLAY_EVENTS:
        raise ValueError("Feedback graphs require at most 4096 attempted typed revisions")
    groups: dict[tuple[str, str, str], dict[str, ReplayFeedback]] = defaultdict(dict)
    for raw in values:
        if not isinstance(raw, ReplayFeedback):
            raise TypeError("Feedback graph checking requires typed ReplayFeedback revisions")
        checked = ReplayFeedback.model_validate(raw)
        revisions = groups[feedback_key(checked)]
        previous = revisions.setdefault(checked.batch.revision_id, checked)
        if previous.to_json() != checked.to_json():
            raise ValueError("Conflicting feedback content or first receipt at one revision identity")
    return tuple(_chain(tuple(groups[key].values())) for key in sorted(groups))


def _chain(values: tuple[ReplayFeedback, ...]) -> tuple[ReplayFeedback, ...]:
    """Require one complete unambiguous chain and stable logical identity."""
    roots = [value for value in values if value.batch.supersedes is None]
    if len(roots) != 1:
        raise ValueError("Feedback lineage requires exactly one root")
    root = roots[0]
    children = _feedback_edges(values, root)
    chain = [root]
    while chain[-1].batch.revision_id in children:
        chain.append(children[chain[-1].batch.revision_id])
    if len(chain) != len(values):
        raise ValueError("Feedback lineage is disconnected or cyclic")
    _shared_schemas(tuple(chain))
    return tuple(chain)


def _feedback_edges(values: tuple[ReplayFeedback, ...], root: ReplayFeedback) -> dict[str, ReplayFeedback]:
    """Validate direct predecessor ownership before following explicit ancestry."""
    by_id = {value.batch.revision_id: value for value in values}
    children: dict[str, ReplayFeedback] = {}
    for value in values:
        _stable_round(root, value)
        predecessor = value.batch.supersedes
        if predecessor is None:
            continue
        parent = by_id.get(predecessor)
        if parent is None or predecessor in children:
            raise ValueError("Feedback lineage has a missing parent or ambiguous fork")
        _stable_edge(parent, value)
        children[predecessor] = value
    return children


def _stable_round(root: ReplayFeedback, value: ReplayFeedback) -> None:
    """Reject aliases or changed target/coverage meaning within one round."""
    if value.batch.feedback_id != root.batch.feedback_id:
        raise ValueError("Different feedback ID aliases cannot create the same logical target round")
    if value.target.model_dump_json() != root.target.model_dump_json():
        raise ValueError("Feedback revision changed its exact target, horizon or dependence groups")
    if value.batch.feedback.kind != root.batch.feedback.kind:
        raise ValueError("Feedback revisions cannot change their declared coverage kind")
    _source_event(root, value)


def _source_event(root: ReplayFeedback, value: ReplayFeedback) -> None:
    """Keep the original event interpretation while corrections arrive later."""
    root_timing, timing = root.batch.timing, value.batch.timing
    if root_timing is None or timing is None or timing.event_at != root_timing.event_at:
        raise ValueError("Feedback revisions must retain the original source event clock")


def _stable_edge(parent: ReplayFeedback, child: ReplayFeedback) -> None:
    """Retain source chronology and shared-coordinate provenance across revisions."""
    parent_timing, child_timing = parent.batch.timing, child.batch.timing
    if parent_timing is None or child_timing is None or child_timing.available_at < parent_timing.available_at:
        raise ValueError("Feedback correction availability cannot precede its parent")


def _stable_origin(parent: ValueOriginEvidence, child: ValueOriginEvidence) -> None:
    """Reject origin/model replacement at an already declared coordinate."""
    old = None if parent.producer is None else parent.producer.model_dump_json()
    new = None if child.producer is None else child.producer.model_dump_json()
    if child.origin != parent.origin or new != old:
        raise ValueError("A feedback correction cannot change a coordinate's origin or model binding")


def _origin_key(value: ValueOriginEvidence) -> tuple[str, str]:
    """Compare exact member/dimension versions without interpreting display names."""
    return ("" if value.member is None else value.member.model_dump_json(), value.dimension.model_dump_json())


def _shared_schemas(chain: tuple[ReplayFeedback, ...]) -> None:
    """Require stable declared dimension schemas and shared-coordinate transforms."""
    from coleman.replay.contracts import outcome_coordinates  # pylint: disable=import-outside-toplevel

    dimensions: dict[str, str] = {}
    transforms: dict[tuple[str, str], str | None] = {}
    origins: dict[tuple[str, str], ValueOriginEvidence] = {}
    for revision in chain:
        _keep_origins(origins, revision.origins)
        for member, value in outcome_coordinates(revision.batch.feedback):
            _keep_dimension(dimensions, value)
            _keep_transform(transforms, member, value)


def _keep_origins(
    retained: dict[tuple[str, str], ValueOriginEvidence], incoming: tuple[ValueOriginEvidence, ...]
) -> None:
    """Keep coordinate provenance even across partial revisions omitting a row."""
    for value in incoming:
        previous = retained.setdefault(_origin_key(value), value)
        _stable_origin(previous, value)


def _keep_dimension(dimensions: dict[str, str], value: OutcomeValue) -> None:
    """Bind each dimension ID to its exact schema/version throughout a chain."""
    schema = value.spec.model_dump_json()
    previous = dimensions.setdefault(value.spec.dimension.id, schema)
    if schema != previous:
        raise ValueError("Feedback revisions/rows changed a shared dimension's schema or version")


def _keep_transform(
    transforms: dict[tuple[str, str], str | None], member: Reference | None, value: OutcomeValue
) -> None:
    """Keep an explicitly declared coordinate transform without executing it."""
    key = ("" if member is None else member.model_dump_json(), value.spec.dimension.model_dump_json())
    transform = None if value.transform is None else value.transform.model_dump_json()
    old = transforms.setdefault(key, transform)
    if old != transform:
        raise ValueError("Feedback corrections cannot change a coordinate's transform identity")


def validate_feedback_graph(
    values: tuple[ReplayFeedback, ...],
    decisions: tuple[ReplayDecisionRecord, ...],
    capability: Capability,
) -> tuple[ReplayFeedback, ...]:
    """Check complete revision graphs against exact actual producing decisions.

    Parameters
    ----------
    values : tuple of ReplayFeedback
        Retained plus incoming bounded revisions, including exact retries.
    decisions : tuple of ReplayDecisionRecord
        Already emitted actual simulated decisions; foreign producers reject.
    capability : Capability
        Exact host's coverage, origin, quantity and selection-evidence requirements.

    Returns
    -------
    tuple of ReplayFeedback
        Unique revisions sorted by logical round key then revision ID. Chain
        ancestry remains explicit even when lexical IDs run in reverse order.

    Raises
    ------
    ValueError, TypeError
        Invalid graph, foreign execution/lineage/target or incompatible coverage,
        quantities, origin, versions, transforms or selection evidence.

    Notes
    -----
    Admission relaxes observed-value, receipt and delay requirements ONLY on a
    temporary core validation profile; nulls/partial/unknown clocks stay pending.
    It never changes retained bytes or authorizes an actual update. Application
    must use the original full capability and its actual apply_at.
    """
    checked_capability = Capability.model_validate(capability)
    producers = {(value.lineage.model_dump_json(), value.executed.fingerprint): value for value in decisions}
    chains = feedback_chains(values)
    unique = []
    for chain in chains:
        for value in chain:
            producer = producers.get((value.lineage.model_dump_json(), value.batch.decision_fingerprint))
            if producer is None or producer.opportunity.target.model_dump_json() != value.target.model_dump_json():
                raise ValueError("Feedback requires an exact registered producing lineage, execution and target")
            _admission(producer.executed, value.batch, checked_capability)
            unique.append(value)
    return tuple(sorted(unique, key=lambda value: (*feedback_key(value), value.batch.revision_id)))


def _admission(executed: ExecutedDecision, batch: FeedbackBatch, capability: Capability) -> None:
    """Reuse core shape/schema validation without turning pending facts into data."""
    feedback_kinds = capability.feedback_kinds
    if isinstance(batch.feedback, AbsentFeedback) and "absent" not in feedback_kinds:
        feedback_kinds = (*feedback_kinds, "absent")
    relaxed = Capability.model_validate(
        capability.model_copy(
            update={
                "requires_observed": False,
                "requires_known_receipt": False,
                "supports_delayed_feedback": True,
                "feedback_kinds": feedback_kinds,
            }
        )
    )
    provisional = FeedbackBatch.model_validate(batch.model_copy(update={"delay_rounds": 0}))
    timing = batch.timing
    decision_timing = executed.proposal.request.timing
    if timing is None or decision_timing is None:
        raise ValueError("Causal replay admission requires declared decision and feedback clocks")
    clocks = [timing.event_at, timing.available_at, timing.mature_at, decision_timing.decision_at]
    if timing.received_at is not None:
        clocks.append(timing.received_at)
    validate_feedback(executed, provisional, relaxed, apply_at=max(clocks))


def latest_visible(chain: tuple[ReplayFeedback, ...], eligible_through: Timestamp) -> ReplayFeedback | None:
    """Select the latest source-visible revision before checking update readiness.

    Parameters
    ----------
    chain : tuple of ReplayFeedback
        Complete root-to-successor validated chain; raw tuple order is not used.
    eligible_through : datetime
        Explicit inclusive UTC-compatible information cutoff, not processing time.

    Returns
    -------
    ReplayFeedback or None
        Latest received/source-available revision, even when its values are null,
        component panel partial or maturity later. Unknown receipt is not visible.

    Raises
    ------
    ValueError
        Invalid cutoff/graph or more than one logical chain.

    Notes
    -----
    Never falls back to an obsolete observed root because a visible correction
    is unready. No host call, clock collection or acceptance occurs.
    """
    cutoff = TypeAdapter(Timestamp).validate_python(eligible_through)
    chains = feedback_chains(chain)
    if len(chains) > 1:
        raise ValueError("Latest-visible selection requires one logical feedback chain")
    for value in reversed(chains[0] if chains else ()):
        if _source_visible(value, cutoff):
            return value
    return None


def _source_visible(value: ReplayFeedback, cutoff: Timestamp) -> bool:
    """Require independent actual receipt before exposing a source correction."""
    timing = value.batch.timing
    return (
        timing is not None and timing.received_at is not None and max(timing.available_at, timing.received_at) <= cutoff
    )


def feedback_eligible_at(value: ReplayFeedback, executed: ExecutedDecision) -> Timestamp | None:
    """Return known clock readiness without asserting values or panel completeness.

    Parameters
    ----------
    value : ReplayFeedback
        Selected source-visible revision, still possibly numerically unavailable.
    executed : ExecutedDecision
        Exact registered producing action with decision and target clocks.

    Returns
    -------
    datetime or None
        Maximum producing decision, target completion and feedback clocks;
        explicit null for unknown receipt. This alone never authorizes an update.

    Raises
    ------
    ValueError
        Mismatched executed binding or missing required temporal envelopes.
    """
    if value.batch.decision_fingerprint != executed.fingerprint:
        raise ValueError("Eligibility requires the exact producing executed decision")
    timing, decision = value.batch.timing, executed.proposal.request.timing
    if timing is None or decision is None:
        raise ValueError("Eligibility requires declared decision and feedback clocks")
    usable = timing.usable_at
    return None if usable is None else max(decision.decision_at, decision.target_end, usable)


def _state_applications(state: ReplayState) -> None:
    """Require retained accepted bytes and full original capability validation."""
    retained = {(*item.logical_key, item.batch.revision_id): item.to_json() for item in state.feedback}
    for update in state.updates:
        application = update.application
        value = application.feedback
        if retained.get((*value.logical_key, value.batch.revision_id)) != value.to_json():
            raise ValueError("Accepted application does not match an exact retained feedback revision")
        if state.policy.delivery == "synchronous" and value.batch.delay_rounds:
            raise ValueError("Synchronous replay cannot retain an asynchronously delayed update")
        validate_feedback(application.executed, value.batch, state.policy.capability, apply_at=application.apply_at)


def validate_opportunity(policy: ReplayPolicy, first: ReplayOpportunity | None, opportunity: ReplayOpportunity) -> None:
    """Share exact stable-request and capability checks across runtime and reader.

    Parameters
    ----------
    policy : ReplayPolicy
        Exact stream, active-universe and delivery declarations.
    first : ReplayOpportunity or None
        First emitted opportunity, or null before an independent first decision.
    opportunity : ReplayOpportunity
        Validated proposed or retained producing request and optional context.

    Raises
    ------
    ValueError
        Changed task/objective/context, undeclared changing active universe,
        incompatible action/coverage/timing or missing/nonobserved required input.

    Notes
    -----
    Supplied scales may vary by producing decision, while their declared semantic
    definitions remain fixed. All advertised feedback kinds are considered; no
    arbitrary first-kind choice rejects a supported mixed-coverage host. No host
    calls, source selection, update, migration or I/O occurs.
    """
    request = opportunity.request
    if request.task != policy.stream:
        raise ValueError("Replay request task must match its exact declared stream")
    if first is not None:
        _opportunity_identity(policy, first, opportunity)
    _opportunity_inputs(policy, opportunity)


def _opportunity_identity(policy: ReplayPolicy, first: ReplayOpportunity, value: ReplayOpportunity) -> None:
    """Preserve objective, declared universe and full semantic definitions."""
    if value.request.task != first.request.task or value.request.objective != first.request.objective:
        raise ValueError("Replay task/objective changes require an explicit new compatible lineage")
    if not policy.specialists and value.request.space != first.request.space:
        raise ValueError("Changing active members requires an explicit specialist replay declaration")
    old = None if first.context is None else first.context.model_dump(mode="json", exclude={"scale_selections"})
    new = None if value.context is None else value.context.model_dump(mode="json", exclude={"scale_selections"})
    if canonical_json(old) != canonical_json(new):
        raise ValueError("Replay semantic-definition changes require an explicit restart/migration profile")


def _opportunity_inputs(policy: ReplayPolicy, value: ReplayOpportunity) -> None:
    """Negotiate actual request shape and all declared feedback alternatives."""
    compatibility = tuple(
        validate_composition(_request_problem(policy, value, kind), policy.capability)
        for kind in policy.capability.feedback_kinds
    )
    if not any(result.compatible for result in compatibility):
        reasons = tuple(result.reasons for result in compatibility)
        raise ValueError(f"Replay request is structurally incompatible: {reasons}")
    required = {spec.dimension.model_dump_json() for spec in policy.capability.required_inputs}
    if policy.capability.requires_observed and any(
        item.status != "observed" for item in value.request.inputs if item.spec.dimension.model_dump_json() in required
    ):
        raise ValueError("Replay required decision inputs must be observed, never imputed")


def _request_problem(policy: ReplayPolicy, value: ReplayOpportunity, kind: FeedbackKind) -> ProblemCapabilities:
    """Describe one advertised coverage without inventing observed outcomes."""
    timing = value.request.timing
    if timing is None:
        raise ValueError("Causal replay requires explicit decision timing")
    capability = policy.capability
    return ProblemCapabilities(
        schema_version=1,
        space=value.request.space,
        feedback_kind=kind,
        values=capability.required_values,
        inputs=tuple(item.spec for item in value.request.inputs),
        origins=capability.origins,
        timing_mode=timing.mode,
        delayed_feedback=policy.delivery == "delayed",
        sampling_statuses=("exact",) if capability.requires_sampling else ("not_required",),
        receipt_status="known",
        state_owners=(capability.state_owner,),
    )


def feedback_value_reason(value: ReplayFeedback, capability: Capability) -> FeedbackValueReason | None:
    """Share explicit pending-value states between live preflight and readers.

    Parameters
    ----------
    value : ReplayFeedback
        Latest source-visible revision with its actual supplied coordinates.
    capability : Capability
        Required schemas and independently declared observed-value requirement.

    Returns
    -------
    str or None
        Absent, partial, missing-required or entirely unobserved reason; null
        means value readiness only, not temporal or actual execution admission.

    Notes
    -----
    Required observed zero and permitted negatives remain valid. Unused null
    diagnostics do not block required coordinates. No values are manufactured,
    dropped from the payload, applied, transformed or delivered to a host.
    """
    payload = value.batch.feedback
    if isinstance(payload, AbsentFeedback):
        return "absent"
    if isinstance(payload, ComponentFeedback) and payload.completeness == "partial":
        return "partial_components"
    return _coordinate_reason(value, capability)


def _coordinate_reason(value: ReplayFeedback, capability: Capability) -> FeedbackValueReason | None:
    """Inspect only admitted coordinate states, preserving optional diagnostics."""
    from coleman.replay.contracts import outcome_coordinates  # pylint: disable=import-outside-toplevel

    values = tuple(item for _, item in outcome_coordinates(value.batch.feedback))
    wanted = {spec.dimension.model_dump_json() for spec in capability.required_values}
    if capability.requires_observed and any(
        item.status != "observed" for item in values if item.spec.dimension.model_dump_json() in wanted
    ):
        return "missing_required_value"
    return None if any(item.status == "observed" for item in values) else "no_observed_value"


def _complete_drain(state: ReplayState, drain: ReplayDrain, updates: list[ReplayUpdateRecord]) -> None:
    """Reconstruct every unconsumed ready round, including omitted-update attacks."""
    expected = _expected_drain(state, drain, drain.sequence - len(updates))
    actual = [
        (
            item.application.eligible_at,
            item.application.executed.proposal.request.round,
            item.application.feedback.logical_key,
            item.application.feedback.to_json(),
        )
        for item in updates
    ]
    if actual != expected:
        raise ValueError("Closed drain omits or changes a complete eligible feedback update")


def _prior_consumed(state: ReplayState, boundary: int) -> set[tuple[str, str, str]]:
    """Retain logical consumption before this drain, excluding its own updates."""
    return {feedback_key(item.application.feedback) for item in state.updates if item.sequence < boundary}


def _drain_producers(state: ReplayState, drain: ReplayDrain) -> dict[tuple[str, str], ReplayDecisionRecord]:
    """Exclude producers registered only after an earlier same-instant drain."""
    return {
        (item.lineage.model_dump_json(), item.executed.fingerprint): item
        for item in state.decisions
        if item.sequence < drain.sequence
    }


def _expected_drain(
    state: ReplayState, drain: ReplayDrain, boundary: int
) -> list[tuple[Timestamp, int, tuple[str, str, str], str]]:
    """Reconstruct one bounded closed cohort using the same pure readiness facts."""
    consumed = _prior_consumed(state, boundary)
    producers = _drain_producers(state, drain)
    expected = []
    for chain in feedback_chains(state.feedback):
        if feedback_key(chain[0]) in consumed:
            continue
        producer = producers.get((chain[0].lineage.model_dump_json(), chain[0].batch.decision_fingerprint))
        if producer is not None:
            candidate = _drain_candidate(chain, producer, state.policy.capability, drain)
            if candidate is not None:
                expected.append(candidate)
    return sorted(expected)


def _drain_candidate(
    chain: tuple[ReplayFeedback, ...], producer: ReplayDecisionRecord, capability: Capability, drain: ReplayDrain
) -> tuple[Timestamp, int, tuple[str, str, str], str] | None:
    """Select source visibility before separately requiring clock/value readiness."""
    selected = latest_visible(chain, drain.eligible_through)
    if selected is None:
        return None
    eligible = feedback_eligible_at(selected, producer.executed)
    if eligible is None or eligible > drain.eligible_through or feedback_value_reason(selected, capability) is not None:
        return None
    expected_delay = drain.decision_count - producer.decision_index - 1
    if selected.batch.delay_rounds != expected_delay:
        raise ValueError("Closed drain selected feedback with inconsistent logical decision delay")
    validate_feedback(producer.executed, selected.batch, capability, apply_at=drain.apply_at)
    return eligible, producer.decision_index, selected.logical_key, selected.to_json()


def _checked_origins(value: ReplayFeedback) -> tuple[ValueOriginEvidence, ...]:
    """Reconcile exact coordinate sets before checking each producer/value binding."""
    from coleman.replay.contracts import outcome_coordinates  # pylint: disable=import-outside-toplevel

    expected = {
        ("" if member is None else member.model_dump_json(), item.spec.dimension.model_dump_json()): item
        for member, item in outcome_coordinates(value.batch.feedback)
    }
    actual = {_origin_key(item): item for item in value.origins}
    if len(actual) != len(value.origins) or set(actual) != set(expected):
        raise ValueError("Origin evidence must cover exactly the supplied unique value coordinates")
    for key, item in expected.items():
        _origin_value(actual[key], item)
    return tuple(actual[key] for key in sorted(actual))


def _origin_value(evidence: ValueOriginEvidence, value: OutcomeValue) -> None:
    """Bind origin metadata to the exact supplied scalar content."""
    from coleman.replay.contracts import outcome_fingerprint  # pylint: disable=import-outside-toplevel

    if evidence.origin != value.origin or evidence.value_fingerprint != outcome_fingerprint(value):
        raise ValueError("Origin evidence differs from its actual value content")


def _application_clocks(value: ReplayApplication) -> None:
    """Require exact producing execution and all independently known clocks."""
    if value.feedback.batch.decision_fingerprint != value.executed.fingerprint:
        raise ValueError("Application feedback differs from the exact executed decision")
    timing = value.executed.proposal.request.timing
    feedback_timing = value.feedback.batch.timing
    if timing is None or feedback_timing is None or feedback_timing.usable_at is None:
        raise ValueError("Application requires producing clocks and known first receipt")
    expected = max(timing.decision_at, timing.target_end, feedback_timing.usable_at)
    if value.eligible_at != expected or value.apply_at < value.eligible_at:
        raise ValueError("Application eligibility or actual processing clock is inconsistent")


def _application_counts(value: ReplayApplication) -> None:
    """Check decision-delay units without treating a receipt as a completed round."""
    delay = value.decision_count_before - value.executed.proposal.request.round - 1
    if delay < 0 or value.feedback.batch.delay_rounds != delay:
        raise ValueError("Application declared logical delay differs from successful decision count")
    if value.update_count_before >= value.decision_count_before:
        raise ValueError("Completed updates cannot exceed producing decision opportunities")
