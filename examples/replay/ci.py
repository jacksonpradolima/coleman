"""A host-owned observed-loss baseline using actual causal replay contracts.

The input is authored synthetic CI evidence. This deterministic, bounded example
is not a shipped learner or evidence of algorithmic efficacy. Its execution is
explicitly offline identity simulation, without running a physical CI action.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field, model_validator

from coleman.core.actions import ChoiceAction, ChoiceSpace, SubsetAction, SubsetSpace
from coleman.core.base import CoreContract
from coleman.core.capabilities import Capability, validate_decision, validate_feedback
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import ChosenFeedback, ComponentFeedback, FeedbackBatch, FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64, ScalarSpec
from coleman.core.primitives import Fingerprint, canonical_json, fingerprint
from coleman.core.sampling import SamplingRecord
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.core.units import UnitSemantics
from coleman.objectives.arithmetic import checked_fraction
from coleman.objectives.bindings import DefinitionBinding
from coleman.observations.contracts import ObservationHistory
from coleman.replay.contracts import (
    ReplayApplication,
    ReplayDecisionRecord,
    ReplayFeedback,
    ReplayOpportunity,
    ReplayPolicy,
    ReplayTarget,
    ValueOriginEvidence,
)
from coleman.replay.engine import CausalReplay

Profile = Literal["full_panel", "chosen", "component"]
Count = Annotated[int, Field(strict=True, ge=0, le=16)]
NOW = datetime(2026, 1, 1, 10, tzinfo=UTC)


def reference(name: str) -> Reference:
    """Return an exact public v1 identity without registry access.

    Parameters
    ----------
    name : str
        Valid bounded core identifier; no source or registry is consulted.

    Returns
    -------
    Reference
        Exact identity and version ``v1``.

    Raises
    ------
    ValueError
        The identifier violates the core contract.
    """
    return Reference(id=name, version="v1")


MEMBERS = tuple(reference("suite-" + name) for name in ("a", "b", "c"))
IMPLEMENTATION = reference("authored-ci-observed-mean")
CODEC = reference("authored-ci-host-state")
HORIZON = reference("authored-ci-target-horizon")
SOURCE = reference("authored-ci-source")
GENERATOR = DefinitionBinding(
    reference=reference("authored-ci-generator"), definition_fingerprint=fingerprint({"fixture": 1})
)
LOSS = ScalarSpec(
    dimension=reference("ci-loss"),
    semantics=UnitSemantics(unit="1", meaning="synthetic-ci-bounded-loss-v1"),
    role="loss",
    minimum=0,
    maximum=1,
)


def space(profile: Profile) -> ChoiceSpace | SubsetSpace:
    """Declare three fixed suites, choosing one or an unordered pair.

    Parameters
    ----------
    profile : {'full_panel', 'chosen', 'component'}
        Component feedback uses a subset of exactly two suites; other profiles
        select a single suite. No outcomes are included.

    Returns
    -------
    ChoiceSpace or SubsetSpace
        Immutable exact-version universe, with no I/O or state changes.
    """
    if profile == "component":
        return SubsetSpace(
            feasible_set=reference("authored-ci-subset"), members=MEMBERS, minimum_selected=2, maximum_selected=2
        )
    return ChoiceSpace(feasible_set=reference("authored-ci-choice"), members=MEMBERS)


def capability(profile: Profile) -> Capability:
    """Declare the example's units, actual coverage and delayed input support.

    Parameters
    ----------
    profile : {'full_panel', 'chosen', 'component'}
        Exact supported feedback shape and corresponding action kind.

    Returns
    -------
    Capability
        Pure declaration requiring observed dimensionless bounded loss, known
        receipts and explicit timing; it executes no update.

    Notes
    -----
    No synchronous-learning theorem is inferred. Origins remain separate even
    though this arithmetic example can consume any explicitly supplied origin.
    """
    return Capability(
        schema_version=1,
        implementation=IMPLEMENTATION,
        action_kinds=("subset",) if profile == "component" else ("choice",),
        feedback_kinds=(profile,),
        required_values=(LOSS,),
        origins=("realized", "modeled_counterfactual", "synthetic"),
        timing_modes=("future", "ongoing"),
        supports_delayed_feedback=True,
        requires_known_receipt=True,
        requires_observed=True,
        state_owner="host",
        requires_sampling=False,
        sampling_requirement="not_required",
    )


def _rows(batch: FeedbackBatch) -> tuple[FeedbackRow, ...]:
    feedback = batch.feedback
    if isinstance(feedback, FullPanel):
        return feedback.rows
    if isinstance(feedback, ChosenFeedback):
        return (feedback.row,)
    if isinstance(feedback, ComponentFeedback) and feedback.completeness == "complete":
        return feedback.rows
    raise ValueError("Host consumes only complete admitted full/chosen/selected-component rounds")


def _statistics(
    applications: tuple[ReplayApplication, ...],
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    counts, totals = [0, 0, 0], [Fraction(0), Fraction(0), Fraction(0)]
    for application in applications:
        for row in _rows(application.feedback.batch):
            index = MEMBERS.index(row.member)
            value = next(value for value in row.values if value.spec.dimension == LOSS.dimension)
            if value.value is None:
                raise ValueError("Unknown loss cannot become zero")
            counts[index] += 1
            totals[index] += Fraction.from_float(value.value)
    return (counts[0], counts[1], counts[2]), (
        checked_fraction(totals[0]),
        checked_fraction(totals[1]),
        checked_fraction(totals[2]),
    )


class HostConsumerState(CoreContract):
    """Retain the complete deterministic example state as versioned values.

    Attributes
    ----------
    profile : {'full_panel', 'chosen', 'component'}
        Fixed coverage/action interpretation; not changed by restoration.
    capability_fingerprint : str
        Exact implementation and capability binding, not authentication.
    executions, applications : tuple
        All bounded emitted simulations and accepted actual update envelopes.
        Future source history, hidden oracle rows and pending values are absent.
    decision_count, update_count : int
        Successful opportunities and completed accepted panels, independently.
    observed_counts, loss_sums : tuple
        Present admitted loss counts and dimensionless sums per MEMBERS order.
        These counts imply neither realized observations nor independence.
    pending : tuple of str
        Exact emitted execution digests lacking an accepted outcome round.

    Notes
    -----
    One round per execution and at most 16 opportunities. All summaries are
    recomputed on ingress. There is no RNG, callback, executable state or I/O.
    """

    profile: Profile
    capability_fingerprint: Fingerprint
    executions: Annotated[tuple[ExecutedDecision, ...], Field(max_length=16)] = ()
    applications: Annotated[tuple[ReplayApplication, ...], Field(max_length=16)] = ()
    decision_count: Count = 0
    update_count: Count = 0
    observed_counts: tuple[Count, Count, Count] = (0, 0, 0)
    loss_sums: tuple[Binary64, Binary64, Binary64] = (0.0, 0.0, 0.0)
    pending: Annotated[tuple[Fingerprint, ...], Field(max_length=16)] = ()

    @model_validator(mode="after")
    def _facts(self):
        """Validate complete state in ingress order without applying any update."""
        declared = capability(self.profile)
        if self.capability_fingerprint != declared.fingerprint:
            raise ValueError("Host snapshot capability/profile mismatch")
        if self.decision_count != len(self.executions) or self.update_count != len(self.applications):
            raise ValueError("Host decision/update counts contradict retained facts")
        by_fingerprint = self._execution_facts(declared)
        used = self._application_facts(declared, by_fingerprint)
        self._summary_facts(used)
        return self

    def _execution_facts(self, declared: Capability) -> dict[Fingerprint, ExecutedDecision]:
        """Check unique sequential simulations before inspecting applications.

        Parameters
        ----------
        declared : Capability
            Already selected fixed host profile, including its exact loss schema.

        Returns
        -------
        dict
            Local execution-digest lookup, never retained or exposed as state.

        Raises
        ------
        ValueError
            Duplicate opportunity, invalid core decision or changed round,
            universe, action or explicit offline mapping. No state is changed.
        """
        identities = [execution.proposal.logical_identity for execution in self.executions]
        by_fingerprint = {execution.fingerprint: execution for execution in self.executions}
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate host opportunity")
        for index, execution in enumerate(self.executions):
            validate_decision(execution.proposal, declared)
            if (
                execution.proposal.request.round != index
                or execution.proposal.request.space != space(self.profile)
                or execution.mapping != "unchanged"
                or execution.action.fingerprint != execution.proposal.action.fingerprint
            ):
                raise ValueError("Host snapshot requires its exact sequential offline simulations")
        return by_fingerprint

    def _application_facts(
        self, declared: Capability, by_fingerprint: dict[Fingerprint, ExecutedDecision]
    ) -> set[Fingerprint]:
        """Validate each original execution's single accepted complete round.

        Parameters
        ----------
        declared : Capability
            Fixed coverage, loss units and timing requirements.
        by_fingerprint : dict
            Checked exact executions from this snapshot, keyed by their digests.

        Returns
        -------
        set
            Consumed execution digests for the pending-summary check.

        Raises
        ------
        ValueError
            Unknown/repeated execution, invalid feedback or inconsistent D/U.
            Validation performs no learner call, state assignment or I/O.
        """
        used = set()
        for index, application in enumerate(self.applications):
            execution = by_fingerprint.get(application.executed.fingerprint)
            if execution is None or execution.fingerprint in used:
                raise ValueError("Host application is unknown or repeated")
            validate_feedback(execution, application.feedback.batch, declared, apply_at=application.apply_at)
            if application.update_count_before != index or not (
                execution.proposal.request.round < application.decision_count_before <= self.decision_count
            ):
                raise ValueError("Host accepted application counters contradict retained history")
            used.add(execution.fingerprint)
        return used

    def _summary_facts(self, used: set[Fingerprint]) -> None:
        """Recompute pending, counts and dimensionless loss sums without mutation.

        Parameters
        ----------
        used : set
            Checked consumed execution digests; never modified by this method.

        Raises
        ------
        ValueError
            Pending order, observed counts or loss sums disagree with retained
            applications. Canonical float comparison also preserves signed-zero
            distinctions; unknown losses are never filled with zero.
        """
        expected_pending = tuple(
            execution.fingerprint for execution in self.executions if execution.fingerprint not in used
        )
        counts, totals = _statistics(self.applications)
        if (
            self.pending != expected_pending
            or self.observed_counts != counts
            or canonical_json(list(self.loss_sums)) != canonical_json(list(totals))
        ):
            raise ValueError("Host pending/statistical summaries contradict retained applications")


def _state(
    profile: Profile,
    executions: tuple[ExecutedDecision, ...] = (),
    applications: tuple[ReplayApplication, ...] = (),
) -> HostConsumerState:
    counts, totals = _statistics(applications)
    consumed = {application.executed.fingerprint for application in applications}
    return HostConsumerState(
        schema_version=1,
        profile=profile,
        capability_fingerprint=capability(profile).fingerprint,
        executions=executions,
        applications=applications,
        decision_count=len(executions),
        update_count=len(applications),
        observed_counts=counts,
        loss_sums=totals,
        pending=tuple(execution.fingerprint for execution in executions if execution.fingerprint not in consumed),
    )


class HostConsumer:
    """Choose from actually observed losses using one complete owned state.

    Parameters
    ----------
    profile : {'full_panel', 'chosen', 'component'}, optional
        Fixed declared coverage, default chosen. Unobserved suites are explored
        lexically; otherwise lowest observed mean wins, with lexical ties.

    Notes
    -----
    Multiple pending decisions are supported. Direct calls mutate only this
    instance; replay owns delivery idempotency, clocks and revision acceptance.
    This is copied host code, not a shipped algorithm or efficacy claim.
    """

    def __init__(self, profile: Profile = "chosen"):
        """Create empty deterministic state for the declared coverage profile."""
        self._state = _state(profile)

    def decide(self, request: DecisionRequest, /) -> Decision:
        """Emit one explicitly simulated proposal, retaining its full pending state.

        Parameters
        ----------
        request : DecisionRequest
            Exact timed opportunity; round equals D, not completed updates.

        Returns
        -------
        Decision
            Deterministic proposal. Offline identity execution is the declared
            matching example/replay policy, not a physical action receipt.

        Raises
        ------
        ValueError
            Wrong round/space/profile or capacity; rejected before assignment.
        """
        request = DecisionRequest.model_validate(request)
        state = self._state
        if state.decision_count >= 16 or request.round != state.decision_count or request.space != space(state.profile):
            raise ValueError("Host requires its fixed space, next decision round and bounded capacity")
        order = sorted(
            range(3),
            key=lambda i: (
                0 if state.observed_counts[i] == 0 else 1,
                0.0 if state.observed_counts[i] == 0 else state.loss_sums[i] / state.observed_counts[i],
                MEMBERS[i].id,
            ),
        )
        action: ChoiceAction | SubsetAction
        if isinstance(request.space, SubsetSpace):
            action = SubsetAction(schema_version=1, space=request.space, selected=tuple(MEMBERS[i] for i in order[:2]))
        elif isinstance(request.space, ChoiceSpace):
            action = ChoiceAction(schema_version=1, space=request.space, chosen=MEMBERS[order[0]])
        else:
            raise ValueError("Host supports only its declared choice or subset space")
        sampling = SamplingRecord(
            mechanism="deterministic",
            mechanism_ref=IMPLEMENTATION,
            status="not_required",
            action_fingerprint=action.fingerprint,
            probability=None,
        )
        proposal = Decision(
            schema_version=1,
            request=request,
            implementation=IMPLEMENTATION,
            action=action,
            sampling=sampling,
            state_owner="host",
        )
        validate_decision(proposal, capability(state.profile))
        executed = ExecutedDecision(
            schema_version=1,
            proposal=proposal,
            action=action,
            sampling=sampling,
            mapping="unchanged",
            sampling_disposition="preserved",
        )
        self._state = _state(state.profile, (*state.executions, executed), state.applications)
        return proposal

    def update(self, application: ReplayApplication, /) -> None:
        """Apply one admitted panel at its explicit actual processing instant.

        Parameters
        ----------
        application : ReplayApplication
            Retained exact execution, outer provenance, batch and real apply_at.
            Latest complete successor may be the first accepted revision.

        Raises
        ------
        ValueError
            Unknown/already consumed execution, wrong counters or invalid
            coverage/units/timing. No partial state assignment occurs.
        """
        application = ReplayApplication.model_validate(application)
        state = self._state
        if application.executed.fingerprint not in state.pending:
            raise ValueError("Host execution is unknown or already consumed")
        if (application.decision_count_before, application.update_count_before) != (
            state.decision_count,
            state.update_count,
        ):
            raise ValueError("Host application decision/update counters mismatch")
        validate_feedback(
            application.executed, application.feedback.batch, capability(state.profile), apply_at=application.apply_at
        )
        _rows(application.feedback.batch)
        self._state = _state(state.profile, state.executions, (*state.applications, application))

    def export_state(self) -> HostConsumerState:
        """Return revalidated complete immutable state without a clock or I/O.

        Returns
        -------
        HostConsumerState
            Full bounded execution/application history and checked summaries.

        Raises
        ------
        ValueError
            Retained state violates its declared invariants.
        """
        return HostConsumerState.model_validate(self._state)

    def restore_state(self, state: HostConsumerState, /) -> None:
        """Validate complete same-profile state before replacing this instance.

        Parameters
        ----------
        state : HostConsumerState
            Complete versioned snapshot. All facts are revalidated before the
            single state replacement; no migration or merge is inferred.

        Raises
        ------
        ValueError
            Unsupported version, invalid summaries or another profile.

        Notes
        -----
        The session additionally checks the separately retained host fingerprint;
        a content digest is consistency evidence, not authentication.
        """
        checked = HostConsumerState.model_validate(state)
        if checked.profile != self._state.profile:
            raise ValueError("Host cannot silently change coverage profile")
        self._state = checked


def instant(minute: float) -> datetime:
    """Return an explicit UTC authored timeline instant, in minute offsets.

    Parameters
    ----------
    minute : float
        Offset from 2026-01-01 10:00 UTC. Fractional and negative offsets are
        useful for receipt ordering and ongoing targets; no clock is read.

    Returns
    -------
    datetime
        Zoned UTC instant at microsecond precision.

    Raises
    ------
    ValueError, OverflowError
        The offset is non-finite or outside datetime/timedelta bounds.
    """
    return NOW + timedelta(minutes=minute)


def history() -> ObservationHistory:
    """Return an empty authored source; no hidden oracle enters the consumer.

    Returns
    -------
    ObservationHistory
        Versioned empty CI input history. No source is queried and no file or
        host state is changed; outcome payloads are admitted separately.
    """
    return ObservationHistory(schema_version=1, source_id=SOURCE.id, source_version=SOURCE.version, revisions=())


def policy(profile: Profile = "chosen", *, lineage: Reference | None = None) -> ReplayPolicy:
    """Declare explicit delayed offline CI replay over one independently owned host.

    Parameters
    ----------
    profile : {'full_panel', 'chosen', 'component'}, optional
        Fixed capability profile, default chosen.
    lineage : Reference or None, optional
        Independent origin identity; None uses the authored live identity.
        Matched child lineage construction belongs to replay's fork operation.

    Returns
    -------
    ReplayPolicy
        Preserve-live, closed-drain declaration with explicit source identity.
        Does not create a host, access storage or modify state.
    """
    return ReplayPolicy(
        schema_version=1,
        stream=reference("authored-ci-stream"),
        lineage=lineage or reference("authored-ci-live"),
        parent=None,
        source=SOURCE,
        capability=capability(profile),
        delivery="delayed",
        specialists=False,
    )


def opportunity(
    index: int,
    *,
    profile: Profile = "chosen",
    cutoff: float | None = None,
    at: float | None = None,
    end: float | None = None,
    source: ObservationHistory | None = None,
) -> ReplayOpportunity:
    """Build a timed CI opportunity from the actual eligible source snapshot.

    Parameters
    ----------
    index : int
        Zero-based decision round, independently of accepted update count.
    profile : {'full_panel', 'chosen', 'component'}, optional
        Exact fixed action/coverage profile, default chosen.
    cutoff, at, end : float or None, optional
        Information, decision and target-end minute offsets. Decision defaults
        to index, cutoff to decision and end to decision; all clocks stay explicit.
    source : ObservationHistory or None, optional
        Actual admitted history used for its inclusive cutoff snapshot digest;
        None selects the empty authored history. It is never passed to the host.

    Returns
    -------
    ReplayOpportunity
        Timed future target and typed request bound to the eligible view.

    Raises
    ------
    ValueError
        Invalid clocks, schema, identifier or round. No host call or I/O occurs.
    """
    at = float(index) if at is None else at
    cutoff = at if cutoff is None else cutoff
    source = history() if source is None else source
    target = ReplayTarget(
        reference=reference(f"build-{index}-target"), horizon=HORIZON, groups=(reference("authored-ci-family"),)
    )
    request = DecisionRequest(
        schema_version=1,
        task=reference("authored-ci-stream"),
        decision_id=f"build-{index}",
        round=index,
        space=space(profile),
        objective=reference("minimize-authored-ci-loss"),
        timing=DecisionTiming(
            decision_at=instant(at),
            information_cutoff=instant(cutoff),
            target_start=instant(at),
            target_end=instant(at if end is None else end),
            eligible_input_fingerprint=source.as_of(instant(cutoff)).fingerprint,
            mode="future",
        ),
    )
    return ReplayOpportunity(schema_version=1, request=request, context=None, target=target)


def feedback(
    record: ReplayDecisionRecord,
    losses: tuple[float | None, float | None, float | None],
    *,
    profile: Profile = "chosen",
    visible: float = 2,
    mature: float | None = None,
    event: float | None = None,
    delay: int = 1,
    revision: str = "root",
    supersedes: str | None = None,
    completeness: Literal["complete", "partial"] = "complete",
) -> ReplayFeedback:
    """Supply only selected authored outcomes, with complete per-value provenance.

    Parameters
    ----------
    record : ReplayDecisionRecord
        Actual registered producer and explicitly simulated execution.
    losses : tuple of float or None
        Driver-owned losses for a/b/c, each observed in [0,1] or unavailable.
        None remains missing, never zero. Unselected coordinates are discarded.
    profile : {'full_panel', 'chosen', 'component'}, optional
        Declared coverage; default chosen.
    visible, mature, event : float or None, optional
        Minute offsets for source availability/known receipt, maturity and event.
        Maturity defaults to visibility; event defaults to producing target end.
    delay : int, optional
        Exact intervening decision count when accepted, default one.
    revision, supersedes : str or None, optional
        Explicit chain identity and predecessor, never an independent sample.
    completeness : {'complete', 'partial'}, optional
        Component completeness; partial retains only its first selected row.

    Returns
    -------
    ReplayFeedback
        Typed synthetic values with exact generator/version/digest per coordinate.

    Raises
    ------
    ValueError
        Invalid loss, clocks, profile/coverage or producer. No update is applied.

    Notes
    -----
    The tuple belongs to the driver. Unselected coordinates never enter the
    returned object, host request, state, callback or filesystem channel.
    """
    action = record.executed.action
    if profile == "full_panel":
        selected = MEMBERS
    elif isinstance(action, ChoiceAction):
        selected = (action.chosen,)
    elif isinstance(action, SubsetAction):
        selected = action.selected
    else:
        raise ValueError("Example feedback requires an actual choice or subset")
    if completeness == "partial":
        selected = selected[:1]
    rows = tuple(
        FeedbackRow(
            member=member,
            values=(
                OutcomeValue(
                    spec=LOSS,
                    value=losses[MEMBERS.index(member)],
                    status="unavailable" if losses[MEMBERS.index(member)] is None else "observed",
                    origin="synthetic",
                    transform=reference("authored-ci-loss-identity"),
                ),
            ),
        )
        for member in selected
    )
    content: FullPanel | ChosenFeedback | ComponentFeedback
    if profile == "full_panel":
        content = FullPanel(rows=rows)
    elif profile == "chosen":
        content = ChosenFeedback(row=rows[0])
    else:
        content = ComponentFeedback(rows=rows, completeness=completeness)
    origins = tuple(
        ValueOriginEvidence(
            member=row.member,
            dimension=value.spec.dimension,
            value_fingerprint=fingerprint(value.model_dump(mode="json")),
            origin=value.origin,
            producer=GENERATOR,
        )
        for row in rows
        for value in row.values
    )
    timing = record.opportunity.request.timing
    if timing is None:
        raise ValueError("Example feedback requires a timed producing decision")
    actual_event = timing.target_end if event is None else instant(event)
    batch = FeedbackBatch(
        schema_version=1,
        decision_fingerprint=record.executed.fingerprint,
        feedback_id="ci-outcome",
        revision_id=revision,
        supersedes=supersedes,
        horizon=HORIZON,
        feedback=content,
        timing=FeedbackTiming(
            event_at=actual_event,
            available_at=instant(visible),
            mature_at=instant(visible if mature is None else mature),
            received_at=instant(visible),
        ),
        delay_rounds=delay,
    )
    return ReplayFeedback(
        schema_version=1, lineage=record.lineage, batch=batch, target=record.opportunity.target, origins=origins
    )


def run() -> dict[str, object]:
    """Execute real chosen-only delayed replay and return its small CI receipt.

    Returns
    -------
    dict[str, object]
        Human display receipt with three choices, independent D/U counts and
        dimensionless observed loss summaries. Durable state remains typed.

    Notes
    -----
    Allocates one private host/session and mutates only their in-memory state.
    Does not run tests physically, read a clock, write files or qualify efficacy.
    """
    host = HostConsumer()
    replay = CausalReplay(host, policy(), state_type=HostConsumerState, state_codec=CODEC)
    replay.admit_history(history())
    first = replay.emit(replay.prepare(opportunity(0, at=0, end=2)), execution="offline_identity_v1")
    second = replay.emit(replay.prepare(opportunity(1, at=1, end=2)), execution="offline_identity_v1")
    replay.admit_feedback((feedback(first, (1.0, 0.0, 0.5)),))
    third = replay.emit(replay.prepare(opportunity(2, cutoff=2, at=3, end=4)), execution="offline_identity_v1")
    state = host.export_state()
    choices = []
    for record in (first, second, third):
        if not isinstance(record.executed.action, ChoiceAction):
            raise ValueError("Chosen example produced another action shape")
        choices.append(record.executed.action.chosen.id)
    return {
        "evidence": "authored_synthetic_ci",
        "execution": "offline_identity_v1",
        "choices": choices,
        "decisions": state.decision_count,
        "accepted_rounds": state.update_count,
        "counts": state.observed_counts,
        "loss_sums": state.loss_sums,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
