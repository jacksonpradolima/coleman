"""Synthetic CI consumers with complete bounded state and explicitly owned RNG.

This copied host is a recovery fixture, not a shipped learner or evidence of
efficacy. Actions are offline identity simulations. Predictive scores are
synthetic priorities, not probabilities; the preloaded reservoir is explicit.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any, Self

from pydantic import Field, model_validator

from coleman.core.actions import ChoiceAction, ChoiceSpace
from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.capabilities import Capability, validate_decision, validate_feedback
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import ChosenFeedback, FeedbackBatch, FeedbackRow, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64, ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.sampling import SamplingRecord
from coleman.core.temporal import Timestamp
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
from coleman.state.random_state import PythonRandomState, capture_random, restore_random

if TYPE_CHECKING:
    from coleman.replay.retained import RetainedReplay
    from coleman.replay.retained_contracts import RetainedContinuation
    from coleman.state.components import ComponentBinding, ComponentValue
    from coleman.state.storage_contracts import CollectionDefinition, ExecutionCompatibility

Count = Annotated[int, Field(strict=True, ge=0)]
ChildIndex = Annotated[int, Field(strict=True, ge=0, le=31)]
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def reference(name: str) -> Reference:
    """Return a public version-one identity without registry access.

    Parameters
    ----------
    name : str
        Explicit logical identifier.

    Returns
    -------
    Reference
        Exact identity/version pair; no I/O or state change.
    """
    return Reference(id=name, version="v1")


PLANS = reference("ci-plan-a"), reference("ci-plan-b")
IMPLEMENTATION = reference("authored-owned-ci-consumer")
CODEC = reference("authored-owned-ci-state")
EPOCH = reference("authored-ci-source-epoch")
SOURCE = reference("authored-ci-source")
HORIZON = reference("authored-ci-horizon")
LOSS = ScalarSpec(
    dimension=reference("ci-loss"),
    semantics=UnitSemantics(unit="1", meaning="authored-synthetic-ci-loss-v1"),
    role="loss",
    minimum=0,
    maximum=1,
)
GENERATOR = DefinitionBinding(
    reference=reference("authored-ci-provided-synthetic-loss"),
    definition_fingerprint=fingerprint(
        {"process": "explicit-host-provided-synthetic-value", "unit": "1", "minimum": 0, "maximum": 1, "version": 1}
    ),
)


def child_reference(child: int) -> Reference:
    """Return the fixed logical consumer identity after strict index validation.

    Parameters
    ----------
    child : int
        Integer 0 through 31; booleans and coerced inputs reject.

    Returns
    -------
    Reference
        Stable recipe identity, independent of enumeration order.

    Raises
    ------
    ValueError
        Invalid child index. No state is changed.
    """
    if type(child) is not int or not 0 <= child < 32:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Child index must be an integer from 0 through 31")
    return reference(f"recipe-{child:02d}")


def instant(ordinal: int) -> datetime:
    """Map an authored second offset to an explicit UTC clock.

    Parameters
    ----------
    ordinal : int
        Second offset from 2026-01-01 UTC; no ambient clock is read.

    Returns
    -------
    datetime
        Zoned authored instant.

    Raises
    ------
    OverflowError
        The offset exceeds datetime bounds.
    """
    return NOW + timedelta(seconds=ordinal)


def space() -> ChoiceSpace:
    """Return the two-plan CI universe without state mutation.

    Returns
    -------
    ChoiceSpace
        Exact public member and feasible-set references; no outcomes.
    """
    return ChoiceSpace(feasible_set=reference("authored-ci-two-plan-space"), members=PLANS)


def capability() -> Capability:
    """Declare delayed chosen-only loss input with explicit known receipt clocks.

    Returns
    -------
    Capability
        Typed dimensionless loss requirement and host-owned state. Sampling
        probabilities are not claimed for the explicitly preloaded stream.
    """
    return Capability(
        schema_version=1,
        implementation=IMPLEMENTATION,
        action_kinds=("choice",),
        feedback_kinds=("chosen",),
        required_values=(LOSS,),
        origins=("synthetic",),
        timing_modes=("future", "ongoing"),
        supports_delayed_feedback=True,
        requires_known_receipt=True,
        requires_observed=True,
        state_owner="host",
        requires_sampling=False,
        sampling_requirement="not_required",
    )


class PathBook(CoreImmutable):
    """Keep sufficient bounded state for a synthetic applied CI path.

    Attributes
    ----------
    releases, count_b, switches : int
        Action releases, B releases and adjacent plan changes, respectively.
    spent_seconds : int
        Exact charge ``3*releases + 2*count_b + 2*switches`` in seconds.
        It is not a fee estimate or dimensionless movement penalty.
    last_plan : Reference or None
        Required prior-plan state for the next switching charge.
    recent : tuple of Reference
        At most eight applied plans; an audit ring, never the full path.

    Notes
    -----
    No physical CI job is submitted. Historical sums are qualified state, not
    authenticated reconstruction of discarded actions. No I/O is performed.
    """

    releases: Count = 0
    count_b: Count = 0
    switches: Count = 0
    spent_seconds: Count = 0
    last_plan: Reference | None = None
    recent: Annotated[tuple[Reference, ...], Field(max_length=8)] = ()

    @model_validator(mode="after")
    def _facts(self) -> Self:
        """Reject inconsistent sufficient counters without reconstructing history."""
        if self.count_b > self.releases or self.switches > max(0, self.releases - 1):
            raise ValueError("Path counters exceed actual releases")
        if self.spent_seconds != 3 * self.releases + 2 * self.count_b + 2 * self.switches:
            raise ValueError("Path seconds contradict the declared resource formula")
        if len(self.recent) != min(8, self.releases) or any(plan not in PLANS for plan in self.recent):
            raise ValueError("Path audit ring has an invalid size or plan")
        expected_last = self.recent[-1] if self.recent else None
        if self.last_plan != expected_last:
            raise ValueError("Path last plan differs from its audit ring")
        return self

    def release(self, plan: Reference) -> PathBook:
        """Return one new offline path release without mutating this book.

        Parameters
        ----------
        plan : Reference
            Exact plan A or B, already selected for offline identity execution.

        Returns
        -------
        PathBook
            Updated resource counters, last plan and ring.

        Raises
        ------
        ValueError
            Unsupported plan; no state is changed.

        Notes
        -----
        Raw calls charge again. Operational retry deduplication belongs to the
        explicitly selected store/replay layer, not this mathematical operation.
        """
        if plan not in PLANS:
            raise ValueError("Path release requires an exact authored CI plan")
        switches = self.switches + int(self.last_plan is not None and plan != self.last_plan)
        count_b = self.count_b + int(plan == PLANS[1])
        releases = self.releases + 1
        return PathBook(
            releases=releases,
            count_b=count_b,
            switches=switches,
            spent_seconds=3 * releases + 2 * count_b + 2 * switches,
            last_plan=plan,
            recent=(*self.recent, plan)[-8:],
        )


class PredictionCache(CoreImmutable):
    """Bind one synthetic priority to its actual input, clock and model version.

    Attributes
    ----------
    feature : float
        Dimensionless known feature; no outcome enters the key.
    cutoff : datetime
        Explicit eligible-input instant, never inferred from a receipt.
    model_updates : int
        Completed observed-loss rounds defining this affine model generation.
    value : float
        Dimensionless priority ``2*feature + model_updates``, not a probability.
    """

    feature: Binary64
    cutoff: Timestamp
    model_updates: Count
    value: Binary64

    @model_validator(mode="after")
    def _value(self) -> Self:
        """Check affine arithmetic with finite explicitly projected values."""
        expected = checked_fraction(2 * Fraction.from_float(self.feature) + self.model_updates)
        if self.value != expected:
            raise ValueError("Cache priority contradicts its input/model binding")
        return self


class PendingChoice(CoreImmutable):
    """Retain one exact executed proposal required by a delayed chosen update.

    Attributes
    ----------
    executed : ExecutedDecision
        Complete offline unchanged execution, including actual request clocks.
    """

    executed: ExecutedDecision


class HostState(CoreContract):
    """Own complete bounded algorithm, predictor, RNG and virtual-path state.

    Attributes
    ----------
    child : int
        Exact fixed consumer configuration, 0 through 31.
    decision_count, update_count : int
        Logical emitted opportunities and complete chosen outcomes separately.
    cadence_count : int
        Completed two-release periods; receipts do not advance this counter.
    observed_counts, loss_sums : tuple
        Two observed sample counts and dimensionless bounded-loss sums.
    rng : PythonRandomState
        Complete owned MT19937 state, including explicit absent Gaussian cache.
    reservoir_word, reservoir_remaining, backend_refills : int
        Unconsumed low bits, remaining bit count and actual 32-bit refill count.
        The initial declared reservoir is word4/remaining3, producing A,A,B.
    pending : tuple of PendingChoice
        At most 128 exact executions awaiting outcomes; required by this host.
    book : PathBook
        This consumer's isolated virtual resource path.
    cache : PredictionCache or None
        One known-input entry; refreshed when an update changes model identity.
    prediction_hits, prediction_misses : int
        Explicit synthetic inference accounting. Only a cache miss costs one
        declared inference unit; these are not measured CPU costs.

    Notes
    -----
    No full growing execution/application/path history is retained. Counts and
    sufficient statistics are validated for internal consistency, not proven
    from discarded observations. A compatibility-bound trusted host owns these
    facts. No serialization, I/O or RNG draw occurs during model validation.
    """

    child: ChildIndex
    decision_count: Count
    update_count: Count
    cadence_count: Count
    observed_counts: tuple[Count, Count]
    loss_sums: tuple[Binary64, Binary64]
    rng: PythonRandomState
    reservoir_word: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
    reservoir_remaining: Annotated[int, Field(strict=True, ge=0, le=32)]
    backend_refills: Count
    pending: Annotated[tuple[PendingChoice, ...], Field(max_length=128)]
    book: PathBook
    cache: PredictionCache | None
    prediction_hits: Count
    prediction_misses: Count

    @model_validator(mode="after")
    def _facts(self) -> Self:
        """Verify bounded state facts without replaying or inferring missing data."""
        if (
            self.update_count != sum(self.observed_counts)
            or self.update_count + len(self.pending) != self.decision_count
        ):
            raise ValueError("Host D/U/pending/observed counts disagree")
        if any(not 0 <= value <= count for value, count in zip(self.loss_sums, self.observed_counts, strict=True)):
            raise ValueError("Host loss sums violate bounded observed sample counts")
        self._random_facts()
        self._pending_facts()
        if self.book.releases != self.decision_count:
            raise ValueError("Virtual book releases differ from emitted host opportunities")
        if self.cadence_count != self.decision_count // 2:
            raise ValueError("Host cadence differs from its two-release period")
        if self.cache is not None and self.cache.model_updates != self.update_count:
            raise ValueError("Cache uses another model generation")
        return self

    def _random_facts(self) -> None:
        """Check reservoir accounting while retaining the actual backend words."""
        extra = max(0, self.decision_count - 3)
        refills = (extra + 31) // 32
        remaining = 3 - self.decision_count if self.decision_count <= 3 else (-extra) % 32
        if (self.backend_refills, self.reservoir_remaining) != (refills, remaining):
            raise ValueError("Host reservoir/refill counts differ from released decisions")
        if self.reservoir_word >= 2**self.reservoir_remaining or self.rng.gauss_next is not None:
            raise ValueError("Host has invalid unused reservoir bits or an unused Gaussian cache")

    def _pending_facts(self) -> None:
        """Validate required exact executions without a second accepted-ID owner."""
        ordinals = []
        for pending in self.pending:
            execution = pending.executed
            validate_decision(execution.proposal, capability())
            ordinal = execution.proposal.request.round
            if (
                execution.proposal.request.task != child_reference(self.child)
                or ordinal >= self.decision_count
                or execution.mapping != "unchanged"
                or execution.action != execution.proposal.action
            ):
                raise ValueError("Pending execution differs from its fixed child/offline profile")
            ordinals.append(ordinal)
        if ordinals != sorted(set(ordinals)):
            raise ValueError("Pending execution ordinals must be unique and canonical")


def _initial(child: int) -> HostState:
    """Create one private complete empty host snapshot with an explicit seed."""
    child_reference(child)
    return HostState(
        schema_version=1,
        child=child,
        decision_count=0,
        update_count=0,
        cadence_count=0,
        observed_counts=(0, 0),
        loss_sums=(0.0, 0.0),
        rng=capture_random(random.Random(100 + child)),
        reservoir_word=4,
        reservoir_remaining=3,
        backend_refills=0,
        pending=(),
        book=PathBook(),
        cache=None,
        prediction_hits=0,
        prediction_misses=0,
    )


def _predicted(state: HostState, cutoff: datetime) -> tuple[PredictionCache, int, int]:
    """Calculate a pure current-model cache hit or one synthetic inference miss."""
    feature = float(state.child + 1)
    entry = PredictionCache(
        feature=feature,
        cutoff=cutoff,
        model_updates=state.update_count,
        value=checked_fraction(Fraction(2 * (state.child + 1) + state.update_count)),
    )
    hit = state.cache == entry
    return entry, state.prediction_hits + int(hit), state.prediction_misses + int(not hit)


class HostConsumer:
    """Implement real direct core protocols with bounded privately owned state.

    Parameters
    ----------
    child : int, optional
        Explicit consumer index0..31, default0. Distinct owners never share a
        mutable random instance. Seeds are configuration, not observations.

    Notes
    -----
    Choosing consumes one buffered bit and records one offline virtual release;
    updating consumes one supplied chosen loss and refreshes its model cache.
    Raw repeated updates reject after the execution is consumed. Replay/store
    supplies its separate retained-retry and durable acknowledgement contract.
    """

    def __init__(self, child: int = 0):
        """Create an independent initial state without reading ambient entropy."""
        self._state = _initial(child)

    def prewarm(self, cutoff: datetime) -> float:
        """Evaluate the declared known input at an explicit cutoff.

        Parameters
        ----------
        cutoff : datetime
            Known zoned input instant. No outcome or ambient clock is read.

        Returns
        -------
        float
            Dimensionless affine synthetic priority.

        Raises
        ------
        ValueError
            Invalid clock or finite arithmetic, before state assignment.

        Notes
        -----
        Changes only this owner's one-entry cache and hit/miss accounting.
        """
        entry, hits, misses = _predicted(self._state, cutoff)
        self._state = HostState.model_validate(
            self._state.model_copy(update={"cache": entry, "prediction_hits": hits, "prediction_misses": misses})
        )
        return entry.value

    def decide(self, request: DecisionRequest, /) -> Decision:
        """Consume one owned bit and emit an offline identity CI proposal.

        Parameters
        ----------
        request : DecisionRequest
            Fixed child/universe, next global round and explicit known timing.

        Returns
        -------
        Decision
            One actual proposal; no physical CI job or probability claim.

        Raises
        ------
        ValueError
            Invalid request, full pending capacity or arithmetic. Candidate
            state and RNG are local until the final validated replacement.
        """
        request = DecisionRequest.model_validate(request)
        state = self._state
        if (
            request.round != state.decision_count
            or request.task != child_reference(state.child)
            or request.space != space()
            or request.timing is None
            or len(state.pending) >= 128
        ):
            raise ValueError("Host requires its next timed child request and bounded pending capacity")
        generator = random.Random(0)
        restore_random(generator, state.rng)
        word, remaining, refills = state.reservoir_word, state.reservoir_remaining, state.backend_refills
        if not remaining:
            # The restored owned stream only selects simulated CI plans.
            word, remaining, refills = (
                generator.getrandbits(32),  # NOSONAR(S2245) deterministic offline choices, never security tokens.
                32,
                refills + 1,
            )
        action = ChoiceAction(schema_version=1, space=space(), chosen=PLANS[word & 1])
        sampling = SamplingRecord(
            mechanism="sampled",
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
        validate_decision(proposal, capability())
        executed = ExecutedDecision(
            schema_version=1,
            proposal=proposal,
            action=action,
            sampling=sampling,
            mapping="unchanged",
            sampling_disposition="preserved",
        )
        entry, hits, misses = _predicted(state, request.timing.information_cutoff)
        self._state = HostState.model_validate(
            state.model_copy(
                update={
                    "decision_count": state.decision_count + 1,
                    "cadence_count": (state.decision_count + 1) // 2,
                    "rng": capture_random(generator),
                    "reservoir_word": word >> 1,
                    "reservoir_remaining": remaining - 1,
                    "backend_refills": refills,
                    "pending": (*state.pending, PendingChoice(executed=executed)),
                    "book": state.book.release(action.chosen),
                    "cache": entry,
                    "prediction_hits": hits,
                    "prediction_misses": misses,
                }
            )
        )
        return proposal

    def update(self, application: ReplayApplication, /) -> None:
        """Consume one explicitly supplied synthetic chosen loss.

        Parameters
        ----------
        application : ReplayApplication
            Actual execution, complete chosen batch and explicit processing
            clock/counters supplied by the replay boundary.

        Raises
        ------
        ValueError
            Unknown/consumed execution, counter mismatch, incompatible loss,
            clocks or coverage. No partial state/RNG replacement occurs.

        Notes
        -----
        Mutates statistics, pending and cache only; no action, RNG draw, cadence
        release or physical resource charge is caused by an outcome receipt.
        """
        application = ReplayApplication.model_validate(application)
        state = self._state
        execution = application.executed
        pending = next((entry for entry in state.pending if entry.executed.fingerprint == execution.fingerprint), None)
        if pending is None:
            raise ValueError("Host execution is unknown or already consumed")
        if (application.decision_count_before, application.update_count_before) != (
            state.decision_count,
            state.update_count,
        ):
            raise ValueError("Host update counters differ from the actual owned frontier")
        validate_feedback(execution, application.feedback.batch, capability(), apply_at=application.apply_at)
        coverage = application.feedback.batch.feedback
        if not isinstance(coverage, ChosenFeedback) or not isinstance(execution.action, ChoiceAction):
            raise ValueError("Host requires an actual chosen-only action/outcome")
        outcome = next(value for value in coverage.row.values if value.spec.dimension == LOSS.dimension)
        if outcome.value is None:
            raise ValueError("Missing loss cannot become zero")
        member = PLANS.index(execution.action.chosen)
        counts, totals = list(state.observed_counts), list(state.loss_sums)
        counts[member] += 1
        totals[member] = checked_fraction(Fraction.from_float(totals[member]) + Fraction.from_float(outcome.value))
        candidate = HostState.model_validate(
            state.model_copy(
                update={
                    "update_count": state.update_count + 1,
                    "observed_counts": tuple(counts),
                    "loss_sums": tuple(totals),
                    "pending": tuple(entry for entry in state.pending if entry != pending),
                    "cache": None,
                }
            )
        )
        entry, hits, misses = _predicted(candidate, application.apply_at)
        self._state = HostState.model_validate(
            candidate.model_copy(update={"cache": entry, "prediction_hits": hits, "prediction_misses": misses})
        )

    def export_state(self) -> HostState:
        """Return the fully validated bounded state without drawing or I/O.

        Returns
        -------
        HostState
            Concrete complete snapshot, including its own RNG and reservoir.

        Raises
        ------
        ValueError
            An internal state invariant has been violated.
        """
        return HostState.model_validate(self._state)

    def restore_state(self, state: HostState, /) -> None:
        """Validate a compatible complete snapshot before one state replacement.

        Parameters
        ----------
        state : HostState
            Exact host codec and child configuration, with no guessed migration.

        Raises
        ------
        TypeError
            Snapshot has another concrete class.
        ValueError
            Invalid state or a different child configuration.

        Notes
        -----
        Changes this instance only; does not restore another component, decode
        executable state or acknowledge durable application progress.
        """
        if type(state) is not HostState:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Host restore requires its exact concrete state codec")
        validated = HostState.model_validate(state)
        if validated.child != self._state.child:
            raise ValueError("Host restore cannot change its child configuration")
        self._state = validated


class DomainState(CoreContract):
    """Keep the bounded actual offline applied path separately from virtual books.

    Attributes
    ----------
    book : PathBook
        One actual interleaved path, not the sum of per-child virtual charges.

    Notes
    -----
    No physical effect or external reconciliation is claimed.
    """

    book: PathBook = Field(default_factory=PathBook)


class Domain:
    """Own one offline applied-path book, independently of every child state."""

    def __init__(self):
        """Create an empty actual path without external side effects."""
        self._state = DomainState(schema_version=1)

    def apply(self, plan: Reference) -> None:
        """Append one actual offline release to this domain's sufficient state.

        Parameters
        ----------
        plan : Reference
            Explicit selected plan A or B. Raw retries charge again.

        Raises
        ------
        ValueError
            Unknown plan; state remains unchanged.
        """
        self._state = DomainState(schema_version=1, book=self._state.book.release(plan))

    def export_state(self) -> DomainState:
        """Return checked bounded actual-domain state without I/O.

        Returns
        -------
        DomainState
            Current resource counters, prior plan and audit ring.
        """
        return DomainState.model_validate(self._state)

    def restore_state(self, state: DomainState, /) -> None:
        """Replace only this domain after exact complete-state validation.

        Parameters
        ----------
        state : DomainState
            Version-one actual-domain snapshot.

        Raises
        ------
        TypeError, ValueError
            Wrong concrete class or invalid resource/path state.
        """
        if type(state) is not DomainState:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Domain restore requires its exact concrete state codec")
        self._state = DomainState.model_validate(state)


def history() -> ObservationHistory:
    """Return a finite empty source, with no future outcome channel.

    Returns
    -------
    ObservationHistory
        Exact source identity and empty typed revision tuple, without I/O.
    """
    return ObservationHistory(schema_version=1, source_id=SOURCE.id, source_version=SOURCE.version, revisions=())


def policy(child: int) -> ReplayPolicy:
    """Declare the consumer's exact delayed offline replay semantics.

    Parameters
    ----------
    child : int
        Fixed recipe index0..31.

    Returns
    -------
    ReplayPolicy
        Child-specific stream/lineage, shared public input source and capability.
    """
    identity = child_reference(child)
    return ReplayPolicy(
        schema_version=1,
        stream=identity,
        lineage=reference(identity.id + "-live"),
        parent=None,
        source=SOURCE,
        capability=capability(),
        delivery="delayed",
        specialists=False,
    )


def opportunity(child: int, ordinal: int, *, at: int | None = None) -> ReplayOpportunity:
    """Author a retained-profile request with exact global ordinal and clocks.

    Parameters
    ----------
    child : int
        Fixed recipe index.
    ordinal : int
        Global next decision count; never reset when a tail is sealed.
    at : int or None, optional
        Explicit clock offset independent of the logical counter. None uses
        ordinal seconds. Delayed receipts may require a later actual clock.

    Returns
    -------
    ReplayOpportunity
        Exact ordinal ID, known-input fingerprint and zero-length future target.

    Raises
    ------
    ValueError
        Invalid child, ordinal or core timing. Performs no host mutation.
    """
    from coleman.replay.retained import ordinal_decision_id  # pylint: disable=import-outside-toplevel

    clock = instant(ordinal if at is None else at)
    return ReplayOpportunity(
        schema_version=1,
        request=DecisionRequest(
            schema_version=1,
            task=child_reference(child),
            decision_id=ordinal_decision_id(EPOCH, ordinal),
            round=ordinal,
            space=space(),
            objective=reference("minimize-authored-ci-loss"),
            timing=DecisionTiming(
                mode="future",
                information_cutoff=clock,
                decision_at=clock,
                target_start=clock,
                target_end=clock,
                eligible_input_fingerprint=history().as_of(clock).fingerprint,
            ),
        ),
        context=None,
        target=ReplayTarget(reference=reference(f"ci-target-{ordinal}"), horizon=HORIZON, groups=()),
    )


def feedback(record: ReplayDecisionRecord, *, decisions: int, available: int, value: float) -> ReplayFeedback:
    """Author one observed synthetic chosen outcome with explicit logical delay.

    Parameters
    ----------
    record : ReplayDecisionRecord
        Actual emitted producer, including exact executed member and target.
    decisions : int
        Actual D before application; delay is D-producer_index-1.
    available : int
        Explicit event/availability/maturity/receipt second offset.
    value : float
        Observed dimensionless loss in[0,1], including zero.

    Returns
    -------
    ReplayFeedback
        Typed observed chosen loss with per-value synthetic origin.

    Raises
    ------
    ValueError
        Invalid action, loss, delay or clocks. No host update or hidden alternate
        outcome is generated.
    """
    if not isinstance(record.executed.action, ChoiceAction):
        raise ValueError("Synthetic fixture expects a real choice execution")
    outcome = OutcomeValue(
        spec=LOSS, value=value, status="observed", origin="synthetic", transform=reference("ci-loss-id")
    )
    batch = FeedbackBatch(
        schema_version=1,
        decision_fingerprint=record.executed.fingerprint,
        feedback_id=f"feedback-{record.decision_index}",
        revision_id="root",
        supersedes=None,
        horizon=HORIZON,
        feedback=ChosenFeedback(row=FeedbackRow(member=record.executed.action.chosen, values=(outcome,))),
        timing=FeedbackTiming(
            event_at=instant(available),
            available_at=instant(available),
            mature_at=instant(available),
            received_at=instant(available),
        ),
        delay_rounds=decisions - record.decision_index - 1,
    )
    return ReplayFeedback(
        schema_version=1,
        lineage=record.lineage,
        batch=batch,
        target=record.opportunity.target,
        origins=(
            ValueOriginEvidence(
                member=record.executed.action.chosen,
                dimension=LOSS.dimension,
                value_fingerprint=fingerprint(outcome.model_dump(mode="json")),
                origin="synthetic",
                producer=GENERATOR,
            ),
        ),
    )


def make_replay(child: int, *, maximum_tail_records: int = 512) -> tuple[RetainedReplay[HostState], HostConsumer]:
    """Create a real retained replay around one independent complete consumer.

    Parameters
    ----------
    child : int
        Fixed recipe index0..31.
    maximum_tail_records : int, optional
        Explicit bounded tail budget, default512; no automatic sealing occurs.

    Returns
    -------
    tuple
        Actual retained replay with its finite source admitted, and its owned
        host. No store, runtime scan, filesystem or future learner is required.

    Raises
    ------
    ValueError
        Invalid identity, bound or source/capability contract.
    """
    from coleman.replay.retained import RetainedReplay  # pylint: disable=import-outside-toplevel
    from coleman.replay.retained_contracts import OrdinalRetention  # pylint: disable=import-outside-toplevel

    host = HostConsumer(child)
    replay = RetainedReplay(
        host,
        policy(child),
        state_type=HostState,
        state_codec=CODEC,
        retention=OrdinalRetention(
            profile="quiescent-ordinal-v1", source_epoch=EPOCH, maximum_tail_records=maximum_tail_records
        ),
    )
    replay.admit_history(history())
    return replay, host


class ReplayOwner:
    """Adapt one actual retained session to the existing Stateful protocol.

    Parameters
    ----------
    child : int
        Exact configured recipe index0..31. Fresh defaults are used only until
        the store restores its supplied/acknowledged complete continuation.

    Notes
    -----
    Owns both the host and its delivery session without duplicating authoritative
    state. This in-process adapter is explicit copied application code.
    """

    def __init__(self, child: int):
        """Create a fresh independent origin, ready for explicit restoration."""
        self.child = child
        self.replay, self.host = make_replay(child)

    def export_state(self) -> RetainedContinuation[HostState]:
        """Return the actual complete session and host with no durable side effect.

        Returns
        -------
        RetainedContinuation
            Concrete generation2 continuation, including all owned host fields.
        """
        return self.replay.export_continuation()

    def restore_state(self, state: RetainedContinuation[HostState], /) -> None:
        """Restore a complete compatible continuation into fresh local owners.

        Parameters
        ----------
        state : RetainedContinuation
            Exact child configuration and host-state codec.

        Raises
        ------
        ValueError, TypeError
            Invalid complete session or wrong child/codec, without replacing
            the previously retained pair.
        """
        from coleman.replay.retained import RetainedReplay  # pylint: disable=import-outside-toplevel

        host = HostConsumer(self.child)
        replay = RetainedReplay.from_continuation(host, state, state_type=HostState, state_codec=CODEC)
        self.host, self.replay = host, replay

    def step(self, value: float, *, at: int | None = None) -> ChoiceAction:
        """Run one actual chosen decision and complete synthetic outcome round.

        Parameters
        ----------
        value : float
            Explicit supplied bounded dimensionless loss, never an alternative
            or a fabricated realized observation.
        at : int or None, optional
            Known processing/input second offset. None uses global D.

        Returns
        -------
        ChoiceAction
            Actual offline unchanged action, ready for the separate domain book.

        Raises
        ------
        ValueError
            Invalid value/clocks or a replay admission/application failure.

        Notes
        -----
        Mutates this pair once through real decider/updater/replay. A caller
        seeking all-owner recovery performs it only in its owned store stage.
        """
        ordinal = self.replay.state.decision_count
        offset = ordinal if at is None else at
        prepared = self.replay.prepare(opportunity(self.child, ordinal, at=offset))
        record = self.replay.emit(prepared, execution="offline_identity_v1")
        self.replay.admit_feedback((feedback(record, decisions=ordinal + 1, available=offset, value=value),))
        self.replay.advance(eligible_through=instant(offset), apply_at=instant(offset))
        if not isinstance(record.executed.action, ChoiceAction):
            raise ValueError("Owned CI example requires its actual choice action")
        return record.executed.action


def owner_implementation() -> DefinitionBinding:
    """Bind the actual copied owner source at an explicit outer boundary.

    Returns
    -------
    DefinitionBinding
        Exact SHA-256 of this example file, not an inferred historical version.

    Raises
    ------
    OSError
        The explicitly selected owner source is unreadable.

    Notes
    -----
    Reads only this file. Direct host state/RNG operations never call it.
    """
    import hashlib  # pylint: disable=import-outside-toplevel
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    return DefinitionBinding(
        reference=IMPLEMENTATION,
        definition_fingerprint="sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def execution_compatibility() -> ExecutionCompatibility:
    """Explicitly capture the actual installed base runtime and copied owner code.

    Returns
    -------
    ExecutionCompatibility
        Generation2 selected-extras-empty runtime plus exact source definition.

    Raises
    ------
    ValueError, OSError
        Installed metadata/resources or copied source cannot be validated.

    Notes
    -----
    Performs package metadata/resource and this-file reads only when requested
    by the outer application. Pins establish consistency, not authentication or
    binary/ABI compatibility. Pure component state never scans the environment.
    """
    from coleman.spec.selected_runtime import SelectedInstalledRuntime  # pylint: disable=import-outside-toplevel
    from coleman.state.storage_contracts import ExecutionCompatibility  # pylint: disable=import-outside-toplevel

    return ExecutionCompatibility(
        schema_version=1,
        runtime=SelectedInstalledRuntime.capture(extras=()),
        owner_implementations=(owner_implementation(),),
    )


def _no_frontier(_state: DomainState) -> None:
    """Declare that the shared domain has no independent delivery frontier."""
    return None


def collection_layout(
    children: tuple[int, ...] = (0,),
) -> tuple[CollectionDefinition, tuple[ComponentBinding[Any, Any], ...]]:
    """Build finite explicit codecs/factories for children and the actual domain.

    Parameters
    ----------
    children : tuple of int, optional
        Required recipe inventory, default one child0. Duplicate children reject;
        declaration order does not change the canonical definition.

    Returns
    -------
    tuple
        Pure collection definition and trusted in-process component bindings.
        The outer continuation codec differs from the inner host codec.

    Raises
    ------
    ValueError, OSError
        Invalid inventory or unreadable explicitly selected owner source.

    Notes
    -----
    Reads this example's source for its implementation binding; creates no
    owner, store or runtime capture. Factories are called only by explicit host
    initialization or the store after complete candidate validation.
    """
    from coleman.replay.retained_contracts import RetainedContinuation  # pylint: disable=import-outside-toplevel
    from coleman.state.components import (  # pylint: disable=import-outside-toplevel
        ComponentBinding,
        ComponentCodec,
        child_frontier,
    )
    from coleman.state.contracts import StateBinding  # pylint: disable=import-outside-toplevel
    from coleman.state.storage_contracts import (  # pylint: disable=import-outside-toplevel
        CollectionDefinition,
        ComponentSpec,
    )

    if len(set(children)) != len(children):
        raise ValueError("Collection child declarations must be unique")
    implementation = owner_implementation()
    bindings, specs = [], []
    for child in sorted(children):
        identity = child_reference(child)
        binding = StateBinding(
            owner=identity,
            implementation=implementation,
            state_codec=reference("authored-ci-retained-continuation"),
            configuration_fingerprint=fingerprint({"child": child, "period": 2, "pending_limit": 128}),
        )
        codec = ComponentCodec(binding=binding, state_type=RetainedContinuation[HostState])
        bindings.append(
            ComponentBinding(
                codec=codec, new_owner=partial(ReplayOwner, child), frontier=partial(child_frontier, child=identity)
            )
        )
        specs.append(
            ComponentSpec(
                binding=binding,
                child=identity,
                roles=("algorithm", "learner_pending", "predictor", "transform", "delivery", "virtual_path"),
                frontier_profile="replay-ordinal-v2",
            )
        )
    domain_binding = StateBinding(
        owner=reference("zz-ci-actual-domain"),
        implementation=implementation,
        state_codec=reference("authored-ci-actual-domain-state"),
        configuration_fingerprint=fingerprint({"unit": "s", "plan_a": 3, "plan_b": 5, "switch": 2}),
    )
    bindings.append(
        ComponentBinding(codec=ComponentCodec(domain_binding, DomainState), new_owner=Domain, frontier=_no_frontier)
    )
    specs.append(
        ComponentSpec(binding=domain_binding, child=None, roles=("domain", "applied_path"), frontier_profile="none")
    )
    definition = CollectionDefinition(
        schema_version=1,
        reference=reference("authored-ci-composed-definition"),
        session=reference("authored-ci-composed-session"),
        source_epoch=EPOCH,
        children=tuple(child_reference(child) for child in sorted(children)),
        components=tuple(specs),
    )
    return definition, tuple(bindings)


def origin_values(bindings: tuple[ComponentBinding[Any, Any], ...]) -> tuple[ComponentValue[Any], ...]:
    """Explicitly supply configured origin snapshots before optional persistence.

    Parameters
    ----------
    bindings : tuple of ComponentBinding
        Finite trusted factory inventory produced by collection_layout.

    Returns
    -------
    tuple of ComponentValue
        Complete actual origin states. Already running hosts instead supply
        their actual snapshots; initialize never chooses a seed-zero fallback.

    Notes
    -----
    Calls the explicit fresh factories and exports their direct state. It does
    not open a store, acknowledge progress or compare execution compatibility.
    """
    from coleman.state.components import ComponentValue  # pylint: disable=import-outside-toplevel

    return tuple(ComponentValue(binding.codec, binding.new_owner().export_state()) for binding in bindings)


def run() -> dict[str, int | float | str]:
    """Execute one small actual owned-RNG/replay lifecycle without persistence.

    Returns
    -------
    dict
        Human demonstration receipt: A,A,B, D3/U3, loss3/4 and13 virtual seconds.
        No statistical or physical-CI efficacy claim follows.
    """
    replay, host = make_replay(0)
    for ordinal in range(3):
        prepared = replay.prepare(opportunity(0, ordinal))
        record = replay.emit(prepared, execution="offline_identity_v1")
        replay.admit_feedback((feedback(record, decisions=ordinal + 1, available=ordinal, value=(ordinal % 4) / 4),))
        replay.advance(eligible_through=instant(ordinal), apply_at=instant(ordinal))
    state = host.export_state()
    return {
        "evidence": "authored-synthetic-ci",
        "decisions": state.decision_count,
        "updates": state.update_count,
        "loss_sum": sum(state.loss_sums),
        "virtual_seconds": state.book.spent_seconds,
        "backend_refills": state.backend_refills,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
