"""Thin existing causal-replay bridge; the consumer never owns a future tape.

The replay action is the actual binary64 constrained point before the separate
exact funding adapter. It is not an exact holding, probability or trade. Later
aggregate feedback contains exogenous authored marks with synthetic provenance.
"""

from __future__ import annotations

from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Self, cast
from weakref import ref

from pydantic import Field, ModelWrapValidatorHandler, model_validator

from coleman.core.actions import VectorAction, VectorSpace
from coleman.core.base import CoreContract
from coleman.core.capabilities import Capability
from coleman.core.decisions import Decision, DecisionRequest
from coleman.core.feedback import AggregateFeedback, FeedbackBatch, OutcomeValue
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.primitives import Fingerprint, canonical_json, fingerprint
from coleman.core.sampling import SamplingRecord
from coleman.core.splits import TemporalObservation
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionBinding
from coleman.observations.contracts import ObservationHistory, ObservationRevision
from coleman.problems.portfolio import PricePanel
from coleman.problems.portfolio.contracts import PortfolioContract
from coleman.replay.contracts import (
    ReplayApplication,
    ReplayContinuation,
    ReplayDecisionRecord,
    ReplayFeedback,
    ReplayOpportunity,
    ReplayPolicy,
    ReplayTarget,
    ValueOriginEvidence,
)
from coleman.replay.engine import CausalReplay
from coleman.results.domain import Measurement
from coleman.state.contracts import StateBinding

from . import model
from .experiment import Experiment, _implementation
from .flow_contracts import ExperimentState, Observation

SOURCE = model.ref("replay-observed-prices")
CODEC = model.ref("portfolio-replay-host-state")
TASK = model.ref("pre-funding-continuous-allocation")
OBJECTIVE = model.ref("declared-continuous-target-distance")
PRICE_SEMANTICS = UnitSemantics(unit="synthetic-credit/asset-unit", meaning="authored-exogenous-current-mark-v1")
INPUTS = tuple(
    ScalarSpec(dimension=model.ref("eligible-price-" + letter), role="feature", minimum=0.0, semantics=PRICE_SEMANTICS)
    for letter in ("a", "b")
)
LABELS = tuple(
    ScalarSpec(dimension=model.ref("next-price-" + letter), role="objective", minimum=0.0, semantics=PRICE_SEMANTICS)
    for letter in ("a", "b")
)
SPACE = VectorSpace(
    feasible_set=model.ref("pre-funding-mathematical-point"),
    signed=False,
    coordinates=tuple(
        ScalarSpec(
            dimension=asset,
            role="action",
            minimum=0.0,
            maximum=1.0,
            semantics=UnitSemantics(unit="1", meaning="pre-funding-unit-balance-allocation-coordinate-v1"),
        )
        for asset in model.ASSETS
    ),
)


def _mark_numbers(values: tuple[int, int]) -> tuple[float, float]:
    """Keep this bridge's integer marks exactly representable in generic scalars."""
    if len(values) != 2 or any(type(value) not in (int,) or not 0 < value <= 2**53 for value in values):
        raise ValueError("Replay bridge requires two positive integer marks no larger than2**53")
    return float(values[0]), float(values[1])


def _panel(index: int, values: tuple[Scalar | OutcomeValue, ...], specs: tuple[ScalarSpec, ...]) -> PricePanel:
    """Require original complete observed units and exact integer price projection."""
    if len(values) != 2 or tuple(value.spec for value in values) != specs:
        raise ValueError("Portfolio replay requires its complete original mark dimensions/units")
    numbers = []
    for value in values:
        number = value.value
        if value.status != "observed" or number is None or not 0 < number <= 2**53 or not number.is_integer():
            raise ValueError(
                "Authored replay mark must be observed, positive and exactly representable integral binary64"
            )
        numbers.append(int(number))
    return model.price_panel(index, (numbers[0], numbers[1]))


def _exported_wire(value: CoreContract) -> str:
    """Observe complete immutable bytes from an actual public state boundary.

    Algorithm values use a full audit export or the controlled learner's explicit
    complete committed-state view. This private live-signature projection does
    not repeat validation through ``fingerprint``. Current book bytes and every
    component are compared with the previously validated owned snapshot; any
    outside mutation refuses host export. Caller-supplied state, restore and
    wire readers never use this observation helper.
    """
    return canonical_json(value.model_dump(mode="json"))


class PortfolioReplayState(PortfolioContract):
    """Own the same domain state plus original outer producers and applied labels."""

    binding: StateBinding
    experiment: ExperimentState
    pending_request: DecisionRequest | None
    applied: Annotated[tuple[ReplayApplication, ...], Field(max_length=15)]

    @model_validator(mode="after")
    def _originals(self) -> Self:
        """Reject a substituted original source, request or feedback-ready frontier."""
        if len(self.applied) != len(self.experiment.observations):
            raise ValueError("Portfolio replay lost actual applied original feedback")
        pending = self.experiment.next_index > 0 and self.experiment.ready_observation is None
        if pending != (self.pending_request is not None):
            raise ValueError("Portfolio replay lost its original pending action producer")
        if self.pending_request is not None and self.pending_request.round != self.experiment.next_index - 1:
            raise ValueError("Portfolio replay pending producer has another original period")
        for index, (application, observation) in enumerate(
            zip(self.applied, self.experiment.observations, strict=True), start=1
        ):
            if application.executed.proposal.request.round != index - 1 or observation.index != index:
                raise ValueError("Portfolio replay observation belongs to another original producer")
            payload = application.feedback.batch.feedback
            if (
                not isinstance(payload, AggregateFeedback)
                or _panel(index, payload.values, LABELS) != observation.prices
            ):
                raise ValueError("Portfolio replay applied marks differ from the actual original domain input")
        return self


class PortfolioReplayHost:
    """Expose the same real domain through existing direct replay protocols.

    Parameters
    ----------
    source_fingerprint : str
        Frozen complete source identity only. No source values, future table or
        feedback oracle enter this consumer; each current mark is supplied later.
    strength : float, optional
        Same fixed development-selected L1 strength as the direct experiment.

    Notes
    -----
    This fixed daily synchronous profile waits for the previous target's actual
    marks before another decision. Late processing beyond that original calendar
    is unsupported. Existing replay owns eligibility, queueing, corrections,
    duplicate delivery and busy ownership. This wrapper adds no scheduler.
    """

    def __init__(self, *, source_fingerprint: Fingerprint, strength: float = 0.05) -> None:
        """Construct only actual kernels and a source identity, with no future rows."""
        self.experiment = Experiment(strength=strength, rows=None, source_fingerprint=source_fingerprint)
        self.pending_request: DecisionRequest | None = None
        self.applied: tuple[ReplayApplication, ...] = ()
        self._cached_snapshot: PortfolioReplayState | None = None
        self._cached_canonical = ""
        self._cached_signature: tuple[object, ...] = ()
        owner_reference = ref(self)

        class OwnedPortfolioReplayState(PortfolioReplayState):
            """Use unchanged typed wire with a private exact-owned-byte reuse gate."""

            @model_validator(mode="wrap")
            @classmethod
            def _reuse_owned(cls, value: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
                """Fully validate every unowned/copied/constructed/JSON candidate."""
                # The closure is private to this actual owner, never selected by
                # wire content. No authority or cache is shared between hosts.
                owner = owner_reference()
                # Exact concrete type is part of the authority check; accepting
                # subclasses would widen it to caller-selected implementations.
                if owner is not None and type(value) is cls and value is owner._cached_snapshot:  # pylint: disable=protected-access,unidiomatic-typecheck
                    current = canonical_json(value.model_dump(mode="json"))
                    if (  # pylint: disable=protected-access
                        current == owner._cached_canonical and owner._cached_signature == owner._live_signature()
                    ):
                        return value
                return handler(value)

        self._state_type = OwnedPortfolioReplayState
        self._refresh_snapshot()

    @property
    def state_type(self) -> type[PortfolioReplayState]:
        """Return this trusted owner's concrete type, never a wire-selected class.

        Returns
        -------
        type of PortfolioReplayState
            Private per-owner subtype with identical versioned fields and JSON.
            Use it consistently for existing replay construction and decoding.
        """
        return self._state_type

    def _live_signature(self) -> tuple[object, ...]:
        """Detect outside-owner mutation without revalidating the report archive."""
        current = self.experiment
        return (
            self.binding.model_dump_json(),
            id(current),
            current.strength,
            current.source_fingerprint,
            id(current.rows),
            id(current.periods),
            id(current.observations),
            id(current.observed_prices),
            id(current.pending),
            id(current.ready_observation),
            id(self.pending_request),
            id(self.applied),
            tuple(
                (name, id(book), id(book.export_state()), _exported_wire(book.book))
                for name, book in sorted(current.books.items())
            ),
            tuple(
                (name, id(learner), _exported_wire(learner.export_state()))
                for name, learner in sorted(current.references.items())
            ),
            tuple(
                (name, id(learner), _exported_wire(learner.view_state()))
                for name, learner in sorted(current.controlled.items())
            ),
            id(current.ridge),
            _exported_wire(current.ridge.export_state()),
        )

    def _refresh_snapshot(self) -> None:
        """Validate and close a complete immutable snapshot after actual mutation."""
        # Constructor validation always takes the normal complete path: the new
        # object cannot be the previously owned cached instance. Capture every
        # inner field, then validate it here once rather than validating the
        # full archive both in the inner export and again at this outer boundary.
        candidate = self._state_type(
            schema_version=1,
            binding=self.binding,
            experiment=self.experiment._capture_state(),  # pylint: disable=protected-access
            pending_request=self.pending_request,
            applied=self.applied,
        )
        self._cached_canonical = canonical_json(candidate.model_dump(mode="json"))
        self._cached_snapshot = candidate
        self._cached_signature = self._live_signature()

    @property
    def binding(self) -> StateBinding:
        """Return exact host, mathematical configuration and concrete codec identity."""
        return StateBinding(
            owner=model.ref("portfolio-causal-owner"),
            implementation=DefinitionBinding(
                reference=model.ref("portfolio-causal-host"),
                definition_fingerprint=fingerprint(
                    {
                        "domain": _implementation().definition_fingerprint,
                        "bridge": sha256(Path(__file__).read_bytes()).hexdigest(),
                    }
                ),
            ),
            state_codec=CODEC,
            configuration_fingerprint=fingerprint(
                {
                    "source": self.experiment.source_fingerprint,
                    "strength": self.experiment.strength,
                    "calendar": "daily-synchronous-v1",
                }
            ),
        )

    @property
    def capability(self) -> Capability:
        """Declare pre-funding vectors and whole-action exogenous-mark feedback."""
        return Capability(
            schema_version=1,
            implementation=self.binding.implementation.reference,
            action_kinds=("vector",),
            feedback_kinds=("aggregate",),
            required_values=LABELS,
            required_inputs=INPUTS,
            origins=("synthetic",),
            timing_modes=("future",),
            supports_delayed_feedback=False,
            requires_known_receipt=True,
            requires_observed=True,
            state_owner="host",
            requires_sampling=False,
            sampling_requirement="not_required",
        )

    def decide(self, request: DecisionRequest, /) -> Decision:
        """Emit actual constrained binary64 coordinates after eligible prior feedback.

        Parameters
        ----------
        request : DecisionRequest
            Exact current mark inputs and fixed daily producing clocks. The
            existing replay has selected their as-of source view and fingerprint.

        Returns
        -------
        Decision
            Actual composed QP coordinates before exact funding. Exact Target,
            cash, holdings, netted trades and paid fees remain separate receipts.

        Raises
        ------
        ValueError
            Missing prior feedback/current mark, changed schema/calendar or
            refused kernel. This wrapper executes no external effect.
        """
        checked = DecisionRequest.model_validate(request)
        index = checked.round
        at = model.START + timedelta(days=index)
        timing = checked.timing
        if checked.task != TASK or checked.objective != OBJECTIVE or checked.space != SPACE or timing is None:
            raise ValueError("Portfolio replay decision has another declared task/objective/space")
        if (timing.decision_at, timing.information_cutoff, timing.target_start, timing.target_end) != (
            at,
            at,
            at,
            at + timedelta(days=1),
        ):
            raise ValueError("Portfolio replay decision differs from the original daily calendar")
        if self.pending_request is not None or index != len(self.experiment.periods):
            raise ValueError("Portfolio replay cannot replace an original unresolved action")
        if index > 0 and self.experiment.ready_observation is None:
            raise ValueError("Portfolio replay waits for eligible original previous-period marks")
        period = self.experiment.step(index, prices=_panel(index, checked.inputs, INPUTS))
        component = next(row for row in period.candidates if row.name == "composed-online")
        assert component.solver is not None and component.solver.executed is not None
        action = VectorAction(
            schema_version=1,
            space=SPACE,
            values=tuple(
                Scalar(spec=spec, status="observed", value=value)
                for spec, value in zip(SPACE.coordinates, component.solver.executed, strict=True)
            ),
        )
        result = Decision(
            schema_version=1,
            request=checked,
            implementation=self.capability.implementation,
            action=action,
            state_owner="host",
            sampling=SamplingRecord(
                mechanism="deterministic",
                mechanism_ref=self.capability.implementation,
                status="not_required",
                action_fingerprint=action.fingerprint,
                probability=None,
            ),
        )
        self.pending_request = checked
        self._refresh_snapshot()
        return result

    def update(self, application: ReplayApplication, /) -> Observation:
        """Apply original later exogenous marks exactly at their explicit clock.

        Parameters
        ----------
        application : ReplayApplication
            Existing replay's actual original execution, selected aggregate mark
            revision, eligibility, receipt and supplied application instant.

        Returns
        -------
        Observation
            Actual signed recipe outcomes, original scales and appended ridge
            label. The complete feedback-ready phase survives before a decision.

        Raises
        ------
        ValueError
            Wrong producer/horizon, missing marks or unsupported late processing.
            A rejected host callback invalidates its enclosing replay session.
        """
        checked = ReplayApplication.model_validate(application)
        original = checked.executed.proposal.request
        if self.pending_request is None or self.pending_request.fingerprint != original.fingerprint:
            raise ValueError("Portfolio replay feedback has another original action producer")
        index = original.round + 1
        payload = checked.feedback.batch.feedback
        if not isinstance(payload, AggregateFeedback) or checked.feedback.batch.horizon != model.HORIZON:
            raise ValueError("Portfolio replay requires original whole-action mark observations")
        quote = _panel(index, payload.values, LABELS)
        if checked.eligible_at != quote.at or checked.apply_at != quote.at:
            raise ValueError("Fixed daily portfolio replay does not backdate late processing")
        result = self.experiment.apply_feedback(index, prices=quote)
        self.pending_request = None
        self.applied += (checked,)
        self._refresh_snapshot()
        return result

    def export_state(self) -> PortfolioReplayState:
        """Capture complete domain, original pending producer and applied lineage.

        Returns
        -------
        PortfolioReplayState
            Known-version complete snapshot, without I/O, ACK or current-clock reads.
        """
        if self._cached_signature != self._live_signature():
            raise ValueError("Portfolio components changed outside their owning replay host")
        assert self._cached_snapshot is not None
        return self._cached_snapshot

    def restore_state(self, state: PortfolioReplayState, /) -> None:
        """Restore the exact same source/law and every original pending phase.

        Parameters
        ----------
        state : PortfolioReplayState
            Complete actual direct/domain plus replay-host state. The enclosing
            replay restores its own queue and closed-drain frontier separately.

        Raises
        ------
        ValueError
            Changed source, implementation, codec, configuration or producer joins.
            All actual owners are staged before replacing live state.
        """
        checked = PortfolioReplayState.model_validate(state)
        if checked.binding != self.binding:
            raise ValueError("Portfolio replay continuation has another exact host binding")
        staged = Experiment(
            strength=self.experiment.strength, rows=None, source_fingerprint=self.experiment.source_fingerprint
        )
        staged.restore_state(checked.experiment)
        self.experiment, self.pending_request, self.applied = staged, checked.pending_request, checked.applied
        self._refresh_snapshot()

    def resume(self, continuation: ReplayContinuation[PortfolioReplayState]) -> CausalReplay[PortfolioReplayState]:
        """Select this fresh owner's concrete type before existing replay restore.

        Parameters
        ----------
        continuation : ReplayContinuation
            Complete inspected in-memory or JSON-read snapshot. This trusted
            host was constructed with the expected source and declared strength;
            the artifact never chooses an import or implementation class.

        Returns
        -------
        CausalReplay
            Actual existing engine restored through the same known typed wire.

        Raises
        ------
        ValueError, TypeError
            Any source, codec, schema, implementation, producer or state mismatch.
            Unowned input receives complete typed validation before acceptance.
        """
        inspected = ReplayContinuation[PortfolioReplayState].model_validate(continuation)
        selected = self.state_type.model_validate(inspected.host_state.model_dump(mode="python"))
        envelope = cast(
            type[ReplayContinuation[PortfolioReplayState]], ReplayContinuation.__class_getitem__(self.state_type)
        )
        current = envelope(schema_version=1, state=inspected.state, host_state=selected)
        return CausalReplay.from_continuation(self, current, state_type=self.state_type, state_codec=CODEC)


def source_history(rows: tuple[tuple[int, int], ...] = model.PRICE_ROWS) -> ObservationHistory:
    """Author source facts outside the consumer, with genuine as-of visibility.

    Parameters
    ----------
    rows : tuple of tuple of int, optional
        Complete authored exogenous source, held by the caller and replay only.

    Returns
    -------
    ObservationHistory
        Independent coordinates of one dependent authored path, not independent
        observations. Each mark becomes available/received on its original date.
    """
    revisions = tuple(
        ObservationRevision(
            schema_version=1,
            metadata=TemporalObservation(
                observation_id=f"price-{index}-{member}",
                group_id="one-authored-dependent-path",
                event_at=model.START + timedelta(days=index),
                available_at=model.START + timedelta(days=index),
            ),
            entity_id=spec.dimension.id,
            received_at=model.START + timedelta(days=index),
            supersedes=None,
            measurement=Measurement(
                name=spec.dimension.id,
                value=value,
                unit=spec.semantics.unit,
                meaning=spec.semantics.meaning,
                direction="none",
                status="observed",
            ),
        )
        for index, values in enumerate(rows)
        for member, (spec, value) in enumerate(zip(INPUTS, _mark_numbers(values), strict=True))
    )
    return ObservationHistory(schema_version=1, source_id=SOURCE.id, source_version=SOURCE.version, revisions=revisions)


def start(
    *, rows: tuple[tuple[int, int], ...] = model.PRICE_ROWS, strength: float = 0.05
) -> tuple[PortfolioReplayHost, CausalReplay[PortfolioReplayState]]:
    """Create the existing replay and pass only a source identity into its host.

    Parameters
    ----------
    rows : tuple of tuple of int, optional
        Authored source owned by this outer adapter, never by the consumer.
    strength : float, optional
        Same previously development-selected constant coefficient penalty.

    Returns
    -------
    tuple
        Actual portfolio host and existing CausalReplay with admitted source.
    """
    host = PortfolioReplayHost(source_fingerprint=fingerprint([list(row) for row in rows]), strength=strength)
    replay = CausalReplay(
        host,
        ReplayPolicy(
            schema_version=1,
            stream=TASK,
            lineage=model.ref("portfolio-replay-lineage"),
            parent=None,
            source=SOURCE,
            capability=host.capability,
            delivery="synchronous",
            specialists=False,
        ),
        state_type=host.state_type,
        state_codec=CODEC,
    )
    replay.admit_history(source_history(rows))
    return host, replay


def opportunity(replay: CausalReplay[PortfolioReplayState], index: int) -> ReplayOpportunity:
    """Derive current typed inputs only from the existing eligible source view.

    Parameters
    ----------
    replay : CausalReplay
        Existing history/eligibility owner; no consumer source table is queried.
    index : int
        Original source decision position0..15.

    Returns
    -------
    ReplayOpportunity
        Exact source-selected current marks, clocks, horizon and dependence group.

    Raises
    ------
    ValueError
        Missing or incompatible current input; it is never supplied as zero.
    """
    at = model.START + timedelta(days=index)
    snapshot = replay.eligible_snapshot(at)
    observations = {row.metadata.observation_id: row for row in snapshot.observations}
    inputs = []
    for member, spec in enumerate(INPUTS):
        row = observations.get(f"price-{index}-{member}")
        if (
            row is None
            or row.measurement.status != "observed"
            or (row.measurement.unit, row.measurement.meaning) != (spec.semantics.unit, spec.semantics.meaning)
        ):
            raise ValueError("Current eligible source mark is missing or incompatible")
        inputs.append(Scalar(spec=spec, value=row.measurement.value, status="observed"))
    return ReplayOpportunity(
        schema_version=1,
        request=DecisionRequest(
            schema_version=1,
            task=TASK,
            decision_id=f"portfolio-decision-{index}",
            round=index,
            space=SPACE,
            objective=OBJECTIVE,
            inputs=tuple(inputs),
            timing=DecisionTiming(
                decision_at=at,
                information_cutoff=at,
                target_start=at,
                target_end=at + timedelta(days=1),
                eligible_input_fingerprint=snapshot.fingerprint,
                mode="future",
            ),
        ),
        context=None,
        target=ReplayTarget(
            reference=model.ref("portfolio-target-" + str(index)),
            horizon=model.HORIZON,
            groups=(model.ref("one-authored-dependent-path"),),
        ),
    )


def feedback(
    record: ReplayDecisionRecord,
    marks: tuple[int, int],
    *,
    revision: str = "initial",
    supersedes: str | None = None,
    receipt_days: int = 0,
) -> ReplayFeedback:
    """Supply later source marks outside the consumer with exact producer joins.

    Parameters
    ----------
    record : ReplayDecisionRecord
        Actual unchanged earlier pre-funding allocation execution.
    marks : tuple of int
        Later authored exogenous marks, not a funded payoff or expert-loss panel.
    revision : str, optional
        Original or correction identity, with separate previous revision.
    supersedes : str or None, optional
        Exact earlier revision, otherwise an independent initial revision.
    receipt_days : int, optional
        Nonnegative delayed correction receipt for preserve-live audit tests.

    Returns
    -------
    ReplayFeedback
        Whole-action exogenous observations with explicit synthetic generator.
        Eligibility/updates remain owned by existing replay, never this helper.
    """
    timing = record.opportunity.request.timing
    assert timing is not None
    at = timing.target_end
    if type(receipt_days) not in (int,) or receipt_days < 0:
        raise ValueError("Correction receipt delay requires nonnegative integral days")
    values = tuple(
        OutcomeValue(spec=spec, value=value, status="observed", origin="synthetic", transform=None)
        for spec, value in zip(LABELS, _mark_numbers(marks), strict=True)
    )
    producer = DefinitionBinding(
        reference=model.ref("authored-mark-generator"),
        definition_fingerprint=fingerprint(
            {
                "profile": "authored-exogenous-mark-revisions-v1",
                "unit": PRICE_SEMANTICS.unit,
                "calendar": "daily-utc-v1",
            }
        ),
    )
    return ReplayFeedback(
        schema_version=1,
        lineage=record.lineage,
        target=record.opportunity.target,
        batch=FeedbackBatch(
            schema_version=1,
            decision_fingerprint=record.executed.fingerprint,
            feedback_id="portfolio-next-exogenous-mark",
            revision_id=revision,
            supersedes=supersedes,
            horizon=model.HORIZON,
            feedback=AggregateFeedback(values=values),
            delay_rounds=0,
            timing=FeedbackTiming(
                event_at=at, available_at=at, mature_at=at, received_at=at + timedelta(days=receipt_days)
            ),
        ),
        origins=tuple(
            ValueOriginEvidence(
                member=None,
                dimension=value.spec.dimension,
                value_fingerprint=fingerprint(value.model_dump(mode="json")),
                origin="synthetic",
                producer=producer,
            )
            for value in values
        ),
    )
