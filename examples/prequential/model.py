"""Host-owned synthetic CI model used by the prequential comparison example.

All duration and reconfiguration quantities are modeled seconds. No physical
test process, financial book, market source or paid charge is represented.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import fsum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from coleman.core.actions import ChoiceAction, ChoiceSpace, MixtureAction, MixtureSpace, Weight
from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision, MemberInputs
from coleman.core.feedback import FeedbackBatch, FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64, Scalar, ScalarSpec
from coleman.core.primitives import canonical_json, fingerprint
from coleman.core.sampling import SamplingRecord
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.core.units import UnitSemantics
from coleman.evaluation.design.contracts import MetricPlan
from coleman.evaluation.prequential.contracts import DomainExecutionReceipt, ExecutionKey, StepEvidence
from coleman.evaluation.prequential.execution import DomainExecutionContext
from coleman.learning.contracts import ReferenceConfig
from coleman.learning.control_contracts import ControlledConfig
from coleman.learning.controlled import ControlledHedge
from coleman.learning.direct import DirectMixture, DirectOptions
from coleman.learning.reference import Hedge
from coleman.learning.replay import MixtureApplication, ReplayMixture
from coleman.learning.schedule import ControlledDirect, ControlSchedule
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage, QuantityMeaning
from coleman.objectives.controls import (
    CoefficientOrigin,
    ExpertBound,
    ExpertBounds,
    L1MovementPenalty,
    MixtureControls,
    MovementAllowance,
    MovementLimit,
    ProjectionOutput,
    RoundCadence,
)
from coleman.objectives.feasibility import (
    FiniteAlternative,
    FiniteFeasibilityRequest,
    ScalarConstraint,
    check_finite_feasibility,
)
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS, DeclaredTargetShare, InitialPrior
from coleman.observations.contracts import ObservationHistory
from coleman.replay.contracts import ReplayFeedback, ReplayOpportunity, ReplayPolicy, ReplayTarget, ValueOriginEvidence
from coleman.replay.retained import RetainedReplay, ordinal_decision_id
from coleman.replay.retained_contracts import OrdinalRetention
from coleman.state.contracts import StateBinding

MODES = ("fixed", "plain", "target", "bounded", "movement", "cadence", "penalized", "online", "accounting_off")
MEMBERS = tuple(Reference(id="ci-expert-" + name, version="1") for name in ("fast", "full"))
PLANS = tuple(Reference(id="ci-plan-" + name, version="1") for name in ("fast", "full"))
EPOCH = Reference(id="constructed-prequential-ci-epoch", version="1")
SOURCE = Reference(id="constructed-prequential-ci-inputs", version="1")
OBJECTIVE = Reference(id="constructed-ci-duration-objective", version="1")
HORIZON = Reference(id="one-modeled-ci-decision", version="1")
DOMAIN_TASK = Reference(id="constructed-ci-plan-execution", version="1")


def ref(name: str) -> Reference:
    """Return a local version-one identity without registration.

    Parameters
    ----------
    name : str
        Explicit logical identifier.

    Returns
    -------
    Reference
        Immutable name/version; no lookup, I/O or source admission."""
    return Reference(id=name, version="1")


def definition(name: str, text: str) -> DefinitionBinding:
    """Bind the complete supplied behavior description without authenticating it.

    Parameters
    ----------
    name, text : str
        Versioned logical name and complete declared rule description.

    Returns
    -------
    DefinitionBinding
        Deterministic description digest, not proof arbitrary code followed it."""
    return DefinitionBinding(reference=ref(name), definition_fingerprint=fingerprint({"definition": text}))


def lineage() -> DefinitionLineage:
    """Keep absent declaration evidence explicit rather than inventing history."""
    return DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)


ADAPTER = definition("constructed-ci-domain-adapter", "argmax p_i*(8-forecast_i); full tie-break; finite projection")
INTERVENTION = definition("constructed-ci-budget-intervention", "retain proposed feasible plan, otherwise choose fast")
TRANSFORM = definition("constructed-ci-duration-normalization", "bounded synthetic expert seconds divided by eight")
GENERATOR = definition(
    "constructed-ci-duration-table", "four authored worlds, three paired expert-duration observations"
)


def loss_spec() -> ScalarSpec:
    """Describe dimensionless expert duration/8, inclusive [0,1], not probability."""
    return ScalarSpec(
        dimension=ref("constructed-expert-duration-loss"),
        role="loss",
        minimum=0.0,
        maximum=1.0,
        semantics=UnitSemantics(unit="1", meaning="synthetic-expert-duration-divided-by-eight-v1"),
    )


def metric(name: str = "terminal_net_seconds") -> MetricPlan:
    """Declare the one-decision primary outcome in modeled seconds.

    Parameters
    ----------
    name : str, optional
        Explicit metric name, default terminal_net_seconds.

    Returns
    -------
    MetricPlan
        Inclusive [0,16] modeled seconds, minimize; no physical-runtime guarantee.
        The comparison separately names which terminal opportunity supplies a cell."""
    return MetricPlan(
        name=name,
        quantity=QuantityMeaning(
            spec=ScalarSpec(
                dimension=ref(name),
                role="objective",
                minimum=0.0,
                maximum=16.0,
                semantics=UnitSemantics(unit="s", meaning=name + "-modeled-v1"),
            ),
            interpretation="raw_outcome",
            horizon=HORIZON,
        ),
        direction="minimize",
        role="executed_objective",
        estimand=definition(name + "-estimand", "One named terminal decision; do not sum its preceding trajectory."),
    )


def metrics() -> tuple[MetricPlan, ...]:
    """Keep raw duration, modeled reconfiguration and coefficient movement distinct."""
    result = [metric()]
    entries: tuple[tuple[str, str, str, Literal["raw_outcome", "diagnostic"], float], ...] = (
        ("gross_seconds", "s", "authored-chosen-plan-duration-v1", "raw_outcome", 8.0),
        ("reconfiguration_seconds", "s", "modeled-coefficient-reconfiguration-seconds-v1", "diagnostic", 8.0),
        ("coefficient_l1", "1", "actual-full-coefficient-L1-v1", "diagnostic", 2.0),
        (
            "proposal_without_intervention_seconds",
            "s",
            "unexecuted-proposal-counterfactual-duration-v1",
            "diagnostic",
            8.0,
        ),
    )
    for name, unit, meaning, role, maximum in entries:
        result.append(
            MetricPlan(
                name=name,
                quantity=QuantityMeaning(
                    spec=ScalarSpec(
                        dimension=ref(name),
                        role="objective",
                        minimum=0.0,
                        maximum=maximum,
                        semantics=UnitSemantics(unit=unit, meaning=meaning),
                    ),
                    interpretation="raw_outcome",
                    horizon=HORIZON,
                ),
                direction="minimize" if role == "raw_outcome" else "none",
                role=role,
                estimand=definition(
                    name + "-estimand",
                    "Separate one-decision component; no implicit physical-cost or fee interpretation.",
                ),
            )
        )
    return tuple(result)


def configuration(mode: str) -> ReferenceConfig | ControlledConfig:
    """Freeze one named ablation; lambda is a declared coefficient parameter.

    Parameters
    ----------
    mode : str
        One of MODES. Online changes only host prediction relative to penalized.

    Returns
    -------
    ReferenceConfig or ControlledConfig
        Actual shipped method law; fixed uses the eta=0 identity endpoint.
    """
    if mode not in MODES:
        raise ValueError("Unknown declared CI comparison mode")
    prior = InitialPrior(
        schema_version=1,
        reference=ref("ci-half-prior"),
        lineage=lineage(),
        weights=tuple(Weight(member=m, value=0.5) for m in MEMBERS),
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )
    if mode in {"fixed", "plain"}:
        return ReferenceConfig(
            schema_version=1,
            reference=ref("ci-" + mode),
            lineage=lineage(),
            profile="hedge-constant",
            prior=prior,
            loss_spec=loss_spec(),
            transform=TRANSFORM,
            eta=0.0 if mode == "fixed" else 1.0,
        )
    target = DeclaredTargetShare(
        schema_version=1,
        reference=ref("declared-ci-target"),
        lineage=lineage(),
        weights=(Weight(member=MEMBERS[0], value=0.6), Weight(member=MEMBERS[1], value=0.4)),
        alpha=0.1,
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )
    bounds = ExpertBounds(
        schema_version=1,
        reference=ref("ci-boxes"),
        lineage=lineage(),
        bounds=tuple(
            ExpertBound(member=m, lower=0.0 if mode == "target" else 0.2, upper=1.0 if mode == "target" else 0.8)
            for m in MEMBERS
        ),
        semantics=COEFFICIENT_SEMANTICS,
    )
    movement = (
        None
        if mode in {"target", "bounded"}
        else MovementAllowance(
            schema_version=1,
            reference=ref("ci-per-release-movement"),
            lineage=lineage(),
            scope="per_release",
            measure="accumulated_absolute",
            limits=tuple(MovementLimit(member=m, maximum=0.1) for m in MEMBERS),
            semantics=COEFFICIENT_SEMANTICS,
        )
    )
    penalized = mode in {"penalized", "online", "accounting_off"}
    origin = CoefficientOrigin(
        schema_version=1,
        reference=ref("ci-initial-origin"),
        lineage=lineage(),
        weights=prior.weights,
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )
    output = (
        L1MovementPenalty(
            schema_version=1,
            reference=ref("development-ci-lambda"),
            lineage=lineage(),
            strength=0.05,
            origin=origin,
            semantics=COEFFICIENT_SEMANTICS,
        )
        if penalized
        else ProjectionOutput()
    )
    cadence = RoundCadence(panels=2 if mode in {"cadence", "penalized", "online", "accounting_off"} else 1)
    controls = MixtureControls(
        schema_version=1,
        reference=ref("ci-controls-" + mode),
        lineage=lineage(),
        initial_prior=prior,
        target=target,
        bounds=bounds,
        movement=movement,
        cadence=cadence,
        output=output,
    )
    return ControlledConfig(
        schema_version=1,
        reference=ref("ci-controlled-" + mode),
        lineage=lineage(),
        profile="controlled-hedge-l1-movement" if penalized else "controlled-hedge-projection",
        controls=controls,
        loss_spec=loss_spec(),
        transform=TRANSFORM,
        eta=1.0,
    )


@dataclass(frozen=True)
class Recipe:
    """Hold one immutable law, complete schedule and BASE candidate identity."""

    mode: str
    config: ReferenceConfig | ControlledConfig
    schedule: ControlSchedule
    base_fingerprint: str
    data_fingerprint: str
    world: str

    @property
    def child(self) -> Reference:
        """Return a distinct actual owner identity for this candidate and world."""
        return ref("ci-" + self.mode + "-" + self.world)


def application(recipe: Recipe) -> MixtureApplication[Any]:
    """Construct a fresh actual shipped learner and original-producer bridge.

    Parameters
    ----------
    recipe : Recipe
        Complete frozen reference/controlled law and explicit schedule.

    Returns
    -------
    DirectMixture or ControlledDirect
        Fresh independently owned actual mathematical and producer state.

    Raises
    ------
    ValueError
        Invalid configuration, unsupported profile or incompatible schedule.

    Notes
    -----
    Performs no source lookup, external action, artifact publication or inference.
    The heterogeneous provider boundary erases the static state parameter only;
    each actual provider still supplies its concrete trusted state_type codec,
    which is retained and validated by the replay and joint checkpoint owners."""
    options = DirectOptions(task=recipe.child, objective=OBJECTIVE, horizon=HORIZON)
    if isinstance(recipe.config, ReferenceConfig):
        return DirectMixture(
            Hedge(recipe.config, owner=recipe.child), task=recipe.child, objective=OBJECTIVE, horizon=HORIZON
        )
    return ControlledDirect(ControlledHedge(recipe.config, owner=recipe.child), recipe.schedule, options)


class ReplayOwner:
    """Own one complete actual mathematical, producer and causal replay chain.

    Parameters
    ----------
    recipe : Recipe
        Immutable fixed method, explicit schedule and independent world identity.

    Notes
    -----
    Construction allocates fresh local owners and an empty admitted input history;
    it performs no learning, storage, physical action or source-outcome lookup."""

    def __init__(self, recipe: Recipe):
        """Create fresh learner, original producer and replay owners without I/O."""
        self.recipe = recipe
        self.application = application(recipe)
        self.consumer = ReplayMixture(self.application)
        policy = ReplayPolicy(
            schema_version=1,
            stream=recipe.child,
            lineage=ref(recipe.child.id + "-lineage"),
            parent=None,
            source=SOURCE,
            capability=self.application.capability,
            delivery="delayed",
            specialists=False,
        )
        self.session = RetainedReplay(
            self.consumer,
            policy,
            state_type=self.application.state_type,
            state_codec=ref("ci-actual-mixture-host"),
            retention=OrdinalRetention(profile="quiescent-ordinal-v1", source_epoch=EPOCH, maximum_tail_records=128),
        )
        self.session.admit_history(
            ObservationHistory(schema_version=1, source_id=SOURCE.id, source_version=SOURCE.version, revisions=())
        )

    def export_state(self):
        """Export the actual complete continuation, including pending producers."""
        return self.session.export_continuation()

    def restore_state(self, state, /) -> None:
        """Restore an exact compatible complete continuation into fresh owners.

        Parameters
        ----------
        state : RetainedContinuation
            Concrete original learner, pending producer and replay state.

        Raises
        ------
        ValueError, TypeError
            Invalid/future version, wrong law/codec or incomplete ownership. Assignment
            occurs only after the fresh chain has successfully restored all state."""
        fresh = application(self.recipe)
        consumer = ReplayMixture(fresh)
        session = RetainedReplay.from_continuation(
            consumer, state, state_type=fresh.state_type, state_codec=ref("ci-actual-mixture-host")
        )
        self.application, self.consumer, self.session = fresh, consumer, session

    def opportunity(self, ordinal: int, at: datetime) -> ReplayOpportunity:
        """Build a timed coefficient request from the actual eligible source view.

        Parameters
        ----------
        ordinal : int
            Logical decision index, independent of feedback/update/release counts.
        at : datetime
            Explicit zoned common logical decision/cutoff tick.

        Returns
        -------
        ReplayOpportunity
            Complete mixture request and modeled build target; no future outcome is
            included. This performs a source-view read without a learner decision."""
        request = DecisionRequest(
            schema_version=1,
            task=self.recipe.child,
            decision_id=ordinal_decision_id(EPOCH, ordinal),
            round=ordinal,
            space=MixtureSpace(
                feasible_set=ref("ci-expert-coefficients"), members=MEMBERS, profile="binary64-simplex-v1"
            ),
            objective=OBJECTIVE,
            timing=DecisionTiming(
                decision_at=at,
                information_cutoff=at,
                target_start=at,
                target_end=at,
                mode="future",
                eligible_input_fingerprint=self.session.eligible_snapshot(at).fingerprint,
            ),
        )
        return ReplayOpportunity(
            schema_version=1,
            request=request,
            context=None,
            target=ReplayTarget(
                reference=ref("ci-target-" + str(ordinal)), horizon=HORIZON, groups=(ref(self.recipe.world),)
            ),
        )


class DomainState(CoreContract):
    """Retain complete host forecasts, emitted actions and separately received outcomes.

    Attributes
    ----------
    binding : StateBinding
        Complete fixed host law and BASE candidate/world identity.
    receipts, observations : tuple
        Decision-time receipts and later original observations; pending receipts
        have no fabricated outcome. Every applied observation remains retained.
    sums, count : tuple of float, int
        Expanding sums/count of past eligible complete expert durations. Frozen
        prediction retains these observations but does not use them to forecast.
    last_coefficients, movements, charges : tuple
        Actual emitted weights, full coefficient L1 and modeled seconds charged
        at each change. Lambda never appears in this accounting formula.
    """

    binding: StateBinding
    receipts: Annotated[tuple[DomainExecutionReceipt, ...], Field(max_length=16)] = ()
    observations: Annotated[tuple[StepEvidence, ...], Field(max_length=16)] = ()
    expert_history: Annotated[tuple[tuple[Binary64, Binary64], ...], Field(max_length=16)] = ()
    sums: tuple[Binary64, Binary64] = (0.0, 0.0)
    count: Annotated[int, Field(strict=True, ge=0, le=16)] = 0
    last_coefficients: tuple[Binary64, Binary64] = (0.5, 0.5)
    movements: Annotated[tuple[Binary64, ...], Field(max_length=16)] = ()
    charges: Annotated[tuple[Binary64, ...], Field(max_length=16)] = ()

    @model_validator(mode="after")
    def _complete(self) -> Self:
        """Reject incomplete parallel accounting arrays and duplicate applications."""
        if len(self.receipts) != len(self.movements) or len(self.receipts) != len(self.charges):
            raise ValueError("Domain accounting must retain every emitted action")
        keys = [row.key.logical_identity for row in self.receipts]
        observed = [row.receipt.key.logical_identity for row in self.observations]
        if len(keys) != len(set(keys)) or len(observed) != len(set(observed)) or not set(observed) <= set(keys):
            raise ValueError("Domain observations require distinct original retained producers")
        if self.count != len(self.observations):
            raise ValueError("Predictor count differs from its complete applied observation history")
        if len(self.expert_history) != self.count:
            raise ValueError("Predictor history must retain each original complete expert panel")
        if any(value < 0.0 or value > 8.0 for panel in self.expert_history for value in panel):
            raise ValueError("Original expert seconds exceed the admitted source range")
        expected_sums = tuple(sum(panel[index] for panel in self.expert_history) for index in range(2))
        if self.sums != expected_sums:
            raise ValueError("Predictor sufficient statistics differ from their complete original history")
        return self


class ModeledComponents(CoreImmutable):
    """Keep one decision's five deterministic quantities in their declared units.

    Attributes
    ----------
    terminal_net_seconds, gross_seconds, reconfiguration_seconds : float
        Modeled seconds. Net is gross plus the separate coefficient-change model.
    coefficient_l1 : float
        Full absolute coefficient movement, dimensionless and never book turnover.
    proposal_without_intervention_seconds : float
        Counterfactual raw seconds of the unexecuted proposal. An infeasible
        proposal is a diagnostic, not an available adoption alternative.
    """

    terminal_net_seconds: Annotated[Binary64, Field(ge=0, le=16)]
    gross_seconds: Annotated[Binary64, Field(ge=0, le=8)]
    reconfiguration_seconds: Annotated[Binary64, Field(ge=0, le=8)]
    coefficient_l1: Annotated[Binary64, Field(ge=0, le=2)]
    proposal_without_intervention_seconds: Annotated[Binary64, Field(ge=0, le=8)]


def modeled_components(
    receipt: DomainExecutionReceipt, durations: tuple[float, float], charge: float, movement: float
) -> ModeledComponents:
    """Compute the declared finite model, without observation or state mutation.

    Parameters
    ----------
    receipt : DomainExecutionReceipt
        Actual finite proposal and separate feasible executed choice.
    durations : tuple of float
        Complete fast/full synthetic outcome panel in modeled seconds.
    charge : float
        This decision's retained modeled reconfiguration seconds, not lambda.
    movement : float
        This decision's actual full coefficient L1.

    Returns
    -------
    ModeledComponents
        Separate raw, net, counterfactual, cost-model and coefficient quantities.

    Raises
    ------
    ValueError
        A shape/value lies outside this example's fixed two-plan model.
    """
    chosen, proposed = receipt.executed.action, receipt.executed.proposal.action
    if not isinstance(chosen, ChoiceAction) or not isinstance(proposed, ChoiceAction):
        raise ValueError("The complete CI example requires finite proposed and executed choices")
    gross = durations[PLANS.index(chosen.chosen)]
    return ModeledComponents(
        terminal_net_seconds=gross + charge,
        gross_seconds=gross,
        reconfiguration_seconds=charge,
        coefficient_l1=movement,
        proposal_without_intervention_seconds=durations[PLANS.index(proposed.chosen)],
    )


class DomainOwner:
    """Own the offline finite CI adapter and its explicit optional predictor.

    Parameters
    ----------
    recipe : Recipe
        Exact fixed BASE treatment and independent world. Online mode uses an
        expanding complete-past mean; accounting_off changes only modeled charges.

    Notes
    -----
    Complete DomainState retains forecasts, original feedback, pending receipts and
    own-path charges. Pure exports and serial calls are host obligations. No files,
    ambient clock, broker, physical CI action or global random generator are used."""

    def __init__(self, recipe: Recipe):
        """Create empty predictor, accounting and receipt state for the frozen recipe."""
        self.recipe = recipe
        self.binding = StateBinding(
            owner=ref(recipe.child.id + "-domain"),
            implementation=ADAPTER,
            state_codec=ref("constructed-ci-domain-state"),
            configuration_fingerprint=fingerprint(
                {
                    "BASE": recipe.base_fingerprint,
                    "world": recipe.world,
                    "predictor": recipe.mode == "online",
                    "accounting": "0 modeled seconds per full coefficient L1"
                    if recipe.mode == "accounting_off"
                    else "4 modeled seconds per full coefficient L1",
                    "intervention": INTERVENTION.model_dump(mode="json"),
                }
            ),
        )
        self._state = DomainState(schema_version=1, binding=self.binding)

    def export_state(self) -> DomainState:
        """Return all owned state without a clock read, observation or side effect."""
        return DomainState.model_validate(self._state)

    def restore_state(self, state: DomainState, /) -> None:
        """Replace only complete state under this exact fixed host law.

        Parameters
        ----------
        state : DomainState
            Original predictor/history, pending receipts and separate own-path charges.

        Raises
        ------
        ValueError
            Invalid/future fields, inconsistent history or a different binding. No
            migration, source read or fresh-learning fallback is inferred."""
        checked = DomainState.model_validate(state)
        if checked.binding != self.binding:
            raise ValueError("Domain snapshot belongs to another fixed law or independent world")
        self._state = checked

    def execution(self, key: ExecutionKey, /) -> DomainExecutionReceipt | None:
        """Inspect a retained logical receipt without charging twice.

        Parameters
        ----------
        key : ExecutionKey
            Exact candidate/world/repetition/task/opportunity identity.

        Returns
        -------
        DomainExecutionReceipt or None
            Original immutable receipt or explicit absence. No mutation or I/O."""
        return next((row for row in self._state.receipts if row.key == key), None)

    def _proposal(self, context: DomainExecutionContext, weights: tuple[float, ...]) -> Decision:
        """Compute a decision from actual coefficients and only earlier applied forecasts."""
        forecasts = (
            (self._state.sums[0] / self._state.count, self._state.sums[1] / self._state.count)
            if (self.recipe.mode == "online" and self._state.count)
            else (4.0, 4.0)
        )
        scores = tuple(p * (8.0 - forecast) for p, forecast in zip(weights, forecasts, strict=True))
        proposed = PLANS[0] if scores[0] > scores[1] else PLANS[1]
        space = ChoiceSpace(feasible_set=ref("finite-ci-plans"), members=PLANS)
        source = context.coefficient.opportunity.request
        timing = source.timing
        if timing is None:
            raise ValueError("The modeled domain action requires the original timed coefficient request")
        request = DecisionRequest(
            schema_version=1,
            task=context.plan.key.task,
            decision_id=context.plan.key.decision_id,
            round=source.round,
            space=space,
            objective=OBJECTIVE,
            member_inputs=forecast_inputs(forecasts),
            timing=timing,
        )
        proposal_action = ChoiceAction(schema_version=1, space=space, chosen=proposed)
        return Decision(
            schema_version=1,
            request=request,
            implementation=ADAPTER.reference,
            action=proposal_action,
            sampling=_sampling(proposal_action),
            state_owner="host",
        )

    def execute(self, context: DomainExecutionContext, /) -> DomainExecutionReceipt:
        """Map actual coefficients to a separate feasible offline CI action.

        Parameters
        ----------
        context : DomainExecutionContext
            Complete earlier coefficient producer and frozen law/source/prior-state
            binding. Only eligible past predictor statistics are used.

        Returns
        -------
        DomainExecutionReceipt
            Original proposal, finite feasibility intervention and executed choice.

        Raises
        ------
        ValueError
            Duplicate new execution, wrong action/profile/timing or infeasible input.

        Notes
        -----
        Appends one receipt and actual coefficient movement plus modeled reconfiguration
        seconds. It receives no future outcomes and performs no physical CI action.
        Joint durable acceptance belongs to the outer existing LocalStateStore."""
        if self.execution(context.plan.key) is not None:
            raise ValueError("Root driver must reconcile retained domain executions before a new call")
        action = context.coefficient.executed.action
        if not isinstance(action, MixtureAction):
            raise ValueError("CI model requires actual mixture coefficients")
        weights = tuple(row.value for row in action.weights)
        proposal = self._proposal(context, weights)
        source = proposal.request
        timing = source.timing
        if timing is None:
            raise ValueError("The modeled domain action requires its timed proposal")
        feasibility = _feasibility(self.recipe.world, source.round)
        if not isinstance(proposal.action, ChoiceAction):
            raise ValueError("The frozen CI proposal must remain an explicit finite choice")
        selected = proposal.action.chosen if proposal.action.chosen in feasibility.feasible_members else PLANS[0]
        executed_action = ChoiceAction(schema_version=1, space=proposal.action.space, chosen=selected)
        changed = selected != proposal.action.chosen
        executed = ExecutedDecision(
            schema_version=1,
            proposal=proposal,
            action=executed_action,
            mapping="projected" if changed else "unchanged",
            sampling_disposition="recomputed" if changed else "preserved",
            sampling=_sampling(executed_action),
        )
        receipt = DomainExecutionReceipt(
            schema_version=1,
            key=context.plan.key,
            adapter=context.plan.adapter,
            base_configuration_fingerprint=context.plan.base_configuration_fingerprint,
            data_fingerprint=context.plan.data_fingerprint,
            coefficient_record_fingerprint=context.coefficient.fingerprint,
            coefficient_execution_fingerprint=context.coefficient.executed.fingerprint,
            prior_domain_state_fingerprint=context.prior_domain_state_fingerprint,
            executed=executed,
            intervention=INTERVENTION,
            feasibility=feasibility,
            executed_at=timing.decision_at,
        )
        movement = fsum(abs(new - old) for new, old in zip(weights, self._state.last_coefficients, strict=True))
        self._state = DomainState.model_validate(
            self._state.model_copy(
                update={
                    "receipts": (*self._state.receipts, receipt),
                    "last_coefficients": weights,
                    "movements": (*self._state.movements, movement),
                    "charges": (
                        *self._state.charges,
                        (0.0 if self.recipe.mode == "accounting_off" else 4.0) * movement,
                    ),
                }
            )
        )
        return receipt

    def observe(self, step: StepEvidence, durations: tuple[float, float]) -> DomainState:
        """Apply one original eligible complete panel to the owned past-only state.

        Parameters
        ----------
        step : StepEvidence
            Separate original domain receipt and all five later outcome components.
        durations : tuple of float
            Original full fast/full panel, each in [0,8] modeled seconds.

        Returns
        -------
        DomainState
            Actual complete immutable successor, usable for the pre-ACK joint check.

        Raises
        ------
        ValueError
            Wrong producer, unknown application, inconsistent model quantities or a
            conflicting duplicate, including changed unchosen-expert feedback.

        Notes
        -----
        Preserves original expert history, updates only the predictor's count/sums and
        retains every component. Fixed prediction still uses its declared 4/4 forecasts;
        online prediction uses only these previously applied panels. No I/O occurs."""
        if self.execution(step.receipt.key) != step.receipt:
            raise ValueError("Observation differs from its original domain producer")
        ordinal = step.receipt.executed.proposal.request.round
        expected = modeled_components(
            step.receipt, durations, self._state.charges[ordinal], self._state.movements[ordinal]
        )
        values = expected.model_dump(mode="json")
        if {row.metric.name for row in step.components} != set(values) or any(
            row.value.value != values[row.metric.name] for row in step.components
        ):
            raise ValueError("Domain components differ from the declared original source/accounting model")
        old = next((row for row in self._state.observations if row.receipt.key == step.receipt.key), None)
        if old is not None:
            index = self._state.observations.index(old)
            if old.to_json() != step.to_json() or canonical_json(
                list(self._state.expert_history[index])
            ) != canonical_json(list(durations)):
                raise ValueError("Conflicting duplicate domain observation")
            return self._state
        if any(
            row.applied_at is None or row.timing.usable_at is None or row.applied_at < row.timing.usable_at
            for row in step.components
        ):
            raise ValueError("Predictor update requires known mature/received/applied observations")
        self._state = DomainState.model_validate(
            self._state.model_copy(
                update={
                    "observations": (*self._state.observations, step),
                    "expert_history": (*self._state.expert_history, durations),
                    "count": self._state.count + 1,
                    "sums": tuple(old_sum + value for old_sum, value in zip(self._state.sums, durations, strict=True)),
                }
            )
        )
        return self._state


def _sampling(action: ChoiceAction) -> SamplingRecord:
    """Bind deterministic action evidence, without inventing a propensity."""
    return SamplingRecord(
        mechanism="deterministic",
        mechanism_ref=ref("ci-argmax-then-feasibility"),
        status="not_required",
        action_fingerprint=action.fingerprint,
        probability=None,
    )


def forecast_inputs(forecasts: tuple[float, float]) -> tuple[MemberInputs, ...]:
    """Expose each actual host forecast with explicit units and meaning.

    Parameters
    ----------
    forecasts : tuple of float
        Fast/full predictions in [0,8] modeled seconds, not calibrated probabilities.

    Returns
    -------
    tuple of MemberInputs
        Complete versioned prediction-role scalars for the two finite CI plans."""
    spec = ScalarSpec(
        dimension=ref("predicted-modeled-plan-seconds"),
        role="prediction",
        minimum=0.0,
        maximum=8.0,
        semantics=UnitSemantics(unit="s", meaning="host-forecast-modeled-duration-v1"),
    )
    return tuple(
        MemberInputs(member=plan, values=(Scalar(spec=spec, value=value, status="observed"),))
        for plan, value in zip(PLANS, forecasts, strict=True)
    )


def _feasibility(world: str, ordinal: int):
    """Check known modeled resource requirements against the frozen case budget."""
    quantity = QuantityMeaning(
        spec=ScalarSpec(
            dimension=ref("required-modeled-ci-seconds"),
            role="resource",
            minimum=0.0,
            semantics=UnitSemantics(unit="s", meaning="constructed-ci-plan-resource-v1"),
        ),
        interpretation="resource",
        horizon=HORIZON,
    )
    maximum = 8.0 if ordinal == 1 or int(world.rsplit("-", 1)[1]) % 2 else 4.0
    constraint = ScalarConstraint(
        schema_version=1,
        reference=ref("ci-resource-budget"),
        lineage=lineage(),
        quantity=quantity,
        lower=None,
        upper=maximum,
    )
    return check_finite_feasibility(
        FiniteFeasibilityRequest(
            schema_version=1,
            universe=PLANS,
            universe_complete=True,
            constraints=(constraint,),
            alternatives=tuple(
                FiniteAlternative(member=member, values=(Scalar(spec=quantity.spec, value=value, status="observed"),))
                for member, value in zip(PLANS, (2.0, 6.0), strict=True)
            ),
        )
    )


def feedback(record, config, durations: tuple[float, float], at: datetime) -> ReplayFeedback:
    """Supply original complete synthetic expert losses after their receipt.

    Parameters
    ----------
    record : ReplayDecisionRecord
        Actual original unchanged coefficient execution and producer target.
    config : ReferenceConfig or ControlledConfig
        Exact producing loss specification and fixed duration/8 transform.
    durations : tuple of float
        Complete fast/full source outcomes in [0,8] modeled seconds.
    at : datetime
        Original common availability/maturity/first-receipt tick.

    Returns
    -------
    ReplayFeedback
        All original expert values with synthetic origins, separate from the
        executed plan's net outcome and physical-cost evidence.

    Raises
    ------
    ValueError
        Invalid producer, values or clocks. Construction performs no update or I/O."""
    rows = tuple(
        FeedbackRow(
            member=member,
            values=(
                OutcomeValue(
                    spec=config.loss_spec,
                    value=value / 8.0,
                    status="observed",
                    origin="synthetic",
                    transform=config.transform.reference,
                ),
            ),
        )
        for member, value in zip(MEMBERS, durations, strict=True)
    )
    timing = record.opportunity.request.timing
    batch = FeedbackBatch(
        schema_version=1,
        decision_fingerprint=record.executed.fingerprint,
        feedback_id="ci-complete-panel",
        revision_id="initial",
        supersedes=None,
        horizon=HORIZON,
        feedback=FullPanel(rows=rows),
        delay_rounds=0,
        timing=FeedbackTiming(event_at=timing.target_end, available_at=at, mature_at=at, received_at=at),
    )
    origins = tuple(
        ValueOriginEvidence(
            member=row.member,
            dimension=value.spec.dimension,
            value_fingerprint=fingerprint(value.model_dump(mode="json")),
            origin="synthetic",
            producer=GENERATOR,
        )
        for row in rows
        for value in row.values
    )
    return ReplayFeedback(
        schema_version=1, lineage=record.lineage, batch=batch, target=record.opportunity.target, origins=origins
    )
