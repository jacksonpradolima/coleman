"""Independently authored synthetic CI plans; no historical or physical claim."""

from datetime import UTC, datetime, timedelta
from typing import Literal

from coleman.core.actions import ChoiceAction, ChoiceSpace
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import Origin, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec, ScalarStatus
from coleman.core.sampling import SamplingRecord
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.core.units import UnitSemantics
from coleman.evaluation.design.contracts import MetricPlan
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.evaluation.design.selection import CellIdentity
from coleman.evaluation.prequential.conformance import ReferenceConformancePlan
from coleman.evaluation.prequential.contracts import (
    ComponentObservation,
    ConformanceTrace,
    DomainExecutionReceipt,
    ExecutionKey,
    StepEvidence,
)
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage, QuantityMeaning
from coleman.objectives.feasibility import (
    FiniteAlternative,
    FiniteFeasibilityRequest,
    ScalarConstraint,
    check_finite_feasibility,
)
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance

NOW = datetime(2026, 11, 8, 12, tzinfo=UTC)
CONFIG = "sha256:" + "b" * 64
DATA = "sha256:" + "d" * 64


def ref(name):
    return Reference(id=name, version="1")


def binding(name):
    return DefinitionBinding(reference=ref(name), definition_fingerprint="sha256:" + "a" * 64)


def lineage():
    return DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)


def changed(original, **updates):
    return type(original).model_validate(original.model_copy(update=updates))


def key(decision_id="build-1", member="repository-a", candidate="baseline"):
    return ExecutionKey(
        cell=CellIdentity(candidate_id=candidate, member_id=member, seed=0),
        task=ref("ci-execution-plans"),
        decision_id=decision_id,
    )


def metric(
    name: str = "net_duration",
    role: Literal[
        "raw_outcome", "surrogate", "executed_objective", "feasibility", "physical_cost", "uncertainty", "diagnostic"
    ] = "executed_objective",
    unit: str = "s",
) -> MetricPlan:
    quantity = QuantityMeaning(
        spec=ScalarSpec(
            dimension=ref(name),
            semantics=UnitSemantics(unit=unit, meaning=name + "-v1"),
            role="objective",
            minimum=-20.0,
            maximum=20.0,
        ),
        interpretation="raw_outcome",
        horizon=ref("one-build"),
    )
    return MetricPlan(name=name, quantity=quantity, direction="minimize", role=role, estimand=binding("terminal-build"))


def receipt(opportunity=None, *, selected="fast", proposed="full", upper=3.0):
    opportunity = key() if opportunity is None else opportunity
    space = ChoiceSpace(feasible_set=ref("ci-plans"), members=(ref("fast"), ref("full")))
    request = DecisionRequest(
        schema_version=1,
        task=opportunity.task,
        decision_id=opportunity.decision_id,
        round=0,
        space=space,
        objective=ref("test-duration"),
        timing=DecisionTiming(
            decision_at=NOW,
            information_cutoff=NOW,
            target_start=NOW,
            target_end=NOW + timedelta(seconds=10),
            eligible_input_fingerprint=DATA,
            mode="future",
        ),
    )
    proposal_action = ChoiceAction(schema_version=1, space=space, chosen=ref(proposed))
    action = ChoiceAction(schema_version=1, space=space, chosen=ref(selected))

    def sampling(content):
        return SamplingRecord(
            mechanism="deterministic",
            mechanism_ref=ref("finite-choice"),
            status="not_required",
            action_fingerprint=content.fingerprint,
            probability=None,
        )

    proposal = Decision(
        schema_version=1,
        request=request,
        implementation=ref("ci-adapter"),
        action=proposal_action,
        sampling=sampling(proposal_action),
        state_owner="host",
    )
    executed = ExecutedDecision(
        schema_version=1,
        proposal=proposal,
        action=action,
        mapping="projected" if selected != proposed else "unchanged",
        sampling_disposition="recomputed" if selected != proposed else "preserved",
        sampling=sampling(action),
    )
    quantity = QuantityMeaning(
        spec=ScalarSpec(
            dimension=ref("duration"),
            semantics=UnitSemantics(unit="s", meaning="predicted-duration-v1"),
            role="resource",
            minimum=0.0,
        ),
        interpretation="resource",
        horizon=ref("one-build"),
    )
    constraint = ScalarConstraint(
        schema_version=1, reference=ref("duration-limit"), lineage=lineage(), quantity=quantity, lower=None, upper=upper
    )
    feasibility = check_finite_feasibility(
        FiniteFeasibilityRequest(
            schema_version=1,
            universe=space.members,
            universe_complete=True,
            constraints=(constraint,),
            alternatives=tuple(
                FiniteAlternative(
                    member=ref(name), values=(Scalar(spec=quantity.spec, value=value, status="observed"),)
                )
                for name, value in (("fast", 2.0), ("full", 6.0))
            ),
        )
    )
    return DomainExecutionReceipt(
        schema_version=1,
        key=opportunity,
        adapter=binding("ci-adapter"),
        base_configuration_fingerprint=CONFIG,
        data_fingerprint=DATA,
        coefficient_record_fingerprint="sha256:" + "c" * 64,
        coefficient_execution_fingerprint="sha256:" + "e" * 64,
        prior_domain_state_fingerprint="sha256:" + "f" * 64,
        executed=executed,
        intervention=binding("bounded-duration"),
        feasibility=feasibility,
        executed_at=NOW,
    )


def observation(
    execution: DomainExecutionReceipt | None = None,
    *,
    value: float | None = 2.0,
    status: ScalarStatus = "observed",
    origin: Origin = "synthetic",
    quantity: MetricPlan | None = None,
    received: bool = True,
) -> ComponentObservation:
    execution = receipt() if execution is None else execution
    quantity = metric() if quantity is None else quantity
    timing = FeedbackTiming(
        event_at=NOW + timedelta(seconds=10),
        available_at=NOW + timedelta(seconds=11),
        mature_at=NOW + timedelta(seconds=12),
        received_at=NOW + timedelta(seconds=13) if received else None,
    )
    record = DomainRecord(
        schema_version=1,
        namespace="constructed.conformance",
        run_id=execution.key.decision_id,
        execution_id="constructed-baseline-seed-0",
        decision_id=execution.key.cell.member_id,
        data_fingerprint=execution.data_fingerprint,
        available_at=timing.available_at,
        provenance=ResultProvenance(
            producer="hand-authored-ci-fixture",
            producer_version="1",
            spec_fingerprint=execution.base_configuration_fingerprint,
            evaluation_fingerprint="sha256:" + "1" * 64,
        ),
        measurements=(
            Measurement(
                name=quantity.name,
                value=value,
                unit=quantity.quantity.spec.semantics.unit,
                meaning=quantity.quantity.spec.semantics.meaning,
                direction=quantity.direction,
                status=status,
            ),
        ),
    )
    return ComponentObservation(
        schema_version=1,
        execution_fingerprint=execution.fingerprint,
        metric=quantity,
        value=OutcomeValue(spec=quantity.quantity.spec, value=value, status=status, origin=origin, transform=None),
        original=OriginalRecordEvidence.from_record(record),
        measurement_name=quantity.name,
        timing=timing,
        applied_at=NOW + timedelta(seconds=14) if received else None,
        physical_group_id="physical-group-1" if quantity.role == "physical_cost" and origin == "realized" else None,
    )


def trace(execution=None, *, components=None):
    execution = receipt() if execution is None else execution
    components = (observation(execution),) if components is None else components
    step = StepEvidence(schema_version=1, receipt=execution, components=components)
    return ConformanceTrace(schema_version=1, recipe=binding("ci-adapter"), steps=(step,))


def plan(**updates):
    original = ReferenceConformancePlan(
        schema_version=1,
        reference=ref("ci-reference-conformance"),
        lineage=lineage(),
        adapter=binding("ci-adapter"),
        reference_recipe=binding("ci-adapter"),
        baseline_candidate_id="baseline",
        base_configuration_fingerprint=CONFIG,
        data_fingerprint=DATA,
        opportunities=(key(),),
        metrics=(metric(),),
    )
    return changed(original, **updates) if updates else original
