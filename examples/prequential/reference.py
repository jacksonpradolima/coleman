"""Independent CI reference arithmetic and explicitly descriptive stage outputs."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from coleman.core.actions import ChoiceAction, ChoiceSpace
from coleman.core.decisions import DecisionRequest
from coleman.core.sampling import SamplingRecord
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.evaluation.prequential.contracts import (
    ConformanceTrace,
    StepEvidence,
)

from . import model
from .profile import REFERENCE_RECIPE, STAGE_PAIRS, TABLE, WORLDS, ObservationInput, SourceTape, primary_component

MODEL = model


def _choice_space(step: StepEvidence) -> ChoiceSpace:
    """Read the explicitly qualified finite domain universe without a type cast."""
    space = step.receipt.executed.action.space
    if not isinstance(space, ChoiceSpace):
        raise ValueError("The CI example requires its declared finite choice space")
    return space


def _add_cost(value: int | float | None) -> int | float:
    """Change a known fault-fixture value without imputing an unavailable cost."""
    if value is None:
        raise ValueError("The deliberate cost fault requires an observed component")
    return value + 0.25


def reference_trace(tape: SourceTape, *, drift=None) -> ConformanceTrace:
    """Calculate the independent fixed reference from retained original inputs.

    Parameters
    ----------
    tape : SourceTape
        Original complete authored panels and captured logical/application clocks.
        The actual trace being tested is not an expected-answer source.
    drift : {'action', 'cost', 'time'} or None, optional
        Deliberate failed-reference diagnostic, separate from immutable inputs.

    Returns
    -------
    ConformanceTrace
        Full fixed-half direct reference with independently calculated actions,
        nominal resource constraints and five modeled quantities.

    Notes
    -----
    Pure arithmetic; original clocks are explicit replay inputs. Neither OS clocks,
    physical costs nor historical unknown availability are reconstructed."""
    originals = tuple(
        original
        for tick in tape.ticks
        for original in tick.observations
        if original.step.receipt.key.cell.candidate_id == "fixed"
    )
    return reference_from_originals(originals, drift=drift)


def reference_from_originals(originals: tuple[ObservationInput, ...], *, drift=None) -> ConformanceTrace:
    """Calculate the independent fixed-half CI reference from original inputs.

    Parameters
    ----------
    originals : tuple of ObservationInput
        Original raw expert panels, logical opportunities and captured clocks.
        Actual output traces being checked are not an expected-answer source.
    drift : {'action', 'cost', 'time'} or None, optional
        Deliberate wrong-reference diagnostic; no source observations change.

    Returns
    -------
    ConformanceTrace
        Fixed-half/frozen-forecast direct reference. Completeness is checked by
        the separately frozen opportunity catalogue, not assumed from a subset.

    Notes
    -----
    Uses a separate literal resource-budget table and direct source arithmetic.
    Original clocks are replay inputs, not reconstructed historical facts.
    """
    steps = [_reference_step(original) for original in originals]
    if drift is not None:
        index = (
            next(i for i, step in enumerate(steps) if step.receipt.executed.mapping == "projected")
            if drift == "action"
            else 0
        )
        steps[index] = _divergent_reference(steps[index], drift)
    return ConformanceTrace(schema_version=1, recipe=REFERENCE_RECIPE, steps=tuple(steps))


def _reference_execution(original_input: ObservationInput):
    """Independently calculate fixed-half proposals and the frozen finite intervention."""
    step = original_input.step
    receipt = step.receipt
    ordinal = receipt.executed.proposal.request.round
    world = WORLDS.index(receipt.key.cell.member_id)
    # Independent fixed-half/frozen-four reference always proposes full.
    can_run_full = ordinal == 1 or world % 2 == 1
    chosen = MODEL.PLANS[1 if can_run_full else 0]
    proposed = MODEL.PLANS[1]
    proposal_action = ChoiceAction(schema_version=1, space=_choice_space(step), chosen=proposed)
    action = ChoiceAction(schema_version=1, space=_choice_space(step), chosen=chosen)
    expected_request = DecisionRequest.model_validate(
        receipt.executed.proposal.request.model_copy(
            update={
                "task": MODEL.DOMAIN_TASK,
                "decision_id": "modeled-build-" + str(ordinal),
                "objective": MODEL.OBJECTIVE,
                "member_inputs": MODEL.forecast_inputs((4.0, 4.0)),
            }
        )
    )
    proposal = type(receipt.executed.proposal).model_validate(
        receipt.executed.proposal.model_copy(
            update={
                "action": proposal_action,
                "sampling": _reference_sampling(proposal_action),
                "request": expected_request,
            }
        )
    )
    return type(receipt.executed).model_validate(
        receipt.executed.model_copy(
            update={
                "proposal": proposal,
                "action": action,
                "sampling": _reference_sampling(action),
                "mapping": "unchanged" if chosen == proposed else "projected",
                "sampling_disposition": "preserved" if chosen == proposed else "recomputed",
            }
        )
    )


def _reference_step(original_input: ObservationInput) -> StepEvidence:
    """Join independent model quantities to the original immutable source clocks."""
    step = original_input.step
    receipt = step.receipt
    ordinal = receipt.executed.proposal.request.round
    world = WORLDS.index(receipt.key.cell.member_id)
    can_run_full = ordinal == 1 or world % 2 == 1
    executed = _reference_execution(original_input)
    expected_receipt = type(receipt).model_validate(
        receipt.model_copy(
            update={
                "executed": executed,
                "feasibility": _reference_feasibility(world, ordinal),
                "intervention": MODEL.INTERVENTION,
            }
        )
    )
    gross = original_input.expert_seconds[1 if can_run_full else 0]
    values = {
        "terminal_net_seconds": gross,
        "gross_seconds": gross,
        "reconfiguration_seconds": 0.0,
        "coefficient_l1": 0.0,
        "proposal_without_intervention_seconds": original_input.expert_seconds[1],
    }
    original = primary_component(step).original.record
    measurements = tuple(
        type(row).model_validate(row.model_copy(update={"value": values[row.name]})) for row in original.measurements
    )
    expected_record = type(original).model_validate(original.model_copy(update={"measurements": measurements}))
    expected_components = tuple(
        type(component).model_validate(
            component.model_copy(
                update={
                    "execution_fingerprint": expected_receipt.fingerprint,
                    "value": type(component.value).model_validate(
                        component.value.model_copy(update={"value": values[component.metric.name]})
                    ),
                    "original": OriginalRecordEvidence.from_record(expected_record),
                }
            )
        )
        for component in step.components
    )
    return StepEvidence(schema_version=1, receipt=expected_receipt, components=expected_components)


def _reference_sampling(action: ChoiceAction) -> SamplingRecord:
    """Independently declare the fixed reference's deterministic choice semantics."""
    return SamplingRecord(
        mechanism="deterministic",
        mechanism_ref=MODEL.ref("ci-argmax-then-feasibility"),
        status="not_required",
        action_fingerprint=action.fingerprint,
        probability=None,
    )


def _reference_feasibility(world: int, ordinal: int):
    """Construct expected nominal resource evidence from a separate literal table."""
    budgets = ((4.0, 8.0, 4.0), (8.0, 8.0, 8.0), (4.0, 8.0, 4.0), (8.0, 8.0, 8.0))
    quantity = MODEL.QuantityMeaning(
        spec=MODEL.ScalarSpec(
            dimension=MODEL.ref("required-modeled-ci-seconds"),
            role="resource",
            minimum=0.0,
            semantics=MODEL.UnitSemantics(unit="s", meaning="constructed-ci-plan-resource-v1"),
        ),
        interpretation="resource",
        horizon=MODEL.HORIZON,
    )
    constraint = MODEL.ScalarConstraint(
        schema_version=1,
        reference=MODEL.ref("ci-resource-budget"),
        lineage=MODEL.lineage(),
        quantity=quantity,
        lower=None,
        upper=budgets[world][ordinal],
    )
    request = MODEL.FiniteFeasibilityRequest(
        schema_version=1,
        universe=MODEL.PLANS,
        universe_complete=True,
        alternatives=(
            MODEL.FiniteAlternative(
                member=MODEL.PLANS[0], values=(MODEL.Scalar(spec=quantity.spec, value=2.0, status="observed"),)
            ),
            MODEL.FiniteAlternative(
                member=MODEL.PLANS[1], values=(MODEL.Scalar(spec=quantity.spec, value=6.0, status="observed"),)
            ),
        ),
        constraints=(constraint,),
    )
    result = MODEL.check_finite_feasibility(request)
    expected_members = MODEL.PLANS if budgets[world][ordinal] == 8.0 else (MODEL.PLANS[0],)
    if result.feasible_members != expected_members:
        raise RuntimeError("Reference finite checker differs from the independent two-plan arithmetic")
    return result


def _divergent_reference(step: StepEvidence, drift: str) -> StepEvidence:
    """Keep schema-valid action, cost or clock drift visible to conformance."""
    receipt = step.receipt
    components = list(step.components)
    if drift == "action":
        proposal_action = ChoiceAction(schema_version=1, space=_choice_space(step), chosen=MODEL.PLANS[0])
        proposal = type(receipt.executed.proposal).model_validate(
            receipt.executed.proposal.model_copy(
                update={"action": proposal_action, "sampling": _reference_sampling(proposal_action)}
            )
        )
        executed = type(receipt.executed).model_validate(
            receipt.executed.model_copy(
                update={
                    "proposal": proposal,
                    "mapping": "unchanged",
                    "sampling_disposition": "preserved",
                    "sampling": _reference_sampling(proposal_action),
                }
            )
        )
        receipt = type(receipt).model_validate(receipt.model_copy(update={"executed": executed}))
    elif drift == "cost":
        record = primary_component(step).original.record
        measurements = tuple(
            type(row).model_validate(row.model_copy(update={"value": _add_cost(row.value)}))
            if row.name == "reconfiguration_seconds"
            else row
            for row in record.measurements
        )
        record = type(record).model_validate(record.model_copy(update={"measurements": measurements}))
        originals = OriginalRecordEvidence.from_record(record)
        components = [
            type(row).model_validate(
                row.model_copy(
                    update={
                        "original": originals,
                        "value": type(row.value).model_validate(
                            row.value.model_copy(update={"value": _add_cost(row.value.value)})
                        )
                        if row.metric.name == "reconfiguration_seconds"
                        else row.value,
                    }
                )
            )
            for row in components
        ]
    elif drift == "time":
        # Unknown receipt would narrow the claim; this is a known wrong application time.
        index = next(i for i, row in enumerate(components) if row.metric.name == "terminal_net_seconds")
        applied = components[index].applied_at
        if applied is None:
            raise ValueError("The deliberate clock fault requires a known original application")
        components[index] = type(components[index]).model_validate(
            components[index].model_copy(update={"applied_at": applied + timedelta(microseconds=1)})
        )
    else:
        raise ValueError("Unknown explicit reference-drift profile")
    components = tuple(
        type(row).model_validate(row.model_copy(update={"execution_fingerprint": receipt.fingerprint}))
        for row in components
    )
    return StepEvidence(schema_version=1, receipt=receipt, components=components)


def trajectory_diagnostics(traces: tuple[ConformanceTrace, ...]) -> dict[str, Any]:
    """Expose every prefix and terminal modeled stage without hiding earlier actions.

    Parameters
    ----------
    traces : tuple of ConformanceTrace
        Original complete candidate trajectories. All five quantities must be
        observed in this authored example; missing values are not imputed.

    Returns
    -------
    dict
        Human JSON presentation with each terminal quantity, every prefix and
        separately named descriptive path totals and frozen matched contrasts.

    Notes
    -----
    The unprojected proposal is a same-source model diagnostic, never an executed
    infeasible action or observed causal effect. No inference or adoption occurs."""
    terminal = {}
    trajectories = {}
    for trace in traces:
        mode = trace.steps[0].receipt.key.cell.candidate_id
        terminal[mode] = []
        trajectories[mode] = []
        for world in WORLDS:
            steps = sorted(
                (step for step in trace.steps if step.receipt.key.cell.member_id == world),
                key=lambda step: step.receipt.executed.proposal.request.round,
            )
            if not steps:
                continue
            panels = [_diagnostic_panel(step) for step in steps]
            terminal[mode].append({"world": world, **panels[-1]})
            trajectories[mode].append(
                {
                    "world": world,
                    "decisions": panels,
                    "path_modeled_gross_seconds": sum(panel["measurements"]["gross_seconds"] for panel in panels),
                    "path_modeled_reconfiguration_seconds": sum(
                        panel["measurements"]["reconfiguration_seconds"] for panel in panels
                    ),
                    "path_coefficient_l1": sum(panel["measurements"]["coefficient_l1"] for panel in panels),
                }
            )
    return {
        "terminal_by_candidate": terminal,
        "complete_trajectories": trajectories,
        "matched_stage_contrasts": _stage_contrasts(terminal),
        "stage_contrast_inference": (
            "Descriptive matched differences only; "
            "the complete protected family compares all eight candidates with fixed."
        ),
        "stage_diagnostic_scope": (
            "Same-source modeled contrast; infeasible proposals are not adoptable or executed; "
            "no observed causal effect."
        ),
        "path_total_scope": "Descriptive prefix totals; the protected primary remains the named terminal decision.",
    }


def _diagnostic_panel(step: StepEvidence) -> dict[str, Any]:
    """Expose one original modeled stage contrast, refusing missing quantities."""
    values: dict[str, int | float] = {}
    for component in step.components:
        if component.value.value is None or component.value.status != "observed":
            raise ValueError("The complete authored diagnostic cannot impute unavailable measurements")
        values[component.metric.name] = component.value.value
    executed = step.receipt.executed
    proposed = executed.proposal.action
    if not isinstance(proposed, ChoiceAction) or not isinstance(executed.action, ChoiceAction):
        raise ValueError("The declared CI diagnostic requires explicit proposed and executed finite choices")
    return {
        "round": executed.proposal.request.round,
        "proposed_plan": proposed.chosen.id,
        "executed_plan": executed.action.chosen.id,
        "proposal_feasible": proposed.chosen in step.receipt.feasibility.feasible_members,
        "measurements": values,
        "intervention_modeled_seconds_difference": values["gross_seconds"]
        - values["proposal_without_intervention_seconds"],
    }


def _stage_contrasts(terminal: dict) -> list[dict[str, Any]]:
    """Calculate only frozen descriptive matched differences, never new tests."""
    contrasts = []
    for stage, baseline, candidate in STAGE_PAIRS:
        if baseline not in terminal or candidate not in terminal:
            continue
        before = {row["world"]: row for row in terminal[baseline]}
        after = {row["world"]: row for row in terminal[candidate]}
        contrasts.append(
            {
                "stage": stage,
                "baseline": baseline,
                "candidate": candidate,
                "terminal_net_seconds_difference_by_world": {
                    world: after[world]["measurements"]["terminal_net_seconds"]
                    - before[world]["measurements"]["terminal_net_seconds"]
                    for world in WORLDS
                    if world in before and world in after
                },
            }
        )
    return contrasts


def human_summary(report, conformance, disposition, traces) -> dict[str, Any]:
    """Present source, conformance, family conclusions and fallback as separate axes.

    Parameters
    ----------
    report : ProtocolReport
        Original complete audited results after authorized disclosure.
    conformance : ReferenceConformanceReport
        Exact independent action/resource/component/time comparison.
    disposition : ComparisonDisposition
        Re-derived frozen fallback and explicitly scoped eligibility.
    traces : tuple of ConformanceTrace
        Original full candidate paths, including modeled costs and raw source units.

    Returns
    -------
    dict
        Human JSON showing all outcomes and the unchanged conservative fallback.

    Notes
    -----
    Presentation only: no source read, experiment, learner transition, physical
    resource measurement, fee charge or automatic candidate deployment."""
    means = {
        mode: sum(
            row.record.measurement(report.panel.design.primary_metric).value
            for row in report.panel.rows
            if row.candidate_id == mode
        )
        / len(WORLDS)
        for mode in MODEL.MODES
    }
    interventions = sum(step.receipt.executed.mapping != "unchanged" for trace in traces for step in trace.steps)
    return {
        "evidence_scope": "constructed modeled CI mechanics; already exposed public inputs",
        "input_table_seconds": TABLE,
        "nominal_plan_reservation_seconds": {"fast": 2.0, "full": 6.0},
        "reservation_semantics": (
            "Feasibility uses nominal plan reservations, not a bound on modeled observed duration."
        ),
        "candidate_modes": MODEL.MODES,
        "independent_declared_worlds": len(WORLDS),
        "seeds_per_world": 1,
        "decisions_per_world": 3,
        "raw_cells": len(report.panel.rows),
        "primary_estimand": "Named terminal decision only; modeled net seconds, minimize",
        "mean_terminal_modeled_seconds": means,
        "nonidentity_feasibility_interventions": interventions,
        "source_reasons": report.source_reasons,
        "confirmatory_eligible": report.confirmatory_eligible,
        "conformance": conformance.status,
        "comparison_status": disposition.status,
        "eligible_candidates": [candidate.candidate_id for candidate in disposition.eligible_candidates],
        "retained_fallback": disposition.baseline.candidate_id,
        "automatic_adoption": False,
        "family_conclusions": {row.hypothesis_id: row.conclusion for row in report.family.results},
        "physical_cost": "unavailable: no physical resource measurements supplied",
        "executed_book_turnover": "not_applicable: this is a CI plan model",
        "coefficient_movement": "Full L1 retained separately in each complete domain checkpoint",
        "penalty": "lambda=0.05 coefficient units for declared penalized modes; never paid cost",
        "required_later_comparison": (
            "Actual interval-cadence domain-book accounting remains a separate downstream qualification"
        ),
        **trajectory_diagnostics(traces),
    }
