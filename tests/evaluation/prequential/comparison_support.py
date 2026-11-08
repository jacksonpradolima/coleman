"""Independent whole-world comparison table and explicit terminal observations."""

from datetime import UTC, datetime, timedelta

from coleman.core.feedback import OutcomeValue
from coleman.core.timing import FeedbackTiming
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.evaluation.prequential.comparison import ComparisonPlan
from coleman.evaluation.prequential.conformance import ReferenceConformancePlan, check_reference_conformance
from coleman.evaluation.prequential.contracts import ComponentObservation, ConformanceTrace, StepEvidence
from coleman.evaluation.protected.contracts import ComparisonHypothesis, FrozenCandidate
from coleman.results.domain import Measurement
from tests.evaluation.design_support import changed, make_selection
from tests.evaluation.prequential.conformance_support import binding, key, lineage, receipt, ref
from tests.evaluation.prequential.conformance_support import metric as conformance_metric
from tests.evaluation.prequential.diagnostic_support import declarations, protected_panel


def comparison_declarations(
    *, units=2, threshold=0.125, unsupported=False, additional=False, seeds=(7,), physical=False
):
    """Freeze a terminal estimand, comparison rule and full family before outcomes."""
    design, selection = declarations(units=units, unsupported=unsupported)
    design = changed(design, seeds=seeds)
    if additional:
        other = changed(
            design.candidate("candidate"),
            candidate=FrozenCandidate.from_configuration("other", {"method": "independent-no-skill-baseline-copy"}),
        )
        extra = ComparisonHypothesis(hypothesis_id="other-baseline", candidate_id="other", baseline_id="baseline")
        design = changed(design, candidates=(*design.candidates, other), hypotheses=(*design.hypotheses, extra))
    design = changed(
        design,
        inference=changed(
            design.inference, practical_effect=design.inference.practical_effect.model_copy(update={"value": threshold})
        ),
    )
    if physical:
        cost = conformance_metric("cpu_seconds", role="physical_cost")
        cost = changed(
            cost,
            quantity=changed(
                cost.quantity, spec=changed(cost.quantity.spec, role="resource", minimum=0.0), interpretation="resource"
            ),
        )
        design = changed(design, metrics=(*design.metrics, cost))
    baseline = design.candidate("baseline").candidate
    opportunities = tuple(
        changed(
            key(decision_id="terminal-" + member.member_id, member=member.member_id),
            cell=changed(key().cell, candidate_id="baseline", member_id=member.member_id, seed=seed),
        )
        for member in design.protected_members
        for seed in design.seeds
    )
    conformance = ReferenceConformancePlan(
        schema_version=1,
        reference=ref("authored-terminal-conformance"),
        lineage=lineage(),
        adapter=binding("ci-adapter"),
        reference_recipe=binding("authored-terminal-reference"),
        baseline_candidate_id="baseline",
        base_configuration_fingerprint=baseline.configuration_fingerprint,
        data_fingerprint=design.snapshot.data_fingerprint,
        opportunities=opportunities,
        metrics=design.metrics,
    )
    plan = ComparisonPlan(
        schema_version=1,
        reference=ref("authored-comparison-rule"),
        lineage=lineage(),
        design_fingerprint=design.fingerprint,
        conformance_plan=conformance,
        baseline=baseline,
        required_hypotheses=tuple(row.hypothesis_id for row in design.hypotheses),
        cell_result_opportunities=opportunities,
    )
    selection = make_selection(design, actual_selected_at=selection.actual_selected_at, rule=plan.binding)
    return design, selection, plan


def hand_values(design, baseline, candidate):
    """Emit the authored complete table; seeds never add independent worlds."""
    return {
        (item.candidate.candidate_id, member.member_id, seed): candidate
        if item.candidate.candidate_id == "candidate"
        else baseline
        for item in design.candidates
        for member in design.protected_members
        for seed in design.seeds
    }


def conformance_for(plan, panel):
    """Bind each terminal primary to the actual original raw cell, not a sum."""
    rows = {(row.member_id, row.seed): row for row in panel.rows if row.candidate_id == "baseline"}
    steps = []
    decision_at = datetime(2020, 1, 1, tzinfo=UTC)
    for opportunity in plan.cell_result_opportunities:
        row = rows[opportunity.cell.member_id, opportunity.cell.seed]
        original = receipt(opportunity)
        request = original.executed.proposal.request
        timing = changed(
            request.timing,
            information_cutoff=decision_at,
            decision_at=decision_at,
            target_start=decision_at,
            target_end=decision_at + timedelta(seconds=10),
        )
        proposal = changed(original.executed.proposal, request=changed(request, timing=timing))
        execution = changed(
            original,
            executed=changed(original.executed, proposal=proposal),
            executed_at=decision_at,
            base_configuration_fingerprint=plan.baseline.configuration_fingerprint,
            data_fingerprint=panel.design.snapshot.data_fingerprint,
        )
        metric = panel.design.metric(panel.design.primary_metric)
        measurement = row.record.measurement(metric.name)
        available = row.record.available_at
        component = ComponentObservation(
            schema_version=1,
            execution_fingerprint=execution.fingerprint,
            metric=metric,
            value=OutcomeValue(
                spec=metric.quantity.spec,
                value=measurement.value,
                status=measurement.status,
                origin="synthetic",
                transform=None,
            ),
            original=row.evidence,
            measurement_name=metric.name,
            timing=FeedbackTiming(
                event_at=decision_at + timedelta(seconds=10),
                available_at=available,
                mature_at=available,
                received_at=available,
            ),
            applied_at=available,
            physical_group_id=None,
        )
        components = [component]
        for attempt in panel.ledger.attempts:
            if opportunity.cell not in attempt.outputs:
                continue
            for original_cost in attempt.costs:
                for cost_metric in plan.conformance_plan.metrics:
                    if cost_metric.role != "physical_cost":
                        continue
                    measured = original_cost.record.measurement(cost_metric.name)
                    cost_available = original_cost.record.available_at
                    components.append(
                        ComponentObservation(
                            schema_version=1,
                            execution_fingerprint=execution.fingerprint,
                            metric=cost_metric,
                            value=OutcomeValue(
                                spec=cost_metric.quantity.spec,
                                value=measured.value,
                                status=measured.status,
                                origin="realized",
                                transform=None,
                            ),
                            original=original_cost,
                            measurement_name=cost_metric.name,
                            timing=FeedbackTiming(
                                event_at=decision_at + timedelta(seconds=10),
                                available_at=cost_available,
                                mature_at=cost_available,
                                received_at=cost_available,
                            ),
                            applied_at=cost_available,
                            physical_group_id=attempt.group_id,
                        )
                    )
        steps.append(StepEvidence(schema_version=1, receipt=execution, components=tuple(components)))
    actual = ConformanceTrace(schema_version=1, recipe=plan.conformance_plan.adapter, steps=tuple(steps))
    reference = ConformanceTrace(schema_version=1, recipe=plan.conformance_plan.reference_recipe, steps=tuple(steps))
    return check_reference_conformance(plan.conformance_plan, reference, actual)


def physical_panel(design, selection, *, values=None):
    """Supply one authored realized-cost input shared by a multi-cell invocation.

    This tests provenance/accounting admission, not a measured CPU benchmark.
    """
    panel = protected_panel(design, selection, values=values)
    metric = design.metric("cpu_seconds")
    baseline = next(row for row in panel.rows if row.candidate_id == "baseline")
    cost_record = changed(
        baseline.record,
        namespace="authored.physical-cost-input",
        execution_id="one-baseline-invocation",
        decision_id="cpu-cost",
        available_at=panel.assembled_at,
        measurements=(
            Measurement(
                name=metric.name,
                value=2.0,
                unit=metric.quantity.spec.semantics.unit,
                meaning=metric.quantity.spec.semantics.meaning,
                direction=metric.direction,
                status="observed",
            ),
        ),
    )
    cost = OriginalRecordEvidence.from_record(cost_record)
    attempts = tuple(
        changed(attempt, group_id="baseline-physical-group", costs=(cost,))
        if attempt.candidate_id == "baseline"
        else attempt
        for attempt in panel.ledger.attempts
    )
    return changed(panel, ledger=changed(panel.ledger, attempts=attempts))
