"""Real protected inference remains distinct from fallback, conformance and scope."""

import json
from pathlib import Path

import pytest

from coleman.evaluation.design.audit import ProtocolAudit
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.evaluation.design.reports import ProtocolReport
from coleman.evaluation.prequential.comparison import ComparisonDisposition, ComparisonPlan, assess_comparison
from coleman.evaluation.prequential.conformance import check_reference_conformance
from coleman.evaluation.prequential.diagnostics import DiagnosticAudit
from coleman.evaluation.protected.contracts import FrozenCandidate
from tests.evaluation.design_support import changed
from tests.evaluation.prequential.comparison_support import (
    comparison_declarations,
    conformance_for,
    hand_values,
    physical_panel,
)
from tests.evaluation.prequential.diagnostic_support import diagnostic_panel, protected_panel

FIXTURES = Path(__file__).parent / "fixtures" / "comparison-v1"


def test_independently_authored_complete_plan_wire_is_canonical():
    payload = (FIXTURES / "plan.json").read_text().strip()
    original = ComparisonPlan.from_json(payload)
    assert original.to_json() == payload
    assert ComparisonPlan.model_validate_json(payload).to_json() == payload


def test_public_preflight_accepts_exact_acyclic_inputs_without_changing_wire(tmp_path):
    design, selection, plan = comparison_declarations()
    before = plan.to_json(), design.to_json(), selection.to_json()
    audit = ProtocolAudit(tmp_path)
    assert plan.validate_for(design, selection) is None
    actual = plan.to_json(), design.to_json(), selection.to_json()
    assert actual == before
    assert audit.sources.history(design.scope).receipts == ()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("family", "required contrasts"),
        ("terminal", "terminal opportunities"),
        ("baseline", "deployable BASE baseline"),
        ("source", "another interpreted source"),
        ("metric", "exact declared primary metric"),
    ],
)
def test_public_preflight_refuses_bound_but_incompatible_metadata_before_outcomes(tmp_path, mismatch, message):
    design, selection, plan = comparison_declarations()
    if mismatch == "family":
        plan = changed(plan, required_hypotheses=("unfrozen-contrast",))
    elif mismatch == "terminal":
        plan = changed(plan, cell_result_opportunities=plan.cell_result_opportunities[:-1])
    elif mismatch == "baseline":
        baseline = FrozenCandidate.from_configuration("baseline", {"method": "another-baseline"})
        reference = changed(plan.conformance_plan, base_configuration_fingerprint=baseline.configuration_fingerprint)
        plan = changed(plan, baseline=baseline, conformance_plan=reference)
    elif mismatch == "source":
        reference = changed(plan.conformance_plan, data_fingerprint="sha256:" + "a" * 64)
        plan = changed(plan, conformance_plan=reference)
    else:
        metric = plan.conformance_plan.metrics[0]
        changed_metric = changed(
            metric, quantity=changed(metric.quantity, spec=changed(metric.quantity.spec, maximum=2.0))
        )
        plan = changed(plan, conformance_plan=changed(plan.conformance_plan, metrics=(changed_metric,)))
    # All inputs are individually valid and the actual selection deliberately
    # binds the bad plan, reproducing the late-refusal gap rather than failing
    # setup on an unrelated selection hash.
    selected = changed(selection, rule=plan.binding)
    audit = ProtocolAudit(tmp_path)
    calls = []

    def load():
        calls.append("outcome-loaded")
        return protected_panel(design, selected)

    def host_operation():
        plan.validate_for(design, selected)
        return audit.evaluate(design, selected, load, actor="preflight-regression", mode="produced")

    with pytest.raises(ValueError, match=message):
        host_operation()
    assert calls == []
    assert audit.sources.history(design.scope).receipts == ()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("part", "message"),
    [
        ("plan", "schema_version"),
        ("design", "hypotheses"),
        ("selection", "selected_candidate_ids"),
    ],
)
def test_public_preflight_revalidates_all_actual_objects(part, message):
    design, selection, plan = comparison_declarations()
    if part == "plan":
        plan = plan.model_copy(update={"schema_version": 2})
    elif part == "design":
        design = design.model_copy(update={"hypotheses": ()})
    else:
        selection = selection.model_copy(update={"selected_candidate_ids": ()})
    with pytest.raises(ValueError, match=message):
        plan.validate_for(design, selection)


def run_protocol(root, design, selection, values):
    """Use the actual reserved loader and existing complete bounded inference."""
    result = ProtocolAudit(root).evaluate(
        design,
        selection,
        lambda: protected_panel(design, selection, values=values),
        actor="synthetic-independent-worlds",
        mode="produced",
    )
    assert isinstance(result, ProtocolReport)
    return result


@pytest.mark.parametrize(
    ("baseline", "candidate", "threshold", "conclusion", "eligible"),
    [
        (0.0, 1.0, 0.125, "evidence_of_benefit", True),
        (1.0, 0.0, 0.125, "evidence_of_harm", False),
        (0.5, 0.5, 0.125, "inconclusive", False),
        (0.5, 0.5, 0.9, "evidence_of_practical_equivalence", False),
    ],
)
def test_authored_unit_differences_support_benefit_harm_tie_equivalence_without_adoption(
    tmp_path,
    baseline,
    candidate,
    threshold,
    conclusion,
    eligible,
):
    design, selection, plan = comparison_declarations(units=24, threshold=threshold)
    report = run_protocol(tmp_path, design, selection, hand_values(design, baseline, candidate))
    assert report.family.results[0].raw_mean_difference == candidate - baseline
    assert report.family.results[0].observed_pairs == 24
    assert report.family.results[0].conclusion == conclusion
    conformance = conformance_for(plan, report.panel)
    disposition = assess_comparison(plan, conformance, report)
    assert disposition.status == "qualified_comparison"
    assert bool(disposition.eligible_candidates) is eligible
    assert disposition.fallback == "baseline"
    assert disposition.baseline.model_dump_json() == design.candidate("baseline").candidate.model_dump_json()
    assert disposition.evidence_basis == "constructed_mechanism"
    assert disposition.source_admission_fingerprint is None
    restored = ComparisonDisposition.from_json(disposition.to_json())
    restored.validate_for(plan, conformance, report)
    assert restored.to_json() == disposition.to_json()


def test_two_worlds_are_insufficient_even_with_more_repetitions(tmp_path):
    design, selection, plan = comparison_declarations(seeds=(7, 11, 13, 17))
    report = run_protocol(tmp_path, design, selection, hand_values(design, 0.0, 1.0))
    conformance = conformance_for(plan, report.panel)
    result = assess_comparison(plan, conformance, report)
    assert len(report.panel.rows) == 16
    assert result.status == "insufficient_evidence"
    assert result.eligible_candidates == ()
    assert result.results[0].observed_pairs == 2


def test_supported_positive_after_real_inspection_remains_descriptive(tmp_path):
    design, selection, plan = comparison_declarations(units=24)
    DiagnosticAudit(tmp_path).evaluate(
        design, selection, lambda: diagnostic_panel(design, selection), actor="inspected", mode="produced"
    )
    report = run_protocol(tmp_path, design, selection, hand_values(design, 0.0, 1.0))
    result = assess_comparison(plan, conformance_for(plan, report.panel), report)
    assert result.status == "descriptive_only"
    assert result.eligible_candidates == ()
    assert not report.confirmatory_eligible


def test_unsupported_complete_diagnostics_keep_the_frozen_baseline(tmp_path):
    design, selection, plan = comparison_declarations(unsupported=True)
    evidence = DiagnosticAudit(tmp_path).evaluate(
        design, selection, lambda: diagnostic_panel(design, selection), actor="unsupported", mode="produced"
    )
    result = assess_comparison(plan, conformance_for(plan, evidence.panel), evidence)
    assert result.status == "unsupported_inference"
    assert "dependent_or_unknown_final_units" in result.reasons
    assert result.results == ()
    assert result.eligible_candidates == ()


def test_action_drift_blocks_comparison_even_with_identical_scores(tmp_path):
    design, selection, plan = comparison_declarations()
    report = run_protocol(tmp_path, design, selection, hand_values(design, 0.0, 1.0))
    conformance = conformance_for(plan, report.panel)
    reference = conformance.reference_trace
    step = reference.steps[0]
    # The actual boundary stays unchanged; only the declared reference sampling
    # action is altered to its other already feasible member, with exact core joins.
    from tests.evaluation.prequential.conformance_support import receipt

    other = receipt(step.receipt.key, selected="fast", proposed="fast")
    changed_receipt = changed(
        step.receipt,
        executed=changed(
            step.receipt.executed,
            proposal=changed(
                step.receipt.executed.proposal,
                action=other.executed.proposal.action,
                sampling=other.executed.proposal.sampling,
            ),
        ),
    )
    components = tuple(changed(item, execution_fingerprint=changed_receipt.fingerprint) for item in step.components)
    changed_step = changed(step, receipt=changed_receipt, components=components)
    wrong_reference = changed(reference, steps=(changed_step, *reference.steps[1:]))
    failed = check_reference_conformance(plan.conformance_plan, wrong_reference, conformance.actual_trace)
    result = assess_comparison(plan, failed, report)
    assert result.status == "failed_conformance"
    assert result.eligible_candidates == ()


def test_persisted_eligible_list_and_plan_rule_are_rejoined_not_trusted(tmp_path):
    design, selection, plan = comparison_declarations(units=24)
    report = run_protocol(tmp_path, design, selection, hand_values(design, 0.0, 1.0))
    conformance = conformance_for(plan, report.panel)
    original = assess_comparison(plan, conformance, report)
    altered = changed(original, eligible_candidates=())
    with pytest.raises(ValueError, match="rederived"):
        altered.validate_for(plan, conformance, report)
    changed_plan = changed(plan, fallback="abstain")
    with pytest.raises(ValueError, match="pre-access selected rule"):
        assess_comparison(changed_plan, conformance, report)
    assert original.eligible_candidates[0].candidate_id == "candidate"


def test_plan_is_acyclic_future_version_unknown_field_and_nonfinite_tolerance_reject():
    _design, _selection, plan = comparison_declarations()
    assert "selection_fingerprint" not in json.loads(plan.to_json())
    assert ComparisonPlan.from_json(plan.to_json()).binding == plan.binding
    for mutation in ({"schema_version": 2}, {"selection_fingerprint": "sha256:" + "f" * 64}):
        data = {**json.loads(plan.to_json()), **mutation}
        with pytest.raises(ValueError):
            ComparisonPlan.model_validate(data)
    data = json.loads(plan.to_json())
    data["conformance_plan"]["clock_tolerance_us"] = float("inf")
    with pytest.raises(ValueError):
        ComparisonPlan.model_validate(data)


def test_nested_legacy_configuration_is_bounded_before_its_parser():
    _design, _selection, plan = comparison_declarations()
    data = json.loads(plan.to_json())
    configuration = dict(data["baseline"], configuration_json="[" * 1000 + "0" + "]" * 1000)
    data["baseline"] = configuration
    with pytest.raises(ValueError, match="depth"):
        ComparisonPlan.model_validate(data)


def test_complete_multiple_family_survives_frozen_eligibility_subset(tmp_path):
    design, selection, plan = comparison_declarations(units=24, additional=True)
    plan = changed(plan, required_hypotheses=("candidate-baseline",))
    selection = changed(selection, rule=plan.binding)
    report = run_protocol(tmp_path, design, selection, hand_values(design, 0.0, 1.0))
    result = assess_comparison(plan, conformance_for(plan, report.panel), report)
    assert tuple(item.hypothesis_id for item in result.results) == ("candidate-baseline", "other-baseline")
    assert tuple(item.candidate_id for item in result.eligible_candidates) == ("candidate",)
    benefit, no_skill = result.results
    assert benefit.p_value is not None
    assert benefit.adjusted_p_value == min(1.0, 2.0 * benefit.p_value)
    assert no_skill.raw_mean_difference == 0.0
    assert no_skill.adjusted_p_value == 1.0
    assert no_skill.conclusion == "inconclusive"


def test_heterogeneous_no_skill_retains_all_signed_unit_outcomes(tmp_path):
    design, selection, plan = comparison_declarations(units=24)
    values = hand_values(design, 0.5, 0.5)
    for index, member in enumerate(design.protected_members):
        values["candidate", member.member_id, 7] = float(index % 2)
    report = run_protocol(tmp_path, design, selection, values)
    result = assess_comparison(plan, conformance_for(plan, report.panel), report)
    assert result.results[0].raw_mean_difference == 0.0
    assert result.results[0].observed_pairs == 24
    assert result.results[0].conclusion == "inconclusive"
    assert result.eligible_candidates == ()


def test_missing_unit_is_not_dropped_even_above_minimum_sample_count(tmp_path):
    design, selection, plan = comparison_declarations(units=24)
    values: dict[tuple[str, str, int], float | None] = dict(hand_values(design, 0.0, 1.0))
    member = design.protected_members[0]
    values["candidate", member.member_id, 7] = None
    report = run_protocol(tmp_path, design, selection, values)
    result = assess_comparison(plan, conformance_for(plan, report.panel), report)
    assert result.results[0].observed_pairs == 23
    assert len(result.results[0].independent_units) == 24
    assert result.status == "insufficient_evidence"
    assert result.eligible_candidates == ()


def test_realized_group_charge_joins_original_ledger_once_for_many_cells(tmp_path):
    design, selection, plan = comparison_declarations(physical=True)
    report = ProtocolAudit(tmp_path).evaluate(
        design, selection, lambda: physical_panel(design, selection), actor="authored-cost-admission", mode="produced"
    )
    assert isinstance(report, ProtocolReport)
    conformance = conformance_for(plan, report.panel)
    result = assess_comparison(plan, conformance, report)
    assert result.status == "insufficient_evidence"
    costs = [cost for attempt in report.panel.ledger.attempts for cost in attempt.costs]
    assert len(costs) == 1
    assert costs[0].record.measurement("cpu_seconds").value == 2.0
    assert len(conformance.actual_trace.steps) == 2
    assert (
        sum(
            component.metric.role == "physical_cost"
            for step in conformance.actual_trace.steps
            for component in step.components
        )
        == 2
    )

    steps = []
    for step in conformance.actual_trace.steps:
        components = tuple(
            changed(item, physical_group_id="another-invocation") if item.metric.role == "physical_cost" else item
            for item in step.components
        )
        steps.append(changed(step, components=components))
    actual = changed(conformance.actual_trace, steps=tuple(steps))
    reference = changed(conformance.reference_trace, steps=tuple(steps))
    wrong = check_reference_conformance(plan.conformance_plan, reference, actual)
    with pytest.raises(ValueError, match="physical group ledger"):
        assess_comparison(plan, wrong, report)

    omitted = tuple(
        changed(step, components=tuple(item for item in step.components if item.metric.role != "physical_cost"))
        for step in conformance.actual_trace.steps
    )
    incomplete = check_reference_conformance(
        plan.conformance_plan,
        changed(conformance.reference_trace, steps=omitted),
        changed(conformance.actual_trace, steps=omitted),
    )
    unknown = assess_comparison(plan, incomplete, report)
    assert unknown.status == "unknown_conformance"
    assert unknown.eligible_candidates == ()


@pytest.mark.parametrize("cost_first", [True, False], ids=["cost-before-outputs", "outputs-before-cost"])
def test_retry_group_retains_failed_charge_once_for_later_successful_outputs(tmp_path, cost_first):
    design, selection, plan = comparison_declarations(physical=True)
    supplied = []

    def load():
        original = physical_panel(design, selection)
        supplied.append(original)
        baseline = next(attempt for attempt in original.ledger.attempts if attempt.candidate_id == "baseline")
        failed = changed(baseline, attempt_id="a-failed" if cost_first else "z-failed", state="failed", outputs=())
        success = changed(baseline, attempt_id="z-success" if cost_first else "a-success", costs=())
        others = tuple(attempt for attempt in original.ledger.attempts if attempt.candidate_id != "baseline")
        return changed(original, ledger=changed(original.ledger, attempts=(failed, success, *others)))

    report = ProtocolAudit(tmp_path).evaluate(
        design, selection, load, actor="authored-retry-cost-admission", mode="produced"
    )
    assert isinstance(report, ProtocolReport)
    # The unchanged original records supply the full conformance trace. The
    # audited ledger separately retains the failed charge and successful retry.
    conformance = conformance_for(plan, supplied[0])
    assert conformance.status == "pass"
    before = report.to_json(), conformance.to_json()
    result = assess_comparison(plan, conformance, report)
    assert result.status == "insufficient_evidence"
    actual = report.to_json(), conformance.to_json()
    assert actual == before
    failed = next(attempt for attempt in report.panel.ledger.attempts if attempt.state == "failed")
    success = next(
        attempt
        for attempt in report.panel.ledger.attempts
        if attempt.candidate_id == "baseline" and attempt.state == "completed"
    )
    assert failed.outputs == ()
    assert len(failed.costs) == 1
    assert success.costs == ()
    assert len(success.outputs) == 2
    assert failed.group_id == success.group_id
    assert failed.costs[0].record.measurement("cpu_seconds").value == 2.0
    duplicate = changed(success, costs=failed.costs)
    others = tuple(attempt for attempt in report.panel.ledger.attempts if attempt.candidate_id != "baseline")
    with pytest.raises(ValueError, match="cannot be charged to several attempts"):
        changed(report.panel.ledger, attempts=(failed, duplicate, *others))
    assert report.to_json() == before[0]


@pytest.mark.parametrize("cost_candidate", ["baseline", "candidate"], ids=["no-group-outputs", "foreign-candidate"])
def test_retry_charge_cannot_borrow_outputs_from_another_group_or_candidate(tmp_path, cost_candidate):
    design, selection, plan = comparison_declarations(physical=True)
    supplied = []

    def load():
        original = physical_panel(design, selection)
        supplied.append(original)
        baseline = next(attempt for attempt in original.ledger.attempts if attempt.candidate_id == "baseline")
        failed = changed(
            baseline,
            attempt_id="failed-no-output",
            candidate_id=cost_candidate,
            group_id="another-physical-group",
            state="failed",
            outputs=(),
        )
        success = changed(baseline, attempt_id="success-with-outputs", costs=())
        others = tuple(attempt for attempt in original.ledger.attempts if attempt.candidate_id != "baseline")
        return changed(original, ledger=changed(original.ledger, attempts=(failed, success, *others)))

    report = ProtocolAudit(tmp_path).evaluate(
        design, selection, load, actor="authored-unrelated-retry", mode="produced"
    )
    assert isinstance(report, ProtocolReport)
    original = conformance_for(plan, supplied[0])
    steps = tuple(
        changed(
            step,
            components=tuple(
                changed(component, physical_group_id="another-physical-group")
                if component.metric.role == "physical_cost"
                else component
                for component in step.components
            ),
        )
        for step in original.actual_trace.steps
    )
    conformance = check_reference_conformance(
        plan.conformance_plan,
        changed(original.reference_trace, steps=steps),
        changed(original.actual_trace, steps=steps),
    )
    assert conformance.status == "pass"
    before = report.to_json(), conformance.to_json()
    with pytest.raises(ValueError, match="physical group ledger"):
        assess_comparison(plan, conformance, report)
    actual = report.to_json(), conformance.to_json()
    assert actual == before


def _terminal_primary_inputs(root, primary_value, shadow_value, candidate_value, units):
    """Keep a named raw estimand and a distinct same-unit quantity in each original."""
    design, selection, plan = comparison_declarations(units=units)
    values: dict[tuple[str, str, int], float | None] = dict(hand_values(design, 0.0, candidate_value))
    for member in design.protected_members:
        values["baseline", member.member_id, 7] = primary_value
    primary_name = design.primary_metric
    shadow_name = "shadow-" + primary_name

    def load():
        panel = protected_panel(design, selection, values=values)
        rows = []
        for row in panel.rows:
            if row.candidate_id == "baseline":
                primary = row.record.measurement(primary_name)
                shadow = type(primary).model_validate(
                    primary.model_copy(update={"name": shadow_name, "value": shadow_value, "status": "observed"})
                )
                record = changed(row.record, measurements=(*row.record.measurements, shadow))
                row = changed(row, evidence=OriginalRecordEvidence.from_record(record))
            rows.append(row)
        return changed(panel, rows=tuple(rows))

    report = ProtocolAudit(root).evaluate(design, selection, load, actor="named-terminal-primary", mode="produced")
    assert isinstance(report, ProtocolReport)
    named = conformance_for(plan, report.panel)
    steps = []
    for step in named.actual_trace.steps:
        components = []
        for component in step.components:
            shadow = component.original.record.measurement(shadow_name)
            value = type(component.value).model_validate(
                component.value.model_copy(update={"value": shadow.value, "status": shadow.status})
            )
            components.append(
                type(component).model_validate(
                    component.model_copy(update={"measurement_name": shadow_name, "value": value})
                )
            )
        steps.append(changed(step, components=tuple(components)))
    selected_shadow = check_reference_conformance(
        plan.conformance_plan,
        changed(named.reference_trace, steps=tuple(steps)),
        changed(named.actual_trace, steps=tuple(steps)),
    )
    return plan, report, named, selected_shadow


@pytest.mark.parametrize(
    ("primary", "shadow", "candidate", "units"),
    [(0.0, 1.0, 1.0, 24), (1.0, 0.0, 0.0, 2), (None, 0.0, 1.0, 2), (0.0, 0.0, 1.0, 2)],
    ids=[
        "observed-zero-primary",
        "observed-nonzero-primary",
        "unavailable-primary-is-not-zero",
        "equal-value-alias-still-not-primary",
    ],
)
def test_terminal_primary_refuses_different_same_unit_measurement(tmp_path, primary, shadow, candidate, units):
    plan, report, _, conformance = _terminal_primary_inputs(tmp_path, primary, shadow, candidate, units)
    # Generic semantic labels may select another measurement. That valid generic
    # contract and reference agreement cannot replace the protected named estimand.
    assert conformance.status == "pass"
    before = report.to_json(), conformance.to_json()
    with pytest.raises(ValueError, match="named raw primary measurement"):
        assess_comparison(plan, conformance, report)
    actual = report.to_json(), conformance.to_json()
    assert actual == before


@pytest.mark.parametrize(
    ("primary", "expected"),
    [(0.0, "insufficient_evidence"), (None, "unknown_conformance")],
    ids=["observed-zero", "unavailable-remains-unknown"],
)
def test_named_terminal_primary_with_secondary_measurement_preserves_original_status(tmp_path, primary, expected):
    plan, report, conformance, _ = _terminal_primary_inputs(tmp_path, primary, 1.0, 1.0, 2)
    before = report.to_json(), conformance.to_json()
    result = assess_comparison(plan, conformance, report)
    assert result.status == expected
    assert result.eligible_candidates == ()
    actual = report.to_json(), conformance.to_json()
    assert actual == before
