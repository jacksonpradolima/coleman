"""Real example arithmetic, original-feedback joins and complete continuation."""

from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from itertools import combinations
from math import exp, isclose, log
from pathlib import Path

import pytest

from coleman.core.actions import RankingAction, SlateAction
from coleman.core.feedback import AbsentFeedback, ComponentFeedback, FullPanel
from coleman.core.primitives import canonical_json
from coleman.learning.direct import DirectMixture
from coleman.learning.reference import Hedge
from coleman.problems.tcpci import map_ranking

SOURCE = Path(__file__).parents[2] / "examples/tcpci/ci.py"
SPEC = importlib.util.spec_from_file_location("tcpci_public_example_tests", SOURCE)
assert SPEC is not None
assert SPEC.loader is not None
EXAMPLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXAMPLE
SPEC.loader.exec_module(EXAMPLE)


@pytest.fixture(name="example", scope="module")
def _example():
    """Run the actual learner, mapping and coverage solver once for these oracles."""
    return EXAMPLE.run_example()


def test_complete_loss_vectors_use_the_original_coefficient_producer(example):
    """Domain selected verdicts and complete expert-loss panels never share joins."""
    expected = ((Fraction(1, 3), Fraction(1, 3)), (Fraction(0), Fraction(2, 3)), (Fraction(1, 3), Fraction(0)))
    for ordinal, (row, losses) in enumerate(zip(example.rounds, expected, strict=True)):
        assert row.expert_feedback.decision_fingerprint == row.coefficient_producer.fingerprint
        assert row.selected_feedback.decision_fingerprint == row.mapping.executed.fingerprint
        assert row.expert_feedback.decision_fingerprint != row.selected_feedback.decision_fingerprint
        assert isinstance(row.expert_feedback.feedback, FullPanel)
        actual = tuple(item.values[0].value for item in row.expert_feedback.feedback.rows)
        assert actual == tuple(float(value) for value in losses)
        assert row.after_state.update_count == ordinal + 1
        assert row.after_state.pending == ()


def test_selected_feedback_contains_only_the_actual_prefix_members(example):
    """The first prefix stops at B despite C fitting the remaining modeled second."""
    expected = (("A",), ("A",), ("C", "A"))
    for row, selected in zip(example.rounds, expected, strict=True):
        assert isinstance(row.mapping.executed.action, SlateAction)
        assert tuple(member.id for member in row.mapping.executed.action.selected) == selected
        assert isinstance(row.selected_feedback.feedback, ComponentFeedback)
        assert {value.member.id for value in row.selected_feedback.feedback.rows} == set(selected)
        assert row.selected_feedback.horizon == row.mapping.problem.mapping.quantity.horizon
        assert row.mapping.executed.sampling_disposition == "invalidated"
        assert row.mapping.executed.sampling.probability is None
    assert example.rounds[0].mapping.estimated_cost.value == 2
    assert example.rounds[0].mapping.remaining_budget.value == 1


def test_final_actual_hedge_weights_match_independent_exponential_formula(example):
    """A hand total of2/3 versus1 losses determines the actual final coefficients."""
    masses = (exp(-log(2) * float(Fraction(2, 3))), exp(-log(2)))
    expected = tuple(value / sum(masses) for value in masses)
    actual = example.rounds[-1].after_state.method.payload.weights
    for coefficient, wanted in zip(actual, expected, strict=True):
        assert isclose(coefficient.value, wanted, rel_tol=0.0, abs_tol=1e-14)


def test_complete_original_state_restores_the_next_actual_producer_and_update(example):
    """Restore one accepted prefix and consume the same original next panel."""
    model = EXAMPLE.model
    learner = Hedge(model.configuration(), owner=model.ref("ci-example-hedge-owner"))
    bridge = DirectMixture(
        learner, task=model.ref("ci-expert-coefficients"), objective=model.OBJECTIVE.reference, horizon=model.HORIZON
    )
    original = example.rounds[0].after_state
    bridge.restore_state(type(original).from_json(original.to_json()))
    next_row = example.rounds[1]
    proposal = bridge.decide(next_row.coefficient_producer.proposal.request)
    assert proposal.to_json() == next_row.coefficient_producer.proposal.to_json()
    bridge.update(next_row.expert_feedback)
    assert bridge.export_state().to_json() == next_row.after_state.to_json()
    with pytest.raises(ValueError, match="original pending producer"):
        bridge.update(next_row.selected_feedback)


def test_empty_slate_is_explicit_absence_and_no_failures_are_real_zero_losses(example):
    """A missing observation is not supplied by the zero-valued complete source."""
    problem = EXAMPLE.model.problem(0, budget="0")
    mapping = map_ranking(problem, EXAMPLE.rank(problem, example.rounds[0].coefficient_producer.proposal))
    feedback = EXAMPLE.selected_feedback(mapping, EXAMPLE.source_panel((0, 0, 0)))
    assert isinstance(feedback.feedback, AbsentFeedback)
    assert mapping.estimated_cost.value == 0
    panel = EXAMPLE.expert_feedback(example.rounds[0].coefficient_producer, problem, EXAMPLE.source_panel((0, 0, 0)))
    assert isinstance(panel.feedback, FullPanel)
    assert tuple(row.values[0].value for row in panel.feedback.rows) == (0.0, 0.0)


def test_coverage_choice_has_an_independent_exhaustive_set_oracle(example):
    """Coverage optimizes a different objective and does not manufacture a prefix."""
    candidates = {"A": (2, {"x", "y"}), "B": (2, {"y", "z"}), "C": (1, {"z"})}
    weights = {"x": Fraction(1, 2), "y": Fraction(3, 10), "z": Fraction(1, 5)}
    rows = []
    for count in range(4):
        for selected in combinations(sorted(candidates), count):
            cost = sum(candidates[name][0] for name in selected)
            if cost <= 3:
                covered = set().union(*(candidates[name][1] for name in selected))
                rows.append((sum(weights[name] for name in covered), selected, cost))
    expected = min(rows, key=lambda value: (-value[0], value[1]))
    actual = (example.coverage.objective.value, example.coverage.selected_ids, example.coverage.cost.value)
    assert actual == expected
    assert example.coverage.status == "optimal"
    assert example.coverage.proof_prefixes
    assert example.coverage.selected_ids != tuple(
        member.id for member in example.rounds[0].mapping.executed.action.selected
    )


def test_example_receipts_are_deterministic_and_planning_inputs_contain_no_labels(example):
    """Unknown clocks/physical time stay unknown; all authored outputs stay synthetic."""
    assert canonical_json(EXAMPLE.full_report(EXAMPLE.run_example())) == canonical_json(EXAMPLE.full_report(example))
    assert EXAMPLE.summary(example)["physical_elapsed_seconds"] is None
    evidence = EXAMPLE.full_report(example)["evidence"]
    for row, retained in zip(example.rounds, evidence, strict=True):
        assert retained["mapping_fingerprint"] == row.mapping.fingerprint
        assert retained["executed_fingerprint"] == row.mapping.executed.fingerprint
        assert retained["mapping"] == row.mapping.model_dump(mode="json")
        assert isinstance(row.mapping.executed.proposal.action, RankingAction)
        assert row.mapping.problem.semantic.request.member_inputs == ()
        assert row.mapping.problem.semantic.request.timing is None
        assert all(test.quote_available_at is None for test in row.mapping.problem.tests)
        assert all(value.origin == "synthetic" for item in row.full_source.rows for value in item.values)


@pytest.mark.parametrize("verdicts", ((0, 1), (0, True, 0), (0, -1, 0), (0, 2, 0)))
def test_authored_full_source_rejects_missing_coerced_or_nonbinary_verdicts(verdicts):
    """A complete binary source cannot be inferred from malformed supplied rows."""
    with pytest.raises(ValueError, match="three binary integer"):
        EXAMPLE.source_panel(verdicts)
