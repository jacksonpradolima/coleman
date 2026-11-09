"""Actual shipped learner/solver composition with independently authored answers."""

import math
from decimal import Decimal, localcontext
from itertools import combinations

import pytest

from coleman.core.actions import MixtureAction, MixtureSpace, SlateAction, Weight
from coleman.core.decisions import DecisionRequest, ExecutedDecision
from coleman.core.feedback import ComponentFeedback, FeedbackBatch, FeedbackRow, FullPanel, OutcomeValue
from coleman.core.quantities import Quantity
from coleman.core.units import UnitSemantics
from coleman.learning.contracts import LearningUpdate, ReferenceConfig, ReferencePayload
from coleman.learning.direct import DirectMixture
from coleman.learning.reference import Hedge
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS, InitialPrior
from coleman.problems.tcpci import TcpciProblem, map_ranking
from coleman.selection import CoverageCandidate, CoverageConstraints, CoverageOutcome, CoverageProblem, SearchLimits
from coleman.selection.solver import solve_coverage
from tests.problems.tcpci._support import data, problem, proposal, ref


def _owned_case(value: TcpciProblem) -> tuple[Hedge, DirectMixture[ReferencePayload], ExecutedDecision]:
    """Prepare a real independently owned method and its original pending producer."""
    identity = value.semantic.context.transforms[0]
    loss = value.semantic.context.objective.terms[0].quantity.spec
    config = ReferenceConfig(
        schema_version=1,
        reference=ref("actual-constant-hedge"),
        lineage=identity.lineage,
        profile="hedge-constant",
        prior=InitialPrior(
            schema_version=1,
            reference=ref("positive-three-test-prior"),
            lineage=identity.lineage,
            weights=tuple(Weight(member=test.member, value=1 / 3) for test in value.tests),
            profile="binary64-simplex-v1",
            semantics=COEFFICIENT_SEMANTICS,
        ),
        loss_spec=loss,
        transform=identity.binding,
        eta=math.log(2),
    )
    method = Hedge(config, owner=ref("independent-hedge-owner"))
    bridge = DirectMixture(
        method,
        task=value.semantic.request.task,
        objective=value.semantic.request.objective,
        horizon=value.mapping.quantity.horizon,
    )
    mixture_request = DecisionRequest(
        schema_version=1,
        task=value.semantic.request.task,
        decision_id="original-mixture-0",
        round=0,
        space=MixtureSpace(
            feasible_set=ref("original-test-mixture"),
            members=tuple(test.member for test in value.tests),
            profile="binary64-simplex-v1",
        ),
        objective=value.semantic.request.objective,
    )
    original = bridge.decide(mixture_request)
    assert isinstance(original.action, MixtureAction)
    executed = ExecutedDecision(
        schema_version=1,
        proposal=original,
        action=original.action,
        mapping="unchanged",
        sampling_disposition="preserved",
        sampling=original.sampling,
    )
    return method, bridge, executed


def _scalar_answer(initial: tuple[float, ...]) -> tuple[float, ...]:
    """Independently substitute actual binary64 inputs at100-digit precision."""
    with localcontext() as context:
        context.prec = 100
        rate = Decimal.from_float(math.log(2))
        scores = tuple(
            Decimal.from_float(prior) * (-rate * loss).exp() for prior, loss in zip(initial, (0, 1, 0), strict=True)
        )
        total = sum(scores, Decimal(0))
        return tuple(float(score / total) for score in scores)


def test_actual_hedge_full_panel_keeps_original_producer_separate_from_slate():
    """Real partial slate feedback cannot impersonate the original mixture panel."""
    value = problem()
    method, bridge, executed = _owned_case(value)
    # This separate named coefficient-rank rule uses stable member IDs for ties.
    order = "".join(item.member.id for item in sorted(method.weights, key=lambda item: (-item.value, item.member.id)))
    mapped = map_ranking(value, proposal(value, order))
    assert isinstance(mapped.executed.action, SlateAction)
    assert tuple(member.id for member in mapped.executed.action.selected) == ("A", "B")
    rows = tuple(
        FeedbackRow(
            member=test.member,
            values=(
                OutcomeValue(
                    spec=method.loss_spec,
                    value=number,
                    status="observed",
                    origin="synthetic",
                    transform=method.transform.reference,
                ),
            ),
        )
        for test, number in zip(value.tests, (0.0, 1.0, 0.0), strict=True)
    )
    censored = FeedbackBatch(
        schema_version=1,
        decision_fingerprint=mapped.executed.fingerprint,
        feedback_id="selected-only",
        revision_id="root",
        supersedes=None,
        horizon=value.mapping.quantity.horizon,
        feedback=ComponentFeedback(rows=rows[:2], completeness="complete"),
    )
    before = bridge.export_state().to_json()
    with pytest.raises(ValueError, match="original pending producer"):
        bridge.update(censored)
    assert bridge.export_state().to_json() == before
    batch = FeedbackBatch(
        schema_version=1,
        decision_fingerprint=executed.fingerprint,
        feedback_id="explicit-complete-synthetic-oracle",
        revision_id="root",
        supersedes=None,
        horizon=value.mapping.quantity.horizon,
        feedback=FullPanel(rows=rows),
    )
    initial = tuple(item.value for item in method.weights)
    result = bridge.update(batch)
    assert isinstance(result, LearningUpdate)
    assert result.status == "accepted"
    assert method.update_count == 1
    assert tuple(item.value for item in method.weights) == pytest.approx(_scalar_answer(initial), abs=3e-16)
    assert tuple(item.value for item in method.weights) == pytest.approx((0.4, 0.2, 0.4), abs=3e-16)
    next_order = "".join(
        item.member.id for item in sorted(method.weights, key=lambda item: (-item.value, item.member.id))
    )
    assert next_order == "ACB"
    raw = data()
    raw["semantic"]["request"]["decision_id"] = "next-domain-opportunity"
    raw["semantic"]["request"]["round"] = 1
    following = TcpciProblem.model_validate(raw)
    mapped = map_ranking(following, proposal(following, next_order))
    assert isinstance(mapped.executed.action, SlateAction)
    assert tuple(member.id for member in mapped.executed.action.selected) == ("A",)


def test_existing_exact_coverage_solver_matches_eight_subset_hand_oracle():
    """Actual solver's lexicographic tie chooses AB, not the cheaper equal-score B."""
    value = problem()
    coverage = {"A": ("x",), "B": ("x", "y"), "C": ("z",)}
    costs = {"A": 1, "B": 2, "C": 4}
    weights = {"x": 1, "y": 2, "z": 4}
    semantics = value.budget.semantics
    credit = UnitSemantics(unit="1", meaning="synthetic-covered-fault-credit-v1")
    request = CoverageProblem(
        schema_version=1,
        data_fingerprint=value.fingerprint,
        objective_semantics=credit,
        candidates=tuple(
            CoverageCandidate(
                candidate_id=name,
                cost=Quantity(value=Decimal(costs[name]), semantics=semantics),
                covers=coverage[name],
            )
            for name in "ABC"
        ),
        outcomes=tuple(
            CoverageOutcome(
                outcome_id=name,
                weight=Quantity(
                    value=Decimal(weights[name]),
                    semantics=credit,
                ),
            )
            for name in "xyz"
        ),
        constraints=CoverageConstraints(budget=value.budget),
    )
    feasible = []
    for count in range(4):
        for subset in combinations("ABC", count):
            cost = sum(costs[name] for name in subset)
            if cost <= 3:
                score = sum(weights[name] for name in {key for member in subset for key in coverage[member]})
                feasible.append((subset, cost, score))
    assert feasible == [((), 0, 0), (("A",), 1, 1), (("B",), 2, 3), (("A", "B"), 3, 3)]
    expected = min(feasible, key=lambda item: (-item[2], item[0]))
    actual = solve_coverage(request, method="branch_bound_v1", limits=SearchLimits(deadline_seconds=None))
    assert actual.status == "optimal"
    assert actual.selected_ids == expected[0]
    assert actual.cost is not None
    assert actual.objective is not None
    assert actual.cost.value == Decimal(expected[1])
    assert actual.objective.value == Decimal(expected[2])
    # Coverage is a set; the independently supplied complete ranking orders it.
    mapped = map_ranking(value, proposal(value, "BAC"))
    assert isinstance(mapped.executed.action, SlateAction)
    assert tuple(member.id for member in mapped.executed.action.selected) == ("B", "A")
