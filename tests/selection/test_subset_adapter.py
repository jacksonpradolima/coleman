"""Real subsets reuse the solver and match an independent exhaustive enumerator."""

from decimal import Decimal
from itertools import combinations

import pytest

from coleman.core.actions import AbstainAction, SubsetAction
from coleman.core.quantities import Quantity
from coleman.core.units import UnitSemantics
from coleman.selection.contracts import CoverageCandidate, CoverageConstraints, CoverageOutcome, CoverageProblem
from coleman.selection.subsets import CoverageSelection, coverage_space, select_coverage_subset

COST = UnitSemantics(unit="seconds", meaning="modeled-additive-ci-runtime-v1")
VALUE = UnitSemantics(unit="faults", meaning="distinct-weighted-faults-v1")


def problem(budget=2, required=()):
    def q(value):
        return Quantity(value=Decimal(str(value)), semantics=COST)

    return CoverageProblem(
        schema_version=1,
        data_fingerprint="sha256:" + "0" * 64,
        objective_semantics=VALUE,
        candidates=(
            CoverageCandidate(candidate_id="A", cost=q(1), covers=("x", "y")),
            CoverageCandidate(candidate_id="B", cost=q(1), covers=("y", "z")),
            CoverageCandidate(candidate_id="C", cost=q(2), covers=("z",)),
        ),
        outcomes=tuple(
            CoverageOutcome(outcome_id=key, weight=Quantity(value=Decimal("1"), semantics=VALUE)) for key in "xyz"
        ),
        constraints=CoverageConstraints(budget=q(budget), required_ids=required),
    )


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4])
def test_actual_coverage_subset_matches_independent_enumeration_and_retains_proof(budget):
    source = problem(budget)
    result = select_coverage_subset(source, method="branch_bound_v1")
    subsets = [rows for count in range(4) for rows in combinations(source.candidates, count)]
    feasible = [rows for rows in subsets if sum(row.cost.value for row in rows) <= budget]
    maximum = max(len({outcome for row in rows for outcome in row.covers}) for rows in feasible)
    assert isinstance(result.action, SubsetAction)
    assert result.result.objective is not None
    assert result.result.selected_ids is not None
    assert result.result.status == "optimal" and result.result.objective.value == maximum
    assert {row.id for row in result.action.selected} == set(result.result.selected_ids)
    assert result.result.problem == source
    assert CoverageSelection.from_json(result.to_json()) == result


def test_empty_subset_is_real_action_while_infeasible_is_abstention():
    empty = select_coverage_subset(problem(0)).action
    assert isinstance(empty, SubsetAction)
    assert empty.selected == ()
    result = select_coverage_subset(problem(0, ("A",)))
    assert isinstance(result.action, AbstainAction)
    assert not result.result.feasible and result.result.status == "infeasible"
    raw = result.model_dump(mode="json")
    raw["action"]["reason"] = "invented"
    with pytest.raises(ValueError, match="differs from"):
        CoverageSelection.model_validate(raw)


def test_restricted_scope_is_preserved_and_not_claimed_as_full():
    source = CoverageProblem.model_validate(problem().model_copy(update={"search_ids": ("C",)}))
    result = select_coverage_subset(source, method="branch_bound_v1")
    assert isinstance(result.action, SubsetAction)
    assert result.result.objective is not None
    assert result.result.scope == "restricted"
    assert tuple(row.id for row in result.action.space.members) == ("C",)
    assert result.result.objective.value == 1
    changed = problem().model_copy(update={"candidates": ()})
    with pytest.raises(ValueError):
        coverage_space(changed)


def test_oversized_declared_scope_is_rejected_without_truncating_candidates():
    source = problem(0)
    source = CoverageProblem.model_validate(
        source.model_copy(
            update={
                "candidates": tuple(
                    CoverageCandidate(
                        candidate_id=f"test-{index:05d}",
                        cost=Quantity(value=Decimal(0), semantics=COST),
                        covers=(),
                    )
                    for index in range(4097)
                ),
                "search_ids": None,
            }
        )
    )
    with pytest.raises(ValueError, match="exceeds 4096"):
        select_coverage_subset(source)
    assert len(source.candidates) == 4097


def test_old_coverage_artifacts_and_checkpoint_keep_their_original_readers_and_meaning():
    from pathlib import Path

    from coleman.selection.results import CoverageCheckpoint, CoverageResult

    directory = Path(__file__).parents[1] / "fixtures/coverage"
    old = CoverageResult.from_json((directory / "optimal-v1.json").read_text())
    checkpoint = CoverageCheckpoint.from_json((directory / "checkpoint-v1.json").read_text())
    original = old.to_json()
    converted = select_coverage_subset(old.problem, method="branch_bound_v1", checkpoint=checkpoint)
    assert converted.result.problem == old.problem
    assert converted.result.objective == old.objective
    assert converted.result.selected_ids == old.selected_ids
    assert CoverageResult.from_json(original).to_json() == original
    assert isinstance(converted.action, SubsetAction)


def test_cancelled_search_keeps_existing_incumbent_status_without_false_optimality():
    converted = select_coverage_subset(problem(), method="branch_bound_v1", cancelled=lambda: True)
    assert converted.result.status != "optimal"
    if converted.result.feasible:
        assert isinstance(converted.action, SubsetAction)
        assert tuple(row.id for row in converted.action.selected) == converted.result.selected_ids
