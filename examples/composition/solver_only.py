"""Call the existing finite coverage solver without learner/runtime/artifacts."""

import json
from decimal import Decimal
from itertools import combinations

from coleman.core.primitives import fingerprint
from coleman.selection import (
    CoverageCandidate,
    CoverageConstraints,
    CoverageOutcome,
    CoverageProblem,
    Quantity,
    SearchLimits,
    UnitSemantics,
    solve_coverage,
)


def problem() -> CoverageProblem:
    """Declare three indivisible synthetic CI suites with overlapping coverage."""
    seconds = UnitSemantics(unit="s", meaning="modeled-serial-suite-duration-v1")
    credit = UnitSemantics(unit="1", meaning="synthetic-covered-fault-credit-v1")
    return CoverageProblem(
        schema_version=1,
        data_fingerprint=fingerprint({"fixture": "synthetic-three-suite-union-v1"}),
        objective_semantics=credit,
        candidates=tuple(
            CoverageCandidate(candidate_id=name, cost=Quantity(value=Decimal(cost), semantics=seconds), covers=covers)
            for name, cost, covers in (("A", "2", ("x", "y")), ("B", "2", ("y", "z")), ("C", "1", ("z",)))
        ),
        outcomes=tuple(
            CoverageOutcome(outcome_id=name, weight=Quantity(value=Decimal(weight), semantics=credit))
            for name, weight in (("x", ".5"), ("y", ".3"), ("z", ".2"))
        ),
        constraints=CoverageConstraints(budget=Quantity(value=Decimal(3), semantics=seconds)),
    )


def independent_answer(request: CoverageProblem) -> tuple[tuple[str, ...], Decimal, Decimal]:
    """Enumerate this tiny fixture independently of production solver helpers."""
    candidates = request.candidates
    if (
        len(candidates) > 3
        or request.constraints.required_ids
        or request.constraints.maximum_selected is not None
        or request.search_scope != "full"
    ):
        raise ValueError("This independent example oracle admits only three full-scope unconstrained suites")
    weights = {row.outcome_id: row.weight.value for row in request.outcomes}
    feasible = []
    for count in range(len(candidates) + 1):
        for subset in combinations(candidates, count):
            cost = sum((row.cost.value for row in subset), Decimal(0))
            if cost <= request.constraints.budget.value:
                covered = {outcome for row in subset for outcome in row.covers}
                score = sum((weights[name] for name in sorted(covered)), Decimal(0))
                feasible.append((tuple(row.candidate_id for row in subset), cost, score))
    return min(feasible, key=lambda row: (-row[2], row[0]))


def main() -> None:
    """Compare actual bounded exact-scope solving with the exhaustive fixture."""
    request = problem()
    result = solve_coverage(request, method="branch_bound_v1", limits=SearchLimits(deadline_seconds=None))
    expected = independent_answer(request)
    assert result.selected_ids == expected[0]
    assert result.cost is not None
    assert result.objective is not None
    assert (result.cost.value, result.objective.value) == expected[1:]
    print(
        json.dumps(
            {"selected": list(expected[0]), "cost_seconds": str(expected[1]), "covered_credit": str(expected[2])}
        )
    )


if __name__ == "__main__":
    main()
