"""Independent exact-prefix arithmetic, temporal and sampling counterexamples."""

from decimal import Decimal, Inexact, Rounded, localcontext
from fractions import Fraction
from itertools import permutations

import pytest

from coleman.core.actions import SlateAction
from coleman.core.decisions import Decision
from coleman.problems.tcpci import TcpciMapping, TcpciProblem, map_ranking, ranking_request
from tests.problems.tcpci._support import FIXTURES, data, problem, proposal


def test_actual_prefix_matches_independent_hand_golden():
    """Cost3/rem0 and ordered B,A are authored independently of mapper code."""
    value = problem()
    actual = map_ranking(value, proposal(value))
    expected = (FIXTURES / "mapping.json").read_text(encoding="utf-8").rstrip("\n")
    assert actual.to_json() == expected
    assert ranking_request(value).to_json() == value.semantic.request.to_json()


@pytest.mark.parametrize("order", ["".join(order) for order in permutations("ABC")])
@pytest.mark.parametrize("budget", ["0", "2.999999999999", "3", "7"])
def test_prefix_agrees_with_independent_rational_oracle(order, budget):
    """All six orders and inclusive boundaries use Fraction constants, no SUT helper."""
    raw = data()
    raw["budget"]["value"] = budget
    value = TcpciProblem.model_validate(raw)
    costs = {"A": Fraction(1), "B": Fraction(2), "C": Fraction(4)}
    expected = []
    used = Fraction(0)
    for name in order:
        candidate = used + costs[name]
        if candidate > Fraction(budget):
            break
        expected.append(name)
        used = candidate
    actual = map_ranking(value, proposal(value, order))
    assert isinstance(actual.executed.action, SlateAction)
    assert tuple(member.id for member in actual.executed.action.selected) == tuple(expected)
    assert Fraction(actual.estimated_cost.value) == used
    assert Fraction(actual.remaining_budget.value) == Fraction(budget) - used


def test_unaffordable_first_member_stops_instead_of_solving_knapsack():
    """C,A,B has no affordable prefix although A and B together fit."""
    value = problem()
    actual = map_ranking(value, proposal(value, "CAB"))
    assert isinstance(actual.executed.action, SlateAction)
    assert actual.executed.action.selected == ()
    assert actual.estimated_cost.value == Decimal(0)
    assert actual.remaining_budget.value == Decimal(3)


def test_zero_costs_empty_universe_and_decimal_context_have_explicit_behavior():
    """Exact tick arithmetic survives hostile Decimal precision/traps and empty input."""
    raw = data()
    raw["tests"][0]["duration_quote"]["value"] = "0.000000000001"
    raw["tests"][1]["duration_quote"]["value"] = "0"
    raw["budget"]["value"] = "0.000000000001"
    value = TcpciProblem.model_validate(raw)
    original = proposal(value, "ABC")
    before = map_ranking(value, original).to_json()
    with localcontext() as context:
        context.prec = 1
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        actual = map_ranking(value, original)
    assert actual.to_json() == before
    assert isinstance(actual.executed.action, SlateAction)
    assert tuple(member.id for member in actual.executed.action.selected) == ("A", "B")
    raw = data()
    raw["tests"] = []
    raw["semantic"]["request"]["space"]["members"] = []
    empty = TcpciProblem.model_validate(raw)
    result = map_ranking(empty, proposal(empty, ""))
    assert isinstance(result.executed.action, SlateAction)
    assert result.executed.action.selected == ()


@pytest.mark.parametrize("status", ["unknown", "unavailable", "not_applicable"])
def test_missing_quote_even_after_prefix_refuses_before_any_result(status):
    """Complete-input profile never silently uses zero or ignores a censored quote."""
    raw = data()
    raw["tests"][2]["duration_quote"] = None
    raw["tests"][2]["duration_status"] = status
    value = TcpciProblem.model_validate(raw)
    original = proposal(value)
    assert ranking_request(value) == value.semantic.request
    with pytest.raises(ValueError, match="complete observed duration"):
        map_ranking(value, original)


@pytest.mark.parametrize("sampled", [False, True])
def test_shape_change_always_invalidates_ranking_mass(sampled):
    """Deterministic postprocessing does not preserve or infer induced slate mass."""
    value = problem()
    original = proposal(value, sampled=sampled)
    actual = map_ranking(value, original)
    assert actual.executed.sampling_disposition == "invalidated"
    assert actual.executed.sampling.status == "unavailable"
    assert actual.executed.sampling.probability is None
    assert actual.executed.sampling.action_fingerprint == actual.executed.action.fingerprint
    assert actual.executed.sampling.action_fingerprint != original.sampling.action_fingerprint
    assert actual.executed.proposal.to_json() == original.to_json()


def _timed_data():
    raw = data()
    raw["semantic"]["request"]["timing"] = {
        "decision_at": "2026-01-01T09:01:00Z",
        "information_cutoff": "2026-01-01T09:00:00Z",
        "target_start": "2026-01-01T09:02:00Z",
        "target_end": "2026-01-01T09:03:00Z",
        "eligible_input_fingerprint": "sha256:" + "a" * 64,
        "mode": "future",
    }
    for test in raw["tests"]:
        test["quote_available_at"] = "2026-01-01T09:00:00Z"
    return raw


def test_original_quote_cutoff_and_unknown_timing_are_not_invented():
    """Known timely inputs map, unknown clocks remain representable but cannot map."""
    raw = _timed_data()
    value = TcpciProblem.model_validate(raw)
    assert map_ranking(value, proposal(value)).estimated_cost.value == Decimal(3)
    raw["tests"][0]["quote_available_at"] = None
    unknown = TcpciProblem.model_validate(raw)
    original = proposal(unknown)
    with pytest.raises(ValueError, match="known original quote availability"):
        map_ranking(unknown, original)


@pytest.mark.parametrize("location", ["quote", "rule", "objective"])
def test_known_future_inputs_or_declarations_reject(location):
    """Original semantic and planning definitions cannot follow the input cutoff."""
    raw = _timed_data()
    future = "2026-01-01T09:00:00.000001Z"
    if location == "quote":
        raw["tests"][0]["quote_available_at"] = future
    elif location == "rule":
        raw["mapping"]["lineage"]["declared_at"] = future
    else:
        raw["semantic"]["context"]["objective"]["lineage"]["declared_at"] = future
    with pytest.raises(ValueError, match="cutoff"):
        TcpciProblem.model_validate(raw)


def test_changed_request_cannot_be_relabelled_as_the_original_producer():
    """A valid different opportunity is refused without mutating its proposal."""
    value = problem()
    raw = data("mapping")["executed"]["proposal"]
    raw["request"]["decision_id"] = "another-opportunity"
    foreign = Decision.model_validate(raw)
    before = foreign.to_json()
    with pytest.raises(ValueError, match="exact original request"):
        map_ranking(value, foreign)
    assert foreign.to_json() == before


def test_reader_cannot_bless_an_altered_complete_problem():
    """Keeping an old result while changing its budget must fail rederivation."""
    raw = data("mapping")
    raw["problem"]["budget"]["value"] = "7"
    with pytest.raises(ValueError, match="prefix profile"):
        TcpciMapping.model_validate(raw)


@pytest.mark.parametrize(
    "change,expected_cost,expected_remaining",
    [("quote", "2.5", "0.5"), ("budget", "3", "0.5"), ("context", "3", "0")],
)
def test_core_action_binding_is_distinct_from_retained_domain_context(change, expected_cost, expected_remaining):
    """Equal core execution bytes do not bind changed costs or definition provenance."""
    original_problem = problem()
    original_proposal = proposal(original_problem)
    original = map_ranking(original_problem, original_proposal)
    raw = data()
    if change == "quote":
        raw["tests"][0]["duration_quote"]["value"] = "0.5"
    elif change == "budget":
        raw["budget"]["value"] = "3.5"
    else:
        # Bare objective Reference stays equal, full definition provenance changes.
        raw["semantic"]["context"]["objective"]["lineage"]["source"] = {
            "id": "another-declared-objective-source",
            "version": "1",
        }
    changed_problem = TcpciProblem.model_validate(raw)
    changed = map_ranking(changed_problem, original_proposal)
    assert changed.executed.to_json() == original.executed.to_json()
    assert changed.executed.fingerprint == original.executed.fingerprint
    assert changed.problem.fingerprint != original.problem.fingerprint
    assert changed.fingerprint != original.fingerprint
    assert changed.estimated_cost.value == Decimal(expected_cost)
    assert changed.remaining_budget.value == Decimal(expected_remaining)
