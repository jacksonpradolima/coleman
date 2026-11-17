"""Read manually authored v1 complete inputs/output and independent arithmetic."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from coleman.optimization.continuous import solve_continuous
from coleman.optimization.continuous_contracts import ContinuousProblem, ContinuousSolution
from tests.optimization._continuous_cases import geometry, problem
from tests.optimization.test_continuous_math import quadratic

FIXTURE = Path(__file__).parent / "fixtures/continuous/v1"


@pytest.mark.parametrize("name,contract", [("problem.json", ContinuousProblem), ("solution.json", ContinuousSolution)])
def test_manual_complete_wire_is_canonical_and_has_no_writer_generated_expectations(name, contract):
    """The closed reader and writer must preserve each authored field/byte."""
    raw = (FIXTURE / name).read_text()
    value = contract.from_json(raw)
    assert value.to_json() + "\n" == raw


def test_actual_coupled_solver_matches_the_independently_authored_complete_solution():
    """Wire equality includes all original constraints, score and KKT evidence."""
    raw = (FIXTURE / "solution.json").read_text()
    expected = ContinuousSolution.from_json(raw)
    actual = solve_continuous(problem())
    assert actual.to_json() == expected.to_json()
    assert actual.problem.to_json() + "\n" == (FIXTURE / "problem.json").read_text()


def test_two_different_active_constraint_cases_match_the_same_literal_rational_derivation():
    """The independent scalar substitution checks optimum and clipping gap."""
    hand = json.loads((FIXTURE / "hand.json").read_text())
    matrix = tuple(tuple(Fraction(x) for x in row) for row in hand["matrix"])
    linear = tuple(Fraction(x) for x in hand["linear"])
    for name in ("simplex", "active_box"):
        row = hand[name]
        optimum = tuple(Fraction(x) for x in row["optimum"])
        assert quadratic(optimum, matrix, linear) == Fraction(row["objective"])
        if name == "simplex":
            declared = problem()
            wrong = tuple(Fraction(x) for x in row["clip_renormalize"])
            gap = Fraction(row["clip_objective_gap"])
        else:
            declared = problem(geometry("signed-box-v1", ((-1.0, 0.5), (-1.0, 2.0))))
            wrong = tuple(Fraction(x) for x in row["clipped"])
            gap = Fraction(row["clipped_objective_gap"])
        actual = solve_continuous(declared)
        assert actual.executed == tuple(float(x) for x in optimum)
        assert quadratic(wrong, matrix, linear) - quadratic(optimum, matrix, linear) == gap
