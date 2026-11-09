"""Compose real forecasts, constrained solves and separate synthetic path charges.

This public deterministic example measures mechanics. Coefficients use a unit
reference balance; it does not model funding, shorting, margin or live trading.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from decimal import Decimal, localcontext
from pathlib import Path

from coleman.allocation import (
    AllocationProblem,
    CurrentAllocation,
    DiscreteAllocation,
    HoldingDefinition,
    JointScenario,
    NamedAmount,
    ProbabilityEvidence,
    RiskConstraints,
    ScenarioPayoff,
    SignedPayoff,
    select_allocation,
)
from coleman.core.numeric import ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.quantities import Quantity
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage
from coleman.optimization.continuous import projected_update, solve_continuous
from coleman.optimization.continuous_contracts import (
    CoefficientInterval,
    ContinuousProblem,
    ContinuousSet,
    ContinuousVariable,
    QuadraticObjective,
)


def load_flow():
    """Load the explicitly selected public chronological example.

    Returns
    -------
    module
        Actual sibling flow.py, with its real predictor and replay.
    """
    name = "authored_continuous_allocation_flow"
    if name not in sys.modules:
        specification = importlib.util.spec_from_file_location(name, Path(__file__).with_name("flow.py"))
        if specification is None or specification.loader is None:
            raise RuntimeError("The explicitly copied flow.py is required")
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
    return sys.modules[name]


def lineage() -> DefinitionLineage:
    """Retain explicit unknown declaration provenance.

    Returns
    -------
    DefinitionLineage
        No invented external source, clock or approval.
    """
    return DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)


def problem(prediction: float, *, coupled_fixture: bool = False) -> ContinuousProblem:
    """Construct a strict convex loss on a declared coefficient feasible set.

    Parameters
    ----------
    prediction : float
        Signed dimensionless forecast; not a calibrated probability.
    coupled_fixture : bool, optional
        Select the independently solved active-box example, defaultFalse.

    Returns
    -------
    ContinuousProblem
        Coupled Q=[[2,1],[1,2]], fixed units, comparator and explicit transition
        model identity. The mathematical score is not relabeled as variance.
    """
    flow = load_flow()
    reference = flow.MODEL.reference

    def interval(lower, upper):
        return CoefficientInterval(schema_version=1, lower=lower, upper=upper)

    upper = (0.5, 2.0) if coupled_fixture else (1.0, 1.0)
    feasible = ContinuousSet(
        schema_version=1,
        reference=reference("constructed-active-box" if coupled_fixture else "constructed-two-coordinate-simplex"),
        lineage=lineage(),
        profile="signed-box-v1" if coupled_fixture else "simplex-v1",
        variables=tuple(
            ContinuousVariable(schema_version=1, member=reference(f"allocation-{index}"), lower=0.0, upper=bound)
            for index, bound in enumerate(upper)
        ),
        net=interval(0.0, 2.5) if coupled_fixture else interval(1.0, 1.0),
        cash=interval(-1.5, 1.0) if coupled_fixture else interval(0.0, 0.0),
        gross_limit=2.5 if coupled_fixture else 1.0,
        leverage_limit=4.0 if coupled_fixture else 1.0,
    )
    return ContinuousProblem(
        schema_version=1,
        feasible_set=feasible,
        objective=QuadraticObjective(
            schema_version=1,
            reference=reference("constructed-coupled-quadratic"),
            lineage=lineage(),
            matrix=((2.0, 1.0), (1.0, 2.0)),
            linear=(-3.0, -3.0) if coupled_fixture else (-prediction, 0.0),
            score=ScalarSpec(
                dimension=reference("constructed-quadratic-score"),
                role="objective",
                semantics=UnitSemantics(unit="1", meaning="constructed-coupled-loss-v1"),
            ),
        ),
        comparator=DefinitionBinding(
            reference=reference("best-fixed-feasible-vector-same-realized-losses"),
            definition_fingerprint=fingerprint({"comparator": "same-realized-quadratics-v1"}),
        ),
        transition=DefinitionBinding(
            reference=reference("constructed-own-book-charge"),
            definition_fingerprint=fingerprint({"charge_credit_per_coefficient_l1": "0.001"}),
        ),
    )


def objective(point: tuple[float, float], response: float) -> float:
    """Evaluate the example score independently from production solver helpers.

    Parameters
    ----------
    point : tuple of float
        Two executed allocation coefficients.
    response : float
        Signed synthetic response in the same dimensionless scale.

    Returns
    -------
    float
        x0²+x0*x1+x1²−response*x0, without a physical fee deduction.
    """
    first, second = point
    return first * first + first * second + second * second - response * first


def finite_baseline(prediction: float) -> tuple[float, float]:
    """Use the existing finite solver on the same forecast and five fixed points.

    Parameters
    ----------
    prediction : float
        Same available prediction as the continuous route.

    Returns
    -------
    tuple of float
        Existing select_allocation choice among x0=0,.25,.5,.75,1.

    Notes
    -----
    Gross modeled credit equals negative quadratic score, rounded once to the
    retained finite contract's1e-12 tick. Absolute rounding is at most5e-13.
    One assumed deterministic state is an encoding of a supplied objective,
    never estimated probabilities or a claimed calibrated outcome forecast.
    """
    credit = UnitSemantics(unit="credit", meaning="constructed-negative-quadratic-credit-v1")
    coefficient = UnitSemantics(unit="1", meaning="constructed-reference-balance-coefficient-v1")

    def amount(value, semantics=credit):
        return Quantity(value=Decimal(str(value)), semantics=semantics)

    def holdings(point):
        return tuple(NamedAmount(name=f"allocation-{i}", amount=amount(x, coefficient)) for i, x in enumerate(point))

    actions = []
    for index in range(5):
        point = (index / 4, 1 - index / 4)
        with localcontext() as context:
            context.prec = 60
            value = Decimal(str(-objective(point, prediction))).quantize(Decimal("1e-12"))
        actions.append(
            DiscreteAllocation(
                action_id=f"grid-{index}",
                holdings=holdings(point),
                resource_use=(),
                capital_required=amount(0),
                transaction_cost=amount(0),
                gross_payoffs=(
                    ScenarioPayoff(scenario_id="supplied-loss", gross=SignedPayoff(value=value, semantics=credit)),
                ),
            )
        )
    request = AllocationProblem(
        schema_version=1,
        data_fingerprint=fingerprint({"prediction": prediction, "grid": 4}),
        value_semantics=credit,
        initial_balance=amount(1),
        current=CurrentAllocation(holdings=holdings((0.5, 0.5)), resource_use=(), reserved_balance=amount(0)),
        resources=(),
        holding_definitions=tuple(HoldingDefinition(name=f"allocation-{i}", semantics=coefficient) for i in range(2)),
        scenarios=(JointScenario(scenario_id="supplied-loss", probability=Decimal(1)),),
        probability_evidence=ProbabilityEvidence(kind="assumed"),
        actions=tuple(actions),
        no_action_id="grid-2",
        risk=RiskConstraints(downside_target=SignedPayoff(value=Decimal(0), semantics=credit)),
    )
    result = select_allocation(request)
    selected = result.choice.selected_action_id
    if selected is None:
        raise ValueError("The bounded complete finite baseline could not select a feasible grid point")
    return tuple(float(row.amount.value) for row in request.action(selected).holdings)


def _chosen(row, index, books, next_online):
    """Use the same eligible forecast at each mode's declared release."""
    solved = solve_continuous(problem(row["prediction"]))
    if solved.status != "optimal" or solved.executed is None:
        raise ValueError("The declared synthetic forecast solve did not obtain a qualified feasible allocation")
    periodic = solve_continuous(problem(row["prediction"])) if index % 2 == 0 else None
    if periodic is not None and (periodic.status != "optimal" or periodic.executed is None):
        raise ValueError("The independent periodic route could not release a qualified allocation")
    actions = {
        "fixed": (0.5, 0.5),
        "periodic": periodic.executed if periodic is not None else books["periodic"],
        "finite": finite_baseline(row["prediction"]),
        "forecast-continuous": solved.executed,
        "projected-online": next_online,
    }
    work = {name: 0 for name in actions}
    work["forecast-continuous"] = solved.faces_examined
    work["periodic"] = 0 if periodic is None else periodic.faces_examined
    return actions, work


def _accounting_row(index, name, point, previous, row):
    """Keep one actual coefficient change and charge on its own synthetic book."""
    movement = math.fsum(abs(x - old) for x, old in zip(point, previous, strict=True))
    charge = 0.001 * movement
    outcomes = (row["target"] / 100, row["target"] / 200 + 0.001)
    gross = math.fsum(x * value for x, value in zip(point, outcomes, strict=True))
    return {
        "step": index,
        "mode": name,
        "executed": list(point),
        "forecast": row["prediction"],
        "realized_same_function_loss": objective(point, row["target"]),
        "coefficient_l1_movement": movement,
        "own_book_charge_credit": charge,
        "gross_credit": gross,
        "net_credit": gross - charge,
        "objective_penalty_paid_as_fee": 0.0,
    }


def _path(forecast):
    """Advance real projected updates only after each executed period's label."""
    names = ("fixed", "periodic", "finite", "forecast-continuous", "projected-online")
    books = {name: (0.5, 0.5) for name in names}
    next_online = (0.5, 0.5)
    trace = []
    for index, row in enumerate(forecast["rows"]):
        chosen, work = _chosen(row, index, books, next_online)
        for name in names:
            point = chosen[name]
            trace.append(_accounting_row(index, name, point, books[name], row))
            trace[-1]["solve_faces_examined"] = work[name]
            trace[-1]["finite_candidates_evaluated"] = 5 if name == "finite" else 0
            trace[-1]["post_label_projection_faces"] = 0
            books[name] = point
        update = projected_update(problem(row["target"]), next_online, step_size=0.25)
        if update.projection.status != "optimal" or update.projection.executed is None:
            raise ValueError("The declared online gradient update did not produce a qualified feasible successor")
        next_online = update.projection.executed
        trace[-1]["post_label_projection_faces"] = update.projection.faces_examined
    return trace


def run() -> dict[str, object]:
    """Compose actual forecasts and solves against fixed/periodic/finite controls.

    Returns
    -------
    dict
        Shared-input synthetic raw path, one charge per own-book coefficient
        change, forecast scores, realized same-function losses and external
        comparator. No stochastic uncertainty, market return or win is claimed.
    """
    flow = load_flow()
    forecast = flow.run_prediction()
    trace = _path(forecast)
    # The offline comparator minimizes the same sum of four realized losses;
    # it is an evaluation result and never an action deployed using the future.
    fixed_comparator = solve_continuous(problem(math.fsum(flow.LABELS) / len(flow.LABELS)))
    if fixed_comparator.executed is None:
        raise ValueError("The independent fixed comparator was unavailable")
    comparator_loss = math.fsum(objective(fixed_comparator.executed, y) for y in flow.LABELS)
    names = tuple(dict.fromkeys(row["mode"] for row in trace))
    return {
        "scope": "constructed-exogenous-path-and-modeled-additive-accounting",
        "capital_reference_credit_each": 1.0,
        "charge_rate_credit_per_coefficient_l1_each": 0.001,
        "tuning_trials_each": 0,
        "online_observations": len(flow.LABELS),
        "forecast": forecast,
        "trace": trace,
        "best_fixed_same_function_comparator": list(fixed_comparator.executed),
        "external_regret_same_realized_losses": {
            name: math.fsum(row["realized_same_function_loss"] for row in trace if row["mode"] == name)
            - comparator_loss
            for name in names
        },
        "comparator_specific_net_wealth_regret": "not_qualified",
        "risk_measure": "no_variance_shortfall_or_cvar_claim",
        "inference": "insufficient_evidence",
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
