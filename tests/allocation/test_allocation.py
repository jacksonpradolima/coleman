"""Hand-calculated and independent joint-scenario allocation evidence tests."""

from __future__ import annotations

import ast
import json
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
from pydantic import ValidationError

from coleman.allocation import (
    AllocationCheckpoint,
    AllocationLimits,
    AllocationProblem,
    AllocationResult,
    CurrentAllocation,
    DiscreteAllocation,
    JointScenario,
    NamedAmount,
    NonnegativeMoment,
    ProbabilityEvidence,
    ResourceCapacity,
    RiskConstraints,
    ScenarioPayoff,
    SignedPayoff,
    UtilitySpec,
    allocation_record,
    evaluate_allocation_action,
    penalty_semantics,
    publish_allocation_result,
    select_allocation,
    variance_semantics,
)
from coleman.artifacts import ArtifactWriter
from coleman.checkpoint import LocalCheckpointStore
from coleman.checkpoint.state import CheckpointPayload
from coleman.manifest import generate_manifest
from coleman.results.domain import fingerprint
from coleman.results.duckdb_sink import DuckDBSink
from coleman.selection import Quantity, UnitSemantics

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "allocation"
CREDIT = UnitSemantics(unit="credit", meaning="synthetic-ci-credit-accounting-v1")
TIME = UnitSemantics(unit="s", meaning="modeled-serial-ci-resource-time-v1")


def amount(value, semantics=CREDIT):
    return Quantity(value=Decimal(str(value)), semantics=semantics)


def signed(value):
    return SignedPayoff(value=Decimal(str(value)), semantics=CREDIT)


def problem(
    *,
    payoffs=None,
    fees=None,
    reserves=None,
    durations=None,
    balance=100,
    probabilities=("0.5", "0.5"),
    estimated=False,
    variance=None,
    shortfall=None,
    floor=None,
    coefficient=None,
    target=0,
    capacity=5,
):
    payoffs = {"no_action": (0, 0), "stable": (2, 2), "variable": (10, -8)} if payoffs is None else payoffs
    fees, reserves, durations = fees or {}, reserves or {}, durations or {}
    scenarios = tuple(
        JointScenario(
            scenario_id=f"s{index}",
            probability=Decimal(value),
            lower=Decimal("0") if estimated else None,
            upper=Decimal("1") if estimated else None,
        )
        for index, value in enumerate(probabilities)
    )
    actions = tuple(
        DiscreteAllocation(
            action_id=name,
            holdings=(),
            resource_use=(NamedAmount(name="runner_time", amount=amount(durations.get(name, 0), TIME)),),
            capital_required=amount(reserves.get(name, fees.get(name, 0))),
            transaction_cost=amount(fees.get(name, 0)),
            gross_payoffs=tuple(
                ScenarioPayoff(scenario_id=f"s{index}", gross=signed(value)) for index, value in enumerate(values)
            ),
        )
        for name, values in payoffs.items()
    )
    return AllocationProblem(
        schema_version=1,
        data_fingerprint=fingerprint(
            {
                "payoffs": {key: list(values) for key, values in payoffs.items()},
                "fees": fees,
                "reserves": reserves,
                "durations": durations,
            }
        ),
        value_semantics=CREDIT,
        initial_balance=amount(balance),
        current=CurrentAllocation(
            holdings=(),
            resource_use=(NamedAmount(name="runner_time", amount=amount(0, TIME)),),
            reserved_balance=amount(0),
        ),
        resources=(ResourceCapacity(name="runner_time", capacity=amount(capacity, TIME)),),
        holding_definitions=(),
        scenarios=scenarios,
        probability_evidence=ProbabilityEvidence(
            kind="estimated" if estimated else "assumed",
            evidence_fingerprint=fingerprint("synthetic-weight-evidence-v1") if estimated else None,
            method_id="synthetic-frequency-estimate-v1" if estimated else None,
        ),
        actions=actions,
        no_action_id="no_action",
        risk=RiskConstraints(
            downside_target=signed(target),
            maximum_variance=NonnegativeMoment(value=Decimal(str(variance)), semantics=variance_semantics(CREDIT))
            if variance is not None
            else None,
            maximum_expected_shortfall=amount(shortfall) if shortfall is not None else None,
            minimum_terminal_balance=amount(floor) if floor is not None else None,
        ),
        utility=UtilitySpec(kind="quadratic_target", coefficient=amount(coefficient, penalty_semantics(CREDIT)))
        if coefficient is not None
        else UtilitySpec(),
    )


def select(request, **limits):
    return select_allocation(request, limits=AllocationLimits(deadline_seconds=None, **limits))


def independent_moments(request, action):
    """Independent Decimal definitions; do not import production arithmetic/forecast helpers."""
    with localcontext() as context:
        context.prec = 110
        gross = {row.scenario_id: row.gross.value for row in action.gross_payoffs}
        values = [
            (scenario.probability, gross[scenario.scenario_id] - action.transaction_cost.value)
            for scenario in request.scenarios
        ]
        mean = sum((weight * value for weight, value in values), Decimal(0))
        variance = sum((weight * (value - mean) ** 2 for weight, value in values), Decimal(0))
        shortage = sum(
            (weight * max(request.risk.downside_target.value - value, Decimal(0)) for weight, value in values),
            Decimal(0),
        )
        utility = mean
        if request.utility.coefficient is not None:
            utility -= request.utility.coefficient.value * sum(
                (weight * max(request.risk.downside_target.value - value, Decimal(0)) ** 2 for weight, value in values),
                Decimal(0),
            )
    return mean, variance, shortage, utility


def test_issue_fixture_signed_mean_variance_and_no_fee_double_counting():
    request = problem()
    rows = {row.forecast.action_id: row.forecast for row in select(request).completed}
    assert rows["no_action"].expected_net.value == 0
    assert rows["no_action"].variance.value == 0
    assert rows["stable"].expected_net.value == 2
    assert rows["stable"].variance.value == 0
    assert rows["variable"].expected_net.value == 1
    assert rows["variable"].variance.value == 81
    assert rows["variable"].expected_target_shortfall.value == 4
    assert rows["variable"].outcomes[1].net.value == -8
    result = select(request)
    assert result.choice.selected_action_id == "stable"
    assert result.choice.status == "optimal"
    assert result.termination_reason == "enumeration_complete"
    assert AllocationResult.from_json(result.to_json()) == result


def test_gross_fee_reservation_and_initial_balance_have_distinct_roles():
    request = problem(payoffs={"no_action": (0, 0), "stable": (3, 3)}, fees={"stable": 1}, reserves={"stable": 4})
    row = evaluate_allocation_action(request, "stable").forecast
    assert row.expected_gross.value == 3
    assert row.expected_net.value == 2
    assert row.outcomes[0].terminal_balance.value == 102
    assert row.variance.value == 0
    # Reserved balance 4 is not charged as another expense; the one fee is already in net.
    assert select(request).choice.selected_action_id == "stable"


def test_no_action_optimal_for_negative_alternatives_and_preferred_on_exact_ties():
    negative = problem(payoffs={"no_action": (0, 0), "stable": (-2, -2), "variable": (1, -7)})
    result = select(negative)
    assert result.choice.selected_action_id == "no_action"
    tied = problem(payoffs={"no_action": (0, 0), "a_change": (0, 0)})
    assert select(tied).choice.selected_action_id == "no_action"
    # A negative optimum remains a valid recorded outcome if even the unchanged baseline loses more.
    losing = select(problem(payoffs={"no_action": (-5, -5), "stable": (-2, -2)}))
    assert losing.choice.selected_action_id == "stable"
    record = allocation_record(losing, run_id="ci", available_at=datetime(2026, 1, 1, tzinfo=UTC))
    assert record.measurement("expected_net_payoff").value == -2
    assert record.learning_reward is None


def test_capital_resources_and_risk_constraints_bind_independently():
    request = problem(
        payoffs={"no_action": (0, 0), "stable": (2, 2), "high": (20, 20)}, reserves={"high": 6, "stable": 1}, balance=5
    )
    assert select(request).choice.selected_action_id == "stable"
    assert evaluate_allocation_action(request, "high").rejection_reasons == ("capital_constraint",)
    resources = problem(durations={"stable": 6}, capacity=5)
    assert evaluate_allocation_action(resources, "stable").rejection_reasons == ("resource_constraint",)
    risky = problem(variance=80, shortfall=3, floor=95)
    rejected = evaluate_allocation_action(risky, "variable")
    assert rejected.rejection_reasons == (
        "variance_constraint",
        "target_shortfall_constraint",
        "terminal_balance_constraint",
    )
    assert rejected.forecast.outcomes[1].net.value == -8
    impossible = select(problem(payoffs={"no_action": (-5, -5)}, floor=100))
    assert impossible.choice.status == "infeasible"
    assert impossible.choice.selected_action_id is None


def test_nonlinear_utility_is_expected_over_scenarios_not_applied_to_expected_payoff():
    request = problem(payoffs={"no_action": (0, 0), "stable": (2, 2), "variable": (12, -6)}, coefficient="0.1")
    variable = evaluate_allocation_action(request, "variable").forecast
    assert variable.expected_net.value == 3
    assert variable.variance.value == 81
    assert variable.utility.value == Decimal("1.2")
    # Applying u to the positive mean would give 3, a different and incorrect selection.
    assert select(request).choice.selected_action_id == "stable"


def test_joint_scenarios_preserve_correlation_and_zero_probability_losses():
    request = problem(payoffs={"no_action": (0, 0), "single": (4, -2), "joint": (8, -4)})
    assert evaluate_allocation_action(request, "single").forecast.variance.value == 9
    assert evaluate_allocation_action(request, "joint").forecast.variance.value == 36
    zero_weight = problem(payoffs={"no_action": (0, 0), "stable": (2, -1000)}, probabilities=("1", "0"), floor=0)
    row = evaluate_allocation_action(zero_weight, "stable")
    assert row.feasible
    assert row.forecast.expected_net.value == 2
    assert row.forecast.variance.value == 0
    assert row.forecast.outcomes[1].net.value == -1000


def test_estimated_weights_retain_bounds_and_force_conservative_insufficient_evidence():
    request = problem(estimated=True)
    result = select(request)
    assert result.choice.status == "insufficient_evidence"
    assert result.choice.selected_action_id == "no_action"
    assert result.choice.conservative_fallback
    assert result.problem.probability_evidence.independent_units is None
    assert all(scenario.lower == 0 and scenario.upper == 1 for scenario in result.problem.scenarios)
    assert len(result.completed) == 3
    assert AllocationResult.from_json(result.to_json()) == result
    unsafe = problem(
        payoffs={"no_action": (0, -8), "stable": (2, 2)}, probabilities=("1", "0"), estimated=True, variance=0
    )
    # Point variance is zero, but the estimated support cannot certify the unchanged fallback.
    assert evaluate_allocation_action(unsafe, "no_action").feasible
    uncertain = select(unsafe)
    assert uncertain.choice.status == "insufficient_evidence"
    assert uncertain.choice.selected_action_id is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["scenarios"][0].update(probability="-0.1"),
        lambda data: data["scenarios"][0].update(probability="NaN"),
        lambda data: data["scenarios"][0].update(probability="0.499999999999"),
        lambda data: data["scenarios"].append(data["scenarios"][0]),
        lambda data: data["actions"].append(data["actions"][0]),
        lambda data: data.update(no_action_id="missing"),
        lambda data: data["actions"][0].update(
            holdings=[
                {"name": "unknown", "amount": {"value": "0", "semantics": {"unit": "count", "meaning": "unknown-v1"}}}
            ]
        ),
        lambda data: data["actions"][0]["transaction_cost"].update(value="1"),
        lambda data: data["actions"][0]["gross_payoffs"][0].update(scenario_id="missing"),
        lambda data: data["actions"][0]["gross_payoffs"].pop(),
        lambda data: data["actions"][0]["gross_payoffs"][0]["gross"].update(value="Infinity"),
        lambda data: data["actions"][0]["resource_use"][0]["amount"]["semantics"].update(unit="ms"),
        lambda data: data["initial_balance"].update(value="-1"),
        lambda data: data["utility"].update(kind="log"),
    ],
)
def test_invalid_probabilities_joint_identity_units_accounting_and_unsupported_objectives(mutation):
    payload = json.loads(problem().to_json())
    mutation(payload)
    with pytest.raises(ValidationError):
        AllocationProblem.model_validate(payload)


def test_zero_null_unavailable_and_sub_tick_moment_precision():
    zero = select(problem(payoffs={"no_action": (0, 0)}))
    empty = select(problem(), max_work=0)
    assert empty.choice.status == "limited"
    assert empty.choice.selected_action_id is None
    when = datetime(2026, 1, 1, tzinfo=UTC)
    assert allocation_record(zero, run_id="ci", available_at=when).measurement("expected_net_payoff").value == 0
    missing = allocation_record(empty, run_id="ci", available_at=when).measurement("expected_net_payoff")
    assert missing.value is None
    assert missing.status == "unavailable"
    request = problem(
        payoffs={"no_action": ("0", "0"), "tiny": ("0.000000000001", "0")},
        probabilities=("0.000000000001", "0.999999999999"),
    )
    with localcontext() as context:
        context.prec = 4
        result = evaluate_allocation_action(request, "tiny").forecast
    assert result.expected_net.value == Decimal("1e-24")
    assert result.variance.value == Decimal("0.000000000000000000000000000000000000999999999999")


def test_partial_work_deadline_cancellation_and_mid_action_interrupt_are_truthful():
    request = problem()
    partial = select(request, max_work=1)
    assert partial.termination_reason == "work_limit"
    assert partial.choice.status == "feasible"
    assert partial.choice.selected_action_id == "no_action"
    assert [row.forecast.action_id for row in partial.completed] == ["no_action"]
    deadline = select_allocation(request, limits=AllocationLimits(deadline_seconds=0))
    assert deadline.termination_reason == "deadline"
    assert deadline.completed == ()
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls == 5

    interrupted = select_allocation(request, cancelled=cancelled)
    assert interrupted.termination_reason == "cancelled"
    assert interrupted.work_used == 2
    assert len(interrupted.completed) == 1

    def keyboard():
        raise KeyboardInterrupt

    stopped = select_allocation(request, cancelled=keyboard)
    assert stopped.termination_reason == "interrupted"
    assert stopped.completed == ()


def test_checkpoint_resume_reuses_verified_progress_and_rejects_corruption_before_preflight(tmp_path):
    request = problem()
    partial = select(request, max_work=1)
    store = LocalCheckpointStore(str(tmp_path))
    store.save(
        CheckpointPayload(run_id="allocation-ci", step=1, extra={"allocation_v1": partial.checkpoint().to_json()})
    )
    saved = store.load("allocation-ci", 0)
    assert saved is not None
    checkpoint = AllocationCheckpoint.from_json(saved.extra["allocation_v1"])
    resumed = select_allocation(request, checkpoint=checkpoint, limits=AllocationLimits(max_work=2))
    assert resumed.choice.selected_action_id == "stable"
    assert resumed.choice.status == "optimal"
    assert resumed.work_used == 2
    payload = checkpoint.model_dump(mode="json")
    payload["completed"][0]["forecast"]["utility"]["value"] = "100"
    corrupt = AllocationCheckpoint.model_validate(payload)
    limits = AllocationLimits(max_working_bytes=0)
    with pytest.raises(ValueError, match="differs"):
        select_allocation(request, checkpoint=corrupt, limits=limits)
    changed_request = problem(balance=99)
    with pytest.raises(ValueError, match="source differs"):
        select_allocation(changed_request, checkpoint=checkpoint)


@pytest.mark.parametrize("stop", ["deadline", "cancelled", "interrupted", "none"])
def test_complete_checkpoint_still_honors_deadline_and_cancellation_without_new_work(stop):
    request = problem()
    previous = select(request)
    checkpoint = previous.checkpoint()
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        if stop == "interrupted":
            raise KeyboardInterrupt
        return stop == "cancelled" and calls == 2

    result = select_allocation(
        request,
        checkpoint=checkpoint,
        limits=AllocationLimits(max_work=0, deadline_seconds=0 if stop == "deadline" else None),
        cancelled=cancelled,
    )
    assert result.termination_reason == ("enumeration_complete" if stop == "none" else stop)
    assert result.work_used == 0
    assert result.completed == previous.completed
    # Cached complete mathematical evidence remains valid even when this invocation stops.
    assert result.choice == previous.choice
    assert AllocationResult.from_json(result.to_json()) == result
    assert calls == {"deadline": 1, "cancelled": 2, "interrupted": 1, "none": 3}[stop]


def test_independent_decimal_oracle_and_metamorphic_joint_order():
    rng = random.Random(254)
    for index in range(60):
        payoffs = {
            "no_action": (0, 0),
            **{f"plan-{row}": (rng.randrange(-10, 11), rng.randrange(-10, 11)) for row in range(4)},
        }
        request = problem(payoffs=payoffs, coefficient="0.1" if index % 2 else None)
        result = select(request)
        utilities = {}
        for row in result.completed:
            expected = independent_moments(request, request.action(row.forecast.action_id))
            assert row.forecast.expected_net.value == expected[0]
            assert row.forecast.variance.value == expected[1]
            assert row.forecast.expected_target_shortfall.value == expected[2]
            assert row.forecast.utility.value == expected[3]
            utilities[row.forecast.action_id] = expected[3]
        maximum = max(utilities.values())
        tied = [name for name, value in utilities.items() if value == maximum]
        selected = "no_action" if "no_action" in tied else min(tied)
        assert result.choice.selected_action_id == selected
        payload = json.loads(request.to_json())
        payload["actions"].reverse()
        payload["scenarios"].reverse()
        for action in payload["actions"]:
            action["gross_payoffs"].reverse()
        reordered = AllocationProblem.model_validate(payload)
        assert reordered.fingerprint == request.fingerprint
        assert select(reordered).to_json() == result.to_json()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["choice"].update(selected_action_id="variable"),
        lambda data: data["completed"][0]["forecast"]["expected_net"].update(value="100"),
        lambda data: data["completed"][0].update(feasible=False),
        lambda data: data["completed"].pop(),
        lambda data: data.update(work_used=0),
        lambda data: data["choice"].update(status="feasible"),
    ],
)
def test_forged_statistics_feasibility_selection_and_completion_fail_closed(mutation):
    payload = json.loads(select(problem()).to_json())
    mutation(payload)
    with pytest.raises(ValidationError):
        AllocationResult.model_validate(payload)


def test_versioned_golden_roundtrip_future_versions_and_duplicate_json():
    expected = json.loads((FIXTURES / "digests-v1.json").read_text())
    models = {
        "problem": AllocationProblem,
        "optimal": AllocationResult,
        "partial": AllocationResult,
        "uncertain": AllocationResult,
        "checkpoint": AllocationCheckpoint,
    }
    for name, model in models.items():
        content = (FIXTURES / f"{name}-v1.json").read_text()
        restored = model.from_json(content)
        assert restored.to_json() + "\n" == content
        assert restored.fingerprint == expected[name]
        for version in (None, 0, 2, True, "1"):
            payload = json.loads(content)
            payload["schema_version"] = version
            serialized = json.dumps(payload)
            with pytest.raises(ValidationError):
                model.from_json(serialized)
    partial = select(problem(), max_work=1)
    fresh = {
        "problem": problem(),
        "optimal": select(problem()),
        "partial": partial,
        "uncertain": select(problem(estimated=True)),
        "checkpoint": partial.checkpoint(),
    }
    assert {name: value.fingerprint for name, value in fresh.items()} == expected
    duplicate = problem().to_json().replace('"schema_version":1', '"schema_version":1,"schema_version":1')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        AllocationProblem.from_json(duplicate)


def _publish(root, serialized):
    return str(publish_allocation_result(AllocationResult.from_json(serialized), ArtifactWriter(Path(root))))


def test_atomic_concurrent_publication_manifest_and_signed_sink_replay(tmp_path):
    result = select(problem(payoffs={"no_action": (-5, -5), "stable": (-2, -2)}))
    with ProcessPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(_publish, [str(tmp_path)] * 6, [result.to_json()] * 6))
    assert len(set(paths)) == 1
    assert AllocationResult.from_json(Path(paths[0]).read_bytes()) == result
    inventory = json.loads(generate_manifest(tmp_path, run_id="ci").read_text())
    assert len([row for row in inventory["artifacts"] if row["type"] == "allocation_result"]) == 1
    record = allocation_record(result, run_id="ci", available_at=datetime(2026, 1, 1, tzinfo=UTC))
    sink = DuckDBSink(out_dir=str(tmp_path / "results"))
    try:
        sink.write_domain_record(record)
        sink.write_domain_record(record)
        assert list(sink.domain_records()) == [record]
        assert record.measurement("expected_net_payoff").value == -2
        assert record.learning_reward is None
    finally:
        sink.close()


def test_interrupted_artifact_never_becomes_completed(tmp_path, monkeypatch):
    result = select(problem())
    writer = ArtifactWriter(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("interrupted")

    monkeypatch.setattr("coleman.artifacts.os.link", fail)
    with pytest.raises(OSError):
        publish_allocation_result(result, writer)
    assert list(tmp_path.rglob("*.json")) == []


def test_transport_import_boundaries_and_hash_seed_determinism(tmp_path):
    for path in (ROOT / "coleman" / "allocation").glob("*.py"):
        tree = ast.parse(path.read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(name.startswith(("coleman.cli", "mcp", "coleman.policy", "coleman.runner")) for name in imports)
    script = """import sys
from coleman.allocation import AllocationProblem, AllocationLimits, select_allocation
problem = AllocationProblem.from_json(sys.stdin.read())
result = select_allocation(problem, limits=AllocationLimits(deadline_seconds=None))
print(result.to_json())
assert 'coleman.cli' not in sys.modules
"""
    outputs = []
    for seed in ("4", "23"):
        outputs.append(
            subprocess.run(
                [sys.executable, "-c", script],
                input=problem().to_json(),
                cwd=tmp_path,
                env=dict(os.environ, PYTHONHASHSEED=seed),
                text=True,
                capture_output=True,
                check=True,
            ).stdout
        )
    assert outputs[0] == outputs[1]


def test_installed_package_native_ci_and_signed_resource_examples(tmp_path):
    wheel_dir = tmp_path / "wheels"
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(wheel_dir)], cwd=ROOT, check=True)
    wheel = next(wheel_dir.glob("*.whl"))
    installed = tmp_path / "installed"
    subprocess.run(["uv", "venv", str(installed), "--python", sys.executable], check=True)
    python = installed / "bin" / "python"
    subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel) + "[ci]"], check=True)
    external = tmp_path / "external"
    external.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for filename in ("ci.py", "resources.py"):
        script = external / filename
        shutil.copy(ROOT / "examples" / "scenario-payoffs" / filename, script)
        completed = subprocess.run(
            [str(python), str(script), str(external / filename.removesuffix(".py"))],
            cwd=external,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
        output = json.loads(completed.stdout.splitlines()[-1])
        assert output["selected_action_id"] == "stable"
        assert output["status"] == "optimal"
        replay = subprocess.run(
            [str(python), str(script), str(external / filename.removesuffix(".py"))],
            cwd=external,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
        assert json.loads(replay.stdout.splitlines()[-1]) == output
        if filename == "ci.py":
            assert output["native_budget_seconds"] == 3


def test_current_holdings_and_no_action_preserve_complete_explicit_vectors():
    payload = json.loads(problem().to_json())
    semantics = {"unit": "slot", "meaning": "reserved-ci-runner-slot-v1"}
    payload["holding_definitions"] = [{"name": "runner_pool", "semantics": semantics}]
    vector = [{"name": "runner_pool", "amount": {"value": "1", "semantics": semantics}}]
    payload["current"]["holdings"] = vector
    for action in payload["actions"]:
        action["holdings"] = json.loads(json.dumps(vector))
    request = AllocationProblem.model_validate(payload)
    assert select(request).choice.selected_action_id == "stable"
    unchanged = next(action for action in payload["actions"] if action["action_id"] == "no_action")
    unchanged["holdings"][0]["amount"]["value"] = "0"
    with pytest.raises(ValidationError, match="No-action must preserve"):
        AllocationProblem.model_validate(payload)


def test_probability_estimation_bounds_and_units_cannot_be_ignored():
    payload = json.loads(problem(estimated=True).to_json())
    payload["scenarios"][0]["lower"] = None
    with pytest.raises(ValidationError, match="both bounds"):
        AllocationProblem.model_validate(payload)
    payload = json.loads(problem(estimated=True).to_json())
    payload["probability_evidence"]["method_id"] = None
    with pytest.raises(ValidationError, match="evidence and method"):
        AllocationProblem.model_validate(payload)
    payload = json.loads(problem(coefficient="0.1").to_json())
    payload["utility"]["coefficient"]["semantics"] = CREDIT.model_dump()
    with pytest.raises(ValidationError, match="inverse payoff units"):
        AllocationProblem.model_validate(payload)
    payload = json.loads(problem(variance=81).to_json())
    payload["risk"]["maximum_variance"]["semantics"] = CREDIT.model_dump()
    with pytest.raises(ValidationError, match="squared payoff units"):
        AllocationProblem.model_validate(payload)


def test_memory_limit_and_future_or_unknown_checkpoint_integrity_before_early_return():
    request = problem()
    limited = select(request, max_working_bytes=0)
    assert limited.termination_reason == "memory_limit"
    assert limited.choice.status == "limited"
    assert limited.completed == ()
    assert limited.choice.selected_action_id is None
    with pytest.raises(ValueError, match="No completed"):
        limited.checkpoint()
    partial = select(request, max_work=1).checkpoint()
    future = partial.model_copy(update={"schema_version": 2})
    limits = AllocationLimits(max_working_bytes=0)
    with pytest.raises(ValidationError):
        select_allocation(request, checkpoint=future, limits=limits)
    data = partial.model_dump(mode="json")
    data["completed"][0]["forecast"]["action_id"] = "unknown"
    unknown = AllocationCheckpoint.model_validate(data)
    with pytest.raises(ValueError, match="Unknown allocation action"):
        select_allocation(request, checkpoint=unknown, limits=limits)


def test_tiny_negative_utility_and_moment_units_survive_low_decimal_context():
    request = problem(
        payoffs={"no_action": (-5, -5), "stable": ("-0.000000000001", "0")},
        probabilities=("0.000000000001", "0.999999999999"),
    )
    with localcontext() as context:
        context.prec = 4
        result = select(request)
    assert result.choice.selected_action_id == "stable"
    row = next(item.forecast for item in result.completed if item.forecast.action_id == "stable")
    assert row.expected_net.value == Decimal("-1e-24")
    assert row.utility.value == Decimal("-1e-24")
    assert row.variance.semantics == variance_semantics(CREDIT)
    assert AllocationResult.from_json(result.to_json()) == result


def test_observation_count_requires_a_named_unit_and_never_comes_from_seeds():
    source_fingerprint = fingerprint("observations")
    with pytest.raises(ValidationError, match="unit definition"):
        ProbabilityEvidence(
            kind="estimated",
            evidence_fingerprint=source_fingerprint,
            method_id="external-estimator-v1",
            independent_units=5,
        )
    evidence = ProbabilityEvidence(
        kind="estimated",
        evidence_fingerprint=source_fingerprint,
        method_id="external-estimator-v1",
        independent_unit="repository",
        independent_units=5,
    )
    assert evidence.independent_units == 5


def test_claimed_work_limit_must_match_the_actual_declared_counter():
    partial = select(problem(), max_work=1)
    payload = json.loads(partial.to_json())
    payload["limits"]["max_work"] = 2
    with pytest.raises(ValidationError, match="completed budget"):
        AllocationResult.model_validate(payload)
