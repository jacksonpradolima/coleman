"""Independent finite exhaustive oracles, proof validation and coverage integration."""

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
from itertools import combinations
from pathlib import Path

import pytest
from pydantic import ValidationError

from coleman.artifacts import ArtifactWriter
from coleman.checkpoint import LocalCheckpointStore
from coleman.checkpoint.state import CheckpointPayload
from coleman.manifest import generate_manifest
from coleman.results.domain import fingerprint
from coleman.results.parquet_sink import ParquetSink
from coleman.selection import (
    CoverageCandidate,
    CoverageCheckpoint,
    CoverageConstraints,
    CoverageOutcome,
    CoverageProblem,
    CoverageResult,
    Quantity,
    SearchLimits,
    UnitSemantics,
    coverage_record,
    publish_coverage_result,
    solve_coverage,
)

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "coverage"
COST = UnitSemantics(unit="s", meaning="modeled-serial-test-duration-v1")
WEIGHT = UnitSemantics(unit="1", meaning="weighted-test-capability-v1")


def quantity(value, semantics=COST):
    return Quantity(value=Decimal(str(value)), semantics=semantics)


def problem(budget=3, *, candidates=None, weights=None, required=(), maximum=None, search=None):
    candidates = [("A", 2, ("x", "y")), ("B", 2, ("y", "z")), ("C", 1, ("z",))] if candidates is None else candidates
    weights = [("x", "0.5"), ("y", "0.3"), ("z", "0.2")] if weights is None else weights
    return CoverageProblem(
        schema_version=1,
        data_fingerprint=fingerprint(json.loads(json.dumps({"candidates": candidates, "weights": weights}))),
        objective_semantics=WEIGHT,
        candidates=tuple(
            CoverageCandidate(candidate_id=name, cost=quantity(cost), covers=covered)
            for name, cost, covered in candidates
        ),
        outcomes=tuple(CoverageOutcome(outcome_id=name, weight=quantity(weight, WEIGHT)) for name, weight in weights),
        constraints=CoverageConstraints(budget=quantity(budget), required_ids=required, maximum_selected=maximum),
        search_ids=search,
    )


def exact(request, **limits):
    return solve_coverage(request, method="branch_bound_v1", limits=SearchLimits(deadline_seconds=None, **limits))


def independent_oracle(request):
    """Test-only combinations oracle, capped explicitly; no production math/solver helper."""
    candidates = {item.candidate_id: item for item in request.candidates}
    names = sorted(request.search_ids)
    if len(names) > 10:
        raise ValueError("Exhaustive test oracle is capped at ten candidates")
    weights = {item.outcome_id: item.weight.value for item in request.outcomes}
    feasible_rows = []
    with localcontext() as context:
        context.prec = 60
        for count in range(len(names) + 1):
            for ids in combinations(names, count):
                if not set(request.constraints.required_ids).issubset(ids):
                    continue
                maximum = request.constraints.maximum_selected
                if maximum is not None and len(ids) > maximum:
                    continue
                cost = sum((candidates[name].cost.value for name in ids), Decimal(0))
                if cost > request.constraints.budget.value:
                    continue
                covered = set().union(*(candidates[name].covers for name in ids))
                score = sum((weights[name] for name in sorted(covered)), Decimal(0))
                feasible_rows.append((ids, cost, score))
    return min(feasible_rows, key=lambda row: (-row[2], row[0])) if feasible_rows else None


@pytest.mark.parametrize("budget,ids,cost,score", [(3, ("A", "C"), "3", "1"), (2, ("A",), "2", "0.8")])
def test_issue_fixture_exact_cost_union_bound_gap_and_lexicographic_oracle(budget, ids, cost, score):
    request = problem(budget)
    result = exact(request)
    assert result.selected_ids == ids
    assert result.cost.value == Decimal(cost)
    assert result.objective.value == Decimal(score)
    assert result.status == "optimal"
    assert result.scope == "full"
    assert result.upper_bound.value == result.objective.value
    assert result.absolute_gap.value == 0
    assert result.proof_prefixes
    assert independent_oracle(request) == (ids, Decimal(cost), Decimal(score))
    assert CoverageResult.from_json(result.to_json()) == result


def test_greedy_is_deterministic_feasible_with_no_approximation_or_optimality_claim():
    result = solve_coverage(problem())
    assert result.status == "feasible"
    assert result.termination_reason == "greedy_complete"
    assert result.selected_ids == ("A", "C")
    assert result.proof_prefixes == ()
    assert result.absolute_gap is not None
    assert result.absolute_gap.value == 0
    assert result == CoverageResult.from_json(result.to_json())


@pytest.mark.parametrize("value", ["-1", "NaN", "Infinity", "-Infinity", "0.0000000000001", "1e25", None, True])
def test_invalid_nonnegative_decimal_quantities(value):
    with pytest.raises(ValidationError):
        Quantity(value=value, semantics=COST)


def test_zero_null_unavailable_and_valid_empty_selection_remain_distinct():
    zero = exact(problem(0))
    assert zero.feasible
    assert zero.selected_ids == ()
    assert zero.objective.value == zero.cost.value == 0
    limited = exact(problem(0), max_working_bytes=0)
    assert limited.status == "limited"
    assert limited.termination_reason == "memory_limit"
    assert limited.selected_ids is limited.cost is limited.objective is None
    now = datetime.now(UTC)
    assert coverage_record(zero, run_id="ci", available_at=now).measurement("selection_cost").value == 0
    missing = coverage_record(limited, run_id="ci", available_at=now).measurement("selection_cost")
    assert missing.value is None
    assert missing.status == "unavailable"


def test_zero_cost_candidates_zero_weights_and_global_lexicographic_ties():
    request = problem(0, candidates=[("free", 0, ("x",)), ("paid", 1, ("y",))])
    result = exact(request)
    assert result.selected_ids == ("free",)
    assert result.cost.value == 0
    assert result.objective.value == Decimal("0.5")
    # The zero-gain free A improves the *global tuple tie*, despite greedy omitting it.
    tied = problem(1, candidates=[("A", 0, ()), ("Z", 1, ("x",))], weights=[("x", 1)])
    assert solve_coverage(tied).selected_ids == ("Z",)
    assert exact(tied).selected_ids == ("A", "Z")
    empty = exact(problem(0, candidates=[], weights=[]))
    assert empty.selected_ids == ()
    assert empty.objective.value == 0
    assert empty.proof_prefixes == ("",)
    all_zero = exact(problem(0, candidates=[("A", 0, ("x",))], weights=[("x", 0)]))
    assert all_zero.selected_ids == ()


@pytest.mark.parametrize(
    "required,budget,maximum,search,reason",
    [
        (("A",), 1, None, None, "mandatory_budget_exceeded"),
        (("A", "B"), 5, 1, None, "mandatory_cardinality_exceeded"),
        (("A",), 5, None, ("B", "C"), "mandatory_outside_scope"),
    ],
)
def test_proven_infeasibility_is_separate_from_computation_limits(required, budget, maximum, search, reason):
    request = problem(budget, required=required, maximum=maximum, search=search)
    result = exact(request)
    assert independent_oracle(request) is None
    assert not result.feasible
    assert result.status == "infeasible"
    assert result.termination_reason == reason
    assert result.selected_ids is result.objective is result.upper_bound is result.absolute_gap is None
    with pytest.raises(ValueError, match="missing feasible"):
        result.checkpoint()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["candidates"].append(data["candidates"][0]),
        lambda data: data["outcomes"].append(data["outcomes"][0]),
        lambda data: data["candidates"][0]["covers"].append("x"),
        lambda data: data["candidates"][0]["covers"].append("unknown"),
        lambda data: data["constraints"].update(required_ids=["unknown"]),
        lambda data: data.update(search_ids=["unknown"]),
        lambda data: data.update(search_ids=["A", "A"]),
        lambda data: data["candidates"][0]["cost"]["semantics"].update(unit="ms"),
        lambda data: data["outcomes"][0]["weight"]["semantics"].update(meaning="different-v1"),
        lambda data: data["constraints"]["budget"].update(value="-1"),
        lambda data: data["candidates"][0]["cost"].update(value="1000000000001"),
        lambda data: data["outcomes"][0]["weight"].update(value="1000000000001"),
    ],
)
def test_invalid_identity_memberships_units_costs_and_weights(mutation):
    data = json.loads(problem().to_json())
    mutation(data)
    with pytest.raises(ValidationError):
        CoverageProblem.model_validate(data)


def test_decimal_arithmetic_and_canonical_units_have_zero_decision_tolerance():
    request = problem("0.3", candidates=[("A", "0.1", ("x",)), ("B", "0.2", ("y",)), ("C", "0.300000000001", ("z",))])
    result = exact(request)
    assert result.selected_ids == ("A", "B")
    assert result.cost.value == Decimal("0.3")
    with localcontext() as context:
        context.prec = 4
        same = exact(request)
    assert same.to_json() == result.to_json()
    payload = json.loads(request.to_json())
    payload["constraints"]["budget"]["value"] = "0.300000000000"
    assert CoverageProblem.model_validate(payload).fingerprint == request.fingerprint


def test_restricted_scope_never_inherits_full_universe_optimality():
    full = exact(problem())
    restricted = exact(problem(search=("B",)))
    assert restricted.status == "optimal"
    assert restricted.scope == "restricted"
    assert restricted.selected_ids == ("B",)
    assert restricted.objective.value < full.objective.value
    assert restricted.scope_fingerprint != full.scope_fingerprint
    payload = json.loads(restricted.to_json())
    payload["scope"] = "full"
    with pytest.raises(ValidationError, match="scope"):
        CoverageResult.model_validate(payload)


@pytest.mark.parametrize(
    "limits,reason",
    [
        ({"max_candidates": 0}, "candidate_limit"),
        ({"max_outcomes": 0}, "outcome_limit"),
        ({"max_memberships": 0}, "membership_limit"),
        ({"max_working_bytes": 0}, "memory_limit"),
    ],
)
def test_resource_preflight_rejects_before_search_without_claiming_infeasibility(limits, reason):
    result = exact(problem(), **limits)
    assert result.status == "limited"
    assert result.termination_reason == reason
    assert result.work_used == 0
    assert result.selected_ids is None


def test_non_overridable_exact_scope_and_certificate_limits_are_truthful():
    too_many = problem(candidates=[(f"test-{index:02d}", 1, ("x",)) for index in range(21)])
    assert exact(too_many).termination_reason == "exact_scope_limit"
    many_outcomes = [(f"o{index}", 1) for index in range(257)]
    too_broad = problem(candidates=[("test", 1, tuple(name for name, _ in many_outcomes))], weights=many_outcomes)
    assert exact(too_broad).termination_reason == "exact_scope_limit"
    limited = exact(problem(), max_certificate_leaves=1)
    assert limited.status == "feasible"
    assert limited.termination_reason == "certificate_limit"
    assert limited.proof_prefixes == ()
    assert limited.selected_ids == ("A", "C")


@pytest.mark.parametrize("method", ["greedy_density_v1", "branch_bound_v1"])
def test_overall_work_deadline_cancellation_and_interruption_keep_complete_incumbents(method):
    request = problem(required=("C",))
    for limits, reason in ((SearchLimits(max_work=0), "work_limit"), (SearchLimits(deadline_seconds=0), "deadline")):
        result = solve_coverage(request, method=method, limits=limits)
        assert result.termination_reason == reason
        assert result.selected_ids == ("C",)
        assert result.status == "feasible"
        assert result.work_used == 0
        assert result.upper_bound is not None
        assert result.upper_bound.value >= independent_oracle(request)[2]
    cancelled = solve_coverage(request, method=method, cancelled=lambda: True)
    assert cancelled.termination_reason == "cancelled"
    assert cancelled.selected_ids == ("C",)

    def interrupt():
        raise KeyboardInterrupt

    interrupted = solve_coverage(request, method=method, cancelled=interrupt)
    assert interrupted.termination_reason == "interrupted"
    assert interrupted.selected_ids == ("C",)
    warm = solve_coverage(problem(), method=method, limits=SearchLimits(max_work=3))
    assert warm.selected_ids == ("A",)
    assert warm.work_used == 3
    assert warm.termination_reason == "work_limit"


def test_checkpoint_continuation_uses_existing_store_and_never_restores_an_unproved_bound(tmp_path):
    request = problem()
    partial = exact(request, max_work=3)
    checkpoint = partial.checkpoint()
    store = LocalCheckpointStore(str(tmp_path))
    store.save(
        CheckpointPayload(run_id="ci-coverage", step=partial.work_used, extra={"coverage_v1": checkpoint.to_json()})
    )
    saved = store.load("ci-coverage", 0)
    assert saved is not None
    restored = CoverageCheckpoint.from_json(saved.extra["coverage_v1"])
    continued = solve_coverage(request, method="branch_bound_v1", checkpoint=restored)
    assert continued.selected_ids == ("A", "C")
    assert continued.status == "optimal"
    assert continued.objective is not None
    assert continued.objective.value >= partial.objective.value
    changed = problem(2)
    with pytest.raises(ValueError, match="Checkpoint differs"):
        solve_coverage(changed, checkpoint=restored)
    forged = restored.model_copy(update={"selected_ids": ("A", "B", "C")})
    with pytest.raises(ValueError, match="not feasible"):
        solve_coverage(request, checkpoint=forged)


def test_exhaustive_oracles_verify_all_bounds_across_finite_random_instances():
    rng = random.Random(253)
    for index in range(80):
        weights = [(f"o{outcome}", str(rng.randrange(6) / 10)) for outcome in range(4)]
        candidates = [
            (f"c{candidate}", rng.randrange(4), tuple(name for name, _ in weights if rng.randrange(2)))
            for candidate in range(rng.randrange(9))
        ]
        required = tuple(name for name, _, _ in candidates if rng.randrange(8) == 0)
        request = problem(
            rng.randrange(8),
            candidates=candidates,
            weights=weights,
            required=required,
            maximum=None if index % 2 else rng.randrange(5),
        )
        expected = independent_oracle(request)
        result = exact(request)
        if expected is None:
            assert result.status == "infeasible"
            continue
        assert result.status == "optimal"
        assert result.selected_ids == expected[0]
        assert result.cost.value == expected[1]
        assert result.objective.value == expected[2]
        for work in (0, 3, 11):
            limited = exact(request, max_work=work)
            assert limited.work_used <= work
            assert limited.upper_bound.value >= expected[2]
            assert limited.objective.value <= expected[2]
        greedy = solve_coverage(request)
        assert greedy.objective is not None
        assert greedy.upper_bound is not None
        assert greedy.objective.value <= expected[2]
        assert greedy.upper_bound.value >= expected[2]


def test_metamorphic_order_scale_budget_and_coverage_union():
    request = problem()
    payload = json.loads(request.to_json())
    payload["candidates"].reverse()
    payload["outcomes"].reverse()
    payload["search_ids"].reverse()
    for candidate in payload["candidates"]:
        candidate["covers"].reverse()
    reordered = CoverageProblem.model_validate(payload)
    assert reordered.fingerprint == request.fingerprint
    assert exact(reordered).to_json() == exact(request).to_json()
    for outcome in payload["outcomes"]:
        outcome["weight"]["value"] = str(Decimal(outcome["weight"]["value"]) * 10)
    scaled = exact(CoverageProblem.model_validate(payload))
    assert scaled.selected_ids == exact(request).selected_ids
    assert scaled.objective.value == 10
    objectives = [exact(problem(budget)).objective.value for budget in range(6)]
    assert objectives == sorted(objectives)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(proof_prefixes=[]),
        lambda data: data.update(proof_prefixes=[""]),
        lambda data: data["proof_prefixes"].pop(),
        lambda data: data["proof_prefixes"].append(data["proof_prefixes"][0]),
        lambda data: data.update(proof_prefixes=["2"]),
        lambda data: data["upper_bound"].update(value="100"),
        lambda data: data["absolute_gap"].update(value="1"),
        lambda data: data["objective"].update(value="1"),
        lambda data: data.update(work_used=10000001),
        lambda data: data.update(termination_reason="deadline"),
    ],
)
def test_forged_or_incomplete_certificates_never_become_optimal_claims(mutation):
    data = json.loads(exact(problem(2)).to_json())
    mutation(data)
    with pytest.raises(ValidationError):
        CoverageResult.model_validate(data)


def test_serialization_versions_and_golden_artifacts():
    expected = json.loads((FIXTURES / "digests-v1.json").read_text())
    models = {
        "problem": CoverageProblem,
        "optimal": CoverageResult,
        "limited": CoverageResult,
        "checkpoint": CoverageCheckpoint,
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
    partial = exact(problem(), max_work=3)
    fresh = {"problem": problem(), "optimal": exact(problem()), "limited": partial, "checkpoint": partial.checkpoint()}
    assert {name: model.fingerprint for name, model in fresh.items()} == expected
    content = problem().to_json().replace('"schema_version":1', '"schema_version":1,"schema_version":1')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        CoverageProblem.from_json(content)


def _publish_worker(root, serialized):
    result = CoverageResult.from_json(serialized)
    return str(publish_coverage_result(result, ArtifactWriter(Path(root))))


def test_atomic_publication_parallel_idempotence_manifest_and_sink_replay(tmp_path):
    result = exact(problem())
    with ProcessPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(_publish_worker, [str(tmp_path)] * 6, [result.to_json()] * 6))
    assert len(set(paths)) == 1
    assert CoverageResult.from_json(Path(paths[0]).read_bytes()) == result
    manifest = json.loads(generate_manifest(tmp_path, run_id="ci").read_text())
    assert len([item for item in manifest["artifacts"] if item["type"] == "coverage_result"]) == 1
    record = coverage_record(result, run_id="ci", available_at=datetime(2026, 1, 1, tzinfo=UTC))
    sink = ParquetSink(out_dir=str(tmp_path / "results"))
    try:
        sink.write_domain_record(record)
        sink.write_domain_record(record)
        assert list(sink.domain_records()) == [record]
        assert record.learning_reward is None
        assert record.measurement("coverage_objective").meaning.endswith(":binary64-projection-v1")
    finally:
        sink.close()


def test_interrupted_publication_never_leaves_a_completed_artifact(tmp_path, monkeypatch):
    result = exact(problem())

    def fail_link(*args, **kwargs):
        raise OSError("interrupted publication")

    monkeypatch.setattr("coleman.artifacts.os.link", fail_link)
    writer = ArtifactWriter(tmp_path)
    with pytest.raises(OSError, match="interrupted"):
        publish_coverage_result(result, writer)
    assert list(tmp_path.rglob("*.json")) == []


def test_core_import_boundaries_and_stable_serialization_across_hash_seeds(tmp_path):
    for file in (ROOT / "coleman" / "selection").glob("*.py"):
        tree = ast.parse(file.read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(
            name.startswith(("coleman.cli", "coleman.runner", "coleman.evaluation", "mcp")) for name in imports
        )
    script = """from coleman.selection import CoverageProblem, SearchLimits, solve_coverage
import sys
r = solve_coverage(CoverageProblem.from_json(sys.stdin.read()), method="branch_bound_v1",
                   limits=SearchLimits(deadline_seconds=None))
print(r.to_json())
assert 'coleman.cli' not in sys.modules
"""
    outputs = []
    for seed in ("1", "999"):
        environment = dict(os.environ, PYTHONHASHSEED=seed)
        outputs.append(
            subprocess.run(
                [sys.executable, "-c", script],
                input=problem().to_json(),
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
        )
    assert outputs[0] == outputs[1]


def test_installed_wheel_native_tcpci_and_synthetic_resource_examples(tmp_path):
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
        shutil.copy(ROOT / "examples" / "finite-coverage" / filename, script)
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
        assert output["selected_ids"] == ["A", "C"]
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
            assert output["native_scheduled"] == ["A", "C"]


def test_proof_budget_and_minimum_node_work_cannot_be_forged():
    original = exact(problem())
    payload = json.loads(original.to_json())
    payload["limits"]["max_certificate_leaves"] = 1
    # Recompute the declared memory estimate independently of the old report.
    from coleman.selection.resources import inspect_workload

    limits = SearchLimits.model_validate(payload["limits"])
    payload["preflight"] = inspect_workload(original.problem, original.method, limits).model_dump()
    with pytest.raises(ValidationError, match="declared proof limit"):
        CoverageResult.model_validate(payload)
    payload = json.loads(original.to_json())
    payload["work_used"] = 0
    with pytest.raises(ValidationError, match="complete proof tree"):
        CoverageResult.model_validate(payload)


def test_future_checkpoint_rejected_even_when_preflight_would_refuse_search():
    request = problem()
    future = exact(request).checkpoint().model_copy(update={"schema_version": 2})
    limits = SearchLimits(max_working_bytes=0)
    with pytest.raises(ValidationError):
        solve_coverage(request, checkpoint=future, limits=limits)


def test_large_decimal_aggregates_preserve_all_digits_under_low_context_precision():
    request = problem(
        2,
        candidates=[("A", 1, ("x",)), ("B", 1, ("y",))],
        weights=[("x", "999999999999.123456789012"), ("y", "999999999999.123456789012")],
    )
    with localcontext() as context:
        context.prec = 4
        result = exact(request)
    assert result.objective.value == Decimal("1999999999998.246913578024")
    assert result.absolute_gap.value == 0
    assert CoverageResult.from_json(result.to_json()) == result


def test_certificate_rejects_wrong_global_lexicographic_tie():
    request = problem(1, candidates=[("A", 0, ()), ("Z", 1, ("x",))], weights=[("x", 1)])
    payload = json.loads(exact(request).to_json())
    payload["selected_ids"] = ["Z"]
    with pytest.raises(ValidationError, match="better feasible selection"):
        CoverageResult.model_validate(payload)


def test_exact_verification_stays_inside_a_small_restricted_scope():
    candidates = [(f"test-{index:04d}", 1, ("x",)) for index in range(200)]
    request = problem(2, candidates=candidates, search=("test-0001", "test-0002"))
    result = exact(request)
    assert result.scope == "restricted"
    assert result.status == "optimal"
    # Equal objective with optional zero-gain additions uses the full tuple tie rule.
    assert result.selected_ids == ("test-0001",)
    assert result.objective.value == Decimal("0.5")
    assert CoverageResult.from_json(result.to_json()) == result


def test_signed_zero_is_canonical_before_binary64_projection():
    original = exact(problem(0))
    payload = json.loads(original.to_json())
    payload["cost"]["value"] = "-0"
    payload["objective"]["value"] = "-0.000000000000"
    normalized = CoverageResult.model_validate(payload)
    assert normalized.to_json() == original.to_json()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert (
        coverage_record(normalized, run_id="ci", available_at=now).to_json()
        == coverage_record(original, run_id="ci", available_at=now).to_json()
    )


@pytest.mark.parametrize("selected", [("unknown",), ("A", "B", "C")])
def test_corrupt_checkpoint_is_rejected_before_resource_preflight(selected):
    request = problem()
    checkpoint = CoverageCheckpoint(schema_version=1, problem_fingerprint=request.fingerprint, selected_ids=selected)
    limits = SearchLimits(max_working_bytes=0)
    with pytest.raises(ValueError, match="checkpoint candidate|not feasible"):
        solve_coverage(request, checkpoint=checkpoint, limits=limits)


def test_corrupt_checkpoint_is_rejected_before_infeasibility_result():
    request = problem(1, required=("A",))
    checkpoint = CoverageCheckpoint(schema_version=1, problem_fingerprint=request.fingerprint, selected_ids=("A",))
    with pytest.raises(ValueError, match="not feasible"):
        solve_coverage(request, checkpoint=checkpoint)
