import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from coleman.core.actions import ChoiceAction
from coleman.core.feedback import ChosenFeedback, FeedbackBatch, FullPanel
from coleman.interchange.codec import wire_fingerprint
from coleman.testing.decision_contracts import check_decision_contracts

ROOT = Path(__file__).parents[2]
GOLDEN = ROOT / "tests/fixtures/interchange/v1"


def _batch(example, consumer, case, index):
    executed = consumer.export_state().pending
    assert executed is not None
    if case.problem.feedback_kind == "chosen":
        delivered = ChosenFeedback(row=next(row for row in case.outcomes if row.member == executed.action.chosen))
    else:
        delivered = FullPanel(rows=case.outcomes)
    return FeedbackBatch(
        schema_version=1,
        decision_fingerprint=executed.fingerprint,
        feedback_id=f"synthetic-feedback-{index}",
        revision_id="root",
        supersedes=None,
        horizon=None,
        feedback=delivered,
    )


def test_full_information_host_learns_from_actual_complete_panels(examples):
    example = examples["learning_only"]
    report = check_decision_contracts(example.HostLearner, example.capability(), example.cases())
    assert [cast(ChoiceAction, decision.action).chosen.id for decision in report.decisions] == ["recipe-a", "recipe-b"]
    assert all(len(cast(FullPanel, batch.feedback).rows) == 2 for batch in report.feedback)
    learner = example.HostLearner()
    learner.decide(example.cases()[0].request)
    learner.update(_batch(example, learner, example.cases()[0], 0))
    assert learner.export_state().totals == (1.0, 0.0)
    before = learner.export_state()
    with pytest.raises(ValueError, match="No example decision"):
        learner.update(report.feedback[0])
    assert learner.export_state() == before
    with pytest.raises(ValueError):
        learner.restore_state(before.model_copy(update={"totals": (-1.0, 0.0)}))
    assert learner.export_state() == before


def test_real_chosen_only_mab_has_sampling_effects_and_complete_rng_restore(examples):
    example = examples["mab_only"]
    cases = example.cases()
    report = check_decision_contracts(example.HostMAB, example.capability(), cases)
    assert [cast(ChoiceAction, decision.action).chosen.id for decision in report.decisions] == ["recipe-a"] * 3 + [
        "recipe-b"
    ] * 5
    assert [decision.sampling.probability for decision in report.decisions] == [0.75] * 3 + [0.25] + [0.75] * 4
    assert all(isinstance(batch.feedback, ChosenFeedback) for batch in report.feedback)
    learner = example.HostMAB()
    resumed = example.HostMAB(seed=999)
    for index, case in enumerate(cases):
        if index == 4:
            snapshot = example.MABState.from_json(learner.export_state().to_json())
            resumed.restore_state(snapshot)
            assert resumed.export_state() == learner.export_state()
        proposal = learner.decide(case.request)
        if index >= 4:
            assert resumed.decide(case.request).to_json() == proposal.to_json()
        pending = example.MABState.from_json(learner.export_state().to_json())
        # Restoring while an outcome is pending preserves RNG, exact action and the update binding.
        pending_host = example.HostMAB(seed=0)
        pending_host.restore_state(pending)
        batch = _batch(example, learner, case, index)
        learner.update(batch)
        pending_host.update(batch)
        assert pending_host.export_state() == learner.export_state()
        if index >= 4:
            resumed.update(batch)
            assert resumed.export_state() == learner.export_state()
    state = learner.export_state()
    assert state.counts == (3, 5)
    assert state.rewards == (0.0, 5.0)
    assert sum(state.counts) == len(cases)
    assert state.words != example.HostMAB().export_state().words
    before = resumed.export_state()
    for update in ({"schema_version": 2}, {"words": before.words[:-1]}, {"counts": (65, 0)}):
        with pytest.raises(ValueError):
            resumed.restore_state(before.model_copy(update=update))
        assert resumed.export_state() == before


def test_chosen_feedback_changes_best_arm_under_same_owned_rng(examples):
    example = examples["mab_only"]
    baseline, changed = example.HostMAB(), example.HostMAB()
    for index, case in enumerate(example.cases()[:4]):
        baseline.decide(case.request)
        changed.decide(case.request)
        batch = _batch(example, baseline, case, index)
        baseline.update(batch)
        assert isinstance(batch.feedback, ChosenFeedback)
        row = batch.feedback.row
        assert row.values[0].value is not None
        inverted = row.model_copy(
            update={"values": (row.values[0].model_copy(update={"value": 1.0 - row.values[0].value}),)}
        )
        altered = batch.model_copy(update={"feedback": ChosenFeedback(row=inverted)})
        changed.update(altered)
    assert baseline.export_state().rewards == (0.0, 1.0)
    assert changed.export_state().rewards == (3.0, 0.0)
    next_request = example.cases()[4].request
    assert baseline.decide(next_request).action.chosen.id == "recipe-b"
    assert changed.decide(next_request).action.chosen.id == "recipe-a"
    assert baseline.export_state().words == changed.export_state().words


def test_existing_finite_solver_independent_oracle_and_candidate_order(examples):
    example = examples["solver_only"]
    from coleman.selection import SearchLimits, solve_coverage

    request = example.problem()
    expected = example.independent_answer(request)
    assert expected[0] == ("A", "C")
    assert expected[1] == 3
    assert expected[2] == 1
    for candidates in (request.candidates, tuple(reversed(request.candidates))):
        supplied = type(request).model_validate(request.model_copy(update={"candidates": candidates}))
        result = solve_coverage(supplied, method="branch_bound_v1", limits=SearchLimits(deadline_seconds=None))
        assert result.selected_ids == expected[0]
        assert result.cost is not None
        assert result.objective is not None
        assert result.cost.value == expected[1]
        assert result.objective.value == expected[2]


def test_foreign_consumer_reads_actual_result_with_no_sdk_import(tmp_path):
    request = GOLDEN / "coverage-request.json"
    response = GOLDEN / "coverage-response.json"
    manifest = json.loads((GOLDEN / "manifest.json").read_text())["fixtures"]
    payload = json.loads(response.read_text())
    arguments = [
        "--request",
        str(request),
        "--response",
        str(response),
        "--request-wire",
        manifest[request.name]["wire_fingerprint"],
        "--response-wire",
        manifest[response.name]["wire_fingerprint"],
        "--request-fingerprint",
        payload["request_fingerprint"],
        "--input-fingerprint",
        payload["input_fingerprint"],
        "--received-at",
        "2026-01-01T10:02:00Z",
    ]
    guard = """
import importlib.abc, runpy, sys
class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname=='coleman' or fullname.startswith('coleman.'):
            raise AssertionError('Foreign host imported SDK: '+fullname)
sys.meta_path.insert(0, NoSDK())
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", guard, str(ROOT / "examples/composition/foreign_consumer.py"), *arguments],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual["selected"] == ["A", "C"]
    assert actual["status"] == "optimal"
    assert actual["cost"] == {"value": "3", "semantics": {"unit": "s", "meaning": "modeled-serial-suite-duration-v1"}}
    assert actual["sdk_imported"] is False


@pytest.mark.parametrize(
    "change",
    [
        "receipt",
        "request_wire",
        "response_wire",
        "request_fingerprint",
        "input_fingerprint",
        "attempt",
        "version",
        "duplicate",
    ],
)
def test_foreign_host_rejects_stale_mismatched_unknown_and_duplicate_evidence(examples, tmp_path, change):
    source = GOLDEN / "coverage-request.json"
    output = GOLDEN / "coverage-response.json"
    request_path, response_path = tmp_path / "request.json", tmp_path / "response.json"
    request_path.write_bytes(source.read_bytes())
    response_path.write_bytes(output.read_bytes())
    response = json.loads(output.read_text())
    arguments = argparse.Namespace(
        request=request_path,
        response=response_path,
        request_wire=wire_fingerprint(request_path.read_bytes()),
        response_wire=wire_fingerprint(response_path.read_bytes()),
        request_fingerprint=response["request_fingerprint"],
        input_fingerprint=response["input_fingerprint"],
        received_at="2026-01-01T10:02:00Z",
    )
    if change == "receipt":
        arguments.received_at = "2026-01-01T10:06:00Z"
    elif change in {"request_wire", "response_wire", "request_fingerprint", "input_fingerprint"}:
        setattr(arguments, change, "sha256:" + "0" * 64)
    elif change == "attempt":
        payload = json.loads(request_path.read_text())
        payload["attempt_id"] = "another-attempt"
        request_path.write_text(json.dumps(payload))
        arguments.request_wire = wire_fingerprint(request_path.read_bytes())
    elif change == "version":
        response["schema_version"] = 2
        response_path.write_text(json.dumps(response))
        arguments.response_wire = wire_fingerprint(response_path.read_bytes())
    else:
        response_path.write_text(
            output.read_text().replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1)
        )
        arguments.response_wire = wire_fingerprint(response_path.read_bytes())
    with pytest.raises(ValueError):
        examples["foreign_consumer"].consume(arguments)


def test_import_direction_keeps_optional_exchange_out_of_core_and_publication_explicit():
    import ast

    core = ROOT / "coleman/core"
    assert list(core.glob("*.py"))
    for path in core.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            else:
                continue
            assert not any(
                name == "coleman.interchange" or name.startswith("coleman.interchange.") for name in imports
            ), path
    for name in ("__init__", "contracts", "codec"):
        path = ROOT / f"coleman/interchange/{name}.py"
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "coleman.artifacts", path


@pytest.mark.parametrize("boundary", ["request_problem", "result", "result_problem"])
@pytest.mark.parametrize("version", [None, True, "1", 1.0, 2])
def test_foreign_host_rejects_unknown_nested_coverage_schemas(examples, tmp_path, boundary, version):
    request = json.loads((GOLDEN / "coverage-request.json").read_text())
    response = json.loads((GOLDEN / "coverage-response.json").read_text())
    if boundary == "request_problem":
        targets = (request["input"]["problem"], response["request"]["input"]["problem"])
    elif boundary == "result":
        targets = (response["reply"]["result"],)
    else:
        targets = (response["reply"]["result"]["problem"],)
    for target in targets:
        if version is None:
            del target["schema_version"]
        else:
            target["schema_version"] = version
    request_path, response_path = tmp_path / "request.json", tmp_path / "response.json"
    request_path.write_text(json.dumps(request))
    response_path.write_text(json.dumps(response))
    arguments = argparse.Namespace(
        request=request_path,
        response=response_path,
        request_wire=wire_fingerprint(request_path.read_bytes()),
        response_wire=wire_fingerprint(response_path.read_bytes()),
        request_fingerprint=response["request_fingerprint"],
        input_fingerprint=response["input_fingerprint"],
        received_at="2026-01-01T10:02:00Z",
    )
    with pytest.raises(ValueError, match="schema"):
        examples["foreign_consumer"].consume(arguments)


def test_tiny_independent_oracle_does_not_ignore_declared_restricted_search(examples):
    example = examples["solver_only"]
    problem = example.problem()
    restricted = type(problem).model_validate(problem.model_copy(update={"search_ids": ("B",)}))
    with pytest.raises(ValueError, match="full-scope"):
        example.independent_answer(restricted)


@pytest.mark.parametrize("name", ["learning_only", "mab_only"])
def test_bounded_host_rejects_a_65th_opportunity_before_changing_owned_state(examples, name):
    example = examples[name]
    consumer = example.HostLearner() if name == "learning_only" else example.HostMAB()
    original = example.cases()[0]
    for index in range(64):
        request = type(original.request).model_validate(
            original.request.model_copy(update={"decision_id": f"bounded-build-{index}", "round": index})
        )
        case = type(original).model_validate(original.model_copy(update={"request": request}))
        consumer.decide(request)
        consumer.update(_batch(example, consumer, case, index))
    before = consumer.export_state().to_json()
    request = type(original.request).model_validate(
        original.request.model_copy(update={"decision_id": "bounded-build-64", "round": 64})
    )
    with pytest.raises(ValueError, match="64"):
        consumer.decide(request)
    assert consumer.export_state().to_json() == before
