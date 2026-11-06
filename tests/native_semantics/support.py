"""Small authored CI inputs and actual native calls, without learner stand-ins."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from coleman.agent import Agent, ContextualAgent
from coleman.bandit import EvaluationMetricBandit
from coleman.evaluation import NAPFDVerdictMetric

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests/fixtures/native-semantics/v1"


def fixture(name: str) -> Any:
    """Read consciously authored public fixture data, never executable state."""
    return json.loads((FIXTURES / name).read_text())


def rng_state(agent: Agent) -> dict[str, Any]:
    """Copy the whole owned PCG64 state, including the cached half-word."""
    return copy.deepcopy(dict(agent._rng.bit_generator.state))


def bandit(verdicts: dict[str, int], *, budget: float | None = None) -> EvaluationMetricBandit:
    """Create real unit-second CI tests and the real bounded verdict metric."""
    metric = NAPFDVerdictMetric()
    metric.update_available_time(float(len(verdicts)) if budget is None else budget)
    return EvaluationMetricBandit(
        [
            {
                "Name": name,
                "Duration": 1.0,
                "Verdict": verdict,
                "CalcPrio": 0,
                "LastRun": "2026-01-01T00:00:00Z",
                "NumRan": 0,
                "NumErrors": verdict,
                "LastResults": "[]",
            }
            for name, verdict in verdicts.items()
        ],
        metric,
    )


def prepare(agent: Agent, verdicts: dict[str, int], *, budget: float | None = None) -> EvaluationMetricBandit:
    """Supply a predeclared dimensionless contextual constant before actual selection."""
    if isinstance(agent, ContextualAgent):
        agent.update_context(pl.DataFrame({"Name": list(verdicts), "signal": [1.0] * len(verdicts)}))
        agent.update_features(["signal"])
    actual = bandit(verdicts, budget=budget)
    agent.update_bandit(actual)
    return actual


def cycle(agent: Any, verdicts: dict[str, int], *, budget: float | None = None) -> dict[str, Any]:
    """Execute choose, real native metric evaluation and real native credit exactly once."""
    actual = prepare(agent, verdicts, budget=budget)
    order = agent.choose()
    outcome = actual.pull(order)
    credits = agent.reward_function.evaluate(outcome, order)
    agent.observe(outcome)
    return {
        "ranking": order,
        "scheduled": list(outcome.scheduled_testcases),
        "unscheduled": list(outcome.unscheduled_testcases),
        "credits": list(credits),
        "actions": agent.actions.to_dicts(),
    }


class _WithoutDocstrings(ast.NodeTransformer):
    """Remove only first-position docstring expressions, retaining all executable constants."""

    def generic_visit(self, node: Any) -> Any:
        super().generic_visit(node)
        if (
            isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node


def mechanism_digest(source: str) -> str:
    """Hash the CPython 3.14 AST after removing docstrings and location metadata only."""
    tree = _WithoutDocstrings().visit(ast.parse(source, feature_version=(3, 14)))
    encoded = ast.dump(tree, annotate_fields=True, include_attributes=False).encode()
    return hashlib.sha256(encoded).hexdigest()
