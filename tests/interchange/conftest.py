import importlib.util
from datetime import datetime
from pathlib import Path

import pytest

from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.interchange.contracts import (
    CoverageCall,
    CoverageReply,
    InterchangeRequest,
    InterchangeResponse,
)
from coleman.selection import SearchLimits, solve_coverage

ROOT = Path(__file__).parents[2]
GOLDEN = ROOT / "tests/fixtures/interchange/v1"


def load_example(name):
    spec = importlib.util.spec_from_file_location("composition_" + name, ROOT / f"examples/composition/{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def examples():
    return {name: load_example(name) for name in ("learning_only", "mab_only", "solver_only", "foreign_consumer")}


@pytest.fixture
def coverage_pair(examples):
    request = InterchangeRequest(
        schema_version=1,
        task=Reference(id="synthetic-ci-suite-coverage", version="v1"),
        exchange_id="build-42-coverage",
        attempt_id="attempt-1",
        implementation=Reference(id="branch_bound_v1", version="coverage-v1"),
        state=None,
        created_at=datetime.fromisoformat("2026-01-01T10:00:00Z"),
        expires_at=datetime.fromisoformat("2026-01-01T10:05:00Z"),
        input=CoverageCall(
            problem=examples["solver_only"].problem(),
            method="branch_bound_v1",
            limits=SearchLimits(deadline_seconds=None),
        ),
    )
    call = request.input
    assert isinstance(call, CoverageCall)
    reply = CoverageReply(result=solve_coverage(call.problem, method=call.method, limits=call.limits))
    response = InterchangeResponse(
        schema_version=1,
        request=request,
        request_fingerprint=request.fingerprint,
        input_fingerprint=request.input_fingerprint,
        reply_fingerprint=fingerprint(reply.model_dump(mode="json")),
        implementation=request.implementation,
        produced_at=datetime.fromisoformat("2026-01-01T10:01:00Z"),
        state=None,
        reply=reply,
    )
    return request, response


@pytest.fixture
def respond():
    def build(request, reply, *, produced_at=None, state=None):
        return InterchangeResponse(
            schema_version=1,
            request=request,
            request_fingerprint=request.fingerprint,
            input_fingerprint=request.input_fingerprint,
            reply_fingerprint=fingerprint(reply.model_dump(mode="json")),
            implementation=request.implementation,
            produced_at=produced_at,
            state=state,
            reply=reply,
        )

    return build
