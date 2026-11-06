"""Load the public host example; every principal state/replay/store stays real."""

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.state.support import pending_collection

ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="session")
def state_example():
    name = "coleman_authored_owned_state_ci"
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples/state/ci.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture(scope="session")
def pending_evidence(state_example):
    sessions, records, domain = pending_collection(state_example)
    return (
        tuple(replay.export_continuation() for replay, _ in sessions),
        tuple(tuple(row) for row in records),
        domain.export_state(),
    )
