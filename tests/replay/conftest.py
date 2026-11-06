"""Load the actual public copied CI host, never a replacement replay engine."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="session")
def example():
    name = "coleman_authored_replay_ci_host"
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples/replay/ci.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)
