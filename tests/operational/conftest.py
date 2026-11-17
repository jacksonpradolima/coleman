"""Explicit source-owned helpers; no replacement mathematical components."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ci_example():
    return load("coleman_operational_ci_tests", ROOT / "examples/operational/ci.py")


@pytest.fixture(scope="module")
def replay_example():
    return load("coleman_operational_replay_tests", ROOT / "examples/replay/ci.py")


@pytest.fixture(scope="module")
def measurement():
    return load("coleman_operational_measurement_tests", ROOT / "examples/operational/measure.py")
