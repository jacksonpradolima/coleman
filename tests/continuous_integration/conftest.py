"""Load the explicitly selected public chronological example, without mocks."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def continuous_example():
    name = "authored_continuous_integration_flow"
    source = Path(__file__).parents[2] / "examples/continuous/flow.py"
    specification = importlib.util.spec_from_file_location(name, source)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module
