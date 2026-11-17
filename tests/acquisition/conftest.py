"""Load the standalone public CI experiment without importing an application."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def ci_pool():
    path = Path(__file__).parents[2] / "examples/acquisition/ci_pool.py"
    spec = importlib.util.spec_from_file_location("coleman_acquisition_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
