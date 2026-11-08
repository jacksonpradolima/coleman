"""Load actual public installed-package examples without a replacement host."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def learning_example():
    name = "coleman_public_learning_ci_example"
    source = Path(__file__).parents[2] / "examples/learning/ci.py"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)
