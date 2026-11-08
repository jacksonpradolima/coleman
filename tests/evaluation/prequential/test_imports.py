"""Cold pure namespaces must not load transports, stores or analytics providers."""

import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest


@pytest.mark.parametrize("leaf", ["", ".contracts", ".conformance", ".evidence", ".comparison", ".execution"])
def test_cold_prequential_boundary_uses_only_its_declared_lightweight_dependencies(leaf):
    script = r"""
import importlib
import importlib.abc
import sys
sys.path[:0] = sys.argv[2:]

blocked = (
    'coleman.cli', 'coleman.transports', 'coleman.application', 'mcp',
    'coleman.state.local', 'coleman.evaluation.design.audit', 'sqlite3',
    'numpy', 'pandas', 'polars', 'pyarrow', 'scipy',
)
assert not any(
    name == prefix or name.startswith(prefix + '.')
    for name in sys.modules for prefix in blocked
), 'Unexpected startup dependency before the tested import'
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Undeclared dependency: ' + fullname)
sys.meta_path.insert(0, Guard())
module = importlib.import_module('coleman.evaluation.prequential' + sys.argv[1])
if not sys.argv[1]:
    before = set(sys.modules)
    assert 'DomainExecutionReceipt' in dir(module)
    assert 'DiagnosticAudit' in dir(module)
    assert set(sys.modules) == before
    assert module.DomainExecutionReceipt.__name__ == 'DomainExecutionReceipt'
    try:
        getattr(module, 'missing_prequential_export')
    except AttributeError:
        pass
    else:
        raise AssertionError('Unknown exports must reject')
assert not any(
    name == prefix or name.startswith(prefix + '.')
    for name in sys.modules for prefix in blocked
)
print('cold boundary passed')
"""
    # Mount selected package paths without running .pth coverage startup hooks.
    # Every preloaded and newly imported module remains visible to the guard.
    paths = [
        str(Path(__file__).resolve().parents[3]),
        *sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}),
    ]
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, leaf, *paths],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.stdout.strip() == "cold boundary passed"
