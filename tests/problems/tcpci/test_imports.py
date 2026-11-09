"""Hostile fresh import proves that the pure domain mapping has no native edge."""

import subprocess
import sys
import sysconfig
from pathlib import Path


def test_pure_domain_import_without_site_startup_or_optional_engines():
    """Block heavy/runtime modules before importing every new public mapping API."""
    root = Path(__file__).parents[3]
    code = r"""
import importlib.abc
import sys
sys.path[:0] = list(dict.fromkeys(sys.argv[1:]))
blocked = (
    'coleman.application', 'coleman.native', 'coleman.agent', 'coleman.selection',
    'coleman.learning', 'coleman.replay', 'coleman.state', 'coleman.tasks',
    'coleman.transports', 'numpy', 'polars', 'pyarrow', 'mcp', 'click', 'rich', 'typer',
)
def forbidden(name):
    return any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)
assert not any(forbidden(name) for name in sys.modules)
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            raise AssertionError('Forbidden pure mapping import: ' + fullname)
        return None
sys.meta_path.insert(0, Guard())
from coleman.problems.tcpci import BudgetPrefix, TcpciMapping, TcpciProblem, TestCase, map_ranking, ranking_request
public = (BudgetPrefix, TcpciMapping, TcpciProblem, TestCase, map_ranking, ranking_request)
assert all(callable(value) for value in public)
assert not any(forbidden(name) for name in sys.modules)
print('pure TCPCI boundary passed')
"""
    actual = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            code,
            str(root),
            sysconfig.get_path("purelib"),
            sysconfig.get_path("platlib"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert actual.returncode == 0, actual.stdout + actual.stderr
    assert actual.stdout.strip() == "pure TCPCI boundary passed"
