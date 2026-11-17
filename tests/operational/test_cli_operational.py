"""Real Click argument boundaries and lazy optional command imports."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SOURCES = ROOT / "examples/operational"


@pytest.mark.parametrize("filename", ("ci.py", "measure.py", "bootstrap.py", "tasks.py"))
def test_operational_api_imports_do_not_require_click_and_missing_cli_is_useful(tmp_path, filename):
    """Reject a Click import through a real import boundary, before any work."""
    program = """
import importlib.abc,json,runpy,sys
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname == 'click' or fullname.startswith('click.'):
            raise ModuleNotFoundError('Click deliberately absent at this import boundary')
sys.meta_path.insert(0,Boundary())
namespace=runpy.run_path(sys.argv[1])
assert 'click' not in sys.modules
if sys.argv[1].endswith('/ci.py'):
    result=namespace['run'](namespace['Workload'](schema_version=1),'noop')
    assert result.final_states[0].payload.update_count == 2
try:
    namespace['main']()
except SystemExit as error:
    assert '[cli] extra' in str(error)
else:
    raise AssertionError('Missing CLI dependency must refuse before command work')
print(json.dumps({'lazy':True,'guidance':True}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(SOURCES / filename)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"lazy": True, "guidance": True}
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    "arguments",
    (
        ("off", "1", "2", "1"),
        ("off", "33", "2", "1"),
        ("off", "2", "0", "1"),
        ("off", "2", "33", "1"),
        ("off", "2", "2", "0"),
        ("off", "2", "2", "5"),
        ("unknown", "2", "2", "1"),
    ),
)
def test_click_child_types_enforce_original_workload_caps_before_math(tmp_path, arguments):
    """Actual Click rejects invalid heterogeneous values without publishing."""
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(SOURCES / "measure.py"), "--child", *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Invalid value" in result.stderr
    assert "Traceback" not in result.stderr
    assert not tuple(tmp_path.iterdir())
