"""Real Click declarations preserve the pure portfolio import boundary."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.synthetic_portfolio.recipe import recipe
from examples.synthetic_portfolio.sample import planning_request

ROOT = Path(__file__).parents[2]
PROGRAM = """
import runpy,sys
sys.path.insert(0,sys.argv[1])
sys.argv=['coleman-portfolio-example',*sys.argv[2:]]
runpy.run_module('examples.synthetic_portfolio.run',run_name='__main__')
"""


def invoke(arguments, directory):
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c", PROGRAM, str(ROOT), *arguments],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_library_import_is_click_free_and_missing_extra_has_guidance_before_work(tmp_path):
    program = """
import importlib,importlib.abc,json,sys
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname == 'click' or fullname.startswith('click.'):
            raise ModuleNotFoundError('Deliberately absent optional CLI extra')
sys.meta_path.insert(0,Boundary())
sys.path.insert(0,sys.argv[1])
module=importlib.import_module('examples.synthetic_portfolio.run')
assert 'click' not in sys.modules
try:
    module.main()
except SystemExit as error:
    assert '[cli] extra' in str(error)
else:
    raise AssertionError('Missing CLI dependency must reject before work')
print(json.dumps({'lazy':True,'guidance':True}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(ROOT)],
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
    ("arguments", "message"),
    [(("--unknown-option",), "No such option"), (("--output",), "requires an argument")],
)
def test_click_rejects_bad_flags_without_experiment_artifacts(tmp_path, arguments, message):
    result = invoke(arguments, tmp_path)
    assert result.returncode == 2 and message in result.stderr
    assert "Traceback" not in result.stderr and not tuple(tmp_path.iterdir())


def test_click_help_retains_all_flags_and_original_default_output(tmp_path):
    result = invoke(("--help",), tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(flag in result.stdout for flag in ("--output", "--recipe", "--task-request", "--protected"))
    assert "synthetic-portfolio-artifacts" in result.stdout and not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("flags", [("--recipe",), ("--recipe", "--task-request", "--protected")])
def test_recipe_declaration_is_complete_library_json_with_original_precedence(tmp_path, flags):
    result = invoke((*flags, "--output", str(tmp_path / "unused")), tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == recipe().to_json() + "\n"
    assert not tuple(tmp_path.iterdir())


def test_request_declaration_is_complete_library_json_without_observations(tmp_path):
    result = invoke(("--task-request", "--protected"), tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == planning_request().to_json() + "\n"
    assert not tuple(tmp_path.iterdir())
