"""Execute literal human-guide inputs and the CI example from an actual base wheel."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.installed_support import build_install_offline_wheel

ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def installed_examples(tmp_path_factory):
    owned = tmp_path_factory.mktemp("installed-registry-human-examples")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_LINK_MODE="copy", UV_NO_PROGRESS="1")
    virtual = owned / "installed"
    wheel = build_install_offline_wheel(uv, owned / "dist", virtual, environment)
    outside = owned / "outside"
    outside.mkdir()
    for origin, target in (
        ("examples/registry/ci.py", "ci.py"),
        ("examples/registry/collection.py", "collection.py"),
        ("examples/state/ci.py", "owned_ci.py"),
    ):
        shutil.copyfile(ROOT / origin, outside / target)
    blocks = re.findall(
        r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60", (ROOT / "docs/registry-lifecycle.md").read_text(), re.DOTALL
    )
    assert len(blocks) == 2, "Qualify every complete literal Python guide example"
    for index, block in enumerate(blocks):
        (outside / f"guide-{index}.py").write_text(block + "\n")
    assert not outside.is_relative_to(ROOT)
    assert wheel.is_file()
    return virtual / "bin/python", outside, environment


def invoke(installed, script, *, timeout=180, cold=False):
    python, outside, environment = installed
    if cold:
        environment = {
            key: value for key, value in environment.items() if not key.startswith(("COVERAGE_", "COV_CORE_"))
        }
    result = subprocess.run(
        [str(python), "-I", "-B", "-c", script],
        cwd=outside,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize("block,expected", [(0, "['ci-plan-a']\nTrue"), (1, "new retained 5")])
def test_every_literal_python_guide_block_uses_the_actual_installed_package(installed_examples, block, expected):
    script = (
        "import runpy,sys;from pathlib import Path;sys.path.insert(0,str(Path.cwd()));"
        "import coleman.registry.composition as c;"
        "assert Path(c.__file__).resolve().is_relative_to(Path(sys.prefix).resolve());"
        f"runpy.run_path('guide-{block}.py')"
    )
    assert invoke(installed_examples, script).strip() == expected


def test_real_six_round_example_retains_missing_zero_and_old_epoch_with_deterministic_identity(installed_examples):
    script = "import runpy;runpy.run_path('ci.py',run_name='__main__')"
    first, second = invoke(installed_examples, script), invoke(installed_examples, script)
    assert first == second
    receipt = json.loads(first)
    assert receipt["loss_unit"] == "1"
    rounds = receipt["rounds"]
    assert [len(row["active"]) for row in rounds] == [2, 1, 2, 2, 3, 0]
    assert [row["update_eligible"] for row in rounds] == [True, True, False, True, True, False]
    assert rounds[1]["losses"] == [0.5]
    assert rounds[2]["losses"] == [0.25, None]
    assert rounds[2]["statuses"] == ["observed", "unavailable"]
    assert rounds[4]["losses"][-1] == 0.0
    assert rounds[4]["statuses"][-1] == "observed"
    assert rounds[5]["action"] == "abstain"
    assert rounds[5]["coefficients"] == []
    assert receipt["delayed"] == {
        "original_round": 2,
        "original_epoch": "ci-plan-b:epoch-0",
        "replacement_epoch": "ci-plan-b:epoch-1",
        "delay_rounds": 3,
        "disposition": "audit_only",
    }
    assert receipt["state_fingerprint"] == "sha256:3e484e5b403eda84b16e616c04385767c5de600b6d3a52829f53d5396a70f26f"


def test_pure_example_does_not_initialize_storage_cli_mcp_or_native_execution(installed_examples):
    script = r"""
import importlib.abc,json,runpy,sys
blocked=('sqlite3','numpy','polars','pyarrow','duckdb','mcp','openai','anthropic',
 'coleman.state.local','coleman.state.components','coleman.artifacts','coleman.spec',
 'coleman.cli','coleman.application','coleman.runner','coleman.native','coleman.policy','coleman.telemetry')
assert not any(name==item or name.startswith(item+'.') for name in sys.modules for item in blocked)
class Boundary(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if any(fullname==item or fullname.startswith(item+'.') for item in blocked):
   raise AssertionError('Pure registry example imported optional boundary: '+fullname)
sys.meta_path.insert(0,Boundary())
example=runpy.run_path('ci.py')
receipt=example['run']()
assert receipt['rounds'][-1]['action']=='abstain'
assert not any(name==item or name.startswith(item+'.') for name in sys.modules for item in blocked)
print('pure registry example passed')
"""
    assert invoke(installed_examples, script, cold=True).strip() == "pure registry example passed"
