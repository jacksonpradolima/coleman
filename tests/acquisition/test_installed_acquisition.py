"""Noneditable base wheel, cold import boundary and complete literal CI guide."""

import hashlib
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
def installed_acquisition(tmp_path_factory):
    owned = tmp_path_factory.mktemp("installed-acquisition")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.setdefault("UV_CACHE_DIR", str(ROOT / ".cache/uv"))
    environment.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_LINK_MODE="copy", UV_NO_PROGRESS="1")
    virtual = owned / "installed"
    wheel = build_install_offline_wheel(uv, owned / "dist", virtual, environment)
    outside = owned / "outside"
    outside.mkdir()
    shutil.copyfile(ROOT / "examples/acquisition/ci_pool.py", outside / "ci_pool.py")
    shutil.copyfile(ROOT / "examples/acquisition/checkpoint.py", outside / "checkpoint.py")
    blocks = re.findall(r"```python\n(.*?)\n```", (ROOT / "docs/subset-acquisition.md").read_text(), re.DOTALL)
    assert len(blocks) == 1
    (outside / "guide.py").write_text(blocks[0] + "\n")
    (owned / "installation.json").write_text(
        json.dumps(
            {
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "wheel": str(wheel),
                "python": str(virtual / "bin/python"),
                "outside": str(outside),
            }
        )
    )
    assert not outside.is_relative_to(ROOT)
    return virtual / "bin/python", outside, environment


def invoke(installed, arguments):
    python, outside, environment = installed
    result = subprocess.run(
        [str(python), "-I", "-B", *arguments],
        cwd=outside,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_installed_base_cold_boundary_runs_actual_terminal_allocation_and_coverage_adapter(installed_acquisition):
    script = """
import importlib.abc,importlib.metadata,runpy,sys
from pathlib import Path
blocked=('sqlite3','numpy','scipy','polars','pyarrow','duckdb','mcp','openai','anthropic',
         'coleman.state','coleman.artifacts','coleman.spec','coleman.cli','coleman.runner',
         'coleman.application','coleman.native','coleman.agent','coleman.policy','coleman.telemetry',
         'coleman.transports')
class Boundary(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if any(fullname==name or fullname.startswith(name+'.') for name in blocked):
   raise AssertionError('Pure acquisition dependency crossed boundary: '+fullname)
sys.meta_path.insert(0,Boundary())
import coleman.acquisition.session as owner
import coleman.selection.subsets as subset
assert Path(owner.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert Path(subset.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not Path(owner.__file__).resolve().is_relative_to(Path.cwd())
installed={item.metadata['Name'].lower() for item in importlib.metadata.distributions()}
assert not {'numpy','scipy','polars','pyarrow','duckdb','mcp'} & installed
runpy.run_path('ci_pool.py',run_name='__main__')
assert not any(name==item or name.startswith(item+'.') for name in sys.modules for item in blocked)
"""
    reports = [line.split(" ", 2) for line in invoke(installed_acquisition, ["-c", script]).splitlines()]
    assert [json.loads(row[2])["known_charge"]["value"] for row in reports] == ["16", "8", "12"]
    assert all(json.loads(row[2])["status"] == "completed" for row in reports)


def test_literal_human_input_cycle_output_example_uses_installed_package(installed_acquisition):
    python, outside, environment = installed_acquisition
    assert invoke((python, outside, environment), ["guide.py"]).strip() == "recipe-A 2"


def test_installed_checkpoint_binding_uses_actual_store_and_receipt_retry(installed_acquisition):
    script = """
import runpy
from pathlib import Path
from coleman.acquisition.contracts import AllocationState
from coleman.acquisition.engine import inspect_allocation
from coleman.state.components import ComponentValue
from coleman.state.local import LocalStateStore
pool=runpy.run_path('ci_pool.py'); checkpoint=runpy.run_path('checkpoint.py')
plan=pool['make_plan'](); definition,bindings=checkpoint['layout'](plan)
compatibility=checkpoint['compatibility'](); state=AllocationState(schema_version=1,plan=plan)
store=LocalStateStore.create(Path('acknowledged'),definition,compatibility)
origin=checkpoint['command'](0,None,state.fingerprint,initialize=True)
head=store.initialize(origin,(ComponentValue(bindings[0].codec,state),),bindings,compatibility=compatibility)
purchase=inspect_allocation(state).next_purchase
receipt=pool['CIPool']().acquire(purchase,plan)
request=checkpoint['command'](1,head.current_head,receipt.fingerprint)
def transition(stage): stage.owner(bindings[0]).accept(receipt)
first=store.commit_transition(request,bindings,transition,compatibility=compatibility)
store=LocalStateStore.open(store.root)
retry=store.commit_transition(request,bindings,transition,compatibility=compatibility)
assert retry.status=='retained' and retry.current_head==first.current_head
current=store.load(bindings,compatibility=compatibility).component(bindings[0].codec)
assert len(current.receipts)==1 and inspect_allocation(current).known_charge.value==1
print('acknowledged-once')
"""
    assert invoke(installed_acquisition, ["-c", script]).strip() == "acknowledged-once"
