"""Installed LRTS source conversion, real native replay and checkpoint recovery outside the checkout."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from coleman.corpus.lrts_contracts import LRTSDataset
from tests.installed_support import build_install_offline_wheel

ROOT = Path(__file__).parents[2]


def command(args, cwd, env):
    """Run a bounded independent installed-package command and preserve errors for the test report."""
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=True, timeout=240)


def test_installed_lrts_boundaries_owned_sources_native_checkpoint_and_documentation(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update({"UV_LINK_MODE": "copy", "UV_NO_PROGRESS": "1"})
    dist = tmp_path / "dist"
    virtual = tmp_path / "installed"
    build_install_offline_wheel(uv, dist, virtual, env, extras=("ci",))
    python = virtual / "bin/python"
    outside = tmp_path / "outside"
    outside.mkdir()
    code = """
import importlib.abc, importlib.util, sys
assert importlib.util.find_spec('mcp') is None
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'coleman.cli' or fullname.startswith(('coleman.transports', 'mcp')):
            raise AssertionError('LRTS domain imported transport: ' + fullname)
sys.meta_path.insert(0, Boundary())
import coleman.corpus.lrts as adapter
import coleman.corpus.lrts_contracts, coleman.corpus.lrts_raw
print(adapter.__file__)
assert not any(name == 'mcp' or name.startswith('mcp.') for name in sys.modules)
"""
    origin = command([str(python), "-c", code], outside, env)
    assert Path(origin.stdout.strip()).resolve().is_relative_to(virtual.resolve())
    script = outside / "ci.py"
    shutil.copyfile(ROOT / "examples/lrts/ci.py", script)
    observations = []
    for _ in range(2):
        result = command([str(python), str(script), "owned-ci", str(tmp_path / "cache")], outside, env)
        outcome = json.loads(result.stdout.strip().splitlines()[-1])
        observations.append(Path(outcome["dataset"]).read_bytes())
        assert outcome["native_rows"] == 1
        assert outcome["reported_cases"] == 3
        assert outcome["metadata_stages"] == 1
        assert outcome["admission"] == "blocked_source_evidence"
        assert outcome["generalization"] == "insufficient_evidence"
        assert outcome["feedback_scope"] == "reported_cases_only"
        dataset = LRTSDataset.from_json(observations[-1])
        assert dataset.spec.history == "constructed"
        assert len(dataset.cases) == 3
        root = Path(outcome["native_root"])
        checkpoints = list((root / "checkpoints").rglob("progress_*.json"))
        assert checkpoints
        assert all(json.loads(path.read_bytes())["step_committed"] == 1 for path in checkpoints)
        manifest = json.loads((root / "manifest.json").read_bytes())
        assert len([item for item in manifest["artifacts"] if item["type"] == "corpus_lrts_dataset"]) == 1
    assert observations[0] == observations[1]
    documentation = (ROOT / "docs/lrts.md").read_text()
    snippets = re.findall(r"```python\n(.*?)```", documentation, flags=re.S)
    assert len(snippets) == 1
    command([str(python), "-c", snippets[0]], outside, env)
