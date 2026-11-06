"""Installed offline controlled source reader, native CI replay and checkpoint continuation."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from coleman.corpus.controlled_contracts import ControlledDataset
from tests.installed_support import build_install_offline_wheel

ROOT = Path(__file__).parents[2]


def command(args, cwd, env):
    """Run a bounded installed-package command and preserve independent process failures."""
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=True, timeout=240)


def test_installed_offline_sources_import_boundary_native_checkpoint_and_sdk(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
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
        if fullname == 'coleman.cli' or fullname.startswith(('coleman.transports','mcp')):
            raise AssertionError('Controlled domain imported transport: '+fullname)
sys.meta_path.insert(0, Boundary())
import coleman.corpus.controlled as adapter
import coleman.corpus.controlled_contracts
print(adapter.__file__)
"""
    origin = command([str(python), "-c", code], outside, env)
    assert Path(origin.stdout.strip()).resolve().is_relative_to(virtual.resolve())
    script = outside / "ci.py"
    shutil.copyfile(ROOT / "examples/controlled_faults/ci.py", script)
    observations = []
    for _ in range(2):
        outcome = json.loads(
            command([str(python), str(script), "owned-ci", str(tmp_path / "cache")], outside, env)
            .stdout.strip()
            .splitlines()[-1]
        )
        observations.append(Path(outcome["dataset"]).read_bytes())
        assert outcome["native_rows"] == 1
        assert outcome["classifications"] == ["expected", "inconclusive"]
        dataset = ControlledDataset.from_json(observations[-1])
        assert dataset.spec.context.fault_origin == "synthetic"
        assert dataset.admission == "not_evaluated"
        assert dataset.generalization == "insufficient_evidence"
        assert len(dataset.observations) == 3
        assert all(item.elapsed.value is None for item in dataset.observations)
        root = Path(outcome["native_root"])
        checkpoints = list((root / "checkpoints").rglob("progress_*.json"))
        assert checkpoints
        assert all(json.loads(p.read_bytes())["step_committed"] == 1 for p in checkpoints)
        manifest = json.loads((root / "manifest.json").read_bytes())
        assert sum(a["type"] == "corpus_controlled_dataset" for a in manifest["artifacts"]) == 1
    assert observations[0] == observations[1]
    snippets = re.findall(r"```python\n(.*?)```", (ROOT / "docs/controlled-faults.md").read_text(), flags=re.S)
    assert len(snippets) == 1
    snippet = snippets[0].replace("/tmp/coleman-controlled-source-cache", str(tmp_path / "cache"))
    command([str(python), "-c", snippet], outside, env)
