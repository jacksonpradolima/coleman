"""Installed CI-capability wheel source conversion and actual isolated native replay/checkpoint continuation."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from coleman.corpus.rtptorrent_contracts import RTPTorrentDataset
from tests.installed_support import build_install_offline_wheel

ROOT = Path(__file__).parents[2]


def command(args, cwd, env):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=True, timeout=240)


def test_installed_adapter_boundaries_owned_source_and_native_checkpoint(tmp_path):
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
            raise AssertionError('Domain imported transport: ' + fullname)
sys.meta_path.insert(0, Boundary())
import coleman.corpus.rtptorrent as adapter
import coleman.corpus.rtptorrent_contracts
print(adapter.__file__)
assert not any(name == 'mcp' or name.startswith('mcp.') for name in sys.modules)
"""
    origin = command([str(python), "-c", code], outside, env)
    assert Path(origin.stdout.strip()).resolve().is_relative_to(virtual.resolve())
    script = outside / "ci.py"
    shutil.copyfile(ROOT / "examples/rtptorrent/ci.py", script)
    observations = []
    for _ in range(2):
        result = command([str(python), str(script), "owned-ci", str(tmp_path / "cache")], outside, env)
        outcome = json.loads(result.stdout.strip().splitlines()[-1])
        observations.append(Path(outcome["dataset"]).read_bytes())
        assert outcome["native_rows"] == 1
        assert outcome["source_jobs"] == 3
        assert outcome["input_rows"] == 6
        assert outcome["repeated_class_executions"] == 1
        assert outcome["admission"] == "blocked_missing_temporal_evidence"
        assert outcome["generalization"] == "insufficient_evidence"
        dataset = RTPTorrentDataset.from_json(observations[-1])
        assert dataset.spec.history == "constructed"
        zero_duration = dataset.job(102)[0].duration
        assert zero_duration is not None
        assert zero_duration.value == 0
        assert dataset.job(103)[0].outcome == "unknown"
        root = Path(outcome["native_root"])
        checkpoints = list((root / "checkpoints").rglob("progress_*.json"))
        assert checkpoints
        assert all(json.loads(p.read_bytes())["step_committed"] == 1 for p in checkpoints)
        manifest = json.loads((root / "manifest.json").read_bytes())
        assert len([p for p in manifest["artifacts"] if p["type"] == "corpus_rtptorrent_dataset"]) == 1
    assert observations[0] == observations[1]
    documentation = (ROOT / "docs/rtptorrent.md").read_text()
    snippets = re.findall(r"```python\n(.*?)```", documentation, flags=re.S)
    assert len(snippets) == 1
    command([str(python), "-c", snippets[0]], outside, env)
