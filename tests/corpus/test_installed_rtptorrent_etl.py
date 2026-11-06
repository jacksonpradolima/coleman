"""Installed CI-capability Parquet/native workflow, with optional actual RTPTorrent cache."""

import json
import os
import shutil
import sys

from tests.corpus.test_installed_materialization import ROOT, command
from tests.installed_support import seed_locked_dependencies


def test_installed_rtptorrent_etl_resources_domain_boundaries_native_resume_and_real_cache(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update({"UV_LINK_MODE": "copy", "UV_NO_PROGRESS": "1"})
    dist = tmp_path / "dist"
    command(
        [uv, "build", "--offline", "--no-build-isolation", "--python", sys.executable, "--out-dir", str(dist)],
        ROOT,
        env,
    )
    virtual = tmp_path / "installed"
    command([uv, "venv", str(virtual), "--python", sys.executable], tmp_path, env)
    python = virtual / "bin/python"
    seed_locked_dependencies(uv, virtual, env, extras=("ci",))
    command(
        [
            uv,
            "pip",
            "install",
            "--offline",
            "--no-index",
            "--no-deps",
            "--python",
            str(python),
            str(next(dist.glob("*.whl"))) + "[ci]",
        ],
        tmp_path,
        env,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.copyfile(ROOT / "examples/rtptorrent-parquet/ci.py", outside / "ci.py")
    outcomes = []
    for _ in range(2):
        result = command([str(python), "ci.py", "owned", str(tmp_path / "cache")], outside, env)
        outcomes.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0]["observed_executions"] == 8
    assert outcomes[0]["replay_executions"] == 4
    assert outcomes[0]["modeled_rounds"] == 2
    script = outside / "verify.py"
    script.write_text(
        """
import importlib.abc, json, sys
from importlib.resources import files
from pathlib import Path
import polars as pl
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'coleman.cli' or fullname.startswith(('coleman.transports', 'mcp')):
            raise AssertionError('ETL domain imported transport: ' + fullname)
boundary = Boundary(); sys.meta_path.insert(0, boundary)
from coleman.corpus.rtptorrent_etl import export_rtptorrent_parquet, validate_rtptorrent_etl
from coleman.corpus.rtptorrent_etl_contracts import RTPTorrentETLRecipe, RTPTorrentETLReceipt
from coleman.corpus.materialization import SourceCache
from coleman.artifacts import ArtifactWriter
assert Path(sys.modules['coleman.corpus.rtptorrent_etl'].__file__).resolve().is_relative_to(Path(sys.prefix))
summary = json.loads(Path('owned-summary.json').read_bytes())
completion = Path(summary['completion'])
receipt = RTPTorrentETLReceipt.from_json(completion.read_bytes())
validate_rtptorrent_etl(receipt, SourceCache(CACHE_ROOT), completion.parent)
assert receipt.observed_executions == 8
assert receipt.replay_executions == 4
assert receipt.excluded_outside_jobs == receipt.excluded_not_run == 1
assert receipt.excluded_unknown_outcome == receipt.excluded_unavailable_duration == 1
native = Path(summary['native_artifacts'])
rows = pl.read_parquet(str(native / 'results/**/*.parquet'), hive_partitioning=True)
assert rows.height == 4
assert rows['step'].n_unique() == 2
assert all(json.loads(p.read_bytes())['step_committed'] == 2 for p in (native / 'checkpoints').rglob('progress_*.json'))
resource = files('coleman.corpus.resources').joinpath('rtptorrent-hikaricp-first-eight-v1.json')
recipe = RTPTorrentETLRecipe.from_json(resource.read_bytes())
assert recipe.source.source.source.size_bytes == 5016963482
real = REAL_CACHE
if real:
    cache = SourceCache(real)
    first = export_rtptorrent_parquet(recipe, cache, ArtifactWriter('real-etl'))
    second = export_rtptorrent_parquet(recipe, cache, ArtifactWriter('real-etl'))
    assert first == second
    real_receipt = RTPTorrentETLReceipt.from_json(first.read_bytes())
    validate_rtptorrent_etl(real_receipt, cache, first.parent)
    assert real_receipt.input_rows == real_receipt.observed_executions == 46995
    assert real_receipt.source_jobs == 1662
    assert real_receipt.replay_executions == 18
    assert real_receipt.native_rounds == 8
    assert real_receipt.excluded_outside_jobs == 46977
    assert real_receipt.admission == 'blocked_missing_temporal_evidence'
    sys.meta_path.remove(boundary)
    from coleman.application.corpus import run_rtptorrent_replay
    before = run_rtptorrent_replay(first, cache, Path('real-native'))
    after = run_rtptorrent_replay(first, cache, Path('real-native'))
    assert before.run_id == after.run_id
    real_root = Path(after.artifacts_dir)
    rows = pl.read_parquet(str(real_root / 'results/**/*.parquet'), hive_partitioning=True)
    assert rows.height == 16
    assert rows['step'].n_unique() == 8
    checkpoints = list((real_root / 'checkpoints').rglob('progress_*.json'))
    assert checkpoints
    assert all(json.loads(p.read_bytes())['step_committed'] == 8 for p in checkpoints)
print(json.dumps({'installed': True, 'real_source': bool(real), 'native_rows': rows.height,
                  'evidence': receipt.evidence, 'admission': receipt.admission}, sort_keys=True))
""".replace("CACHE_ROOT", repr(str(tmp_path / "cache"))).replace(
            "REAL_CACHE", repr(os.environ.get("COLEMAN_RTPTORRENT_ETL_REAL_CACHE"))
        )
    )
    (outside / "owned-summary.json").write_text(json.dumps(outcomes[0]))
    result = command([str(python), str(script)], outside, env)
    proof = json.loads(result.stdout.strip().splitlines()[-1])
    assert proof["installed"]
    assert proof["real_source"] == bool(os.environ.get("COLEMAN_RTPTORRENT_ETL_REAL_CACHE"))
    assert proof["native_rows"] == (16 if proof["real_source"] else 4)
