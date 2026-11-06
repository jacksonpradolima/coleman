"""Installed actual Parquet ETL/native replay outside checkout; optional complete real Karaf source slice."""

import json
import os
import shutil
import sys

from tests.corpus.test_installed_controlled import ROOT, command
from tests.corpus.test_lrts_etl import owned_recipe
from tests.installed_support import seed_locked_dependencies


def test_installed_lrts_etl_resources_boundaries_parquet_and_native_recovery(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(UV_LINK_MODE="copy", UV_NO_PROGRESS="1")
    dist = tmp_path / "dist"
    command(
        [
            uv,
            "build",
            "--offline",
            "--no-build-isolation",
            "--python",
            sys.executable,
            "--wheel",
            "--out-dir",
            str(dist),
        ],
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
    recipe, archive, metadata = owned_recipe()
    (outside / "recipe.json").write_text(recipe.to_json())
    (outside / "outer.zip").write_bytes(archive)
    (outside / "dataset.csv").write_bytes(metadata)
    script = """
import importlib.abc,json,sys
from pathlib import Path
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='coleman.cli' or fullname.startswith(('coleman.transports','mcp')):
            raise AssertionError('ETL domain imported transport: '+fullname)
boundary=Boundary();sys.meta_path.insert(0,boundary)
from coleman.corpus.lrts_etl import export_lrts_parquet,validate_lrts_etl
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt,LRTSETLRecipe
from coleman.corpus.archive_selection import materialize_selected_packages
from coleman.corpus.materialization import SourceCache,materialize_source
assert Path(sys.modules['coleman.corpus.lrts_etl'].__file__).resolve().is_relative_to(Path(sys.prefix))
sys.meta_path.remove(boundary)
from coleman.application.corpus import run_lrts_replay
from coleman.artifacts import ArtifactWriter
from coleman.transports.lrts_etl import load_lrts_etl_recipe
import polars as pl
recipe=LRTSETLRecipe.from_json(Path('recipe.json').read_bytes())
cache=SourceCache('source-cache')
materialize_source(recipe.metadata,cache,lambda:Path('dataset.csv').open('rb'))
materialize_selected_packages(recipe.reports,cache,lambda:Path('outer.zip').open('rb'))
first=export_lrts_parquet(recipe,cache,ArtifactWriter('etl'))
assert export_lrts_parquet(recipe,cache,ArtifactWriter('etl'))==first
receipt=LRTSETLReceipt.from_json(first.read_bytes());validate_lrts_etl(receipt,cache,first.parent)
assert receipt.observed_cases==10 and receipt.replay_cases==4
before=run_lrts_replay(first,cache,Path('native'))
after=run_lrts_replay(first,cache,Path('native'))
assert before.run_id==after.run_id
rows=pl.read_parquet(str(Path(after.artifacts_dir)/'results/**/*.parquet'),hive_partitioning=True)
assert rows.height==4
real=False
if sys.argv[1]!='-':
    preset=load_lrts_etl_recipe();cache=SourceCache(sys.argv[1])
    first=export_lrts_parquet(preset,cache,ArtifactWriter('real-etl'))
    second=export_lrts_parquet(preset,cache,ArtifactWriter('real-etl'))
    assert first==second
    receipt=LRTSETLReceipt.from_json(first.read_bytes());validate_lrts_etl(receipt,cache,first.parent)
    assert (receipt.observed_cases,receipt.replay_cases,receipt.excluded_not_run)==(6876,6567,309)
    assert receipt.metadata_accounting.input_rows==57437
    assert receipt.metadata_accounting.selected_rows==8
    a=run_lrts_replay(first,cache,Path('real-native'));b=run_lrts_replay(first,cache,Path('real-native'))
    assert a.run_id==b.run_id
    rows=pl.read_parquet(str(Path(b.artifacts_dir)/'results/**/*.parquet'),hive_partitioning=True)
    assert rows.height==16 and rows['step'].n_unique()==8
    real=True
print(json.dumps({'installed':True,'real_source':real,'evidence':receipt.evidence,'observed_cases':receipt.observed_cases,'replay_cases':receipt.replay_cases,'native_rows':rows.height},sort_keys=True))
"""
    result = command([str(python), "-c", script, env.get("COLEMAN_LRTS_ETL_REAL_CACHE", "-")], outside, env)
    proof = json.loads(result.stdout.strip().splitlines()[-1])
    assert proof["installed"]
    assert proof["real_source"] == ("COLEMAN_LRTS_ETL_REAL_CACHE" in env)
    (tmp_path / "installed-lrts-etl-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
