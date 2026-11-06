"""Fresh CI-capability wheel outside checkout: stage Parquet, import boundaries and actual native checkpoints."""

import json
import os
import shutil
import sys

from tests.corpus.test_installed_controlled import ROOT, command
from tests.corpus.test_lrts_etl import owned_recipe
from tests.corpus.test_lrts_stage_etl import stage_recipe
from tests.installed_support import seed_locked_dependencies


def test_installed_stage_sources_parquet_native_checkpoints_and_legacy_compatibility(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
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
    recipe, archive, metadata = stage_recipe()
    (outside / "recipe.json").write_text(recipe.to_json())
    (outside / "outer.zip").write_bytes(archive)
    (outside / "dataset.csv").write_bytes(metadata)
    old, old_archive, old_metadata = owned_recipe()
    (outside / "old-recipe.json").write_text(old.to_json())
    (outside / "old-outer.zip").write_bytes(old_archive)
    (outside / "old-dataset.csv").write_bytes(old_metadata)
    (outside / "old-completion.json").write_bytes(
        (ROOT / "tests/corpus/fixtures/lrts-etl-owned-completion-v1.json").read_bytes()
    )
    script = """
import importlib.abc,json,sys
from pathlib import Path
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='coleman.cli' or fullname.startswith(('coleman.transports','mcp')):
            raise AssertionError('Domain imported transport: '+fullname)
boundary=Boundary();sys.meta_path.insert(0,boundary)
from coleman.corpus.lrts_stage_etl import export_lrts_stage_parquet,validate_lrts_stage_etl
from coleman.corpus.lrts_stage_contracts import LRTSStageRecipe,LRTSStageReceipt
assert Path(sys.modules['coleman.corpus.lrts_stage_etl'].__file__).resolve().is_relative_to(Path(sys.prefix))
sys.meta_path.remove(boundary)
from coleman.application.corpus import run_lrts_stage_replay
from coleman.artifacts import ArtifactWriter
from coleman.corpus.archive_selection import materialize_selected_packages
from coleman.corpus.materialization import SourceCache,materialize_source
from coleman.transports.lrts_stage_etl import load_lrts_stage_recipe
import polars as pl
from coleman.corpus.lrts_etl import export_lrts_parquet,validate_lrts_etl
from coleman.corpus.lrts_etl_contracts import LRTSETLRecipe,LRTSETLReceipt
old=LRTSETLRecipe.from_json(Path('old-recipe.json').read_bytes());old_cache=SourceCache('old-cache')
materialize_source(old.metadata,old_cache,lambda:Path('old-dataset.csv').open('rb'))
materialize_selected_packages(old.reports,old_cache,lambda:Path('old-outer.zip').open('rb'))
old_path=export_lrts_parquet(old,old_cache,ArtifactWriter('old-etl'))
legacy_bytes=Path('old-completion.json').read_bytes()
legacy_receipt=LRTSETLReceipt.from_json(legacy_bytes)
current_receipt=LRTSETLReceipt.from_json(old_path.read_bytes())
# Implementation pins include materialization source bytes, including docstrings.
# A new producer identity must not rewrite the old wire record or its table bytes.
assert (legacy_receipt.to_json()+'\\n').encode()==legacy_bytes
assert current_receipt.producer_sha256!=legacy_receipt.producer_sha256
assert legacy_bytes.count(legacy_receipt.producer_sha256.encode())==1
assert old_path.read_bytes()==legacy_bytes.replace(
    legacy_receipt.producer_sha256.encode(),current_receipt.producer_sha256.encode())
validate_lrts_etl(legacy_receipt,old_cache,old_path.parent)
recipe=LRTSStageRecipe.from_json(Path('recipe.json').read_bytes())
cache=SourceCache('cache')
materialize_source(recipe.source.metadata,cache,lambda:Path('dataset.csv').open('rb'))
materialize_selected_packages(recipe.source.reports,cache,lambda:Path('outer.zip').open('rb'))
path=export_lrts_stage_parquet(recipe,cache,ArtifactWriter('etl'))
assert export_lrts_stage_parquet(recipe,cache,ArtifactWriter('etl'))==path
receipt=LRTSStageReceipt.from_json(path.read_bytes());validate_lrts_stage_etl(receipt,cache,path.parent)
assert receipt.observed_cases==12
assert receipt.replay_cases==4
before=run_lrts_stage_replay(path,cache,Path('native'));after=run_lrts_stage_replay(path,cache,Path('native'))
assert before.run_id==after.run_id
rows=pl.read_parquet(str(Path(after.artifacts_dir)/'results/**/*.parquet'),hive_partitioning=True)
assert rows.height==4
assert rows['step'].n_unique()==2
proof=[]
for preset in ('activemq','log4j-ubuntu','log4j-windows'):
    r=load_lrts_stage_recipe(preset=preset)
    assert r.source.history=='observed'
    if sys.argv[1]=='-':continue
    real_cache=SourceCache(sys.argv[1])
    p=export_lrts_stage_parquet(r,real_cache,ArtifactWriter('real-'+preset))
    assert export_lrts_stage_parquet(r,real_cache,ArtifactWriter('real-'+preset))==p
    receipt=LRTSStageReceipt.from_json(p.read_bytes());validate_lrts_stage_etl(receipt,real_cache,p.parent)
    expected={'activemq':(18901,18668,0,233,4),'log4j-ubuntu':(30066,14957,15014,95,8),
              'log4j-windows':(30066,14906,15052,108,8)}
    assert (receipt.observed_cases,receipt.replay_cases,receipt.excluded_outside_stage,
            receipt.excluded_not_run,receipt.source_stages)==expected[preset]
    assert receipt.excluded_unknown_outcome==0
    assert receipt.excluded_unavailable_duration==0
    assert receipt.metadata_accounting.input_rows==57437
    assert receipt.metadata_accounting.selected_rows==receipt.source_stages
    a=run_lrts_stage_replay(p,real_cache,Path('native-'+preset));b=run_lrts_stage_replay(p,real_cache,Path('native-'+preset))
    assert a.run_id==b.run_id
    rows=pl.read_parquet(str(Path(b.artifacts_dir)/'results/**/*.parquet'),hive_partitioning=True)
    assert rows.height==8
    assert rows['step'].n_unique()==4
    markers=list((Path(b.artifacts_dir)/'checkpoints').rglob('progress_*.json'))
    assert markers
    assert all(json.loads(m.read_bytes())['step_committed']==4 for m in markers)
    proof.append(dict(preset=preset,observed=receipt.observed_cases,replay=receipt.replay_cases,
                      excluded_outside_stage=receipt.excluded_outside_stage,excluded_not_run=receipt.excluded_not_run,
                      logical_rows=receipt.logical_rows_sha256,receipt=receipt.fingerprint,producer=receipt.producer_sha256,
                      source_stages=receipt.source_stages,source_datasets=receipt.source_datasets,native_rows=rows.height,
                      native_run_id=b.run_id,admission=receipt.admission,temporal=receipt.temporal.model_dump(mode='json')))
print(json.dumps(dict(installed=True,real_sources=proof),sort_keys=True))
"""
    result = command([str(python), "-c", script, env.get("COLEMAN_LRTS_STAGE_REAL_CACHE", "-")], outside, env)
    proof = json.loads(result.stdout.strip().splitlines()[-1])
    assert proof["installed"]
    assert len(proof["real_sources"]) == (3 if "COLEMAN_LRTS_STAGE_REAL_CACHE" in env else 0)
    (tmp_path / "installed-lrts-stage-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    # Explicit source transport is also exercised from a fresh process, followed by an offline retry.
    args = [
        str(python),
        "-m",
        "coleman.transports.lrts_stage_etl",
        "--recipe",
        "recipe.json",
        "--cache",
        "cli-cache",
        "--output",
        "cli-output",
        "--run",
    ]
    first = command(args + ["--archive", "outer.zip", "--metadata", "dataset.csv"], outside, env)
    second = command(args, outside, env)
    a = json.loads(first.stdout.strip().splitlines()[-1])
    b = json.loads(second.stdout.strip().splitlines()[-1])
    assert a == b
    assert a["observed_cases"] == 12
    assert a["replay_cases"] == 4
