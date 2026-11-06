"""Base installed wheel outside checkout: real Parquet/cache, process death, offline CLI and import boundary."""

import json
import os
import shutil
import subprocess
import sys

from tests.corpus.test_installed_controlled import ROOT, command
from tests.corpus.westermo_support import owned_recipe


def test_installed_westermo_offline_recovery_and_import_boundary(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(UV_LINK_MODE="copy", UV_NO_PROGRESS="1", UV_OFFLINE="1")
    dist = tmp_path / "dist"
    command([uv, "build", "--wheel", "--out-dir", str(dist)], ROOT, env)
    virtual = tmp_path / "installed"
    command([uv, "venv", str(virtual), "--python", sys.executable], tmp_path, env)
    python = virtual / "bin/python"
    command(
        [uv, "pip", "install", "--offline", "--python", str(python), str(next(dist.glob("*.whl"))) + "[ci]"],
        tmp_path,
        env,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    recipe, payload = owned_recipe()
    (outside / "recipe.json").write_text(recipe.to_json())
    (outside / "source.tsv").write_bytes(payload)
    script = r"""
import importlib.abc, importlib.util, sys, os
from pathlib import Path
assert importlib.util.find_spec('mcp') is None
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='coleman.cli' or fullname.startswith(('coleman.transports','coleman.application','mcp')):
            raise AssertionError('Westermo domain imported transport/application: '+fullname)
sys.meta_path.insert(0,Boundary())
from coleman.corpus.westermo_etl import export_westermo_parquet, validate_westermo_etl, load_westermo_recipe
from coleman.corpus.westermo_contracts import WestermoRecipe, WestermoReceipt
from coleman.corpus.materialization import SourceCache,materialize_source
from coleman.artifacts import ArtifactWriter
assert Path(sys.modules['coleman.corpus.westermo_etl'].__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert len(load_westermo_recipe().sessions)==16
recipe=WestermoRecipe.from_json(Path('recipe.json').read_bytes())
cache=SourceCache('cache')
materialize_source(recipe.source,cache,lambda:Path('source.tsv').open('rb'))
class Interrupted(ArtifactWriter):
    def write_immutable_stream(self,path,chunks,*,maximum_bytes):
        if Path(path).name=='sessions.parquet':
            os._exit(73)
        return super().write_immutable_stream(path,chunks,maximum_bytes=maximum_bytes)
if sys.argv[1]=='interrupt':
    export_westermo_parquet(recipe,cache,Interrupted('output'))
    raise AssertionError('Process did not interrupt')
p=export_westermo_parquet(recipe,cache,ArtifactWriter('output'))
r=WestermoReceipt.from_json(p.read_bytes());validate_westermo_etl(r,cache,p.parent)
assert r.counts.selected_rows==5 and r.counts.invalid==1 and r.counts.unmappable==1
assert r.counts.incomplete_sessions==1
assert r.admission=='blocked_source_evidence'
before={str(q):q.read_bytes() for q in Path('output').rglob('*') if q.is_file()}
assert export_westermo_parquet(recipe,cache,ArtifactWriter('output'))==p
assert before=={str(q):q.read_bytes() for q in Path('output').rglob('*') if q.is_file()}
print(r.logical_rows_sha256)
"""
    interrupted = subprocess.run(
        [str(python), "-I", "-c", script, "interrupt"], cwd=outside, env=env, capture_output=True, text=True, timeout=60
    )
    assert interrupted.returncode == 73, interrupted.stderr
    assert list((outside / "output").rglob("observations.parquet"))
    assert not list((outside / "output").rglob("complete.json"))
    completed = command([str(python), "-c", script, "resume"], outside, env)
    assert completed.stdout.strip().startswith("sha256:")
    args = [
        str(python),
        "-m",
        "coleman.transports.westermo_etl",
        "--recipe-stdin",
        "--source",
        "source.tsv",
        "--cache",
        "cli-cache",
        "--output",
        "cli-output",
    ]

    def invoke():
        """Pass owned recipe bytes to the real installed CLI, independently of its source/archive filename."""
        return subprocess.run(
            args,
            input=recipe.to_json(),
            cwd=outside,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )

    first = json.loads(invoke().stdout)
    args.remove("--source")
    args.remove("source.tsv")
    (outside / "source.tsv").unlink()
    second = json.loads(invoke().stdout)
    assert first == second
    assert first["counts"]["selected_rows"] == 5
    assert first["logical_rows"] == completed.stdout.strip()
    assert first["admission"] == "blocked_source_evidence"
