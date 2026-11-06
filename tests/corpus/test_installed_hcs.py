"""Fresh CI-capability wheel outside checkout: HCS Parquet, actual strategies/checkpoints and import boundary."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from tests.corpus.hcs_raw_reference import verify_libssh_raw_reference
from tests.corpus.hcs_support import hcs_recipe
from tests.corpus.test_hcs_archive_trace import owned_archive
from tests.corpus.test_hcs_trace import OWNED
from tests.corpus.test_hcs_trace_artifacts import owned_log_spec
from tests.corpus.test_installed_controlled import ROOT, command
from tests.installed_support import build_install_offline_wheel


def test_installed_hcs_sources_native_recovery_and_domain_import_boundary(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(UV_LINK_MODE="copy", UV_NO_PROGRESS="1")
    dist = tmp_path / "dist"
    virtual = tmp_path / "installed"
    build_install_offline_wheel(uv, dist, virtual, env, extras=("ci",))
    python = virtual / "bin/python"
    outside = tmp_path / "outside"
    outside.mkdir()
    recipe, csv_bytes, jobs_bytes = hcs_recipe()
    (outside / "recipe.json").write_text(recipe.to_json())
    (outside / "executions.csv").write_bytes(csv_bytes)
    (outside / "jobs.json").write_bytes(jobs_bytes)
    (outside / "owned-ctest.log").write_bytes(OWNED)
    (outside / "owned-log-spec.json").write_text(owned_log_spec().to_json())
    _, _, archive_spec, archive_payload = owned_archive(tmp_path / "archive-source")
    (outside / "archive-spec.json").write_text(archive_spec.to_json())
    (outside / "archive.zip").write_bytes(archive_payload)
    shutil.copyfile(ROOT / "tests/corpus/fixtures/hcs-ctest-owned-v1.json", outside / "ctest-golden.json")
    script = """
import importlib.abc,importlib.util,json,sys,os
from pathlib import Path
assert importlib.util.find_spec('mcp') is None
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='coleman.cli' or fullname.startswith(('coleman.transports','coleman.application','mcp')):
            raise AssertionError('HCS domain imported transport: '+fullname)
boundary=Boundary();sys.meta_path.insert(0,boundary)
from coleman.corpus.hcs_etl import export_hcs_parquet,validate_hcs_etl
from coleman.corpus.hcs_contracts import HCSReceipt,HCSRecipe
from coleman.corpus.hcs_trace import parse_hcs_ctest_trace,validate_hcs_ctest_trace
from coleman.corpus.hcs_trace_contracts import HCSTraceEvidence
from coleman.corpus.hcs_trace_correspondence import corroborate_hcs_ctest_trace
from coleman.corpus.hcs_trace_artifacts import HCSLogSpec,publish_hcs_trace_evidence,validate_hcs_trace_completion
from coleman.corpus.hcs_archive_trace import (
    HCSArchiveLogSpec,publish_hcs_archive_trace,validate_hcs_archive_trace,load_hcs_archive_log_spec,
)
assert load_hcs_archive_log_spec().job_id==63565028
trace=HCSTraceEvidence.from_json(Path('ctest-golden.json').read_bytes())
raw=Path('owned-ctest.log').read_bytes()
assert parse_hcs_ctest_trace(raw,trace.job)==trace
validate_hcs_ctest_trace(trace,raw)
assert trace.runs[0].executions[0].elapsed.value==0
assert trace.predecision_membership=='unverified'
assert Path(sys.modules['coleman.corpus.hcs_etl'].__file__).resolve().is_relative_to(Path(sys.prefix))
sys.meta_path.remove(boundary)
from coleman.application.hcs import run_hcs_replay
from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.artifacts import ArtifactWriter
from coleman.corpus.materialization import SourceCache,materialize_source
from coleman.transports.hcs_etl import load_hcs_recipe
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt
import polars as pl
old=LRTSETLReceipt.from_json(Path('legacy-completion.json').read_bytes())
assert old.schema_version==1
assert old.recipe.feedback.value=='budget-censored-native-RNFail-zero-nondetection'
recipe=HCSRecipe.from_json(Path('recipe.json').read_bytes());cache=SourceCache('cache')
for package,name in [(recipe.executions,'executions.csv'),(recipe.jobs,'jobs.json')]:
    materialize_source(package,cache,lambda name=name:Path(name).open('rb'))
from coleman.corpus.hcs_raw import read_hcs_slice
trace_proof=corroborate_hcs_ctest_trace(trace,read_hcs_slice(recipe,cache))
assert trace_proof.matched_observations==3
assert trace_proof.duplicate_rows==1
assert trace_proof.admission=='not_granted'
log_spec=HCSLogSpec.from_json(Path('owned-log-spec.json').read_bytes())
class InterruptedTrace(ArtifactWriter):
    def write_immutable(self,path,content):
        if path.name=='correspondence.json':
            os._exit(73)
        return super().write_immutable(path,content)
trace_writer=InterruptedTrace('corroboration') if sys.argv[-1]=='trace-interrupt' else ArtifactWriter('corroboration')
trace_path=publish_hcs_trace_evidence(recipe,log_spec,raw,cache,trace_writer)
trace_receipt=validate_hcs_trace_completion(trace_path,recipe,raw,cache)
assert trace_receipt.log_spec.terms.status=='pending'
trace_before={str(p):p.read_bytes() for p in trace_path.parent.iterdir()}
assert publish_hcs_trace_evidence(recipe,log_spec,raw,cache,trace_writer)==trace_path
assert trace_before=={str(p):p.read_bytes() for p in trace_path.parent.iterdir()}
archive_spec=HCSArchiveLogSpec.from_json(Path('archive-spec.json').read_bytes())
materialize_source(archive_spec.container,cache,lambda:Path('archive.zip').open('rb'))
class InterruptedArchive(ArtifactWriter):
    def write_immutable(self,path,content):
        if path.parent.parent.name=='hcs-archive-trace':
            os._exit(74)
        return super().write_immutable(path,content)
archive_writer=(InterruptedArchive('archive-corroboration') if sys.argv[-1]=='archive-interrupt'
    else ArtifactWriter('archive-corroboration'))
archive_path=publish_hcs_archive_trace(recipe,archive_spec,cache,archive_writer)
archive_receipt=validate_hcs_archive_trace(archive_path,recipe,cache)
assert archive_receipt.corroboration.files[0].rows==3
assert archive_receipt.corroboration.log_spec.terms==archive_spec.container.license
assert archive_receipt.admission=='not_granted'
p=export_hcs_parquet(recipe,cache,ArtifactWriter('etl'))
assert export_hcs_parquet(recipe,cache,ArtifactWriter('etl'))==p
r=HCSReceipt.from_json(p.read_bytes());validate_hcs_etl(r,cache,p.parent)
assert r.accounting.replay_rows==10 and r.accounting.selected_rows==17
assert r.evidence.setup_seconds is None and r.admission=='blocked_source_evidence'
proof=[]
for strategy in ['wts','vts']:
    original=LocalCheckpointStore.save
    def interrupted(self,payload):
        raise OSError('owned interruption after result flush before first checkpoint')
    LocalCheckpointStore.save=interrupted
    try:
        run_hcs_replay(p,cache,Path('native')/strategy,strategy=strategy)
    except OSError as error:
        assert str(error)=='owned interruption after result flush before first checkpoint'
    else:
        raise AssertionError('checkpoint failure was not injected')
    finally:
        LocalCheckpointStore.save=original
    assert not list((Path('native')/strategy).rglob('manifest.json'))
    a=run_hcs_replay(p,cache,Path('native')/strategy,strategy=strategy)
    b=run_hcs_replay(p,cache,Path('native')/strategy,strategy=strategy)
    assert a.run_id==b.run_id
    rows=pl.read_parquet(str(Path(b.artifacts_dir)/'results/**/*.parquet'),hive_partitioning=True)
    assert rows.height=={'wts':24,'vts':16}[strategy]
    assert set(rows['budget_value'])=={0.5,1.0}
    identities={(x['scenario'],x['variant'],x['policy'],x['budget_value'],x['step']) for x in rows.to_dicts()}
    assert len(identities)==rows.height
    markers=list(Path(b.artifacts_dir).rglob('progress_ex1.json'))
    assert markers and all('budget-v1' in str(m) for m in markers)
    proof.append(dict(strategy=strategy,rows=rows.height,run_id=b.run_id))
real=[]
if sys.argv[1]!='-':
    recipe=load_hcs_recipe();cache=SourceCache(sys.argv[1]);p=export_hcs_parquet(recipe,cache,ArtifactWriter('real-etl'))
    assert export_hcs_parquet(recipe,cache,ArtifactWriter('real-etl'))==p
    r=HCSReceipt.from_json(p.read_bytes());validate_hcs_etl(r,cache,p.parent)
    counts=r.accounting
    assert (counts.input_rows,counts.selected_rows,counts.replay_rows,counts.other_attempt_rows)==(278950,885,856,29)
    assert len(r.variants)==13
    cells=pl.read_parquet(p.parent/'cells.parquet');assert cells.height==208
    assert cells.filter(pl.col('cell_status')=='eligible').height==44
    for strategy in ['wts','vts']:
        a=run_hcs_replay(p,cache,Path('real-native')/strategy,strategy=strategy)
        b=run_hcs_replay(p,cache,Path('real-native')/strategy,strategy=strategy)
        assert a.run_id==b.run_id
        rows=pl.read_parquet(str(Path(b.artifacts_dir)/'results/**/*.parquet'),hive_partitioning=True)
        assert rows.height=={'wts':240,'vts':176}[strategy]
        assert set(rows['budget_value'])=={0.5,1.0}
        real.append(dict(strategy=strategy,run_id=b.run_id,rows=rows.height,artifacts_dir=b.artifacts_dir))
print(json.dumps(dict(installed=True,owned=proof,real=real,real_completion=str(p) if real else None),sort_keys=True))
"""
    shutil.copyfile(
        ROOT / "tests/corpus/fixtures/lrts-etl-owned-completion-v1.json", outside / "legacy-completion.json"
    )
    interrupted = subprocess.run(
        [str(python), "-c", script, "-", "trace-interrupt"],
        cwd=outside,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert interrupted.returncode == 73, interrupted.stderr
    assert list((outside / "corroboration").rglob("trace.json"))
    assert not list((outside / "corroboration").rglob("complete.json"))
    archive_interrupted = subprocess.run(
        [str(python), "-c", script, "-", "archive-interrupt"],
        cwd=outside,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert archive_interrupted.returncode == 74, archive_interrupted.stderr
    assert list((outside / "archive-corroboration" / "hcs-trace").rglob("complete.json"))
    assert not list((outside / "archive-corroboration" / "hcs-archive-trace").rglob("complete.json"))
    result = command([str(python), "-c", script, env.get("COLEMAN_HCS_REAL_CACHE", "-")], outside, env)
    proof = json.loads(result.stdout.strip().splitlines()[-1])
    assert proof["installed"]
    assert len(proof["real"]) == (2 if "COLEMAN_HCS_REAL_CACHE" in env else 0)
    if proof["real"]:
        reference = verify_libssh_raw_reference(
            env["COLEMAN_HCS_REAL_CACHE"], proof["real_completion"], proof["real"], outside
        )
        assert reference["decimal_boundary_differences"] == []
        (tmp_path / "installed-hcs-raw-reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    (tmp_path / "installed-hcs-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    args = [
        str(python),
        "-m",
        "coleman.transports.hcs_etl",
        "--recipe",
        "recipe.json",
        "--cache",
        "cli-cache",
        "--output",
        "cli-output",
        "--run",
    ]
    first = command(args + ["--executions", "executions.csv", "--jobs", "jobs.json"], outside, env)
    second = command(args, outside, env)
    assert json.loads(first.stdout.strip().splitlines()[-1]) == json.loads(second.stdout.strip().splitlines()[-1])
    assert Path(virtual).resolve() != ROOT.resolve()
