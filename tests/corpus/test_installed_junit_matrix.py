"""Fresh CI-capability wheel: offline owned matrix, domain boundaries, native CI and optional actual Java sources."""

import json
import os
import shutil
import subprocess
import sys

from coleman.transports import junit_matrix as matrix
from tests.corpus.test_installed_controlled import ROOT, command
from tests.corpus.test_junit_matrix import owned_capture


def test_installed_matrix_validation_native_ci_and_optional_real_sources(tmp_path, monkeypatch):
    with monkeypatch.context() as external:
        plan, cache, writer, _ = owned_capture(tmp_path, external)
        owned = matrix.capture_junit_matrix(plan, cache, writer)
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(UV_LINK_MODE="copy", UV_NO_PROGRESS="1")
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
    script = """
import importlib.abc,json,sys
from pathlib import Path
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='coleman.cli' or fullname.startswith(('coleman.transports','mcp')):
            raise AssertionError('Controlled domain imported a transport: '+fullname)
boundary=Boundary();sys.meta_path.insert(0,boundary)
from coleman.corpus.controlled import validate_controlled_dataset
from coleman.corpus.controlled_contracts import ControlledDataset
assert Path(sys.modules['coleman.corpus.controlled'].__file__).resolve().is_relative_to(Path(sys.prefix))
sys.meta_path.remove(boundary)
from coleman.transports.junit_matrix import capture_junit_matrix,load_junit_matrix,validate_junit_matrix
from coleman.transports.junit_matrix_contracts import JunitMatrixRecipe,JunitMatrixReceipt
from coleman.transports.junit_source import digest,harness_bytes
from coleman.artifacts import ArtifactWriter
from coleman.corpus.materialization import SourceCache
assert Path(sys.modules['coleman.transports.junit_matrix'].__file__).resolve().is_relative_to(Path(sys.prefix))
owned=JunitMatrixReceipt.from_json(Path(sys.argv[3]).read_bytes())
cache=SourceCache(sys.argv[1]);output=Path(sys.argv[2])
validate_junit_matrix(owned,cache,output)
# This must complete without querying Docker: the original owned image does not exist.
replay=capture_junit_matrix(owned.plan,cache,ArtifactWriter(output))
assert replay.receipt==owned
assert replay.completion_path is not None
assert replay.completion_path.read_bytes()==Path(sys.argv[3]).read_bytes()
plan=load_junit_matrix()
assert sorted(len(cell.recipe.test_ids) for cell in plan.cells)==[95,96,166]
assert all(cell.recipe.harness_sha256==digest(harness_bytes()) for cell in plan.cells)
assert JunitMatrixRecipe.from_json(plan.to_json())==plan
future=json.loads(plan.to_json());future['schema_version']=99
try:JunitMatrixRecipe.from_json(json.dumps(future))
except ValueError:pass
else:raise AssertionError('Future matrix schema was accepted')
proof={'installed':True,'owned_offline_validation':True,'real_source_capture':False}
if sys.argv[4]!='-':
    real_cache=SourceCache(sys.argv[4]);writer=ArtifactWriter('real-java-matrix')
    first=capture_junit_matrix(plan,real_cache,writer)
    second=capture_junit_matrix(plan,real_cache,writer)
    assert first==second
    validate_junit_matrix(first.receipt,real_cache,writer.run_root)
    assert first.receipt.status=='collected'
    assert first.receipt.declared_projects==2
    assert first.receipt.curated_focal_fixes==3
    expected={'cli-13':(1,92,2,0,0),'cli-51':(0,93,2,0,1),'codec-65':(3,163,0,0,0)}
    for cell in first.receipt.cells:
        counts=cell.pairs
        values=(counts.expected,counts.both_pass,counts.both_fail,counts.reversed,counts.inconclusive)
        assert values==expected[cell.cell_id]
        assert cell.unobserved_pairs==0
    assert sum(cell.pairs.total for cell in first.receipt.cells)==357
    assert first.receipt.admission=='not_evaluated'
    assert first.receipt.generalization=='insufficient_evidence'
    proof.update(real_source_capture=True,real_receipt=str(first.receipt_path.resolve()),
                 actual_matrix=first.receipt.model_dump(mode='json'))
print(json.dumps(proof,sort_keys=True))
"""
    source_cache = env.get("COLEMAN_JUNIT_REAL_CACHE", "-")
    result = command(
        [str(python), "-c", script, str(cache.root), str(writer.run_root), str(owned.receipt_path), source_cache],
        outside,
        env,
    )
    proof = json.loads(result.stdout)
    assert proof["installed"]
    assert proof["owned_offline_validation"]
    assert proof["real_source_capture"] == (source_cache != "-")
    cli = subprocess.run(
        [
            str(python),
            "-m",
            "coleman.transports.junit_matrix",
            "--validate",
            "--cache",
            str(cache.root),
            "--output",
            str(writer.run_root),
        ],
        cwd=outside,
        env=env,
        input=owned.receipt_path.read_text(),
        capture_output=True,
        text=True,
        check=True,
        timeout=240,
    )
    assert json.loads(cli.stdout)["integrity"] == "verified"
    # Native CI uses its own owned exact-answer fixture, never Java elapsed counters as CI costs.
    shutil.copyfile(ROOT / "examples/controlled_faults/ci.py", outside / "ci.py")
    runs = [
        json.loads(
            command([str(python), "ci.py", "native-owned", str(tmp_path / "native-cache")], outside, env)
            .stdout.strip()
            .splitlines()[-1]
        )
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    assert runs[0]["native_rows"] == 1
    proof["owned_native_ci_checkpoint"] = runs[0]
    (tmp_path / "installed-junit-matrix-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
