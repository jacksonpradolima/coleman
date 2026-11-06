"""Fresh installed base wheel: offline owned replay, import boundary, native CI and opt-in real C source."""

import json

from coleman.artifacts import ArtifactWriter
from coleman.transports.c_capture import capture_c_source
from tests.corpus.installed_source_support import (
    installed_source_environment,
    verify_native_checkpoint,
    verify_offline_source_cli,
)
from tests.corpus.test_c_capture import prepared_inputs, substitute_runtime
from tests.corpus.test_installed_controlled import command


def test_installed_c_source_replay_native_checkpoints_and_optional_real_c_source(tmp_path, monkeypatch):
    spec, cache, environment = prepared_inputs(tmp_path)
    writer = ArtifactWriter(tmp_path / "owned-output")
    with monkeypatch.context() as external:
        substitute_runtime(external, environment)
        owned = capture_c_source(spec, cache, writer, "owned-installed")
    python, outside, env = installed_source_environment(tmp_path, extras=("ci",))
    script = """
import importlib.abc,json,os,signal,subprocess,sys,time
from collections import Counter
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
from coleman.artifacts import ArtifactWriter
from coleman.corpus import SourceCache
from coleman.transports.c import PRESETS,load_c_recipe
from coleman.transports.c_capture import capture_c_source,validate_c_capture
from coleman.transports.c_contracts import CCapturePlan,CCaptureReceipt
assert Path(sys.modules['coleman.transports.c_capture'].__file__).resolve().is_relative_to(Path(sys.prefix))
cache=SourceCache(sys.argv[1]);output=Path(sys.argv[2])
plan=CCapturePlan.from_json(Path(sys.argv[3]).read_bytes())
first=capture_c_source(plan.recipe,cache,ArtifactWriter(output),plan.trial_id)
assert first.receipt.status=='collected'
validate_c_capture(first.receipt,plan,first.dataset,cache)
assert first.captures[0].producer_elapsed.value==0
assert first.dataset.pairs[0].classification=='expected'
future=json.loads(plan.recipe.to_json());future['schema_version']=99
try:type(plan.recipe).from_json(json.dumps(future))
except ValueError:pass
else:raise AssertionError('Future C schema was accepted')
proof={'installed':True,'owned_offline_replay':True,'real_source_capture':False}
if sys.argv[4]!='-':
    real_cache=SourceCache(sys.argv[4]);real_writer=ArtifactWriter('real-c')
    cells=[]
    for preset in PRESETS:
        recipe=load_c_recipe(preset)
        result=capture_c_source(recipe,real_cache,real_writer,'installed-c-qualification')
        assert result.receipt.status=='collected'
        assert len(result.dataset.observations)==12
        assert all(case.outcome=='passed' for case in result.captures[1].cases)
        assert Counter(pair.classification for pair in result.dataset.pairs)=={'both_pass':5,'expected':1}
        assert result.dataset.spec.context.fault_origin=='synthetic'
        classification='six-correlated-boundary-inputs-one-synthetic-template'
        assert capture_c_source(recipe,real_cache,real_writer,'installed-c-qualification')==result
        frozen=CCapturePlan.from_json((result.state_directory/'plan.json').read_bytes())
        validate_c_capture(result.receipt,frozen,result.dataset,real_cache)
        cells.append({'preset':preset,'receipt':str((result.state_directory/'receipt.json').resolve()),
                      'context':result.dataset.spec.context.fingerprint,'classification':classification})
    # Kill a real SDK producer after durable ownership, without replacing its main component.
    child_code='''import sys
from coleman.artifacts import ArtifactWriter
from coleman.corpus import SourceCache
from coleman.transports.c import load_c_recipe
from coleman.transports.c_capture import capture_c_source
capture_c_source(load_c_recipe(),SourceCache(sys.argv[1]),ArtifactWriter(sys.argv[2]),'installed-interrupted')
'''
    interrupted=Path('real-interrupted')
    with Path('interrupted-process.log').open('wb') as log:
        child=subprocess.Popen([sys.executable,'-c',child_code,str(real_cache.root),str(interrupted)],
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        marker=None
        deadline=time.monotonic()+90
        try:
            while child.poll() is None and time.monotonic()<deadline:
                markers=list(interrupted.glob('c-captures/*/buggy.started.json'))
                if markers:
                    marker=markers[0]
                    os.kill(child.pid,signal.SIGSTOP)
                    break
                time.sleep(0.005)
        finally:
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
            child.wait(timeout=15)
    assert marker is not None,'Actual producer did not reach its durable ownership marker'
    assert not (marker.parent/'buggy.json').exists(),'Role committed before interruption; do not fabricate recovery'
    assert not (marker.parent/'complete.json').exists()
    recovered=capture_c_source(load_c_recipe(),real_cache,ArtifactWriter(interrupted),'installed-interrupted')
    assert recovered.receipt.status=='incomplete'
    assert [r.status for r in recovered.captures]==['interrupted','collected']
    assert recovered.captures[0].producer_elapsed.value is None
    assert recovered.dataset.pairs[0].classification=='inconclusive'
    assert not (recovered.state_directory/'complete.json').exists()
    replay=capture_c_source(load_c_recipe(),real_cache,ArtifactWriter(interrupted),'installed-interrupted')
    assert replay==recovered
    proof.update(real_source_capture=True,cells=cells,actual_interruption_recovery=True,
                 incomplete_receipt=str((recovered.state_directory/'receipt.json').resolve()))
print(json.dumps(proof,sort_keys=True))
"""
    source_cache = env.get("COLEMAN_C_REAL_CACHE", "-")
    proof = json.loads(
        command(
            [
                str(python),
                "-c",
                script,
                str(cache.root),
                str(writer.run_root),
                str(owned.state_directory / "plan.json"),
                source_cache,
            ],
            outside,
            env,
        ).stdout
    )
    assert proof["installed"]
    assert proof["owned_offline_replay"]
    assert proof["real_source_capture"] == (source_cache != "-")
    verify_offline_source_cli(python, outside, env, "c", cache, writer, spec)
    proof["native_ci_checkpoint"] = verify_native_checkpoint(python, outside, env, tmp_path)
    (tmp_path / "installed-c-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
