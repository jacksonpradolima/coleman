"""Fresh CI-capability wheel outside checkout: real 50K native replay, process interruption and old artifacts."""

import json
import os
import re
import shutil
import sys

from tests.corpus.test_installed_controlled import ROOT, command
from tests.installed_support import seed_locked_dependencies

SCRIPT = r"""
import hashlib, importlib.abc, importlib.util, json, os, subprocess, sys
from pathlib import Path
assert importlib.util.find_spec('mcp') is None
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'coleman.cli' or fullname.startswith(('coleman.transports', 'mcp')):
            raise AssertionError('Stress domain imported transport: ' + fullname)
boundary = Boundary(); sys.meta_path.insert(0, boundary)
from coleman.corpus.stress_artifacts import export_stress, validate_stress
from coleman.corpus.stress_contracts import StressCase, StressFactors, StressReceipt
from coleman.corpus.stress_design import covering_design
from coleman.corpus.stress_feedback import feedback_pages, visible_feedback, decision_time
from coleman.corpus.stress_replay import schedule_build
from coleman.corpus.stress_world import generate_build
assert Path(sys.modules['coleman.corpus.stress_world'].__file__).resolve().is_relative_to(Path(sys.prefix))
from coleman.artifacts import ArtifactWriter
from coleman.application.stress import run_stress_replay
from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt
import polars as pl

def make(**factors):
    return StressCase(schema_version=1, factors=StressFactors.model_validate(factors), builds=2)

def independent(result, case):
    rows = pl.read_parquet(str(Path(result.artifacts_dir)/'results/**/*.parquet'), hive_partitioning=True)
    for row in rows.iter_rows(named=True):
        order = json.loads(row['prioritization_order_top_k'])
        truth = {r.execution_name:r for r in generate_build(case, row['step']-1)}
        assert len(set(order)) == len(order) == len(truth)
        assert set(order) == truth.keys()
        total = sum(r.duration_seconds for r in truth.values())
        numerator, denominator = (1,16) if case.factors.budget == 'severe' else (1,1)
        selected, occupied, detected = [], 0, 0
        for name in order:
            r = truth[name]
            if (occupied+r.duration_seconds)*denominator > total*numerator:
                break
            occupied += r.duration_seconds; detected += r.verdict; selected.append(name)
        assert row['tests_ran'] == len(selected)
        assert row['tests_not_ran'] == len(truth)-len(selected)
        assert row['detected'] == detected
        assert row['detected']+row['missed'] == sum(r.verdict for r in truth.values())
        assert row['total_build_duration'] == total
    return rows

small = make(variants='changing', churn='collisions')
completion = export_stress(small, ArtifactWriter('small-export'))
if sys.argv[1] == 'interrupt':
    def stop_before_commit(self, payload):
        os._exit(73)
    LocalCheckpointStore.save = stop_before_commit
    run_stress_replay(completion, Path('small-native'))
    raise AssertionError('Expected real process interruption')
assert covering_design().to_json() == Path('stress-design-v1.json').read_text().strip()
legacy = LRTSETLReceipt.from_json(Path('legacy-completion.json').read_bytes())
assert legacy.schema_version == 1
assert legacy.recipe.feedback.value == 'budget-censored-native-RNFail-zero-nondetection'
assert export_stress(small, ArtifactWriter('small-export')) == completion
validate_stress(StressReceipt.from_json(completion.read_bytes()), completion.parent)
interrupted = subprocess.run([sys.executable, __file__, 'interrupt'], capture_output=True, timeout=120)
assert interrupted.returncode == 73, interrupted.stderr.decode()
assert not list(Path('small-native').rglob('manifest.json'))
assert list(Path('small-native').rglob('*.parquet'))
result = run_stress_replay(completion, Path('small-native'))
assert independent(result, small).height == 4
stable = {p:p.read_bytes() for p in Path(result.artifacts_dir).rglob('*')
          if p.is_file() and p.suffix in {'.parquet','.pkl'}}
assert run_stress_replay(completion, Path('small-native')).run_id == result.run_id
assert stable == {p:p.read_bytes() for p in stable}

# A real 50,000-test run, not a mocked scheduler, with a sparse dependent variant assignment.
large = make(size='large', failures='rare', duration='heavy_tail', variants='sparse', budget='full')
large_completion = export_stress(large, ArtifactWriter('large-export'))
validate_stress(StressReceipt.from_json(large_completion.read_bytes()), large_completion.parent)
large_result = run_stress_replay(large_completion, Path('large-native'), policies=('Random',))
large_rows = independent(large_result, large)
assert large_rows.height == 2 and large_rows['tests_ran'].to_list() == [50000,50000]
assert large_rows['detected'].to_list() == [1,1]
assert run_stress_replay(large_completion, Path('large-native'), policies=('Random',)).run_id == large_result.run_id

late = make(feedback='corrected', budget='full', variants='changing')
pages = tuple(feedback_pages(late, 0))
assert all(not s.observations for s in visible_feedback(pages, decision_time(late, 0)))
before = visible_feedback(pages, decision_time(late, 1))[0]
after = visible_feedback(pages, decision_time(late, 4))[0]
assert before.fingerprint != after.fingerprint
late_completion = export_stress(late, ArtifactWriter('late-export'))
try:
    run_stress_replay(late_completion, Path('ineligible-must-not-exist'))
except ValueError as error:
    assert 'cannot consume' in str(error)
else:
    raise AssertionError('Unsupported native corrected feedback was silently scored')
assert not Path('ineligible-must-not-exist').exists()
print(json.dumps(dict(installed=True, design_rows=32, small_rows=4, large_rows=2,
                     large_tests=50000, real_interruption_exit=73, run_id=large_result.run_id), sort_keys=True))
"""


def test_installed_offline_stress_transport_boundary_50k_and_real_checkpoint_process(tmp_path):
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
    (outside / "stress_check.py").write_text(SCRIPT)
    fixture = ROOT / "tests/corpus/fixtures"
    shutil.copyfile(fixture / "lrts-etl-owned-completion-v1.json", outside / "legacy-completion.json")
    shutil.copyfile(fixture / "stress-design-v1.json", outside / "stress-design-v1.json")
    result = command([str(python), str(outside / "stress_check.py"), "qualify"], outside, env)
    proof = json.loads(result.stdout.strip().splitlines()[-1])
    assert proof["installed"]
    assert proof["large_tests"] == 50000
    assert proof["real_interruption_exit"] == 73
    examples = re.findall(r"```python\n(.*?)\n```", (ROOT / "docs/ci-stress-laboratory.md").read_text(), re.S)
    assert len(examples) == 2
    docs = outside / "literal-doc-examples"
    docs.mkdir()
    for example in examples:
        command([str(python), "-c", example], docs, env)
