"""Installed JUnit resources/import boundaries plus an explicitly enabled real offline source reproduction."""

import json
import os
import shutil

from tests.corpus.test_installed_controlled import command
from tests.installed_support import build_install_offline_wheel


def test_installed_producer_resources_domain_boundary_and_optional_real_pair(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update({"UV_LINK_MODE": "copy", "UV_NO_PROGRESS": "1"})
    dist = tmp_path / "dist"
    virtual = tmp_path / "installed"
    build_install_offline_wheel(uv, dist, virtual, env)
    python = virtual / "bin/python"
    outside = tmp_path / "outside"
    outside.mkdir()
    script = """
import importlib.abc, json, sys
from collections import Counter
from pathlib import Path
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'coleman.cli' or fullname.startswith(('coleman.transports','mcp')):
            raise AssertionError('Source domain imported transport: '+fullname)
boundary = Boundary()
sys.meta_path.insert(0, boundary)
from coleman.corpus.source_views import SourceViewSpec, source_view_bytes
assert source_view_bytes((('owned.txt',b'owned'),))
from coleman.corpus.archives import deterministic_zip_bytes
import io, zipfile
large = b'x'*(8*1024**2+1)
entries = (('diagnostic.json',large),)
try: source_view_bytes(entries)
except ValueError: pass
else: raise AssertionError('Installed source-view limits were widened')
envelope = deterministic_zip_bytes(entries,maximum_member_bytes=16*1024**2,maximum_total_bytes=40*1024**2)
with zipfile.ZipFile(io.BytesIO(envelope)) as archive:
    assert archive.read('diagnostic.json') == large
sys.meta_path.remove(boundary)
from coleman.transports.junit import load_junit_recipe
from coleman.transports.junit_source import digest, harness_bytes
recipe = load_junit_recipe()
assert recipe.harness_sha256 == digest(harness_bytes())
assert len(recipe.test_ids) == 95
assert Path(sys.modules['coleman.transports.junit'].__file__).resolve().is_relative_to(Path(sys.prefix))
proof = {'installed': True, 'tests': len(recipe.test_ids), 'real_source_capture': False}
if sys.argv[1] != '-':
    from uuid import uuid4
    from coleman.artifacts import ArtifactWriter
    from coleman.corpus import SourceCache
    from coleman.transports.junit_capture import capture_junit_source
    cache = SourceCache(sys.argv[1])
    writer = ArtifactWriter('real-junit-source')
    trial = 'installed-source-' + uuid4().hex
    first = capture_junit_source(recipe, cache, writer, trial)
    second = capture_junit_source(recipe, cache, writer, trial)
    assert first == second
    assert [item.status for item in first.captures] == ['collected','collected']
    pairs = dict(Counter(pair.classification for pair in first.dataset.pairs))
    assert pairs == {'both_pass':92,'both_fail':2,'expected':1}
    assert len(first.dataset.observations) == 190
    assert first.dataset.admission == 'not_evaluated'
    assert first.dataset.generalization == 'insufficient_evidence'
    proof.update(real_source_capture=True, pairs=pairs, dataset=str(first.dataset_path.resolve()))
print(json.dumps(proof,sort_keys=True))
"""
    source_cache = env.get("COLEMAN_JUNIT_REAL_CACHE", "-")
    completed = command([str(python), "-c", script, source_cache], outside, env)
    proof = json.loads(completed.stdout)
    assert proof["installed"]
    assert proof["tests"] == 95
    assert proof["real_source_capture"] == (source_cache != "-")
    (tmp_path / "installed-junit-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
