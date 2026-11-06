"""Fresh CI-capability wheel ZIP64 selection and actual native checkpoint recovery outside the checkout."""

import json
import os
import shutil
import sys

from tests.corpus.test_installed_materialization import ROOT, command


def test_installed_zip64_selected_ci_fixture_native_replay_and_boundaries(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update({"UV_LINK_MODE": "copy", "UV_NO_PROGRESS": "1"})
    dist = tmp_path / "dist"
    command([uv, "build", "--out-dir", str(dist)], ROOT, env)
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
    boundary = command(
        [
            str(python),
            "-c",
            """
import importlib.abc, sys
class NoTransports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('coleman.cli', 'coleman.transports', 'mcp', 'httpx', 'requests')):
            raise AssertionError('Domain imported transport: ' + fullname)
sys.meta_path.insert(0, NoTransports())
import coleman.corpus.zip64_contracts, coleman.corpus.zip64_selection
from importlib.resources import files
recipe = coleman.corpus.zip64_contracts.Zip64SelectionSpec.from_json(
    files('coleman.corpus.resources').joinpath('rtptorrent-hikaricp-zip64-selection-v1.json').read_bytes())
assert recipe.source.source.size_bytes == 5016963482
print(coleman.corpus.zip64_selection.__file__)
""",
        ],
        outside,
        env,
    )
    assert str(virtual) in boundary.stdout
    shutil.copyfile(ROOT / "examples/zip64-selection/ci.py", outside / "zip64_ci.py")
    shutil.copyfile(ROOT / "examples/corpus-materialization/ci.py", outside / "native_ci.py")
    cache = tmp_path / "cache"
    outcomes = []
    for _ in range(2):
        result = command([str(python), "zip64_ci.py", "selected", str(cache)], outside, env)
        outcomes.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0]["integrity"] == "verified-selected-membership"
    script = outside / "native_selected.py"
    script.write_text(
        """
import json
from importlib.resources import files
from pathlib import Path
from coleman.api import run
from coleman.corpus import MaterializationReceipt, SourceCache, load_registry, validate_fixture_source
from coleman.corpus.zip64_contracts import Zip64SelectionSpec
from coleman.corpus.zip64_selection import materialize_zip64_files, validate_zip64_files
from coleman.corpus.rtptorrent import convert_rtptorrent_source
from coleman.corpus.rtptorrent_contracts import RTPTorrentSpec
from native_ci import _native_spec, _check_native
root = Path.cwd()
selection = Zip64SelectionSpec.from_json((root / 'selected/selection.json').read_bytes())
receipt = MaterializationReceipt.from_json((root / 'selected/raw-0.json').read_bytes())
cache = SourceCache(CACHE_ROOT)
validate_zip64_files(selection, cache)
source = validate_fixture_source(load_registry().card('ci-serial-boundary-v1').manifest,
                                cache.member_path(receipt, 'serial.json').read_bytes())
spec = _native_spec(root, source, receipt)
first = run(spec)
second = run(spec)
assert first.run_id == second.run_id
directory = _check_native(second, source)
checkpoints = list((directory / 'checkpoints').rglob('progress_*.json'))
assert checkpoints
assert all(json.loads(path.read_bytes())['step_committed'] == 2 for path in checkpoints)
real = REAL_CACHE
if real:
    frozen = Zip64SelectionSpec.from_json(
        files('coleman.corpus.resources').joinpath('rtptorrent-hikaricp-zip64-selection-v1.json').read_bytes())
    real_cache = SourceCache(real)
    validate_zip64_files(frozen, real_cache)
    raw_receipt, = materialize_zip64_files(frozen, real_cache)
    dataset = convert_rtptorrent_source(RTPTorrentSpec(
        schema_version=1, materialization=raw_receipt, member='hikaricp-class-results.csv',
        project='brettwooldridge/HikariCP', history='observed'), real_cache)
    assert dataset.input_rows == 46995 and len(dataset.job_ids) == 1662
    assert dataset.repeated_class_executions == 23302
    assert dataset.reported_zero_failure_jobs == 1530
    try:
        dataset.require_temporal_evaluation()
    except ValueError:
        pass
    else:
        raise AssertionError('Unexpected temporal admission')
print(json.dumps({'native_rows': 2, 'checkpoint_reopened': True, 'real_source_verified': bool(real)}))
""".replace("CACHE_ROOT", repr(str(cache))).replace(
            "REAL_CACHE", repr(os.environ.get("COLEMAN_RTPTORRENT_ZIP64_REAL_CACHE"))
        )
    )
    result = command([str(python), str(script)], outside, env)
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "native_rows": 2,
        "checkpoint_reopened": True,
        "real_source_verified": bool(os.environ.get("COLEMAN_RTPTORRENT_ZIP64_REAL_CACHE")),
    }
