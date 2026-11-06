"""CI-capability wheel materialization and actual checkpointed native CI outside the repository, without network."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from coleman.corpus import MaterializationReceipt
from tests.installed_support import build_install_offline_wheel

ROOT = Path(__file__).parents[2]


def command(args, cwd, env, *, check=True):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=check, timeout=240)


def test_installed_external_cache_owned_zip_native_resume_and_commands(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update({"UV_LINK_MODE": "copy", "UV_NO_PROGRESS": "1"})
    dist = tmp_path / "dist"
    virtual = tmp_path / "installed"
    build_install_offline_wheel(uv, dist, virtual, env, extras=("ci",))
    python = virtual / "bin/python"
    outside = tmp_path / "outside"
    outside.mkdir()
    code = (
        "import coleman,importlib.util,sys; print(coleman.__file__); "
        "assert importlib.util.find_spec('mcp') is None; import coleman.corpus; "
        "assert 'coleman.cli' not in sys.modules; assert 'coleman.transports.corpus_http' not in sys.modules"
    )
    origin = command([str(python), "-c", code], outside, env)
    assert Path(origin.stdout.strip()).is_relative_to(virtual)
    script = outside / "ci.py"
    shutil.copyfile(ROOT / "examples/corpus-materialization/ci.py", script)
    cache = tmp_path / "external-cache"
    outcomes = []
    for _ in range(2):
        result = command([str(python), str(script), "owned-ci", str(cache)], outside, env)
        outcome = json.loads(result.stdout.strip().splitlines()[-1])
        assert outcome["native_rows"] == 2
        assert outcome["members_verified"] == 2
        assert outcome["admission"] == "not_evaluated"
        assert outcome["generalization"] == "insufficient_evidence"
        outcomes.append(MaterializationReceipt.from_json(Path(outcome["receipt"]).read_bytes()))
    assert outcomes[0] == outcomes[1]
    native_root = Path(outcome["native_root"])
    checkpoints = list((native_root / "checkpoints").rglob("progress_*.json"))
    assert checkpoints
    assert all(json.loads(path.read_bytes())["step_committed"] == 2 for path in checkpoints)
    assert len(list(cache.glob("blobs/*"))) == 3
    package = outside / "owned-ci/package.json"
    source = outside / "owned-ci/owned-source.zip"
    prefix = [str(python), "-m", "coleman.corpus"]
    schema = command([*prefix, "materialization-schema"], outside, env)
    assert json.loads(schema.stdout)["properties"]["schema_version"]["maximum"] == 1
    for acquisition in (["--source", str(source)], []):
        fetch = command(
            [*prefix, "fetch", str(package), "--cache", str(cache), "--output", "cli-artifacts", *acquisition],
            outside,
            env,
        )
        summary = json.loads(fetch.stdout)
        assert summary["integrity"] == "verified"
        assert summary["admission"] == "not_evaluated"
        validation = command([*prefix, "validate-cache", summary["artifact"], "--cache", str(cache)], outside, env)
        assert json.loads(validation.stdout)["integrity"] == "verified"
    altered = outside / "altered.zip"
    altered.write_bytes(b"different source bytes")
    rejected = command(
        [*prefix, "fetch", str(package), "--source", str(altered), "--cache", str(cache)], outside, env, check=False
    )
    assert rejected.returncode == 2
    rejected = command(
        [*prefix, "fetch", str(package), "--cache", str(cache), "--output", "../escape"], outside, env, check=False
    )
    assert rejected.returncode == 2
    assert not (tmp_path / "escape").exists()
    docs = (ROOT / "docs/corpus-materialization.md").read_text()
    snippets = re.findall(r"```python\n(.*?)```", docs, flags=re.S)
    assert snippets
    for snippet in snippets:
        command([str(python), "-c", snippet], outside, env)
