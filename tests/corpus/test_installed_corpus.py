"""Noneditable CI-capability wheel corpus commands and a real native CI run outside the checkout."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from coleman.corpus import CorpusFixtureReport, CorpusProfileInventory, NativeSourceProfile
from tests.installed_support import build_install_offline_wheel

ROOT = Path(__file__).parents[2]


def command(args, cwd, env, *, check=True):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=check, timeout=240)


def test_installed_registry_correctness_fixtures_and_real_native_ci(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update({"UV_LINK_MODE": "copy", "UV_NO_PROGRESS": "1"})
    distributions = tmp_path / "dist"
    virtual = tmp_path / "installed"
    build_install_offline_wheel(uv, distributions, virtual, env, extras=("ci",))
    python = virtual / "bin/python"
    outside = tmp_path / "outside"
    outside.mkdir()
    origin = command([str(python), "-c", "import coleman; print(coleman.__file__)"], outside, env)
    assert Path(origin.stdout.strip()).resolve().is_relative_to(virtual.resolve())
    command([str(python), "-c", "import importlib.util; assert importlib.util.find_spec('mcp') is None"], outside, env)
    registered = command([str(python), "-m", "coleman.corpus", "list"], outside, env)
    initial = json.loads(registered.stdout)
    assert len(initial["registered_sources"]) == 12
    assert len(initial["blocked_sources"]) == 10
    assert initial["evidence_status"] == "insufficient_diversity"
    assert len(initial["uncovered_required_regimes"]) == 7
    shown = command([str(python), "-m", "coleman.corpus", "show", "ci-serial-boundary-v1"], outside, env)
    assert json.loads(shown.stdout)["manifest"]["benchmark_id"] == "ci-serial-boundary-v1"
    schema = command([str(python), "-m", "coleman.corpus", "schema"], outside, env)
    assert json.loads(schema.stdout)["properties"]["schema_version"]["maximum"] == 1
    for source in ("ci-serial-boundary-v1", "ci-parallel-occupancy-v1"):
        reports = []
        for _ in range(2):
            result = command(
                [str(python), "-m", "coleman.corpus", "validate", source, "--output", "verified"], outside, env
            )
            payload = json.loads(result.stdout)
            assert payload["fixture_status"] == "verified"
            assert payload["generalization"] == "insufficient_evidence"
            reports.append(CorpusFixtureReport.from_json(Path(payload["artifact"]).read_bytes()))
        assert reports[0] == reports[1]
    rejected = command([str(python), "-m", "coleman.corpus", "validate", "../../arbitrary"], outside, env, check=False)
    assert rejected.returncode == 2
    escaped = command(
        [str(python), "-m", "coleman.corpus", "validate", "ci-serial-boundary-v1", "--output", "../escape"],
        outside,
        env,
        check=False,
    )
    assert escaped.returncode == 2
    assert not (tmp_path / "escape").exists()
    example = outside / "ci.py"
    shutil.copyfile(ROOT / "examples/corpus-foundation/ci.py", example)
    native = command([str(python), str(example), "native-output"], outside, env)
    final = json.loads(native.stdout.strip().splitlines()[-1])
    assert final["native_rows"] == 2
    assert final["fixture_sources_verified"] == 2
    assert final["evidence_status"] == "insufficient_diversity"
    assert final["uncovered_regimes"] == 7
    run_root = Path(final["artifacts_dir"])
    manifest = json.loads((run_root / "manifest.json").read_bytes())
    assert len([entry for entry in manifest["artifacts"] if entry["type"] == "corpus_fixture_report"]) == 2

    entries = manifest["artifacts"]
    assert len([entry for entry in entries if entry["type"] == "corpus_source_profile"]) == 1
    assert len([entry for entry in entries if entry["type"] == "corpus_profile_inventory"]) == 1
    source_profile = NativeSourceProfile.from_json(next(run_root.glob("corpus-profiles/*.json")).read_bytes())
    assert source_profile.rows == 6
    assert source_profile.logical_builds == 2
    assert source_profile.duration_median.unit == "s"
    assert source_profile.card.history == "constructed"
    assert source_profile.card.manifest.dependence_group == "coleman-tiny-ci-fixture-family-v1"
    assert source_profile.generalization == "insufficient_evidence"
    assert list((run_root / "checkpoints").rglob("progress_*.json"))
    source = outside / "source.parquet"
    shutil.copyfile(ROOT / "examples/fakedata/features-engineered.parquet", source)
    for _ in range(2):
        described = command(
            [
                str(python),
                "-m",
                "coleman.corpus",
                "profile",
                "native-example-e970afe1686d7fe6",
                str(source),
                "--output",
                "profiles",
            ],
            outside,
            env,
        )
        description = json.loads(described.stdout)
        assert description["profile_status"] == "descriptive_only"
        assert description["rows"] == 29
        assert NativeSourceProfile.from_json(Path(description["artifact"]).read_bytes()).raw_duration.zero_count == 3
    escaped = command(
        [
            str(python),
            "-m",
            "coleman.corpus",
            "profile",
            "native-example-e970afe1686d7fe6",
            str(source),
            "--output",
            "../escape",
        ],
        outside,
        env,
        check=False,
    )
    assert escaped.returncode == 2
    inventory_script = outside / "inventory.py"
    shutil.copyfile(ROOT / "examples/corpus-profiles/inventory.py", inventory_script)
    first = command([str(python), str(inventory_script), str(ROOT), "inventory"], outside, env)
    second = command([str(python), str(inventory_script), str(ROOT), "inventory"], outside, env)
    assert json.loads(first.stdout) == json.loads(second.stdout)
    inventory = CorpusProfileInventory.from_json(Path(json.loads(first.stdout)["inventory"]).read_bytes())
    assert len(inventory.profiles) == 10
    assert len(inventory.uncovered_required_regimes) == 7

    documentation = (ROOT / "docs/benchmark-corpus.md").read_text()
    snippets = re.findall(r"```python\n(.*?)```", documentation, flags=re.S)
    assert len(snippets) == 2
    for snippet in snippets:
        command([str(python), "-c", snippet], outside, env)
