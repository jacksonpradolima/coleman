"""Installed native corpus comparison, protected retry and transport-free domain imports."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from coleman.corpus.repeated import RepeatedCorpusReport
from coleman.evaluation.protected import FamilyReport

ROOT = Path(__file__).parents[2]


def test_installed_native_repetitions_and_checkpoint_continuation_outside_checkout(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_NO_PROGRESS="1", UV_LINK_MODE="copy")

    def command(arguments, *, cwd=tmp_path, extra_environment=None):
        result = subprocess.run(
            arguments,
            cwd=cwd,
            env={**env, **(extra_environment or {})},
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    wheel_dir = tmp_path / "wheels"
    command(
        [uv, "build", "--offline", "--no-build-isolation", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=ROOT,
    )
    wheel = next(wheel_dir.glob("*.whl"))
    environment = tmp_path / "installed"
    command([uv, "venv", str(environment), "--python", sys.executable])
    python = environment / "bin" / "python"
    # CI's frozen sync caches pinned wheel URLs, not registry metadata. Require
    # the real offline profile to work with indexes forbidden, even on a warm
    # developer cache, before installing the noneditable wheel without resolving.
    command(
        [
            uv,
            "sync",
            "--frozen",
            "--offline",
            "--no-index",
            "--no-default-groups",
            "--no-dev",
            "--no-install-project",
            "--extra",
            "ci",
            "--extra",
            "analytics",
            "--python",
            str(python),
        ],
        cwd=ROOT,
        extra_environment={"UV_PROJECT_ENVIRONMENT": str(environment)},
    )
    command([uv, "pip", "install", "--offline", "--no-index", "--no-deps", "--python", str(python), str(wheel)])
    command([uv, "pip", "check", "--python", str(python)])
    external = tmp_path / "external"
    external.mkdir()
    script = external / "ci.py"
    shutil.copy(ROOT / "examples/corpus-evaluation/ci.py", script)
    output = external / "output"

    def execute():
        result = subprocess.run(
            [str(python), "-I", str(script), str(output)],
            cwd=external,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        return json.loads(result.stdout.splitlines()[-1])

    first = execute()
    assert first["raw_cells"] == 16
    assert first["independent_units"] == 1
    assert first["conclusion"] == "insufficient_evidence"
    assert first["retry_confirmatory"] is False
    report = RepeatedCorpusReport.from_json(Path(first["raw_evidence"]).read_bytes())
    assert len(report.rows) == 16
    assert report.design.evidence_basis == "constructed_mechanism"
    assert report.design.source_admission_fingerprint is None
    for candidate in report.frozen.candidates:
        configurations = json.loads(candidate.configuration_json)["resolved_runs"]
        assert all(
            config["extensions"]["corpus_source"]["data_fingerprint"] == report.design.data_fingerprint
            for config in configurations
        )
    assert all(row.record.run_id == report.frozen.fingerprint for row in report.rows)
    for project in {member.project_id for member in report.design.builds}:
        timeline = sorted(
            (member.metadata for member in report.design.builds if member.project_id == project),
            key=lambda metadata: metadata.event_at,
        )
        assert timeline[0].available_at < timeline[1].event_at
    comparison = FamilyReport.from_json(Path(first["comparison"]).read_bytes())
    assert len(comparison.inputs) == 2
    protected = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (output / "native").rglob("*")
        if path.is_file() and (path.suffix in {".parquet", ".pkl"} or path.name.startswith("progress_"))
    }
    assert sum(path.suffix == ".pkl" for path in protected) == 8
    second = execute()
    assert second["conclusion"] == "insufficient_evidence"
    assert second["raw_cells"] == 16
    assert second["retry_confirmatory"] is False
    second_report = RepeatedCorpusReport.from_json(Path(second["raw_evidence"]).read_bytes())
    assert {row.record.logical_key for row in second_report.rows}.isdisjoint(
        row.record.logical_key for row in report.rows
    )
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected} == protected
    assert set(path for path in (output / "native").rglob("*.parquet")) == {
        p for p in protected if p.suffix == ".parquet"
    }
    assert Path(first["raw_evidence"]).read_bytes() == (report.to_json() + "\n").encode()
    # Kill a real installed process while its immutable writer is still consuming
    # content. Only an uncommitted temporary file may survive; recovery ignores it.
    crash_root = external / "interrupted-publication"
    crash = """
import os, sys
from pathlib import Path
from coleman.artifacts import ArtifactWriter
from coleman.corpus.repeated import RepeatedCorpusReport, publish_repeated_corpus
report = RepeatedCorpusReport.from_json(Path(sys.argv[1]).read_bytes())
writer = ArtifactWriter(sys.argv[2])
def interrupted(path, content):
    def chunks():
        yield content[:100].encode()
        os._exit(73)
    writer.write_immutable_stream(path, chunks(), maximum_bytes=len(content.encode()))
writer.write_immutable = interrupted
publish_repeated_corpus(report, writer)
"""
    stopped = subprocess.run(
        [str(python), "-I", "-c", crash, first["raw_evidence"], str(crash_root)],
        cwd=external,
        env=env,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert stopped.returncode == 73
    assert not tuple(crash_root.rglob("*.json"))
    assert len(tuple(crash_root.rglob("*.tmp"))) == 1
    recovery = """
import sys
from pathlib import Path
from coleman.artifacts import ArtifactWriter
from coleman.corpus.repeated import RepeatedCorpusReport, publish_repeated_corpus
report = RepeatedCorpusReport.from_json(Path(sys.argv[1]).read_bytes())
path = publish_repeated_corpus(report, ArtifactWriter(sys.argv[2]))
assert RepeatedCorpusReport.from_json(path.read_bytes()) == report
print(path)
"""
    recovered = subprocess.run(
        [str(python), "-I", "-c", recovery, first["raw_evidence"], str(crash_root)],
        cwd=external,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert Path(recovered.stdout.strip()).read_bytes() == Path(first["raw_evidence"]).read_bytes()
    guard = """
import importlib.abc, pathlib, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('coleman.cli', 'coleman.transports', 'mcp')):
            raise AssertionError('Domain attempted transport import: ' + fullname)
sys.meta_path.insert(0, Guard())
import coleman.corpus.repeated as domain
import coleman.corpus.repeated_contracts
assert pathlib.Path(domain.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())
assert not any(name.startswith(('coleman.cli', 'coleman.transports', 'mcp')) for name in sys.modules)
"""
    subprocess.run([str(python), "-I", "-c", guard], cwd=external, env=env, capture_output=True, check=True, timeout=30)
