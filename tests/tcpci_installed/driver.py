"""Copied outside-checkout verifier for the actual base-wheel CI composition."""

from __future__ import annotations

import importlib.abc
import importlib.metadata
import importlib.util
import json
import sys
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from zipfile import ZipFile

BLOCKED = (
    "numpy",
    "polars",
    "pyarrow",
    "duckdb",
    "scipy",
    "mcp",
    "sqlite3",
    "coleman.application",
    "coleman.workflows",
    "coleman.runner",
    "coleman.api",
    "coleman.agent",
    "coleman.policy",
    "coleman.reward",
    "coleman.bandit",
    "coleman.environment",
    "coleman.scenarios",
    "coleman.cli",
    "coleman.transports",
    "coleman.artifacts",
    "coleman.telemetry",
    "coleman.corpus",
    "coleman.native",
    "coleman.state.local",
    "coleman.results.parquet_sink",
    "coleman.results.duckdb_sink",
    "coleman.results.clickhouse_sink",
)


class Boundary(importlib.abc.MetaPathFinder):
    """Fail on a real forbidden native, transport or persistence import."""

    def find_spec(self, fullname, path=None, target=None):
        """Refuse the blocked package and every submodule before loader lookup."""
        del path, target
        if any(fullname == name or fullname.startswith(name + ".") for name in BLOCKED):
            raise AssertionError("Independent CI composition crossed a boundary: " + fullname)


def _verify_archive(wheel):
    """Compare every actual installed SDK resource with the noneditable archive."""
    package = import_module("coleman")
    origin = package.__file__
    assert origin is not None
    site = Path(origin).resolve().parent.parent
    assert site.is_relative_to(Path(sys.prefix).resolve())
    assert not site.is_relative_to(Path.cwd())
    distribution = importlib.metadata.distribution("coleman")
    metadata = distribution.read_text("direct_url.json")
    assert metadata is not None
    direct = json.loads(metadata)
    assert direct.get("dir_info", {}).get("editable") is not True
    assert direct["url"] == wheel.resolve().as_uri()
    sdk = []
    with ZipFile(wheel) as archive:
        for member in archive.infolist():
            if member.filename.startswith("coleman/") and not member.is_dir():
                assert archive.read(member) == (site / member.filename).read_bytes(), member.filename
                sdk.append(member.filename)
    assert sdk
    return {
        "installed_noneditable": True,
        "SDK_resources_verified": len(sdk),
        "wheel_sha256": sha256(wheel.read_bytes()).hexdigest(),
    }


def _load_example():
    """Load only the two explicitly copied public example files under -I."""
    spec = importlib.util.spec_from_file_location("installed_ci_composition", Path("ci.py"))
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    return example


def _verify_run(example):
    """Execute actual methods and verify original producer/feedback/state joins."""
    actions = import_module("coleman.core.actions")
    feedback = import_module("coleman.core.feedback")
    primitives = import_module("coleman.core.primitives")
    problems = import_module("coleman.problems.tcpci")
    result = example.run_example()
    body = example.full_report(result)
    wire = primitives.canonical_json(body)
    assert primitives.canonical_json(example.full_report(example.run_example())) == wire
    for row in result.rounds:
        assert isinstance(row.mapping.executed.action, actions.SlateAction)
        assert isinstance(row.selected_feedback.feedback, feedback.ComponentFeedback)
        assert isinstance(row.expert_feedback.feedback, feedback.FullPanel)
        assert row.selected_feedback.decision_fingerprint == row.mapping.executed.fingerprint
        assert row.expert_feedback.decision_fingerprint == row.coefficient_producer.fingerprint
        assert row.expert_feedback.decision_fingerprint != row.selected_feedback.decision_fingerprint
        assert problems.TcpciMapping.from_json(row.mapping.to_json()).to_json() == row.mapping.to_json()
        assert row.selected_feedback.horizon == row.mapping.problem.mapping.quantity.horizon
        assert row.mapping.executed.sampling_disposition == "invalidated"
    assert result.rounds[-1].after_state.update_count == 3
    assert result.coverage.selected_ids == ("A", "C")
    assert result.coverage.status == "optimal"
    assert not any(
        any(name == blocked or name.startswith(blocked + ".") for blocked in BLOCKED) for name in sys.modules
    )
    return {
        "full_report_sha256": sha256(wire.encode()).hexdigest(),
        "boundary_clear": True,
        "summary": example.summary(result),
        "full_report": body,
    }


def main():
    """Run actual current-wheel methods under a native/transport/storage guard."""
    for name in ("numpy", "polars", "pyarrow", "duckdb", "scipy", "mcp"):
        assert importlib.util.find_spec(name) is None, name
    sys.meta_path.insert(0, Boundary())
    installation = _verify_archive(Path(sys.argv[1]))
    before = sorted(path.name for path in Path.cwd().iterdir())
    result = _verify_run(_load_example())
    assert sorted(path.name for path in Path.cwd().iterdir()) == before
    primitives = import_module("coleman.core.primitives")
    print(primitives.canonical_json({**installation, **result}))


if __name__ == "__main__":
    main()
