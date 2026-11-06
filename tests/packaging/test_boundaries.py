"""Exercise optional selection and extraction without bootstrapping a runtime."""

import importlib
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from coleman.capabilities import CapabilityUnavailableError, installation_capabilities, require_capability

if TYPE_CHECKING:
    from coleman.runner import RunnerExtension


def test_capability_selection_is_closed_and_absence_is_actionable(monkeypatch):
    reads = []

    def metadata(name):
        reads.append(name)
        if name in {"numpy", "pyarrow"}:
            raise PackageNotFoundError(name)
        return "installed"

    monkeypatch.setattr("coleman.capabilities.version", metadata)
    assert tuple(item.name for item in installation_capabilities()) == (
        "ci",
        "analytics",
        "mcp",
        "telemetry",
        "clickhouse",
    )
    assert reads == []
    with pytest.raises(ValueError, match="Unknown COLEMAN.*optimizer"):
        require_capability("optimizer")
    assert reads == []
    with pytest.raises(CapabilityUnavailableError, match=r"numpy, pyarrow.*GitHub.*\[ci\]"):
        require_capability("ci")
    assert reads == ["duckdb", "numpy", "polars", "pyarrow", "pyyaml"]
    require_capability("mcp")


def test_metadata_corruption_is_not_misreported_as_missing_extra(monkeypatch):
    def corrupt(_name):
        raise RuntimeError("unreadable metadata")

    monkeypatch.setattr("coleman.capabilities.version", corrupt)
    with pytest.raises(RuntimeError, match="unreadable metadata"):
        require_capability("ci")


def test_missing_native_capability_precedes_artifacts_and_workers(monkeypatch, tmp_path):
    from coleman.api import run, run_many, run_with_extension
    from coleman.spec.models import RunSpec

    def absent(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("coleman.capabilities.version", absent)
    output = tmp_path / "must-not-exist"
    spec = RunSpec.model_validate({"results": {"out_dir": str(output)}})
    with pytest.raises(CapabilityUnavailableError):
        run(spec)
    with pytest.raises(CapabilityUnavailableError):
        run_with_extension(spec, cast("RunnerExtension", object()))
    with pytest.raises(CapabilityUnavailableError):
        run_many([spec], max_workers=2)
    assert run_many([], max_workers=1) == []
    assert not output.exists()


def test_cli_help_and_all_selected_pure_namespaces_have_cold_boundaries(tmp_path):
    code = """
import importlib, importlib.abc, sys
sys.path.insert(0,sys.argv[1])
blocked=('numpy','polars','pyarrow','duckdb','scipy','mcp','yaml','opentelemetry','clickhouse_connect')
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname==name or fullname.startswith(name+'.') for name in blocked):
            raise AssertionError('Optional dependency imported: '+fullname)
sys.meta_path.insert(0,Boundary())
for name in ('coleman.selection','coleman.allocation','coleman.observations','coleman.evaluation',
             'coleman.evaluation.protected','coleman.spec','coleman.application','coleman.workflows','coleman.corpus'):
    module=importlib.import_module(name)
    assert set(module.__all__) <= set(dir(module))
    try: getattr(module,'unknown_optional_provider')
    except AttributeError: pass
    else: raise AssertionError('Unknown export accepted')
from coleman.core.quantities import Quantity
from coleman.selection.contracts import Quantity as LegacyQuantity
from coleman.core.splits import TemporalObservation,TemporalPlan
from coleman.evaluation.protected.temporal import TemporalObservation as LegacyTemporal
assert Quantity is LegacyQuantity
assert TemporalObservation is LegacyTemporal
from coleman.observations import ObservationHistory,fit_quantity_mean
from coleman.evaluation.protected import BoundedMeanProtocol,holm_adjust,FamilyReport
from coleman.selection import solve_coverage
from coleman.allocation import select_allocation
from coleman.spec import RunSpec,compute_run_id
from coleman.api import sweep
from coleman.application.contracts import Submit
from coleman.corpus.admission_contracts import RealityFact,ObservationProfile
from coleman.corpus.profiles import NumericProfile,NativeSourceProfile
from coleman.capabilities import installation_capabilities
assert holm_adjust((0.01,0.2)) == (0.02,0.2)
import coleman.cli
try: coleman.cli.main(['--help'])
except SystemExit as exc: assert exc.code==0
try: coleman.cli.main(['workflow','--help'])
except SystemExit as exc: assert exc.code==0
assert not any(any(name==x or name.startswith(x+'.') for x in blocked) for name in sys.modules)
native=('coleman.runner','coleman.policy','coleman.agent','coleman.transports')
assert not any(name.startswith(native) for name in sys.modules)
"""
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(Path(__file__).parents[2])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "workflow" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_shared_aliases_preserve_decimal_and_temporal_schema():
    from coleman.core import quantities, splits

    quantity_edge = importlib.import_module("coleman.selection.contracts")
    temporal_edge = importlib.import_module("coleman.evaluation.protected.temporal")
    for name in (
        "Quantity",
        "ticks",
        "decimal_value",
        "decimal_text",
        "validate_decimal_places",
        "validate_decimal_schema",
    ):
        assert getattr(quantity_edge, name) is getattr(quantities, name)
    for name in ("TemporalObservation", "SplitDecision", "TemporalPlan"):
        assert getattr(temporal_edge, name) is getattr(splits, name)
    assert Path(quantities.__file__).parent == Path(splits.__file__).parent


def test_missing_analytics_does_not_reserve_protected_access(monkeypatch, tmp_path):
    from coleman.evaluation.protected import EvaluationAudit, FrozenEvaluation, evaluate

    fixture = Path(__file__).parents[1] / "fixtures/protected-evaluation/freeze-v1.json"
    frozen = FrozenEvaluation.from_json(fixture.read_bytes())
    audit = EvaluationAudit(tmp_path / "untouched-audit")

    def absent(name):
        raise PackageNotFoundError(name)

    def protected_loader():
        raise AssertionError("Missing installation must not load protected outcomes")

    monkeypatch.setattr("coleman.capabilities.version", absent)
    with pytest.raises(CapabilityUnavailableError, match="analytics"):
        evaluate(frozen, audit, protected_loader, actor="installation-test")
    assert audit.history(frozen.protocol.holdout_id) == ()
    assert not audit.root.exists()


def test_missing_sdk_is_reported_before_mcp_workspace_effects(monkeypatch, tmp_path):
    from coleman.transports.mcp import main

    def absent(name):
        raise PackageNotFoundError(name)

    workspace = tmp_path / "uncreated-workspace"
    monkeypatch.setattr("coleman.capabilities.version", absent)
    monkeypatch.setattr(sys, "argv", ["coleman-mcp", "--workspace", str(workspace)])
    with pytest.raises(CapabilityUnavailableError, match="mcp"):
        main()
    assert not workspace.exists()
