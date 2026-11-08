"""Complete original evidence survives descriptive use without invented inference."""

import json
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from coleman.evaluation._repetition import repetition_execution_id
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.evaluation.design.panel import RawPanel
from coleman.evaluation.design.projection import project_frozen, qualify_inference
from coleman.evaluation.prequential.evidence import DiagnosticLocator, DiagnosticPanel
from tests.evaluation.design_support import changed, make_selection
from tests.evaluation.prequential.diagnostic_support import declarations, diagnostic_panel, protected_panel


def test_authored_result_free_locator_wire():
    path = Path(__file__).parent / "fixtures" / "comparison-v1" / "diagnostic-locator.json"
    payload = path.read_text().strip()
    assert DiagnosticLocator.from_json(payload).to_json() == payload


def test_shared_grid_keeps_supported_wire_and_original_omissions():
    design, selection = declarations()
    protected = protected_panel(design, selection)
    diagnostic = DiagnosticPanel.model_validate(protected.model_dump(mode="json"))
    assert diagnostic.to_json() == protected.to_json()
    assert tuple(row.evidence.record.to_json() for row in diagnostic.rows) == tuple(
        row.evidence.record.to_json() for row in protected.rows
    )
    assert all("reason" not in row for row in json.loads(diagnostic.to_json())["rows"])
    assert DiagnosticPanel.from_json(diagnostic.to_json()).to_json() == diagnostic.to_json()


def test_unsupported_stays_original_descriptive_and_old_raw_panel_refuses():
    design, selection = declarations(unsupported=True)
    diagnostic = diagnostic_panel(design, selection)
    data = diagnostic.model_dump(mode="json")
    with pytest.raises(ValueError, match="Unsupported evaluation assumptions: dependent_or_unknown_final_units"):
        RawPanel.model_validate(data)
    assert len(diagnostic.rows) == design.raw_cells
    assert "confidence" not in diagnostic.to_json()
    assert "family" not in json.loads(diagnostic.to_json())


@pytest.mark.parametrize("malformation", ["missing", "duplicate", "extra"])
def test_shared_complete_grid_refuses_missing_duplicate_or_extra_cells(malformation):
    design, selection = declarations()
    original = protected_panel(design, selection)
    rows = original.rows[:-1] if malformation == "missing" else (*original.rows, original.rows[0])
    if malformation == "extra":
        row = original.rows[0]
        record = changed(row.record, execution_id=repetition_execution_id(row.candidate_id, 99))
        extra = changed(row, seed=99, evidence=OriginalRecordEvidence.from_record(record))
        rows = (*original.rows, extra)
    data = original.model_dump(mode="python")
    data["rows"] = rows
    for reader in (RawPanel, DiagnosticPanel):
        with pytest.raises(ValueError, match="required panel cell|unexpected or duplicate"):
            reader.model_validate(data)
    assert original.rows == tuple(original.rows)


def test_wrong_base_configuration_is_not_excused_by_unsupported_inference():
    design, selection = declarations(unsupported=True)
    original = diagnostic_panel(design, selection)
    row = original.rows[0]
    record = changed(row.record, provenance=changed(row.record.provenance, spec_fingerprint="sha256:" + "a" * 64))
    wrong = changed(row, evidence=OriginalRecordEvidence.from_record(record))
    data = original.model_dump(mode="python")
    data["rows"] = (wrong, *original.rows[1:])
    with pytest.raises(ValueError, match="BASE candidate"):
        DiagnosticPanel.model_validate(data)


@pytest.mark.parametrize("version", [0, 2])
def test_diagnostic_version_and_unknown_fields_reject(version):
    design, selection = declarations()
    original = diagnostic_panel(design, selection)
    data = original.model_dump(mode="json")
    data["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        DiagnosticPanel.model_validate(data)
    unknown = original.model_dump(mode="json")
    unknown["inferential_winner"] = "candidate"
    with pytest.raises(ValueError, match="inferential_winner"):
        DiagnosticPanel.model_validate(unknown)


def test_observed_zero_and_missing_required_value_remain_different():
    design, selection = declarations(unsupported=True)
    key = ("candidate", design.protected_members[0].member_id, design.seeds[0])
    zero = diagnostic_panel(design, selection, values={key: 0.0})
    absent = diagnostic_panel(design, selection, values={key: None}, states={key: "failed"})
    left = next(row for row in zero.rows if (row.candidate_id, row.member_id, row.seed) == key)
    right = next(row for row in absent.rows if (row.candidate_id, row.member_id, row.seed) == key)
    assert left.record.measurement(design.primary_metric).value == 0.0
    assert right.record.measurement(design.primary_metric).status == "unavailable"
    assert len(zero.rows) == len(absent.rows)


@pytest.mark.parametrize("reader", [DiagnosticPanel.from_json, DiagnosticPanel.model_validate_json])
def test_actual_diagnostic_json_routes_bound_depth(reader):
    payload = "[" * 1000 + "0" + "]" * 1000
    with pytest.raises(ValueError, match="depth"):
        reader(payload)


@pytest.mark.parametrize("assumption", ["serial", "clustered", "adaptive_partial", "optional_stopping", "unbounded"])
def test_original_grid_can_remain_descriptive_under_explicit_unsupported_assumptions(assumption):
    design, selection = declarations()
    original = diagnostic_panel(design, selection)
    if assumption in {"serial", "clustered"}:
        design = changed(design, inference=changed(design.inference, dependence=assumption))
    elif assumption == "adaptive_partial":
        design = changed(design, inference=changed(design.inference, assignment=assumption))
    elif assumption == "optional_stopping":
        design = changed(design, inference=changed(design.inference, analysis=assumption))
    else:
        metric = design.metric(design.primary_metric)
        unbounded = changed(
            metric, quantity=changed(metric.quantity, spec=changed(metric.quantity.spec, minimum=None, maximum=None))
        )
        design = changed(design, metrics=(unbounded,), inference=changed(design.inference, bounds_basis=None))
    selected = make_selection(design, actual_selected_at=selection.actual_selected_at)
    descriptive = DiagnosticPanel(
        schema_version=1,
        design=design,
        selection=selected,
        rows=original.rows,
        ledger=original.ledger,
        mode=original.mode,
        assembled_at=original.assembled_at,
    )
    assert qualify_inference(design, selected).status == "unsupported"
    assert tuple(row.record_digest for row in descriptive.rows) == tuple(row.record_digest for row in original.rows)
    with pytest.raises(ValueError, match="Unsupported evaluation assumptions"):
        project_frozen(design, selected)


def test_nonfinite_raw_diagnostic_does_not_reach_original_record_relabeling():
    design, selection = declarations(unsupported=True)
    original = diagnostic_panel(design, selection)
    data = original.model_dump(mode="json")
    data["rows"][0]["evidence"]["record"]["measurements"][0]["value"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        DiagnosticPanel.model_validate(data)


@pytest.mark.parametrize("module", ["evidence", "comparison"])
def test_new_pure_declarations_have_genuine_cold_dependency_boundary(module):
    script = """
import importlib
import importlib.abc
import sys
sys.path[:0] = sys.argv[2:]
blocked = ('coleman.replay', 'coleman.state', 'coleman.cli', 'coleman.mcp',
           'coleman.native', 'coleman.artifacts', 'sqlite3',
           'numpy', 'pandas', 'pyarrow', 'scipy', 'duckdb')
def forbidden(name):
    return any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)
preloaded = sorted(name for name in sys.modules if forbidden(name))
assert not preloaded, preloaded
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            raise AssertionError('Pure declaration imported ' + fullname)
sys.meta_path.insert(0, Guard())
importlib.import_module('coleman.evaluation.prequential.' + sys.argv[1])
loaded = sorted(name for name in sys.modules if forbidden(name))
assert not loaded, loaded
print('pure-declaration-ok')
"""
    repository = Path(__file__).resolve().parents[3]
    paths = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, module, str(repository), *paths],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "pure-declaration-ok"
