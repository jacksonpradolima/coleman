"""Keep the compact published evidence readable without claiming raw downloads."""

import hashlib
import json
from pathlib import Path

from coleman.results.domain import Measurement
from examples.research.expert_rate_process_records import DevelopmentChoice, OracleResult, StudyReport

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "examples/research/expert-rate-entry-result-v1.json"


def test_compact_public_result_preserves_original_typed_records_and_conservative_decision():
    data = json.loads(SOURCE.read_bytes())
    assert type(data["schema_version"]) is int
    assert data["schema_version"] == 1
    assert data["artifact_kind"] == "coleman-public-expert-rate-entry-result"
    report = StudyReport.from_json(json.dumps(data["study_report"]))
    oracle = OracleResult.from_json(json.dumps(data["oracle_report"]))
    choice = DevelopmentChoice.from_json(json.dumps(data["development_choice"]))
    assert report.status == "completed"
    assert len(report.attempts) == 15
    assert report.known_attempted_calls == 77112
    assert report.reserved_calls == 77112
    assert report.unknown_attempts == 0
    assert oracle.checked_steps == 77112
    assert oracle.repetitions_equal is True
    assert report.gate == oracle.gate
    assert oracle.gate.selected_family == choice.selected_family == "fixed-share"
    assert oracle.gate.selected_slots == choice.selected_slots
    assert oracle.gate.decision == data["decision"] == "retain_baselines"
    assert data["selected_additional_method"] is None
    assert oracle.gate.independent_empirical_units == 0
    assert all(0 < gap < oracle.gate.threshold for _, gap in oracle.gate.shape_excess)
    assert data["raw_artifact_access"]["status"] == "not_committed"
    assert data["raw_artifact_access"]["retained_output_bytes"] == 38059756
    protocol = ROOT / data["source"]["protocol_path"]
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == data["source"]["protocol_raw_sha256"]
    assert json.loads(protocol.read_bytes())["content_digest"] == report.protocol_content


def test_resource_projection_keeps_all_physical_slots_units_missingness_and_source_hashes():
    data = json.loads(SOURCE.read_bytes())
    attempts = data["study_report"]["attempts"]
    projections = data["family_resources"]
    assert tuple((r["family"], r["repetition"]) for r in projections) == tuple(
        (a["family"], a["repetition"]) for a in attempts
    )
    for resource, attempt in zip(projections, attempts, strict=True):
        assert resource["raw_family_artifact"] == attempt["artifact"]
        assert resource["attempted_calls"] == attempt["attempted_calls"]
        measured = tuple(Measurement.model_validate(row) for row in resource["worker_measurements"])
        assert tuple((m.name, m.unit) for m in measured) == (
            ("process_wall", "s"),
            ("process_cpu", "s"),
            ("peak_rss", "MiB"),
        )
        assert all(m.status == "observed" for m in measured)
        assert all(m.value is not None and m.value >= 0 for m in measured)
        assert all(type(value) is int and value > 0 for value in resource["logical_sizes_bytes"].values())
    unavailable = tuple(Measurement.model_validate(row) for row in data["oracle_resource_availability"])
    assert tuple((m.name, m.unit) for m in unavailable) == (("oracle_cpu", "s"), ("oracle_peak_rss", "MiB"))
    assert all(m.status == "unavailable" and m.value is None for m in unavailable)
    assert data["source"]["sealed_driver_files"] == data["study_report"]["source_files"]
    assert len(data["source"]["sealed_driver_files"]) == 6
