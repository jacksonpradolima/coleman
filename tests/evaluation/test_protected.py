"""Leakage, paired inference, frozen identities and durable protected access tests."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats
from statsmodels.stats.multitest import multipletests

from coleman.evaluation.protected import (
    AccessReceipt,
    ComparisonHypothesis,
    EvaluationAudit,
    EvaluationProtocol,
    FamilyReport,
    FrozenCandidate,
    FrozenEvaluation,
    PairedObservation,
    TemporalObservation,
    TemporalPlan,
    evaluate,
    freeze,
    holm_adjust,
    named_seed,
)
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance, fingerprint
from coleman.results.parquet_sink import ParquetSink

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "protected-evaluation"


def plan():
    return TemporalPlan.model_validate(
        {
            "train_end": "2026-01-02T23:59:59Z",
            "validation_end": "2026-01-04T23:59:59Z",
            "test_end": "2026-01-06T23:59:59Z",
        }
    )


def metadata(identifier="build", group="project", event="2026-01-05T12:00:00Z", available=None, **overrides):
    return TemporalObservation.model_validate(
        {
            "observation_id": identifier,
            "group_id": group,
            "event_at": event,
            "available_at": available or event,
            **overrides,
        }
    )


def frozen_contract(**protocol_overrides):
    protocol = {
        "protocol_id": "paired-ci-v1",
        "holdout_id": "public-ci-v1",
        "metric_name": "napfd",
        "unit": "1",
        "meaning": "napfd-verdict-v1",
        "direction": "maximize",
        "independent_unit": "project",
        "dependence": "independent",
        "normal_differences_assumed": True,
        "practical_effect": 0.01,
        "minimum_units": 2,
        **protocol_overrides,
    }
    candidates = (
        FrozenCandidate.from_configuration("A", {"policy": "Greedy", "preprocessor": "identity-v1"}),
        FrozenCandidate.from_configuration("B", {"policy": "Random", "seed": 42}),
    )
    return freeze(
        candidates,
        fingerprint({"dataset": "synthetic-independent-projects-v1"}),
        plan(),
        EvaluationProtocol.model_validate(protocol),
        (ComparisonHypothesis(hypothesis_id="A-B", candidate_id="A", baseline_id="B"),),
    )


def observations(frozen, differences=(0.1, 0.2, 0.15, 0.25), *, baseline=0.3, unavailable_index=None):
    rows = []
    for index, difference in enumerate(differences):
        meta = metadata(identifier=f"build-{index}", group=f"project-{index:04d}")
        for candidate_id, value in (("A", baseline + difference), ("B", baseline)):
            missing = candidate_id == "A" and index == unavailable_index
            record = DomainRecord(
                schema_version=1,
                namespace="ci.independent_projects",
                run_id=f"run-{candidate_id}",
                execution_id=meta.group_id,
                decision_id=meta.observation_id,
                data_fingerprint=frozen.data_fingerprint,
                available_at=meta.available_at,
                provenance=ResultProvenance(
                    producer="ci-fixture",
                    producer_version="1",
                    spec_fingerprint=frozen.candidate(candidate_id).configuration_fingerprint,
                    evaluation_fingerprint=frozen.fingerprint,
                ),
                measurements=(
                    Measurement(
                        name=frozen.protocol.metric_name,
                        value=None if missing else value,
                        status="unavailable" if missing else "observed",
                        unit=frozen.protocol.unit,
                        meaning=frozen.protocol.meaning,
                        direction=frozen.protocol.direction,
                    ),
                ),
            )
            rows.append(PairedObservation(candidate_id=candidate_id, metadata=meta, record=record))
    return tuple(rows)


def test_issue_fixture_late_availability_and_deterministic_freeze():
    temporal = plan()
    late = metadata(event="2026-01-02T12:00:00Z", available="2026-01-03T12:00:00Z", group="g1")
    assert not temporal.training_eligible(late)
    decision = temporal.decide(late)
    assert decision.partition == "train"
    assert decision.reason == "outcome_not_available"
    frozen = frozen_contract()
    same = freeze(frozen.candidates, frozen.data_fingerprint, frozen.plan, frozen.protocol, frozen.comparison_family)
    assert same.fingerprint == frozen.fingerprint
    assert named_seed(42, "candidate-A", "evaluation") == 10676855509616717645


@pytest.mark.parametrize(
    "timestamp,partition",
    [
        ("2026-01-02T23:59:59Z", "train"),
        ("2026-01-03T00:00:00Z", "validation"),
        ("2026-01-04T23:59:59Z", "validation"),
        ("2026-01-05T00:00:00Z", "test"),
        ("2026-01-06T23:59:59Z", "test"),
        ("2026-01-07T00:00:00Z", None),
    ],
)
def test_cutoff_boundaries_and_utc_normalization(timestamp, partition):
    observation = metadata(event=timestamp)
    decision = plan().decide(observation)
    assert decision.partition == partition
    assert decision.eligible is (partition is not None)
    zoned = metadata(event="2026-01-05T09:00:00-03:00")
    assert zoned == metadata()


def test_overlapping_targets_revisions_and_group_conflicts():
    temporal = plan()
    original = metadata(event="2026-01-02T12:00:00Z")
    future_revision = metadata(event="2026-01-02T12:00:00Z", available="2026-01-05T00:00:00Z", revision_id="revision-2")
    crossing = metadata(
        identifier="crossing", group="second", event="2026-01-02T12:00:00Z", target_end="2026-01-05T00:00:00Z"
    )
    decisions = temporal.split((future_revision, crossing, original))
    assert {
        item.observation.revision_id: item.eligible for item in decisions if item.observation.observation_id == "build"
    } == {
        "original": True,
        "revision-2": False,
    }
    assert temporal.decide(crossing).reason == "overlapping_target"
    other = metadata(identifier="other", event="2026-01-05T12:00:00Z")
    with pytest.raises(ValueError, match="spans"):
        temporal.split((future_revision, other))
    duplicate = (original, original)
    with pytest.raises(ValueError, match="Duplicate"):
        temporal.split(duplicate)
    changed = metadata(event="2026-01-02T13:00:00Z", revision_id="revision-2")
    with pytest.raises(ValueError, match="identity"):
        temporal.split((original, changed))


def test_configurable_mapping_missing_fields_and_invalid_chronology():
    temporal = TemporalPlan.model_validate(
        {
            **plan().model_dump(),
            "event_field": "event",
            "availability_field": "label",
            "group_field": "repository",
            "observation_field": "build",
        }
    )
    values = {
        "event": "2026-01-02T12:00:00Z",
        "label": "2026-01-02T12:00:00Z",
        "repository": "ci",
        "build": "1",
        "unused_raw_outcome": -3.0,
    }
    assert temporal.training_eligible(temporal.observation(values))
    del values["label"]
    with pytest.raises(KeyError):
        temporal.observation(values)
    for timestamp in ("2026-01-02", "1767355200", 1767355200):
        with pytest.raises(ValidationError):
            metadata(event=timestamp)
    with pytest.raises(ValidationError, match="precede"):
        metadata(event="2026-01-05T12:00:00Z", available="2026-01-04T12:00:00Z")
    with pytest.raises(ValidationError, match="ends"):
        metadata(target_end="2026-01-04T12:00:00Z")
    reversed_plan = {**plan().model_dump(), "train_end": plan().test_end}
    with pytest.raises(ValidationError):
        TemporalPlan.model_validate(reversed_plan)
    duplicate_fields = {**plan().model_dump(), "group_field": "event_at"}
    with pytest.raises(ValidationError):
        TemporalPlan.model_validate(duplicate_fields)


def test_freeze_canonical_order_immutable_config_and_changed_protocols():
    original = frozen_contract()
    reordered = freeze(
        tuple(reversed(original.candidates)),
        original.data_fingerprint,
        original.plan,
        original.protocol,
        tuple(reversed(original.comparison_family)),
    )
    assert reordered.to_json() == original.to_json()
    assert FrozenEvaluation.from_json(original.to_json()) == original
    config = {"parameters": {"weight": 1}, "preprocessor": "identity-v1"}
    candidate = FrozenCandidate.from_configuration("C", config)
    config["parameters"]["weight"] = 999
    assert json.loads(candidate.configuration_json)["parameters"]["weight"] == 1
    assert frozen_contract(practical_effect=0.5).fingerprint != original.fingerprint
    assert frozen_contract(independent_unit="seed").fingerprint != original.fingerprint
    payload = json.loads(original.to_json())
    payload["candidates"][0]["configuration_json"] = '{"policy":"Greedy","learned_state":"changed"}'
    changed = FrozenEvaluation.from_json(json.dumps(payload))
    assert changed.fingerprint != original.fingerprint
    with pytest.raises(KeyError):
        original.candidate("missing")


@pytest.mark.parametrize("version", [0, 2, True, "1", None])
def test_freeze_future_unknown_and_ambiguous_versions_rejected(version):
    payload = json.loads(frozen_contract().to_json())
    payload["schema_version"] = version
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError):
        FrozenEvaluation.from_json(serialized)


def test_duplicate_and_nonjson_configuration_family_references():
    for value in ('{"x":1,"x":2}', "[]", '{"x":NaN}'):
        with pytest.raises((ValidationError, ValueError)):
            FrozenCandidate(candidate_id="A", configuration_json=value)
    frozen = frozen_contract()
    with pytest.raises(ValueError, match="Duplicate"):
        FrozenEvaluation.from_json('{"schema_version":1,"schema_version":1}')
    payload = json.loads(frozen.to_json())
    payload["comparison_family"][0]["candidate_id"] = "unfrozen"
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError, match="unfrozen"):
        FrozenEvaluation.from_json(serialized)
    duplicate_family = frozen.comparison_family * 2
    with pytest.raises(ValidationError):
        freeze(frozen.candidates, frozen.data_fingerprint, frozen.plan, frozen.protocol, duplicate_family)
    duplicate_candidates = frozen.candidates[:1] * 2
    with pytest.raises(ValidationError):
        freeze(duplicate_candidates, frozen.data_fingerprint, frozen.plan, frozen.protocol, frozen.comparison_family)
    with pytest.raises(ValidationError):
        ComparisonHypothesis(hypothesis_id="A-A", candidate_id="A", baseline_id="A")


def test_named_seed_order_workers_and_stream_separation():
    names = ("A", "B", "C")
    expected = {name: named_seed(42, name, "evaluation") for name in names}
    assert {name: named_seed(42, name, "evaluation") for name in reversed(names)} == expected
    with ProcessPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(named_seed, [42] * 3, names, ["evaluation"] * 3))
    assert dict(zip(names, actual, strict=True)) == expected
    assert len(set(expected.values())) == 3
    assert named_seed(42, "A", "training") != expected["A"]
    assert named_seed(43, "A", "evaluation") != expected["A"]
    for raw_root in (-1, True, "42"):
        root = cast(Any, raw_root)
        with pytest.raises(ValueError):
            named_seed(root, "A", "evaluation")
    for raw_candidate, purpose in (("", "training"), ("A", ""), (None, "training")):
        candidate = cast(Any, raw_candidate)
        with pytest.raises(ValueError):
            named_seed(42, candidate, purpose)


def test_holm_matches_independent_reference_and_metamorphic_order():
    values = (0.01, 0.04, 0.03, 0.4, 1.0, 0.03)
    expected = multipletests(values, method="holm")[1]
    assert holm_adjust(values) == pytest.approx(expected)
    assert tuple(reversed(holm_adjust(tuple(reversed(values))))) == pytest.approx(expected)
    assert holm_adjust(()) == ()
    for value in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            holm_adjust((value,))


def test_paired_mean_interval_and_pvalue_match_scipy(tmp_path):
    frozen = frozen_contract()
    differences = (0.1, 0.2, 0.15, 0.25)
    report = evaluate(frozen, EvaluationAudit(tmp_path), lambda: observations(frozen, differences), actor="ci")
    result = report.results[0]
    expected = stats.ttest_rel(np.asarray(differences) + 0.3, np.full(4, 0.3))
    interval = expected.confidence_interval(confidence_level=0.95)
    assert result.raw_mean_difference == pytest.approx(np.mean(differences))
    assert result.ci_lower == pytest.approx(interval.low)
    assert result.ci_upper == pytest.approx(interval.high)
    assert result.adjusted_p_value == pytest.approx(expected.pvalue)
    assert result.conclusion == "evidence_of_benefit"
    assert report.confirmatory_eligible
    assert FamilyReport.from_json(report.to_json()) == report


@pytest.mark.parametrize(
    "direction,effect,conclusion",
    [
        ("maximize", -0.2, "evidence_of_harm"),
        ("minimize", -0.2, "evidence_of_benefit"),
        ("minimize", 0.2, "evidence_of_harm"),
    ],
)
def test_signed_negative_results_and_minimization(tmp_path, direction, effect, conclusion):
    frozen = frozen_contract(
        direction=direction, metric_name="net_ci_credit", unit="credit", meaning="synthetic-net-ci-credit-v1"
    )
    differences = tuple(effect + noise for noise in (-0.02, 0.01, 0.03, -0.01, 0.0))
    report = evaluate(
        frozen, EvaluationAudit(tmp_path), lambda: observations(frozen, differences, baseline=-2), actor="ci"
    )
    result = report.results[0]
    assert result.raw_mean_difference == pytest.approx(np.mean(differences))
    assert result.conclusion == conclusion
    records = report.domain_records("ci-campaign")
    sink = ParquetSink(out_dir=str(tmp_path / "results"))
    sink.write_domain_record(records[0])
    sink.write_domain_record(records[0])
    assert list(sink.domain_records()) == list(records)
    sink.close()


@pytest.mark.parametrize(
    "differences,missing,reason",
    [
        ((), None, "too_few_independent_units"),
        ((0.0,), None, "too_few_independent_units"),
        ((0.0, 0.0, 0.0), None, "degenerate_difference_variance"),
        ((-2.0, -2.0, -2.0), None, "degenerate_difference_variance"),
        ((0.1, 0.2, 0.3), 1, "incomplete_outcomes"),
    ],
)
def test_insufficient_zero_null_and_unknown_uncertainty(tmp_path, differences, missing, reason):
    frozen = frozen_contract()
    report = evaluate(
        frozen,
        EvaluationAudit(tmp_path),
        lambda: observations(frozen, differences, unavailable_index=missing),
        actor="ci",
    )
    result = report.results[0]
    assert result.reason == reason
    assert result.conclusion == "insufficient_evidence"
    assert result.ci_lower is None
    assert result.adjusted_p_value is None
    if differences and missing is None:
        assert result.raw_mean_difference == pytest.approx(np.mean(differences))
    else:
        assert result.raw_mean_difference is None
    assert report.domain_records("r")[0].measurement("ci_lower").status == "unavailable"


def test_large_practical_threshold_does_not_promote_small_significant_difference(tmp_path):
    frozen = frozen_contract(practical_effect=10.0)
    report = evaluate(frozen, EvaluationAudit(tmp_path), lambda: observations(frozen), actor="ci")
    assert report.results[0].conclusion == "inconclusive"
    adjusted_p = report.results[0].adjusted_p_value
    assert adjusted_p is not None
    assert adjusted_p < 0.05


def test_holdout_reuse_tuning_and_candidate_revision_are_persisted(tmp_path):
    frozen = frozen_contract()
    audit = EvaluationAudit(tmp_path)
    audit.access(frozen, partition="train", purpose="development", actor="ci")
    audit.access(frozen, partition="validation", purpose="tuning", actor="ci")
    first = evaluate(frozen, audit, lambda: observations(frozen), actor="ci")
    assert first.access.prior_test_accesses == 0
    changed = frozen_contract(practical_effect=0.02)
    second = evaluate(changed, EvaluationAudit(tmp_path), lambda: observations(changed), actor="ci")
    assert second.access.prior_test_accesses == 1
    assert not second.confirmatory_eligible
    assert second.results[0].conclusion == "inconclusive"
    assert second.results[0].reason == "reused_holdout_no_confirmatory_claim"
    history = audit.history(frozen.protocol.holdout_id)
    assert len(history) == 4
    assert audit.load_freeze(first.frozen.fingerprint) == first.frozen
    exposed = EvaluationAudit(tmp_path / "tuning")
    exposed.access(frozen, partition="test", purpose="tuning", actor="ci")
    tainted = evaluate(frozen, exposed, lambda: observations(frozen), actor="ci")
    assert not tainted.confirmatory_eligible


@pytest.mark.parametrize("dependence,normal", [("serial", True), ("clustered", True), ("independent", False)])
def test_unsupported_dependence_or_assumptions_reject_before_loading(tmp_path, dependence, normal):
    frozen = frozen_contract(dependence=dependence, normal_differences_assumed=normal)

    def forbidden():
        raise AssertionError("Outcomes must not be loaded")

    audit = EvaluationAudit(tmp_path)
    with pytest.raises(NotImplementedError):
        evaluate(frozen, audit, forbidden, actor="ci")
    assert list(tmp_path.rglob("*.json")) == []


def test_loader_failure_reservation_prevents_falsely_virgin_retry(tmp_path):
    frozen = frozen_contract()
    audit = EvaluationAudit(tmp_path)

    def fail():
        raise RuntimeError("interrupted outcome loading")

    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate(frozen, audit, fail, actor="ci")
    retry = evaluate(frozen, EvaluationAudit(tmp_path), lambda: observations(frozen), actor="ci")
    assert retry.access.prior_test_accesses == 1
    assert not retry.confirmatory_eligible


def test_durable_access_precedes_loading_and_io_failure_cannot_read(tmp_path, monkeypatch):
    frozen = frozen_contract()
    audit = EvaluationAudit(tmp_path)

    def load():
        assert len(audit.history(frozen.protocol.holdout_id)) == 1
        return observations(frozen)

    evaluate(frozen, audit, load, actor="ci")

    def interrupted(*_args):
        raise OSError("disk error")

    monkeypatch.setattr(os, "link", interrupted)

    def forbidden():
        raise AssertionError("No outcome access after failed reservation")

    failed_audit = EvaluationAudit(tmp_path / "failure")
    with pytest.raises(OSError, match="disk error"):
        evaluate(frozen, failed_audit, forbidden, actor="ci")


def test_provenance_future_labels_compatibility_duplicate_seed_and_pairing_errors(tmp_path):
    frozen = frozen_contract()
    base = observations(frozen)
    bad_payloads = []
    for field, value in (("data_fingerprint", fingerprint([])),):
        payload = base[0].model_dump(mode="json")
        payload["record"][field] = value
        bad_payloads.append(payload)
    for field, value in (("evaluation_fingerprint", fingerprint([])), ("spec_fingerprint", fingerprint([]))):
        payload = base[0].model_dump(mode="json")
        payload["record"]["provenance"][field] = value
        bad_payloads.append(payload)
    for field, value in (("unit", "ms"), ("meaning", "other-v2"), ("direction", "minimize")):
        payload = base[0].model_dump(mode="json")
        payload["record"]["measurements"][0][field] = value
        bad_payloads.append(payload)
    payload = base[0].model_dump(mode="json")
    payload["candidate_id"] = "unfrozen-candidate"
    bad_payloads.append(payload)
    payload = base[0].model_dump(mode="json")
    payload["metadata"]["available_at"] = "2026-01-07T12:00:00Z"
    payload["record"]["available_at"] = payload["metadata"]["available_at"]
    bad_payloads.append(payload)
    for index, payload in enumerate(bad_payloads):
        bad = PairedObservation.model_validate(payload)
        rows = (bad, *base[1:])
        audit = EvaluationAudit(tmp_path / str(index))
        with pytest.raises(ValueError):
            evaluate(frozen, audit, lambda rows=rows: rows, actor="ci")
    repeated = (*base, base[0])
    seed_audit = EvaluationAudit(tmp_path / "seeds")
    with pytest.raises(ValueError, match="not independent"):
        evaluate(frozen, seed_audit, lambda: repeated, actor="ci")
    unpaired = base[:-1]
    unpaired_audit = EvaluationAudit(tmp_path / "unpaired")
    with pytest.raises(ValueError, match="same paired"):
        evaluate(frozen, unpaired_audit, lambda: unpaired, actor="ci")


def _reserve_access(root, serialized):
    audit = EvaluationAudit(root)
    frozen = FrozenEvaluation.from_json(serialized)
    return audit.access(frozen, partition="test", purpose="confirmatory", actor="worker").sequence


def test_parallel_process_reservations_have_one_first_access(tmp_path):
    frozen = frozen_contract()
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_reserve_access, str(tmp_path), frozen.to_json()) for _ in range(6)]
        sequences = [future.result(timeout=60) for future in futures]
    assert sorted(sequences) == list(range(1, 7))
    history = EvaluationAudit(tmp_path).history(frozen.protocol.holdout_id)
    assert [item.prior_test_accesses for item in history] == list(range(6))


@pytest.mark.parametrize("stage", ["before", "after"])
def test_process_interruption_reservation_is_conservative_on_resume(tmp_path, stage):
    frozen = frozen_contract()
    script = """
import os, sys
from coleman.evaluation.protected import EvaluationAudit, FrozenEvaluation
original = os.link
def interrupt(source, destination):
    if "evaluation-access" in str(destination):
        if sys.argv[3] == "after":
            original(source, destination)
        os._exit(73)
    return original(source, destination)
os.link = interrupt
frozen = FrozenEvaluation.from_json(sys.argv[1])
EvaluationAudit(sys.argv[2]).access(frozen, partition="test", purpose="confirmatory", actor="ci")
"""
    result = subprocess.run([sys.executable, "-c", script, frozen.to_json(), str(tmp_path), stage], check=False)
    assert result.returncode == 73
    audit = EvaluationAudit(tmp_path)
    receipt = audit.access(frozen, partition="test", purpose="confirmatory", actor="retry")
    assert receipt.prior_test_accesses == (1 if stage == "after" else 0)
    assert list(tmp_path.rglob("*.tmp"))


def test_corrupt_gap_future_history_and_fingerprint_mismatch_fail_closed(tmp_path):
    frozen = frozen_contract()
    audit = EvaluationAudit(tmp_path)
    audit.access(frozen, partition="test", purpose="confirmatory", actor="ci")
    second = audit.access(frozen, partition="test", purpose="confirmatory", actor="ci")
    path = next(tmp_path.rglob("000000000002.json"))
    payload = json.loads(path.read_text())
    payload["previous_fingerprint"] = None
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="chain"):
        audit.history(frozen.protocol.holdout_id)
    path.write_text(second.to_json())
    payload = json.loads(path.read_text())
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValidationError):
        audit.history(frozen.protocol.holdout_id)
    path.write_text(second.to_json())
    next(tmp_path.rglob("000000000001.json")).unlink()
    with pytest.raises(ValueError, match="chain"):
        audit.history(frozen.protocol.holdout_id)
    for digest in ("../../escape", "sha256:" + "g" * 64):
        with pytest.raises(ValueError):
            audit.load_freeze(digest)
    path = next((tmp_path / "evaluation-contracts").glob("*.json"))
    path.write_text(frozen_contract(practical_effect=1.0).to_json())
    with pytest.raises(ValueError, match="mismatch"):
        audit.load_freeze(frozen.fingerprint)


def test_import_boundary_no_transport_and_existing_native_run_id_unchanged():
    for path in (ROOT / "coleman" / "evaluation" / "protected").glob("*.py"):
        tree = ast.parse(path.read_text())
        names = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        names += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert not any(name and any(part in name.lower() for part in ("cli", "mcp", "runner")) for name in names)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import coleman.evaluation.protected; assert 'coleman.cli' not in sys.modules",
        ],
        check=True,
    )


def test_report_persistence_raw_references_and_manifest(tmp_path):
    from coleman.manifest import generate_manifest

    frozen = frozen_contract()
    audit = EvaluationAudit(tmp_path)
    rows = observations(frozen)
    report = evaluate(frozen, audit, lambda: rows, actor="ci")
    artifact = audit.save_report(report)
    assert audit.save_report(report) == artifact
    by_key = {row.record.logical_key: row.record for row in rows}
    assert {ref.record_key for ref in report.inputs} == set(by_key)
    assert all(ref.record_digest == by_key[ref.record_key].digest for ref in report.inputs)
    assert FamilyReport.from_json(artifact.read_bytes()) == report
    manifest = json.loads(generate_manifest(tmp_path, run_id="campaign").read_text())
    types = {item["type"] for item in manifest["artifacts"]}
    assert {"evaluation_contract", "evaluation_access", "evaluation_report"} <= types
    other_audit = EvaluationAudit(tmp_path / "unreserved")
    with pytest.raises(ValueError, match="not reserved"):
        other_audit.save_report(report)
    payload = json.loads(report.to_json())
    payload["access"]["evaluation_fingerprint"] = fingerprint([])
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError):
        FamilyReport.from_json(serialized)


def test_report_forged_reuse_claim_and_missing_input_references_rejected(tmp_path):
    frozen = frozen_contract()
    audit = EvaluationAudit(tmp_path)
    rows = observations(frozen)
    evaluate(frozen, audit, lambda: rows, actor="ci")
    reused = evaluate(frozen, audit, lambda: rows, actor="ci")
    payload = json.loads(reused.to_json())
    payload["results"][0]["conclusion"] = "evidence_of_benefit"
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError, match="confirmatory"):
        FamilyReport.from_json(serialized)
    payload = json.loads(reused.to_json())
    payload["inputs"].pop()
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError, match="unit counts"):
        FamilyReport.from_json(serialized)


def test_full_family_correction_and_intervals_include_all_predeclared_pairs(tmp_path):
    base = frozen_contract()
    third = FrozenCandidate.from_configuration("C", {"policy": "Conservative"})
    family = (*base.comparison_family, ComparisonHypothesis(hypothesis_id="A-C", candidate_id="A", baseline_id="C"))
    frozen = freeze((*base.candidates, third), base.data_fingerprint, base.plan, base.protocol, family)
    rows = list(observations(frozen))
    for row in tuple(rows):
        if row.candidate_id == "B":
            payload = row.model_dump(mode="json")
            payload["candidate_id"] = "C"
            payload["record"]["run_id"] = "run-C"
            payload["record"]["provenance"]["spec_fingerprint"] = third.configuration_fingerprint
            rows.append(PairedObservation.model_validate(payload))
    paired = tuple(rows)
    audit = EvaluationAudit(tmp_path)
    report = evaluate(frozen, audit, lambda: paired, actor="ci")
    raw_p = report.results[0].p_value
    assert raw_p is not None
    assert report.results[0].adjusted_p_value == pytest.approx(min(1.0, 2 * raw_p))
    expected = stats.ttest_rel(np.asarray([0.1, 0.2, 0.15, 0.25]) + 0.3, np.full(4, 0.3))
    interval = expected.confidence_interval(confidence_level=0.975)
    assert report.results[0].ci_lower == pytest.approx(interval.low)
    assert report.results[0].ci_upper == pytest.approx(interval.high)
    reordered = freeze(
        tuple(reversed(frozen.candidates)),
        frozen.data_fingerprint,
        frozen.plan,
        frozen.protocol,
        tuple(reversed(family)),
    )
    assert reordered.fingerprint == frozen.fingerprint
    assert FamilyReport.from_json(report.to_json()) == report
    payload = json.loads(report.to_json())
    for result in payload["results"]:
        result.update(p_value=0.03, adjusted_p_value=0.03)
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError, match="complete frozen Holm family"):
        FamilyReport.from_json(serialized)
    forged_results = tuple(
        result.model_copy(update={"p_value": 0.03, "adjusted_p_value": 0.03}) for result in report.results
    )
    unchecked_report = report.model_copy(update={"results": forged_results})
    with pytest.raises(ValidationError, match="complete frozen Holm family"):
        audit.save_report(unchecked_report)
    assert list(tmp_path.rglob("evaluation-reports/*.json")) == []
    for result in payload["results"]:
        result.update(adjusted_p_value=0.06, conclusion="inconclusive")
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError, match="statistics differ"):
        FamilyReport.from_json(serialized)


def test_metamorphic_unit_scale_translation_and_row_order(tmp_path):
    frozen = frozen_contract(metric_name="net_ci_credit", unit="credit", meaning="synthetic-net-ci-credit-v1")
    differences = (0.1, 0.2, 0.15, 0.25)
    first = evaluate(frozen, EvaluationAudit(tmp_path / "one"), lambda: observations(frozen, differences), actor="ci")
    scaled_differences = tuple(value * 10 for value in differences)
    second = evaluate(
        frozen,
        EvaluationAudit(tmp_path / "two"),
        lambda: tuple(reversed(observations(frozen, scaled_differences, baseline=100))),
        actor="ci",
    )
    assert first.results[0].raw_mean_difference is not None
    assert first.results[0].ci_lower is not None
    assert second.results[0].raw_mean_difference == pytest.approx(first.results[0].raw_mean_difference * 10)
    assert second.results[0].p_value == pytest.approx(first.results[0].p_value)
    assert second.results[0].ci_lower == pytest.approx(first.results[0].ci_lower * 10)
    assert second.results[0].independent_units == first.results[0].independent_units


def test_normal_null_and_known_effect_simulations_calibrate_inference(tmp_path):
    # Independent simulated experiments validate declared normal-difference
    # assumptions. These are synthetic credit quantities, not unbounded NAPFD.
    rng = np.random.default_rng(named_seed(42, "paired-t-validation", "simulation"))
    rejected = covered = detected = 0
    repetitions = 160
    for index in range(repetitions):
        null = tuple(float(value) for value in rng.normal(0, 1, 24))
        frozen = frozen_contract(
            metric_name="net_ci_credit",
            unit="credit",
            meaning="synthetic-credit-v1",
            holdout_id=f"independent-null-simulation-{index}",
            practical_effect=0.0,
        )
        payload = frozen.model_dump()
        payload["data_fingerprint"] = fingerprint({"differences": list(null), "baseline": 0.0})
        frozen = FrozenEvaluation.model_validate(payload)
        report = evaluate(
            frozen,
            EvaluationAudit(tmp_path),
            lambda null=null, frozen=frozen: observations(frozen, null, baseline=0),
            actor="simulation",
        )
        result = report.results[0]
        rejected += result.conclusion in {"evidence_of_benefit", "evidence_of_harm"}
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        covered += result.ci_lower <= 0 <= result.ci_upper
        known = tuple(value + 1.0 for value in null)
        protocol = {**frozen.protocol.model_dump(), "holdout_id": f"independent-known-simulation-{index}"}
        alternative = freeze(
            frozen.candidates,
            fingerprint({"differences": list(known), "baseline": 0.0}),
            frozen.plan,
            EvaluationProtocol.model_validate(protocol),
            frozen.comparison_family,
        )
        report = evaluate(
            alternative,
            EvaluationAudit(tmp_path),
            lambda known=known, alternative=alternative: observations(alternative, known, baseline=0),
            actor="simulation",
        )
        detected += report.results[0].conclusion == "evidence_of_benefit"
    assert rejected / repetitions < 0.12
    assert covered / repetitions > 0.88
    assert detected / repetitions > 0.9


def test_installed_package_native_tcpci_protected_workflow(tmp_path):
    import shutil

    wheel_dir = tmp_path / "wheels"
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(wheel_dir)], cwd=ROOT, check=True)
    wheel = next(wheel_dir.glob("*.whl"))
    installed = tmp_path / "installed"
    subprocess.run(["uv", "venv", str(installed), "--python", sys.executable], check=True)
    python = installed / "bin" / "python"
    subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel) + "[ci,analytics]"], check=True)
    external = tmp_path / "external"
    external.mkdir()
    script = external / "ci.py"
    shutil.copy(ROOT / "examples" / "protected-evaluation" / "ci.py", script)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [str(python), str(script), str(external / "output")],
        cwd=external,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    output = json.loads(completed.stdout.splitlines()[-1])
    assert output["result"] == "insufficient_evidence"
    assert output["independent_units"] == 4
    assert output["repeat_is_confirmatory"] is False
    report = FamilyReport.from_json(Path(output["report"]).read_bytes())
    assert len(report.inputs) == 8
    manifest = json.loads(Path(output["manifest"]).read_text())
    access_entries = [entry for entry in manifest["artifacts"] if entry["type"] == "evaluation_access"]
    assert len(access_entries) == 2
    assert report.results[0].ci_lower is None


def test_versioned_golden_fixtures_roundtrip_identity_and_named_seed():
    expected = json.loads((FIXTURES / "digests-v1.json").read_text())
    for name, model in (("freeze", FrozenEvaluation), ("access", AccessReceipt), ("report", FamilyReport)):
        content = (FIXTURES / f"{name}-v1.json").read_text()
        restored = model.from_json(content)
        assert restored.to_json() + "\n" == content
        assert restored.fingerprint == expected[name]
    assert named_seed(42, "A", "evaluation") == expected["named_seed_42_A_evaluation"]
    assert 0 <= named_seed(42, "A", "evaluation") <= 2**64 - 1


@pytest.mark.parametrize("name,model", [("access", AccessReceipt), ("report", FamilyReport)])
def test_unknown_and_missing_versions_never_migrated(name, model):
    payload = json.loads((FIXTURES / f"{name}-v1.json").read_text())
    for version in (None, 0, 2, True):
        payload["schema_version"] = version
        serialized = json.dumps(payload)
        with pytest.raises(ValidationError):
            model.from_json(serialized)
    del payload["schema_version"]
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError):
        model.from_json(serialized)


def test_validation_rejects_forged_report_interval_and_claims():
    original = json.loads((FIXTURES / "report-v1.json").read_text())
    mutations = (
        {"observed_pairs": 1000},
        {"ci_lower": 1000.0},
        {"ci_upper": None},
        {"adjusted_p_value": 0.9},
        {"ci_lower": 0.0},
        {"conclusion": "evidence_of_harm"},
        {"independent_units": ["project-0000"] * 4},
    )
    for mutation in mutations:
        payload = json.loads(json.dumps(original))
        payload["results"][0].update(mutation)
        serialized = json.dumps(payload)
        with pytest.raises(ValidationError):
            FamilyReport.from_json(serialized)
    payload = json.loads(json.dumps(original))
    payload["inputs"].append(payload["inputs"][0])
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError):
        FamilyReport.from_json(serialized)


def test_extreme_finite_inputs_and_unstable_uncertainty_are_explicit(tmp_path):
    frozen = frozen_contract(metric_name="net_ci_credit", unit="credit", meaning="synthetic-credit-v1", alpha=1e-300)
    rows = observations(frozen, differences=(1.0, 2.0, 3.0), baseline=0.0)
    report = evaluate(frozen, EvaluationAudit(tmp_path / "uncertainty"), lambda: rows, actor="ci")
    assert report.results[0].raw_mean_difference == pytest.approx(2.0)
    assert report.results[0].reason == "numerically_unstable_uncertainty"
    assert report.results[0].ci_lower is None
    rows = list(observations(frozen, differences=(0.0, 0.0), baseline=0.0))
    first, second = rows[0].model_dump(), rows[1].model_dump()
    first["record"]["measurements"][0]["value"] = 1e308
    second["record"]["measurements"][0]["value"] = -1e308
    rows[0], rows[1] = PairedObservation.model_validate(first), PairedObservation.model_validate(second)
    extreme = tuple(rows)
    overflow_audit = EvaluationAudit(tmp_path / "overflow")
    with pytest.raises(ValueError, match="finite numeric range"):
        evaluate(frozen, overflow_audit, lambda: extreme, actor="ci")


def test_shared_domain_compatibility_preserves_unknown_values_and_strict_comparison():
    frozen = frozen_contract()
    rows = observations(frozen, differences=(0.1,), unavailable_index=0)
    unknown = rows[0].record
    observed = rows[1].record
    unknown.require_compatible(observed, frozen.protocol.metric_name)
    with pytest.raises(ValueError, match="observed"):
        unknown.require_comparable(observed, frozen.protocol.metric_name)
    payload = observed.model_dump(mode="json")
    payload["measurements"][0]["unit"] = "ms"
    incompatible = DomainRecord.model_validate(payload)
    with pytest.raises(ValueError, match="Incompatible"):
        unknown.require_compatible(incompatible, frozen.protocol.metric_name)


@pytest.mark.parametrize(
    "points",
    [
        {"raw_mean_difference": None, "improvement": None},
        {"raw_mean_difference": -0.175, "improvement": -0.175},
        {"raw_mean_difference": 0.175, "improvement": -0.175},
        {"raw_mean_difference": 100.0, "improvement": 100.0},
    ],
)
def test_confirmatory_report_requires_coherent_point_estimates_before_persistence(tmp_path, points):
    frozen = frozen_contract()
    audit = EvaluationAudit(tmp_path)
    rows = observations(frozen)
    original = evaluate(frozen, audit, lambda: rows, actor="ci")
    payload = json.loads(original.to_json())
    payload["results"][0].update(points)
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError):
        FamilyReport.from_json(serialized)
    unchecked_result = original.results[0].model_copy(update=points)
    unchecked_report = original.model_copy(update={"results": (unchecked_result,)})
    with pytest.raises(ValidationError):
        audit.save_report(unchecked_report)
    assert list(tmp_path.rglob("evaluation-reports/*.json")) == []


def test_relabeling_one_raw_record_cannot_create_independent_units(tmp_path):
    frozen = frozen_contract()
    original = observations(frozen)
    rows = list(original)
    for row in original[:2]:
        payload = row.model_dump(mode="json")
        payload["metadata"].update(group_id="duplicate-group", observation_id="duplicate-build")
        rows.append(PairedObservation.model_validate(payload))
    duplicated = tuple(rows)
    audit = EvaluationAudit(tmp_path / "duplicate-evaluation")
    with pytest.raises(ValueError, match="Repeated raw result identity"):
        evaluate(frozen, audit, lambda: duplicated, actor="ci")
    assert len(audit.history(frozen.protocol.holdout_id)) == 1
    report_audit = EvaluationAudit(tmp_path / "duplicate-report")
    report = evaluate(frozen, report_audit, lambda: original, actor="ci")
    payload = json.loads(report.to_json())
    candidate_inputs = [item for item in payload["inputs"] if item["candidate_id"] == "A"]
    for field in ("record_key", "record_digest", "record"):
        candidate_inputs[1][field] = candidate_inputs[0][field]
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError, match="Repeated raw result identity"):
        FamilyReport.from_json(serialized)
    unchecked_inputs = list(report.inputs)
    first, second = [index for index, item in enumerate(unchecked_inputs) if item.candidate_id == "A"][:2]
    unchecked_inputs[second] = unchecked_inputs[second].model_copy(
        update={
            "record_key": unchecked_inputs[first].record_key,
            "record_digest": unchecked_inputs[first].record_digest,
            "record": unchecked_inputs[first].record,
        }
    )
    unchecked_report = report.model_copy(update={"inputs": tuple(unchecked_inputs)})
    with pytest.raises(ValidationError, match="Repeated raw result identity"):
        report_audit.save_report(unchecked_report)
    assert list(tmp_path.rglob("evaluation-reports/*.json")) == []


def test_claimed_statistics_are_recomputed_from_digest_checked_raw_records(tmp_path):
    frozen = frozen_contract()
    audit = EvaluationAudit(tmp_path)
    rows = observations(frozen)
    report = evaluate(frozen, audit, lambda: rows, actor="ci")
    forgery = dict(
        raw_mean_difference=100.0,
        improvement=100.0,
        ci_lower=99.0,
        ci_upper=101.0,
        p_value=0.0,
        adjusted_p_value=0.0,
        conclusion="evidence_of_benefit",
    )
    payload = json.loads(report.to_json())
    payload["results"][0].update(forgery)
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError, match="statistics differ"):
        FamilyReport.from_json(serialized)
    unchecked = report.model_copy(update={"results": (report.results[0].model_copy(update=forgery),)})
    with pytest.raises(ValidationError, match="statistics differ"):
        audit.save_report(unchecked)
    assert list(tmp_path.rglob("evaluation-reports/*.json")) == []
    payload = json.loads(report.to_json())
    payload["inputs"][0]["record"]["measurements"][0]["value"] = 100.0
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError, match="referenced identity or digest"):
        FamilyReport.from_json(serialized)


def test_derived_availability_is_completion_after_the_loader_and_inference(tmp_path):
    frozen = frozen_contract()
    loaded_at = None

    def load():
        nonlocal loaded_at
        rows = observations(frozen)
        loaded_at = datetime.now(UTC)
        return rows

    report = evaluate(frozen, EvaluationAudit(tmp_path), load, actor="ci")
    assert loaded_at is not None
    assert report.completed_at >= loaded_at > report.access.requested_at
    assert all(record.available_at == report.completed_at for record in report.domain_records("ci"))
    payload = json.loads(report.to_json())
    payload["completed_at"] = "2026-01-01T00:00:00Z"
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError, match="completion predates"):
        FamilyReport.from_json(serialized)
    payload["access"]["requested_at"] = "2026-01-04T00:00:00Z"
    payload["completed_at"] = "2026-01-04T12:00:00Z"
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError, match="completion predates"):
        FamilyReport.from_json(serialized)


def test_raw_logical_identity_cannot_collide_between_candidates(tmp_path):
    frozen = frozen_contract()
    rows = observations(frozen)
    colliding = tuple(
        row.model_copy(update={"record": row.record.model_copy(update={"run_id": "run-A"})})
        if row.candidate_id == "B"
        else row
        for row in rows
    )
    audit = EvaluationAudit(tmp_path)
    with pytest.raises(ValueError, match="Repeated raw result identity"):
        evaluate(frozen, audit, lambda: colliding, actor="ci")
    valid = evaluate(frozen, EvaluationAudit(tmp_path / "valid"), lambda: rows, actor="ci")
    inputs = []
    for item in valid.inputs:
        if item.candidate_id == "B":
            record = item.record.model_copy(update={"run_id": "run-A"})
            item = item.model_copy(
                update={"record": record, "record_key": record.logical_key, "record_digest": record.digest}
            )
        inputs.append(item)
    unchecked = valid.model_copy(update={"inputs": tuple(inputs)})
    serialized = json.dumps(unchecked.model_dump(mode="json"))
    with pytest.raises(ValidationError, match="Repeated raw result identity"):
        FamilyReport.from_json(serialized)
    with pytest.raises(ValidationError, match="Repeated raw result identity"):
        audit.save_report(unchecked)
    assert list(tmp_path.rglob("evaluation-reports/*.json")) == []
