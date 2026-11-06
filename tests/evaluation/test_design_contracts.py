"""Independent wire, chronology, membership, budgets and identity admission cases."""

import copy
import hashlib
import json
import math
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from coleman.core.primitives import canonical_json
from coleman.evaluation.design._wire import DesignContract, guard_graph
from coleman.evaluation.design.contracts import (
    CandidatePlan,
    ExperimentDesign,
    InferenceQualification,
    MetricPlan,
    Threshold,
)
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.evaluation.design.limits import DesignCounts, ProtocolLimits
from coleman.evaluation.design.selection import (
    AttemptLedger,
    CellIdentity,
    ConfigurationTrial,
    PhysicalAttempt,
    SelectionManifest,
)
from coleman.evaluation.design.source_contracts import (
    SourcePartitionManifest,
    SourcePrefix,
    SourceReceipt,
    SourceRequest,
    SourceScope,
)
from coleman.evaluation.protected.contracts import FrozenCandidate
from coleman.results.domain import DomainRecord, LearningReward, Measurement, ResultProvenance

FIXTURES = Path(__file__).parent / "fixtures/design-v1"
GOLDENS = (
    ("design", ExperimentDesign),
    ("selection", SelectionManifest),
    ("partition", SourcePartitionManifest),
    ("prefix", SourcePrefix),
)


def raw_design():
    return json.loads((FIXTURES / "design.json").read_text())


def original_record(value=0.0, status="observed"):
    return DomainRecord(
        schema_version=1,
        namespace="authored.ci",
        run_id="original-run",
        execution_id="original-execution",
        decision_id="original-build",
        data_fingerprint="sha256:" + "1" * 64,
        available_at=datetime(2026, 11, 6, 12, tzinfo=UTC),
        provenance=ResultProvenance(
            producer="authored-producer",
            producer_version="1",
            spec_fingerprint="sha256:" + "2" * 64,
            evaluation_fingerprint="sha256:" + "3" * 64,
        ),
        measurements=(
            Measurement(
                name="quality",
                value=value,
                unit="1",
                meaning="authored-quality-v1",
                direction="maximize",
                status=status,
            ),
        ),
    )


@pytest.mark.parametrize("name,contract", GOLDENS)
def test_hand_authored_v1_roots_have_independent_wire_and_digest_goldens(name, contract):
    content = (FIXTURES / (name + ".json")).read_bytes()
    expected = json.loads((FIXTURES / "manifest.json").read_text())["files"][name + ".json"]
    assert hashlib.sha256(content).hexdigest() == expected["file_sha256"]
    decoded = contract.from_json(content)
    assert (decoded.to_json() + "\n").encode() == content
    assert decoded.fingerprint == expected["model_fingerprint"]
    assert contract.model_validate_json(content) == decoded


@pytest.mark.parametrize("name,contract", GOLDENS)
@pytest.mark.parametrize("version", [None, True, "1", 0, 2, 999])
def test_all_new_root_readers_reject_missing_coerced_old_and_future_versions(name, contract, version):
    data = json.loads((FIXTURES / (name + ".json")).read_text())
    if version is None:
        data.pop("schema_version")
    else:
        data["schema_version"] = version
    content = canonical_json(data)
    with pytest.raises(ValueError):
        contract.from_json(content)
    with pytest.raises(ValueError):
        contract.model_validate_json(content)


@pytest.mark.parametrize("extra", ["ignore", "allow"])
def test_new_reader_cannot_relax_unknown_fields_or_strict_versions(extra):
    content = (FIXTURES / "design.json").read_bytes()
    with pytest.raises(ValueError, match="forbid"):
        ExperimentDesign.model_validate_json(content, extra=extra)
    with pytest.raises(ValueError, match="strict"):
        ExperimentDesign.model_validate_json(content, strict=False)
    duplicate = content.decode().replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1)
    with pytest.raises(ValueError, match="Duplicate"):
        ExperimentDesign.from_json(duplicate)


def test_set_permutations_preserve_design_identity_but_actual_horizon_changes_it():
    data = raw_design()
    before = ExperimentDesign.model_validate(data)
    data["members"].reverse()
    data["candidates"].reverse()
    after = ExperimentDesign.model_validate(data)
    assert before.fingerprint == after.fingerprint
    data["metrics"][0]["quantity"]["horizon"]["version"] = "2"
    assert ExperimentDesign.model_validate(data).fingerprint != before.fingerprint


@pytest.mark.parametrize("field", ["train_end", "validation_end", "event_field", "availability_field", "group_field"])
def test_only_comparison_test_deadline_may_change(field):
    data = raw_design()
    data["comparison_plan"][field] = "2018-06-01T00:00:00Z" if field.endswith("end") else "different_source_column"
    with pytest.raises(ValueError):
        ExperimentDesign.model_validate(data)


def test_wider_comparison_window_does_not_admit_a_later_source_member():
    data = raw_design()
    later = copy.deepcopy(data["members"][0])
    later.update(member_id="later-build", path={"id": "later-path", "version": "1"})
    later["metadata"].update(
        observation_id="later-observation",
        group_id="later-group",
        event_at="2020-01-01T00:00:00Z",
        available_at="2020-01-02T00:00:00Z",
    )
    data["members"].append(later)
    design = ExperimentDesign.model_validate(data)
    assert design.comparison_plan.decide(design.members[-1].metadata).eligible
    assert "later-build" not in {member.member_id for member in design.protected_members}
    assert design.raw_cells == 4


def test_excluded_late_training_row_still_cannot_share_a_protected_leakage_group():
    data = raw_design()
    training = copy.deepcopy(data["members"][0])
    training.update(member_id="late-training", unit_id=None)
    training["metadata"].update(
        observation_id="late-training-observation",
        event_at="2017-01-01T00:00:00Z",
        available_at="2020-01-01T00:00:00Z",
    )
    data["members"].append(training)
    with pytest.raises(ValueError, match="group spans"):
        ExperimentDesign.model_validate(data)


def test_task_or_version_changes_cannot_split_one_globally_scoped_path_into_units():
    data = raw_design()
    data["members"][1]["path"] = {**data["members"][0]["path"], "version": "2"}
    data["members"][1]["task"] = {"id": "renamed-task", "version": "2"}
    with pytest.raises(ValueError, match="logical path"):
        ExperimentDesign.model_validate(data)


def test_protected_member_requires_unit_and_duplicate_source_revision_cannot_make_new_member():
    data = raw_design()
    data["members"][0]["unit_id"] = None
    with pytest.raises(ValueError, match="unit_id"):
        ExperimentDesign.model_validate(data)
    data = raw_design()
    duplicate = copy.deepcopy(data["members"][0])
    duplicate["member_id"] = "different-label"
    duplicate["metadata"]["revision_id"] = "corrected"
    data["members"].append(duplicate)
    with pytest.raises(ValueError, match="selected revision"):
        ExperimentDesign.model_validate(data)


@pytest.mark.parametrize("name", ["fitness", "cost", "rewards", "process_cpu_time_seconds"])
def test_native_columns_cannot_impersonate_new_generic_metric_names(name):
    metric = raw_design()["metrics"][0]
    metric["name"] = name
    with pytest.raises(ValueError, match="reserved"):
        MetricPlan.model_validate(metric)


@pytest.mark.parametrize("unit,meaning", [("USD", "authored-ci-quality-v1"), ("1", "different-quantity-v1")])
def test_practical_effect_requires_exact_raw_metric_units_and_meaning(unit, meaning):
    data = raw_design()
    data["inference"]["practical_effect"]["semantics"] = {"unit": unit, "meaning": meaning}
    with pytest.raises(ValueError, match="threshold units"):
        ExperimentDesign.model_validate(data)


@pytest.mark.parametrize("invalid", [True, "0.1", math.inf, math.nan, -0.1])
def test_threshold_rejects_nonfinite_coerced_and_negative_values(invalid):
    with pytest.raises(ValueError):
        Threshold.model_validate({"value": invalid, "semantics": {"unit": "1", "meaning": "authored-quality-v1"}})


def test_coupled_byte_budget_uses_actual_grid_and_retains_required_unused_margin():
    defaults = ProtocolLimits(schema_version=1)
    assert defaults.reserve(128, 128) == 1_966_080
    broad = ProtocolLimits(
        schema_version=1,
        max_cells=8192,
        max_aggregate_inputs=4096,
        max_raw_record_bytes=4096,
        max_metadata_bytes=8 * 1024**2,
        max_source_receipts=512,
        max_audit_receipts=512,
        max_directory_entries=576,
        max_directory_bytes=2 * 1024**2,
        result_reserve_bytes=2 * 1024**2,
        admission_margin_bytes=1024**2,
        max_envelope_bytes=64 * 1024**2,
    )
    with pytest.raises(ValueError, match="unused margin"):
        broad.reserve(8192, 4096)
    assert broad.reserve(7987, 4096) + broad.admission_margin_bytes <= broad.max_envelope_bytes
    with pytest.raises(ValueError, match="capacity"):
        defaults.require_counts(DesignCounts(candidates=3, protected_members=30, seeds=20, units=30))


def test_raised_directory_receipt_count_cannot_create_an_unreadable_full_prefix_profile():
    with pytest.raises(ValueError, match="headroom"):
        ProtocolLimits(schema_version=1, max_source_receipts=512)
    with pytest.raises(ValueError, match="complete declared receipt prefix"):
        ProtocolLimits(schema_version=1, max_directory_bytes=100)
    with pytest.raises(ValueError, match="publication headroom"):
        ProtocolLimits(schema_version=1, max_directory_bytes=128 * 2048)
    with pytest.raises(ValueError, match="one MiB"):
        ProtocolLimits(schema_version=1, max_envelope_bytes=64 * 1024**2)


@pytest.mark.parametrize("bad", [True, -1, 1.0, "1"])
def test_reservation_counts_are_strict_nonnegative_integers(bad):
    limits = ProtocolLimits(schema_version=1)
    with pytest.raises(ValueError, match="strict bounded"):
        limits.reserve(bad, 1)


def test_graph_preflight_counts_shared_aliases_and_never_calls_model_serializer():
    class Trap(BaseModel):
        values: list[int]

        def model_dump(self, *args, **kwargs):
            raise AssertionError("Unchecked serializer must not execute")

    value = Trap(values=[1, 2, 3])
    guard_graph(value, maximum_bytes=100, maximum_items=20)
    alias = [1, 2, 3]
    guard_graph([alias, alias], maximum_bytes=100, maximum_items=9)
    with pytest.raises(ValueError, match="item budget"):
        guard_graph([alias, alias], maximum_bytes=100, maximum_items=8)


def test_graph_preflight_utf8_escaping_cycles_and_depth_are_bounded_before_parsing():
    text = "é\\\n𝄞"
    encoded = len(canonical_json(text).encode())
    guard_graph(text, maximum_bytes=encoded, maximum_items=1)
    with pytest.raises(ValueError, match="byte budget"):
        guard_graph(text, maximum_bytes=encoded - 1, maximum_items=1)
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="cyclic"):
        guard_graph(cycle, maximum_bytes=100, maximum_items=100)
    nested = 0
    for _ in range(64):
        nested = [nested]
    guard_graph(nested, maximum_bytes=200, maximum_items=100)
    with pytest.raises(ValueError, match="depth"):
        guard_graph([nested], maximum_bytes=200, maximum_items=100)


@pytest.mark.parametrize(
    "status,value",
    [
        ("observed", 0.0),
        ("observed", -0.0),
        ("observed", -0.5),
        ("unknown", None),
        ("unavailable", None),
        ("not_applicable", None),
    ],
)
def test_original_wire_enclosure_preserves_raw_statuses_and_omitted_optional_digests(status, value):
    class Envelope(DesignContract):
        evidence: OriginalRecordEvidence

    source = original_record(value, status)
    before = source.to_json()
    envelope = Envelope(schema_version=1, evidence=OriginalRecordEvidence.from_record(source))
    restored = Envelope.from_json(envelope.to_json())
    assert restored.evidence.record.to_json() == before
    assert restored.evidence.record_digest == source.digest
    assert '"learning_reward"' not in before and '"minimum"' not in before
    assert restored.evidence.record.measurement("quality").status == status


@pytest.mark.parametrize("token", ["1e-400", "9007199254740993", "true", "NaN", "Infinity"])
def test_new_original_evidence_guard_rejects_lossy_numeric_tokens_before_legacy_float_coercion(token):
    record = original_record()
    evidence = OriginalRecordEvidence.from_record(record)
    wire = canonical_json(evidence.model_dump(mode="json")).replace('"value":0.0', '"value":' + token)
    with pytest.raises(ValueError):
        OriginalRecordEvidence.model_validate_json(wire)


@pytest.mark.parametrize("route", ["nested_typed_record", "unchecked_typed_evidence", "from_record"])
@pytest.mark.parametrize("field", ["value", "minimum", "learning_reward"])
@pytest.mark.parametrize("unsafe", [Decimal("1e-400"), 9007199254740993])
def test_new_evidence_checks_unparsed_typed_values_before_old_float_reader(route, field, unsafe):
    clean = original_record(float(unsafe) if field == "value" else 1e20)
    if field == "learning_reward":
        reward = LearningReward(value=float(unsafe), unit="1", transform_id="authored-identity-v1")
        clean = DomainRecord.model_validate(clean.model_copy(update={"learning_reward": reward}))
        unchecked = clean.model_copy(update={"learning_reward": reward.model_copy(update={"value": unsafe})})
    else:
        if field == "minimum":
            measurement = clean.measurements[0].model_copy(update={"minimum": float(unsafe)})
            clean = DomainRecord.model_validate(clean.model_copy(update={"measurements": (measurement,)}))
        unchecked = clean.model_copy(
            update={"measurements": (clean.measurements[0].model_copy(update={field: unsafe}),)}
        )
    supplied: dict[str, object] | OriginalRecordEvidence = {
        "record_key": clean.logical_key,
        "record_digest": clean.digest,
        "record": unchecked,
    }
    if route == "unchecked_typed_evidence":
        supplied = OriginalRecordEvidence.model_construct(
            record_key=clean.logical_key, record_digest=clean.digest, record=unchecked
        )
    with pytest.raises(ValueError):
        if route == "from_record":
            OriginalRecordEvidence.from_record(unchecked)
        else:
            OriginalRecordEvidence.model_validate(supplied)


def test_selection_preserves_complete_final_family_and_requires_development_evidence():
    design = ExperimentDesign.from_json((FIXTURES / "design.json").read_bytes())
    selection = SelectionManifest.from_json((FIXTURES / "selection.json").read_bytes())
    selection.require_design(design)
    corrupted = selection.model_copy(update={"selected_candidate_ids": ("baseline", "unfrozen")})
    with pytest.raises(ValueError, match="complete final"):
        corrupted.require_design(design)
    protected = AttemptLedger(schema_version=1, stage="protected")
    data = selection.model_dump(mode="json")
    data["ledger"] = protected.model_dump(mode="json")
    with pytest.raises(ValueError, match="development evidence"):
        SelectionManifest.model_validate(data)


def test_source_requests_without_known_manifests_remain_potential_protected_exposure():
    request = json.loads((FIXTURES / "prefix.json").read_text())["receipts"][0]["request"]
    request.update(purpose="inspection", partitions=["train"], partition_manifest_fingerprint=None)
    assert SourceRequest.model_validate(request).potential_test_exposure
    request["partitions"] = ["unknown", "train"]
    with pytest.raises(ValueError, match="unknown must stand alone"):
        SourceRequest.model_validate(request)


def test_source_prefix_keeps_ordinal_order_and_rejects_gaps_and_forged_predecessors():
    raw = json.loads((FIXTURES / "prefix.json").read_text())
    first = SourcePrefix.model_validate(raw).receipts[0]
    second = copy.deepcopy(raw["receipts"][0])
    second.update(
        sequence=2,
        access_id="actual-other-attempt",
        previous_fingerprint=first.fingerprint,
        requested_at="2026-11-06T12:01:59Z",
    )
    raw["receipts"].append(second)
    prefix = SourcePrefix.model_validate(raw)
    assert prefix.receipts[-1].requested_at < prefix.receipts[0].requested_at
    raw["receipts"][-1]["sequence"] = 3
    with pytest.raises(ValueError, match="source-access prefix"):
        SourcePrefix.model_validate(raw)


def test_inference_support_is_not_a_fabricated_protocol_or_scientific_claim():
    result = InferenceQualification(schema_version=1, status="unsupported", protocol=None, reasons=("dependent_units",))
    assert result.protocol is None
    with pytest.raises(ValueError, match="support requires"):
        InferenceQualification(schema_version=1, status="supported", protocol=None, reasons=())


def test_physical_attempt_has_many_cell_outputs_without_inventing_independent_jobs():
    attempt = PhysicalAttempt(
        attempt_id="actual-process-1",
        group_id="one-native-policy-seed",
        trial_id=None,
        candidate_id="candidate",
        state="completed",
        started_at=None,
        completed_at=None,
        outputs=(
            CellIdentity(candidate_id="candidate", member_id="build-a", seed=7),
            CellIdentity(candidate_id="candidate", member_id="build-b", seed=7),
        ),
    )
    ledger = AttemptLedger(schema_version=1, stage="protected", attempts=(attempt,))
    assert len(ledger.attempts) == 1 and ledger.output_references == 2
    assert ledger.attempts[0].costs == ()


def test_same_upstream_scope_versions_and_prehistory_never_rotate_the_logical_holdout():
    raw = raw_design()["scope"]
    before = SourceScope.model_validate(raw)
    raw["reference"]["version"] = "2"
    raw["prehistory"] = "known_exposed"
    after = SourceScope.model_validate(raw)
    assert before.logical_key == after.logical_key
    assert before.holdout_id == after.holdout_id
    assert before.fingerprint != after.fingerprint


def test_unexposed_is_a_host_declaration_and_empirical_basis_requires_a_separate_reference():
    raw = raw_design()
    raw["evidence_basis"] = "empirical_protected"
    with pytest.raises(ValueError, match="independent source assessment"):
        ExperimentDesign.model_validate(raw)
    raw["source_admission_fingerprint"] = "sha256:" + "4" * 64
    declared = ExperimentDesign.model_validate(raw)
    assert declared.evidence_basis == "empirical_protected"
    assert declared.scope.prehistory == "declared_unexposed"
    # The pure DTO retains two declarations; it performs no authentication/read.
    raw["scope"]["prehistory"] = "qualified_unexposed"
    with pytest.raises(ValueError):
        ExperimentDesign.model_validate(raw)


@pytest.mark.parametrize("invalid", [None, 1, "true"])
def test_manifest_verification_at_reservation_is_required_and_strict(invalid):
    receipt = json.loads((FIXTURES / "prefix.json").read_text())["receipts"][0]
    if invalid is None:
        receipt.pop("partition_manifest_verified")
    else:
        receipt["partition_manifest_verified"] = invalid
    with pytest.raises(ValueError):
        SourceReceipt.model_validate(receipt)


def test_later_manifest_binding_cannot_rewrite_an_old_unverified_receipt():
    receipt = json.loads((FIXTURES / "prefix.json").read_text())["receipts"][0]
    receipt["request"].update(purpose="inspection", partitions=["validation"], partition_manifest_fingerprint=None)
    old = SourceReceipt.model_validate(receipt)
    assert old.partition_manifest_verified is False
    receipt["request"]["partition_manifest_fingerprint"] = "sha256:" + "5" * 64
    supplied_later = SourceReceipt.model_validate(receipt)
    assert not supplied_later.partition_manifest_verified
    assert supplied_later.fingerprint != old.fingerprint
    receipt["partition_manifest_verified"] = True
    verified = SourceReceipt.model_validate(receipt)
    assert verified.fingerprint != supplied_later.fingerprint
    receipt["request"]["partitions"] = ["test"]
    with pytest.raises(ValueError, match="potentially narrow"):
        SourceReceipt.model_validate(receipt)


def test_nominal_protected_member_needs_unit_even_when_late_availability_excludes_it():
    raw = raw_design()
    raw["members"][0]["unit_id"] = None
    raw["members"][0]["metadata"]["available_at"] = "2020-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="unit_id"):
        ExperimentDesign.model_validate(raw)


def test_new_wire_cap_includes_one_framing_lf_and_counts_utf8_not_characters():
    class Tiny(DesignContract):
        _maximum_bytes: ClassVar[int] = 64
        label: str

    empty = canonical_json({"schema_version": 1, "label": ""})
    count = 63 - len(empty.encode())
    raw = canonical_json({"schema_version": 1, "label": "a" * count})
    value = Tiny.from_json(raw + "\n")
    assert len(value.to_json().encode()) + 1 == 64
    with pytest.raises(ValueError):
        Tiny.from_json(raw + "\n ")
    with pytest.raises(ValueError):
        Tiny(schema_version=1, label="a" * (count + 1))
    with pytest.raises(ValueError):
        Tiny(schema_version=1, label="é" * count)


def test_graph_preflight_rejects_a_cycle_in_unchecked_model_storage():
    class Node(BaseModel):
        child: object = None

    value = Node()
    value.__dict__["child"] = value
    with pytest.raises(ValueError, match="cyclic model"):
        guard_graph(value, maximum_bytes=100, maximum_items=100)


def test_trial_outcome_must_bind_the_inspected_base_configuration():
    design = ExperimentDesign.from_json((FIXTURES / "design.json").read_bytes())
    raw = {
        "trial_id": "development-trial",
        "candidate": design.candidates[0],
        "status": "completed",
        "partition": "validation",
        "completed_at": None,
        "results": (OriginalRecordEvidence.from_record(original_record()),),
    }
    with pytest.raises(ValueError, match="actual base candidate"):
        ConfigurationTrial.model_validate(raw)


def test_selection_cannot_include_future_physical_cost_or_start_and_duplicate_cost_is_not_paid_twice():
    selection = SelectionManifest.from_json((FIXTURES / "selection.json").read_bytes())
    design = ExperimentDesign.from_json((FIXTURES / "design.json").read_bytes())
    trial = ConfigurationTrial(
        trial_id="physical-trial",
        candidate=design.candidates[0],
        status="declared",
        partition="validation",
        completed_at=None,
    )
    cost = original_record().model_copy(update={"available_at": datetime(2026, 11, 6, 12, 5, tzinfo=UTC)})
    attempt = PhysicalAttempt(
        attempt_id="measured-process",
        group_id="measured-group",
        trial_id=trial.trial_id,
        candidate_id=trial.candidate.candidate.candidate_id,
        state="failed",
        started_at=None,
        completed_at=None,
        costs=(OriginalRecordEvidence.from_record(cost),),
    )
    ledger = AttemptLedger(schema_version=1, stage="development", trials=(trial,), attempts=(attempt,))
    with pytest.raises(ValueError, match="actual development"):
        SelectionManifest.model_validate(selection.model_copy(update={"ledger": ledger}))
    other = attempt.model_copy(update={"attempt_id": "different-physical-retry"})
    with pytest.raises(ValueError, match="several attempts"):
        AttemptLedger(schema_version=1, stage="development", trials=(trial,), attempts=(attempt, other))


@pytest.mark.parametrize("route", ["nested_typed_legacy", "unchecked_typed_plan"])
def test_unchecked_typed_base_configuration_is_depth_guarded_before_legacy_parser(route):
    design = ExperimentDesign.from_json((FIXTURES / "design.json").read_bytes())
    original = design.candidates[0]
    unsupported = '{"nested":' + "[" * 1200 + "0" + "]" * 1200 + "}"
    legacy = FrozenCandidate.model_construct(candidate_id="baseline", configuration_json=unsupported)
    values = {"candidate": legacy, "role": original.role, "strategy": original.strategy}
    supplied = (
        values
        if route == "nested_typed_legacy"
        else CandidatePlan.model_construct(candidate=legacy, role=original.role, strategy=original.strategy)
    )
    with pytest.raises(ValueError, match="depth"):
        CandidatePlan.model_validate(supplied)
