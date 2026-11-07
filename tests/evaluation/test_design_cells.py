"""Cell identity, original-wire preservation and new numeric ingress guards."""

from datetime import UTC, datetime

import pytest

from coleman.core.primitives import canonical_json
from coleman.evaluation._repetition import repetition_execution_id
from coleman.evaluation.design.cells import RawCell
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance


def producer_record(*, member="member-a", candidate="baseline", seed=11, value=0.5, **extra):
    return DomainRecord(
        schema_version=1,
        namespace="constructed.design.raw",
        run_id="authored-record-fixture",
        execution_id=repetition_execution_id(candidate, seed),
        decision_id=member,
        data_fingerprint="sha256:" + "1" * 64,
        available_at=datetime(2026, 11, 6, tzinfo=UTC),
        provenance=ResultProvenance(
            producer="authored-grid-fixture",
            producer_version="1",
            spec_fingerprint="sha256:" + "2" * 64,
            evaluation_fingerprint="sha256:" + "3" * 64,
            **extra,
        ),
        measurements=(
            Measurement(
                name="quality",
                value=value,
                unit="1",
                meaning="authored-quality-v1",
                direction="maximize",
                status="observed",
            ),
        ),
    )


def cell_for(record, **extra):
    return RawCell(
        candidate_id="baseline",
        member_id=record.decision_id,
        seed=11,
        state="completed",
        evidence=OriginalRecordEvidence.from_record(record),
        **extra,
    )


def test_raw_cell_round_trip_retains_original_record_optional_omission_and_digest():
    original = producer_record()
    cell = cell_for(original)
    wire = canonical_json(cell.model_dump(mode="json", exclude_unset=True))
    restored = RawCell.model_validate_json(wire)
    assert restored.record.to_json() == original.to_json()
    assert restored.record_digest == original.digest
    assert "reason" not in restored.model_fields_set
    assert "source_revision" not in restored.record.provenance.model_fields_set
    assert "minimum" not in restored.record.measurements[0].model_fields_set


def test_explicit_null_provenance_and_reason_are_not_silently_omitted():
    original = producer_record(source_revision=None)
    cell = cell_for(original, reason=None)
    restored = RawCell.model_validate_json(canonical_json(cell.model_dump(mode="json", exclude_unset=True)))
    assert restored.record.to_json() == original.to_json()
    assert "source_revision" in restored.record.provenance.model_fields_set
    assert "reason" in restored.model_fields_set
    assert restored.reason is None


@pytest.mark.parametrize(
    "changes",
    [{"candidate_id": "another"}, {"member_id": "another"}, {"seed": 42}],
)
def test_original_record_cannot_be_reassigned_to_another_candidate_member_or_seed(changes):
    cell = cell_for(producer_record())
    with pytest.raises(ValueError, match="declared repetition cell"):
        RawCell.model_validate(cell.model_copy(update=changes))


@pytest.mark.parametrize("seed", [True, -1, 2**64, 1.5, "11"])
def test_cell_seed_is_a_strict_unsigned_coordinate(seed):
    cell = cell_for(producer_record())
    with pytest.raises(ValueError):
        RawCell.model_validate(cell.model_copy(update={"seed": seed}))


def test_raw_body_provenance_growth_is_not_mistaken_for_header_bytes():
    short = cell_for(producer_record())
    long = cell_for(producer_record(source_revision="reviewed-source-" + "x" * 1000))
    assert long.evidence.record_bytes > short.evidence.record_bytes + 1000
    assert long.header_bytes == short.header_bytes


def test_header_counts_utf8_bytes_in_both_member_references():
    ascii_cell = cell_for(producer_record(member="a" * 10))
    unicode_cell = cell_for(producer_record(member="é" * 10))
    assert unicode_cell.header_bytes == ascii_cell.header_bytes + 20


@pytest.mark.parametrize(
    ("value", "old_token", "new_token", "message"),
    [
        (0.0, '"value":0.0', '"value":1e-9999', "underflows"),
        (
            float(2**53),
            '"value":9007199254740992.0',
            '"value":9007199254740993',
            "not exactly representable",
        ),
    ],
)
def test_new_envelope_rejects_tokens_that_legacy_float_coercion_could_hide(value, old_token, new_token, message):
    cell = cell_for(producer_record(value=value))
    original = canonical_json(cell.model_dump(mode="json", exclude_unset=True))
    assert old_token in original
    hostile = original.replace(old_token, new_token)
    with pytest.raises(ValueError, match=message):
        RawCell.model_validate_json(hostile)
