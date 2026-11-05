"""Portable scientific evidence rejects self-consistent corruption, not just broken digests."""

import base64
import copy
import hashlib
import io

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from coleman.application.workflow_exports import EvidenceBundle
from coleman.application.workflow_validation import expected_trials_for_report
from coleman.workflows.lifecycle import ExecutePlan, WorkflowReport
from tests.workflows.test_execution import setup_workflow, wait


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    service, preview = setup_workflow(tmp_path_factory.mktemp("portable-boundaries"))
    job = service.execute(ExecutePlan(schema_version=1, request_id="portable", plan_id=preview.plan_id))
    wait(service, job.job_id)
    return service.export(job.job_id)


def changed_bundle(bundle, case):
    data = copy.deepcopy(bundle.model_dump(mode="json"))
    receipt = data["report"]["trials"][0]
    if case == "future":
        data["schema_version"] = 2
    elif case == "wrong_plan":
        data["report"]["plan_fingerprint"] = "sha256:" + "0" * 64
    elif case == "duplicate_file":
        data["files"].append(data["files"][0])
    elif case == "missing_file":
        data["files"].pop()
    elif case == "duplicate_trial":
        data["report"]["trials"].append(receipt)
    elif case == "unknown_trial":
        receipt["trial_id"] = "sha256:" + "0" * 64
    elif case == "incomplete_rows":
        receipt["rows"].pop()
    elif case == "wrong_stage":
        receipt["stage"] = "evaluation"
    elif case == "missing_trial":
        omitted = data["report"]["trials"].pop()
        names = {pin["name"] for pin in omitted["files"]}
        data["files"] = [file for file in data["files"] if file["name"] not in names]
    elif case == "wrong_metric":
        receipt["rows"][0]["napfd"] = -0.1234
    elif case == "wrong_pin":
        receipt["files"][0]["sha256"] = "sha256:" + "0" * 64
    elif case == "wrong_units":
        data["plan"]["request"]["budget"]["unit"] = "s"
    elif case == "missing_hive":
        previous = receipt["files"][0]["name"]
        receipt["files"][0]["name"] = "native.parquet"
        next(file for file in data["files"] if file["name"] == previous)["name"] = "native.parquet"
    elif case == "duplicate_physical_attempt":
        data["report"]["physical_attempts"].append(data["report"]["physical_attempts"][0])
    elif case == "missing_physical_attempts":
        data["report"]["physical_attempts"] = []
    elif case == "unplanned_selection":
        data["report"]["selected_configuration"] = "sha256:" + "0" * 64
    elif case == "unplanned_fallback":
        data["report"]["fallback_configuration"] = "sha256:" + "0" * 64
    elif case == "ucb_fallback":
        data["report"]["fallback_configuration"] = next(
            trial.configuration_id for trial in bundle.plan.trials if trial.policy == "UCB"
        )
    elif case == "excess_attempt_ordinal":
        data["report"]["physical_attempts"][0]["ordinal"] = 8
    elif case == "excess_worker_invocation":
        data["report"]["physical_attempts"][0]["worker_invocation"] = 8
    elif case == "wrong_attempt_stage":
        data["report"]["physical_attempts"][0]["stage"] = "evaluation"
    elif case == "wrong_interpretation":
        data["report"]["interpretation"] = "insufficient_evidence"
    return data


@pytest.mark.parametrize(
    "case",
    [
        "future",
        "wrong_plan",
        "duplicate_file",
        "missing_file",
        "duplicate_trial",
        "unknown_trial",
        "incomplete_rows",
        "wrong_stage",
        "missing_trial",
        "wrong_metric",
        "wrong_pin",
        "wrong_units",
        "missing_hive",
        "duplicate_physical_attempt",
        "missing_physical_attempts",
        "unplanned_selection",
        "unplanned_fallback",
        "ucb_fallback",
        "excess_attempt_ordinal",
        "excess_worker_invocation",
        "wrong_attempt_stage",
        "wrong_interpretation",
    ],
)
def test_corrupt_or_unplanned_portable_membership_rejects(evidence, case):
    changed_payload = changed_bundle(evidence, case)
    with pytest.raises(ValueError):
        EvidenceBundle.model_validate(changed_payload)


def replace_first_bytes(bundle, payload):
    data = copy.deepcopy(bundle.model_dump(mode="json"))
    pin = data["report"]["trials"][0]["files"][0]
    file = next(file for file in data["files"] if file["name"] == pin["name"])
    digest = hashlib.sha256(payload).hexdigest()
    pin.update(sha256="sha256:" + digest, size_bytes=len(payload))
    file.update(sha256=digest, size_bytes=len(payload), content_base64=base64.b64encode(payload).decode("ascii"))
    return data


def test_native_schema_mismatch_rejects_even_when_file_and_pin_digests_match(evidence):
    output = io.BytesIO()
    pl.DataFrame({"unrecognized_column": [1]}).write_parquet(output)
    schema_payload = replace_first_bytes(evidence, output.getvalue())
    with pytest.raises(ValueError, match="schema"):
        EvidenceBundle.model_validate(schema_payload)
    broken_footer_payload = replace_first_bytes(evidence, b"PAR1 broken footer")
    with pytest.raises(ValueError):
        EvidenceBundle.model_validate(broken_footer_payload)


def test_compressed_native_decode_bomb_is_rejected_from_footer_before_materialization(evidence, monkeypatch):
    output = io.BytesIO()
    pq.write_table(pa.table({"payload": ["x" * (65 * 1024**2)]}), output, compression="zstd", write_statistics=False)
    payload = output.getvalue()
    assert len(payload) < 2097152

    def forbidden(*args, **kwargs):
        raise AssertionError("Oversized native payload must not be decoded")

    monkeypatch.setattr(pl, "read_parquet", forbidden)
    oversized_payload = replace_first_bytes(evidence, payload)
    with pytest.raises(ValueError, match="decode bounds"):
        EvidenceBundle.model_validate(oversized_payload)


def test_portable_native_aggregate_bytes_reject_before_native_decode(evidence, monkeypatch):
    import coleman.application.workflow_exports as exports

    data = replace_first_bytes(evidence, b"x" * 1048577)

    def forbidden(*args, **kwargs):
        raise AssertionError("Aggregate encoded bound must precede native decode")

    monkeypatch.setattr(exports, "_native_table", forbidden)
    with pytest.raises(ValueError, match="1 MiB"):
        EvidenceBundle.model_validate(data)


def test_decoded_allowance_applies_before_materializing_each_file(evidence, monkeypatch):
    from coleman.application.workflow_exports import _native_table

    def forbidden(*args, **kwargs):
        raise AssertionError("Remaining aggregate allowance must precede native decode")

    monkeypatch.setattr(pl, "read_parquet", forbidden)
    with pytest.raises(ValueError, match="decode bounds"):
        _native_table(evidence.files[0], maximum_decoded=0)


def test_physical_attempt_identity_and_descriptive_inference_cannot_be_forged(evidence):
    plan, report = evidence.plan, evidence.report
    first = report.trials[0]
    row = first.rows[0].model_copy(update={"attempt_id": "not-a-reserved-attempt"})
    modified = first.model_copy(update={"rows": (row, *first.rows[1:])})
    bad = WorkflowReport.model_validate({**report.model_dump(), "trials": (modified, *report.trials[1:])})
    with pytest.raises(ValueError, match="physical attempt"):
        expected_trials_for_report(plan, bad)
    bad = report.model_copy(update={"physical_attempts": (*report.physical_attempts, report.physical_attempts[0])})
    with pytest.raises(ValueError, match="duplicate/unplanned"):
        expected_trials_for_report(plan, bad)
