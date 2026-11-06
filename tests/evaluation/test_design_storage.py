"""Actual bounded legacy audit acquisition, CAS capacity and compatibility."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from coleman.evaluation.design._storage import (
    AuditCapacity,
    BoundedEvaluationAudit,
    ProtocolCapacityError,
    bounded_paths,
    read_bounded,
)
from coleman.evaluation.protected.audit import EvaluationAudit
from coleman.evaluation.protected.comparison import FamilyReport, evaluate
from coleman.evaluation.protected.contracts import FrozenCandidate, FrozenEvaluation
from coleman.evaluation.protected.observations import PairedObservation

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Named pipes require a POSIX filesystem")
def test_direct_bounded_reader_refuses_real_fifo_without_waiting_for_writer(tmp_path):
    path = tmp_path / "not-a-regular-artifact.json"
    os.mkfifo(path)
    script = """
import sys
from pathlib import Path
from coleman.evaluation.design._storage import read_bounded
try:
 read_bounded(Path(sys.argv[1]),1024)
except OSError as error:
 assert 'regular files' in str(error)
else:
 raise AssertionError('FIFO was accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def capacity(*, receipts=2, entries=8):
    return AuditCapacity(receipts, 2048, 65536, 2 * 1024**2, entries, 16384, 4096)


def frozen_fixture():
    return FrozenEvaluation.from_json((FIXTURES / "bounded-freeze-v1.json").read_bytes())


def reserve(audit, frozen):
    return audit.access(frozen, partition="test", purpose="verification", actor="bounded-mechanics")


def test_exact_full_history_readable_but_next_actual_cas_refuses(tmp_path):
    frozen = frozen_fixture()
    audit = BoundedEvaluationAudit(tmp_path, capacity())
    first, second = reserve(audit, frozen), reserve(audit, frozen)
    assert second.sequence == 2
    assert second.prior_test_accesses == 1
    assert audit.history(frozen.protocol.holdout_id) == (first, second)
    with pytest.raises(ProtocolCapacityError, match="slots exhausted"):
        reserve(audit, frozen)
    assert audit.history(frozen.protocol.holdout_id) == (first, second)


def test_default_v1_reader_writer_remains_usable_above_new_selected_cap(tmp_path):
    frozen = frozen_fixture()
    bounded = BoundedEvaluationAudit(tmp_path, capacity())
    reserve(bounded, frozen)
    reserve(bounded, frozen)
    legacy = EvaluationAudit(tmp_path)
    third = reserve(legacy, frozen)
    assert third.sequence == 3
    assert len(legacy.history(frozen.protocol.holdout_id)) == 3
    with pytest.raises(ProtocolCapacityError, match="history capacity"):
        bounded.history(frozen.protocol.holdout_id)


def test_family_report_at_last_valid_receipt_can_still_be_saved_and_read(tmp_path):
    original = FamilyReport.from_json((FIXTURES / "bounded-report-v1.json").read_bytes())
    audit = BoundedEvaluationAudit(tmp_path, capacity())
    reserve(audit, original.frozen)
    observations = tuple(
        PairedObservation(candidate_id=item.candidate_id, metadata=item.metadata, record=item.record)
        for item in original.inputs
    )
    report = evaluate(original.frozen, audit, lambda: observations, actor="actual-last-slot")
    assert report.access.sequence == 2
    target = audit.save_report(report)
    loaded = FamilyReport.from_json(read_bounded(target, audit.capacity.report_bytes))
    assert loaded.fingerprint == report.fingerprint
    assert len(audit.history(original.frozen.protocol.holdout_id)) == 2
    assert not loaded.confirmatory_eligible


def test_raw_count_and_configuration_caps_precede_legacy_validation_or_writes(tmp_path):
    frozen = frozen_fixture()
    audit = BoundedEvaluationAudit(tmp_path / "absent", capacity())
    too_many = FrozenEvaluation.model_construct(**(vars(frozen) | {"candidates": (frozen.candidates[0],) * 65}))
    with pytest.raises(ProtocolCapacityError, match="candidates collection"):
        audit.save_freeze(too_many)
    oversized = FrozenCandidate.model_construct(
        candidate_id=frozen.candidates[0].candidate_id, configuration_json='{"x":"' + "x" * 4097 + '"}'
    )
    bad_configuration = FrozenEvaluation.model_construct(
        **(vars(frozen) | {"candidates": (oversized, *frozen.candidates[1:])})
    )
    with pytest.raises(ProtocolCapacityError, match="configuration"):
        audit.save_freeze(bad_configuration)
    assert not audit.root.exists()


def test_a_grown_receipt_is_bounded_during_actual_acquisition(tmp_path):
    frozen = frozen_fixture()
    audit = BoundedEvaluationAudit(tmp_path, capacity())
    reserve(audit, frozen)
    path = audit._scope_root(frozen.protocol.holdout_id) / "000000000001.json"
    path.write_text('"' + "x" * 2048 + '"')
    with pytest.raises(ProtocolCapacityError, match="bounded UTF-8"):
        audit.history(frozen.protocol.holdout_id)


def test_complete_wire_exact_byte_limit_and_utf8_limit_plus_one(tmp_path):
    path = tmp_path / "evidence.json"
    original = '{"meaning":"cost-Δ","value":0}\n'.encode()
    path.write_bytes(original)
    assert read_bounded(path, len(original)) == original
    with pytest.raises(ProtocolCapacityError):
        read_bounded(path, len(original) - 1)


def test_temporary_and_foreign_names_count_toward_scan_capacity(tmp_path):
    for index in range(8):
        (tmp_path / f"foreign-{index}.tmp").write_bytes(b"x")
    assert bounded_paths(tmp_path, capacity()) == []
    extra = tmp_path / "foreign-9.bin"
    extra.write_bytes(b"x")
    with pytest.raises(ProtocolCapacityError, match="entry capacity"):
        bounded_paths(tmp_path, capacity())
    assert extra.read_bytes() == b"x"


def test_known_final_and_temporary_name_headroom_refuses_before_ack(tmp_path):
    frozen = frozen_fixture()
    audit = BoundedEvaluationAudit(tmp_path, capacity(entries=4))
    first = reserve(audit, frozen)
    scope = audit._scope_root(frozen.protocol.holdout_id)
    for number in range(2):
        (scope / f"foreign-{number}.tmp").write_bytes(b"x")
    with pytest.raises(ProtocolCapacityError, match="temporary-name"):
        reserve(audit, frozen)
    assert audit.history(frozen.protocol.holdout_id) == (first,)


def test_scope_directory_byte_guard_counts_foreign_content_without_reading_it(tmp_path):
    foreign = tmp_path / "unrelated.bin"
    foreign.write_bytes(b"x" * 16385)
    with pytest.raises(ProtocolCapacityError, match="directory byte capacity"):
        bounded_paths(tmp_path, capacity())
    assert foreign.stat().st_size == 16385


def test_links_and_subdirectories_are_unsupported_acquisition(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"{}")
    directory = tmp_path / "receipt-directory"
    directory.mkdir()
    link = directory / "000000000001.json"
    link.symlink_to(target)
    with pytest.raises(OSError, match="regular files"):
        bounded_paths(directory, capacity())
    with pytest.raises(OSError, match="symbolic links"):
        read_bounded(link, 2048)
    link.unlink()
    (directory / "nested").mkdir()
    with pytest.raises(OSError, match="regular files"):
        bounded_paths(directory, capacity())


@pytest.mark.parametrize("changes", [{"receipts": True}, {"directory_entries": 3}, {"directory_bytes": 8191}])
def test_internal_capacity_rejects_infeasible_limits_before_filesystem_access(changes):
    values = {
        "receipts": 2,
        "receipt_bytes": 2048,
        "freeze_bytes": 65536,
        "report_bytes": 2 * 1024**2,
        "directory_entries": 8,
        "directory_bytes": 16384,
        "configuration_bytes": 4096,
    }
    with pytest.raises(ValueError):
        AuditCapacity(**(values | changes))


def test_actual_concurrent_last_slot_has_one_winner_and_no_max_plus_one_file(tmp_path):
    frozen = frozen_fixture()
    root = tmp_path / "audit"
    audit = BoundedEvaluationAudit(root, capacity())
    reserve(audit, frozen)
    freeze_file = tmp_path / "frozen.json"
    freeze_file.write_text(frozen.to_json())
    script = r"""
import json,sys,time
from pathlib import Path
from coleman.evaluation.design._storage import AuditCapacity,BoundedEvaluationAudit,ProtocolCapacityError
from coleman.evaluation.protected.contracts import FrozenEvaluation
root, frozen_file, gate, ready = map(Path,sys.argv[1:])
frozen=FrozenEvaluation.from_json(frozen_file.read_bytes())
audit=BoundedEvaluationAudit(root,AuditCapacity(2,2048,65536,2*1024**2,8,16384,4096))
ready.write_text('ready')
deadline=time.monotonic()+30
while not gate.exists():
 if time.monotonic()>deadline:raise TimeoutError('parent gate')
 time.sleep(.01)
try:
 receipt=audit.access(frozen,partition='test',purpose='verification',actor='actual-contender')
except ProtocolCapacityError:
 print(json.dumps({'refused':True}))
else:
 print(json.dumps({'accepted':receipt.sequence}))
"""
    gate = tmp_path / "start"
    workers = []
    try:
        for number in range(2):
            ready = tmp_path / f"ready-{number}"
            workers.append(
                subprocess.Popen(
                    [sys.executable, "-c", script, str(root), str(freeze_file), str(gate), str(ready)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        deadline = time.monotonic() + 30
        while not all((tmp_path / f"ready-{number}").exists() for number in range(2)):
            assert time.monotonic() < deadline
            assert all(worker.poll() is None for worker in workers)
            time.sleep(0.01)
        gate.write_text("actual concurrent admission")
        outcomes = []
        for worker in workers:
            output, error = worker.communicate(timeout=30)
            assert worker.returncode == 0, output + error
            outcomes.append(json.loads(output))
        assert outcomes.count({"accepted": 2}) == 1
        assert outcomes.count({"refused": True}) == 1
        assert len(audit.history(frozen.protocol.holdout_id)) == 2
        assert not (audit._scope_root(frozen.protocol.holdout_id) / "000000000003.json").exists()
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=10)
