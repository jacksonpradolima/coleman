"""Actual last-slot source reservations across independent local processes."""

import json
import subprocess
import sys
import time

import pytest
from design_support import changed, make_design

from coleman.evaluation.design.audit import ProtocolCapacityError, SourceAudit
from coleman.evaluation.design.source_contracts import SourceRequest


def test_actual_concurrent_source_last_slot_has_one_winner_and_full_prefix_stays_readable(tmp_path):
    design = make_design()
    limits = changed(design.limits, max_source_receipts=2, max_audit_receipts=2)
    request = SourceRequest(
        schema_version=1,
        scope=design.scope,
        snapshot=changed(design.snapshot, captured_at=None),
        purpose="inspection",
        partitions=("test",),
        partition_manifest_fingerprint=None,
        actor="actual-source-contender",
        selection_fingerprint=None,
        evaluation_fingerprint=None,
        report_fingerprint=None,
    )
    root = tmp_path / "source-audit"
    sources = SourceAudit(root, limits=limits)
    first = sources.access(request)
    request_file = tmp_path / "request.json"
    request_file.write_text(request.to_json(), encoding="utf-8")
    script = r"""
import json,sys,time
from pathlib import Path
from coleman.evaluation.design.audit import ProtocolCapacityError,SourceAudit
from coleman.evaluation.design.limits import ProtocolLimits
from coleman.evaluation.design.source_contracts import SourceRequest
root,request_file,gate,ready=map(Path,sys.argv[1:])
request=SourceRequest.from_json(request_file.read_bytes())
audit=SourceAudit(root,limits=ProtocolLimits(schema_version=1,max_source_receipts=2,max_audit_receipts=2))
ready.write_text('ready',encoding='utf-8')
deadline=time.monotonic()+30
while not gate.exists():
 if time.monotonic()>deadline:raise TimeoutError('source parent gate')
 time.sleep(.01)
try:
 receipt=audit.access(request)
except ProtocolCapacityError:
 print(json.dumps({'refused':True}))
else:
 print(json.dumps({'accepted':receipt.sequence}))
"""
    gate = tmp_path / "start-source-reservations"
    workers = []
    try:
        for number in range(2):
            ready = tmp_path / f"source-ready-{number}"
            workers.append(
                subprocess.Popen(
                    [sys.executable, "-c", script, str(root), str(request_file), str(gate), str(ready)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        deadline = time.monotonic() + 30
        while not all((tmp_path / f"source-ready-{number}").exists() for number in range(2)):
            assert time.monotonic() < deadline
            assert all(worker.poll() is None for worker in workers)
            time.sleep(0.01)
        gate.write_text("actual concurrent source admission", encoding="utf-8")
        outcomes = []
        for worker in workers:
            output, error = worker.communicate(timeout=30)
            assert worker.returncode == 0, output + error
            outcomes.append(json.loads(output))
        assert outcomes.count({"accepted": 2}) == 1
        assert outcomes.count({"refused": True}) == 1
        prefix = sources.history(request.scope)
        assert len(prefix.receipts) == 2
        assert prefix.receipts[0] == first
        assert prefix.receipts[1].previous_fingerprint == first.fingerprint
        with pytest.raises(ProtocolCapacityError, match="slots exhausted"):
            sources.access(request)
        assert sources.history(request.scope) == prefix
        assert not (root / "source-access" / request.scope.logical_key[7:] / "000000000003.json").exists()
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=10)
