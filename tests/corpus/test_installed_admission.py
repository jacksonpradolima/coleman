"""Fresh CI-capability wheel outside checkout: offline source audit, real publication crash and native checkpoints."""

import hashlib
import json
import os
import re
import shutil
import subprocess

import pytest

from coleman.application.admission import OriginalCaptureVerifier
from coleman.artifacts import ArtifactWriter
from coleman.corpus.admission import SourceEvidence, assess_source, audit_corpus, publish_corpus_admission
from coleman.corpus.admission_contracts import SourceAssessment
from coleman.corpus.admission_reality import production_facts
from coleman.corpus.admission_semantics import admitted_regimes, historical_blockers, qualification
from coleman.corpus.admission_sources import source_card
from coleman.corpus.controlled_contracts import ControlledDataset
from coleman.corpus.registry import load_registry
from coleman.transports import junit_matrix as matrix
from tests.corpus.admission_support import changed, lrts_source, review
from tests.corpus.test_installed_controlled import ROOT, command
from tests.corpus.test_junit_matrix import owned_capture
from tests.installed_support import build_install_offline_wheel

SCRIPT = r"""
import hashlib, importlib.abc, importlib.metadata, importlib.util, json, os, subprocess, sys
from datetime import UTC, datetime
from pathlib import Path
assert importlib.util.find_spec('mcp') is None
class Boundary(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.startswith(('coleman.cli', 'coleman.transports', 'mcp')) or fullname=='coleman.corpus.__main__':
   raise AssertionError('Domain/application imported a transport: '+fullname)
sys.meta_path.insert(0,Boundary())
import coleman, polars as pl
from coleman.application.corpus import run_lrts_replay
from coleman.artifacts import ArtifactWriter
from coleman.corpus.admission import (
 SourceEvidence,audit_corpus,publish_corpus_admission,validate_admission_completion)
from coleman.corpus.admission_contracts import CorpusAdmission,SourceReview
from coleman.corpus.lrts_etl_contracts import LRTSETLReceipt
from coleman.corpus.materialization import SourceCache
from coleman.corpus.registry import load_registry
base=Path(coleman.__file__).resolve().parent.parent
assert base.is_relative_to(Path(sys.prefix).resolve())
expected=json.loads(Path('expected-sdk.json').read_text())
for name,digest in expected.items():
 assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest,name
locations=json.loads(Path('source-locations.json').read_text())
completion=Path(locations['completion']);cache=SourceCache(locations['cache'])
payload=LRTSETLReceipt.from_json(completion.read_bytes())
review=SourceReview(schema_version=1,payload_fingerprint=payload.fingerprint,status='reviewed',
 reviewed_at=datetime(2026,10,18,tzinfo=UTC),history='constructed',
 reference='urn:coleman:owned-installed-admission-qualification',fault_reference=None,subject_language=None,
 limitations=('Owned post-hoc source/contract qualification; no real CI, independent faults or efficacy claim.',))
binding=SourceEvidence(payload,review,completion.parent,cache)
report=audit_corpus(load_registry(),(binding,))
assert report.assessments[0].profile.rows==10 and report.covered_regimes==()
unknown=next(m.value for m in report.assessments[0].profile.measurements if m.name=='source_unknown_outcomes')
assert unknown==4
assert len(report.uncovered_regimes)==7 and report.observed_source_models==()
assert CorpusAdmission.from_json(report.to_json())==report
writer=ArtifactWriter(Path('audit-output'))
if sys.argv[1]=='interrupt':
 original=ArtifactWriter.write_immutable
 def stopped(self,path,text):
  result=original(self,path,text)
  if path.name=='report.md':os._exit(73)
  return result
 ArtifactWriter.write_immutable=stopped
 publish_corpus_admission(report,(binding,),writer)
 raise AssertionError('Expected a genuine process exit before completion publication')
child=subprocess.run([sys.executable,__file__,'interrupt'],capture_output=True,timeout=120)
assert child.returncode==73,(child.stdout,child.stderr)
assert not list(writer.run_root.rglob('complete.json'))
partial={p.relative_to(writer.run_root):p.read_bytes() for p in writer.run_root.rglob('*') if p.is_file()}
assert len(partial)==2
marker=publish_corpus_admission(report,(binding,),writer)
assert validate_admission_completion(marker,(binding,))==report
assert all((writer.run_root/p).read_bytes()==value for p,value in partial.items())
assert publish_corpus_admission(report,(binding,),writer)==marker
result=run_lrts_replay(completion,cache,Path('native-output'))
where=Path(result.artifacts_dir)
rows=pl.read_parquet(str(where/'results/**/*.parquet'),hive_partitioning=True)
assert rows.height==4 and rows.select(pl.struct(['step','policy']).n_unique()).item()==4
protected={p.relative_to(where):p.read_bytes() for p in where.rglob('*') if p.suffix in {'.pkl','.parquet'}}
assert protected and any(p.suffix=='.pkl' for p in protected)
again=run_lrts_replay(completion,cache,Path('native-output'))
assert result.run_id==again.run_id
assert protected=={p.relative_to(where):p.read_bytes() for p in where.rglob('*') if p.suffix in {'.pkl','.parquet'}}
markers=list(where.rglob('progress_ex1.json'))
assert markers and all(json.loads(p.read_bytes())['step_committed']==2 for p in markers)
print(json.dumps(dict(installed=True,sdk_files=len(expected),domain_transport_boundary=True,
 actual_publication_interruption_exit=child.returncode,retained_rows=10,unknown_rows=int(unknown),
 native_rows=rows.height,native_checkpoint_step=2,protected_files=len(protected),run_id=result.run_id),sort_keys=True))
"""


def test_installed_offline_admission_crash_recovery_and_actual_native_checkpoint(tmp_path, monkeypatch):
    evidence, completion = lrts_source(tmp_path / "source")
    # Raw producer protocol fixture: only external Docker collection is
    # substituted. Original-source validation below uses actual unmodified
    # cache, SDK, raw-wire parser, normalized report and admission services.
    with monkeypatch.context() as external:
        plan, raw_cache, raw_writer, _ = owned_capture(tmp_path / "raw-matrix", external)
        captured = matrix.capture_junit_matrix(plan, raw_cache, raw_writer)
    assert evidence.cache is not None
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(UV_LINK_MODE="copy", UV_NO_PROGRESS="1")
    dist = tmp_path / "dist"
    virtual = tmp_path / "installed"
    build_install_offline_wheel(uv, dist, virtual, env, extras=("ci",))
    python = virtual / "bin/python"
    _original_wire_and_resigned_metadata_attack(
        captured, raw_cache, raw_writer, python, next(dist.glob("*.whl")), tmp_path
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "audit_check.py").write_text(SCRIPT)
    (outside / "source-locations.json").write_text(
        json.dumps({"completion": str(completion), "cache": str(evidence.cache.root)})
    )
    paths = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "--", "coleman"], cwd=ROOT, text=True
    )
    expected = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths.splitlines()}
    (outside / "expected-sdk.json").write_text(json.dumps(expected))
    result = command([str(python), str(outside / "audit_check.py"), "qualify"], outside, env)
    proof = json.loads(result.stdout.strip().splitlines()[-1])
    assert proof["installed"]
    assert proof["domain_transport_boundary"]
    assert proof["actual_publication_interruption_exit"] == 73
    assert proof["native_rows"] == 4
    assert proof["native_checkpoint_step"] == 2
    assert proof["sdk_files"] == len(expected)
    examples = re.findall(r"```python\n(.*?)\n```", (ROOT / "docs/corpus-admission.md").read_text(), re.S)
    assert len(examples) == 2
    docs = outside / "literal-doc-examples"
    docs.mkdir()
    env.update(COLEMAN_CI_COMPLETION=str(completion), COLEMAN_CI_CACHE=str(evidence.cache.root))
    for example in examples:
        command([str(python), "-c", example], docs, env)


def _original_wire_and_resigned_metadata_attack(captured, cache, writer, python, wheel, tmp_path):
    """Verify an original installed SDK and show that re-signed metadata cannot clear real source validation."""
    digest = "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest()
    verifier = OriginalCaptureVerifier(
        "junit_matrix",
        python,
        wheel,
        digest,
        "urn:coleman:owned-installed-verifier-wheel",
        captured.receipt_path,
        output=writer.run_root,
    )
    key = captured.receipt.cells[0].dataset_fingerprint.removeprefix("sha256:")
    payload = ControlledDataset.from_json((writer.run_root / "corpus-controlled" / (key + ".json")).read_bytes())
    binding = SourceEvidence(payload, review(payload), writer.run_root, cache, verifier)
    assessment = assess_source(binding)
    assert assessment.scope == "constructed_functional"
    assert assessment.producer_verification is not None
    assert assessment.producer_verification.package.sha256 == digest
    assert assessment.producer_verification.producer_sha256 == captured.receipt.producer_sha256
    report = audit_corpus(load_registry(), (binding,))
    assert report.covered_regimes == ()  # Owned synthetic wire fixture is not a real-fault corpus.

    # Deserialization validates attestations, not source truth. Forge coherent
    # metadata and a matching dataset pin, then require actual cache/original
    # producer validation to reject it before a completion artifact can exist.
    for language, regime in (("Java", "controlled_java_real_faults"), ("Python", "controlled_non_java_real_faults")):
        context = changed(payload.spec.context, fault_origin="real", language=language)
        forged_payload = changed(payload, spec=changed(payload.spec, context=context))
        declared = review(
            forged_payload, subject_language=language, fault_reference="urn:coleman:owned-forged-curation"
        )
        proof = changed(assessment.producer_verification, payload_fingerprint=forged_payload.fingerprint)
        scope, reasons = qualification(forged_payload, assessment.profile, declared, proof)
        metadata = SourceAssessment(
            schema_version=1,
            payload=forged_payload,
            review=declared,
            producer_verification=proof,
            profile=assessment.profile,
            card=source_card(forged_payload, assessment.profile, declared),
            reality=production_facts(forged_payload, declared),
            scope=scope,
            admission_reasons=reasons,
            historical_blockers=historical_blockers(forged_payload),
        )
        assert admitted_regimes((metadata,)) == (regime,)
        forged_binding = SourceEvidence(forged_payload, declared, writer.run_root, cache, verifier)
        with pytest.raises(ValueError, match="context fingerprint"):
            assess_source(forged_binding)
    wrong = OriginalCaptureVerifier(
        "junit_matrix",
        python,
        wheel,
        "sha256:" + "0" * 64,
        "urn:coleman:owned-wrong-wheel",
        captured.receipt_path,
        output=writer.run_root,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        wrong.verify(payload, cache)
    incomplete = SourceEvidence(payload, binding.review, writer.run_root, cache)
    assert "original_raw_producer_verification_missing" in assess_source(incomplete).admission_reasons
    incomplete_writer = ArtifactWriter(tmp_path / "must-not-complete-raw")
    with pytest.raises(ValueError, match="bindings"):
        publish_corpus_admission(report, (incomplete,), incomplete_writer)
    assert not list((tmp_path / "must-not-complete-raw").rglob("complete.json"))
