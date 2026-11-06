"""Real source-audit corruption, compatibility and prepublication refusals."""

import pytest
from design_support import changed, make_design

from coleman.evaluation.design.audit import ProtocolCapacityError, SourceAudit
from coleman.evaluation.design.source_contracts import SourceRequest


def source_request(*, purpose="inspection"):
    """Keep source capture unknown and all optional selected-operation joins explicit."""
    design = make_design()
    return SourceRequest(
        schema_version=1,
        scope=design.scope,
        snapshot=changed(design.snapshot, captured_at=None),
        purpose=purpose,
        partitions=("test",),
        partition_manifest_fingerprint=None,
        actor="source-recovery-fixture",
        selection_fingerprint="sha256:" + "1" * 64,
        evaluation_fingerprint="sha256:" + "2" * 64,
        report_fingerprint="sha256:" + "3" * 64,
    )


def test_missing_immutable_header_cannot_turn_retained_receipt_into_empty_history(tmp_path):
    request = source_request()
    audit = SourceAudit(tmp_path)
    receipt = audit.access(request)
    header = tmp_path / "source-scopes" / (request.scope.logical_key[7:] + ".json")
    original = tmp_path / "source-access" / request.scope.logical_key[7:] / "000000000001.json"
    retained = original.read_bytes()
    header.unlink()

    with pytest.raises(ValueError, match="lost its immutable scope header"):
        audit.history(request.scope)

    assert not header.exists()
    assert original.read_bytes() == retained
    assert receipt.sequence == 1


def test_ordinal_filename_gap_is_not_repaired_or_returned_as_a_partial_prefix(tmp_path):
    request = source_request()
    audit = SourceAudit(tmp_path)
    audit.access(request)
    directory = tmp_path / "source-access" / request.scope.logical_key[7:]
    original = directory / "000000000001.json"
    retained = original.read_bytes()
    displaced = directory / "000000000002.json"
    original.rename(displaced)

    with pytest.raises(ValueError, match="ordinal filename"):
        audit.history(request.scope)

    assert not original.exists()
    assert displaced.read_bytes() == retained


def test_same_logical_scope_with_changed_definition_cannot_read_the_original_header(tmp_path):
    request = source_request()
    audit = SourceAudit(tmp_path)
    first = audit.access(request)
    header = tmp_path / "source-scopes" / (request.scope.logical_key[7:] + ".json")
    retained = header.read_bytes()
    replacement = changed(
        request.scope,
        upstream=changed(request.scope.upstream, definition_fingerprint="sha256:" + "f" * 64),
    )
    assert replacement.logical_key == request.scope.logical_key

    with pytest.raises(ValueError, match="header cannot change"):
        audit.history(replacement)

    assert header.read_bytes() == retained
    assert audit.history(request.scope).receipts == (first,)


def test_valid_manifest_bytes_at_another_digest_path_fail_content_identity_without_deletion(tmp_path):
    manifest = make_design().partition_manifest
    audit = SourceAudit(tmp_path)
    original = audit.save_manifest(manifest)
    retained = original.read_bytes()
    other_digest = "sha256:" + "a" * 64
    assert other_digest != manifest.fingerprint
    misplaced = tmp_path / "source-manifests" / (other_digest[7:] + ".json")
    misplaced.write_bytes(retained)

    with pytest.raises(ValueError, match="manifest identity"):
        audit.load_manifest(other_digest)

    assert original.read_bytes() == misplaced.read_bytes() == retained
    assert audit.load_manifest(manifest.fingerprint) == manifest


def test_narrow_claim_for_a_different_snapshot_remains_unverified_and_retained(tmp_path):
    design = make_design()
    manifest = design.partition_manifest
    audit = SourceAudit(tmp_path)
    original = audit.save_manifest(manifest)
    retained = original.read_bytes()
    request = changed(
        source_request(),
        snapshot=changed(
            manifest.snapshot,
            captured_at=None,
            reference=changed(manifest.snapshot.reference, version="different-selected-revision"),
        ),
        partitions=("validation",),
        partition_manifest_fingerprint=manifest.fingerprint,
    )
    assert not request.potential_test_exposure
    assert audit.verified_manifest(request) is None

    receipt = audit.access(request)

    assert not receipt.partition_manifest_verified
    assert receipt.request.snapshot == request.snapshot
    assert audit.history(request.scope).receipts == (receipt,)
    assert original.read_bytes() == retained


def test_selected_member_cap_refuses_manifest_before_any_scope_or_body_publication(tmp_path):
    design = make_design()
    manifest = design.partition_manifest
    assert len(manifest.members) == 2
    limits = changed(design.limits, max_source_members=1)
    root = tmp_path / "unpublished-source-store"
    audit = SourceAudit(root, limits=limits)

    with pytest.raises(ProtocolCapacityError, match="member capacity"):
        audit.save_manifest(manifest)

    assert not root.exists()


def test_stricter_reader_rejects_existing_manifest_members_without_changing_original_bytes(tmp_path):
    design = make_design()
    manifest = design.partition_manifest
    original = SourceAudit(tmp_path).save_manifest(manifest)
    retained = original.read_bytes()
    strict = SourceAudit(tmp_path, limits=changed(design.limits, max_source_members=1))

    with pytest.raises(ValueError, match="member capacity mismatch"):
        strict.load_manifest(manifest.fingerprint)

    assert original.read_bytes() == retained
    assert SourceAudit(tmp_path).load_manifest(manifest.fingerprint) == manifest


@pytest.mark.parametrize("digest", ["SHA256:" + "a" * 64, "sha256:" + "g" * 64])
def test_malformed_manifest_digest_refuses_before_file_acquisition_or_store_creation(tmp_path, digest):
    root = tmp_path / "malformed-digest-store"
    audit = SourceAudit(root)

    with pytest.raises(ValueError, match="SHA-256"):
        audit.load_manifest(digest)

    assert not root.exists()


@pytest.mark.parametrize(
    "purpose,missing",
    [
        ("protected_evaluation", "selection_fingerprint"),
        ("protected_evaluation", "evaluation_fingerprint"),
        ("result_disclosure", "report_fingerprint"),
    ],
)
def test_access_revalidates_unchecked_typed_operation_joins_before_publication(tmp_path, purpose, missing):
    valid = source_request(purpose=purpose)
    unchecked = valid.model_copy(update={missing: None})
    assert getattr(valid, missing) is not None
    root = tmp_path / "invalid-selected-operation"
    audit = SourceAudit(root)

    with pytest.raises(ValueError, match="requires"):
        audit.access(unchecked)

    assert not root.exists()
