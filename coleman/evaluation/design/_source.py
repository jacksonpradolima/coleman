"""Optional bounded source reservations, including accesses before selection.

The scientific scope header is immutable. Snapshot, encoding, candidate and
version changes cannot erase known exposure. This is a cooperative local audit,
not authentication, a global data-access monitor or protection from rollback.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from coleman.artifacts import ArtifactWriter
from coleman.evaluation.design._storage import (
    AuditCapacity,
    ProtocolCapacityError,
    bounded_paths,
    publish_bounded,
    read_contract,
)
from coleman.evaluation.design._wire import guard_graph
from coleman.evaluation.design.limits import ProtocolLimits
from coleman.evaluation.design.source_contracts import (
    SourcePartitionManifest,
    SourcePrefix,
    SourceReceipt,
    SourceRequest,
    SourceScope,
)


def audit_capacity(limits: ProtocolLimits, *, source: bool = False) -> AuditCapacity:
    """Translate one validated profile into the existing audit's selected bounds.

    Parameters
    ----------
    limits : ProtocolLimits
        Valid complete-envelope and actual acquisition limits.
    source : bool, optional
        Select source ordinals rather than legacy evaluation ordinals.

    Returns
    -------
    AuditCapacity
        Immutable private adapter bounds. No I/O occurs.
    """
    return AuditCapacity(
        receipts=limits.max_source_receipts if source else limits.max_audit_receipts,
        receipt_bytes=limits.max_receipt_bytes,
        freeze_bytes=limits.max_freeze_bytes,
        report_bytes=limits.max_envelope_bytes,
        directory_entries=limits.max_directory_entries,
        directory_bytes=limits.max_directory_bytes,
        configuration_bytes=limits.max_configuration_bytes,
        candidates=limits.max_candidates,
        hypotheses=limits.max_hypotheses,
        inputs=limits.max_aggregate_inputs,
        graph_items=limits.max_graph_items,
    )


def digest_path(root: Path, namespace: str, digest: str) -> Path:
    """Use only an identified lowercase SHA-256 as an immutable filename.

    Parameters
    ----------
    root : pathlib.Path
        Trusted local artifact root.
    namespace : str
        Adapter-owned literal namespace.
    digest : str
        Complete identified SHA-256, never a caller-supplied relative path.

    Returns
    -------
    pathlib.Path
        Local destination; no filesystem operation occurs.

    Raises
    ------
    ValueError
        Malformed or unidentified digest.
    """
    if not isinstance(digest, str) or len(digest) != 71 or not digest.startswith("sha256:"):
        raise ValueError("Expected an identified SHA-256 fingerprint")
    if any(character not in "0123456789abcdef" for character in digest[7:]):
        raise ValueError("Invalid SHA-256 digest")
    return root / namespace / (digest[7:] + ".json")


class SourceAudit:
    """Reserve every mediated source read before any result-bearing loader.

    Parameters
    ----------
    artifact_root : str or pathlib.Path
        Trusted cooperative local store; construction performs no I/O.
    limits : ProtocolLimits, optional
        Selected versioned acquisition/publication profile.

    Notes
    -----
    Failed loaders and abandoned reservations remain exposure. Receipt ordinals
    use the existing immutable ArtifactWriter's create-if-absent publication;
    at most 1000 conflicts are retried. Source and evaluation chains are separate
    conservative reservations, not one transaction. An empty directory never
    establishes unexposed prehistory. Caller-owned declarations and completeness
    of upstream aliases/external history remain trust boundaries.
    """

    def __init__(self, artifact_root: str | Path, *, limits: ProtocolLimits | None = None) -> None:
        """Validate the selected profile and configure paths without creating files."""
        self.root = Path(artifact_root)
        self.limits = ProtocolLimits(schema_version=1) if limits is None else ProtocolLimits.model_validate(limits)
        self.capacity = audit_capacity(self.limits, source=True)
        self.writer = ArtifactWriter(self.root)

    def _scope_path(self, scope: SourceScope) -> Path:
        """Keep the complete header at a key independent of snapshot/version."""
        return digest_path(self.root, "source-scopes", scope.logical_key)

    def _receipt_root(self, scope: SourceScope) -> Path:
        """Return the same chain directory for every version of one logical scope."""
        return self.root / "source-access" / scope.logical_key[7:]

    def _save_scope(self, scope: SourceScope) -> None:
        """Publish an immutable exact scope; incompatible replacements conflict."""
        guard_graph(scope, maximum_bytes=8192, maximum_items=self.limits.max_graph_items)
        publish_bounded(self.writer, self._scope_path(scope), scope.to_json() + "\n", 8192)

    def save_manifest(self, manifest: SourcePartitionManifest) -> Path:
        """Persist complete source metadata before permitting a narrow read claim.

        Parameters
        ----------
        manifest : SourcePartitionManifest
            Complete known metadata, including development and excluded rows.

        Returns
        -------
        pathlib.Path
            Immutable manifest destination; identical retries are idempotent.

        Raises
        ------
        ValueError, OSError, ProtocolCapacityError
            Invalid metadata, changed scope, capacity or publication failure.
            No source outcome is loaded.
        """
        guard_graph(manifest, maximum_bytes=self.limits.max_design_bytes, maximum_items=self.limits.max_graph_items)
        manifest = SourcePartitionManifest.model_validate(manifest)
        if len(manifest.members) > self.limits.max_source_members:
            raise ProtocolCapacityError("Source metadata exceeds its member capacity")
        self._save_scope(manifest.scope)
        path = digest_path(self.root, "source-manifests", manifest.fingerprint)
        publish_bounded(self.writer, path, manifest.to_json() + "\n", self.limits.max_design_bytes)
        return path

    def load_manifest(self, digest: str) -> SourcePartitionManifest:
        """Read and fingerprint-check bounded persisted metadata without outcomes.

        Parameters
        ----------
        digest : str
            Exact immutable manifest fingerprint.

        Returns
        -------
        SourcePartitionManifest
            Complete validated metadata. No result body is read.

        Raises
        ------
        ValueError, OSError, ProtocolCapacityError
            Missing, corrupt, changed or oversized manifest.
        """
        path = digest_path(self.root, "source-manifests", digest)
        result = read_contract(path, SourcePartitionManifest, self.limits.max_design_bytes, self.limits.max_graph_items)
        if result.fingerprint != digest or len(result.members) > self.limits.max_source_members:
            raise ValueError("Source manifest identity or member capacity mismatch")
        return result

    def verified_manifest(self, request: SourceRequest) -> SourcePartitionManifest | None:
        """Return the actual exact persisted support for a narrow partition claim.

        Parameters
        ----------
        request : SourceRequest
            Validated source request before outcome access.

        Returns
        -------
        SourcePartitionManifest or None
            Matching complete metadata, or conservative unknown when absent or
            mismatched. Corrupt/oversized existing files fail closed.
        """
        if request.potential_test_exposure:
            return None
        digest = request.partition_manifest_fingerprint
        if digest is None:
            return None
        try:
            manifest = self.load_manifest(digest)
        except FileNotFoundError:
            return None
        if manifest.scope != request.scope or manifest.snapshot != request.snapshot:
            return None
        return manifest

    def history(self, scope: SourceScope) -> SourcePrefix:
        """Read a complete bounded source prefix, including exactly-full chains.

        Parameters
        ----------
        scope : SourceScope
            Exact stable header expected by this caller.

        Returns
        -------
        SourcePrefix
            Contiguous chain in ordinal order; no timestamp sorting or repair.

        Raises
        ------
        ValueError, OSError, ProtocolCapacityError
            Changed header, malformed name/link, corruption or acquisition limit.
        """
        guard_graph(scope, maximum_bytes=8192, maximum_items=self.limits.max_graph_items)
        scope = SourceScope.model_validate(scope)
        header = self._scope_path(scope)
        if header.exists() and read_contract(header, SourceScope, 8192, self.limits.max_graph_items) != scope:
            raise ValueError("Source scope header cannot change within a logical namespace")
        receipts = []
        for sequence, path in enumerate(bounded_paths(self._receipt_root(scope), self.capacity), start=1):
            if path.name != f"{sequence:012d}.json":
                raise ValueError("Invalid source receipt ordinal filename")
            receipts.append(
                read_contract(path, SourceReceipt, self.capacity.receipt_bytes, self.limits.max_graph_items)
            )
        if receipts and not header.exists():
            raise ValueError("Source chain has lost its immutable scope header")
        return SourcePrefix(schema_version=1, scope=scope, receipts=tuple(receipts))

    def access(self, request: SourceRequest) -> SourceReceipt:
        """Durably reserve one actual read; no result-bearing loader is invoked.

        Parameters
        ----------
        request : SourceRequest
            Import/inspection/tuning may precede configuration selection. Missing
            narrow support is preserved and conservatively counts as exposure.

        Returns
        -------
        SourceReceipt
            Newly published actual-time reservation, including for future failed
            reads. Calling again creates a new attempt, never an invisible retry.

        Raises
        ------
        ValueError, OSError, ProtocolCapacityError, RuntimeError
            Invalid/future source capture, changed scope, capacity, filesystem
            failure or exhausted contention. No source outcome is read.
        """
        guard_graph(request, maximum_bytes=2048, maximum_items=self.limits.max_graph_items)
        request = SourceRequest.model_validate(request)
        now = datetime.now(UTC)
        if request.snapshot.captured_at is not None and request.snapshot.captured_at > now:
            raise ValueError("Source capture/import cannot be in the future at reservation")
        manifest_verified = self.verified_manifest(request) is not None
        self._save_scope(request.scope)
        access_id = uuid4().hex
        for _ in range(1000):
            prefix = self.history(request.scope)
            if len(prefix.receipts) >= self.capacity.receipts:
                raise ProtocolCapacityError("Source access slots exhausted")
            receipt = SourceReceipt(
                schema_version=1,
                request=request,
                partition_manifest_verified=manifest_verified,
                sequence=len(prefix.receipts) + 1,
                access_id=access_id,
                requested_at=now,
                previous_fingerprint=prefix.receipts[-1].fingerprint if prefix.receipts else None,
            )
            path = self._receipt_root(request.scope) / f"{receipt.sequence:012d}.json"
            try:
                publish_bounded(
                    self.writer,
                    path,
                    receipt.to_json() + "\n",
                    self.capacity.receipt_bytes,
                    receipt_capacity=self.capacity,
                )
            except ValueError:
                if path.exists():
                    continue
                raise
            return receipt
        raise RuntimeError("Source reservation exceeded its contention limit")
