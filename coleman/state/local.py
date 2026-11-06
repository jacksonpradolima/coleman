"""Optional local acknowledgement adapter; direct learner state needs no store."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Self

from coleman.artifacts import ArtifactWriter
from coleman.state import _disk, _garbage, _prepare
from coleman.state.components import MigrationBindings, PreparedCollection, StagedCollection
from coleman.state.storage_contracts import (
    AcknowledgedHead,
    CleanupReceipt,
    CollectionDefinition,
    CollectionManifest,
    CommitIntent,
    CommitReceipt,
    CommitRequest,
    ExecutionCompatibility,
    MigrationReceipt,
    MigrationRequest,
    StorageLimits,
    semantic_frontiers_fingerprint,
)


class LocalStateStore:
    """Commit complete fresh local state collections with one accepted SQL head.

    Parameters
    ----------
    root : pathlib.Path
        Trusted suitable local filesystem root selected by create/open.
    header : StoreHeader
        Validated immutable adapter metadata; applications use create/open.

    Attributes
    ----------
    root : pathlib.Path
        Absolute adapter-owned flat artifact/SQLite directory.
    limits : StorageLimits
        Persisted finite capacities; never inferred from current free space.

    Notes
    -----
    SQLite COMMIT is the sole acceptance point. Precommit immutable artifacts
    are orphans; postcommit response loss does not roll back accepted learning.
    Callbacks execute once per attempt on fresh owners and must be bounded,
    local and free of external side effects. This is neither an external
    exactly-once protocol nor a distributed/network-filesystem store. Qualified
    SQLite WAL engines and FULL synchronization are required. Whole-store
    rollback requires an independent trusted minimum to detect.
    """

    def __init__(self, root: Path, header: _disk.StoreHeader) -> None:
        """Keep validated local adapter metadata; create/open perform I/O.

        Parameters
        ----------
        root : pathlib.Path
            Trusted absolute root.
        header : StoreHeader
            Validated immutable initial inventory and persisted limits.
        """
        self.root = root
        self.limits = header.limits
        self._header = header

    @classmethod
    def create(
        cls,
        root: Path,
        definition: CollectionDefinition,
        compatibility: ExecutionCompatibility,
        *,
        limits: StorageLimits | None = None,
    ) -> Self:
        """Explicitly create a virgin local store without initializing learners.

        Parameters
        ----------
        root : pathlib.Path
            Missing or empty trusted root; its parent must already exist.
        definition : CollectionDefinition
            Complete finite initial owner/child inventory.
        compatibility : ExecutionCompatibility
            Actual outer-host runtime and copied-owner identity capture.
        limits : StorageLimits or None, optional
            Persisted capacities; None selects the documented bounded defaults.

        Returns
        -------
        LocalStateStore
            Virgin adapter with no accepted learning or generation.

        Raises
        ------
        ValueError, OSError, sqlite3.Error
            Nonempty root, unsupported SQLite, invalid inventory/headroom or I/O.
        """
        _disk.qualify_sqlite()
        definition = CollectionDefinition.model_validate(definition)
        compatibility = ExecutionCompatibility.model_validate(compatibility)
        selected = StorageLimits(schema_version=1) if limits is None else StorageLimits.model_validate(limits)
        _disk.validate_capacity(definition, selected)
        path = Path(root).absolute()
        if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
            raise ValueError("State create requires a missing or empty trusted root")
        path.mkdir(exist_ok=True)
        (path / "blobs").mkdir()
        (path / "manifests").mkdir()
        header = _disk.StoreHeader(
            schema_version=1,
            profile=_disk.PROFILE,
            definition=definition,
            compatibility=compatibility,
            limits=selected,
            sqlite_version=sqlite3.sqlite_version,
        )
        ArtifactWriter(path).write_immutable(path / _disk.HEADER, header.to_json())
        _disk.initialize_database(path)
        _disk.usage(path, selected)
        return cls(path, header)

    @classmethod
    def open(cls, root: Path) -> Self:
        """Open existing metadata without fresh-state fallback or host restore.

        Parameters
        ----------
        root : pathlib.Path
            Complete existing trusted adapter root.

        Returns
        -------
        LocalStateStore
            Metadata-only handle, not execution compatibility approval.

        Raises
        ------
        ValueError, OSError, sqlite3.Error
            Missing/corrupt metadata, future profile/version or unsupported engine.
        """
        _disk.qualify_sqlite()
        path = Path(root).absolute()
        if path.is_symlink() or not path.is_dir() or not (path / _disk.DATABASE).is_file():
            raise ValueError("Existing state store metadata is missing; explicit create is required")
        header = _disk.StoreHeader.from_json(_disk.acquire(path / _disk.HEADER, 2 * _disk.MIB))
        if header.profile != _disk.PROFILE:
            raise ValueError("Unknown local state storage profile")
        _disk.validate_capacity(header.definition, header.limits)
        store = cls(path, header)
        store.read_head()
        return store

    def read_head(self) -> AcknowledgedHead | None:
        """Read integrity-checked ack metadata without restoring any owner.

        Returns
        -------
        AcknowledgedHead or None
            Actual current head or explicit virgin metadata.

        Raises
        ------
        ValueError, OSError, sqlite3.Error
            Corrupt/missing accepted artifacts or ownership acquisition failure.

        Notes
        -----
        This metadata inspection is not an operational compatibility receipt.
        """
        with _disk.ownership(self.root, self.limits.lock_wait_seconds, limits=self.limits) as database:
            current, _ = self._current(database)
            return current

    def initialize(
        self,
        request: CommitRequest,
        values: _prepare.Values,
        bindings: _prepare.Bindings,
        *,
        compatibility: ExecutionCompatibility,
    ) -> CommitReceipt:
        """Accept complete supplied snapshots, including already-pending learners.

        Parameters
        ----------
        request : CommitRequest
            Generation-zero initialize command, expected=None.
        values : tuple of ComponentValue
            Complete actual typed state, never replaced with factory defaults.
        bindings : tuple of ComponentBinding
            Every concrete codec, fresh factory and pure frontier inspector.
        compatibility : ExecutionCompatibility
            Explicit actual current environment/owner identity capture.

        Returns
        -------
        CommitReceipt
            New accepted head or exact retained original initialization receipt.

        Raises
        ------
        ValueError, TypeError, OSError, sqlite3.Error
            Incompatible/stale/conflicting input, failed restore or publication.
            Trusted validator/factory errors propagate; no acknowledgement occurs.
        """
        request = CommitRequest.model_validate(request)
        if request.intent.kind != "initialize" or request.intent.generation != 0 or request.expected is not None:
            raise ValueError("Initialization requires generation zero, initialize intent and no predecessor")
        with _disk.ownership(self.root, self.limits.lock_wait_seconds, limits=self.limits) as database:
            current, old = self._current(database)
            self._compatible(compatibility, old)
            retained = self._request(database, request, current)
            if retained is not None:
                return retained
            if current is not None or request.expected is not None or request.intent.kind != "initialize":
                raise ValueError("Only an explicit virgin initialize command can accept supplied initial state")
            _garbage.reserve(self.root, database, self.limits, current)
            prepared = _prepare.build(self._header.definition, compatibility, request, values, bindings, self.limits)
            staged = _prepare.stage(prepared, bindings)
            staged._close()  # pylint: disable=protected-access
            return self._publish(database, request, prepared)

    def load(self, bindings: _prepare.Bindings, *, compatibility: ExecutionCompatibility) -> PreparedCollection:
        """Decode all acknowledged owners without constructing or restoring any.

        Parameters
        ----------
        bindings : tuple of ComponentBinding
            Exact finite complete concrete codecs and pure inspectors.
        compatibility : ExecutionCompatibility
            Actual explicit current execution binding, checked before inspectors.

        Returns
        -------
        PreparedCollection
            Complete typed immutable state for explicit host inspection.

        Raises
        ------
        ValueError, TypeError, OSError, sqlite3.Error
            Virgin/corrupt/incompatible state; no fallback or owner call occurs.
        """
        with _disk.ownership(self.root, self.limits.lock_wait_seconds, limits=self.limits) as database:
            _, manifest = self._current(database)
            self._compatible(compatibility, manifest)
            if manifest is None:
                raise ValueError("Virgin state must be explicitly initialized before load")
            prepared = _prepare.decode(self, manifest, _prepare.bound_codecs(bindings))
            _prepare.verify(prepared, bindings)
            return prepared

    def commit_transition(
        self,
        request: CommitRequest,
        bindings: _prepare.Bindings,
        transition: Callable[[StagedCollection], None],
        *,
        compatibility: ExecutionCompatibility,
    ) -> CommitReceipt:
        """Run one bounded transition or explicit seal on fresh restored owners.

        Parameters
        ----------
        request : CommitRequest
            Exact predecessor, next generation and transition/seal command.
        bindings : tuple of ComponentBinding
            Complete exact concrete current inventory.
        transition : callable
            Trusted local bounded callback, invoked once after stale rejection.
        compatibility : ExecutionCompatibility
            Actual current runtime/implementation binding, required on retries too.

        Returns
        -------
        CommitReceipt
            New acknowledgement or retained receipt without callback/rewind.

        Raises
        ------
        ValueError, TypeError, OSError, sqlite3.Error
            Stale/conflicting/expired command, capacity, relation or I/O failure.
            Callback exceptions propagate. An ambiguous commit/response failure
            requires reopening and reconciling the exact command; no body retry.
        """
        request = CommitRequest.model_validate(request)
        if request.intent.kind not in {"transition", "seal"}:
            raise ValueError("State transition requires a transition/seal intent")
        with _disk.ownership(self.root, self.limits.lock_wait_seconds, limits=self.limits) as database:
            current, manifest = self._current(database)
            self._compatible(compatibility, manifest)
            retained = self._request(database, request, current)
            if retained is not None:
                return retained
            if manifest is None or request.intent.kind not in {"transition", "seal"}:
                raise ValueError("State transition requires an accepted head and transition/seal intent")
            _garbage.reserve(self.root, database, self.limits, current)
            old = _prepare.decode(self, manifest, _prepare.bound_codecs(bindings))
            stage = _prepare.stage(old, bindings)
            try:
                transition(stage)
                values = stage._export()  # pylint: disable=protected-access
            finally:
                stage._close()  # pylint: disable=protected-access
            new = _prepare.build(manifest.definition, compatibility, request, values, bindings, self.limits)
            self._validate_successor(old, new)
            return self._publish(database, request, new)

    def migrate(
        self,
        request: MigrationRequest,
        bindings: MigrationBindings,
        migrate: Callable[[PreparedCollection], _prepare.Values],
        *,
        compatibility: ExecutionCompatibility,
    ) -> CommitReceipt:
        """Apply an explicit known typed mapping without implicit field guessing.

        Parameters
        ----------
        request : MigrationRequest
            Exact old ack, qualified mapping and complete target definition.
        bindings : MigrationBindings
            Finite known source codecs and fresh concrete target owners.
        migrate : callable
            Pure bounded mapping from all decoded source state to all targets.
        compatibility : ExecutionCompatibility
            Actual target runtime, target owner definitions and mapping identity.
            Known source codecs read historical bytes without executing old owners.

        Returns
        -------
        CommitReceipt
            Accepted target or exact retained migration acknowledgement.

        Raises
        ------
        ValueError, TypeError, OSError, sqlite3.Error
            Unknown/incompatible mapping, changed semantics, pending replay,
            stale input or failure. Old ack survives every precommit failure.

        Notes
        -----
        Direct typed mappings may preserve pending work. Replay mapping requires
        an empty-tail quiescent v2 seal and explicit bounded rebind evidence.
        """
        request = MigrationRequest.model_validate(request)
        with _disk.ownership(self.root, self.limits.lock_wait_seconds, limits=self.limits) as database:
            current, manifest = self._current(database)
            existing = database.execute(
                "SELECT 1 FROM receipts WHERE generation=?", (request.commit.intent.generation,)
            ).fetchone()
            if existing is not None:
                self._compatible(compatibility, manifest)
            else:
                self._migration_compatible(compatibility, request)
            retained = self._request(database, request.commit, current)
            if retained is not None:
                accepted = _disk.manifest(self.root, retained.accepted_head, self.limits)
                if (
                    accepted.migration is None
                    or accepted.migration.migration != request.migration
                    or accepted.definition != request.target_definition
                ):
                    raise ValueError("Retained migration request conflicts with its original target/mapping")
                return retained
            if manifest is None or current is None:
                raise ValueError("Migration requires an existing acknowledged collection")
            _disk.validate_capacity(request.target_definition, self.limits)
            _garbage.reserve(self.root, database, self.limits, current)
            old = _prepare.decode(self, manifest, bindings.old_codecs)
            values = migrate(old)
            new = _prepare.build(
                request.target_definition,
                compatibility,
                request.commit,
                values,
                bindings.new_bindings,
                self.limits,
                self._migration_receipt(request, current, manifest),
            )
            self._validate_successor(old, new)
            staged = _prepare.stage(new, bindings.new_bindings)
            staged._close()  # pylint: disable=protected-access
            return self._publish(database, request.commit, new)

    @staticmethod
    def _migration_receipt(
        request: MigrationRequest, current: AcknowledgedHead, manifest: CollectionManifest
    ) -> MigrationReceipt:
        """Retain historical source and exact known mapping as immutable evidence."""
        return MigrationReceipt(
            schema_version=1,
            migration=request.migration,
            source_head=current,
            source_definition_fingerprint=manifest.definition.fingerprint,
            target_definition_fingerprint=request.target_definition.fingerprint,
            preserved_frontiers_fingerprint=semantic_frontiers_fingerprint(manifest.frontiers),
        )

    def cleanup(self, expected: AcknowledgedHead) -> CleanupReceipt:
        """Prune unreachable owned bytes while protecting the exact current ack.

        Parameters
        ----------
        expected : AcknowledgedHead
            Exact current metadata required before receipt retirement/pruning.

        Returns
        -------
        CleanupReceipt
            Actual removed counts/bytes and independent operational receipt floor.

        Raises
        ------
        ValueError, OSError, sqlite3.Error
            Stale/corrupt head or local filesystem failure; no child floor moves.
        """
        expected = AcknowledgedHead.model_validate(expected)
        with _disk.ownership(self.root, self.limits.lock_wait_seconds, limits=self.limits) as database:
            current, _ = self._current(database)
            if current is None or current != expected:
                raise ValueError("Cleanup expected head is stale")
            floor = max(0, current.generation - self.limits.receipt_window + 1)
            database.execute("DELETE FROM receipts WHERE generation < ?", (floor,))
            _disk.commit(database, self.limits.lock_wait_seconds)
        self._barrier("cleanup_after_metadata_commit")
        with _disk.ownership(self.root, self.limits.lock_wait_seconds, limits=self.limits) as database:
            actual, _ = self._current(database)
            if actual != current:
                raise ValueError("Cleanup head changed before protected pruning")
            blobs, files, size = _garbage.prune(self.root, database, self.limits, current)
            _disk.usage(self.root, self.limits)
            return CleanupReceipt(
                schema_version=1,
                current_head=current,
                removed_blobs=blobs,
                removed_files=files,
                removed_bytes=size,
                retired_receipt_floor=floor,
            )

    def _current(self, database: sqlite3.Connection) -> tuple[AcknowledgedHead | None, CollectionManifest | None]:
        """Verify actual ack, complete current artifacts and retained metadata."""
        header = _disk.StoreHeader.from_json(_disk.acquire(self.root / _disk.HEADER, 2 * _disk.MIB))
        if header != self._header:
            raise ValueError("Immutable state header is missing, corrupt or changed")
        current = _disk.head(database)
        generations = tuple(
            row[0]
            for row in database.execute(
                "SELECT generation FROM receipts ORDER BY generation LIMIT ?", (self.limits.receipt_window + 1,)
            )
        )
        if current is None:
            if generations:
                raise ValueError("Virgin state metadata contains operational receipts")
            return None, None
        if (current.session, current.source_epoch) != (
            self._header.definition.session,
            self._header.definition.source_epoch,
        ):
            raise ValueError("Acknowledged session or epoch differs from immutable store header")
        floor = max(0, current.generation - self.limits.receipt_window + 1)
        if generations != tuple(range(floor, current.generation + 1)):
            raise ValueError("Retained state receipt metadata is missing or inconsistent")
        self._receipt_integrity(database, current)
        manifest = _disk.manifest(self.root, current, self.limits)
        _disk.validate_capacity(manifest.definition, self.limits)
        _garbage.protected(self.root, database, self.limits, current)
        _disk.usage(self.root, self.limits)
        return current, manifest

    def _receipt_integrity(self, database: sqlite3.Connection, current: AcknowledgedHead) -> None:
        """Bind retained SQL metadata to exact canonical accepted manifests."""
        lengths = database.execute(
            "SELECT length(CAST(intent AS BLOB)),length(CAST(predecessor AS BLOB)),"
            "length(CAST(head AS BLOB)) FROM receipts"
        )
        if any(size is not None and size > self.limits.maximum_manifest_bytes for row in lengths for size in row):
            raise ValueError("Retained receipt metadata exceeds its persisted byte ceiling")
        previous = None
        for generation, intent_json, predecessor_json, head_json in database.execute(
            "SELECT generation,intent,predecessor,head FROM receipts ORDER BY generation"
        ):
            intent = CommitIntent.from_json(intent_json)
            predecessor = None if predecessor_json is None else AcknowledgedHead.from_json(predecessor_json)
            accepted = AcknowledgedHead.from_json(head_json)
            manifest = _disk.manifest(self.root, accepted, self.limits)
            if (intent.to_json(), accepted.to_json(), None if predecessor is None else predecessor.to_json()) != (
                intent_json,
                head_json,
                predecessor_json,
            ):
                raise ValueError("Retained receipt metadata must use exact canonical representations")
            actual = (generation, generation, intent, predecessor, accepted.session, accepted.source_epoch)
            expected = (
                intent.generation,
                accepted.generation,
                manifest.intent,
                manifest.predecessor,
                current.session,
                current.source_epoch,
            )
            if actual != expected or (previous is not None and predecessor != previous):
                raise ValueError("Retained receipt contradicts its exact accepted manifest or predecessor")
            previous = accepted
        if previous != current:
            raise ValueError("Current state acknowledgement differs from its retained receipt")

    def _compatible(self, actual: ExecutionCompatibility, manifest: CollectionManifest | None) -> None:
        """Check explicit actual execution binding before operational receipts."""
        actual = ExecutionCompatibility.model_validate(actual)
        expected = self._header.compatibility if manifest is None else manifest.compatibility
        if actual != expected:
            raise ValueError("State execution compatibility differs; use a qualified explicit migration environment")

    @staticmethod
    def _migration_compatible(actual: ExecutionCompatibility, request: MigrationRequest) -> None:
        """Admit known old readers without pretending old executable code runs."""
        actual = ExecutionCompatibility.model_validate(actual)
        required = (request.migration,) + tuple(
            item.binding.implementation for item in request.target_definition.components
        )
        if any(item not in actual.owner_implementations for item in required):
            raise ValueError("Migration environment must pin the mapping and every actual target owner implementation")

    def _request(
        self, database: sqlite3.Connection, request: CommitRequest, current: AcknowledgedHead | None
    ) -> CommitReceipt | None:
        """Return exact retained intent before stale rejection, never rewinding."""
        row = database.execute(
            "SELECT intent,predecessor,head FROM receipts WHERE generation=?", (request.intent.generation,)
        ).fetchone()
        if row is not None:
            predecessor = None if row[1] is None else AcknowledgedHead.from_json(row[1])
            if row[0] != request.intent.to_json() or predecessor != request.expected:
                raise ValueError("Publication identity conflicts with its retained intent or original predecessor")
            if current is None:
                raise ValueError("Retained receipt has no actual current acknowledgement")
            return CommitReceipt(
                schema_version=1,
                status="retained",
                accepted_head=AcknowledgedHead.from_json(row[2]),
                current_head=current,
            )
        if request.expected != current:
            raise ValueError("Expected state head is stale or its operational receipt has expired")
        expected_generation = 0 if current is None else current.generation + 1
        if request.intent.generation != expected_generation:
            raise ValueError("State publication requires the next consecutive generation")
        return None

    def _publish(
        self, database: sqlite3.Connection, request: CommitRequest, prepared: PreparedCollection
    ) -> CommitReceipt:
        """Publish complete bytes then atomically accept head, receipt and floor."""
        self._barrier("before_blobs")
        content = _prepare.encoded(prepared.manifest.definition, prepared.values, self.limits)
        for index, item in enumerate(content):
            _disk.publish(self.root, "blobs", item.content, self.limits.maximum_component_bytes)
            self._barrier("after_blob", component=index)
        _disk.publish(
            self.root, "manifests", prepared.manifest.to_json().encode("utf-8"), self.limits.maximum_manifest_bytes
        )
        self._barrier("after_manifest")
        _garbage.candidate_limits(self.root, database, self.limits)
        accepted = _disk.make_head(prepared.manifest)
        database.execute("UPDATE state SET ever_initialized=1,head=? WHERE singleton=1", (accepted.to_json(),))
        database.execute(
            "INSERT INTO receipts VALUES (?,?,?,?)",
            (
                request.intent.generation,
                request.intent.to_json(),
                None if request.expected is None else request.expected.to_json(),
                accepted.to_json(),
            ),
        )
        floor = max(0, accepted.generation - self.limits.receipt_window + 1)
        database.execute("DELETE FROM receipts WHERE generation < ?", (floor,))
        self._barrier("after_head_update")
        _disk.usage(self.root, self.limits)
        _disk.commit(database, self.limits.lock_wait_seconds)
        self._barrier("after_commit")
        return CommitReceipt(schema_version=1, status="new", accepted_head=accepted, current_head=accepted)

    @staticmethod
    def _validate_successor(old: PreparedCollection, new: PreparedCollection) -> None:
        """Delegate exact standard replay relations without importing transports."""
        from coleman.state._relations import validate_successor  # pylint: disable=import-outside-toplevel

        migration = new.manifest.migration
        validate_successor(
            old, new, new.manifest.intent.kind, migration=None if migration is None else migration.migration
        )

    def _barrier(self, phase: str, *, component: int | None = None) -> None:
        """Expose a private no-op process-fault observation point for qualification."""
