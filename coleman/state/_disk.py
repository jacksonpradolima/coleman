"""Private bounded local publication and SQLite ownership primitives."""

from __future__ import annotations

import hashlib
import math
import sqlite3
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.core.base import CoreContract
from coleman.core.reading import bounded_json_bytes
from coleman.state.storage_contracts import (
    MIB,
    AcknowledgedHead,
    CollectionDefinition,
    CollectionManifest,
    ExecutionCompatibility,
    StorageLimits,
)

PROFILE = "local-sqlite-state-v1"
DATABASE = "acknowledgements.sqlite3"
HEADER = "store.json"


class StoreHeader(CoreContract):
    """Retain immutable creation metadata independently of the current manifest.

    Attributes
    ----------
    profile : str
        Exact local adapter profile, checked when opening the root.
    definition : CollectionDefinition
        Initial complete owner inventory, used only until generation zero exists.
    compatibility : ExecutionCompatibility
        Explicit initial environment and copied-owner bindings, not ambient capture.
    limits : StorageLimits
        Persisted byte, entry, receipt and acquisition bounds.
    sqlite_version : str
        Actually linked SQLite version at explicit creation.

    Notes
    -----
    Schema1 is declarative. Construction does not open a database or prove that
    current executable code matches this historical creation environment.
    """

    profile: str
    definition: CollectionDefinition
    compatibility: ExecutionCompatibility
    limits: StorageLimits
    sqlite_version: str


def qualify_sqlite() -> None:
    """Check the linked engine for the qualified WAL-reset fix.

    Raises
    ------
    ValueError
        SQLite is older than3.51.3 without a3.50.7+/3.44.6+ backport in those lines.

    Notes
    -----
    Reads standard-library version metadata only; opens no database, calls no
    owner and does not certify filesystem or power-loss behavior.
    """
    version = sqlite3.sqlite_version_info
    fixed = version >= (3, 51, 3) or ((3, 50, 7) <= version < (3, 51, 0))
    fixed = fixed or (3, 44, 6) <= version < (3, 45, 0)
    if not fixed:
        raise ValueError("Local WAL state requires SQLite3.51.3+ or fixed3.50.7+/3.44.6+ backport lines")


def digest(content: bytes) -> str:
    """Hash the complete supplied bytes using COLEMAN's digest spelling.

    Parameters
    ----------
    content : bytes
        Exact immutable encoded content, including any framing supplied by caller.

    Returns
    -------
    str
        SHA-256 with the sha256 prefix; no decoding, I/O or authentication occurs.
    """
    return "sha256:" + hashlib.sha256(content).hexdigest()


def artifact(root: Path, family: str, fingerprint: str) -> Path:
    """Derive a flat owned artifact path from a validated SHA-256 digest.

    Parameters
    ----------
    root : pathlib.Path
        Trusted absolute adapter root; no directory is created here.
    family : {'blobs', 'manifests'}
        Closed destination namespace.
    fingerprint : str
        Exact sha256 prefix and64 lowercase hexadecimal characters.

    Returns
    -------
    pathlib.Path
        Digest-named JSON path under the selected family.

    Raises
    ------
    ValueError
        Digest spelling or family is invalid. No filesystem lookup occurs.
    """
    if not fingerprint.startswith("sha256:") or len(fingerprint) != 71:
        raise ValueError("State artifact requires a SHA-256 digest")
    suffix = fingerprint[7:]
    if any(character not in "0123456789abcdef" for character in suffix) or family not in {"blobs", "manifests"}:
        raise ValueError("Invalid state artifact digest or family")
    return root / family / (suffix + ".json")


def acquire(path: Path, maximum: int, *, expected: str | None = None) -> bytes:
    """Bound file acquisition before validating UTF-8 content and digest.

    Parameters
    ----------
    path : pathlib.Path
        Existing regular owned file; symlinks are rejected.
    maximum : int
        Persisted positive encoded-byte ceiling, not a character count.
    expected : str or None, optional
        Exact raw-content digest, or None for separately validated header content.

    Returns
    -------
    bytes
        Complete content after byte/depth/UTF-8 and optional digest checks.

    Raises
    ------
    ValueError, TypeError, UnicodeError
        Invalid bounds/content, missing regular file or inconsistent digest.
    OSError
        File acquisition fails. Previously accepted bytes are never modified.

    Notes
    -----
    Reads at most maximum+1 bytes on the trusted local filesystem. It does not
    decode a model, migrate state or restore an owner.
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError("Acknowledged state file is missing or is not a regular owned file")
    with path.open("rb") as stream:
        content = stream.read(maximum + 1)
    if len(content) > maximum:
        raise ValueError("State artifact exceeds its persisted byte ceiling")
    bounded_json_bytes(content, maximum)
    if expected is not None and digest(content) != expected:
        raise ValueError("Acknowledged state artifact digest mismatch")
    return content


def publish(root: Path, family: str, content: bytes, maximum: int) -> str:
    """Publish complete immutable content without acknowledging progress.

    Parameters
    ----------
    root : pathlib.Path
        Trusted owned adapter root.
    family : {'blobs', 'manifests'}
        Closed flat content-addressed namespace.
    content : bytes
        Complete canonical encoded component or manifest.
    maximum : int
        Persisted maximum encoded bytes for this artifact.

    Returns
    -------
    str
        Exact raw-content SHA-256 used to address the artifact.

    Raises
    ------
    ValueError, TypeError
        Invalid namespace/bound, oversized bytes or conflicting immutable content.
    OSError
        Publication/link/fsync fails; no SQLite acknowledgement is created.

    Notes
    -----
    Uses ArtifactWriter's unique temporary file, fsync and immutable link. Equal
    bytes converge. A crash may leave orphan bytes; acceptance requires COMMIT.
    """
    fingerprint = digest(content)
    ArtifactWriter(root).write_immutable_stream(artifact(root, family, fingerprint), (content,), maximum_bytes=maximum)
    return fingerprint


def validate_capacity(definition: CollectionDefinition, limits: StorageLimits) -> None:
    """Validate inventory and simultaneous receipt/candidate headroom.

    Parameters
    ----------
    definition : CollectionDefinition
        Complete current or explicitly proposed target component inventory.
    limits : StorageLimits
        Persisted byte/entry bounds and retained receipt window.

    Raises
    ------
    ValueError
        Inventory exceeds its ceilings or quotas cannot hold retained roots,
        one candidate, temporary bytes and metadata reserve.

    Notes
    -----
    Pure arithmetic only; this neither reserves disk space nor scans free space.
    """
    if len(definition.children) > limits.maximum_children or len(definition.components) > limits.maximum_components:
        raise ValueError("State inventory exceeds its persisted component or child ceiling")
    byte_floor = (
        (limits.receipt_window + 1) * (limits.maximum_collection_bytes + limits.maximum_manifest_bytes)
        + limits.maximum_orphan_bytes
        + 2 * limits.maximum_component_bytes
        + 16 * MIB
    )
    entry_floor = (limits.receipt_window + 1) * (len(definition.components) + 1) + limits.maximum_orphan_entries + 16
    if limits.maximum_root_bytes < byte_floor or limits.maximum_entries < entry_floor:
        raise ValueError("State limits cannot reserve retained receipts, candidate, temporary and metadata headroom")


def initialize_database(root: Path) -> None:
    """Create virgin acknowledgement tables with verified WAL/FULL.

    Parameters
    ----------
    root : pathlib.Path
        Explicitly created empty adapter root with creation header already written.

    Raises
    ------
    ValueError, TimeoutError, sqlite3.Error
        Unsupported engine/profile, exhausted acquisition budget or schema/I/O error.

    Notes
    -----
    Creates the database and tables with ever_initialized=false and no head.
    No learner is initialized and no logical generation is acknowledged.
    """
    with connection(root, 5.0, create=True) as database:
        database.execute("PRAGMA user_version=1")
        database.execute(
            "CREATE TABLE state (singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
            "ever_initialized INTEGER NOT NULL, head TEXT)"
        )
        database.execute("INSERT INTO state VALUES (1,0,NULL)")
        database.execute(
            "CREATE TABLE receipts (generation INTEGER PRIMARY KEY, intent TEXT NOT NULL, "
            "predecessor TEXT, head TEXT NOT NULL)"
        )


@contextmanager
def connection(root: Path, wait: float, *, create: bool = False) -> Iterator[sqlite3.Connection]:
    """Open and close a qualified SQLite connection without callback retries.

    Parameters
    ----------
    root : pathlib.Path
        Trusted absolute root containing the adapter database.
    wait : float
        Positive finite seconds available for connection/profile acquisition.
    create : bool, optional
        True only during explicit virgin creation; False uses URI mode rw and
        refuses a deleted database rather than creating fresh metadata.

    Yields
    ------
    sqlite3.Connection
        Autocommit-mode connection with WAL/FULL set and read back.

    Raises
    ------
    ValueError, TimeoutError, sqlite3.Error
        Symlink/unsupported engine, invalid profile or exhausted acquisition budget.

    Notes
    -----
    Changes connection pragmas, performs database I/O and always closes the handle.
    It invokes no component, restore, transition or RNG callback.
    """
    qualify_sqlite()
    path = root / DATABASE
    if path.is_symlink():
        raise ValueError("State acknowledgement database cannot be a symlink")
    mode = "rwc" if create else "rw"
    deadline = time.monotonic() + wait
    database = sqlite3.connect(path.as_uri() + "?mode=" + mode, timeout=0, isolation_level=None, uri=True)
    try:
        _busy(database, deadline)
        mode = database.execute("PRAGMA journal_mode=WAL").fetchone()
        _busy(database, deadline)
        database.execute("PRAGMA synchronous=FULL")
        synchronous = database.execute("PRAGMA synchronous").fetchone()
        if mode != ("wal",) or synchronous != (2,):
            raise ValueError("State store requires verified SQLite WAL plus FULL synchronization")
        yield database
    finally:
        database.close()


@contextmanager
def ownership(root: Path, wait: float, *, limits: StorageLimits) -> Iterator[sqlite3.Connection]:
    """Own one writer transaction before head-dependent component work.

    Parameters
    ----------
    root : pathlib.Path
        Trusted existing adapter root.
    wait : float
        Finite positive seconds shared across acquisition steps using a monotonic
        deadline; this does not preempt callback CPU time.
    limits : StorageLimits
        Persisted root capacities, checked before SQLite opens/scans metadata
        and again under ownership before its integrity check.

    Yields
    ------
    sqlite3.Connection
        BEGIN IMMEDIATE transaction after schema-version/integrity checks.

    Raises
    ------
    ValueError, TimeoutError, sqlite3.Error
        Unsupported/corrupt metadata or acquisition failure before user callbacks.

    Notes
    -----
    Caller controls the sole COMMIT. Any still-active transaction rolls back on
    exit, then the connection closes; the body is never automatically retried.
    """
    deadline = time.monotonic() + wait
    _database_size(root, limits)
    with connection(root, wait) as database:
        _busy(database, deadline)
        database.execute("BEGIN IMMEDIATE")
        try:
            usage(root, limits)
            if database.execute("PRAGMA user_version").fetchone() != (1,):
                raise ValueError("Unknown local acknowledgement database version")
            if database.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("State acknowledgement database is corrupt")
            yield database
        finally:
            if database.in_transaction:
                database.rollback()


def _database_size(root: Path, limits: StorageLimits) -> None:
    """Bound SQLite page/WAL acquisition before opening or integrity scanning.

    A cooperative connection may remove a sidecar before acquisition; stat
    without following links and tolerate that disappearance. Full root usage
    is checked only under ownership, avoiding volatile temporary-file races.
    """
    total = 0
    for name in (DATABASE, DATABASE + "-wal", DATABASE + "-shm"):
        try:
            metadata = (root / name).stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("State metadata must consist of regular owned files")
        total += metadata.st_size
    if total > limits.maximum_root_bytes:
        raise ValueError("SQLite metadata exceeds the persisted root byte ceiling before acquisition")


def commit(database: sqlite3.Connection, wait: float) -> None:
    """Retry only COMMIT contention without repeating the transition body.

    Parameters
    ----------
    database : sqlite3.Connection
        Exclusively owned active transaction containing the complete candidate ack.
    wait : float
        Positive finite seconds for this commit's monotonic contention budget.

    Raises
    ------
    TimeoutError, sqlite3.Error
        Deadline expires, noncontention error occurs or the transaction is no
        longer active. Caller must reconcile an ambiguous outcome before retrying.

    Notes
    -----
    Only numeric BUSY/LOCKED results may retry the same COMMIT. Success is the
    sole accepted-progress point; no component callback or publication is repeated.
    """
    deadline = time.monotonic() + wait
    while True:
        try:
            _busy(database, deadline)
            database.execute("COMMIT")
            return
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                raise
            if not database.in_transaction or time.monotonic() >= deadline:
                raise
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


def _busy(database: sqlite3.Connection, deadline: float) -> None:
    """Apply the remaining finite monotonic budget to this connection.

    Parameters
    ----------
    database : sqlite3.Connection
        Current owned connection; its busy_timeout pragma is updated.
    deadline : float
        Absolute monotonic seconds, not wall-clock time.

    Raises
    ------
    TimeoutError
        No positive allowance remains before the next SQLite operation.
    sqlite3.Error
        Updating the bounded millisecond timeout fails.

    Notes
    -----
    Rounds the remaining allowance to SQLite milliseconds within its integer
    range. Performs no learner call and does not renew the original deadline.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Local state ownership/commit contention exceeded its finite wait allowance")
    milliseconds = min(2**31 - 1, math.ceil(min(remaining, 2_147_483.647) * 1000))
    database.execute(f"PRAGMA busy_timeout={milliseconds}")


def head(database: sqlite3.Connection) -> AcknowledgedHead | None:
    """Read bounded acknowledged metadata without treating corruption as origin.

    Parameters
    ----------
    database : sqlite3.Connection
        Owned schema1 database, before any owner restoration.

    Returns
    -------
    AcknowledgedHead or None
        Actual parsed head, or None only for consistent virgin metadata.

    Raises
    ------
    ValueError, sqlite3.Error
        Missing/oversized/contradictory metadata, invalid wire or query failure.

    Notes
    -----
    Checks encoded head length before loading its TEXT. This metadata read does
    not restore an owner or independently verify referenced artifacts.
    """
    size = database.execute("SELECT length(CAST(head AS BLOB)) FROM state WHERE singleton=1").fetchone()
    if size is None or (size[0] is not None and size[0] > 2 * MIB):
        raise ValueError("State acknowledgement metadata is missing or oversized")
    row = database.execute("SELECT ever_initialized,head FROM state WHERE singleton=1").fetchone()
    if row is None or row[0] not in {0, 1} or (row[0] == 1 and row[1] is None):
        raise ValueError("State acknowledgement metadata is missing or corrupt")
    if row[0] == 0 and row[1] is not None:
        raise ValueError("Virgin state metadata contains a contradictory acknowledged head")
    return None if row[1] is None else AcknowledgedHead.from_json(row[1])


def manifest(root: Path, acknowledged: AcknowledgedHead, limits: StorageLimits) -> CollectionManifest:
    """Read the exact acknowledged canonical manifest without fallback.

    Parameters
    ----------
    root : pathlib.Path
        Trusted existing adapter root under the caller's ownership boundary.
    acknowledged : AcknowledgedHead
        Exact head selected by the actual acknowledgement database.
    limits : StorageLimits
        Persisted manifest encoded-byte ceiling.

    Returns
    -------
    CollectionManifest
        Canonical validated manifest matching every field of the supplied head.

    Raises
    ------
    ValueError, TypeError, UnicodeError, OSError
        Missing/oversized/corrupt/noncanonical bytes or contradictory head binding.

    Notes
    -----
    Reads no component payload and invokes no owner. An old valid manifest is
    not selected merely because the acknowledged current file is unavailable.
    """
    content = acquire(
        artifact(root, "manifests", acknowledged.manifest_fingerprint),
        limits.maximum_manifest_bytes,
        expected=acknowledged.manifest_fingerprint,
    )
    result = CollectionManifest.from_json(content)
    if result.to_json().encode("utf-8") != content or make_head(result) != acknowledged:
        raise ValueError("Acknowledged manifest is noncanonical or contradicts its head")
    if any(item.byte_count > limits.maximum_component_bytes for item in result.components):
        raise ValueError("Acknowledged component exceeds its persisted byte ceiling")
    if sum(item.byte_count for item in result.components) > limits.maximum_collection_bytes:
        raise ValueError("Acknowledged collection exceeds its persisted byte ceiling")
    return result


def make_head(value: CollectionManifest) -> AcknowledgedHead:
    """Derive complete acknowledgement metadata from canonical manifest data.

    Parameters
    ----------
    value : CollectionManifest
        Complete validated candidate or acknowledged manifest.

    Returns
    -------
    AcknowledgedHead
        Exact session/epoch, generation and definition/manifest/intent digests.

    Notes
    -----
    Pure derivation only. Creating this value does not publish files or COMMIT.
    """
    return AcknowledgedHead(
        schema_version=1,
        session=value.definition.session,
        source_epoch=value.definition.source_epoch,
        definition_fingerprint=value.definition.fingerprint,
        generation=value.intent.generation,
        manifest_fingerprint=value.fingerprint,
        intent_fingerprint=value.intent.fingerprint,
    )


def owned_files(root: Path, limits: StorageLimits) -> tuple[Path, ...]:
    """Enumerate the trusted flat root within its persisted entry bound.

    Parameters
    ----------
    root : pathlib.Path
        Existing owned root, inspected while collection ownership is held.
    limits : StorageLimits
        Maximum counted entries, including fixed directories and foreign files.

    Returns
    -------
    tuple of pathlib.Path
        Root entries and flat blob/manifest files without recursive traversal.

    Raises
    ------
    ValueError
        Symlink, foreign/nested directory, nonregular artifact or entry overflow.
    OSError
        Enumeration/stat fails. This function never removes any path.

    Notes
    -----
    Counting a foreign regular file does not make it eligible for cleanup.
    """
    result: list[Path] = []
    for path in root.iterdir():
        result.append(path)
        if path.is_symlink():
            raise ValueError("State roots cannot contain symlinks")
        if path.is_dir():
            if path.name not in {"blobs", "manifests"}:
                raise ValueError("State root contains a foreign directory")
            for child in path.iterdir():
                if child.is_symlink() or not child.is_file():
                    raise ValueError("State artifact directories must remain flat regular files")
                result.append(child)
                if len(result) > limits.maximum_entries:
                    raise ValueError("State root exceeds its persisted entry ceiling")
        if len(result) > limits.maximum_entries:
            raise ValueError("State root exceeds its persisted entry ceiling")
    return tuple(result)


def usage(root: Path, limits: StorageLimits) -> tuple[int, int]:
    """Measure logical regular-file bytes and bounded root entries.

    Parameters
    ----------
    root : pathlib.Path
        Existing trusted root under collection ownership.
    limits : StorageLimits
        Persisted entry and total encoded-file byte ceilings.

    Returns
    -------
    tuple of int
        Sum of regular-file sizes in bytes and number of counted entries; includes
        DB/WAL/SHM, temporary/orphan bytes and foreign regular files.

    Raises
    ------
    ValueError, OSError
        Invalid layout, capacity overflow or enumeration/stat failure.

    Notes
    -----
    Measures logical file sizes, not allocated disk blocks, free space or RSS.
    Reads metadata only and does not acknowledge or remove anything.
    """
    paths = owned_files(root, limits)
    size = sum(path.stat().st_size for path in paths if path.is_file())
    if size > limits.maximum_root_bytes:
        raise ValueError("State root exceeds its persisted total byte ceiling")
    return size, len(paths)
