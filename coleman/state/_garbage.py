"""Private reachability cleanup under the same acknowledgement ownership."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from coleman.state import _disk
from coleman.state.storage_contracts import AcknowledgedHead, StorageLimits


def recognized(path: Path) -> bool:
    """Recognize the adapter's reserved flat artifact/temporary names.

    Parameters
    ----------
    path : pathlib.Path
        Candidate under blobs or manifests, inspected without opening the file.

    Returns
    -------
    bool
        True only for64-lowercase-hex JSON names or reserved tmp+eight-character
        lowercase/digit/underscore temporary names ending in .tmp.

    Notes
    -----
    Pure filename classification, not ownership authentication. Foreign names
    such as notes.tmp remain false; callers keep the reserved namespace exclusive.
    """
    if path.parent.name not in {"blobs", "manifests"}:
        return False
    if re.fullmatch(r"tmp[a-z0-9_]{8}\.tmp", path.name):
        return True
    return path.suffix == ".json" and len(path.stem) == 64 and all(c in "0123456789abcdef" for c in path.stem)


def protected(
    root: Path, database: sqlite3.Connection, limits: StorageLimits, current: AcknowledgedHead | None
) -> set[Path]:
    """Pin current and retained-receipt manifests and every referenced blob.

    Parameters
    ----------
    root : pathlib.Path
        Trusted root under the same exclusive acknowledgement ownership boundary.
    database : sqlite3.Connection
        Owned transaction with validated retained receipt metadata.
    limits : StorageLimits
        Persisted acquisition bounds for manifests and components.
    current : AcknowledgedHead or None
        Actual current head, or None for consistent virgin metadata.

    Returns
    -------
    set of pathlib.Path
        Exact manifest/blob paths that cleanup must preserve.

    Raises
    ------
    ValueError, TypeError, UnicodeError, OSError, sqlite3.Error
        Corrupt/missing/oversized protected evidence or database query failure.

    Notes
    -----
    Reads and verifies bounded complete bytes. No owner is restored and no file
    is deleted; a missing acknowledged file never triggers older-state fallback.
    """
    heads = [AcknowledgedHead.from_json(row[0]) for row in database.execute("SELECT head FROM receipts")]
    if current is not None:
        heads.append(current)
    result: set[Path] = set()
    for acknowledged in heads:
        manifest = _disk.manifest(root, acknowledged, limits)
        result.add(_disk.artifact(root, "manifests", acknowledged.manifest_fingerprint))
        for item in manifest.components:
            path = _disk.artifact(root, "blobs", item.fingerprint)
            content = _disk.acquire(
                path, min(item.byte_count, limits.maximum_component_bytes), expected=item.fingerprint
            )
            if len(content) != item.byte_count:
                raise ValueError("Protected acknowledged component byte count differs")
            result.add(path)
    return result


def prune(
    root: Path, database: sqlite3.Connection, limits: StorageLimits, current: AcknowledgedHead | None
) -> tuple[int, int, int]:
    """Remove only unreachable reserved files under collection ownership.

    Parameters
    ----------
    root : pathlib.Path
        Trusted flat root, with no concurrent adapter reader/writer outside ownership.
    database : sqlite3.Connection
        Owned validated acknowledgement transaction.
    limits : StorageLimits
        Bounded file enumeration/acquisition profile.
    current : AcknowledgedHead or None
        Actual current protected head, or consistent virgin metadata.

    Returns
    -------
    tuple of int
        Removed blob count, total removed-file count and logical removed bytes.

    Raises
    ------
    ValueError, TypeError, UnicodeError, OSError, sqlite3.Error
        Invalid layout, protected corruption, query/stat or unlink failure.

    Notes
    -----
    Unlinks only unreachable reserved names; foreign files remain. It changes
    neither acknowledged metadata nor child floors. Interrupted pruning leaves
    possible garbage, not permission to adopt an unacknowledged candidate.
    """
    pinned = protected(root, database, limits, current)
    blobs = files = size = 0
    for path in _disk.owned_files(root, limits):
        if path not in pinned and path.is_file() and recognized(path):
            byte_count = path.stat().st_size
            path.unlink()
            files += 1
            size += byte_count
            blobs += int(path.parent.name == "blobs" and path.suffix == ".json")
    return blobs, files, size


def reserve(root: Path, database: sqlite3.Connection, limits: StorageLimits, current: AcknowledgedHead | None) -> None:
    """Prune safe garbage and check candidate/temp quota before callbacks.

    Parameters
    ----------
    root : pathlib.Path
        Trusted flat adapter root under exclusive ownership.
    database : sqlite3.Connection
        Owned transaction containing current validated metadata.
    limits : StorageLimits
        Persisted byte/entry limits and worst-case next-candidate capacities.
    current : AcknowledgedHead or None
        Actual protected head, or consistent virgin metadata.

    Raises
    ------
    ValueError, TypeError, UnicodeError, OSError, sqlite3.Error
        Invalid protected data/layout, cleanup failure or insufficient headroom.

    Notes
    -----
    May delete unreachable reserved bytes, never referenced/current bytes or
    foreign files. The check is not disk preallocation or free-space assurance.
    No component factory/restore/transition/RNG callback runs here.
    """
    prune(root, database, limits, current)
    size, entries = _disk.usage(root, limits)
    candidate = limits.maximum_collection_bytes + limits.maximum_manifest_bytes + limits.maximum_component_bytes
    if size + candidate + 16 * _disk.MIB > limits.maximum_root_bytes:
        raise ValueError("Insufficient persisted root byte headroom before state callback")
    if entries + limits.maximum_components + 2 > limits.maximum_entries:
        raise ValueError("Insufficient persisted entry headroom before state callback")


def candidate_limits(root: Path, database: sqlite3.Connection, limits: StorageLimits) -> None:
    """Check actual unacknowledged publication size before accepting progress.

    Parameters
    ----------
    root : pathlib.Path
        Trusted owned root after complete candidate publication.
    database : sqlite3.Connection
        Current owned transaction before candidate head/receipt COMMIT.
    limits : StorageLimits
        Orphan byte/entry and total root/file ceilings.

    Raises
    ------
    ValueError, TypeError, UnicodeError, OSError, sqlite3.Error
        Corrupt protected evidence, invalid layout or capacity overflow.

    Notes
    -----
    Counts recognized unreferenced candidate/temp bytes without deleting them.
    A refusal leaves the old acknowledged head valid; bytes alone are not acceptance.
    """
    current = _disk.head(database)
    pinned = protected(root, database, limits, current)
    paths = _disk.owned_files(root, limits)
    orphan = [path for path in paths if path.is_file() and recognized(path) and path not in pinned]
    if len(orphan) > limits.maximum_orphan_entries:
        raise ValueError("Unacknowledged state exceeds its persisted orphan entry ceiling")
    if sum(path.stat().st_size for path in orphan) > limits.maximum_orphan_bytes:
        raise ValueError("Unacknowledged state exceeds its persisted orphan byte ceiling")
    _disk.usage(root, limits)
