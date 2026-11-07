"""Coordinate real DirEntry disappearance without replacing scanner or writer.

Only os.scandir is narrowly wrapped to schedule cleanup of actual temporary
files between enumeration and the scanner's no-follow stat. Every yielded entry
is a real os.DirEntry; ArtifactWriter performs its actual stream/link/cleanup.
No receipt is fabricated or silently discarded by a mocked principal component.
"""

import os
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.evaluation.design._storage import (
    AuditCapacity,
    ProtocolCapacityError,
    _directory_view,
    bounded_paths,
)


def _capacity():
    return AuditCapacity(
        receipts=2,
        receipt_bytes=128,
        freeze_bytes=65536,
        report_bytes=2 * 1024**2,
        directory_entries=4,
        directory_bytes=512,
        configuration_bytes=4096,
    )


class _CoordinatedScan:
    """Yield actual entries after one explicitly scheduled real cleanup."""

    def __init__(self, iterator, disappearing, cleanup):
        self.iterator = iterator
        self.disappearing = disappearing
        self.cleanup = cleanup

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.iterator.close()

    def __iter__(self):
        entries = tuple(self.iterator)
        # The temp is enumerated first, so its reservation must still affect
        # subsequent byte/entry guards after the real writer removes it.
        for entry in sorted(entries, key=lambda item: (item.name != self.disappearing.name, item.name)):
            assert isinstance(entry, os.DirEntry)
            if entry.name == self.disappearing.name:
                self.cleanup()
                assert not self.disappearing.exists()
            yield entry


def _cleanup_after_enumeration(monkeypatch, directory, disappearing, cleanup):
    """Wrap only the selected scandir; principal scanner and real entries remain."""
    actual_scandir = os.scandir

    def scandir(path):
        listing = actual_scandir(path)
        if Path(path) == directory:
            return _CoordinatedScan(listing, disappearing, cleanup)
        return listing

    monkeypatch.setattr(os, "scandir", scandir)


@contextmanager
def _actual_writer_temp(directory):
    """Pause an actual immutable writer while its uncommitted temp exists."""
    entered = threading.Event()
    release = threading.Event()
    failures = []
    final = directory / "000000000002.json"
    writer = ArtifactWriter(directory)

    def chunks():
        entered.set()
        if not release.wait(10):
            raise TimeoutError("The coordinated scanner did not release the actual writer")
        yield b"{}"

    def publish():
        try:
            writer.write_immutable_stream(final, chunks(), maximum_bytes=128)
        except BaseException as error:  # Retain thread failures for the owning test.
            failures.append(error)

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    try:
        assert entered.wait(10), "Actual writer did not reach its stream boundary"
        temporaries = tuple(directory.glob("*.tmp"))
        assert len(temporaries) == 1
        temporary = temporaries[0]

        def complete():
            release.set()
            thread.join(10)
            assert not thread.is_alive()
            assert not failures, failures
            assert final.read_bytes() == b"{}"

        yield temporary, complete
    finally:
        release.set()
        thread.join(10)
        assert not thread.is_alive()
        assert not failures, failures


@pytest.mark.parametrize("api", ["directory_view", "bounded_paths"])
def test_actual_writer_temp_cleanup_is_tolerated_with_conservative_reservation(tmp_path, monkeypatch, api):
    receipt = tmp_path / "000000000001.json"
    receipt.write_bytes(b"{}")
    with _actual_writer_temp(tmp_path) as (temporary, complete):
        _cleanup_after_enumeration(monkeypatch, tmp_path, temporary, complete)
        if api == "directory_view":
            view = _directory_view(tmp_path, _capacity())
            assert view.paths == (receipt,)
            assert view.entries == 2
            assert view.byte_count == len(b"{}") + _capacity().receipt_bytes
        else:
            assert bounded_paths(tmp_path, _capacity()) == [receipt]
    # The scan was enumerated before this real final link appeared. A later
    # scan can discover it; the earlier bounded snapshot is not rewritten.
    assert (tmp_path / "000000000002.json").read_bytes() == b"{}"


def test_vanished_writer_temp_reservation_still_exhausts_actual_byte_budget(tmp_path, monkeypatch):
    (tmp_path / "000000000001.json").write_bytes(b"{}")
    (tmp_path / "foreign.bin").write_bytes(b"x" * 385)
    with _actual_writer_temp(tmp_path) as (temporary, complete):
        _cleanup_after_enumeration(monkeypatch, tmp_path, temporary, complete)
        with pytest.raises(ProtocolCapacityError, match="byte capacity"):
            bounded_paths(tmp_path, _capacity())


def test_vanished_writer_temp_still_consumes_an_entry_before_later_count_guard(tmp_path, monkeypatch):
    (tmp_path / "000000000001.json").write_bytes(b"{}")
    for number in range(3):
        (tmp_path / f"foreign-{number}.bin").write_bytes(b"")
    with _actual_writer_temp(tmp_path) as (temporary, complete):
        _cleanup_after_enumeration(monkeypatch, tmp_path, temporary, complete)
        with pytest.raises(ProtocolCapacityError, match="entry capacity"):
            bounded_paths(tmp_path, _capacity())


@pytest.mark.parametrize("name", ["000000000001.json", "foreign.bin"])
def test_disappearing_non_temporary_entries_are_never_hidden(tmp_path, monkeypatch, name):
    disappearing = tmp_path / name
    disappearing.write_bytes(b"{}")
    _cleanup_after_enumeration(monkeypatch, tmp_path, disappearing, disappearing.unlink)
    with pytest.raises(FileNotFoundError):
        bounded_paths(tmp_path, _capacity())


@pytest.mark.parametrize("shape", ["symlink", "directory"])
def test_stable_temporary_symlinks_and_directories_remain_unsupported(tmp_path, shape):
    directory = tmp_path / "scan"
    directory.mkdir()
    temporary = directory / "foreign.tmp"
    if shape == "symlink":
        target = tmp_path / "target"
        target.write_bytes(b"{}")
        temporary.symlink_to(target)
    else:
        temporary.mkdir()
    with pytest.raises(OSError, match="regular files"):
        bounded_paths(directory, _capacity())


@pytest.mark.parametrize("name", ["present.tmp", "foreign.bin"])
@pytest.mark.parametrize("size", [7, 257])
def test_present_foreign_and_temporary_files_charge_exact_actual_bytes(tmp_path, name, size):
    (tmp_path / name).write_bytes(b"x" * size)
    view = _directory_view(tmp_path, _capacity())
    assert view.paths == ()
    assert view.entries == 1
    assert view.byte_count == size


@pytest.mark.parametrize("name", ["present.tmp", "foreign.bin"])
def test_present_foreign_and_temporary_files_keep_actual_byte_guard(tmp_path, name):
    (tmp_path / name).write_bytes(b"x" * 513)
    with pytest.raises(ProtocolCapacityError, match="byte capacity"):
        bounded_paths(tmp_path, _capacity())


def test_canonical_receipt_count_guard_remains_in_force(tmp_path):
    for number in range(3):
        (tmp_path / f"{number + 1:012d}.json").write_bytes(b"{}")
    with pytest.raises(ProtocolCapacityError, match="history capacity"):
        bounded_paths(tmp_path, _capacity())


def test_stable_foreign_entries_keep_the_actual_entry_guard(tmp_path):
    for number in range(5):
        (tmp_path / f"foreign-{number}.bin").write_bytes(b"")
    with pytest.raises(ProtocolCapacityError, match="entry capacity"):
        bounded_paths(tmp_path, _capacity())
