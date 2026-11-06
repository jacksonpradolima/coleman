"""Explicit outer publication never becomes a mandatory replay dependency."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.replay.codec import read_continuation
from tests.replay.support import emit, session


def test_identical_concurrent_complete_envelope_publication_is_idempotent_and_conflicts_preserve_original(
    example, tmp_path
):
    replay, host = session(example)
    emit(example, replay, 0, at=0, end=1)
    continuation = replay.export_continuation()
    encoded = continuation.to_json().encode()
    writer = ArtifactWriter(tmp_path)
    path = tmp_path / "checkpoint.json"

    def publish(_):
        writer.write_immutable_stream(path, (encoded,), maximum_bytes=len(encoded))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(publish, range(8)))
    assert path.read_bytes() == encoded
    decoded = read_continuation(path.read_bytes(), state_type=example.HostConsumerState)
    assert decoded.fingerprint == continuation.fingerprint
    emit(example, replay, 1, at=2, end=3)
    changed = replay.export_continuation().to_json().encode()
    maximum_bytes = len(changed)
    with pytest.raises(ValueError, match="Conflicting immutable"):
        writer.write_immutable_stream(path, (changed,), maximum_bytes=maximum_bytes)
    assert path.read_bytes() == encoded
    assert host.export_state().decision_count == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_iterator_failure_does_not_publish_a_false_completed_aggregate(example, tmp_path):
    replay, _ = session(example)
    emit(example, replay, 0, at=0, end=1)
    encoded = replay.export_continuation().to_json().encode()
    writer = ArtifactWriter(tmp_path)
    path = tmp_path / "checkpoint.json"

    def failed_input():
        yield encoded[: len(encoded) // 2]
        raise RuntimeError("Complete envelope validation failed")

    chunks = failed_input()
    maximum_bytes = len(encoded)
    with pytest.raises(RuntimeError, match="Complete envelope"):
        writer.write_immutable_stream(path, chunks, maximum_bytes=maximum_bytes)
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))
