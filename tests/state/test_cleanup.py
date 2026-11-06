"""Receipt retirement prunes owned garbage without changing learner frontiers."""

import sqlite3

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.state.components import ComponentValue
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import StorageLimits
from tests.state.support import command
from tests.state.test_store_boundaries import TinyState, tiny_layout, tiny_store


def test_receipt_expiration_cannot_resurrect_progress_but_current_retry_stays_idempotent(state_example, tmp_path):
    definition, bindings, compatibility = tiny_layout(state_example)
    limits = StorageLimits(schema_version=1, receipt_window=2)
    store = LocalStateStore.create(tmp_path / "retirement", definition, compatibility, limits=limits)
    values = tuple(ComponentValue(binding.codec, TinyState(schema_version=1, text="origin")) for binding in bindings)
    initial = command(0, None, kind="initialize")
    receipt = store.initialize(initial, values, bindings, compatibility=compatibility)
    old = receipt.current_head
    latest_request = initial
    for generation in range(1, 5):
        latest_request = command(generation, receipt.current_head)

        def advance(stage, generation=generation):
            for binding in bindings:
                stage.owner(binding).value = TinyState(schema_version=1, text=f"generation-{generation}")

        receipt = store.commit_transition(latest_request, bindings, advance, compatibility=compatibility)
    current = receipt.current_head
    before = store.load(bindings, compatibility=compatibility)
    result = store.cleanup(current)
    assert result.retired_receipt_floor == 3
    assert result.current_head == current
    with sqlite3.connect(store.root / "acknowledgements.sqlite3") as database:
        retained = database.execute("SELECT generation FROM receipts ORDER BY generation").fetchall()
    assert retained == [(3,), (4,)]
    calls = []

    def never(stage):
        calls.append("unexpected callback")

    replay = store.commit_transition(latest_request, (), never, compatibility=compatibility)
    assert replay.status == "retained"
    assert replay.current_head == current
    with pytest.raises(ValueError, match="expired|retired|generation"):
        store.initialize(initial, (), (), compatibility=compatibility)
    with pytest.raises(ValueError, match="stale"):
        store.cleanup(old)
    assert calls == []
    after = store.load(bindings, compatibility=compatibility)
    assert tuple(value.value.to_json() for value in after.values) == tuple(
        value.value.to_json() for value in before.values
    )


def test_cleanup_removes_only_unreachable_reserved_artifacts_and_keeps_current_exact_bytes(state_example, tmp_path):
    store, bindings, compatibility, receipt = tiny_store(state_example, tmp_path / "cleanup")
    current = store.load(bindings, compatibility=compatibility)
    protected = {path: path.read_bytes() for path in (store.root / "blobs").iterdir()}
    writer = ArtifactWriter(store.root)
    orphan = store.root / "blobs" / ("0" * 64 + ".json")
    writer.write_immutable(orphan, "unacknowledged bytes")
    temporary = store.root / "manifests" / "tmpabcdefgh.tmp"
    temporary.write_bytes(b"interrupted temporary")
    foreign = store.root / "blobs" / "notes.tmp"
    foreign.write_bytes(b"host note outside reserved names")
    result = store.cleanup(receipt.current_head)
    assert result.removed_blobs == 1
    assert result.removed_files == 2
    assert result.removed_bytes == len(b"unacknowledged bytesinterrupted temporary")
    assert not orphan.exists()
    assert not temporary.exists()
    assert foreign.read_bytes() == b"host note outside reserved names"
    assert {path: path.read_bytes() for path in protected} == protected
    assert store.load(bindings, compatibility=compatibility).manifest == current.manifest


@pytest.mark.parametrize("version", [(3, 44, 5), (3, 45, 0), (3, 49, 2), (3, 50, 6), (3, 51, 2)])
def test_unfixed_linked_sqlite_refuses_before_store_creation_or_owner_calls(
    state_example, tmp_path, monkeypatch, version
):
    definition, _, compatibility = tiny_layout(state_example)
    monkeypatch.setattr(sqlite3, "sqlite_version_info", version)
    root = tmp_path / "unsupported-engine"
    with pytest.raises(ValueError, match="WAL state requires"):
        LocalStateStore.create(root, definition, compatibility)
    assert not root.exists()


@pytest.mark.parametrize("version", [(3, 44, 6), (3, 44, 9), (3, 50, 7), (3, 50, 9), (3, 51, 3), (3, 53, 4)])
def test_qualified_backport_version_metadata_still_uses_the_actual_linked_store_engine(
    state_example, tmp_path, monkeypatch, version
):
    definition, _, compatibility = tiny_layout(state_example)
    monkeypatch.setattr(sqlite3, "sqlite_version_info", version)
    store = LocalStateStore.create(tmp_path / "qualified-metadata", definition, compatibility)
    assert store.read_head() is None
