"""Real local-store metadata, filesystem and SQL failures cannot invent an ack."""

import json
import sqlite3
import time
from contextlib import closing

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.core.primitives import fingerprint
from coleman.state import _disk, _garbage
from coleman.state.components import ComponentBinding, ComponentValue
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import StorageLimits
from tests.state.support import command
from tests.state.test_store_boundaries import TinyState, tiny_layout


@pytest.fixture(scope="module")
def layout(state_example):
    return tiny_layout(state_example)


def small_limits():
    """Keep real capacity tests in tens of MiB, with valid persisted headroom."""
    return StorageLimits(
        schema_version=1,
        maximum_children=0,
        maximum_components=3,
        maximum_component_bytes=4096,
        maximum_collection_bytes=16384,
        maximum_manifest_bytes=1024**2,
        receipt_window=1,
        maximum_root_bytes=32 * 1024**2,
        maximum_entries=64,
        maximum_orphan_bytes=2 * 1024**2,
        maximum_orphan_entries=8,
    )


def initialized(layout, root):
    definition, bindings, compatibility = layout
    store = LocalStateStore.create(root, definition, compatibility, limits=small_limits())
    values = tuple(ComponentValue(item.codec, TinyState(schema_version=1, text="origin")) for item in bindings)
    receipt = store.initialize(command(0, None, kind="initialize"), values, bindings, compatibility=compatibility)
    return store, receipt


def sqlite_change(store, statement, arguments=()):
    with closing(sqlite3.connect(store.root / _disk.DATABASE)) as database, database:
        database.execute(statement, arguments)


def watched_bindings(bindings, calls):
    def factory(binding):
        calls.append("factory")
        return binding.new_owner()

    return tuple(ComponentBinding(item.codec, lambda item=item: factory(item), item.frontier) for item in bindings)


@pytest.mark.parametrize("name", ["not-a-digest", "sha256:" + "G" * 64, "sha256:" + "0" * 63])
def test_invalid_content_addresses_cannot_choose_a_filesystem_path(tmp_path, name):
    with pytest.raises(ValueError, match="digest|family"):
        _disk.artifact(tmp_path, "blobs", name)
    assert list(tmp_path.iterdir()) == []


def test_valid_digest_cannot_select_an_unrecognized_artifact_namespace(tmp_path):
    with pytest.raises(ValueError, match="family"):
        _disk.artifact(tmp_path, "../outside", fingerprint("payload"))
    expected = tmp_path / "blobs" / (fingerprint("payload")[7:] + ".json")
    assert _disk.artifact(tmp_path, "blobs", fingerprint("payload")) == expected


@pytest.mark.parametrize("shape", ["missing", "directory", "symlink", "oversized", "changed"])
def test_bounded_file_acquisition_rejects_unusable_or_changed_acknowledged_bytes(tmp_path, shape):
    path = tmp_path / "acknowledged.json"
    exact = b'{"value":0}'
    if shape == "directory":
        path.mkdir()
    elif shape == "symlink":
        target = tmp_path / "target.json"
        target.write_bytes(exact)
        path.symlink_to(target)
    elif shape == "oversized":
        path.write_bytes(exact + b" ")
    elif shape == "changed":
        path.write_bytes(b'{"value":1}')
    with pytest.raises(ValueError, match="regular|missing|ceiling|digest"):
        _disk.acquire(path, len(exact), expected=_disk.digest(exact))


@pytest.mark.parametrize("shape", ["file", "nonempty", "symlink"])
def test_creation_never_overwrites_existing_user_bytes(layout, tmp_path, shape):
    definition, _, compatibility = layout
    root = tmp_path / "root"
    if shape == "file":
        root.write_text("retain this file")
        protected = root
    elif shape == "nonempty":
        root.mkdir()
        protected = root / "notes.txt"
        protected.write_text("retain this file")
    else:
        target = tmp_path / "target"
        target.mkdir()
        protected = target / "notes.txt"
        protected.write_text("retain this file")
        root.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="empty trusted root"):
        LocalStateStore.create(root, definition, compatibility)
    assert protected.read_text() == "retain this file"


def test_open_requires_existing_metadata_instead_of_creating_a_virgin_store(tmp_path):
    absent = tmp_path / "absent"
    with pytest.raises(ValueError, match="explicit create"):
        LocalStateStore.open(absent)
    assert not absent.exists()


def test_unknown_profile_and_changed_in_memory_header_fail_before_owner_work(layout, tmp_path):
    store, receipt = initialized(layout, tmp_path / "header")
    path = store.root / _disk.HEADER
    original = path.read_bytes()
    payload = json.loads(original)
    payload["profile"] = "future-local-profile"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="storage profile"):
        LocalStateStore.open(store.root)
    path.write_bytes(original)
    payload = json.loads(original)
    payload["sqlite_version"] = "changed-creation-metadata"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Immutable state header"):
        store.read_head()
    path.write_bytes(original)
    assert store.read_head() == receipt.current_head


@pytest.mark.parametrize("operation", ["wrong_initialize", "wrong_transition", "virgin_load", "virgin_transition"])
def test_explicit_virgin_and_transition_operations_have_distinct_admission(layout, tmp_path, operation):
    definition, bindings, compatibility = layout
    store = LocalStateStore.create(tmp_path / operation, definition, compatibility, limits=small_limits())
    calls = []
    watched = watched_bindings(bindings, calls)

    def callback(stage):
        calls.append("transition")

    with pytest.raises(ValueError, match="Initialization|transition|Virgin"):
        if operation == "wrong_initialize":
            store.initialize(command(0, None, kind="transition"), (), watched, compatibility=compatibility)
        elif operation == "wrong_transition":
            store.commit_transition(command(0, None, kind="initialize"), watched, callback, compatibility=compatibility)
        elif operation == "virgin_load":
            store.load(watched, compatibility=compatibility)
        else:
            store.commit_transition(command(0, None), watched, callback, compatibility=compatibility)
    assert calls == []
    assert store.read_head() is None


def test_skipping_a_generation_rejects_before_factory_and_transition(layout, tmp_path):
    _, bindings, compatibility = layout
    store, receipt = initialized(layout, tmp_path / "generation")
    calls = []
    with pytest.raises(ValueError, match="next consecutive generation"):
        store.commit_transition(
            command(2, receipt.current_head),
            watched_bindings(bindings, calls),
            lambda stage: calls.append("transition"),
            compatibility=compatibility,
        )
    assert calls == []
    assert store.read_head() == receipt.current_head


@pytest.mark.parametrize("change", ["missing_row", "oversized_head", "invalid_flag", "virgin_with_head", "future_db"])
def test_corrupt_sql_acknowledgement_metadata_is_not_an_origin(layout, tmp_path, change):
    store, _ = initialized(layout, tmp_path / change)
    if change == "missing_row":
        sqlite_change(store, "DELETE FROM state")
    elif change == "oversized_head":
        sqlite_change(store, "UPDATE state SET head=?", (" " * (2 * _disk.MIB + 1),))
    elif change == "invalid_flag":
        sqlite_change(store, "UPDATE state SET ever_initialized=2")
    elif change == "virgin_with_head":
        sqlite_change(store, "UPDATE state SET ever_initialized=0")
    else:
        sqlite_change(store, "PRAGMA user_version=2")
    with pytest.raises(ValueError, match="metadata|database version"):
        store.read_head()


def test_quick_check_rejects_a_real_duplicate_btree_root_before_state_queries(layout, tmp_path):
    store, _ = initialized(layout, tmp_path / "btree")
    with closing(sqlite3.connect(store.root / _disk.DATABASE)) as database, database:
        database.execute("CREATE TABLE unrelated(value TEXT)")
        database.execute("PRAGMA writable_schema=ON")
        database.execute(
            "UPDATE sqlite_master SET rootpage=(SELECT rootpage FROM sqlite_master WHERE name='state') "
            "WHERE name='unrelated'"
        )
        database.execute("PRAGMA writable_schema=OFF")
    with pytest.raises(ValueError, match="database is corrupt"):
        store.read_head()


@pytest.mark.parametrize("field", ["session", "source_epoch"])
def test_acknowledged_namespace_cannot_disagree_with_the_immutable_header(layout, state_example, tmp_path, field):
    store, receipt = initialized(layout, tmp_path / field)
    changed = type(receipt.current_head).model_validate(
        receipt.current_head.model_copy(update={field: state_example.reference("different-namespace")})
    )
    sqlite_change(store, "UPDATE state SET head=?", (changed.to_json(),))
    with pytest.raises(ValueError, match="session or epoch"):
        store.read_head()


def test_virgin_database_cannot_already_contain_an_operational_receipt(layout, tmp_path):
    source, receipt = initialized(layout, tmp_path / "source")
    definition, _, compatibility = layout
    target = LocalStateStore.create(tmp_path / "virgin", definition, compatibility, limits=small_limits())
    with closing(sqlite3.connect(source.root / _disk.DATABASE)) as database:
        row = database.execute("SELECT generation,intent,predecessor,head FROM receipts").fetchone()
    sqlite_change(target, "INSERT INTO receipts VALUES (?,?,?,?)", row)
    with pytest.raises(ValueError, match="Virgin state metadata contains operational receipts"):
        target.read_head()
    assert receipt.current_head.generation == 0


@pytest.mark.parametrize("change", ["oversized", "noncanonical", "contradictory_intent", "contradictory_parent"])
def test_receipt_chain_verifies_size_canonical_bytes_and_actual_manifest_joins(layout, tmp_path, change):
    store, receipt = initialized(layout, tmp_path / change)
    if change == "oversized":
        sqlite_change(
            store,
            "UPDATE receipts SET intent=?",
            (" " * (store.limits.maximum_manifest_bytes + 1),),
        )
    elif change == "noncanonical":
        with closing(sqlite3.connect(store.root / _disk.DATABASE)) as database:
            old = database.execute("SELECT intent FROM receipts").fetchone()[0]
        sqlite_change(store, "UPDATE receipts SET intent=?", (json.dumps(json.loads(old), indent=2),))
    elif change == "contradictory_parent":
        sqlite_change(store, "UPDATE receipts SET predecessor=?", (receipt.current_head.to_json(),))
    else:
        changed = command(0, None, kind="initialize", content="different accepted command")
        sqlite_change(store, "UPDATE receipts SET intent=?", (changed.intent.to_json(),))
    with pytest.raises(ValueError, match="byte ceiling|canonical|contradicts"):
        store.read_head()


def test_current_head_must_be_the_head_of_the_actual_last_receipt(layout, tmp_path):
    _, bindings, compatibility = layout
    store, receipt = initialized(layout, tmp_path / "unreceipted-head")
    old = store.load(bindings, compatibility=compatibility).manifest
    intent = command(0, None, kind="initialize", content="not actually receipted").intent
    alternative = type(old).model_validate(old.model_copy(update={"intent": intent}))
    path = _disk.artifact(store.root, "manifests", alternative.fingerprint)
    ArtifactWriter(store.root).write_immutable(path, alternative.to_json())
    sqlite_change(store, "UPDATE state SET head=?", (_disk.make_head(alternative).to_json(),))
    with pytest.raises(ValueError, match="differs from its retained receipt"):
        store.read_head()
    assert receipt.current_head.manifest_fingerprint != alternative.fingerprint


@pytest.mark.parametrize(
    "change", ["root_link", "foreign_directory", "nested_directory", "nested_link", "root_entries", "artifact_entries"]
)
def test_invalid_flat_layout_and_entry_overflow_reject_before_any_factory(layout, tmp_path, change):
    _, bindings, compatibility = layout
    store, receipt = initialized(layout, tmp_path / change)
    if change == "root_link":
        (store.root / "linked-note").symlink_to(store.root / _disk.HEADER)
    elif change == "foreign_directory":
        (store.root / "unexpected-directory").mkdir()
    elif change == "nested_directory":
        (store.root / "blobs" / "nested-directory").mkdir()
    elif change == "nested_link":
        (store.root / "blobs" / "linked-note").symlink_to(store.root / _disk.HEADER)
    else:
        parent = store.root if change == "root_entries" else store.root / "blobs"
        for index in range(store.limits.maximum_entries):
            (parent / f"foreign-{index:03d}.txt").write_bytes(b"retained foreign bytes")
    calls = []
    with pytest.raises(ValueError, match="symlink|directory|flat regular|entry ceiling"):
        store.commit_transition(
            command(1, receipt.current_head),
            watched_bindings(bindings, calls),
            lambda stage: calls.append("transition"),
            compatibility=compatibility,
        )
    assert calls == []


@pytest.mark.parametrize("name", [_disk.DATABASE + "-wal", _disk.DATABASE + "-shm"])
def test_nonregular_sidecar_is_rejected_before_sqlite_acquisition(layout, tmp_path, name):
    store, _ = initialized(layout, tmp_path / name)
    sidecar = store.root / name
    assert not sidecar.exists()
    sidecar.mkdir()
    with pytest.raises(ValueError, match="regular owned files"):
        store.read_head()


def test_database_symlink_is_refused_by_the_connection_boundary(layout, tmp_path):
    store, _ = initialized(layout, tmp_path / "linked-database")
    original = store.root / _disk.DATABASE
    saved = store.root / "saved-database"
    original.rename(saved)
    original.symlink_to(saved)
    with pytest.raises(ValueError, match="cannot be a symlink"), _disk.connection(store.root, 1.0):
        pytest.fail("The linked acknowledgement database must never be acquired")


@pytest.mark.parametrize("location", ["sidecar", "foreign"])
def test_byte_admission_precedes_callbacks_and_keeps_every_foreign_byte(layout, tmp_path, location):
    _, bindings, compatibility = layout
    store, receipt = initialized(layout, tmp_path / location)
    path = store.root / (_disk.DATABASE + "-wal" if location == "sidecar" else "foreign.bin")
    with path.open("wb") as stream:
        stream.truncate(store.limits.maximum_root_bytes + 1)
    calls = []
    with pytest.raises(ValueError, match="metadata exceeds|total byte ceiling"):
        store.commit_transition(
            command(1, receipt.current_head),
            watched_bindings(bindings, calls),
            lambda stage: calls.append("transition"),
            compatibility=compatibility,
        )
    assert calls == []
    assert path.stat().st_size == store.limits.maximum_root_bytes + 1


@pytest.mark.parametrize("capacity", ["bytes", "entries"])
def test_foreign_files_cannot_consume_the_next_candidates_reserved_headroom(layout, tmp_path, capacity):
    _, bindings, compatibility = layout
    store, receipt = initialized(layout, tmp_path / capacity)
    foreign = []
    if capacity == "bytes":
        path = store.root / "foreign.bin"
        with path.open("wb") as stream:
            stream.truncate(20 * _disk.MIB)
        foreign.append(path)
    else:
        while len(_disk.owned_files(store.root, store.limits)) < 58:
            path = store.root / f"foreign-{len(foreign):03d}.txt"
            path.write_bytes(b"keep")
            foreign.append(path)
    calls = []
    with pytest.raises(ValueError, match="headroom before state callback"):
        store.commit_transition(
            command(1, receipt.current_head),
            watched_bindings(bindings, calls),
            lambda stage: calls.append("transition"),
            compatibility=compatibility,
        )
    assert calls == []
    assert all(path.is_file() for path in foreign)


@pytest.mark.parametrize("capacity", ["bytes", "entries"])
def test_unacknowledged_publication_overflow_cannot_commit_a_candidate(layout, tmp_path, capacity):
    _, bindings, compatibility = layout
    store, receipt = initialized(layout, tmp_path / capacity)

    class InterruptedPublication(LocalStateStore):
        def _barrier(self, phase, *, component=None):
            if phase == "after_manifest":
                count = self.limits.maximum_orphan_entries + 1 if capacity == "entries" else 1
                for index in range(count):
                    path = self.root / "blobs" / (f"{index:064x}" + ".json")
                    with path.open("wb") as stream:
                        stream.truncate(self.limits.maximum_orphan_bytes + 1 if capacity == "bytes" else 1)

    principal = InterruptedPublication.open(store.root)
    calls = []
    with pytest.raises(ValueError, match="orphan .* ceiling"):
        principal.commit_transition(
            command(1, receipt.current_head),
            bindings,
            lambda stage: calls.append("transition"),
            compatibility=compatibility,
        )
    assert calls == ["transition"]
    assert store.read_head() == receipt.current_head
    with closing(sqlite3.connect(store.root / _disk.DATABASE)) as database:
        assert database.execute("SELECT generation FROM receipts").fetchall() == [(0,)]


def test_cleanup_detects_an_actual_intervening_commit_before_pruning(layout, tmp_path):
    _, bindings, compatibility = layout
    store, receipt = initialized(layout, tmp_path / "cleanup-race")

    class InterveningWriter(LocalStateStore):
        def _barrier(self, phase, *, component=None):
            if phase == "cleanup_after_metadata_commit":
                self.commit_transition(
                    command(1, receipt.current_head), bindings, lambda stage: None, compatibility=compatibility
                )

    racing = InterveningWriter.open(store.root)
    with pytest.raises(ValueError, match="head changed before protected pruning"):
        racing.cleanup(receipt.current_head)
    current = store.read_head()
    assert current is not None and current.generation == 1
    assert store.load(bindings, compatibility=compatibility).manifest.intent.generation == 1


def test_owned_transaction_rolls_back_metadata_when_the_body_fails(layout, tmp_path):
    store, receipt = initialized(layout, tmp_path / "rollback")
    with (
        pytest.raises(RuntimeError, match="after actual SQL mutation"),
        _disk.ownership(store.root, 1.0, limits=store.limits) as database,
    ):
        database.execute("DELETE FROM receipts")
        raise RuntimeError("after actual SQL mutation")
    assert store.read_head() == receipt.current_head


def test_unfinished_returning_statement_produces_real_busy_then_only_commit_is_retried():
    class ObservedConnection(sqlite3.Connection):
        blocker: sqlite3.Cursor | None = None

        def execute(self, statement, *arguments):
            try:
                return super().execute(statement, *arguments)
            except sqlite3.OperationalError as error:
                if statement == "COMMIT":
                    failures.append(error.sqlite_errorcode)
                    assert self.blocker is not None
                    self.blocker.fetchall()
                    self.blocker = None
                raise

    failures = []
    statements = []
    database = sqlite3.connect(":memory:", isolation_level=None, factory=ObservedConnection)
    try:
        database.set_trace_callback(statements.append)
        database.execute("CREATE TABLE updates(value INTEGER)")
        database.execute("BEGIN IMMEDIATE")
        database.blocker = database.execute("INSERT INTO updates VALUES (1),(2) RETURNING value")
        assert database.blocker.fetchone() == (1,)
        _disk.commit(database, 1.0)
        assert failures == [sqlite3.SQLITE_BUSY]
        assert statements.count("COMMIT") == 2
        assert sum(statement.startswith("INSERT") for statement in statements) == 1
        assert database.execute("SELECT value FROM updates ORDER BY value").fetchall() == [(1,), (2,)]
        assert not database.in_transaction
    finally:
        database.close()


def test_noncontention_commit_error_is_propagated_without_an_automatic_body_retry():
    database = sqlite3.connect(":memory:", isolation_level=None)
    statements = []
    database.set_trace_callback(statements.append)
    try:
        with pytest.raises(sqlite3.OperationalError, match="no transaction is active"):
            _disk.commit(database, 1.0)
        assert statements.count("COMMIT") == 1
        assert not database.in_transaction
    finally:
        database.close()


def test_busy_commit_does_not_renew_its_original_monotonic_budget():
    database = sqlite3.connect(":memory:", isolation_level=None)
    try:
        database.execute("CREATE TABLE updates(value INTEGER)")
        database.execute("BEGIN IMMEDIATE")
        pending = database.execute("INSERT INTO updates VALUES (1),(2) RETURNING value")
        assert pending.fetchone() == (1,)
        with pytest.raises((TimeoutError, sqlite3.OperationalError)):
            _disk.commit(database, 0.02)
        assert database.in_transaction
        pending.close()
        database.rollback()
        assert database.execute("SELECT count(*) FROM updates").fetchone() == (0,)
        with pytest.raises(TimeoutError, match="finite wait allowance"):
            _disk._busy(database, time.monotonic() - 1)
    finally:
        database.close()


def test_only_reserved_flat_names_are_garbage_candidates(tmp_path):
    names = [tmp_path / "notes.tmp", tmp_path / ("0" * 64 + ".json"), tmp_path / "tmpabcdefgh.tmp"]
    assert not any(_garbage.recognized(path) for path in names)
    assert _garbage.recognized(tmp_path / "blobs" / ("0" * 64 + ".json"))
    assert _garbage.recognized(tmp_path / "manifests" / "tmpabcdefgh.tmp")
    assert not _garbage.recognized(tmp_path / "blobs" / "notes.tmp")
