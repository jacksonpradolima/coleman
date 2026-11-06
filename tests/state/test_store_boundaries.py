"""Real bounded ingress, metadata corruption and complete staging fail safely."""

import hashlib
import json
import sqlite3
from functools import partial
from pathlib import Path
from typing import Annotated, Any, cast

import pytest
from pydantic import Field

from coleman.artifacts import ArtifactWriter
from coleman.core.base import CoreContract
from coleman.core.primitives import fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.components import ComponentBinding, ComponentCodec, ComponentValue
from coleman.state.contracts import StateBinding
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CollectionDefinition, ComponentSpec, ExecutionCompatibility, StorageLimits
from tests.state.support import command


class TinyState(CoreContract):
    """Actual caller-owned publication-only payload for small quota witnesses."""

    text: Annotated[str, Field(strict=True, max_length=2048)]


class TinyOwner:
    def __init__(self):
        self.value = TinyState(schema_version=1, text="origin")

    def export_state(self):
        return TinyState.model_validate(self.value)

    def restore_state(self, state):
        self.value = TinyState.model_validate(state)


def tiny_layout(example, count=3, *, observer=None):
    implementation = DefinitionBinding(
        reference=example.reference("authored-tiny-publication-owner"),
        definition_fingerprint="sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    compatibility = ExecutionCompatibility(
        schema_version=1, runtime=example.execution_compatibility().runtime, owner_implementations=(implementation,)
    )

    class ObservedCodec(ComponentCodec[TinyState]):
        def encode(self, state):
            if observer is not None:
                observer.append(self.binding.owner.id)
            return super().encode(state)

    bindings, specifications = [], []
    for index in range(count):
        binding = StateBinding(
            owner=example.reference(f"tiny-{index:02d}"),
            implementation=implementation,
            state_codec=example.reference("authored-tiny-state"),
            configuration_fingerprint=fingerprint(index),
        )
        codec = ObservedCodec(binding, TinyState)
        bindings.append(ComponentBinding(codec, TinyOwner, lambda state: None))
        specifications.append(ComponentSpec(binding=binding, child=None, roles=("algorithm",), frontier_profile="none"))
    definition = CollectionDefinition(
        schema_version=1,
        reference=example.reference("tiny-definition"),
        session=example.reference("tiny-session"),
        source_epoch=example.EPOCH,
        children=(),
        components=tuple(specifications),
    )
    return definition, tuple(bindings), compatibility


def tiny_store(example, root, *, limits=None):
    definition, bindings, compatibility = tiny_layout(example)
    store = LocalStateStore.create(root, definition, compatibility, limits=limits)
    values = tuple(ComponentValue(binding.codec, TinyState(schema_version=1, text="old")) for binding in bindings)
    receipt = store.initialize(command(0, None, kind="initialize"), values, bindings, compatibility=compatibility)
    return store, bindings, compatibility, receipt


@pytest.mark.parametrize("mode", ["component", "collection"])
def test_incremental_encoding_stops_at_first_small_overflow_before_later_codecs_or_factories(
    state_example, tmp_path, mode
):
    observed, factories = [], []
    definition, bindings, compatibility = tiny_layout(state_example, observer=observed)
    limits = StorageLimits(schema_version=1, maximum_component_bytes=300, maximum_collection_bytes=400)
    values = tuple(
        ComponentValue(binding.codec, TinyState(schema_version=1, text="x" * (400 if mode == "component" else 250)))
        for binding in bindings
    )

    def factory(binding):
        factories.append(binding.codec.binding.owner.id)
        return binding.new_owner()

    watched = tuple(
        ComponentBinding(binding.codec, partial(factory, binding), binding.frontier) for binding in bindings
    )
    store = LocalStateStore.create(tmp_path / mode, definition, compatibility, limits=limits)
    observed.clear()
    request = command(0, None, kind="initialize")
    with pytest.raises(ValueError, match="encoded byte ceiling"):
        store.initialize(request, values, watched, compatibility=compatibility)
    assert observed == (["tiny-00"] if mode == "component" else ["tiny-00", "tiny-01"])
    assert factories == []
    assert store.read_head() is None
    assert list((store.root / "blobs").iterdir()) == []


def replace_acknowledged_manifest(store, prepared, manifest):
    """Corrupt only an owned test store with self-consistent hash metadata."""
    old = store.read_head()
    target = type(old).model_validate(old.model_copy(update={"manifest_fingerprint": manifest.fingerprint}))
    path = store.root / "manifests" / (manifest.fingerprint[7:] + ".json")
    ArtifactWriter(store.root).write_immutable(path, manifest.to_json())
    with sqlite3.connect(store.root / "acknowledgements.sqlite3") as connection:
        connection.execute("UPDATE state SET head=? WHERE singleton=1", (target.to_json(),))
        connection.execute("UPDATE receipts SET head=? WHERE generation=?", (target.to_json(), target.generation))


@pytest.mark.parametrize("mode", ["component", "collection"])
def test_manifest_quota_failure_precedes_any_child_file_acquisition(state_example, tmp_path, monkeypatch, mode):
    limits = StorageLimits(schema_version=1, maximum_component_bytes=300, maximum_collection_bytes=400)
    store, bindings, compatibility, _ = tiny_store(state_example, tmp_path / mode, limits=limits)
    prepared = store.load(bindings, compatibility=compatibility)
    count = 301 if mode == "component" else 200
    changed = tuple(
        type(blob).model_validate(blob.model_copy(update={"byte_count": count}))
        for blob in prepared.manifest.components
    )
    manifest = type(prepared.manifest).model_validate(prepared.manifest.model_copy(update={"components": changed}))
    replace_acknowledged_manifest(store, prepared, manifest)
    opened = []
    real_open = Path.open

    def observe(path, *args, **kwargs):
        if path.parent == store.root / "blobs":
            opened.append(path.name)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe)
    with pytest.raises(ValueError, match="ceiling|byte"):
        store.load(bindings, compatibility=compatibility)
    assert opened == []


@pytest.mark.parametrize(
    "change", ["missing_head", "missing_receipt", "changed_receipt", "future_head", "missing_table"]
)
def test_corrupt_acknowledged_metadata_never_falls_back_to_new_or_prior_learning(state_example, tmp_path, change):
    store, _, _, receipt = tiny_store(state_example, tmp_path / change)
    with sqlite3.connect(store.root / "acknowledgements.sqlite3") as connection:
        if change == "missing_head":
            connection.execute("UPDATE state SET head=NULL WHERE singleton=1")
        elif change == "missing_receipt":
            connection.execute("DELETE FROM receipts")
        elif change == "changed_receipt":
            connection.execute("UPDATE receipts SET intent='{}' WHERE generation=0")
        elif change == "future_head":
            payload = receipt.current_head.model_dump(mode="json")
            payload["schema_version"] = 2
            connection.execute("UPDATE state SET head=? WHERE singleton=1", (json.dumps(payload),))
        else:
            connection.execute("DROP TABLE state")
    principal = LocalStateStore.open
    root = store.root
    with pytest.raises((ValueError, sqlite3.Error)):
        principal(root)


@pytest.mark.parametrize("version", [2, True, "1"])
def test_future_or_coerced_header_version_rejects_before_any_restore(state_example, tmp_path, version):
    store, _, _, _ = tiny_store(state_example, tmp_path / "future-header")
    header = store.root / "store.json"
    assert header.exists()
    payload = json.loads(header.read_bytes())
    payload["schema_version"] = version
    header.write_text(json.dumps(payload))
    root = store.root
    with pytest.raises(ValueError):
        LocalStateStore.open(root)


def test_finite_inventory_rejects_an_unbounded_iterable_without_enumerating_it(state_example, tmp_path):
    definition, bindings, compatibility = tiny_layout(state_example)
    store = LocalStateStore.create(tmp_path / "finite-inventory", definition, compatibility)
    values = tuple(ComponentValue(binding.codec, TinyState(schema_version=1, text="origin")) for binding in bindings)
    enumerated = []

    def unbounded():
        enumerated.append("unexpected enumeration")
        yield from bindings

    unsupported = cast(tuple[ComponentBinding[Any, Any], ...], unbounded())
    request = command(0, None, kind="initialize")
    with pytest.raises((ValueError, TypeError), match="tuple|finite|bounded"):
        store.initialize(request, values, unsupported, compatibility=compatibility)
    assert enumerated == []
    assert store.read_head() is None


@pytest.mark.parametrize("index", [0, 15, 31])
def test_all32_supplied_payloads_validate_before_any_fresh_factory(state_example, pending_evidence, tmp_path, index):
    example = state_example
    continuations, _, domain = pending_evidence
    definition, bindings = example.collection_layout(tuple(range(32)))
    compatibility = example.execution_compatibility()
    values = [
        ComponentValue(binding.codec, state) for binding, state in zip(bindings, (*continuations, domain), strict=True)
    ]
    invalid = object.__new__(ComponentValue)
    object.__setattr__(invalid, "codec", values[index].codec)
    object.__setattr__(invalid, "value", values[index].value.model_copy(update={"schema_version": 99}))
    values[index] = invalid
    calls = []

    def factory(binding):
        calls.append(binding.codec.binding.owner.id)
        return binding.new_owner()

    watched = tuple(
        ComponentBinding(binding.codec, partial(factory, binding), binding.frontier) for binding in bindings
    )
    store = LocalStateStore.create(tmp_path / f"bad-child-{index}", definition, compatibility)
    offered = tuple(values)
    request = command(0, None, kind="initialize")
    with pytest.raises(ValueError):
        store.initialize(request, offered, watched, compatibility=compatibility)
    assert calls == []
    assert store.read_head() is None


def test_last_real_child_restore_failure_discards_staging_and_acknowledges_nothing(
    state_example, pending_evidence, tmp_path
):
    example = state_example
    continuations, _, domain = pending_evidence
    definition, bindings = example.collection_layout(tuple(range(32)))
    compatibility = example.execution_compatibility()
    values = tuple(
        ComponentValue(binding.codec, state) for binding, state in zip(bindings, (*continuations, domain), strict=True)
    )
    restored = []

    class FaultAfterRealRestore(example.ReplayOwner):
        def restore_state(self, state):
            super().restore_state(state)
            restored.append(self.child)
            if self.child == 31:
                raise RuntimeError("fault-after-real-child31-restore")

    watched = tuple(
        ComponentBinding(binding.codec, partial(FaultAfterRealRestore, child), binding.frontier)
        for child, binding in enumerate(bindings[:-1])
    ) + (bindings[-1],)
    store = LocalStateStore.create(tmp_path / "last-restore", definition, compatibility)
    request = command(0, None, kind="initialize")
    with pytest.raises(RuntimeError, match="real-child31"):
        store.initialize(request, values, watched, compatibility=compatibility)
    assert restored == list(range(32))
    assert store.read_head() is None
    assert list((store.root / "blobs").iterdir()) == []


def test_root_capacity_is_validated_before_creating_the_owned_store(state_example, tmp_path):
    definition, _, compatibility = tiny_layout(state_example)
    limits = StorageLimits(schema_version=1, maximum_root_bytes=1024)
    path = tmp_path / "too-small"
    with pytest.raises(ValueError, match="headroom|reserve"):
        LocalStateStore.create(path, definition, compatibility, limits=limits)
    assert not path.exists()
