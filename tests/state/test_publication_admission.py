"""Real transition publication retains selected readers, bytes and fault boundaries."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from contextlib import contextmanager
from decimal import Decimal
from types import MethodType, SimpleNamespace
from typing import ClassVar, Self

import pytest
from pydantic import Field, StrictInt

from coleman.core.base import CoreContract
from coleman.state import _prepare
from coleman.state.components import ComponentBinding, ComponentCodec, ComponentValue, StateBytes
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CollectionDefinition, ComponentSpec, StorageLimits
from tests.state.support import command
from tests.state.test_codec_symmetry import layout


class _Probe:
    events: ClassVar[list[tuple]] = []
    publishing = False
    read_hook: ClassVar[Callable[[_State], None] | None] = None
    frontier_hook: ClassVar[Callable[[_State], None] | None] = None


class _State(CoreContract):
    """Declared fields include values invisible to JSON or ordinary equality."""

    key: StrictInt
    value: StrictInt = 0
    hidden: StrictInt = Field(default=5, exclude=True)
    amount: Decimal = Decimal("2.00")
    zero: float = -0.0
    pending: tuple[StrictInt, ...] = (-2, 0)
    nullable: str | None = None
    text: str = ""

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        value = super().from_json(payload)
        _Probe.events.append(("reader", value.key, _Probe.publishing))
        if _Probe.publishing and _Probe.read_hook is not None:
            _Probe.read_hook(value)
        return value


class _WriterState(_State):
    def to_json(self) -> str:
        _Probe.events.append(("writer", self.value))
        return super().to_json()


class _Codec(ComponentCodec):
    def encode(self, state):
        _Probe.events.append(("codec", state.value))
        return super().encode(state)


class _Owner:
    def __init__(self):
        self.value = None

    def restore_state(self, value):
        self.value = value

    def export_state(self):
        return self.value


class _InterruptedError(RuntimeError):
    """An explicit local publication fault, never a successful acknowledgement."""


class _Store(LocalStateStore):
    def __init__(self, root, header):
        super().__init__(root, header)
        self.hook: Callable[[str, int | None], None] | None = None
        self.fault: tuple[str, int | None] | None = None

    def _barrier(self, phase, *, component=None):
        _Probe.events.append(("barrier", phase, component))
        if phase == "before_blobs":
            _Probe.publishing = True
        if self.hook is not None:
            self.hook(phase, component)
        if self.fault == (phase, component):
            raise _InterruptedError(phase)


def _reset():
    _Probe.events = []
    _Probe.publishing = False
    _Probe.read_hook = None
    _Probe.frontier_hook = None


@pytest.fixture(autouse=True)
def reset_probe():
    _reset()
    yield
    _reset()


def _case(example, root, *, count=2, state_type=_State, codec_type=ComponentCodec, limits=None):
    base = layout(example, state_type)
    bindings, values, specs = [], [], []

    def frontier(value):
        if value.value < 0 and _Probe.frontier_hook is not None:
            _Probe.frontier_hook(value)
        return None

    for index in range(count):
        selected = type(base.codec.binding).model_validate(
            base.codec.binding.model_copy(update={"owner": example.reference("publication-" + str(index))})
        )
        codec = codec_type(selected, state_type)
        bindings.append(ComponentBinding(codec, _Owner, frontier))
        values.append(ComponentValue(codec, state_type(schema_version=1, key=index)))
        specs.append(ComponentSpec(binding=selected, child=None, roles=("domain",), frontier_profile="none"))
    definition = CollectionDefinition.model_validate(base.definition.model_copy(update={"components": tuple(specs)}))
    store = _Store.create(root, definition, base.compatibility, limits=limits)
    initial = store.initialize(
        command(0, None, kind="initialize"), tuple(values), tuple(bindings), compatibility=base.compatibility
    )
    result = SimpleNamespace(
        store=store,
        bindings=tuple(bindings),
        compatibility=base.compatibility,
        state_type=state_type,
        initial=initial,
        original_blobs={item.name for item in (root / "blobs").iterdir()},
    )
    _reset()
    return result


def _change(case, *, text="", held=None, calls=None, shift=0):
    def transition(stage):
        if calls is not None:
            calls.append("actual-transition")
        for index, binding in enumerate(case.bindings):
            value = case.state_type(schema_version=1, key=index, value=-index - 1 - shift, text=text)
            stage.owner(binding).value = value
            if held is not None:
                held.append(value)

    return transition


@contextmanager
def _publication_calls():
    """Observe original code starts without wrapping guarded methods or codecs."""
    functions = (_prepare._build_for_publish, _prepare._publication_content)
    names = {function.__code__: function.__name__ for function in functions}
    calls = dict.fromkeys(names.values(), 0)
    tool = next(index for index in (4, 3, 2, 5, 0, 1) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "coleman-publication-test")

    def began(code, _offset):
        calls[names[code]] += 1

    try:
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, began)
        for code in names:
            sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
        yield calls
    finally:
        sys.monitoring.free_tool_id(tool)


def _assert_old(case):
    assert case.store.read_head() == case.initial.current_head
    assert {item.name for item in (case.store.root / "blobs").iterdir()} == case.original_blobs
    _Probe.read_hook = None
    _Probe.publishing = False
    loaded = case.store.load(case.bindings, compatibility=case.compatibility)
    assert all(item.value.value == 0 for item in loaded.values)


def test_real_stock_transition_uses_own_publication_and_actual_reader_after_fault_barrier(state_example, tmp_path):
    case = _case(state_example, tmp_path / "stock")
    request = command(1, case.initial.current_head)
    with _publication_calls() as calls:
        receipt = case.store.commit_transition(request, case.bindings, _change(case), compatibility=case.compatibility)
    assert calls == {"_build_for_publish": 1, "_publication_content": 1}
    phases = [event for event in _Probe.events if event[0] == "barrier"]
    assert phases == [
        ("barrier", "before_blobs", None),
        ("barrier", "after_blob", 0),
        ("barrier", "after_blob", 1),
        ("barrier", "after_manifest", None),
        ("barrier", "after_head_update", None),
        ("barrier", "after_commit", None),
    ]
    first = _Probe.events.index(phases[0])
    second = _Probe.events.index(phases[1])
    assert _Probe.events[first + 1 : second] == [("reader", 0, True), ("reader", 1, True)]
    _Probe.publishing = False
    prepared = case.store.load(case.bindings, compatibility=case.compatibility)
    manifest_path = case.store.root / "manifests" / (receipt.accepted_head.manifest_fingerprint[7:] + ".json")
    manifest = manifest_path.read_bytes()
    assert "sha256:" + hashlib.sha256(manifest).hexdigest() == receipt.accepted_head.manifest_fingerprint
    assert len(json.loads(manifest)["components"]) == 2
    for index, item in enumerate(prepared.manifest.components):
        raw = (case.store.root / "blobs" / (item.fingerprint[7:] + ".json")).read_bytes()
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == item.fingerprint
        assert len(raw) == item.byte_count
        assert item.binding == case.bindings[index].codec.binding
        assert json.loads(raw)["value"] == -index - 1
        state = prepared.component(case.bindings[index].codec)
        assert state.hidden == 5
        assert state.amount.as_tuple() == Decimal("2.00").as_tuple()
        assert state.zero.hex() == "-0x0.0p+0"
        assert state.pending == (-2, 0)
        assert state.nullable is None


@pytest.mark.parametrize("field,value", [("hidden", 6), ("amount", Decimal("2.0")), ("zero", 0.0)])
def test_actual_publication_selected_reader_cannot_erase_complete_typed_fields(state_example, tmp_path, field, value):
    case = _case(state_example, tmp_path / field)
    _Probe.read_hook = lambda decoded: object.__setattr__(decoded, field, value)
    with _publication_calls() as calls, pytest.raises(ValueError):
        case.store.commit_transition(
            command(1, case.initial.current_head), case.bindings, _change(case), compatibility=case.compatibility
        )
    assert calls["_publication_content"] == 1
    assert any(event[0] == "reader" and event[2] for event in _Probe.events)
    _assert_old(case)


@pytest.mark.parametrize("phase", ["frontier", "before_blobs"])
def test_trusted_callback_cannot_mutate_actual_captured_state_before_publication(state_example, tmp_path, phase):
    case = _case(state_example, tmp_path / phase)
    held = []
    if phase == "frontier":
        _Probe.frontier_hook = lambda value: object.__setattr__(value, "hidden", 6)
    else:
        case.store.hook = lambda name, _index: object.__setattr__(held[0], "hidden", 6) if name == phase else None
    with _publication_calls() as calls, pytest.raises(ValueError):
        case.store.commit_transition(
            command(1, case.initial.current_head),
            case.bindings,
            _change(case, held=held),
            compatibility=case.compatibility,
        )
    assert calls["_build_for_publish"] == 1
    _assert_old(case)


@pytest.mark.parametrize("target", ["earlier_original", "earlier_decoded"])
def test_later_actual_publication_reader_cannot_mutate_an_earlier_compared_value(state_example, tmp_path, target):
    case = _case(state_example, tmp_path / target)
    held, decoded = [], []

    def mutate(value):
        if value.key == 0:
            decoded.append(value)
        else:
            object.__setattr__((held if target == "earlier_original" else decoded)[0], "hidden", 6)

    _Probe.read_hook = mutate
    with _publication_calls() as calls, pytest.raises(ValueError):
        case.store.commit_transition(
            command(1, case.initial.current_head),
            case.bindings,
            _change(case, held=held),
            compatibility=case.compatibility,
        )
    assert calls["_publication_content"] == 1
    assert len(decoded) == 1
    _assert_old(case)


@pytest.mark.parametrize("mode", ["component", "collection"])
def test_actual_admitted_byte_ceilings_reject_before_any_new_blob_or_ack(state_example, tmp_path, mode):
    limits = StorageLimits(schema_version=1, maximum_component_bytes=256, maximum_collection_bytes=300)
    case = _case(state_example, tmp_path / mode, limits=limits)
    text = "x" * (1000 if mode == "component" else 100)
    with pytest.raises(ValueError, match="encoded byte ceiling"):
        case.store.commit_transition(
            command(1, case.initial.current_head),
            case.bindings,
            _change(case, text=text),
            compatibility=case.compatibility,
        )
    assert not any(event[0] == "barrier" for event in _Probe.events)
    _assert_old(case)


@pytest.mark.parametrize("phase", ["before_blobs", "after_blob", "after_manifest", "after_head_update", "after_commit"])
def test_actual_optimized_fault_barriers_keep_ack_truth_and_same_command_retry(state_example, tmp_path, phase):
    case = _case(state_example, tmp_path / phase)
    case.store.fault = (phase, 0 if phase == "after_blob" else None)
    request = command(1, case.initial.current_head)
    bodies = []
    with _publication_calls() as calls, pytest.raises(_InterruptedError, match=phase):
        case.store.commit_transition(
            request, case.bindings, _change(case, calls=bodies), compatibility=case.compatibility
        )
    assert calls["_build_for_publish"] == 1
    assert calls["_publication_content"] == (0 if phase == "before_blobs" else 1)
    acknowledged = phase == "after_commit"
    assert case.store.read_head().generation == int(acknowledged)
    reopened = _Store.open(case.store.root)
    _reset()
    with _publication_calls() as retried:
        result = reopened.commit_transition(
            request, case.bindings, _change(case, calls=bodies), compatibility=case.compatibility
        )
    assert result.status == ("retained" if acknowledged else "new")
    assert len(bodies) == (1 if acknowledged else 2)
    assert retried["_build_for_publish"] == (0 if acknowledged else 1)
    assert result.accepted_head.generation == 1
    _Probe.publishing = False
    restored = reopened.load(case.bindings, compatibility=case.compatibility)
    assert [item.value.value for item in restored.values] == [-1, -2]


def test_custom_codec_retains_literal_original_encode_callbacks(state_example, tmp_path):
    case = _case(state_example, tmp_path / "codec", count=1, codec_type=_Codec)
    case.store.commit_transition(
        command(1, case.initial.current_head), case.bindings, _change(case), compatibility=case.compatibility
    )
    assert [event[1] for event in _Probe.events if event[0] == "codec"] == [0, 0, 0, 0, -1, -1, -1]


def test_custom_method_writer_retains_literal_original_callbacks(state_example, tmp_path):
    case = _case(state_example, tmp_path / "writer", count=1, state_type=_WriterState)
    case.store.commit_transition(
        command(1, case.initial.current_head), case.bindings, _change(case), compatibility=case.compatibility
    )
    assert [event[1] for event in _Probe.events if event[0] == "writer"] == [0] * 8 + [-1] * 6


def test_custom_value_construction_hook_retains_original_sequence(state_example, tmp_path, monkeypatch):
    case = _case(state_example, tmp_path / "value-hook", count=1)
    original = ComponentValue.__post_init__
    observed = []

    def construct(self):
        observed.append(self.value.value)
        original(self)

    monkeypatch.setattr(ComponentValue, "__post_init__", construct)
    case.store.commit_transition(
        command(1, case.initial.current_head), case.bindings, _change(case), compatibility=case.compatibility
    )
    assert observed == [0, 0, -1]


def test_before_blobs_outer_binding_mutation_cannot_relabel_captured_bytes(state_example, tmp_path):
    case = _case(state_example, tmp_path / "binding")
    binding = case.bindings[0].codec.binding
    original = binding.state_codec

    def change(phase, _component):
        if phase == "before_blobs":
            object.__setattr__(binding, "state_codec", state_example.reference("foreign-codec"))

    case.store.hook = change
    try:
        with _publication_calls() as calls, pytest.raises(ValueError):
            case.store.commit_transition(
                command(1, case.initial.current_head), case.bindings, _change(case), compatibility=case.compatibility
            )
        assert calls["_publication_content"] == 1
    finally:
        object.__setattr__(binding, "state_codec", original)
    _assert_old(case)


def test_custom_publish_bytes_must_match_actual_admitted_manifest(state_example, tmp_path):
    class DifferentBytes(ComponentCodec):
        def encode(self, state):
            content = super().encode(state)
            if _Probe.publishing and state.value < 0:
                changed = json.loads(content.content)
                changed["value"] += 1
                raw = json.dumps(changed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                return StateBytes(self.binding, raw)
            return content

    case = _case(state_example, tmp_path / "different-bytes", count=1, codec_type=DifferentBytes)
    with pytest.raises(ValueError, match="manifest"):
        case.store.commit_transition(
            command(1, case.initial.current_head), case.bindings, _change(case), compatibility=case.compatibility
        )
    _assert_old(case)


def test_later_transition_ignores_unpassed_public_snapshot_and_reuses_no_old_bytes(state_example, tmp_path):
    case = _case(state_example, tmp_path / "two-calls")
    first = case.store.commit_transition(
        command(1, case.initial.current_head), case.bindings, _change(case), compatibility=case.compatibility
    )
    _Probe.publishing = False
    public = case.store.load(case.bindings, compatibility=case.compatibility)
    original_digests = tuple(item.fingerprint for item in public.manifest.components)
    object.__setattr__(public.values[0].value, "hidden", 6)
    _reset()
    with _publication_calls() as calls:
        second = case.store.commit_transition(
            command(2, first.current_head), case.bindings, _change(case, shift=3), compatibility=case.compatibility
        )
    assert calls == {"_build_for_publish": 1, "_publication_content": 1}
    assert second.accepted_head.generation == 2
    _Probe.publishing = False
    current = case.store.load(case.bindings, compatibility=case.compatibility)
    assert [item.value.value for item in current.values] == [-4, -5]
    assert all(item.value.hidden == 5 for item in current.values)
    assert tuple(item.fingerprint for item in current.manifest.components) != original_digests


@pytest.mark.parametrize("mode", ["subclass", "class", "instance"])
def test_genuine_custom_publish_dispatch_keeps_original_path(state_example, tmp_path, monkeypatch, mode):
    case = _case(state_example, tmp_path / mode)
    original = LocalStateStore._publish
    observed = []

    def publish(self, database, request, prepared):
        observed.append(prepared.manifest.intent.generation)
        return original(self, database, request, prepared)

    if mode == "class":
        monkeypatch.setattr(LocalStateStore, "_publish", publish)
    elif mode == "instance":
        case.store._publish = MethodType(publish, case.store)
    else:

        class OtherStore(_Store):
            _publish = publish

        case.store = OtherStore.open(case.store.root)
    with _publication_calls() as calls:
        case.store.commit_transition(
            command(1, case.initial.current_head), case.bindings, _change(case), compatibility=case.compatibility
        )
    assert observed == [1]
    assert calls == {"_build_for_publish": 0, "_publication_content": 0}
