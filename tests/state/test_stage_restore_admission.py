"""Actual restoration retains full fields, readers, wire bytes and custom hooks."""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from functools import partial
from types import MethodType, SimpleNamespace
from typing import ClassVar, Self

import pytest
from pydantic import Field, StrictInt

from coleman.core.base import CoreContract
from coleman.state import _prepare
from coleman.state.components import ComponentBinding, ComponentCodec, ComponentValue, StagedCollection
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CollectionDefinition, ComponentSpec, StorageLimits
from tests.state.support import command
from tests.state.test_codec_symmetry import layout


class CompleteState(CoreContract):
    amount: Decimal = Decimal("2.00")
    hidden: StrictInt = Field(default=5, exclude=True)
    nullable: str | None = None
    pending: tuple[StrictInt, ...] = (1, -2)
    signed: StrictInt = 0
    zero: float = -0.0
    events: ClassVar[list[str]] = []
    reads: ClassVar[int] = 0
    change_on: ClassVar[int | None] = None
    mutate_target: ClassVar[CompleteState | None] = None

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        cls.reads += 1
        cls.events.append("reader:" + str(json.loads(payload)["signed"]))
        result = super().from_json(payload)
        if cls.reads == cls.change_on:
            if cls.mutate_target is None:
                object.__setattr__(result, "hidden", 6)
            else:
                object.__setattr__(cls.mutate_target, "hidden", 6)
        return result


@pytest.fixture(autouse=True)
def clear_reader_controls():
    CompleteState.events = []
    CompleteState.reads = 0
    CompleteState.change_on = None
    CompleteState.mutate_target = None
    yield
    CompleteState.events = []
    CompleteState.change_on = None
    CompleteState.mutate_target = None


def case_for(example, *, count=1, state_type=CompleteState, codec_type=ComponentCodec):
    """Build real complete inventories and actual independent mutable owners."""
    base = layout(example, state_type)
    events, owners = [], []

    class Owner:
        def __init__(self, name):
            events.append("factory:" + str(name))
            self.name = name
            self.value = None
            owners.append(self)

        def restore_state(self, value):
            events.append("restore:" + str(self.name))
            self.value = state_type.model_validate(value)

        def export_state(self):
            events.append("export:" + str(self.name))
            return self.value

    bindings, values, specs = [], [], []
    for index in range(count):
        state_binding = type(base.codec.binding).model_validate(
            base.codec.binding.model_copy(update={"owner": example.reference("complete-" + str(index))})
        )
        codec = codec_type(state_binding, state_type)
        bindings.append(ComponentBinding(codec, partial(Owner, index), lambda _: None))
        values.append(ComponentValue(codec, state_type(schema_version=1, signed=-3 * index)))
        specs.append(ComponentSpec(binding=state_binding, child=None, roles=("algorithm",), frontier_profile="none"))
    definition = CollectionDefinition.model_validate(base.definition.model_copy(update={"components": tuple(specs)}))
    prepared = _prepare.build(
        definition,
        base.compatibility,
        command(0, None, kind="initialize"),
        tuple(values),
        tuple(bindings),
        StorageLimits(schema_version=1),
    )
    CompleteState.events = events
    CompleteState.reads = 0
    events.clear()
    return SimpleNamespace(
        prepared=prepared,
        bindings=tuple(bindings),
        definition=definition,
        compatibility=base.compatibility,
        events=events,
        owners=owners,
    )


def captured_stages(monkeypatch, *, subclass=None, instance_hook=None):
    """Observe real stage lifetime without replacing any principal operation."""
    stages = []

    def construct(entries):
        stage = (subclass or StagedCollection)(entries)
        if instance_hook is not None:
            stage._export = MethodType(instance_hook, stage)
        stages.append(stage)
        return stage

    monkeypatch.setattr(_prepare, "StagedCollection", construct)
    return stages


def test_stock_stage_preserves_export_all_order_three_actual_readers_and_literal_complete_wire(state_example):
    case = case_for(state_example, count=2)
    stage = _prepare.stage(case.prepared, case.bindings)
    try:
        assert case.events == [
            "factory:0",
            "factory:1",
            "restore:0",
            "restore:1",
            "export:0",
            "reader:0",
            "export:1",
            "reader:-3",
            "reader:0",
            "reader:0",
            "reader:-3",
            "reader:-3",
        ]
        assert CompleteState.reads == 6
        state = stage.owner(case.bindings[0]).value
        assert state.to_json() == (
            '{"amount":"2.00","nullable":null,"pending":[1,-2],"schema_version":1,"signed":0,"zero":-0.0}'
        )
        assert state.zero.hex() == "-0x0.0p+0" and state.amount.as_tuple() == Decimal("2.00").as_tuple()
        assert state.hidden == 5 and state.pending == (1, -2)
    finally:
        stage._close()


def test_complete_signature_inspection_grows_linearly_with_independent_owner_count(state_example, monkeypatch):
    original = _prepare._state_signature
    observations = []
    for count in (1, 4, 8):
        case = case_for(state_example, count=count)
        calls = []

        def inspect(value, calls=calls):
            calls.append(value)
            return original(value)

        monkeypatch.setattr(_prepare, "_state_signature", inspect)
        stage = _prepare.stage(case.prepared, case.bindings)
        stage._close()
        monkeypatch.undo()
        observations.append((count, len(calls)))
    first_count = observations[0][1]
    assert first_count > 0
    assert all(actual == first_count * count for count, actual in observations)
    assert all(actual <= 12 * count for count, actual in observations)


@pytest.mark.parametrize("mutate_original", [False, True])
def test_second_actual_reader_retains_hidden_default_and_capture_before_alias_mutation(
    state_example, monkeypatch, mutate_original
):
    case = case_for(state_example)
    stages = captured_stages(monkeypatch)
    CompleteState.change_on = 3
    if mutate_original:
        original = case.bindings[0].new_owner

        def factory():
            owner = original()
            export = owner.export_state

            def actual():
                value = export()
                CompleteState.mutate_target = value
                return value

            owner.export_state = actual
            return owner

        case.bindings = (replace(case.bindings[0], new_owner=factory),)
    with pytest.raises(ValueError, match="exact complete state"):
        _prepare.stage(case.prepared, case.bindings)
    assert CompleteState.reads == 3
    assert len(stages) == 1 and not stages[0]._active


@pytest.mark.parametrize(
    "update",
    [
        {"signed": 1},
        {"pending": ()},
        {"zero": 0.0},
        {"amount": Decimal("2.0")},
        {"nullable": "changed"},
        {"hidden": 6},
        {"schema_version": 2},
        {"zero": float("nan")},
        {"zero": float("inf")},
    ],
)
def test_restore_cannot_omit_or_change_any_declared_field(state_example, monkeypatch, update):
    case = case_for(state_example)
    original = case.bindings[0].new_owner

    def factory():
        owner = original()
        export = owner.export_state
        owner.export_state = lambda: export().model_copy(update=update)
        return owner

    stages = captured_stages(monkeypatch)
    with pytest.raises(ValueError):
        _prepare.stage(case.prepared, (replace(case.bindings[0], new_owner=factory),))
    assert len(stages) == 1 and not stages[0]._active


@pytest.mark.parametrize("phase", ["factory", "restore", "later-export", "inspector"])
def test_mutating_original_or_earlier_export_is_refused_with_live_stage_closed(state_example, monkeypatch, phase):
    case = case_for(state_example, count=2)
    stages = captured_stages(monkeypatch)
    original = case.bindings[1].new_owner

    def mutate():
        target = case.owners[0].value if phase == "later-export" else case.prepared.values[0].value
        object.__setattr__(target, "nullable", "substituted")

    def factory():
        if phase == "factory":
            mutate()
        owner = original()
        restore, export = owner.restore_state, owner.export_state

        def restoring(value):
            mutate()
            restore(value)

        def exporting():
            mutate()
            return export()

        if phase == "restore":
            owner.restore_state = restoring
        elif phase == "later-export":
            owner.export_state = exporting
        return owner

    changed = replace(case.bindings[1], new_owner=factory)
    if phase == "inspector":
        changed = replace(changed, frontier=lambda _: mutate())
    with pytest.raises(ValueError, match="changed|exact complete"):
        _prepare.stage(case.prepared, (case.bindings[0], changed))
    assert not stages if phase == "inspector" else len(stages) == 1 and not stages[0]._active


@pytest.mark.parametrize("target", ["expected", "earlier-actual"])
def test_reader_cannot_mutate_an_already_compared_or_original_state(state_example, monkeypatch, target):
    case = case_for(state_example, count=2)
    stages = captured_stages(monkeypatch)
    if target == "expected":
        CompleteState.mutate_target = case.prepared.values[0].value
        CompleteState.change_on = 3
    else:
        original = case.bindings[0].new_owner

        def factory():
            owner = original()
            export = owner.export_state

            def actual():
                value = export()
                CompleteState.mutate_target = value
                return value

            owner.export_state = actual
            return owner

        case.bindings = (replace(case.bindings[0], new_owner=factory), case.bindings[1])
        CompleteState.change_on = 6
    with pytest.raises(ValueError, match="changed|exact complete"):
        _prepare.stage(case.prepared, case.bindings)
    assert len(stages) == 1 and not stages[0]._active


@pytest.mark.parametrize("mode", ["subclass", "class", "instance"])
def test_custom_codec_encode_callbacks_retain_exact_old_three_encode_sequence(state_example, monkeypatch, mode):
    called = []
    original = ComponentCodec.encode

    def encode(self, state):
        called.append(state.signed)
        return original(self, state)

    class CustomCodec(ComponentCodec):
        def encode(self, state):
            return encode(self, state)

    case = case_for(state_example, codec_type=CustomCodec if mode == "subclass" else ComponentCodec)
    if mode == "class":
        monkeypatch.setattr(ComponentCodec, "encode", encode)
    elif mode == "instance":
        object.__setattr__(case.bindings[0].codec, "encode", MethodType(encode, case.bindings[0].codec))
    called.clear()
    stage = _prepare.stage(case.prepared, case.bindings)
    stage._close()
    assert called == [0, 0, 0]
    assert case.events == ["factory:0", "restore:0", "export:0", "reader:0", "reader:0", "reader:0"]


@pytest.mark.parametrize("mode", ["class", "instance", "subclass"])
def test_custom_stage_export_hook_runs_once_through_legacy_path(state_example, monkeypatch, mode):
    case = case_for(state_example)
    original = StagedCollection._export
    calls = []

    def export(self):
        calls.append("custom-export")
        return original(self)

    class OtherStage(StagedCollection):
        _export = export

    if mode == "class":
        monkeypatch.setattr(StagedCollection, "_export", export)
    else:
        captured_stages(
            monkeypatch,
            subclass=OtherStage if mode == "subclass" else None,
            instance_hook=export if mode == "instance" else None,
        )
    stage = _prepare.stage(case.prepared, case.bindings)
    stage._close()
    assert calls == ["custom-export"] and CompleteState.reads == 3


@pytest.mark.parametrize("refuse_at", [1, 2, 3])
def test_custom_encode_refusal_keeps_legacy_callback_count_and_closes_stage(state_example, monkeypatch, refuse_at):
    case = case_for(state_example)
    stages = captured_stages(monkeypatch)
    original = ComponentCodec.encode
    calls = []

    def refusing(self, state):
        calls.append(state.signed)
        if len(calls) == refuse_at:
            raise LookupError("actual custom encode refusal")
        return original(self, state)

    monkeypatch.setattr(ComponentCodec, "encode", refusing)
    with pytest.raises(LookupError, match="actual custom"):
        _prepare.stage(case.prepared, case.bindings)
    assert len(calls) == refuse_at
    assert len(stages) == 1 and not stages[0]._active


def test_custom_value_post_init_hook_and_custom_method_writer_are_not_bypassed(state_example, monkeypatch):
    case = case_for(state_example)
    original = ComponentValue.__post_init__
    calls = []

    def post_init(self):
        calls.append("value-hook")
        original(self)

    monkeypatch.setattr(ComponentValue, "__post_init__", post_init)
    stage = _prepare.stage(case.prepared, case.bindings)
    stage._close()
    assert calls == ["value-hook"] and CompleteState.reads == 3
    monkeypatch.undo()

    class MethodWriter(CompleteState):
        writes: ClassVar[int] = 0

        def to_json(self):
            type(self).writes += 1
            return super().to_json()

    case = case_for(state_example, state_type=MethodWriter)
    MethodWriter.writes = 0
    stage = _prepare.stage(case.prepared, case.bindings)
    stage._close()
    assert MethodWriter.writes == 6


def test_actual_store_refuses_bad_restoration_without_ack_or_transition_and_recovers(state_example, tmp_path):
    case = case_for(state_example)
    store = LocalStateStore.create(tmp_path / "actual", case.definition, case.compatibility)
    first = store.initialize(
        command(0, None, kind="initialize"), case.prepared.values, case.bindings, compatibility=case.compatibility
    )
    original = case.bindings[0].new_owner

    def factory():
        owner = original()
        export = owner.export_state
        owner.export_state = lambda: export().model_copy(update={"zero": 0.0})
        return owner

    bodies = []
    request = command(1, first.current_head)
    with pytest.raises(ValueError, match="exact complete state"):
        store.commit_transition(
            request,
            (replace(case.bindings[0], new_owner=factory),),
            lambda _: bodies.append("must-not-execute"),
            compatibility=case.compatibility,
        )
    assert store.read_head() == first.current_head and bodies == []
    result = store.commit_transition(
        request, case.bindings, lambda _: bodies.append("actual-once"), compatibility=case.compatibility
    )
    assert result.status == "new" and bodies == ["actual-once"]
    loaded = store.load(case.bindings, compatibility=case.compatibility).component(case.bindings[0].codec)
    assert loaded.zero.hex() == "-0x0.0p+0" and loaded.pending == (1, -2)
