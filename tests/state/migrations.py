"""Two actual caller-owned layouts and an explicit reversible typed mapping."""

import hashlib
from functools import partial
from pathlib import Path
from typing import Literal

from coleman.core.base import CoreContract
from coleman.objectives.bindings import DefinitionBinding
from coleman.replay.retained import RetainedReplay, rebind_retained
from coleman.replay.retained_contracts import OrdinalRetention, RetainedContinuation
from coleman.state.components import ComponentBinding, ComponentCodec, ComponentValue, child_frontier
from coleman.state.contracts import StateBinding
from coleman.state.storage_contracts import CollectionDefinition, ComponentSpec, ExecutionCompatibility


class WrappedState[P: CoreContract](CoreContract):
    """Known alternative caller-owned layout retaining the entire actual payload."""

    layout: Literal["authored-complete-host-wrapper-v2"]
    payload: P


def wrapped(example, state):
    return WrappedState[example.HostState](schema_version=1, layout="authored-complete-host-wrapper-v2", payload=state)


def target_codec(example):
    return example.reference("authored-owned-ci-host-wrapper-v2")


def definition(example, name):
    return DefinitionBinding(
        reference=example.reference(name),
        definition_fingerprint="sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


class WrappedConsumer:
    def __init__(self, example, child):
        self.example = example
        self.host = example.HostConsumer(child)

    def decide(self, request):
        return self.host.decide(request)

    def update(self, application):
        self.host.update(application)

    def export_state(self):
        return wrapped(self.example, self.host.export_state())

    def restore_state(self, state):
        selected = WrappedState[self.example.HostState]
        if type(state) is not selected:
            raise TypeError("Known wrapped owner requires its exact concrete layout")
        validated = selected.model_validate(state)
        self.host.restore_state(validated.payload)


class WrappedReplayOwner:
    def __init__(self, example, child):
        self.example, self.child = example, child
        self.host = WrappedConsumer(example, child)
        self.replay = RetainedReplay(
            self.host,
            example.policy(child),
            state_type=WrappedState[example.HostState],
            state_codec=target_codec(example),
            retention=OrdinalRetention(
                profile="quiescent-ordinal-v1", source_epoch=example.EPOCH, maximum_tail_records=512
            ),
        )
        self.replay.admit_history(example.history())

    def export_state(self):
        return self.replay.export_continuation()

    def restore_state(self, state):
        host = WrappedConsumer(self.example, self.child)
        replay = RetainedReplay.from_continuation(
            host, state, state_type=WrappedState[self.example.HostState], state_codec=target_codec(self.example)
        )
        self.host, self.replay = host, replay


def target_layout(example, source_definition, bindings, compatibility):
    old = bindings[0]
    implementation = definition(example, "authored-wrapped-ci-owner")
    migration = definition(example, "authored-wrap-complete-ci-state")
    target_binding = StateBinding(
        owner=old.codec.binding.owner,
        implementation=implementation,
        state_codec=example.reference("authored-wrapped-retained-continuation-v2"),
        configuration_fingerprint=old.codec.binding.configuration_fingerprint,
    )
    state_type = RetainedContinuation[WrappedState[example.HostState]]
    target = ComponentBinding(
        ComponentCodec(target_binding, state_type),
        partial(WrappedReplayOwner, example, 0),
        partial(child_frontier, child=example.child_reference(0)),
    )
    specifications = tuple(
        ComponentSpec.model_validate(item.model_copy(update={"binding": target_binding}))
        if item.binding.owner == target_binding.owner
        else item
        for item in source_definition.components
    )
    current_definition = CollectionDefinition.model_validate(
        source_definition.model_copy(update={"components": specifications})
    )
    current_compatibility = ExecutionCompatibility.model_validate(
        compatibility.model_copy(
            update={"owner_implementations": (*compatibility.owner_implementations, implementation, migration)}
        )
    )
    return current_definition, (target, *bindings[1:]), current_compatibility, migration


def migrate_values(example, prepared, old_bindings, new_bindings, migration):
    old = prepared.component(old_bindings[0].codec)
    target = rebind_retained(
        old, wrapped(example, old.host_state), target_codec=target_codec(example), migration=migration
    )
    return (
        ComponentValue(new_bindings[0].codec, target),
        *(ComponentValue(binding.codec, prepared.component(binding.codec)) for binding in new_bindings[1:]),
    )
