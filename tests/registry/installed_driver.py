"""Copied process runner for the real installed registry/state/CI composition."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import time
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from coleman.artifacts import ArtifactWriter
from coleman.core.primitives import canonical_json, fingerprint
from coleman.registry.composition import RecipeFeedback, RegistryCheckpoint
from coleman.registry.lifecycle import LifecycleFact, Operation
from coleman.registry.ownership import DeclaredRootGuard, RootInventory
from coleman.registry.state import OwnedBindings, OwnerInspection
from coleman.state.components import ComponentBinding
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import (
    AcknowledgedHead,
    CollectionManifest,
    CommitIntent,
    CommitKind,
    CommitRequest,
    StorageLimits,
)

PROCESS_OBSERVATION_SECONDS = 600


def load_example():
    name = "copied_public_registry_collection"
    specification = importlib.util.spec_from_file_location(name, Path(__file__).with_name("collection.py"))
    if specification is None or specification.loader is None:
        raise RuntimeError("The explicitly copied public registry example is required")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def command(generation, expected, *, kind: CommitKind = "transition", content="complete-pending32"):
    return CommitRequest(
        schema_version=1,
        expected=expected,
        intent=CommitIntent(
            schema_version=1,
            generation=generation,
            kind=kind,
            content_fingerprint=fingerprint([content, generation, kind]),
        ),
    )


def log(path, event):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(event + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def wait(path, *, seconds=PROCESS_OBSERVATION_SECONDS):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Explicit process barrier was not released: {path.name}")
        time.sleep(0.01)


def watched_owned(layout, diagnostic):
    """Observe actual factories and inspectors, including zero-callback losers."""
    inspected = set()

    def factory(binding):
        log(diagnostic, "factory:" + binding.codec.binding.owner.id)
        return binding.new_owner()

    def inspect(inspection, owner):
        identity = inspection.binding.codec.binding.owner.id
        if identity not in inspected:
            log(diagnostic, "inspect:" + identity)
            inspected.add(identity)
        return inspection.inspect(owner)

    inspections = tuple(
        OwnerInspection(
            ComponentBinding(item.binding.codec, partial(factory, item.binding), item.binding.frontier),
            partial(inspect, item),
        )
        for item in layout.inspections
    )
    return OwnedBindings(layout.ownership, inspections), tuple(item.binding for item in inspections)


def initialize(root):
    example = load_example()
    layout = example.collection_layout()
    owners, values = example.pending_values(layout)
    compatibility = layout.compatibility()
    store = LocalStateStore.create(
        root, layout.definition, compatibility, limits=StorageLimits(schema_version=1, lock_wait_seconds=30.0)
    )
    owned = layout.owned(live=owners)
    receipt = store.initialize(
        command(0, None, kind="initialize", content="registry-pending32"),
        values,
        owned.bindings,
        compatibility=compatibility,
    )
    (root.parent / "initial-head.json").write_text(receipt.current_head.to_json())
    return summary(store, layout, compatibility)


def summary(store, layout, compatibility):
    prepared = store.load(layout.bindings, compatibility=compatibility)
    children = []
    for binding in layout.bindings[:-2]:
        state = prepared.component(binding.codec)
        host = state.host_state
        children.append(
            {
                "child": host.child,
                "D": host.decision_count,
                "U": host.update_count,
                "loss": sum(host.loss_sums),
                "pending": len(host.pending),
                "cadence": host.cadence_count,
                "rng": host.rng.fingerprint,
                "reservoir": [host.reservoir_word, host.reservoir_remaining],
                "virtual_seconds": host.book.spent_seconds,
                "previous_plan": None if host.book.last_plan is None else host.book.last_plan.id,
                "priority": None if host.cache is None else host.cache.value,
                "continuation": state.fingerprint,
            }
        )
    domain = prepared.component(layout.bindings[-2].codec)
    registry = prepared.component(layout.bindings[-1].codec)
    aggregate = canonical_json(
        [prepared.component(binding.codec).model_dump(mode="json") for binding in layout.bindings]
    )
    (store.root.parent / "observed-states.json").write_text(aggregate + "\n")
    registry_blob = next(
        item for item in prepared.manifest.components if item.binding.owner == layout.bindings[-1].codec.binding.owner
    )
    return {
        "generation": store.read_head().generation,
        "children": children,
        "D": sum(child["D"] for child in children),
        "U": sum(child["U"] for child in children),
        "pending": sum(child["pending"] for child in children),
        "loss": sum(child["loss"] for child in children),
        "cadence": sum(child["cadence"] for child in children),
        "virtual_seconds": sum(child["virtual_seconds"] for child in children),
        "actual_seconds": domain.book.spent_seconds,
        "components": len(layout.bindings),
        "declared_roots": len(registry.ownership.resources),
        "registry_origins": len(registry.executions),
        "registry_bytes": registry_blob.byte_count,
        "registry_blob": registry_blob.fingerprint,
        "registry_fingerprint": registry.fingerprint,
        "registry_facts": sum(len(item.facts) for item in registry.registries),
    }


def observe_contention(store, request, bindings, callback, compatibility, diagnostic):
    """Observe the actual finite pre-callback SQLite ownership failure."""
    started = time.monotonic()
    try:
        store.commit_transition(request, bindings, callback, compatibility=compatibility)
    except sqlite3.OperationalError as error:
        if error.sqlite_errorcode != sqlite3.SQLITE_BUSY:
            raise
        outcome, code = "sqlite_busy", error.sqlite_errorcode
    except TimeoutError as error:
        if str(error) != "Local state ownership/commit contention exceeded its finite wait allowance":
            raise
        outcome, code = "ownership_timeout", None
    else:
        raise AssertionError("A contender was accepted while the other writer still held ownership")
    if diagnostic.exists():
        raise AssertionError("Acquisition failure invoked a loser factory, inspector or transition")
    return {
        "outcome": outcome,
        "sqlite_errorcode": code,
        "lock_wait_seconds": store.limits.lock_wait_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "callbacks": [],
    }


def reject(error, diagnostic, *, stage):
    """Report the actual admission boundary without hiding its rejection reason."""
    result = {
        "rejected": str(error),
        "rejection_stage": stage,
        "error_type": type(error).__name__,
        "callbacks": diagnostic.read_text().splitlines() if diagnostic.exists() else [],
    }
    if isinstance(error, ValidationError):
        result["validation_model"] = error.title
        result["validation_errors"] = error.errors(include_url=False, include_context=False)
    print(json.dumps(result, sort_keys=True))


def publish_future_registry(root):
    """Prepare an owned generation-zero fixture whose registry bytes alone use v2.

    The selected v1 codec deliberately does not encode these future bytes. The
    actual immutable writer publishes them with exact SHA/count receipts, and
    the typed manifest/head retain every other component and binding. This test
    fixture follows the state boundary tests' acknowledged-manifest replacement;
    it is not a supported application migration or an accepted learner update.
    """
    example = load_example()
    layout = example.collection_layout()
    store = LocalStateStore.open(root)
    old_head = store.read_head()
    if old_head is None or old_head.generation != 0:
        raise AssertionError("Future-schema fixture requires the copied acknowledged generation zero")
    old_manifest_path = root / "manifests" / (old_head.manifest_fingerprint[7:] + ".json")
    manifest = CollectionManifest.from_json(old_manifest_path.read_bytes())
    binding = layout.bindings[-1].codec.binding
    old_blob = next(item for item in manifest.components if item.binding == binding)
    old_blob_path = root / "blobs" / (old_blob.fingerprint[7:] + ".json")
    raw = json.loads(old_blob_path.read_bytes())
    raw["schema_version"] = 2
    content = canonical_json(raw).encode("utf-8")
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    writer = ArtifactWriter(root)
    writer.write_immutable_stream(
        root / "blobs" / (digest[7:] + ".json"), (content,), maximum_bytes=store.limits.maximum_component_bytes
    )
    changed = type(old_blob).model_validate(
        old_blob.model_copy(update={"fingerprint": digest, "byte_count": len(content)})
    )
    manifest = CollectionManifest.model_validate(
        manifest.model_copy(
            update={"components": tuple(changed if item.binding == binding else item for item in manifest.components)}
        )
    )
    writer.write_immutable(root / "manifests" / (manifest.fingerprint[7:] + ".json"), manifest.to_json())
    target = AcknowledgedHead.model_validate(old_head.model_copy(update={"manifest_fingerprint": manifest.fingerprint}))
    with sqlite3.connect(root / "acknowledgements.sqlite3") as connection:
        assert connection.execute("UPDATE state SET head=? WHERE singleton=1", (target.to_json(),)).rowcount == 1
        assert connection.execute("UPDATE receipts SET head=? WHERE generation=0", (target.to_json(),)).rowcount == 1
    (root.parent / "initial-head.json").write_text(target.to_json())
    # Remove only the two superseded fixture artifacts before the unchanged-byte
    # baseline; ordinary reserve/GC may otherwise prune these unreferenced files.
    old_blob_path.unlink()
    old_manifest_path.unlink()
    if LocalStateStore.open(root).read_head() != target:
        raise AssertionError("Well-hashed future component must pass metadata-only open")
    result = {
        "schema_version": 2,
        "generation": target.generation,
        "registry_blob": digest,
        "registry_bytes": len(content),
        "manifest_fingerprint": manifest.fingerprint,
        "metadata_opened": True,
    }
    (root.parent / "future-publication.json").write_text(json.dumps(result, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


def transition(root, *, fault=None, actor="attempt", divergent=False):
    example = load_example()
    layout = example.collection_layout()
    diagnostic = root.parent / (actor + "-calls.log")
    owned, selected = watched_owned(layout, diagnostic)
    compatibility = layout.compatibility()
    registry_stream = False

    class FaultStore(LocalStateStore):
        def _barrier(self, phase, *, component=None):
            nonlocal registry_stream
            if fault == "registry-mid-stream" and phase == "after_blob" and component == 32:
                registry_stream = True
            selected = phase if component is None else phase + ":" + str(component)
            if fault == selected:
                (root.parent / "fault-barrier.json").write_text(json.dumps({"phase": phase, "component": component}))
                os._exit(87)
            if fault == "hold-lock" and phase == "before_blobs":
                (root.parent / "lock-held").write_text("actual callback completed under SQLite ownership")
                wait(root.parent / "release-lock")

    if fault == "registry-mid-stream":
        original_stream = ArtifactWriter.write_immutable_stream

        def interrupted(self: ArtifactWriter, path: Path, chunks: Iterable[bytes], *, maximum_bytes: int) -> None:
            if not registry_stream:
                return original_stream(self, path, chunks, maximum_bytes=maximum_bytes)

            def real_chunks():
                for chunk in chunks:
                    yield chunk[: max(1, len(chunk) // 2)]
                    (root.parent / "fault-barrier.json").write_text(
                        json.dumps({"phase": "registry-mid-stream", "component": 33})
                    )
                    os._exit(88)

            return original_stream(self, path, real_chunks(), maximum_bytes=maximum_bytes)

        cast(Any, ArtifactWriter).write_immutable_stream = interrupted

    (root.parent / (actor + "-attempt")).write_text("entering actual owned-store operation")
    try:
        store = FaultStore.open(root)
    except (ValueError, OSError) as error:
        reject(error, diagnostic, stage="open")
        return
    expected = AcknowledgedHead.from_json((root.parent / "initial-head.json").read_bytes())
    request = command(1, expected, content="divergent" if divergent else "complete-pending32")

    def callback(stage):
        log(diagnostic, "actual-transition")
        owners = tuple(owned.owner(stage, binding) for binding in selected)
        example.complete_pending(layout, owners)

    if fault == "contend":
        (root.parent / "loser-ready").write_text("metadata handle opened before the competing writer")
        wait(root.parent / "start-contention")
        observed = observe_contention(store, request, owned.bindings, callback, compatibility, diagnostic)
        observed_path = root.parent / "loser-busy.json"
        temporary = observed_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(observed, sort_keys=True))
        temporary.replace(observed_path)
        wait(root.parent / "retry-loser")
    try:
        receipt = store.commit_transition(request, owned.bindings, callback, compatibility=compatibility)
    except (ValueError, OSError) as error:
        reject(error, diagnostic, stage="commit")
        return
    result = summary(store, layout, compatibility)
    result["status"] = receipt.status
    print(json.dumps(result, sort_keys=True))


def inspect(root):
    example = load_example()
    layout = example.collection_layout()
    print(json.dumps(summary(LocalStateStore.open(root), layout, layout.compatibility()), sort_keys=True))


def direct_traces(root):
    """Compare every actual isolated/interleaved/reordered and restored child."""
    example = load_example()
    layout = example.collection_layout()
    reference_owners, reference_values = example.pending_values(layout, isolated=True)
    expected = tuple(item.value.to_json() for item in reference_values[:-2])
    for order in (tuple(range(32)), tuple(reversed(range(32)))):
        old, values = example.pending_values(layout, order=order)
        if tuple(item.value.to_json() for item in values[:-2]) != expected:
            raise AssertionError("Interleaving/reordering changed an actual isolated child's complete trace")
        if values[-1].value.to_json() != reference_values[-1].value.to_json():
            raise AssertionError("Scheduling unrelated tasks changed their complete canonical outer origins")
        guard = DeclaredRootGuard(layout.ownership, inventory(layout, old))
        fresh = []
        for binding in layout.bindings:
            fresh.append(binding.new_owner())
            guard.check_partial(inventory(layout, tuple(fresh)).roots)
        guard.check(inventory(layout, tuple(fresh)))
        for binding, value, owner in zip(layout.bindings, values, fresh, strict=True):
            owner.restore_state(binding.codec.decode(binding.codec.encode(value.value).content))
            guard.check(inventory(layout, tuple(fresh)))
        if tuple(owner.export_state().to_json() for owner in fresh[:-2]) != expected:
            raise AssertionError("Actual complete restore changed a child continuation")
        # Existing pending traces keep their original outer producing epochs.
        example.complete_pending(layout, tuple(fresh))
        example.complete_pending(layout, old)
        if tuple(owner.export_state().to_json() for owner in fresh[:-2]) != tuple(
            owner.export_state().to_json() for owner in old[:-2]
        ):
            raise AssertionError("Resumed delayed feedback differs from uninterrupted actual replay")
    untouched = reference_owners[1].export_state().to_json()
    reference_owners[0].restore_state(layout.bindings[0].new_owner().export_state())
    if reference_owners[1].export_state().to_json() != untouched:
        raise AssertionError("Resetting another actual child changed unrelated complete state/RNG")
    result = {
        "children": 32,
        "orders": 2,
        "origins": len(reference_values[-1].value.executions),
        "preserved_complete_child_traces": True,
        "restored_pending": 47,
    }
    (root.parent / "direct-qualification.json").write_text(json.dumps(result, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


def inventory(layout, owners):
    if len(owners) > len(layout.inspections):
        raise ValueError("Actual owner inventory exceeds the selected declared component capacity")
    reports = tuple(inspection.inspect(owner) for inspection, owner in zip(layout.inspections, owners, strict=False))
    recipes = {recipe.child.id: recipe for report in reports for recipe in report.recipes}
    return RootInventory(
        tuple(sorted(recipes.values(), key=lambda recipe: recipe.child.id)),
        tuple(root for report in reports for root in report.roots),
    )


def blocked_lifecycle_emissions(layout, live):
    """Check inactive, unavailable and retired gates on the same real owners."""
    registry, owner = live[-1], live[0]
    host = layout.host
    registry.inactivate_next(owner)
    rejected = []
    transitions: tuple[tuple[str, tuple[Operation, ...]], ...] = (
        ("inactive", ()),
        ("unavailable", ("resume", "unavailable")),
        ("retired", ("retire",)),
    )
    for status, operations in transitions:
        for operation in operations:
            state = next(
                item
                for item in registry.export_state().registries
                if item.policy.task == host.child_reference(owner.child)
            )
            recipe = next(item for item in layout.ownership.recipes if item.child == state.policy.task)
            boundary = owner.replay.state.decision_count
            registry.replace_registry(
                state.append(
                    LifecycleFact(
                        schema_version=1,
                        event_id="gate-" + operation,
                        sequence=len(state.facts),
                        member=recipe.expert.reference,
                        definition_fingerprint=recipe.expert.definition_fingerprint,
                        operation=operation,
                        effective_from_round=boundary,
                        admitted_before_round=boundary,
                        known_at=host.instant(boundary + 1),
                        reason="explicit-pre-emission-gate-check",
                        state_epoch=None,
                        predecessor_epoch=None,
                        disposition=None,
                    )
                )
            )
        before = tuple(value.export_state().to_json() for value in live)
        try:
            registry.emit(owner, at=owner.replay.state.decision_count + 1)
        except ValueError:
            rejected.append(status + "-before-emission")
        else:
            raise AssertionError(status + " recipe reached the actual adaptive emission")
        if tuple(value.export_state().to_json() for value in live) != before:
            raise AssertionError("Rejected lifecycle emission changed a child, RNG, path or registry origin")
    return rejected


def reject_policy_signed_zero(layout):
    """Require canonical policy identity even when Python model equality agrees."""
    owner = layout.bindings[0].new_owner()
    before = owner.host.export_state().to_json()
    original = owner.replay.state.policy
    scalar = original.capability.required_values[0].model_copy(update={"minimum": -0.0})
    capability = original.capability.model_copy(update={"required_values": (scalar,)})
    policy = type(original).model_validate(original.model_copy(update={"capability": capability}))
    if policy != original or policy.fingerprint == original.fingerprint:
        raise AssertionError("The actual valid signed-zero policy fixture did not retain distinct canonical content")
    owner.replay._state = owner.replay.state.model_copy(update={"policy": policy})  # pylint: disable=protected-access
    try:
        layout.inspections[0].inspect(owner)
    except ValueError:
        if owner.host.export_state().to_json() != before:
            raise AssertionError("Rejected policy inspection changed actual host state") from None
        return "policy-signed-zero"
    raise AssertionError("Numerically equal but canonically changed actual policy passed ownership inspection")


def rejection_checks(root):
    """Reject actual wrong joins and staged aliases without altering live owners."""
    example = load_example()
    layout = example.collection_layout()
    live, values = example.pending_values(layout)
    original = tuple(owner.export_state().to_json() for owner in live)
    origin = values[-1].value.executions[0]
    different_recipe = next(recipe for recipe in layout.ownership.recipes if recipe.child != origin.recipe.child)
    rejected = []
    for field, changed in (
        ("child", different_recipe.child),
        ("expert", different_recipe.expert),
        ("controls", different_recipe.expert),
        ("state_epoch", layout.host.reference("wrong-epoch")),
    ):
        recipe = origin.recipe.model_copy(update={field: changed})
        candidate = origin.model_copy(update={"recipe": recipe})
        try:
            RegistryCheckpoint.model_validate(
                values[-1].value.model_copy(
                    update={
                        "executions": (candidate, *values[-1].value.executions[1:]),
                    }
                )
            )
        except ValueError:
            rejected.append(field)
        else:
            raise AssertionError("Actual checkpoint accepted a substituted " + field)
    wrong_catalogue = origin.snapshot.catalogue.model_copy(update={"definition_fingerprint": fingerprint("wrong")})
    candidate = origin.model_copy(
        update={"snapshot": origin.snapshot.model_copy(update={"catalogue": wrong_catalogue})}
    )
    try:
        RegistryCheckpoint.model_validate(
            values[-1].value.model_copy(
                update={
                    "executions": (candidate, *values[-1].value.executions[1:]),
                }
            )
        )
    except ValueError:
        rejected.append("catalogue")
    else:
        raise AssertionError("Actual checkpoint accepted a stale catalogue origin")
    host = layout.host
    record = live[0].replay.state.decisions[0]
    offered = host.feedback(record, decisions=live[0].replay.state.decision_count, available=5, value=0.5)
    origin = values[-1].value.for_execution(record.executed.fingerprint)
    wrong_producer = next(item for item in values[-1].value.executions if item.recipe.child != origin.recipe.child)
    try:
        RecipeFeedback(schema_version=1, producer_fingerprint=origin.fingerprint, batch=offered.batch).validate_for(
            wrong_producer, host.capability(), apply_at=host.instant(5)
        )
    except ValueError:
        rejected.append("feedback-child")
    else:
        raise AssertionError("Actual outer producer accepted another child's feedback")
    for role in ("host", "replay"):
        fresh = tuple(binding.new_owner() for binding in layout.bindings)
        setattr(fresh[0], role, getattr(live[0], role))
        guard = DeclaredRootGuard(layout.ownership, inventory(layout, live))
        try:
            guard.check(inventory(layout, fresh))
        except ValueError:
            rejected.append("live-" + role)
        else:
            raise AssertionError("Actual distinct wrappers hid a retained live alias")
    if tuple(owner.export_state().to_json() for owner in live) != original:
        raise AssertionError("Rejected joins/aliases changed retained actual live owners")
    raw = json.loads(values[-1].value.to_json())
    raw["schema_version"] = 2
    try:
        RegistryCheckpoint.from_json(canonical_json(raw))
    except ValueError:
        rejected.append("future-registry")
    else:
        raise AssertionError("Actual installed registry codec accepted a future version")
    rejected.extend(blocked_lifecycle_emissions(layout, live))
    rejected.append(reject_policy_signed_zero(layout))
    print(json.dumps({"rejected": rejected, "unchanged_live_owners": len(live)}, sort_keys=True))


def main():
    mode, root = sys.argv[1], Path(sys.argv[2])
    root.parent.mkdir(parents=True, exist_ok=True)
    if mode == "initialize32":
        print(json.dumps(initialize(root), sort_keys=True))
    elif mode == "inspect":
        inspect(root)
    elif mode == "direct-traces":
        direct_traces(root)
    elif mode == "rejections":
        rejection_checks(root)
    elif mode == "publish-future-registry":
        publish_future_registry(root)
    else:
        fault = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "none" else None
        actor = sys.argv[4] if len(sys.argv) > 4 else "attempt"
        divergent = len(sys.argv) > 5 and sys.argv[5] == "divergent"
        transition(root, fault=fault, actor=actor, divergent=divergent)


if __name__ == "__main__":
    main()
