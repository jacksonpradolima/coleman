"""Bounded real-process qualification of the public shipped32-mixture example."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import sqlite3
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

from coleman.core.primitives import canonical_json, decode_json
from coleman.core.reading import bounded_json_bytes
from coleman.registry.ownership import DeclaredRootGuard, RootInventory
from coleman.registry.state import OwnedBindings, OwnerInspection
from coleman.state.components import ComponentBinding, ComponentValue
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import AcknowledgedHead, CommitIntent, CommitKind, CommitRequest, StorageLimits

PROCESS_OBSERVATION_SECONDS = 600


def load_example():
    name = "authored_installed_learning_collection"
    if name not in sys.modules:
        copied = Path(__file__).with_name("collection.py")
        source = copied if copied.is_file() else Path(__file__).parents[2] / "examples/learning/collection.py"
        spec = importlib.util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError("The explicitly copied public collection is required")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def atomic_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    if len(encoded.encode("utf-8")) > 64 * 1024**2:
        raise ValueError("The complete qualification artifact exceeds its 64 MiB ceiling")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def command(generation, expected, *, kind: CommitKind = "transition", content="complete-shipped32"):
    return CommitRequest(
        schema_version=1,
        expected=expected,
        intent=CommitIntent(
            schema_version=1,
            generation=generation,
            kind=kind,
            content_fingerprint=load_example().fingerprint([content, generation, kind]),
        ),
    )


def state_wires(values):
    wires = tuple(item.codec.encode(item.value).content for item in values)
    if sum(map(len, wires)) > 64 * 1024**2:
        raise ValueError("The complete actual collection exceeds its 64 MiB profile")
    return wires


def states(layout, owners):
    return tuple(
        ComponentValue(binding.codec, owner.export_state())
        for binding, owner in zip(layout.bindings, owners, strict=True)
    )


def observation(layout, values):
    records = []
    for spec, item in zip(layout.specifications, values[:-1], strict=True):
        host = item.value.host_state
        direct = host.direct if isinstance(spec.configuration, load_example().ControlledConfig) else host
        payload = direct.method.payload
        records.append(
            {
                "child": spec.child,
                "profile": spec.configuration.profile,
                "D": item.value.state.decision_count,
                "U": payload.update_count,
                "pending": len(direct.pending),
                "R": getattr(payload, "release_count", None),
                "usage": None
                if not hasattr(payload, "exact_path")
                else [row.atoms_decimal for row in payload.exact_path.used],
            }
        )
    return {
        "children": len(records),
        "components": len(values),
        "declared_roots": len(layout.ownership.resources),
        "D": sum(item["D"] for item in records),
        "U": sum(item["U"] for item in records),
        "pending": sum(item["pending"] for item in records),
        "R_controlled": sum(item["R"] for item in records if item["R"] is not None),
        "origins": len(values[-1].value.executions),
        "records": records,
    }


def inventory(layout, owners):
    reports = tuple(item.inspect(owner) for item, owner in zip(layout.inspections, owners, strict=True))
    return RootInventory(owners[-1].recipe_bindings, tuple(root for report in reports for root in report.roots))


def restore(layout, values, live):
    owners = tuple(binding.new_owner() for binding in layout.bindings)
    guard = DeclaredRootGuard(layout.ownership, inventory(layout, live))
    guard.check(inventory(layout, owners))
    for binding, owner, value in zip(layout.bindings, owners, values, strict=True):
        owner.restore_state(binding.codec.decode(binding.codec.encode(value.value).content))
        guard.check(inventory(layout, owners))
    if state_wires(states(layout, owners)) != state_wires(values):
        raise AssertionError("Complete actual restored state differs from its original bytes")
    return owners


def _identity(layout):
    return {
        "profile": "actual-shipped32-complete-recovery-v1",
        "definition": layout.definition.fingerprint,
        "compatibility": layout.compatibility().fingerprint,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _write_values(path, values):
    wires = state_wires(values)
    atomic_json(path, [decode_json(wire) for wire in wires])
    return {
        "name": path.name,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "component_sha256": [hashlib.sha256(wire).hexdigest() for wire in wires],
    }


def _read_values(layout, root, artifact):
    path = root / artifact["name"]
    payload = bounded_json_bytes(path.read_bytes(), 64 * 1024**2)
    if hashlib.sha256(payload).hexdigest() != artifact["file_sha256"]:
        raise AssertionError("A complete phase state artifact changed bytes")
    decoded = decode_json(payload)
    if type(decoded) is not list or len(decoded) != len(layout.bindings):
        raise AssertionError("A phase must retain all33 complete component states")
    values = tuple(
        ComponentValue(binding.codec, binding.codec.decode(canonical_json(value).encode("utf-8")))
        for binding, value in zip(layout.bindings, decoded, strict=True)
    )
    if [hashlib.sha256(wire).hexdigest() for wire in state_wires(values)] != artifact["component_sha256"]:
        raise AssertionError("A complete component differs from its producing phase hash")
    return values


def _previous(root, order, phase, identity):
    result = decode_json(bounded_json_bytes((root / f"{order}-{phase}.json").read_bytes(), 1024 * 1024))
    if result["identity"] != identity or (result["order"], result["phase"]) != (order, phase):
        raise AssertionError("A phase receipt belongs to another source, profile or scheduling order")
    return result


def _bootstrap(layout, values):
    owners = tuple(binding.new_owner() for binding in layout.bindings)
    guard = DeclaredRootGuard(layout.ownership)
    guard.check(inventory(layout, owners))
    for owner, item in zip(owners, values, strict=True):
        owner.restore_state(item.value)
        guard.check(inventory(layout, owners))
    if state_wires(states(layout, owners)) != state_wires(values):
        raise AssertionError("A bootstrapped actual32 live collection lost complete state")
    return owners


def _reference_phase(layout, root, order):
    example = load_example()
    owners, initial = example.pending_values(layout, isolated=True)
    DeclaredRootGuard(layout.ownership).check(inventory(layout, owners))
    before = example.summary(layout, owners)
    _require_full_counts(before, pending=True)
    initial_file = _write_values(root / f"{order}-reference-initial.json", initial)
    example.complete_pending(layout, owners)
    after = example.summary(layout, owners)
    _require_full_counts(after, pending=False)
    final_file = _write_values(root / f"{order}-reference-final.json", states(layout, owners))
    return {"before": before, "after": after, "initial": initial_file, "final": final_file}


def _interleaved_phase(layout, root, order, identity):
    example = load_example()
    reference = _previous(root, order, "reference", identity)
    initial = _read_values(layout, root, reference["initial"])
    selected = tuple(spec.child for spec in layout.specifications)
    selected = tuple(reversed(selected)) if order == "reverse" else selected
    owners, values = example.pending_values(layout, order=selected)
    DeclaredRootGuard(layout.ownership).check(inventory(layout, owners))
    if state_wires(values) != state_wires(initial):
        raise AssertionError("The actual interleaved32 prefix differs from its complete isolated reference")
    return {"initial": _write_values(root / f"{order}-interleaved-initial.json", values)}


def _restored_phase(layout, root, order, identity):
    previous = _previous(root, order, "interleaved", identity)
    values = _read_values(layout, root, previous["initial"])
    live = _bootstrap(layout, values)
    first = restore(layout, values, live)
    second = restore(layout, states(layout, first), first)
    live_ids = {id(item.root) for item in inventory(layout, live).roots}
    if live_ids.intersection(id(item.root) for item in inventory(layout, second).roots):
        raise AssertionError("A repeated restored chain aliases the retained actual live roots")
    return {
        "initial": _write_values(root / f"{order}-restored-initial.json", states(layout, second)),
        "repeated_complete_restores": 2,
        "actual_live_root_checks": True,
    }


def _completed_phase(layout, root, order, identity):
    example = load_example()
    reference = _previous(root, order, "reference", identity)
    restored = _previous(root, order, "restored", identity)
    original = _read_values(layout, root, restored["initial"])
    expected = _read_values(layout, root, reference["final"])
    owners = _bootstrap(layout, original)
    example.complete_pending(layout, owners)
    after = example.summary(layout, owners)
    _require_full_counts(after, pending=False)
    values = states(layout, owners)
    if state_wires(values) != state_wires(expected):
        raise AssertionError("Restored actual32 delayed completion differs from uninterrupted full state")
    final_file = _write_values(root / f"{order}-completed-final.json", values)
    other_before = state_wires(values[1:])
    owners[0].restore_state(original[0].value)
    DeclaredRootGuard(layout.ownership).check(inventory(layout, owners))
    if state_wires(states(layout, owners)[1:]) != other_before:
        raise AssertionError("Explicit restore of one complete child changed another child or registry")
    return {"after": after, "final": final_file, "other_children_reset_unchanged": True}


def direct(root: Path, order: str, phase: str):
    if order not in {"forward", "reverse"}:
        raise ValueError("Direct32 order must be forward or reverse")
    layout = load_example().collection_layout()
    identity = _identity(layout)
    result: dict[str, Any]
    if phase == "reference":
        result = _reference_phase(layout, root, order)
    elif phase == "interleaved":
        result = _interleaved_phase(layout, root, order, identity)
    elif phase == "restored":
        result = _restored_phase(layout, root, order, identity)
    elif phase == "completed":
        result = _completed_phase(layout, root, order, identity)
    else:
        raise ValueError("Unknown explicit direct32 qualification phase")
    result.update(identity=identity, order=order, phase=phase)
    atomic_json(root / f"{order}-{phase}.json", result)
    return result


def _require_full_counts(result, *, pending):
    expected = {
        "children": 32,
        "components": 33,
        "declared_roots": 173,
        "D": 128,
        "U": 64 if pending else 128,
        "pending": 64 if pending else 0,
        "R_controlled": 18 if pending else 36,
        "origins": 128,
    }
    for key, value in expected.items():
        if result[key] != value:
            raise AssertionError(f"Actual32 {key} differs from the declared full fixture")
    if not result["moving_exact_usage_positive"] or not result["optimal_holds_spend_zero"]:
        raise AssertionError("Actual positive-L1 moving/holding ledgers failed")
    if (result["moving_children"], result["holding_children"]) != (6, 6):
        raise AssertionError("Both required actual controlled groups must retain all six children")


def watched_owned(layout, calls):
    inspected = set()

    def factory(binding):
        calls.append("factory:" + binding.codec.binding.owner.id)
        return binding.new_owner()

    def inspect(item, owner):
        identity = item.binding.codec.binding.owner.id
        if identity not in inspected:
            calls.append("inspect:" + identity)
            inspected.add(identity)
        return item.inspect(owner)

    inspections = tuple(
        OwnerInspection(
            ComponentBinding(item.binding.codec, partial(factory, item.binding), item.binding.frontier),
            partial(inspect, item),
        )
        for item in layout.inspections
    )
    return OwnedBindings(layout.ownership, inspections), tuple(item.binding for item in inspections)


def initialize(root: Path, *, layout=None):
    example = load_example()
    layout = example.collection_layout() if layout is None else layout
    owners, values = example.pending_values(layout)
    compatibility = layout.compatibility()
    store = LocalStateStore.create(
        root, layout.definition, compatibility, limits=StorageLimits(schema_version=1, lock_wait_seconds=30.0)
    )
    guarded = layout.owned(live=owners)
    receipt = store.initialize(
        command(0, None, kind="initialize", content="pending-shipped32"),
        values,
        guarded.bindings,
        compatibility=compatibility,
    )
    atomic_json(root.parent / "initial-head.json", receipt.current_head.model_dump(mode="json"))
    initial = store.load(layout.bindings, compatibility=compatibility)
    atomic_json(root.parent / "initial-states.json", [item.value.model_dump(mode="json") for item in initial.values])
    return store, layout, compatibility, receipt.current_head


def _transition(store, layout, compatibility, expected, calls, *, content="complete-shipped32", fail=False):
    example = load_example()
    owned, bindings = watched_owned(layout, calls)

    def apply(stage):
        calls.append("transition")
        owners = tuple(owned.owner(stage, binding) for binding in bindings)
        example.complete_pending(layout, owners)
        if fail:
            raise RuntimeError("injected-after-real-complete-before-acknowledgement")

    receipt = store.commit_transition(
        command(1, expected, content=content), owned.bindings, apply, compatibility=compatibility
    )
    return receipt


def _staged_store(root: Path, layout):
    layout = load_example().collection_layout() if layout is None else layout
    store = LocalStateStore.open(root)
    compatibility = layout.compatibility()
    initial = AcknowledgedHead.from_json((root.parent / "initial-head.json").read_bytes())
    if store.read_head() != initial:
        raise AssertionError("The staged fault case requires its exact original acknowledged head")
    return store, layout, compatibility, initial


def failed_transition(root: Path, *, layout=None):
    store, layout, compatibility, initial = _staged_store(root, layout)
    before = state_wires(store.load(layout.bindings, compatibility=compatibility).values)
    calls = []
    try:
        _transition(store, layout, compatibility, initial, calls, fail=True)
    except RuntimeError as error:
        if str(error) != "injected-after-real-complete-before-acknowledgement":
            raise
    else:
        raise AssertionError("The actual injected pre-acknowledgement failure was not raised")
    if (
        store.read_head() != initial
        or state_wires(store.load(layout.bindings, compatibility=compatibility).values) != before
    ):
        raise AssertionError("A failed actual33-owner transition published a false completion")
    result = {
        "identity": _identity(layout),
        "original_head": initial.model_dump(mode="json"),
        "initial_sha256": [hashlib.sha256(wire).hexdigest() for wire in before],
        "callbacks": calls,
        "failure_kept_original_head_and_all_states": True,
    }
    atomic_json(root.parent / "failed-transition.json", result)
    return result


def retry_transition(root: Path, *, layout=None):
    store, layout, compatibility, initial = _staged_store(root, layout)
    failure_path = root.parent / "failed-transition.json"
    failure_bytes = bounded_json_bytes(failure_path.read_bytes(), 1024 * 1024)
    failure = decode_json(failure_bytes)
    if failure["identity"] != _identity(layout) or failure["original_head"] != initial.model_dump(mode="json"):
        raise AssertionError("The retry must join its actual failed transition and complete source identity")
    before = state_wires(store.load(layout.bindings, compatibility=compatibility).values)
    if [hashlib.sha256(wire).hexdigest() for wire in before] != failure["initial_sha256"]:
        raise AssertionError("The cross-process retry lost the complete unchanged failed-transition state")
    calls = []
    receipt = _transition(store, layout, compatibility, initial, calls)
    values = store.load(layout.bindings, compatibility=compatibility).values
    snapshot = state_wires(values)
    callbacks = len(calls)
    retained = _transition(store, layout, compatibility, initial, calls)
    if retained.status != "retained" or len(calls) != callbacks:
        raise AssertionError("Exact acknowledged retry invoked a factory, inspector or transition")
    if state_wires(store.load(layout.bindings, compatibility=compatibility).values) != snapshot:
        raise AssertionError("Retained retry changed the complete acknowledged source state")
    result = {
        "status": receipt.status,
        "retry": retained.status,
        "generation": receipt.current_head.generation,
        "callbacks": failure["callbacks"] + calls,
        "failure_kept_original_head_and_all_states": True,
        "failed_transition_sha256": hashlib.sha256(failure_bytes).hexdigest(),
        "retained_retry_callbacks": 0,
        "observation": observation(layout, values),
        "final_sha256": [hashlib.sha256(wire).hexdigest() for wire in snapshot],
    }
    atomic_json(root.parent / "failure-retry.json", result)
    atomic_json(root.parent / "completed-states.json", [item.value.model_dump(mode="json") for item in values])
    return result


def failed_then_retry(root: Path, *, layout=None):
    initialize(root, layout=layout)
    failed_transition(root, layout=layout)
    return retry_transition(root, layout=layout)


def wait(path: Path, *, seconds=PROCESS_OBSERVATION_SECONDS):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Actual writer barrier was not released: " + path.name)
        time.sleep(0.01)


def writer(root: Path, token: str, *, divergent: bool):
    example = load_example()
    layout = example.collection_layout()
    compatibility = layout.compatibility()

    class HeldStore(LocalStateStore):
        def _barrier(self, phase, *, component=None):
            if token == "winner" and phase == "after_manifest":
                atomic_json(root.parent / "lock-held.json", {"phase": phase, "acknowledged": False})
                wait(root.parent / "release-lock")

    store = HeldStore.open(root)
    expected = AcknowledgedHead.from_json((root.parent / "initial-head.json").read_bytes())
    atomic_json(root.parent / f"{token}-ready.json", {"ready": True})
    calls = []
    content = "divergent-shipped32" if divergent else "complete-shipped32"
    if token == "loser":
        wait(root.parent / "start-contention")
        started = time.monotonic()
        try:
            _transition(store, layout, compatibility, expected, calls, content=content)
        except (sqlite3.OperationalError, TimeoutError) as error:
            if calls:
                raise AssertionError("An actual locked-out writer invoked an owner callback") from error
            busy = {
                "exception": type(error).__name__,
                "sqlite_errorcode": getattr(error, "sqlite_errorcode", None),
                "elapsed_seconds": time.monotonic() - started,
                "lock_wait_seconds": store.limits.lock_wait_seconds,
                "callbacks": calls,
            }
            if isinstance(error, sqlite3.OperationalError) and busy["sqlite_errorcode"] != sqlite3.SQLITE_BUSY:
                raise
            atomic_json(root.parent / "loser-busy.json", busy)
        else:
            raise AssertionError("The actual contending writer was not refused under held SQLite ownership")
        wait(root.parent / "retry-loser")
    try:
        receipt = _transition(store, layout, compatibility, expected, calls, content=content)
    except ValueError as error:
        if "conflicts with its retained intent" not in str(error) or calls:
            raise
        result = {"status": "rejected", "callbacks": calls, "reason": str(error)}
    else:
        result = {"status": receipt.status, "generation": receipt.current_head.generation, "callbacks": calls}
    atomic_json(root.parent / f"{token}-result.json", result)
    return result


def _require_installed():
    import coleman.learning.reference as module

    source = Path(module.__file__).resolve()
    if not source.is_relative_to(Path(sys.prefix).resolve()) or source.is_relative_to(Path.cwd().resolve()):
        raise AssertionError("The driver requires the actual noneditable installed SDK outside the checkout")
    metadata = importlib.metadata.distribution("coleman")
    direct_url = json.loads(metadata.read_text("direct_url.json") or "{}")
    if direct_url.get("dir_info", {}).get("editable"):
        raise AssertionError("An editable package cannot qualify this installed driver")


def _store_observation(store, layout, compatibility):
    values = store.load(layout.bindings, compatibility=compatibility).values
    result = observation(layout, values)
    result["component_sha256"] = [hashlib.sha256(wire).hexdigest() for wire in state_wires(values)]
    return result


def main():
    _require_installed()
    mode, root = sys.argv[1], Path(sys.argv[2])
    if mode == "direct":
        result = direct(root, sys.argv[3], sys.argv[4])
    elif mode == "initialize":
        store, layout, compatibility, head = initialize(root)
        result = _store_observation(store, layout, compatibility) | {"generation": head.generation}
    elif mode == "failure-retry":
        result = failed_then_retry(root)
    elif mode == "failed-transition":
        result = failed_transition(root)
    elif mode == "retry-transition":
        result = retry_transition(root)
    elif mode == "writer":
        result = writer(root, sys.argv[3], divergent=sys.argv[4] == "different")
    elif mode == "inspect":
        layout = load_example().collection_layout()
        store = LocalStateStore.open(root)
        head = store.read_head()
        if head is None:
            raise AssertionError("Inspection requires the acknowledged actual store")
        result = _store_observation(store, layout, layout.compatibility()) | {"generation": head.generation}
    else:
        raise ValueError("Unknown bounded installed collection mode")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
