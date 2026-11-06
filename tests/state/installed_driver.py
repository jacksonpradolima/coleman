"""Copied process runner using the installed state/replay/store and actual host."""

from __future__ import annotations

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

from coleman.artifacts import ArtifactWriter
from coleman.core.actions import ChoiceAction
from coleman.core.base import CoreContract
from coleman.core.primitives import canonical_json, fingerprint
from coleman.state.components import ComponentBinding, ComponentValue
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CommitIntent, CommitKind, CommitRequest

# Test process observation is separate from the persisted SQLite wait allowance.
PROCESS_OBSERVATION_SECONDS = 600


class DirectCheckpoint[H: CoreContract, D: CoreContract](CoreContract):
    """Caller-owned aggregate JSON, without any SQLite acknowledgement claim."""

    continuation: H
    domain: D


def load_example():
    name = "copied_authored_owned_ci"
    specification = importlib.util.spec_from_file_location(name, Path(__file__).with_name("ci.py"))
    if specification is None or specification.loader is None:
        raise RuntimeError("The explicitly copied public host is required")
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


def watched_bindings(bindings, diagnostic):
    def factory(binding):
        log(diagnostic, "factory:" + binding.codec.binding.owner.id)
        return binding.new_owner()

    return tuple(ComponentBinding(binding.codec, partial(factory, binding), binding.frontier) for binding in bindings)


def initialize(root, *, many):
    example = load_example()
    children = tuple(range(32)) if many else (0,)
    definition, bindings = example.collection_layout(children)
    compatibility = example.execution_compatibility()
    if many:
        owners = [example.ReplayOwner(child) for child in children]
        domain = example.Domain()
        records = [[] for _ in children]
        for ordinal in range(4):
            for child, owner in enumerate(owners):
                if ordinal < 2 + child % 3:
                    replay = owner.replay
                    record = replay.emit(
                        replay.prepare(example.opportunity(child, ordinal)), execution="offline_identity_v1"
                    )
                    records[child].append(record)
                    if not isinstance(record.executed.action, ChoiceAction):
                        raise ValueError("Authored fixture requires actual choice actions")
                    domain.apply(record.executed.action.chosen)
        for child, owner in enumerate(owners):
            decisions, updates = 2 + child % 3, 1 + child % 2
            offered = tuple(
                example.feedback(record, decisions=decisions, available=decisions, value=((child + index) % 4) / 4)
                for index, record in enumerate(records[child][:updates])
            )
            owner.replay.admit_feedback(offered)
            owner.replay.advance(eligible_through=example.instant(decisions), apply_at=example.instant(decisions))
        snapshots = (*(owner.export_state() for owner in owners), domain.export_state())
        values = tuple(ComponentValue(binding.codec, value) for binding, value in zip(bindings, snapshots, strict=True))
    else:
        values = example.origin_values(bindings)
    from coleman.state.storage_contracts import StorageLimits  # pylint: disable=import-outside-toplevel

    limits = StorageLimits(schema_version=1, lock_wait_seconds=30.0)
    store = LocalStateStore.create(root, definition, compatibility, limits=limits)
    request = command(0, None, kind="initialize", content="pending32" if many else "origin-one")
    receipt = store.initialize(request, values, bindings, compatibility=compatibility)
    (root.parent / "initial-head.json").write_text(receipt.current_head.to_json())
    return summary(store, bindings, compatibility)


def complete_pending(example, stage, bindings):
    for child, binding in enumerate(bindings[:-1]):
        owner = stage.owner(binding)
        before = owner.host.export_state()
        if not before.pending:
            continue
        by_fingerprint = {record.executed.fingerprint: record for record in owner.replay.state.decisions}
        offered = tuple(
            example.feedback(
                by_fingerprint[pending.executed.fingerprint],
                decisions=before.decision_count,
                available=before.decision_count + 1,
                value=((child + pending.executed.proposal.request.round) % 4) / 4,
            )
            for pending in before.pending
        )
        owner.replay.admit_feedback(offered)
        at = example.instant(before.decision_count + 1)
        owner.replay.advance(eligible_through=at, apply_at=at)


def summary(store, bindings, compatibility):
    prepared = store.load(bindings, compatibility=compatibility)
    children = []
    for binding in bindings[:-1]:
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
    domain = prepared.component(bindings[-1].codec)
    aggregate = canonical_json([prepared.component(binding.codec).model_dump(mode="json") for binding in bindings])
    (store.root.parent / "observed-states.json").write_text(aggregate + "\n")
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
    }


def observe_contention(store, request, bindings, callback, compatibility, diagnostic):
    """Observe finite acquisition failure while another process owns SQLite.

    Only the documented pre-callback busy/deadline outcomes qualify. A corrupt
    database, a different SQL error, any owner call or an unexpected acceptance
    fails this process. The caller explicitly retries the original request only
    after the parent observes the winning process's committed completion.
    """
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
        raise AssertionError("Acquisition failure invoked a loser factory, restore, export or transition")
    return {
        "outcome": outcome,
        "sqlite_errorcode": code,
        "lock_wait_seconds": store.limits.lock_wait_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "callbacks": [],
    }


def transition(root, *, fault=None, actor="attempt", divergent=False):
    example = load_example()
    _, original = example.collection_layout(tuple(range(32)))
    diagnostic = root.parent / (actor + "-calls.log")
    bindings = watched_bindings(original, diagnostic)
    compatibility = example.execution_compatibility()

    class FaultStore(LocalStateStore):
        def _barrier(self, phase, *, component=None):
            selected = phase if component is None else phase + ":" + str(component)
            if fault == selected:
                (root.parent / "fault-barrier.json").write_text(json.dumps({"phase": phase, "component": component}))
                os._exit(87)
            if fault == "hold-lock" and phase == "before_blobs":
                (root.parent / "lock-held").write_text("actual callback completed under SQLite ownership")
                wait(root.parent / "release-lock")

    if fault == "mid-stream":
        original_stream = ArtifactWriter.write_immutable_stream

        def interrupted(self: ArtifactWriter, path: Path, chunks: Iterable[bytes], *, maximum_bytes: int) -> None:
            def real_chunks():
                for chunk in chunks:
                    midpoint = max(1, len(chunk) // 2)
                    yield chunk[:midpoint]
                    (root.parent / "fault-barrier.json").write_text(json.dumps({"phase": "mid-stream"}))
                    os._exit(88)

            return original_stream(self, path, real_chunks(), maximum_bytes=maximum_bytes)

        # Process-local injection still calls the real immutable writer. The
        # fixed attribute is not selected by serialized input or used in SDKs.
        cast(Any, ArtifactWriter).write_immutable_stream = interrupted

    (root.parent / (actor + "-attempt")).write_text("entering actual owned-store operation")
    store = FaultStore.open(root)
    from coleman.state.storage_contracts import AcknowledgedHead  # pylint: disable=import-outside-toplevel

    expected = AcknowledgedHead.from_json((root.parent / "initial-head.json").read_bytes())
    request = command(1, expected, content="divergent" if divergent else "complete-pending32")

    def callback(stage):
        log(diagnostic, "actual-transition")
        complete_pending(example, stage, bindings)

    if fault == "contend":
        (root.parent / "loser-ready").write_text("metadata handle opened before the competing writer")
        wait(root.parent / "start-contention")
        observed = observe_contention(store, request, bindings, callback, compatibility, diagnostic)
        observed_path = root.parent / "loser-busy.json"
        temporary = observed_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(observed, sort_keys=True))
        temporary.replace(observed_path)
        wait(root.parent / "retry-loser")

    try:
        receipt = store.commit_transition(request, bindings, callback, compatibility=compatibility)
    except ValueError as error:
        print(
            json.dumps(
                {
                    "rejected": str(error),
                    "callbacks": diagnostic.read_text().splitlines() if diagnostic.exists() else [],
                }
            )
        )
        return
    result = summary(store, original, compatibility)
    result["status"] = receipt.status
    print(json.dumps(result, sort_keys=True))


def inspect(root):
    example = load_example()
    _, bindings = example.collection_layout(tuple(range(32)))
    store = LocalStateStore.open(root)
    print(json.dumps(summary(store, bindings, example.execution_compatibility()), sort_keys=True))


def long_run(root, *, chunk_size=1, stop=4352):
    example = load_example()
    from coleman.replay.retained_contracts import RetainedContinuation  # pylint: disable=import-outside-toplevel

    definition, bindings = example.collection_layout()
    compatibility = example.execution_compatibility()
    started = time.monotonic()
    checkpoint_type = DirectCheckpoint[RetainedContinuation[example.HostState], example.DomainState]
    prior = [
        ordinal for ordinal in (2048, 4096) if ordinal < stop and (root.parent / f"direct-{ordinal}.json").exists()
    ]
    owner, domain = example.ReplayOwner(0), example.Domain()
    if prior:
        checkpoint = checkpoint_type.from_json((root.parent / f"direct-{max(prior)}.json").read_bytes())
        owner.restore_state(checkpoint.continuation)
        domain.restore_state(checkpoint.domain)
    start = owner.replay.state.decision_count
    old_path = root.parent / "retired-feedback.json"
    if start and old_path.exists():
        from coleman.replay.contracts import ReplayFeedback  # pylint: disable=import-outside-toplevel

        old = ReplayFeedback.from_json(old_path.read_bytes())
        before = owner.export_state().to_json()
        try:
            owner.replay.admit_feedback((old,))
        except ValueError:
            if owner.export_state().to_json() != before:
                raise AssertionError("Expired feedback mutated the actual restored state") from None
        else:
            raise AssertionError("Expired feedback reentered a compacted restored stream")
    maximum_tail = 0
    for ordinal in range(start, stop):
        action = owner.step((ordinal % 4) / 4)
        domain.apply(action.chosen)
        state = owner.replay.state
        maximum_tail = max(
            maximum_tail, len(state.decisions) + len(state.feedback) + len(state.updates) + len(state.drains)
        )
        if ordinal == 0:
            ArtifactWriter(root.parent).write_immutable(old_path, state.feedback[0].to_json())
        if (ordinal + 1) % chunk_size == 0 or ordinal + 1 == stop:
            owner.replay.compact()
    state = owner.export_state()
    checkpoint = checkpoint_type(schema_version=1, continuation=state, domain=domain.export_state())
    ArtifactWriter(root.parent).write_immutable_stream(
        root.parent / f"direct-{stop}.json", (checkpoint.to_json().encode(),), maximum_bytes=16 * 1024**2
    )
    direct = {
        "D": state.host_state.decision_count,
        "U": state.host_state.update_count,
        "loss": sum(state.host_state.loss_sums),
        "cadence": state.host_state.cadence_count,
        "count_b": state.host_state.book.count_b,
        "switches": state.host_state.book.switches,
        "spent_seconds": state.host_state.book.spent_seconds,
        "backend_refills": state.host_state.backend_refills,
        "remaining_bits": state.host_state.reservoir_remaining,
        "maximum_tail_records": maximum_tail,
        "state_bytes": len(state.to_json().encode()),
        "pending": len(state.host_state.pending),
    }
    if stop < 4352:
        direct["elapsed_seconds"] = time.monotonic() - started
        print(json.dumps(direct, sort_keys=True))
        return
    suffix_owner = example.HostConsumer(0)
    suffix_owner.restore_state(state.host_state)
    suffix = []
    for ordinal in range(4352, 4362):
        proposal = suffix_owner.decide(example.opportunity(0, ordinal).request)
        if not isinstance(proposal.action, ChoiceAction):
            raise ValueError("Literal suffix requires actual chosen CI proposals")
        suffix.append(int(proposal.action.chosen == example.PLANS[1]))
    direct["next_bits"] = suffix
    store = LocalStateStore.create(root, definition, compatibility)
    values = (ComponentValue(bindings[0].codec, state), ComponentValue(bindings[-1].codec, domain.export_state()))
    accepted = store.initialize(
        command(0, None, kind="initialize", content="actual-direct4352-state"),
        values,
        bindings,
        compatibility=compatibility,
    )

    def extend(stage):
        restored = stage.owner(bindings[0])
        actual_domain = stage.owner(bindings[-1])
        for ordinal in (4352, 4353):
            selected = restored.step((ordinal % 4) / 4)
            actual_domain.apply(selected.chosen)

    accepted = store.commit_transition(command(1, accepted.current_head), bindings, extend, compatibility=compatibility)
    seal = command(2, accepted.current_head, kind="seal")
    store.commit_transition(
        seal, bindings, lambda stage: stage.owner(bindings[0]).replay.compact(), compatibility=compatibility
    )
    result = summary(store, bindings, compatibility)
    result["direct"] = direct
    result["elapsed_seconds"] = time.monotonic() - started
    result["owned_bytes"] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(json.dumps(result, sort_keys=True))


def main():
    mode, root = sys.argv[1], Path(sys.argv[2])
    root.parent.mkdir(parents=True, exist_ok=True)
    if mode == "initialize32":
        print(json.dumps(initialize(root, many=True), sort_keys=True))
    elif mode == "initialize-one":
        print(json.dumps(initialize(root, many=False), sort_keys=True))
    elif mode == "inspect":
        inspect(root)
    elif mode == "long-run":
        long_run(root, stop=int(sys.argv[3]), chunk_size=int(sys.argv[4]))
    else:
        fault = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "none" else None
        actor = sys.argv[4] if len(sys.argv) > 4 else "attempt"
        divergent = len(sys.argv) > 5 and sys.argv[5] == "divergent"
        transition(root, fault=fault, actor=actor, divergent=divergent)


if __name__ == "__main__":
    main()
