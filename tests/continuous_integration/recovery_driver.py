"""Real forecast/store recovery process; source fallback is explicit and test-only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from functools import partial
from hashlib import sha256
from math import log
from pathlib import Path
from typing import Any, Literal

OBSERVATION_SECONDS = 60.0


def load_flow(path: Path):
    """Load only the explicitly selected public example file."""
    name = "continuous_recovery_flow"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("The selected public flow.py is required")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def command(
    generation, expected, *, kind: Literal["initialize", "transition"] = "transition", content="append-original-label"
):
    """Bind an actual store command independently from its callback."""
    from coleman.core.primitives import fingerprint  # pylint: disable=import-outside-toplevel
    from coleman.state.storage_contracts import CommitIntent, CommitRequest  # pylint: disable=import-outside-toplevel

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


def event(path: Path, value: str) -> None:
    """Flush process observations independently from the real SQLite ACK."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(value + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def wait_for(path: Path) -> None:
    """Bound only this test barrier, leaving store ownership limits unchanged."""
    deadline = time.monotonic() + OBSERVATION_SECONDS
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Unreleased test barrier: {path.name}")
        time.sleep(0.01)


def hedge_inputs(flow):
    """Declare the existing real Hedge law and bounded public synthetic losses."""
    from coleman.core.actions import Weight  # pylint: disable=import-outside-toplevel
    from coleman.core.feedback import FeedbackRow, FullPanel, OutcomeValue  # pylint: disable=import-outside-toplevel
    from coleman.core.numeric import ScalarSpec  # pylint: disable=import-outside-toplevel
    from coleman.core.primitives import fingerprint  # pylint: disable=import-outside-toplevel
    from coleman.core.units import UnitSemantics  # pylint: disable=import-outside-toplevel
    from coleman.learning.contracts import ReferenceConfig  # pylint: disable=import-outside-toplevel
    from coleman.objectives.bindings import (  # pylint: disable=import-outside-toplevel
        DefinitionBinding,
        DefinitionLineage,
    )
    from coleman.objectives.preferences import (  # pylint: disable=import-outside-toplevel
        COEFFICIENT_SEMANTICS,
        InitialPrior,
    )

    ref = flow.MODEL.reference
    lineage = DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)
    config = ReferenceConfig(
        schema_version=1,
        reference=ref("recovery-real-constant-hedge"),
        lineage=lineage,
        profile="hedge-constant",
        prior=InitialPrior(
            schema_version=1,
            reference=ref("recovery-positive-prior"),
            lineage=lineage,
            weights=tuple(
                Weight(member=ref(f"recovery-expert-{i}"), value=value) for i, value in enumerate((0.5, 0.25, 0.25))
            ),
            profile="binary64-simplex-v1",
            semantics=COEFFICIENT_SEMANTICS,
        ),
        loss_spec=ScalarSpec(
            dimension=ref("recovery-full-loss"),
            role="loss",
            minimum=0.0,
            maximum=1.0,
            semantics=UnitSemantics(unit="1", meaning="recovery-complete-synthetic-loss-v1"),
        ),
        transform=DefinitionBinding(
            reference=ref("recovery-identity-loss"), definition_fingerprint=fingerprint("bounded-identity-v1")
        ),
        eta=log(2.0),
        alpha=None,
        anytime_multiplier=None,
    )
    panel = FullPanel(
        rows=tuple(
            FeedbackRow(
                member=weight.member,
                values=(
                    OutcomeValue(
                        spec=config.loss_spec,
                        value=value,
                        status="observed",
                        origin="synthetic",
                        transform=config.transform.reference,
                    ),
                ),
            )
            for weight, value in zip(config.prior.weights, (0.0, 0.5, 1.0), strict=True)
        )
    )
    return config, panel


def collection(flow, count: int, with_hedge: bool) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
    """Compose at most32 predictor children with optional32 real Hedge states."""
    from coleman.learning.reference import Hedge  # pylint: disable=import-outside-toplevel
    from coleman.state.components import ComponentBinding, ComponentCodec  # pylint: disable=import-outside-toplevel
    from coleman.state.storage_contracts import (  # pylint: disable=import-outside-toplevel
        CollectionDefinition,
        ComponentSpec,
    )

    definition, forecasts = flow.collection_layout(tuple(range(count)))
    if not with_hedge:
        return definition, forecasts, ()
    config, _ = hedge_inputs(flow)
    methods = tuple(Hedge(config, owner=flow.MODEL.reference(f"recovery-hedge-{index:02}")) for index in range(count))
    hedges = tuple(
        ComponentBinding(
            ComponentCodec(method.binding, type(method.export_state())),
            partial(Hedge, config, owner=method.binding.owner),
            lambda _: None,
        )
        for method in methods
    )
    definition = CollectionDefinition.model_validate(
        {
            **definition.model_dump(),
            "components": (
                *definition.components,
                *(
                    ComponentSpec(binding=method.binding, child=child, roles=("algorithm",), frontier_profile="none")
                    for method, child in zip(methods, definition.children, strict=True)
                ),
            ),
        }
    )
    return definition, forecasts, hedges


def compatibility_for(flow, bindings):
    """Retain actual runtime capture plus every actual component implementation."""
    from coleman.objectives.bindings import DefinitionBinding  # pylint: disable=import-outside-toplevel

    value = flow.compatibility()
    definitions = {item.reference.id: item for item in value.owner_implementations}
    driver = DefinitionBinding(
        reference=flow.MODEL.reference("constructed-forecast-recovery-driver"),
        definition_fingerprint="sha256:" + sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    definitions[driver.reference.id] = driver
    definitions.update(
        (item.codec.binding.implementation.reference.id, item.codec.binding.implementation) for item in bindings
    )
    return type(value).model_validate(value.model_copy(update={"owner_implementations": tuple(definitions.values())}))


def initialize(flow, root: Path, count: int = 1, with_hedge: bool = False):
    """Initialize from an actual running predictor with one original pending x."""
    from coleman.state.components import ComponentValue  # pylint: disable=import-outside-toplevel
    from coleman.state.local import LocalStateStore  # pylint: disable=import-outside-toplevel

    definition, forecasts, hedges = collection(flow, count, with_hedge)
    bindings = (*forecasts, *hedges)
    owners = tuple(item.new_owner() for item in bindings)
    for owner in owners[:count]:
        record = flow.emit(owner, 0)
        if count == 32:
            owner.replay.admit_feedback((flow.feedback(record, 2.0),))
            owner.replay.advance(eligible_through=flow.instant(1.5), apply_at=flow.instant(1.5))
            flow.emit(owner, 1)
    if hedges:
        _, panel = hedge_inputs(flow)
        for owner in owners[count:]:
            owner.observe(panel)
    compatibility = compatibility_for(flow, bindings)
    store = LocalStateStore.create(root, definition, compatibility)
    receipt = store.initialize(
        command(0, None, kind="initialize", content="actual-pending-forecast"),
        tuple(ComponentValue(item.codec, owner.export_state()) for item, owner in zip(bindings, owners, strict=True)),
        bindings,
        compatibility=compatibility,
    )
    (root.parent / "original-head.json").write_text(receipt.current_head.to_json(), encoding="utf-8")
    return inspect(flow, root, count, with_hedge)


def inspect(flow, root: Path, count: int = 1, with_hedge: bool = False):
    """Read the actual acknowledged full predictor/replay snapshot."""
    from coleman.state.local import LocalStateStore  # pylint: disable=import-outside-toplevel

    _, forecasts, hedges = collection(flow, count, with_hedge)
    bindings = (*forecasts, *hedges)
    store = LocalStateStore.open(root)
    prepared = store.load(bindings, compatibility=compatibility_for(flow, bindings))
    states = tuple(prepared.component(item.codec) for item in forecasts)
    hedge_states = tuple(prepared.component(item.codec) for item in hedges)
    return {
        "head": prepared.manifest.fingerprint,
        "generation": prepared.manifest.intent.generation,
        "state": states[0].model_dump(mode="json"),
        "state_fingerprint": states[0].fingerprint,
        "states": tuple(item.model_dump(mode="json") for item in states),
        "state_fingerprints": tuple(item.fingerprint for item in states),
        "samples": sum(item.host_state.ridge.payload.sample_count for item in states),
        "pending": sum(len(item.host_state.pending) for item in states),
        "hedge_states": tuple(item.model_dump(mode="json") for item in hedge_states),
        "hedge_updates": sum(item.payload.update_count for item in hedge_states),
        "components": len(bindings),
        "children": count,
    }


def transition(
    flow, root: Path, *, actor: str, fault: str | None, divergent: bool, count: int = 1, with_hedge: bool = False
):
    """Append the real original label once, optionally dying before actual COMMIT."""
    from coleman.state.local import LocalStateStore  # pylint: disable=import-outside-toplevel
    from coleman.state.storage_contracts import AcknowledgedHead  # pylint: disable=import-outside-toplevel

    _, forecasts, hedges = collection(flow, count, with_hedge)
    bindings = (*forecasts, *hedges)

    class ObservedStore(LocalStateStore):
        """Observe real publication without replacing validation or SQL ownership."""

        def _barrier(self, phase, *, component=None):
            if phase == "after_head_update" and fault == "before-ack":
                event(root.parent / "fault.jsonl", phase)
                os._exit(87)
            if phase == "before_blobs" and fault == "hold":
                event(root.parent / "held.jsonl", phase)
                wait_for(root.parent / "release")

    store = ObservedStore.open(root)
    expected = AcknowledgedHead.from_json((root.parent / "original-head.json").read_bytes())
    request = command(1, expected, content="different-command" if divergent else "append-original-label")

    def append(stage):
        event(root.parent / (actor + "-calls.jsonl"), "append")
        for binding in forecasts:
            owner = stage.owner(binding)
            producer = owner.replay.state.decisions[-1]
            owner.replay.admit_feedback((flow.feedback(producer, 0.0 if count == 32 else 2.0),))
            owner.replay.advance(eligible_through=flow.instant(4), apply_at=flow.instant(4))
        if hedges:
            _, panel = hedge_inputs(flow)
            for binding in hedges:
                stage.owner(binding).observe(panel)

    compatibility = compatibility_for(flow, bindings)
    if fault == "standby":
        event(root.parent / (actor + "-ready.jsonl"), "opened")
        wait_for(root.parent / "start-contender")
    event(root.parent / (actor + "-attempt.jsonl"), "commit")
    try:
        receipt = store.commit_transition(request, bindings, append, compatibility=compatibility)
    except ValueError as error:
        return {"rejected": str(error)}
    return {**inspect(flow, root, count, with_hedge), "status": receipt.status}


def main() -> None:
    """Run one explicit source/installed process operation, without installation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("initialize", "inspect", "transition"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--flow", type=Path, default=Path(__file__).with_name("flow.py"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--actor", default="attempt")
    parser.add_argument("--fault", choices=("before-ack", "hold", "standby"))
    parser.add_argument("--children", type=int, choices=(1, 32), default=1)
    parser.add_argument("--with-hedge", action="store_true")
    parser.add_argument("--divergent", action="store_true")
    arguments = parser.parse_args()
    if arguments.source_root is not None:
        # Explicit source-unit route. Installed qualification omits this option.
        sys.path.insert(0, str(arguments.source_root.resolve(strict=True)))
    flow = load_flow(arguments.flow.resolve(strict=True))
    if arguments.operation == "initialize":
        result = initialize(flow, arguments.root, arguments.children, arguments.with_hedge)
    elif arguments.operation == "inspect":
        result = inspect(flow, arguments.root, arguments.children, arguments.with_hedge)
    else:
        result = transition(
            flow,
            arguments.root,
            actor=arguments.actor,
            fault=arguments.fault,
            divergent=arguments.divergent,
            count=arguments.children,
            with_hedge=arguments.with_hedge,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
