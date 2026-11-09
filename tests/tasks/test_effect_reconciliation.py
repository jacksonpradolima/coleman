"""A bounded fake-sink contract around actual Hedge and actual local ACKs.

This is test-only local effect reconciliation, not a production effect engine or
network delivery claim. Runtime compatibility comes from the explicit declared
source fixture; no installed-image capture is claimed. Every instrumented store
method delegates the real SDK method exactly once. Sink calls are forbidden
during those actual factory/restore/transition/publication scopes.
"""

from __future__ import annotations

import dataclasses
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Self

import pytest
from pydantic import model_validator

from coleman.artifacts import ArtifactWriter
from coleman.core._wire import BoundedContract
from coleman.core.base import CoreImmutable
from coleman.core.primitives import Fingerprint, canonical_json, decode_json, fingerprint
from coleman.learning.contracts import ReferencePayload
from coleman.learning.reference import Hedge
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.components import ComponentBinding, ComponentCodec, ComponentValue
from coleman.state.contracts import AlgorithmState, StateBinding
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CollectionDefinition, CommitIntent, CommitRequest, ComponentSpec
from coleman.tasks.contracts import OwnedCommitProof, TaskFrontier
from coleman.tasks.protocols import CommittedProgress

from .test_reference_executor import attempt, case, direct
from .test_reference_executor import source as source


def _digest(body: str) -> str:
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read(path: Path) -> str:
    with path.open("rb") as stream:
        content = stream.read(128 * 1024 + 1)
    if len(content) > 128 * 1024:
        raise ValueError("Fake sink artifact exceeded128KiB")
    return content.decode("utf-8")


def _no_read_factory():
    raise AssertionError("Read-only receipt decoding must construct zero state owners")


class SinkArtifact(BoundedContract):
    """Bound this local fixture's complete keys/intents/receipts to128KiB."""

    _maximum_bytes = 128 * 1024


class EffectKey(SinkArtifact):
    """Name one logical external operation without embedding its payload hash."""

    plan_fingerprint: Fingerprint
    coordinate_fingerprint: Fingerprint
    frontier: TaskFrontier


class EffectIntent(SinkArtifact):
    """Freeze the original payload and independently selected fake sink."""

    key: EffectKey
    payload_fingerprint: Fingerprint
    sink: DefinitionBinding


class EffectReceipt(SinkArtifact):
    """Retain an actual fake sink's accepted exact operation, not a store ACK."""

    key: EffectKey
    payload_fingerprint: Fingerprint
    sink: DefinitionBinding


class Inspection(CoreImmutable):
    """Keep unavailable evidence distinct from authoritative absence/presence."""

    status: Literal["present", "absent", "unknown"]
    receipt: EffectReceipt | None

    @model_validator(mode="after")
    def _status(self) -> Self:
        if (self.status == "present") != (self.receipt is not None):
            raise ValueError("Only a known present inspection contains a receipt")
        return self


class JournalState(BoundedContract):
    """Checkpoint full real math/proof and an intent with optional sink receipt."""

    binding: StateBinding
    math: AlgorithmState[ReferencePayload]
    proof: OwnedCommitProof
    intent: EffectIntent
    receipt: EffectReceipt | None
    _maximum_bytes = 512 * 1024

    @model_validator(mode="after")
    def _joins(self) -> Self:
        key = self.intent.key
        if (
            self.intent.payload_fingerprint != self.math.fingerprint
            or key.plan_fingerprint != self.proof.plan_fingerprint
            or key.coordinate_fingerprint != self.proof.coordinate_fingerprint
            or key.frontier != self.proof.frontier
            or key.frontier.position != self.math.payload.update_count
            or key.frontier.coordinate != self.math.binding.owner
        ):
            raise ValueError("Effect intent must join the complete real mathematical ACK")
        if self.receipt is not None and self.receipt.model_dump(exclude={"schema_version"}) != (
            self.intent.model_dump(exclude={"schema_version"})
        ):
            raise ValueError("Effect receipt differs from its original key, payload or sink")
        return self


class Scope:
    """Instrument actual store calls; this is no production sandbox."""

    def __init__(self):
        self.depth = 0
        self.stages = []
        self.external_calls = []

    @contextmanager
    def stage(self, name):
        self.stages.append(name)
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    def external(self, name):
        self.external_calls.append((name, self.depth))
        assert self.depth == 0, "External sink call entered actual state-store staging"


@pytest.fixture
def scope(monkeypatch):
    gate = Scope()

    def wrap(name, original):
        def actual(store, *args, **kwargs):
            with gate.stage(name):
                return original(store, *args, **kwargs)

        return actual

    for name in ("initialize", "commit_transition"):
        monkeypatch.setattr(LocalStateStore, name, wrap(name, getattr(LocalStateStore, name)))
    return gate


class FakeSink:
    """Persist exact accepted keys locally and deliberately lose a response."""

    def __init__(self, root, identity, gate, *, create=False, lose_response=False, unknown=False):
        self.root, self.expected_identity, self.gate = root, identity, gate
        self.lose_response, self.unknown = lose_response, unknown
        self.applies = 0
        self.inspections = 0
        if create:
            ArtifactWriter(root).write_immutable(
                root / "identity.json", canonical_json(identity.model_dump(mode="json"))
            )

    def _identity(self):
        self.gate.external("identity")
        actual = DefinitionBinding.model_validate(decode_json(_read(self.root / "identity.json")))
        if actual != self.expected_identity:
            raise ValueError("Actual persisted fake sink identity changed")
        return actual

    def _path(self, key):
        return self.root / "effects" / (key.fingerprint[7:] + ".json")

    def inspect(self, intent):
        self.gate.external("inspect")
        self.inspections += 1
        if self._identity() != intent.sink:
            raise ValueError("Fake sink inspection has the wrong selected identity")
        if self.unknown:
            return Inspection(status="unknown", receipt=None)
        path = self._path(intent.key)
        if not path.exists():
            return Inspection(status="absent", receipt=None)
        receipt = EffectReceipt.from_json(_read(path))
        if receipt.model_dump(exclude={"schema_version"}) != intent.model_dump(exclude={"schema_version"}):
            raise ValueError("Fake sink key already accepted a divergent payload")
        return Inspection(status="present", receipt=receipt)

    def apply(self, intent, payload):
        self.gate.external("apply")
        self.applies += 1
        if len(payload) > 128 * 1024 or len(payload.encode("utf-8")) > 128 * 1024:
            raise ValueError("Fake sink payload exceeds128KiB")
        intent = EffectIntent.model_validate(intent)
        if self._identity() != intent.sink or _digest(payload) != intent.payload_fingerprint:
            raise ValueError("Fake sink requires its exact identity and original payload digest")
        receipt = EffectReceipt(schema_version=1, **intent.model_dump(exclude={"schema_version"}))
        ArtifactWriter(self.root).write_immutable(self._path(intent.key), receipt.to_json())
        if self.lose_response:
            self.lose_response = False
            raise RuntimeError("Fake sink accepted durably, response lost before receipt checkpoint")
        return receipt


class JournalOwner:
    """Stage only complete immutable receipt state, without sink callbacks."""

    def __init__(self, initial, gate):
        self.gate = gate
        with gate.stage("receipt_factory"):
            self.state = initial

    def export_state(self):
        with self.gate.stage("receipt_export"):
            return self.state

    def restore_state(self, state):
        with self.gate.stage("receipt_restore"):
            checked = JournalState.model_validate(state)
            if checked.binding != self.state.binding:
                raise ValueError("Receipt restore binding changed")
            self.state = checked


class Adapter:
    """Test-only two-store handoff, with no atomic cross-store effect promise."""

    def __init__(self, example, item, sink, gate):
        self.example, self.item, self.sink, self.gate = example, item, sink, gate
        self.root = item.context.job_root / "effect-receipt-state"

    def _binding(self):
        return StateBinding(
            owner=self.example.recipe.reference("fake-effect-receipts"),
            implementation=self.item.context.implementation,
            state_codec=self.example.recipe.reference("fake-effect-receipt-state"),
            configuration_fingerprint=fingerprint(
                {
                    "plan": self.item.context.plan_fingerprint,
                    "coordinate": self.item.coordinate.coordinate_fingerprint,
                    "sink": self.sink.expected_identity.model_dump(mode="json"),
                }
            ),
        )

    def _read(self):
        store = LocalStateStore.open(self.root)
        codec = ComponentCodec(self._binding(), JournalState)
        current = store.load(
            (ComponentBinding(codec, _no_read_factory, lambda _: None),),
            compatibility=self.item.context.compatibility,
        ).component(codec)
        return store, current

    def _initialize(self, result):
        key = EffectKey(
            schema_version=1,
            plan_fingerprint=result.proof.plan_fingerprint,
            coordinate_fingerprint=result.proof.coordinate_fingerprint,
            frontier=result.proof.frontier,
        )
        initial = JournalState(
            schema_version=1,
            binding=self._binding(),
            math=result.checkpoint.learner,
            proof=result.proof,
            intent=EffectIntent(
                schema_version=1,
                key=key,
                payload_fingerprint=result.checkpoint.learner.fingerprint,
                sink=self.sink.expected_identity,
            ),
            receipt=None,
        )
        selected = ComponentBinding(
            ComponentCodec(self._binding(), JournalState), lambda: JournalOwner(initial, self.gate), lambda _: None
        )
        definition = CollectionDefinition(
            schema_version=1,
            reference=self.example.recipe.reference("fake-effect-collection"),
            session=self.example.recipe.reference("effect-" + key.fingerprint[7:]),
            source_epoch=self.item.inputs.reference,
            children=(),
            components=(
                ComponentSpec(binding=self._binding(), child=None, roles=("domain",), frontier_profile="none"),
            ),
        )
        store = LocalStateStore.create(self.root, definition, self.item.context.compatibility)
        request = CommitRequest(
            schema_version=1,
            expected=None,
            intent=CommitIntent(
                schema_version=1,
                generation=0,
                kind="initialize",
                content_fingerprint=initial.fingerprint,
            ),
        )
        receipt = store.initialize(
            request,
            (ComponentValue(selected.codec, initial),),
            (selected,),
            compatibility=self.item.context.compatibility,
        )
        assert receipt.status == "new"
        assert receipt.current_head == store.read_head()

    def _validate_math(self, current):
        observed = self.item.prepared.read_completed(self.item.trajectory, self.item.coordinate)
        assert observed is not None
        if (observed.checkpoint.learner.to_json(), observed.proof.to_json()) != (
            current.math.to_json(),
            current.proof.to_json(),
        ):
            raise ValueError("Receipt reconciliation requires its complete actual original mathematical ACK")
        self.item.prepared.validate(
            CommittedProgress(
                current.proof.plan_fingerprint,
                current.proof.coordinate_fingerprint,
                current.proof.frontier,
                current.proof,
            ),
            self.item.coordinate,
        )

    def checkpoint(self, receipt):
        store, old = self._read()
        selected = ComponentBinding(
            ComponentCodec(self._binding(), JournalState), lambda: JournalOwner(old, self.gate), lambda _: None
        )
        head = store.read_head()
        assert head is not None
        request = CommitRequest(
            schema_version=1,
            expected=head,
            intent=CommitIntent(
                schema_version=1,
                generation=head.generation + 1,
                kind="transition",
                content_fingerprint=receipt.fingerprint,
            ),
        )

        def accept(stage):
            with self.gate.stage("receipt_transition"):
                owner = stage.owner(selected)
                owner.state = JournalState.model_validate({**owner.state.model_dump(), "receipt": receipt})

        acknowledged = store.commit_transition(
            request, (selected,), accept, compatibility=self.item.context.compatibility
        )
        assert acknowledged.status == "new"
        assert acknowledged.current_head == store.read_head()
        current = self._read()[1]
        assert current.receipt is not None
        assert current.receipt.to_json() == receipt.to_json()
        return acknowledged

    def execute(self):
        result = self.item.prepared.advance(self.item.trajectory, attempt(self.item), lambda _: None)
        self.item.prepared.publish_coordinate(self.item.trajectory, result, self.item.coordinate)
        if not self.root.exists():
            self._initialize(result)
        return self._deliver(allow_apply=True)

    def recover(self):
        return self._deliver(allow_apply=False)

    def _deliver(self, *, allow_apply):
        _, current = self._read()
        self._validate_math(current)
        inspection = self.sink.inspect(current.intent)
        if inspection.status == "unknown":
            return None
        if inspection.status == "absent":
            if not allow_apply:
                return None
            receipt = self.sink.apply(current.intent, current.math.to_json())
        else:
            receipt = inspection.receipt
        if current.receipt is None:
            self.checkpoint(receipt)
        elif current.receipt.to_json() != receipt.to_json():
            raise ValueError("Retained receipt differs from actual persisted sink acceptance")
        return self._read()[1]


def _setup(example, tmp_path, gate, *, lose_response=False, unknown=False, config=None):
    item = case(example, tmp_path, config)
    identity = example.recipe.binding("fake-local-idempotent-sink", {"law": "stable-key-exact-payload-v1"})
    sink = FakeSink(tmp_path / "fake-sink", identity, gate, create=True, lose_response=lose_response, unknown=unknown)
    return item, sink, Adapter(example, item, sink, gate)


def test_accepted_effect_response_loss_restart_inspects_actual_sink_without_reobserve_or_reapply(
    source,
    tmp_path,
    scope,
    monkeypatch,
):
    item, sink, adapter = _setup(source, tmp_path, scope, lose_response=True)
    with pytest.raises(RuntimeError, match="accepted durably"):
        adapter.execute()
    store, pending = adapter._read()
    old_journal_head = store.read_head()
    assert old_journal_head is not None
    assert pending.receipt is None
    assert old_journal_head.generation == 0
    assert pending.math.to_json() == direct(item).to_json()
    old_math_head = pending.proof.observed_head
    assert pending.math.payload.update_count == 3
    assert old_math_head.generation == 2
    assert len(tuple((sink.root / "effects").glob("*.json"))) == 1
    assert sink.applies == 1

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Recovery repeated accepted math or external application")

    monkeypatch.setattr(Hedge, "observe", forbidden)
    monkeypatch.setattr(source.state.HedgeOwner, "__init__", forbidden)
    item.prepared = source.package.bootstrap().prepare(item.config, item.inputs, item.context)
    restarted_sink = FakeSink(sink.root, sink.expected_identity, scope)
    monkeypatch.setattr(restarted_sink, "apply", forbidden)
    recovered = Adapter(source, item, restarted_sink, scope).recover()
    assert recovered is not None
    assert recovered.receipt is not None
    assert restarted_sink.inspections == 1
    assert restarted_sink.applies == 0
    assert recovered.math.to_json() == pending.math.to_json()
    assert recovered.proof.to_json() == pending.proof.to_json()
    assert item.prepared.reconcile(item.coordinate).proof.observed_head == old_math_head
    new_head = store.read_head()
    assert new_head is not None
    assert new_head.generation == 1
    assert new_head != old_journal_head
    second = Adapter(source, item, restarted_sink, scope).recover()
    assert second is not None
    assert second.to_json() == recovered.to_json()
    assert store.read_head() == new_head
    assert {"initialize", "commit_transition", "receipt_factory", "receipt_restore", "receipt_transition"} <= set(
        scope.stages
    )
    assert scope.external_calls
    assert all(depth == 0 for _, depth in scope.external_calls)


def test_fake_sink_exact_retry_is_idempotent_and_divergent_payload_same_typed_key_refuses(source, tmp_path, scope):
    _, sink, adapter = _setup(source, tmp_path, scope)
    current = adapter.execute()
    assert current is not None
    assert current.receipt is not None
    head = adapter._read()[0].read_head()
    receipt = sink.apply(current.intent, current.math.to_json())
    assert receipt.to_json() == current.receipt.to_json()
    assert len(tuple((sink.root / "effects").glob("*.json"))) == 1
    changed_math = AlgorithmState[ReferencePayload].model_validate(
        {
            **current.math.model_dump(),
            "payload": {
                **current.math.payload.model_dump(),
                "cumulative_losses": (1.5, 1.75, 1.5),
            },
        }
    )
    divergent_body = changed_math.to_json()
    divergent = EffectIntent.model_validate(
        {**current.intent.model_dump(), "payload_fingerprint": _digest(divergent_body)}
    )
    assert divergent.key.to_json() == current.intent.key.to_json()
    with pytest.raises(ValueError, match="Conflicting immutable"):
        sink.apply(divergent, divergent_body)
    inspected = sink.inspect(current.intent)
    assert inspected.receipt is not None
    assert inspected.receipt.to_json() == receipt.to_json()
    assert adapter._read()[0].read_head() == head
    assert adapter._read()[1].to_json() == current.to_json()


def test_unknown_outcome_stays_unresolved_without_blind_retry_then_known_acceptance_recovers(source, tmp_path, scope):
    item, sink, adapter = _setup(source, tmp_path, scope, lose_response=True)
    with pytest.raises(RuntimeError, match="response lost"):
        adapter.execute()
    old_head, old_state = adapter._read()[0].read_head(), adapter._read()[1].to_json()
    uncertain = FakeSink(sink.root, sink.expected_identity, scope, unknown=True)
    recovery = Adapter(source, item, uncertain, scope)
    assert recovery.recover() is None
    assert recovery.recover() is None
    assert uncertain.applies == 0
    assert adapter._read()[0].read_head() == old_head
    assert adapter._read()[1].to_json() == old_state
    uncertain.unknown = False
    recovered = recovery.recover()
    assert recovered is not None
    assert recovered.receipt is not None
    assert uncertain.applies == 0


def test_wrong_actual_sink_identity_refuses_before_receipt_checkpoint_and_preserves_math(source, tmp_path, scope):
    _, sink, adapter = _setup(source, tmp_path, scope, lose_response=True)
    with pytest.raises(RuntimeError, match="accepted durably"):
        adapter.execute()
    store, original = adapter._read()
    head = store.read_head()
    wrong = source.recipe.binding("wrong-sink", {"other": "identity"})
    (sink.root / "identity.json").write_text(canonical_json(wrong.model_dump(mode="json")))
    with pytest.raises(ValueError, match="Actual persisted fake sink identity"):
        adapter.recover()
    assert store.read_head() == head
    assert adapter._read()[1].to_json() == original.to_json()


def test_numeric_failure_before_panel_ack_causes_zero_external_calls(source, tmp_path, scope):
    base = source.sample.configuration()
    config = source.contracts.HedgeConfiguration.model_validate(
        {
            **base.model_dump(),
            "reference": {**base.reference.model_dump(), "eta": 1e308},
        }
    )
    item, sink, adapter = _setup(source, tmp_path, scope, config=config)
    with pytest.raises(ValueError, match="numerically"):
        adapter.execute()
    assert sink.applies == 0
    assert sink.inspections == 0
    assert scope.external_calls == []
    assert not adapter.root.exists()
    progress = item.prepared.reconcile(item.coordinate)
    assert progress.frontier.position == 0
    assert progress.proof.observed_head.generation == 0
    layout = source.state.Layout(item.config, item.trajectory, item.coordinate)
    assert layout.checkpoint(layout.open()).learner.to_json() == item.trajectory.initial_state.to_json()


def test_state_batch_failure_before_ack_causes_zero_external_calls(source, tmp_path, scope, monkeypatch):
    item, sink, adapter = _setup(source, tmp_path, scope)
    checks = []

    def guard():
        checks.append("guard")
        if len(checks) == 3:
            raise RuntimeError("bounded state attempt interrupted")

    original = item.prepared.advance

    def interrupt(coordinate, context, committed):
        return original(coordinate, dataclasses.replace(context, guard=guard), committed)

    monkeypatch.setattr(type(item.prepared), "advance", lambda _self, *args: interrupt(*args))
    with pytest.raises(RuntimeError, match="state attempt interrupted"):
        adapter.execute()
    assert sink.applies == 0
    assert sink.inspections == 0
    assert scope.external_calls == []
    assert not adapter.root.exists()
    progress = item.prepared.reconcile(item.coordinate)
    assert progress.frontier.position == 0
    assert progress.proof.observed_head.generation == 0
    layout = source.state.Layout(item.config, item.trajectory, item.coordinate)
    assert layout.checkpoint(layout.open()).learner.to_json() == item.trajectory.initial_state.to_json()


def test_staging_instrument_detects_real_forbidden_sink_call_and_no_partial_receipt_ack(source, tmp_path, scope):
    _, sink, adapter = _setup(source, tmp_path, scope, unknown=True)
    assert adapter.execute() is None
    store, pending = adapter._read()
    old_head = store.read_head()
    selected = ComponentBinding(
        ComponentCodec(pending.binding, JournalState), lambda: JournalOwner(pending, scope), lambda _: None
    )
    request = CommitRequest(
        schema_version=1,
        expected=old_head,
        intent=CommitIntent(
            schema_version=1,
            generation=1,
            kind="transition",
            content_fingerprint=fingerprint({"forbidden": True}),
        ),
    )

    def forbidden_effect(stage):
        stage.owner(selected)
        sink.apply(pending.intent, pending.math.to_json())

    with pytest.raises(AssertionError, match="actual state-store staging"):
        store.commit_transition(
            request, (selected,), forbidden_effect, compatibility=adapter.item.context.compatibility
        )
    assert store.read_head() == old_head
    assert adapter._read()[1].to_json() == pending.to_json()
    assert not (sink.root / "effects").exists()
