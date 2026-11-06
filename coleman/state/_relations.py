"""Compare actual validated old/new snapshots before acknowledging publication."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from coleman.core.primitives import canonical_json
from coleman.objectives.bindings import DefinitionBinding
from coleman.replay.retained_contracts import RetainedContinuation, RetainedReplayState
from coleman.state.components import ComponentValue, PreparedCollection
from coleman.state.storage_contracts import AcknowledgedHead, CommitKind, semantic_frontiers_fingerprint


def _wire(value: BaseModel | None) -> str:
    """Preserve complete canonical content, including signed zero and null."""
    if value is None:
        return "null"
    return canonical_json(value.model_dump(mode="json"))


def _same(old: BaseModel | None, new: BaseModel | None, label: str) -> None:
    """Reject a changed explicit semantic boundary rather than comparing names."""
    if _wire(old) != _wire(new):
        raise ValueError(f"State successor changes {label}")


def _prefix(old: Sequence[BaseModel], new: Sequence[BaseModel], label: str) -> None:
    """Require exact old causal records, never just maximum counters."""
    if len(new) < len(old) or any(_wire(left) != _wire(right) for left, right in zip(old, new, strict=False)):
        raise ValueError(f"State successor does not preserve the exact {label} prefix")


def _owners(collection: PreparedCollection) -> dict[str, ComponentValue[Any]]:
    """Match the actual complete values to their declared unique owner IDs."""
    values = {item.codec.binding.owner.id: item for item in collection.values}
    if len(values) != len(collection.values):
        raise ValueError("State successor contains duplicate actual owner values")
    expected = {item.binding.owner.id: item.binding for item in collection.manifest.definition.components}
    if set(values) != set(expected) or any(values[key].codec.binding != binding for key, binding in expected.items()):
        raise ValueError("State successor lacks its exact complete declared component inventory")
    return values


def validate_successor(
    old: PreparedCollection,
    new: PreparedCollection,
    kind: CommitKind,
    *,
    migration: DefinitionBinding | None = None,
) -> None:
    """Check a concrete successor relation before immutable publication/SQL ack.

    Parameters
    ----------
    old, new : PreparedCollection
        Complete decoded source and validated target values, paired by stable
        owner ID. No factory, restore, inspector or learner is called here.
    kind : {'transition', 'seal', 'codec_migration'}
        Selected accepted relation; initialize has no predecessor and rejects.
    migration : DefinitionBinding or None, optional
        Exact known mapping required only for codec_migration. It does not prove
        arbitrary equivalence of host statistics, RNG, cache or pending data.

    Raises
    ------
    ValueError, TypeError
        Incomplete inventory, changed semantic policy, causal rewind, hidden
        seal, missing/mismatched migration receipt or an unsupported relation.

    Notes
    -----
    Replay delivery has explicit ordinal acceptance. Direct custom components
    receive publication semantics only; their raw updates are not implicitly
    idempotent. Codecs/roles/layout/runtime are additionally checked by the local
    adapter before this pure comparison. Unchanged components are legal and may
    reuse verified immutable blobs. No file, RNG or application owner is touched.
    """
    if kind not in {"transition", "seal", "codec_migration"}:
        raise ValueError("A state successor requires transition, seal or codec_migration")
    if new.manifest.intent.kind != kind:
        raise ValueError("State successor relation differs from its actual publication intent")
    if (kind == "codec_migration") != (migration is not None):
        raise ValueError("Exactly codec_migration requires its explicit known mapping")
    if migration is not None:
        DefinitionBinding.model_validate(migration)
    _predecessor(old, new)
    if kind == "codec_migration":
        _migration_projection(old, new, migration)
    _definition_identity(old, new)
    before, after = _owners(old), _owners(new)
    if set(before) != set(after):
        raise ValueError("State successor cannot change stable component owner IDs")
    for identity, previous in before.items():
        current = after[identity]
        if previous.codec.binding.owner != current.codec.binding.owner:
            raise ValueError("State successor cannot change an owner identity/version")
        _component_successor(previous, current, kind, migration)


def _predecessor(old: PreparedCollection, new: PreparedCollection) -> None:
    """Bind the successor to the actual old complete manifest, not its spelling."""
    manifest = old.manifest
    expected = AcknowledgedHead(
        schema_version=1,
        session=manifest.definition.session,
        source_epoch=manifest.definition.source_epoch,
        definition_fingerprint=manifest.definition.fingerprint,
        generation=manifest.intent.generation,
        manifest_fingerprint=manifest.fingerprint,
        intent_fingerprint=manifest.intent.fingerprint,
    )
    if new.manifest.predecessor != expected:
        raise ValueError("State successor does not bind its actual complete acknowledged predecessor")


def _migration_projection(
    old: PreparedCollection, new: PreparedCollection, migration: DefinitionBinding | None
) -> None:
    """Preserve exactly the semantic frontier projection of the actual source."""
    receipt = new.manifest.migration
    if receipt is None:
        raise ValueError("State migration requires its actual source-bound receipt")
    _same(receipt.migration, migration, "qualified collection migration mapping")
    expected = semantic_frontiers_fingerprint(old.manifest.frontiers)
    if expected != semantic_frontiers_fingerprint(new.manifest.frontiers):
        raise ValueError("State migration changes the actual preserved semantic frontiers")
    if receipt.preserved_frontiers_fingerprint != expected:
        raise ValueError("Migration receipt does not bind the actual source semantic frontiers")


def _definition_identity(old: PreparedCollection, new: PreparedCollection) -> None:
    """Keep component semantics fixed when known code/codec bindings migrate."""
    before, after = old.manifest.definition, new.manifest.definition
    if (before.session, before.source_epoch, before.children) != (after.session, after.source_epoch, after.children):
        raise ValueError("State successor cannot change session/source epoch or exact child identities")
    expected = {
        item.binding.owner.id: (item.binding.owner, item.child, item.roles, item.frontier_profile)
        for item in before.components
    }
    actual = {
        item.binding.owner.id: (item.binding.owner, item.child, item.roles, item.frontier_profile)
        for item in after.components
    }
    if expected != actual:
        raise ValueError("State successor cannot change owner/child/role or acceptance namespaces")


def _component_successor(
    old: ComponentValue[Any], new: ComponentValue[Any], kind: CommitKind, migration: DefinitionBinding | None
) -> None:
    """Keep ordinary code bindings fixed and dispatch only explicit replay data."""
    if kind != "codec_migration" and old.codec.binding != new.codec.binding:
        raise ValueError("Ordinary state publication cannot change codec/configuration/code bindings")
    left, right = old.value, new.value
    if isinstance(left, RetainedContinuation) != isinstance(right, RetainedContinuation):
        raise ValueError("A successor cannot add or remove a replay acceptance interpretation")
    if _wire(left) == _wire(right) and old.codec.binding == new.codec.binding:
        return
    if isinstance(left, RetainedContinuation) and isinstance(right, RetainedContinuation):
        _replay_successor(left, right, kind, migration)
    elif kind == "seal":
        raise ValueError("Sealing replay cannot change a direct component's mathematical state")


def _replay_successor(
    old: RetainedContinuation[Any],
    new: RetainedContinuation[Any],
    kind: CommitKind,
    migration: DefinitionBinding | None,
) -> None:
    """Compare actual replay semantics, independent of frontier label claims."""
    left, right = old.state, new.state
    _same(left.policy, right.policy, "replay policy/capability/delivery/correction")
    _same(left.retention, right.retention, "source epoch/retention profile")
    if kind == "transition":
        _transition(left, right)
    elif kind == "seal":
        _seal(left, right)
        _same(old.host_state, new.host_state, "host data during seal")
    else:
        _migration(left, right, migration)


def _transition(old: RetainedReplayState, new: RetainedReplayState) -> None:
    """Verify bounded causal extension without implicit compaction or recoding."""
    _same(old.state_codec, new.state_codec, "host codec")
    _same(old.seal, new.seal, "sealed prefix/floor")
    _prefix(old.rebinds, new.rebinds, "codec receipt")
    if len(old.rebinds) != len(new.rebinds):
        raise ValueError("Ordinary state transition cannot append codec receipts")
    _prefix(old.decisions, new.decisions, "decision")
    _prefix(old.updates, new.updates, "update")
    _prefix(old.drains, new.drains, "drain")
    _feedback_extension(old, new)
    _history_extension(old, new)
    _monotone_frontiers(old, new)
    _host_extension(old, new)


def _host_extension(old: RetainedReplayState, new: RetainedReplayState) -> None:
    """Anchor a causal extension to the actual acknowledged host, even at origin."""
    added = sorted(
        (record for record in (*new.decisions, *new.updates, *new.drains) if record.sequence >= old.next_sequence),
        key=lambda record: record.sequence,
    )
    if not added:
        if new.host_fingerprint != old.host_fingerprint:
            raise ValueError("State transition changes the host without a new causal record")
    elif added[0].host_before != old.host_fingerprint:
        raise ValueError("First new causal record does not start from the acknowledged host")


def _feedback_extension(old: RetainedReplayState, new: RetainedReplayState) -> None:
    """Preserve canonical graph facts by identity despite insertion ordering."""
    actual = {(item.logical_key, item.batch.revision_id): _wire(item) for item in new.feedback}
    if any(actual.get((item.logical_key, item.batch.revision_id)) != _wire(item) for item in old.feedback):
        raise ValueError("State transition changes or deletes an admitted feedback revision")


def _history_extension(old: RetainedReplayState, new: RetainedReplayState) -> None:
    """Keep every source fact and first receipt immutable without cutting parents."""
    if old.history is None:
        return
    if new.history is None:
        raise ValueError("State transition deletes its complete source history")
    if (old.history.source_id, old.history.source_version) != (new.history.source_id, new.history.source_version):
        raise ValueError("State transition changes its original source identity/version")
    actual = {(item.metadata.observation_id, item.metadata.revision_id): _wire(item) for item in new.history.revisions}
    if any(
        actual.get((item.metadata.observation_id, item.metadata.revision_id)) != _wire(item)
        for item in old.history.revisions
    ):
        raise ValueError("State transition changes or deletes an admitted source revision")


def _monotone_frontiers(old: RetainedReplayState, new: RetainedReplayState) -> None:
    """Reject valid-looking old checkpoints published under a later generation."""
    if (
        new.decision_count < old.decision_count
        or new.update_count < old.update_count
        or new.next_sequence < old.next_sequence
    ):
        raise ValueError("State successor cannot rewind global replay counters")
    for before, after in ((old.eligible_through, new.eligible_through), (old.apply_at, new.apply_at)):
        if before is not None and (after is None or after < before):
            raise ValueError("State successor cannot rewind either replay clock")


def _fixed_boundary(old: RetainedReplayState, new: RetainedReplayState) -> None:
    """Hold semantic counts, two clocks and mathematical commitment exactly."""
    if (
        old.decision_count,
        old.update_count,
        old.next_sequence,
        old.eligible_through,
        old.apply_at,
        old.causal_prefix_fingerprint,
    ) != (
        new.decision_count,
        new.update_count,
        new.next_sequence,
        new.eligible_through,
        new.apply_at,
        new.causal_prefix_fingerprint,
    ):
        raise ValueError("Seal/migration changes the exact replay counters/clocks/causal boundary")
    _same(old.history, new.history, "complete source inventory at the boundary")


def _seal(old: RetainedReplayState, new: RetainedReplayState) -> None:
    """Require the one actual predecessor's exact explicit compacted boundary."""
    _fixed_boundary(old, new)
    _same(old.state_codec, new.state_codec, "host codec during seal")
    if old.host_fingerprint != new.host_fingerprint:
        raise ValueError("Sealing cannot change the mathematical host state")
    if new.decisions or new.feedback or new.updates or new.drains or new.rebinds:
        raise ValueError("A new acknowledged seal requires an empty live tail and receipt chain")
    seal = new.seal
    if seal is None or old.decision_count != old.update_count or not old.drains:
        raise ValueError("Sealing requires actual completed quiescence and a new real closed drain")
    if seal.source_state_fingerprint != old.fingerprint:
        raise ValueError("Seal does not commit the actual complete acknowledged predecessor")
    if seal.host_fingerprint != old.host_fingerprint or seal.state_codec != old.state_codec:
        raise ValueError("Seal changed the actual original host/codec binding")
    if seal.causal_commitment != old.causal_prefix_fingerprint:
        raise ValueError("Seal changed the actual predecessor causal commitment")
    _same(seal.last_drain, old.drains[-1], "actual final closed drain")
    first = old.seal.first_opportunity if old.seal is not None else old.decisions[0].opportunity
    _same(seal.first_opportunity, first, "original first opportunity")
    _same(seal.sealed_history, old.history, "actual complete sealed source anchor")


def _migration(old: RetainedReplayState, new: RetainedReplayState, migration: DefinitionBinding | None) -> None:
    """Bind one actual old-to-new codec mapping without rewriting old records."""
    _fixed_boundary(old, new)
    if old.seal is None or old.decision_count != old.update_count:
        raise ValueError("Replay codec migration requires a quiescent generation-two seal")
    if any(
        (old.decisions, old.feedback, old.updates, old.drains, new.decisions, new.feedback, new.updates, new.drains)
    ):
        raise ValueError("Replay codec migration requires both live tails empty")
    _same(old.seal, new.seal, "literal historical seal and floor")
    _prefix(old.rebinds, new.rebinds, "codec receipt")
    if len(new.rebinds) != len(old.rebinds) + 1:
        raise ValueError("Replay codec migration requires exactly one appended receipt")
    receipt = new.rebinds[-1]
    _same(receipt.migration, migration, "qualified migration definition")
    if receipt.source_state_fingerprint != old.fingerprint:
        raise ValueError("Migration receipt does not bind the actual complete acknowledged source")
    if (
        receipt.source_codec,
        receipt.target_codec,
        receipt.source_host_fingerprint,
        receipt.target_host_fingerprint,
    ) != (old.state_codec, new.state_codec, old.host_fingerprint, new.host_fingerprint):
        raise ValueError("Migration receipt changed its actual source/target codec or host binding")
