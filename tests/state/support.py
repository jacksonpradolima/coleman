"""Small actual host/replay flows shared by independent recovery witnesses."""

from coleman.core.actions import ChoiceAction
from coleman.core.primitives import fingerprint
from coleman.state.storage_contracts import CommitIntent, CommitKind, CommitRequest


def emit(example, replay, child, ordinal, *, at=None):
    prepared = replay.prepare(example.opportunity(child, ordinal, at=at))
    return replay.emit(prepared, execution="offline_identity_v1")


def complete(example, replay, record, value, *, at):
    batch = example.feedback(record, decisions=replay.state.decision_count, available=at, value=value)
    replay.admit_feedback((batch,))
    return replay.advance(eligible_through=example.instant(at), apply_at=example.instant(at))


def command(generation, expected, *, kind: CommitKind = "transition", content="authored-command"):
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


def pending_collection(example):
    """Author the actual95 decisions/48 updates, in the declared interleaving."""
    sessions = [example.make_replay(child) for child in range(32)]
    records = [[] for _ in range(32)]
    domain = example.Domain()
    for ordinal in range(4):
        for child, (replay, _) in enumerate(sessions):
            if ordinal < 2 + child % 3:
                record = emit(example, replay, child, ordinal)
                records[child].append(record)
                assert isinstance(record.executed.action, ChoiceAction)
                domain.apply(record.executed.action.chosen)
    for child, (replay, _) in enumerate(sessions):
        decisions, updates = 2 + child % 3, 1 + child % 2
        offered = tuple(
            example.feedback(record, decisions=decisions, available=decisions, value=((child + index) % 4) / 4)
            for index, record in enumerate(records[child][:updates])
        )
        replay.admit_feedback(offered)
        replay.advance(eligible_through=example.instant(decisions), apply_at=example.instant(decisions))
    return sessions, records, domain
