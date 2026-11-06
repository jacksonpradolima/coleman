"""Strict replay readers validate actual retained contracts and causal policy."""

import json
from pathlib import Path
from typing import cast

import pytest

from coleman.core.base import CoreContract
from coleman.replay.codec import read_continuation, read_replay_feedback, read_replay_state
from tests.replay.support import emit, session


def trace(example):
    replay, _ = session(example)
    first = emit(example, replay, 0, at=0, end=1)
    emit(example, replay, 1, at=1, end=2)
    replay.admit_feedback((example.feedback(first, (1.0, 0.0, 0.5), visible=2, delay=1),))
    emit(example, replay, 2, at=3, end=4)
    return replay


@pytest.fixture(scope="module")
def populated_wire(example):
    return trace(example).export_continuation().to_json()


@pytest.mark.parametrize("mutation", ["synchronous", "objective", "universe", "required_input"])
def test_state_reader_rejects_live_impossible_policy_or_request_mutations(example, mutation):
    document = json.loads(trace(example).state.to_json())
    if mutation == "synchronous":
        document["policy"]["delivery"] = "synchronous"
    elif mutation == "objective":
        replacement = example.reference("different-later-objective").model_dump(mode="json")
        document["decisions"][-1]["opportunity"]["request"]["objective"] = replacement
        document["decisions"][-1]["executed"]["proposal"]["request"]["objective"] = replacement
    elif mutation == "required_input":
        document["policy"]["capability"]["required_inputs"] = [example.LOSS.model_dump(mode="json")]
    else:
        # Keep each individual core proposal/action internally consistent. The
        # error belongs to replay's undeclared transition between decisions.
        last = document["decisions"][-1]
        old = type(example.space("chosen")).model_validate(last["opportunity"]["request"]["space"])
        changed = type(old).model_validate(old.model_copy(update={"members": old.members[:2]}))
        from coleman.core.actions import ChoiceAction

        action = ChoiceAction.model_validate(last["executed"]["action"])
        action = ChoiceAction.model_validate(action.model_copy(update={"space": changed}))
        for location in (last["opportunity"]["request"], last["executed"]["proposal"]["request"]):
            location["space"] = changed.model_dump(mode="json")
        last["executed"]["action"] = action.model_dump(mode="json")
        last["executed"]["proposal"]["action"] = action.model_dump(mode="json")
        last["executed"]["sampling"]["action_fingerprint"] = action.fingerprint
        last["executed"]["proposal"]["sampling"]["action_fingerprint"] = action.fingerprint
    wire = json.dumps(document)
    with pytest.raises(ValueError):
        read_replay_state(wire)


@pytest.mark.parametrize("version", [2, 0, True, 1.0, "1"])
@pytest.mark.parametrize(
    "boundary",
    [
        "outer",
        "state",
        "policy",
        "decision",
        "opportunity",
        "request",
        "executed",
        "feedback",
        "batch",
        "update",
        "application",
        "drain",
        "host_state",
    ],
)
def test_continuation_rejects_unknown_or_non_integer_versions_at_every_root(example, populated_wire, version, boundary):
    document = json.loads(populated_wire)
    target = {
        "outer": document,
        "state": document["state"],
        "policy": document["state"]["policy"],
        "decision": document["state"]["decisions"][0],
        "opportunity": document["state"]["decisions"][0]["opportunity"],
        "request": document["state"]["decisions"][0]["opportunity"]["request"],
        "executed": document["state"]["decisions"][0]["executed"],
        "feedback": document["state"]["feedback"][0],
        "batch": document["state"]["feedback"][0]["batch"],
        "update": document["state"]["updates"][0],
        "application": document["state"]["updates"][0]["application"],
        "drain": document["state"]["drains"][0],
        "host_state": document["host_state"],
    }[boundary]
    target["schema_version"] = version
    wire = json.dumps(document)
    with pytest.raises(ValueError):
        read_continuation(wire, state_type=example.HostConsumerState)


@pytest.mark.parametrize("reader", [read_replay_state, read_replay_feedback])
@pytest.mark.parametrize("wire", [b"", b"\xff", b'{"schema_version":1,"schema_version":1}', b"[" * 65 + b"]" * 65])
def test_bounded_readers_reject_empty_non_utf8_duplicate_and_deep_wire(reader, wire):
    with pytest.raises((ValueError, UnicodeError)):
        reader(wire)


@pytest.mark.parametrize("limit", [0, -1, True, 1.0])
def test_reader_requires_a_strict_positive_byte_limit(example, limit):
    wire = example.policy().to_json()
    with pytest.raises(ValueError, match="positive integer"):
        read_replay_state(wire, maximum_bytes=limit)


def test_readers_use_encoded_bytes_and_never_load_a_path_or_dynamic_state_class(example):
    replay = trace(example)
    wire = replay.state.to_json()
    encoded = wire.encode()
    maximum_bytes = len(encoded)
    assert read_replay_state(encoded, maximum_bytes=maximum_bytes).to_json() == wire
    with pytest.raises(ValueError, match="exceeds"):
        read_replay_state(wire, maximum_bytes=maximum_bytes - 1)
    multibyte = '"' + "é" * 8 + '"'
    character_count = len(multibyte)
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        read_replay_state(multibyte, maximum_bytes=character_count)
    unsupported_path = cast(str, Path("untrusted.json"))
    with pytest.raises(TypeError, match="not a path"):
        read_replay_state(unsupported_path)
    continuation = replay.export_continuation().to_json()
    with pytest.raises(TypeError, match="concrete"):
        read_continuation(continuation, state_type=CoreContract)


def test_feedback_round_trip_preserves_unknown_receipt_and_null_instead_of_inventing_observation(example):
    replay, _ = session(example)
    first = emit(example, replay, 0, at=0, end=1)
    feedback = example.feedback(first, (None, 0.0, 0.5), visible=2, delay=0)
    from tests.replay.support import set_clocks

    feedback = set_clocks(feedback, received_at=None)
    decoded = read_replay_feedback(feedback.to_json())
    assert decoded.to_json() == feedback.to_json()
    assert decoded.batch.timing is not None
    assert decoded.batch.timing.received_at is None
    replay.admit_feedback((decoded,))
    replay.advance(eligible_through=example.instant(3), apply_at=example.instant(3))
    assert replay.state.update_count == 0
    assert len(replay.pending) == 1


def test_reader_rejects_ready_feedback_injected_into_a_drain_that_claimed_to_close_without_it(example):
    replay, host = session(example)
    first = emit(example, replay, 0, at=0, end=1)
    emit(example, replay, 1, at=1, end=2)
    emit(example, replay, 2, cutoff=2, at=3, end=4)
    # F0 would have been accepted just before D2: D_before=2, producer=0,
    # so exactly one intervening decision. D2's complete cohort is now closed.
    backfill = example.feedback(first, (1.0, 0.0, 0.5), visible=2, delay=1)
    before = host.export_state().to_json()
    with pytest.raises(ValueError, match="backfill"):
        replay.admit_feedback((backfill,))
    document = json.loads(replay.state.to_json())
    document["feedback"] = [backfill.model_dump(mode="json")]
    wire = json.dumps(document)
    with pytest.raises(ValueError):
        read_replay_state(wire)
    assert host.export_state().to_json() == before
