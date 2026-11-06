"""Owned subprocess driver: actual replay, complete host state and local publication."""

import importlib.util
import json
import os
import sys
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.core.actions import ChoiceAction
from coleman.replay.codec import read_continuation
from coleman.replay.engine import CausalReplay


def load_host():
    specification = importlib.util.spec_from_file_location("authored_ci", Path(__file__).with_name("ci.py"))
    if specification is None or specification.loader is None:
        raise RuntimeError("The copied public host is required")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def choose(example, replay, index, at, end):
    return replay.emit(replay.prepare(example.opportunity(index, at=at, end=end)), execution="offline_identity_v1")


def selected(record):
    action = record.executed.action
    if not isinstance(action, ChoiceAction):
        raise ValueError("The installed driver requires its explicit chosen-only profile")
    return action.chosen.id


def publish(writer, path, continuation, *, fault=None):
    encoded = continuation.to_json().encode("utf-8")

    def chunks():
        midpoint = len(encoded) // 2
        yield encoded[:midpoint]
        if fault == "before":
            os._exit(81)
        yield encoded[midpoint:]

    writer.write_immutable_stream(path, chunks(), maximum_bytes=len(encoded))
    if fault == "after":
        os._exit(82)


def execute(mode, owned):
    example = load_host()
    owned.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(owned)
    initial, middle = owned / "initial.json", owned / "middle.json"
    if mode == "resume":
        checkpoint = middle if middle.exists() else initial
        continuation = read_continuation(checkpoint.read_bytes(), state_type=example.HostConsumerState)
        host = example.HostConsumer()
        replay = CausalReplay.from_continuation(
            host, continuation, state_type=example.HostConsumerState, state_codec=example.CODEC
        )
    else:
        host = example.HostConsumer()
        replay = CausalReplay(host, example.policy(), state_type=example.HostConsumerState, state_codec=example.CODEC)
        replay.admit_history(example.history())
        choose(example, replay, 0, 0, 1)
        publish(writer, initial, replay.export_continuation())
    if replay.state.decision_count == 1:
        choose(example, replay, 1, 1, 2)
        first = replay.state.decisions[0]
        replay.admit_feedback((example.feedback(first, (1.0, 0.0, 0.5), visible=2, delay=1),))
        choose(example, replay, 2, 3, 4)
        fault = {"crash-before": "before", "crash-after": "after"}.get(mode)
        publish(writer, middle, replay.export_continuation(), fault=fault)
    # Exact replay of already accepted input must never perform another update.
    first = replay.state.decisions[0]
    replay.admit_feedback((example.feedback(first, (1.0, 0.0, 0.5), visible=2, delay=1),))
    second = replay.state.decisions[1]
    replay.admit_feedback((example.feedback(second, (0.0, 0.0, 0.5), visible=4, delay=1),))
    choose(example, replay, 3, 5, 6)
    third = replay.state.decisions[2]
    replay.admit_feedback((example.feedback(third, (1.0, 0.0, 0.5), visible=6, delay=1),))
    replay.advance(eligible_through=example.instant(6), apply_at=example.instant(6))
    choose(example, replay, 4, 7, 8)
    final = replay.export_continuation()
    publish(writer, owned / "final.json", final)
    state = host.export_state()
    print(
        json.dumps(
            {
                "choices": [selected(row) for row in replay.state.decisions],
                "decisions": state.decision_count,
                "updates": state.update_count,
                "counts": state.observed_counts,
                "sums": state.loss_sums,
                "prefix": replay.state.causal_prefix_fingerprint,
                "continuation": final.fingerprint,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    execute(sys.argv[1], Path(sys.argv[2]))
