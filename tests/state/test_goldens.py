"""Versioned canonical fixtures retain full actual state and independent CI facts."""

import hashlib
import json
import random
from pathlib import Path

import pytest

from coleman.replay.retained_codec import read_retained_continuation
from coleman.state.random_state import PythonRandomState, restore_random
from tests.state.migrations import WrappedState

FIXTURES = Path(__file__).parents[1] / "fixtures/state"


@pytest.mark.parametrize(
    "name",
    [
        "v1/python-random-cached.json",
        "v1/host-two-pending.json",
        "v2/retained-one-pending.json",
        "v2/retained-sealed.json",
        "v2/retained-known-migration.json",
    ],
)
def test_complete_versioned_goldens_round_trip_with_single_file_lf_and_exact_manifest_digest(state_example, name):
    example = state_example
    manifest = json.loads((FIXTURES / "manifest.json").read_bytes())
    assert manifest["schema_version"] == 1
    item = next(row for row in manifest["fixtures"] if row["path"] == name)
    raw = (FIXTURES / name).read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert len(raw) == item["byte_count"]
    assert hashlib.sha256(raw).hexdigest() == item["file_sha256"]
    if name == "v1/python-random-cached.json":
        value = PythonRandomState.from_json(raw)
    elif name == "v1/host-two-pending.json":
        value = example.HostState.from_json(raw)
    else:
        state_type = WrappedState[example.HostState] if "migration" in name else example.HostState
        value = read_retained_continuation(raw, state_type=state_type)
    assert value.schema_version == item["schema_version"]
    assert value.fingerprint == item["model_fingerprint"]
    assert value.to_json().encode() + b"\n" == raw
    assert manifest["example_sha256"] == hashlib.sha256(Path(example.__file__).read_bytes()).hexdigest()


def test_golden_oracles_have_literal_pending_buffer_cache_path_and_sealed_loss_facts(state_example):
    example = state_example
    host = example.HostState.from_json((FIXTURES / "v1/host-two-pending.json").read_bytes())
    assert host.decision_count == 2
    assert host.update_count == 0
    assert len(host.pending) == 2
    assert host.book.spent_seconds == 6
    assert host.book.count_b == 0
    assert host.cadence_count == 1
    assert host.reservoir_remaining == 1
    assert host.reservoir_word == 1
    assert host.backend_refills == 0
    assert host.loss_sums == (0.0, 0.0)
    assert host.cache is not None
    assert host.cache.model_updates == 0
    assert host.cache.value == 2.0
    sealed = read_retained_continuation(
        (FIXTURES / "v2/retained-sealed.json").read_bytes(), state_type=example.HostState
    )
    assert sealed.state.decision_count == 2
    assert sealed.state.update_count == 2
    assert sealed.state.seal is not None
    assert sealed.state.seal.decision_count == 2
    assert sealed.host_state.pending == ()
    assert sealed.host_state.observed_counts == (2, 0)
    assert sealed.host_state.loss_sums == (0.25, 0.0)
    assert sealed.host_state.book.spent_seconds == 6
    assert sealed.host_state.rng.to_json() == host.rng.to_json()


def test_golden_cached_gaussian_is_consumed_without_advancing_backend_words():
    value = PythonRandomState.from_json((FIXTURES / "v1/python-random-cached.json").read_bytes())
    assert value.index == 4
    assert value.gauss_next == -0.1729036003315193
    target = random.Random(0)
    restore_random(target, value)
    assert target.gauss(0.0, 1.0) == -0.1729036003315193
    assert target.getstate()[1] == (*value.words, value.index)
