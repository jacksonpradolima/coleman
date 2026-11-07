"""Coupled finite admission, strict versions and bounded duplicate-safe ingress."""

import pytest

from coleman.registry.definitions import Catalogue, MemberDefinition
from coleman.registry.envelopes import seal_request
from coleman.registry.lifecycle import LifecycleFact, RegistryState
from coleman.registry.limits import RegistryLimits, admit_bytes
from tests.registry.test_envelopes import registered_request
from tests.registry.test_lifecycle import fact, history


@pytest.mark.parametrize(
    "field,hard",
    [
        ("maximum_definitions", 4096),
        ("maximum_facts", 4096),
        ("maximum_producers", 8192),
        ("maximum_resources", 4096),
        ("maximum_publications", 4096),
        ("maximum_prefix_work", 1048576),
        ("maximum_checkpoint_work", 1048576),
        ("maximum_bytes", 16 * 1024 * 1024),
    ],
)
def test_selected_limits_are_positive_strict_and_finitely_hard_bounded(field, hard):
    for value in (0, -1, True, "1", hard + 1):
        with pytest.raises(ValueError):
            RegistryLimits.model_validate({field: value})
    assert getattr(RegistryLimits.model_validate({field: hard}), field) == hard
    assert RegistryLimits().maximum_producers >= 32 * 95


def test_actual_bytes_counts_fact_and_producer_caps_refuse_without_eviction():
    original = history(count=1)
    exact_bytes = len(original.to_json().encode("utf-8"))
    admit_bytes(original, RegistryLimits(maximum_bytes=exact_bytes))
    smaller_bytes = RegistryLimits(maximum_bytes=exact_bytes - 1)
    with pytest.raises(ValueError, match="maximum_bytes"):
        admit_bytes(original, smaller_bytes)
    for limits in (RegistryLimits(maximum_definitions=1), RegistryLimits(maximum_bytes=128)):
        candidate = original.model_dump() | {"limits": limits}
        with pytest.raises(ValueError):
            RegistryState.model_validate(candidate)
    limited = RegistryState.model_validate(original.model_dump() | {"limits": RegistryLimits(maximum_facts=1)})
    before = limited.to_json()
    assert limited.append(limited.facts[0]) == limited
    another_fact = fact(limited, name="B")
    with pytest.raises(ValueError, match="capacity"):
        limited.append(another_fact)
    assert limited.to_json() == before
    limited = RegistryState.model_validate(original.model_dump() | {"limits": RegistryLimits(maximum_producers=1)})
    sealed = seal_request(limited, registered_request(limited))
    assert seal_request(sealed, registered_request(limited)) == sealed
    another_request = registered_request(sealed, 1, "opportunity-1")
    with pytest.raises(ValueError, match="producer capacity"):
        seal_request(sealed, another_request)
    assert len(sealed.producers) == 1


@pytest.mark.parametrize("reader", [RegistryState.from_json, RegistryState.model_validate_json])
def test_bounded_readers_reject_duplicate_keys_depth_future_versions_and_wrong_kinds(reader):
    original = history(count=1).to_json()
    assert reader(original) == history(count=1)
    for payload in (
        original.replace('"schema_version":1', '"schema_version":2', 1),
        original.replace('"sequence":0', '"sequence":0,"sequence":1'),
        "[" * 65 + "0" + "]" * 65,
        b"\xff",
    ):
        with pytest.raises((ValueError, UnicodeError)):
            reader(payload)
    unsupported = object()
    with pytest.raises(TypeError):
        reader(unsupported)


def test_leaf_writer_and_both_readers_share_the_hard_byte_ceiling():
    leaf = history(count=0).catalogue.definitions[0]
    huge = MemberDefinition.model_validate(
        leaf.model_dump()
        | {
            "member": {"id": "a" * (16 * 1024 * 1024), "version": "v1"},
        }
    )
    with pytest.raises(ValueError, match="maximum_bytes"):
        huge.to_json()
    oversized = '{"schema_version":1,"padding":"' + "a" * (16 * 1024 * 1024) + '"}'
    for reader in (MemberDefinition.from_json, MemberDefinition.model_validate_json):
        with pytest.raises(ValueError, match="maximum_bytes"):
            reader(oversized)


def test_prefix_work_is_admitted_before_hashes_and_equal_prefixes_are_cached_once(monkeypatch):
    import coleman.registry.lifecycle as lifecycle

    state = history(count=2)
    state = RegistryState.model_validate(state.model_dump() | {"limits": RegistryLimits(maximum_prefix_work=2)})
    state = seal_request(state, registered_request(state))
    state = seal_request(state, registered_request(state, decision_id="other-opportunity"))
    assert len(state.producers) == 2
    original = lifecycle._history_fingerprint
    calls = []

    def counted(policy, facts):
        calls.append(len(facts))
        return original(policy, facts)

    monkeypatch.setattr(lifecycle, "_history_fingerprint", counted)
    assert RegistryState.model_validate(state.model_dump()) == state
    assert calls == [2]
    calls.clear()
    insufficient_work = state.model_dump() | {"limits": RegistryLimits(maximum_prefix_work=1)}
    with pytest.raises(ValueError, match="maximum_prefix_work"):
        RegistryState.model_validate(insufficient_work)
    assert not calls
    invalid = state.model_dump()
    for producer in invalid["producers"]:
        producer["fact_count"] = 3
    with pytest.raises(ValueError, match="available history"):
        RegistryState.model_validate(invalid)


@pytest.mark.parametrize("contract", [Catalogue, MemberDefinition, LifecycleFact, RegistryState])
def test_new_contracts_keep_unknown_field_policy_and_do_not_offer_relaxed_readers(contract):
    state = history(count=1)
    sample = {
        Catalogue: state.catalogue,
        MemberDefinition: state.catalogue.definitions[0],
        LifecycleFact: state.facts[0],
        RegistryState: state,
    }[contract]
    unknown = sample.model_dump() | {"untyped_metadata": {"hidden": "value"}}
    with pytest.raises(ValueError):
        contract.model_validate(unknown)
    serialized = sample.to_json()
    for changes in ({"strict": False}, {"extra": "allow"}):
        with pytest.raises(ValueError):
            contract.model_validate_json(serialized, **changes)
