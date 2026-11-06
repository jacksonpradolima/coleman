import itertools
import json
import math

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.core.primitives import fingerprint
from coleman.core.units import UnitSemantics
from coleman.native.decisions import FrozenNativeDecision, NativeDecisionContext, native_binding
from coleman.native.publication import publish_native_decision, read_native_decision
from coleman.native.semantics import NATIVE_REWARD_NAMES


def test_roundtrip_is_complete_frozen_and_does_not_impute_loss_or_propensity(record):
    wire = record.to_json()
    actual = read_native_decision(wire.encode("utf-8"))
    assert actual == record
    assert actual.to_json() == wire
    assert actual.fingerprint == fingerprint(json.loads(wire))
    assert actual.observation_round == 0
    assert actual.loss.value is None and actual.loss.status == "not_defined"
    assert actual.sampling.value is None and actual.sampling.status == "not_recorded"
    assert actual.context.binding.reward.semantics.unit == "1"
    assert actual.context.binding.policy.theory_status == "not_claimed"
    with pytest.raises(ValueError, match="frozen"):
        actual.ranking = ("A", "B", "C")


@pytest.mark.parametrize("version", [0, 2, -1, True, False, "1", 1.0, None])
@pytest.mark.parametrize("location", ["record", "context", "policy", "reward"])
def test_all_artifact_versions_are_explicit_and_strict(record, version, location):
    payload = json.loads(record.to_json())
    target = {
        "record": payload,
        "context": payload["context"],
        "policy": payload["context"]["binding"]["policy"],
        "reward": payload["context"]["binding"]["reward"],
    }[location]
    target["schema_version"] = version
    wire = json.dumps(payload)
    with pytest.raises(ValueError):
        read_native_decision(wire)


@pytest.mark.parametrize("location", ["policy", "route", "feedback", "reward"])
def test_unknown_behavior_versions_reject_instead_of_guessing(record, location):
    payload = json.loads(record.to_json())
    binding = payload["context"]["binding"]
    reference = binding["route"]["feedback_ref"] if location == "feedback" else binding[location]["semantic_ref"]
    reference["version"] = "future-5.99"
    wire = json.dumps(payload)
    with pytest.raises(ValueError):
        read_native_decision(wire)


@pytest.mark.parametrize("section", ["loss", "sampling"])
@pytest.mark.parametrize("value", [0, 0.0, -1, False, "unavailable", math.inf, math.nan])
def test_missing_native_loss_and_mass_cannot_be_replaced_with_zero_or_diagnostic_scores(record, section, value):
    payload = json.loads(record.to_json())
    payload[section]["value"] = value
    wire = json.dumps(payload)
    with pytest.raises(ValueError):
        read_native_decision(wire)


@pytest.mark.parametrize("count", [-1, True, False, 0.0, "0", None, math.inf, -math.inf, math.nan])
def test_observation_count_has_step_units_and_never_coerces_or_imputes(record, count):
    payload = json.loads(record.to_json())
    payload["observation_round"] = count
    wire = json.dumps(payload)
    with pytest.raises(ValueError):
        read_native_decision(wire)


@pytest.mark.parametrize(
    "semantics",
    [
        {"unit": "s", "meaning": "coleman.native.accepted_observation_count.v1"},
        {"unit": "step", "meaning": "independent_observations"},
    ],
)
def test_same_count_with_incompatible_units_or_meaning_rejects(record, semantics):
    payload = json.loads(record.to_json())
    payload["observation_semantics"] = semantics
    wire = json.dumps(payload)
    with pytest.raises(ValueError, match="semantics"):
        read_native_decision(wire)


@pytest.mark.parametrize("names", [("A", "A", "C"), ("A", None, "C"), ("A", 4, "C")])
def test_candidate_names_are_strict_and_duplicates_are_never_hidden(record, names):
    payload = json.loads(record.to_json())
    payload["candidates"] = names
    wire = json.dumps(payload)
    with pytest.raises(ValueError):
        read_native_decision(wire)


@pytest.mark.parametrize("ranking", [("A", "B"), ("A", "B", "retired"), ("A", "A", "B")])
def test_completed_ranking_must_cover_exact_current_members(record, ranking):
    payload = json.loads(record.to_json())
    payload["ranking"] = ranking
    wire = json.dumps(payload)
    with pytest.raises(ValueError, match="permutation"):
        read_native_decision(wire)


def test_verbatim_names_are_not_reinterpreted_as_reference_ids_or_paths(record):
    names = ("", "  test/α  ", "../CI/B", 'escaped"[{\\name')
    altered = FrozenNativeDecision.model_validate(
        record.model_copy(update={"candidates": names, "ranking": names[::-1]})
    )
    actual = read_native_decision(altered.to_json())
    assert actual.candidates == names
    assert actual.ranking == names[::-1]
    assert altered.logical_identity == record.logical_identity
    assert altered.fingerprint != record.fingerprint


def test_native_record_has_explicit_byte_limit_and_no_generic_4096_member_cap(record):
    names = tuple(f"test-{index}" for index in range(5001))
    large = FrozenNativeDecision.model_validate(record.model_copy(update={"candidates": names, "ranking": names[::-1]}))
    wire = large.to_json()
    assert len(read_native_decision(wire).ranking) == 5001
    with pytest.raises(ValueError, match="maximum_bytes"):
        read_native_decision(wire, maximum_bytes=len(wire.encode("utf-8")) - 1)


def test_all_ranking_permutations_change_content_but_preserve_logical_identity(record):
    values = [
        FrozenNativeDecision.model_validate(record.model_copy(update={"ranking": order}))
        for order in itertools.permutations(record.candidates)
    ]
    assert len({value.fingerprint for value in values}) == math.factorial(len(record.candidates))
    assert {value.logical_identity for value in values} == {record.logical_identity}
    assert all(read_native_decision(value.to_json()).ranking == value.ranking for value in values)


@pytest.mark.parametrize(
    "change",
    [
        {"purpose": "forecast"},
        {"configuration_fingerprint": fingerprint({"gamma": 0.3})},
        {"input_fingerprint": fingerprint({"build": 2})},
        {"producer_fingerprint": fingerprint({"producer": 2})},
    ],
)
def test_retained_content_changes_conflict_at_same_logical_decision(record, change):
    context = NativeDecisionContext.model_validate(record.context.model_copy(update=change))
    changed = FrozenNativeDecision.model_validate(record.model_copy(update={"context": context}))
    assert changed.logical_identity == record.logical_identity
    assert changed.fingerprint != record.fingerprint


@pytest.mark.parametrize("field", ["execution_id", "decision_id"])
def test_new_host_coordinate_creates_new_logical_identity(record, field):
    context = NativeDecisionContext.model_validate(record.context.model_copy(update={field: "different"}))
    changed = FrozenNativeDecision.model_validate(record.model_copy(update={"context": context}))
    assert changed.logical_identity != record.logical_identity


def test_object_key_order_is_not_semantic_but_candidate_order_is(record):
    payload = json.loads(record.to_json())
    reversed_keys = {key: payload[key] for key in reversed(payload)}
    assert read_native_decision(json.dumps(reversed_keys)).fingerprint == record.fingerprint
    altered = FrozenNativeDecision.model_validate(record.model_copy(update={"candidates": record.candidates[::-1]}))
    assert altered.fingerprint != record.fingerprint


@pytest.mark.parametrize("prefix", ['{"schema_version":1,', '{"context":{"schema_version":1,"schema_version":1},'])
def test_duplicate_keys_cannot_disappear_during_read(record, prefix):
    wire = prefix + record.to_json()[1:]
    with pytest.raises(ValueError, match="Duplicate"):
        read_native_decision(wire)


@pytest.mark.parametrize("unknown", ["future_field", "realized_credits", "paid_cost", "propensities"])
def test_unknown_fields_and_post_evaluation_claims_are_not_silently_accepted(record, unknown):
    payload = json.loads(record.to_json())
    payload[unknown] = 0
    wire = json.dumps(payload)
    with pytest.raises(ValueError):
        read_native_decision(wire)


@pytest.mark.parametrize("limit", [0, -1, True, 2.0, "100", None])
def test_invalid_acquisition_bound_rejects(record, limit):
    wire = record.to_json()
    with pytest.raises(ValueError, match="positive integer"):
        read_native_decision(wire, maximum_bytes=limit)


@pytest.mark.parametrize("wire", [b"", b"\xff", "[" * 65 + "]" * 65, "]", "not JSON"])
def test_bad_utf8_depth_or_grammar_rejects_before_any_native_work(wire):
    with pytest.raises((ValueError, UnicodeError)):
        read_native_decision(wire)


def test_reader_refuses_paths_and_decodes_utf8_size_before_recursive_validation(record):
    invalid = json.loads('{"schema_version":1}')
    with pytest.raises(TypeError):
        read_native_decision(invalid)
    payload = json.loads(record.to_json())
    payload["loss"]["reason"] = "α" * 1000
    wire = json.dumps(payload, ensure_ascii=False)
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        read_native_decision(wire, maximum_bytes=len(wire))


def test_unchecked_instance_is_revalidated_before_creating_artifact_directory(tmp_path, record):
    invalid = record.model_copy(update={"schema_version": True})
    destination = tmp_path / "not-created"
    with pytest.raises(ValueError):
        publish_native_decision(invalid, ArtifactWriter(destination))
    assert not destination.exists()


def test_unknown_route_names_are_not_approximately_selected():
    with pytest.raises(ValueError, match="no audited route"):
        native_binding("EXP3", "RNFail", route="manual-window")


@pytest.mark.parametrize("reward", NATIVE_REWARD_NAMES)
@pytest.mark.parametrize("field,value", [("unit", "s"), ("meaning", "elapsed_seconds")])
def test_fixed_native_reward_identity_rejects_incompatible_units_at_the_actual_reader(record, reward, field, value):
    binding = native_binding("EXP3", reward)
    context = NativeDecisionContext.model_validate(record.context.model_copy(update={"binding": binding}))
    valid = FrozenNativeDecision.model_validate(record.model_copy(update={"context": context}))
    payload = json.loads(valid.to_json())
    payload["context"]["binding"]["reward"]["semantics"][field] = value
    malformed = json.dumps(payload)
    assert read_native_decision(valid.to_json()).context.binding.reward.name == reward
    with pytest.raises(ValueError):
        read_native_decision(malformed)


def test_publisher_revalidates_nested_fixed_reward_units_before_any_write(record, tmp_path):
    original = record.context.binding.reward
    wrong_units = UnitSemantics(unit="s", meaning=original.semantics.meaning)
    invalid_reward = original.model_copy(update={"semantics": wrong_units})
    invalid_binding = record.context.binding.model_copy(update={"reward": invalid_reward})
    invalid_context = record.context.model_copy(update={"binding": invalid_binding})
    invalid_record = record.model_copy(update={"context": invalid_context})
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        publish_native_decision(invalid_record, ArtifactWriter(destination))
    assert not destination.exists()
