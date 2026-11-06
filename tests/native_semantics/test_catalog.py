"""Typed native source declarations, exact aliases and explicit unsupported interpretations."""

import inspect
import math
from pathlib import Path

import pytest

from coleman.core.primitives import canonical_json
from coleman.native import semantics
from coleman.native.semantics import (
    BASELINE_REVISION,
    NATIVE_POLICY_NAMES,
    NATIVE_REWARD_NAMES,
    NATIVE_SEMANTIC_VERSION,
    BooleanDefault,
    IntegerDefault,
    NativeAgentRoute,
    NativeCatalog,
    NativeLifecycleProfile,
    NativeParameterDefinition,
    NativePolicyEntry,
    NativeRewardDefinition,
    NativeSourcePin,
    NumberDefault,
    RequiredDefault,
    get_native_policy,
    get_native_reward,
    list_native_policies,
    list_native_rewards,
    native_catalog,
)
from tests.native_semantics.support import ROOT, fixture


def test_complete_catalog_round_trip_sorting_and_all_exact_public_aliases():
    catalog = native_catalog()
    assert len(catalog.policies) == 37
    assert len(catalog.rewards) == 6
    assert tuple(row.name for row in catalog.policies) == tuple(sorted(NATIVE_POLICY_NAMES))
    assert tuple(row.name for row in catalog.rewards) == tuple(sorted(NATIVE_REWARD_NAMES))
    assert NativeCatalog.from_json(catalog.to_json()).to_json() == catalog.to_json()
    payload = catalog.model_dump(mode="json")
    payload["policies"].reverse()
    payload["rewards"].reverse()
    assert NativeCatalog.model_validate(payload).to_json() == catalog.to_json()
    expected = {row["name"] for row in fixture("legacy-dispatch.json")["policies"]}
    assert {row.name for row in list_native_policies()} == expected
    for row in catalog.policies:
        assert get_native_policy(row.name) == row
        assert row.theory_status == "not_claimed"
        assert row.semantic_ref.version == NATIVE_SEMANTIC_VERSION
        assert row.source_evidence
        assert row.guidance
    for row in catalog.rewards:
        assert get_native_reward(row.name) == row
        assert row.source_evidence
        assert row.coverage


def test_policy_and_reward_parameters_match_real_constructor_defaults_without_running_the_algorithms():
    import coleman.policy as policies
    import coleman.reward as rewards

    for owner, entries, suffix in (
        (policies, list_native_policies(), "Policy"),
        (rewards, list_native_rewards(), "Reward"),
    ):
        for entry in entries:
            actual = inspect.signature(getattr(owner, entry.name + suffix)).parameters
            assert {row.name for row in entry.parameters} == set(actual), entry.name
            for parameter in entry.parameters:
                value = actual[parameter.name].default
                declared = parameter.default
                if value is inspect.Parameter.empty:
                    assert isinstance(declared, RequiredDefault)
                    assert declared.value is None
                else:
                    assert declared.value == value, (entry.name, parameter.name)
                    expected = {bool: BooleanDefault, int: IntegerDefault, float: NumberDefault}[type(value)]
                    assert isinstance(declared, expected)


def test_all_declared_source_pins_match_independent_public_baseline_bytes_and_line_ranges():
    sources = {row["path"]: row for row in fixture("catalog-source-evidence.json")["resources"]}
    used = set()
    for entry in (*list_native_policies(), *list_native_rewards()):
        pins = list(entry.source_evidence)
        if isinstance(entry, NativePolicyEntry):
            pins.extend(pin for route in entry.routes for pin in route.source_evidence)
        for pin in pins:
            baseline = sources[pin.path]
            assert pin.revision == BASELINE_REVISION
            assert pin.fingerprint == "sha256:" + baseline["source_sha256"]
            assert 1 <= pin.first_line <= pin.last_line <= baseline["line_count"]
            used.add(pin.path)
    assert used == set(sources)


def test_claimed_lifecycle_inherited_context_and_imported_helpers_have_explicit_owner_pins():
    shared_consumers = {
        "BayesianUCB",
        "BootstrappedThompson",
        "ChangeDetectionUCB",
        "CombinatorialThompson",
        "CombinatorialUCB",
        "DiscountedUCB",
        "DuelingUCB",
        "EXP3",
        "EXP3IX",
        "EpsilonDecreasing",
        "KLUCB",
        "MOSSUCB",
        "PHE",
        "PairwiseThompsonRanking",
        "PortfolioUCB",
        "Pursuit",
        "Softmax",
        "ThompsonSampling",
        "UCBTuned",
        "UCBV",
    }
    contextual = {"ContextualEpsilonGreedy", "LinTS", "LinUCB", "SWContextualEpsilonGreedy", "SWLinTS", "SWLinUCB"}
    for entry in list_native_policies():
        paths = {pin.path for pin in entry.source_evidence}
        assert {"coleman/agent/base.py", "coleman/policy/base.py", "coleman/environment.py"} <= paths, entry.name
        reset = [pin for pin in entry.source_evidence if pin.path == "coleman/environment.py"]
        assert any(pin.first_line <= 223 and pin.last_line >= 237 for pin in reset), entry.name
        assert ("coleman/policy/mab/shared.py" in paths) == (entry.name in shared_consumers), entry.name
        assert any(route.context_required for route in entry.routes) == (entry.name in contextual), entry.name
        for route in entry.routes:
            route_paths = {pin.path for pin in route.source_evidence}
            assert "coleman/policy/base.py" in route_paths, route.semantic_ref.id
            if entry.name in shared_consumers:
                assert "coleman/policy/mab/shared.py" in route_paths, route.semantic_ref.id
            if route.context_required:
                assert "coleman/agent/contextual.py" in route_paths, route.semantic_ref.id


def test_all_41_routes_pin_the_real_method_owners_in_their_actual_agent_mro():
    from coleman.agent import ContextualAgent, RewardAgent, RewardSlidingWindowAgent, SlidingWindowContextualAgent

    actual_classes = {
        agent.__module__ + "." + agent.__qualname__: agent
        for agent in (RewardAgent, ContextualAgent, RewardSlidingWindowAgent, SlidingWindowContextualAgent)
    }
    count = 0
    for entry in list_native_policies():
        for route in entry.routes:
            count += 1
            actual_class = actual_classes[route.agent_class]
            paths = {pin.path for pin in route.source_evidence}
            for method in ("choose", "update_actions", "update_bandit", "reset", "observe"):
                source = inspect.getsourcefile(getattr(actual_class, method))
                assert source is not None
                owner = Path(source).relative_to(ROOT).as_posix()
                assert owner in paths, (route.semantic_ref.id, method, owner)
    assert count == 41


@pytest.mark.parametrize("name", ["exp3", "Unknown", "../Random", None, 0])
def test_unknown_or_nonstring_native_policy_aliases_reject(name):
    with pytest.raises(ValueError, match="alias|Unknown"):
        get_native_policy(name)


@pytest.mark.parametrize("name", ["rnfail", "Unknown", "../RNFail", None, 0])
def test_unknown_or_nonstring_native_reward_aliases_reject(name):
    with pytest.raises(ValueError, match="alias|Unknown"):
        get_native_reward(name)


@pytest.mark.parametrize("use", ["sampled_choice", "subset", "mixture", "pairwise", "probability", None, 0])
def test_native_ranking_lookup_refuses_unsupported_generic_interpretations(use):
    with pytest.raises(ValueError, match="native_ranking only"):
        get_native_policy("EXP3", use=use)


@pytest.mark.parametrize("version", [0, 2, 999, True, "1", None])
def test_catalog_rejects_unknown_and_coerced_artifact_versions(version):
    payload = native_catalog().model_dump(mode="json")
    payload["schema_version"] = version
    with pytest.raises(ValueError):
        NativeCatalog.model_validate(payload)


@pytest.mark.parametrize("section", ["policies", "rewards"])
def test_catalog_rejects_duplicates_even_when_the_inventory_length_is_unchanged(section):
    payload = native_catalog().model_dump(mode="json")
    payload[section][0] = payload[section][1]
    with pytest.raises(ValueError, match="unique"):
        NativeCatalog.model_validate(payload)


@pytest.mark.parametrize("section", ["policies", "rewards"])
def test_catalog_requires_every_policy_and_reward(section):
    payload = native_catalog().model_dump(mode="json")
    payload[section].pop()
    with pytest.raises(ValueError):
        NativeCatalog.model_validate(payload)


@pytest.mark.parametrize("field,value", [("baseline_revision", "0" * 40), ("future_extension", 1)])
def test_catalog_rejects_another_source_revision_and_unknown_fields(field, value):
    payload = native_catalog().model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValueError):
        NativeCatalog.model_validate(payload)


@pytest.mark.parametrize("model,alias", [(NativePolicyEntry, "EXP3"), (NativeRewardDefinition, "RNFail")])
@pytest.mark.parametrize(
    "mutation", ["unknown_name", "unknown_version", "wrong_class", "empty_evidence", "future_schema"]
)
def test_entry_identity_class_source_and_artifact_version_are_closed(model, alias, mutation):
    entry = get_native_policy(alias) if model is NativePolicyEntry else get_native_reward(alias)
    payload = entry.model_dump(mode="json")
    if mutation == "unknown_name":
        payload["name"] = "Unregistered"
    elif mutation == "unknown_version":
        payload["semantic_ref"]["version"] = "future"
    elif mutation == "wrong_class":
        payload["policy_class" if model is NativePolicyEntry else "reward_class"] = "external.provider.Unregistered"
    elif mutation == "empty_evidence":
        payload["source_evidence"] = []
    else:
        payload["schema_version"] = 2
    with pytest.raises(ValueError):
        model.model_validate(payload)


def test_policy_parameters_and_routes_sort_but_duplicate_or_foreign_declarations_reject():
    policy = get_native_policy("FRRMAB")
    payload = policy.model_dump(mode="json")
    payload["parameters"].reverse()
    payload["routes"].reverse()
    assert NativePolicyEntry.model_validate(payload).to_json() == policy.to_json()
    duplicate_parameter = policy.model_dump(mode="json")
    duplicate_parameter["parameters"].append(duplicate_parameter["parameters"][0])
    with pytest.raises(ValueError, match="Duplicate native constructor parameter"):
        NativePolicyEntry.model_validate(duplicate_parameter)
    duplicate_route = policy.model_dump(mode="json")
    duplicate_route["routes"].append(duplicate_route["routes"][0])
    with pytest.raises(ValueError, match="Duplicate native agent route"):
        NativePolicyEntry.model_validate(duplicate_route)
    foreign_route = policy.model_dump(mode="json")
    foreign_route["routes"] = [get_native_policy("Random").routes[0].model_dump(mode="json")]
    with pytest.raises(ValueError, match="another policy"):
        NativePolicyEntry.model_validate(foreign_route)


@pytest.mark.parametrize("mutation", ["version", "feedback", "agent_class"])
def test_route_versions_feedback_identity_and_builtin_class_boundary(mutation):
    payload = get_native_policy("EXP3").routes[0].model_dump(mode="json")
    if mutation == "version":
        payload["semantic_ref"]["version"] = "future"
    elif mutation == "feedback":
        payload["feedback_ref"]["id"] += ".different"
    else:
        payload["agent_class"] = "external.agent.Custom"
    with pytest.raises(ValueError):
        NativeAgentRoute.model_validate(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("first_line", 0),
        ("first_line", True),
        ("last_line", 0),
        ("last_line", "1"),
        ("path", "../coleman/policy/base.py"),
        ("path", "external.py"),
        ("revision", "0" * 40),
        ("fingerprint", "sha256:missing"),
    ],
)
def test_source_pins_keep_explicit_public_revision_paths_hashes_and_line_units(field, value):
    payload = get_native_policy("EXP3").source_evidence[0].model_dump(mode="json")
    assert payload["revision"] == BASELINE_REVISION
    payload[field] = value
    with pytest.raises(ValueError):
        NativeSourcePin.model_validate(payload)


def test_source_range_does_not_silently_reverse_its_line_bounds():
    payload = get_native_policy("EXP3").source_evidence[0].model_dump(mode="json")
    payload.update(first_line=10, last_line=9)
    with pytest.raises(ValueError, match="reversed"):
        NativeSourcePin.model_validate(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0.2"])
def test_number_defaults_reject_nonfinite_and_coerced_values(value):
    payload = {"kind": "number", "value": value}
    with pytest.raises(ValueError):
        NumberDefault.model_validate(payload)


def test_default_representation_retains_negative_finite_zero_and_required_null_distinctly():
    negative = NumberDefault(kind="number", value=-2.5)
    zero = NumberDefault(kind="number", value=0.0)
    signed_zero = NumberDefault(kind="number", value=-0.0)
    missing = RequiredDefault(kind="required", value=None)
    assert negative.value == -2.5
    assert zero.value == 0.0
    assert math.copysign(1.0, signed_zero.value) == -1.0
    encoded_zero = canonical_json(zero.model_dump(mode="json"))
    assert canonical_json(signed_zero.model_dump(mode="json")) != encoded_zero
    assert canonical_json(missing.model_dump(mode="json")) != encoded_zero
    # This is literal representation, not admission of -2.5 by any constructor.
    assert NumberDefault.model_validate(negative.model_dump(mode="json")) == negative


@pytest.mark.parametrize(
    "model,kind,value",
    [
        (IntegerDefault, "integer", True),
        (IntegerDefault, "integer", 1.0),
        (BooleanDefault, "boolean", 0),
        (RequiredDefault, "required", 0),
    ],
)
def test_constructor_default_tags_do_not_conflate_counts_booleans_or_required_values(model, kind, value):
    with pytest.raises(ValueError):
        model.model_validate({"kind": kind, "value": value})


@pytest.mark.parametrize(
    "kind,default",
    [
        ("integer", {"kind": "number", "value": 0.5}),
        ("boolean", {"kind": "integer", "value": 0}),
        ("policy_sequence", {"kind": "number", "value": 1.0}),
    ],
)
def test_parameter_rejects_a_well_typed_default_for_the_wrong_parameter_shape(kind, default):
    payload = get_native_policy("EXP3").parameters[0].model_dump(mode="json")
    payload.update(kind=kind, default=default)
    with pytest.raises(ValueError, match="wrong representation"):
        NativeParameterDefinition.model_validate(payload)


def test_reward_fields_and_parameters_are_unique_and_reported_names_remain_separate_from_aliases():
    reward = get_native_reward("TopKRNFail")
    payload = reward.model_dump(mode="json")
    payload["input_fields"].append(payload["input_fields"][0])
    with pytest.raises(ValueError, match="Duplicate native reward input field"):
        NativeRewardDefinition.model_validate(payload)
    payload = reward.model_dump(mode="json")
    payload["parameters"].append(payload["parameters"][0])
    with pytest.raises(ValueError, match="Duplicate native constructor parameter"):
        NativeRewardDefinition.model_validate(payload)
    assert all(reward.reported_name and reward.reward_class for reward in list_native_rewards())


@pytest.mark.parametrize("name", NATIVE_REWARD_NAMES)
@pytest.mark.parametrize("field,value", [("unit", "s"), ("meaning", "elapsed_seconds")])
def test_retained_native_reward_identity_rejects_changed_credit_units_or_meaning(name, field, value):
    payload = get_native_reward(name).model_dump(mode="json")
    payload["semantics"][field] = value
    with pytest.raises(ValueError):
        NativeRewardDefinition.model_validate(payload)


def test_lifecycle_rejects_unqualified_claims_instead_of_promoting_constructor_pickle_evidence():
    payload = get_native_policy("DuelingUCB").lifecycle.model_dump(mode="json")
    payload["checkpoint"] = "all_states_verified"
    with pytest.raises(ValueError):
        NativeLifecycleProfile.model_validate(payload)
    assert get_native_policy("PairwiseThompsonRanking").lifecycle.checkpoint == "unsupported"
    assert get_native_policy("FRRMAB").migration == "restricted"


def test_bundled_catalog_limit_and_missing_resource_are_actual_reader_errors(tmp_path, monkeypatch):
    resource = tmp_path / "catalog-v1.json"
    resource.write_bytes(b" " * (1024 * 1024 + 1))
    native_catalog.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(semantics, "files", lambda package: Path(tmp_path))
        with pytest.raises(ValueError, match="one-MiB"):
            native_catalog()
        resource.unlink()
        with pytest.raises(OSError):
            native_catalog()
    native_catalog.cache_clear()
    assert len(native_catalog().policies) == 37
