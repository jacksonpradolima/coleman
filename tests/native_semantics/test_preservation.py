"""Pinned source mechanisms and public fixtures, independently of runtime floating-point traces."""

import hashlib

import pytest

from coleman.core.primitives import fingerprint
from coleman.native.publication import read_native_decision
from coleman.native.semantics import list_native_policies, list_native_rewards, native_catalog
from tests.native_semantics.support import FIXTURES, ROOT, fixture, mechanism_digest


def test_all_46_python_mechanisms_and_11_markers_retain_public_v4_behavior():
    manifest = fixture("source-preservation.json")
    assert manifest["source_revision"] == "1edbee874129aa85e26e592d9649826e85e1ac6b"
    assert manifest["ast_profile"] == "cpython-3.14-without-docstrings-v1"
    python_rows = [row for row in manifest["resources"] if row["kind"] == "python"]
    markers = [row for row in manifest["resources"] if row["kind"] == "marker"]
    assert len(python_rows) == 46
    assert len(markers) == 11
    for row in python_rows:
        assert mechanism_digest((ROOT / row["path"]).read_text()) == row["mechanism_sha256"], row["path"]
    for row in markers:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["source_sha256"], row["path"]


def test_fixture_manifest_retains_complete_authored_inputs_and_their_provenance():
    manifest = fixture("manifest.json")
    for row in manifest["files"]:
        assert hashlib.sha256((FIXTURES / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    inputs = fixture("legacy-ci-inputs.json")
    assert len(inputs["rows"]) == 12
    assert {row["BuildId"] for row in inputs["rows"]} == {1, 2, 3, 4}
    assert {row["Name"] for row in inputs["rows"]} == {"unit", "integration", "system"}
    assert inputs["duration_unit"] == "s"
    assert inputs["budget"] == {"mode": "ratio", "value": 0.5, "modeled_seconds": 3.0}
    assert inputs["evidence_basis"] == "authored_synthetic_ci"


def test_dispatch_fixture_distinguishes_completion_factory_failures_and_frrmab_variability():
    cases = fixture("legacy-dispatch.json")
    names = [row["name"] for row in cases["policies"]]
    assert len(names) == len(set(names)) == 37
    completed = [row for row in cases["policies"] if row["factory_outcome"] == "completed"]
    failed = [row for row in cases["policies"] if row["factory_outcome"] == "failed"]
    assert len(completed) == 34
    assert {row["name"] for row in failed} == {"SlMAB", "SWLinTS", "SWContextualEpsilonGreedy"}
    variable = [row for row in cases["policies"] if row["single_trace_reproducibility"] == "not_qualified"]
    assert [row["name"] for row in variable] == ["FRRMAB"]
    assert cases["comparison_scope"] == "same_host_cpython_3_14_6_macos_arm64_native_pins"


def test_preservation_hash_removes_documentation_only_not_executable_string_constants():
    source = '"""old module"""\ndef method():\n    """old method"""\n    return "actual-value"\n'
    changed_docs = source.replace("old module", "new module").replace("old method", "new method")
    changed_value = source.replace("actual-value", "different-value")
    assert mechanism_digest(source) == mechanism_digest(changed_docs)
    assert mechanism_digest(source) != mechanism_digest(changed_value)


@pytest.mark.parametrize("name", ["exp3-decision.json", "contextual-decision.json", "window-decision.json"])
def test_portable_frozen_codec_goldens_retain_full_producer_inputs_and_never_claim_sampling_mass(name):
    payload = (FIXTURES / name).read_bytes()
    record = read_native_decision(payload)
    assert (record.to_json() + "\n").encode() == payload
    producer = fixture("fixture-producer.json")
    case = next(row for row in producer["cases"] if row["file"] == name)
    assert record.context.producer_fingerprint == fingerprint(producer)
    assert record.context.input_fingerprint == fingerprint(case)
    assert record.context.configuration_fingerprint == fingerprint(
        {
            "policy": case["policy"],
            "parameters": case["parameters"],
            "window_size": case["window_size"],
            "seed": producer["seed"],
        }
    )
    assert record.context.binding.catalog_fingerprint == producer["catalog_fingerprint"]
    assert record.observation_round == len(case["prior_observations"])
    assert record.window_size == case["window_size"]
    assert set(record.candidates) == set(case["current_verdicts"])
    assert set(record.ranking) == set(record.candidates)
    assert record.sampling.status == "not_recorded"
    assert record.sampling.value is None
    assert record.loss.status == "not_defined"
    assert record.loss.value is None


def test_compact_catalog_lookup_golden_retains_every_exact_name_route_and_credit_definition():
    expected = fixture("catalog-lookups.json")
    assert expected["schema_version"] == 1
    assert expected["catalog_fingerprint"] == native_catalog().fingerprint
    actual_policies = [
        {
            "name": row.name,
            "semantic_ref": row.semantic_ref.model_dump(mode="json"),
            "policy_class": row.policy_class,
            "theory_status": row.theory_status,
            "routes": [
                {
                    "semantic_ref": route.semantic_ref.model_dump(mode="json"),
                    "status": route.status,
                    "feedback_ref": route.feedback_ref.model_dump(mode="json"),
                    "agent_class": route.agent_class,
                }
                for route in row.routes
            ],
        }
        for row in list_native_policies()
    ]
    actual_rewards = [
        {
            "name": row.name,
            "semantic_ref": row.semantic_ref.model_dump(mode="json"),
            "reward_class": row.reward_class,
        }
        for row in list_native_rewards()
    ]
    assert len(expected["policies"]) == 37
    assert expected["policies"] == actual_policies
    assert len(expected["rewards"]) == 6
    assert expected["rewards"] == actual_rewards
