"""Versioned independently authored finite public CI lifecycle oracle."""

import hashlib
import json
from itertools import permutations

import pytest

from coleman.registry.definitions import Catalogue
from coleman.registry.envelopes import RegisteredExecution, RegisteredFeedback, RegistryRequest
from coleman.registry.lifecycle import RegistryState
from tests.registry.test_envelopes import capability
from tests.registry.test_lifecycle import FIXTURES


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_literal_catalogue_history_oracle_exact_bytes_and_stdlib_hashes():
    oracle = json.loads((FIXTURES / "oracle.json").read_text(encoding="utf-8"))
    for name, contract in (("catalogue", Catalogue), ("history", RegistryState)):
        literal = (FIXTURES / (name + ".json")).read_text(encoding="utf-8").rstrip("\n")
        record = contract.from_json(literal)
        assert record.to_json() == literal
        assert record.fingerprint == oracle[name + "_fingerprint"] == _digest(json.loads(literal))
    state = RegistryState.from_json((FIXTURES / "history.json").read_bytes())
    for row in oracle["rounds"]:
        snapshot = state.snapshot(row["round"])
        selected = [member for member in snapshot.members if member.status == "active"]
        projection = {
            "schema_version": 1,
            "policy": state.policy.model_dump(mode="json"),
            "active": [
                {
                    "definition": state.catalogue.resolve("expert", member.member).model_dump(mode="json"),
                    "state_epoch": member.state_epoch.model_dump(mode="json") if member.state_epoch else None,
                }
                for member in selected
            ],
        }
        assert _digest(projection) == row["active_contract_fingerprint"]
        assert snapshot.feasible_set("public-space").version == row["active_contract_fingerprint"]
    assert oracle["outcomes"][1]["rows"] == [{"member": "A", "value": 0.5, "status": "observed"}]
    assert oracle["outcomes"][2]["rows"][1] == {"member": "B", "value": None, "status": "unavailable"}
    assert oracle["empty_round5"] == {"action": "abstain", "adaptive_updates": 0, "physical_cost": None}


def test_independently_authored_outer_producer_goldens_preserve_all_inner_core_hashes():
    oracle = json.loads((FIXTURES / "producer-oracle.json").read_text(encoding="utf-8"))
    records = {}
    for name, contract in (
        ("request", RegistryRequest),
        ("execution", RegisteredExecution),
        ("feedback", RegisteredFeedback),
    ):
        literal = (FIXTURES / (name + ".json")).read_text(encoding="utf-8").rstrip("\n")
        record = contract.from_json(literal)
        assert record.to_json() == literal
        assert record.fingerprint == oracle[name + "_fingerprint"] == _digest(json.loads(literal))
        records[name] = record
    assert records["request"].request.fingerprint == oracle["core_request_fingerprint"]
    assert records["execution"].executed.fingerprint == oracle["core_execution_fingerprint"]
    assert records["feedback"].batch.fingerprint == oracle["core_feedback_fingerprint"]
    records["feedback"].validate_for(records["execution"], capability())
    literal = (FIXTURES / "sealed-history.json").read_text(encoding="utf-8").rstrip("\n")
    retained = RegistryState.from_json(literal)
    assert retained.to_json() == literal
    assert retained.fingerprint == oracle["sealed_history_fingerprint"] == _digest(json.loads(literal))
    assert retained.producers[0].history_fingerprint == records["request"].snapshot.history_fingerprint


def test_set_permutation_is_invariant_and_bijective_relabel_preserves_active_cardinality_and_schedule():
    literal = json.loads((FIXTURES / "history.json").read_text(encoding="utf-8"))
    original = RegistryState.model_validate(literal)
    for order in permutations(literal["catalogue"]["definitions"]):
        changed = RegistryState.model_validate(literal | {"catalogue": literal["catalogue"] | {"definitions": order}})
        assert changed.to_json() == original.to_json()
    for new_names in (("renamed-X", "renamed-Y", "renamed-Z"), ("C", "B", "A")):
        renamed = json.loads(json.dumps(literal))
        mapping = dict(zip(("A", "B", "C"), new_names, strict=True))
        for leaf in renamed["catalogue"]["definitions"]:
            leaf["member"]["id"] = mapping[leaf["member"]["id"]]
        renamed_catalogue = Catalogue.model_validate(renamed["catalogue"])
        for event in renamed["facts"]:
            event["member"]["id"] = mapping[event["member"]["id"]]
            event["definition_fingerprint"] = renamed_catalogue.resolve("expert", event["member"]).fingerprint
            for key in ("state_epoch", "predecessor_epoch"):
                if event[key] is not None:
                    old_name, epoch = event[key]["id"].split(":")
                    event[key]["id"] = mapping[old_name] + ":" + epoch
        changed = RegistryState.model_validate(renamed)
        for round_number in range(6):
            assert {member.id for member in changed.snapshot(round_number).active} == {
                mapping[member.id] for member in original.snapshot(round_number).active
            }
        assert changed.fingerprint != original.fingerprint
    for contract in (Catalogue, RegistryState):
        values = original.catalogue.model_dump() if contract is Catalogue else original.model_dump()
        values["schema_version"] = 2
        with pytest.raises(ValueError):
            contract.model_validate(values)
