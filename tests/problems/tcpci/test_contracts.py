"""Strict readers preserve complete hand-authored TCPCI inputs and results."""

import json
from decimal import Decimal

import pytest

from coleman.problems.tcpci import BudgetPrefix, TcpciMapping, TcpciProblem
from coleman.problems.tcpci import TestCase as Case
from tests.problems.tcpci._support import FIXTURES, data, problem


@pytest.mark.parametrize("name,contract", [("problem", TcpciProblem), ("mapping", TcpciMapping)])
def test_hand_authored_wire_roundtrips_all_public_readers(name, contract):
    """Original values and independent canonical action hashes survive readers."""
    raw = (FIXTURES / f"{name}.json").read_text(encoding="utf-8").rstrip("\n")
    expected = json.loads(raw)
    for incoming in (raw, raw.encode(), bytearray(raw.encode())):
        actual = contract.model_validate_json(incoming)
        assert actual.to_json() == raw
        assert json.loads(actual.to_json()) == expected
    assert contract.from_json(raw).to_json() == raw


@pytest.mark.parametrize("route", ["python", "from_json", "model_validate_json"])
@pytest.mark.parametrize("change", ["missing_version", "future_version", "extra"])
def test_unknown_missing_versions_and_fields_reject_before_use(route, change):
    """Every public input reader retains closed schema/version semantics."""
    raw = data()
    if change == "missing_version":
        del raw["schema_version"]
    elif change == "future_version":
        raw["schema_version"] = 2
    else:
        raw["invented"] = "ignored-fields-would-lose-evidence"
    incoming = json.dumps(raw)
    operation = {
        "python": lambda: TcpciProblem.model_validate(raw),
        "from_json": lambda: TcpciProblem.from_json(incoming),
        "model_validate_json": lambda: TcpciProblem.model_validate_json(incoming),
    }[route]
    with pytest.raises(ValueError):
        operation()


@pytest.mark.parametrize("path", ["tests", "mapping", "semantic"])
def test_nested_future_versions_are_not_silently_migrated(path):
    """A nested future generation rejects the complete original envelope."""
    raw = data()
    target = raw[path][0] if path == "tests" else raw[path]
    target["schema_version"] = 2
    incoming = json.dumps(raw)
    with pytest.raises(ValueError):
        TcpciProblem.from_json(incoming)


@pytest.mark.parametrize("kind", ["member_duplicate", "member_version", "member_missing", "label_duplicate"])
def test_reference_identity_is_exact_and_independent_of_labels(kind):
    """Equal display labels are valid; changed or repeated identities reject."""
    raw = data()
    if kind == "member_duplicate":
        raw["tests"][1]["member"] = raw["tests"][0]["member"]
    elif kind == "member_version":
        raw["tests"][1]["member"]["version"] = "foreign"
    elif kind == "member_missing":
        raw["tests"].pop()
    else:
        raw["tests"][1]["display_name"] = raw["tests"][0]["display_name"]
        actual = TcpciProblem.model_validate(raw)
        assert actual.tests[1].display_name == actual.tests[0].display_name
        return
    with pytest.raises(ValueError):
        TcpciProblem.model_validate(raw)


@pytest.mark.parametrize("value", ["-1", "NaN", "Infinity", "0.0000000000001", "1000000000001"])
def test_invalid_or_over_precision_planning_amounts_reject(value):
    """Exact decimal nonnegative/coupled limits are not float tolerances."""
    raw = data()
    raw["tests"][0]["duration_quote"]["value"] = value
    with pytest.raises(ValueError):
        TcpciProblem.model_validate(raw)


@pytest.mark.parametrize("status,value", [("unknown", "1"), ("observed", None), ("not_applicable", "0")])
def test_quote_status_cannot_impute_or_erase_a_value(status, value):
    """Only observed has a nonnull quantity, including exact zero."""
    raw = data()["tests"][0]
    raw["duration_status"] = status
    raw["duration_quote"] = None if value is None else raw["duration_quote"] | {"value": value}
    with pytest.raises(ValueError, match="Exactly observed"):
        Case.model_validate(raw)


@pytest.mark.parametrize("location", ["budget", "quote", "rule_unit", "rule_interpretation", "horizon"])
def test_units_meanings_roles_and_horizons_are_not_interchangeable(location):
    """Same numerical value cannot change seconds, scientific meaning or horizon."""
    raw = data()
    if location == "budget":
        raw["budget"]["semantics"]["meaning"] = "paid-fee"
    elif location == "quote":
        raw["tests"][0]["duration_quote"]["semantics"]["unit"] = "ms"
    elif location == "rule_unit":
        raw["mapping"]["quantity"]["spec"]["semantics"]["unit"] = "1"
    elif location == "rule_interpretation":
        raw["mapping"]["quantity"]["interpretation"] = "raw_outcome"
    else:
        raw["mapping"]["quantity"]["horizon"]["version"] = "another"
    with pytest.raises(ValueError):
        TcpciProblem.model_validate(raw)


@pytest.mark.parametrize("location", ["cost", "remaining", "action", "sampling", "problem_quote"])
def test_mapping_reader_rederives_accounting_and_full_action(location):
    """Forged equal-shape results do not pass as independently checked mappings."""
    raw = data("mapping")
    if location == "cost":
        raw["estimated_cost"]["value"] = "2"
    elif location == "remaining":
        raw["remaining_budget"]["value"] = "1"
    elif location == "action":
        raw["executed"]["action"]["selected"].reverse()
    elif location == "sampling":
        raw["executed"]["sampling"]["mechanism_ref"]["version"] = "foreign"
    else:
        raw["problem"]["tests"][0]["duration_quote"]["value"] = "2"
    incoming = json.dumps(raw)
    with pytest.raises(ValueError):
        TcpciMapping.from_json(incoming)


def test_decimal_zero_canonicalizes_without_changing_retained_labels():
    """Decimal spelling normalizes while arbitrary bounded labels stay verbatim."""
    raw = data()["tests"][0]
    raw["display_name"] = " \tUnicode λ \t"
    raw["duration_quote"]["value"] = "-0.000000000000"
    actual = Case.model_validate(raw)
    assert actual.display_name == " \tUnicode λ \t"
    assert actual.duration_quote is not None
    assert actual.duration_quote.value == Decimal(0)
    assert json.loads(actual.to_json())["duration_quote"]["value"] == "0"


@pytest.mark.parametrize("contract", [TcpciProblem, TcpciMapping, Case, BudgetPrefix])
def test_new_bytearray_reader_guards_length_before_copy(contract):
    """All new public JSON roots retain the hard pre-copy16MiB bound."""
    incoming = bytearray(16 * 1024**2 + 1)
    with pytest.raises(ValueError, match="before copying"):
        contract.model_validate_json(incoming)


def test_duplicate_json_keys_deep_graphs_and_relaxed_options_reject():
    """Shared core guards remain active on both strict public routes."""
    duplicate = '{"schema_version":1,"schema_version":1}'
    deep = "[" * 65 + "0" + "]" * 65
    raw = (FIXTURES / "problem.json").read_bytes()
    with pytest.raises(ValueError, match="Duplicate"):
        TcpciProblem.from_json(duplicate)
    with pytest.raises(ValueError, match="depth"):
        TcpciProblem.model_validate_json(deep)
    with pytest.raises(ValueError, match="strict"):
        TcpciProblem.model_validate_json(raw, strict=False)
    with pytest.raises(ValueError, match="unknown fields"):
        TcpciProblem.model_validate_json(raw, extra="ignore")


def test_complete_catalogue_order_canonicalizes_but_profile_content_remains_bound():
    """A set permutation is invariant while declaration changes alter identity."""
    original = problem()
    raw = data()
    raw["tests"].reverse()
    permuted = TcpciProblem.model_validate(raw)
    assert permuted.to_json() == original.to_json()
    raw["mapping"]["lineage"]["source"] = {"id": "another-source", "version": "1"}
    changed = TcpciProblem.model_validate(raw)
    assert changed.fingerprint != original.fingerprint


@pytest.mark.parametrize("minimum", [None, 1.0])
def test_planning_schema_requires_an_explicit_zero_lower_bound(minimum):
    """A missing or positive schema floor cannot silently change the fixed profile."""
    raw = data()["mapping"]
    raw["quantity"]["spec"]["minimum"] = minimum
    with pytest.raises(ValueError, match="explicit minimum zero"):
        BudgetPrefix.model_validate(raw)


@pytest.mark.parametrize("location", ["budget", "quote"])
def test_exact_declared_schema_upper_bound_applies_to_each_planning_amount(location):
    """Bounds check exact decimal amounts before any prefix search or projection."""
    raw = data()
    raw["mapping"]["quantity"]["spec"]["maximum"] = 2.5
    if location == "quote":
        raw["budget"]["value"] = "2"
    with pytest.raises(ValueError, match="declared resource schema maximum"):
        TcpciProblem.model_validate(raw)


def test_independent_problem_refuses_a_different_valid_action_family():
    """A valid generic choice universe does not become a TCPCI complete ranking."""
    raw = data()
    raw["semantic"]["request"]["space"]["kind"] = "choice"
    with pytest.raises(ValueError, match="complete ranking space"):
        TcpciProblem.model_validate(raw)


def test_reader_refuses_callback_bearing_bytearray_before_copy_or_decode():
    """A conversion-bearing subclass cannot execute code through JSON ingress."""
    callbacks = []

    class CallbackBytes(bytearray):
        """Retain a real conversion hook that must remain uncalled."""

        def __bytes__(self):
            callbacks.append("copied")
            return b"{}"

    incoming = CallbackBytes((FIXTURES / "problem.json").read_bytes())
    with pytest.raises(TypeError, match="built-in"):
        TcpciProblem.model_validate_json(incoming)
    assert not callbacks
