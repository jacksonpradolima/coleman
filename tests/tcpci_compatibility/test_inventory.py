"""Compatibility declarations must match real retained APIs, artifacts and profiles."""

from __future__ import annotations

import ast
import hashlib
import re
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from coleman.application.contracts import Discovery
from coleman.application.executor_capture_types import CAPTURE_PROFILE
from coleman.core.primitives import decode_json
from coleman.native.semantics import native_catalog
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.selected_runtime import SelectedInstalledRuntime, read_runtime, require_native_runtime

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs/assets/tcpci-compatibility-v1.json"


def inventory() -> dict[str, Any]:
    """Read the static documentation artifact with the existing strict decoder."""
    value = decode_json(INVENTORY.read_bytes())
    assert isinstance(value, dict)
    return value


def source(path: str) -> ast.Module:
    """Inspect declarations without importing optional native or transport code."""
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


def literal(path: str, name: str) -> Any:
    """Read one explicit literal declaration from its real module."""
    node = next(
        item
        for item in source(path).body
        if isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in item.targets)
    )
    return ast.literal_eval(node.value)


def tools(path: str, owner: str) -> list[str]:
    """Read only tool-decorated functions owned by the selected real server."""
    function = next(item for item in source(path).body if isinstance(item, ast.FunctionDef) and item.name == owner)
    return [
        item.name
        for item in function.body
        if isinstance(item, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "tool"
            for decorator in item.decorator_list
        )
    ]


def test_inventory_retains_all_38_routes_with_existing_local_evidence():
    data = inventory()
    assert data["schema_version"] == 1
    assert data["artifact_type"] == "tcpci_compatibility_inventory"
    assert data["baselines"]["v4"]["source_commit"] == "1edbee874129aa85e26e592d9649826e85e1ac6b"
    assert data["baselines"]["entry"]["source_commit"] == "3ad8e9e94d658e910adaad4a85ba7395e5ad7abc"
    expected = [f"P{index:02}" for index in range(1, 16)] + [f"E{index:02}" for index in range(1, 24)]
    assert [row["id"] for row in data["rows"]] == expected
    for row in data["rows"]:
        assert row["disposition"] in {"retained", "shimmed", "read-only", "breaking"}
        assert row["surface"] and row["limits"]
        assert row["profiles"]
        assert set(row["profiles"]) <= data["profiles"].keys()
        assert row["source_paths"] and row["test_paths"]
        for name in row["source_paths"] + row["test_paths"]:
            relative = PurePosixPath(name)
            assert not relative.is_absolute()
            assert ".." not in relative.parts
            assert (ROOT / name).is_file()
        assert set(row["frozen_evidence"]) <= data["frozen_evidence"].keys()
    human = (ROOT / "docs/v5-compatibility.md").read_text(encoding="utf-8")
    assert re.findall(r"^\| ([PE]\d\d) \|", human, re.MULTILINE) == expected


def test_recorded_entry_exports_still_match_real_public_namespaces():
    for path, names in inventory()["entry_exports"].items():
        assert literal(path, "__all__") == names
        assert len(names) == len(set(names))
    spec_names = inventory()["entry_exports"]["coleman/spec/__init__.py"]
    assert {"RunSpec", "load_spec", "save_resolved", "expand_sweep", "resolve_packs"} <= set(spec_names)


def test_old_mcp_catalogues_and_console_routes_are_separate_from_tasks():
    data = inventory()
    catalogues = data["tool_catalogues"]
    for catalogue in catalogues.values():
        assert tools(catalogue["source"], catalogue["owner"]) == catalogue["names"]
    assert len(catalogues["application"]["names"]) == 7
    assert len(catalogues["workflow_additions"]["names"]) == 14
    assert len(catalogues["separate_tasks"]["names"]) == 9
    assert not set(catalogues["application"]["names"]) & set(catalogues["workflow_additions"]["names"])
    server = next(
        item
        for item in source("coleman/transports/mcp.py").body
        if isinstance(item, ast.FunctionDef) and item.name == "create_server"
    )
    assert [argument.arg for argument in server.args.kwonlyargs] == ["caller", "workflows"]
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    assert all(scripts[name] == target for name, target in data["entry_scripts"].items())


def test_legacy_inventory_matches_real_catalog_and_retained_factory_outcomes():
    expected = inventory()["native_inventory"]
    actual = native_catalog()
    assert [row.name for row in actual.policies] == expected["policies"]
    assert [row.name for row in actual.rewards] == expected["rewards"]
    assert len(actual.policies) == 37
    assert len(actual.rewards) == 6
    assert sum(len(row.routes) for row in actual.policies) == expected["agent_routes"] == 41
    dispatch = decode_json((ROOT / "tests/fixtures/native-semantics/v1/legacy-dispatch.json").read_bytes())
    completed = [row["name"] for row in dispatch["policies"] if row["factory_outcome"] == "completed"]
    failed = sorted(row["name"] for row in dispatch["policies"] if row["factory_outcome"] == "failed")
    assert len(completed) == 34
    assert failed == expected["factory_failures"] == ["SWContextualEpsilonGreedy", "SWLinTS", "SlMAB"]
    assert expected["numeric_scope"] == dispatch["comparison_scope"]
    assert expected["single_trace_unqualified"] == ["FRRMAB"]
    preservation = decode_json((ROOT / "tests/fixtures/native-semantics/v1/source-preservation.json").read_bytes())
    assert sum(row["kind"] == "python" for row in preservation["resources"]) == expected["source_mechanisms"] == 46
    assert sum(row["kind"] == "marker" for row in preservation["resources"]) == expected["exact_markers"] == 11


@pytest.mark.parametrize("identifier", sorted(inventory()["frozen_evidence"]))
def test_original_golden_bytes_remain_pinned_without_rewriting(identifier):
    evidence = inventory()["frozen_evidence"][identifier]
    payload = (ROOT / evidence["path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == evidence["sha256"]


def test_measured_numeric_reference_is_bound_to_original_source_and_selected_scope():
    data = inventory()
    entry = data["native_numeric_reference"]
    evidence = data["frozen_evidence"][entry["frozen_evidence"]]
    reference = decode_json((ROOT / evidence["path"]).read_bytes())
    assert reference["contract"] == "coleman-native-numeric-compatibility-v1"
    assert reference["profile"] == entry["profile"] == "legacy-native-numeric-v1"
    assert reference["reference_source_revision"] == entry["reference_source_revision"]
    assert entry["reference_source_revision"] == data["baselines"]["v4"]["source_commit"]
    assert reference["reference_wheel_sha256"] == entry["reference_wheel_sha256"]
    assert reference["reference_SDK_resources"] == entry["reference_SDK_resources"] == 358
    input_bytes = (ROOT / entry["input_fixture"]).read_bytes()
    assert hashlib.sha256(input_bytes).hexdigest() == entry["input_sha256"] == reference["input_sha256"]
    cases = reference["cases"]
    assert len(cases) == entry["cases"] == 8
    assert sum(len(case["rows"]) for case in cases) == entry["rows"] == 32
    assert {(case["policy"], case["budget_mode"], case["budget_value"]) for case in cases} == {
        (policy, mode, value)
        for policy in ("Random", "UCB")
        for mode, value in (("ratio", 0.5), ("ratio", 1), ("fixed_time", 3), ("subset_size", 2))
    }
    for case in cases:
        assert case["api_metrics_empty"] is True
        assert len(case["projected_fields"]) == entry["selected_field_count"] == 22
        assert "wall_time_seconds" in case["excluded_fields"]
        assert "process_memory_peak_rss_mib" in case["excluded_fields"]
    assert entry["limits"] == reference["comparison"]


def test_runtime_reader_generations_never_become_historical_execution_approval():
    data = inventory()
    paths = data["frozen_evidence"]
    old_wire = (ROOT / paths["runtime-v1"]["path"]).read_bytes()
    selected_wire = (ROOT / paths["runtime-v2"]["path"]).read_bytes()
    old = read_runtime(old_wire)
    current = read_runtime(selected_wire)
    assert isinstance(old, InstalledRuntime)
    assert isinstance(current, SelectedInstalledRuntime)
    assert (old.to_json() + "\n").encode() == old_wire
    assert (current.to_json() + "\n").encode() == selected_wire
    assert data["native_runtime_generations"]["historical"]["schema_version"] == old.schema_version == 1
    assert data["native_runtime_generations"]["historical"]["inventory"] == "base-only"
    assert data["native_runtime_generations"]["current_native"]["schema_version"] == current.schema_version == 2
    assert list(current.selected_extras) == data["native_runtime_generations"]["current_native"]["selected_extras"]
    with pytest.raises(ValueError, match="pinned v4 or create a new CI plan"):
        require_native_runtime(old)


def test_ephemeral_application_evidence_is_not_generic_task_application_evidence():
    data = inventory()
    row = next(row for row in data["rows"] if row["id"] == "P11")
    assert row["frozen_evidence"] == ["application-coverage", "application-digests"]
    coverage = data["frozen_evidence"]["application-coverage"]
    assert coverage["path"] == "tests/fixtures/application/coverage-v1.json"
    old = decode_json((ROOT / coverage["path"]).read_bytes())
    task = decode_json((ROOT / "tests/fixtures/tasks/application-v1/discovery.json").read_bytes())
    assert old["kind"] == "coverage"
    assert task["contract"] == "coleman-tasks-v1"
    native_contract = Discovery.model_fields["transport_contract"].default
    assert native_contract == "coleman-application-v1"
    assert native_contract != task["contract"]


def test_optional_remote_backends_remain_interface_only_and_generic_profile_is_explicit():
    data = inventory()
    profiles = data["profiles"]
    assert profiles["clickhouse-interface"]["status"] == "interface-only"
    assert profiles["telemetry-interface"]["status"] == "interface-only"
    assert profiles["pinned-original"]["status"] == "pinned-original"
    generic = profiles["generic-directory-wheel"]
    assert generic["inventory_profile"] == CAPTURE_PROFILE == "directory-wheel-closure-v1"
    assert generic["example_python_flags"] == ["-I", "-B"]
    assert generic["checked_context"]["bytecode_writes"] is False
    assert generic["checked_context"]["optimization"] == 0
    assert generic["checked_context"]["standard_meta_finders"] == ["BuiltinImporter", "FrozenImporter", "PathFinder"]
    inference = next(row for row in data["rows"] if row["id"] == "E19")
    assert inference["profiles"] == ["base-math", "analytics-inference"]
    assert profiles["base-math"]["extras"] == []
    assert profiles["analytics-inference"]["extras"] == ["analytics"]
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert {"numpy", "scipy"} <= {item.split("==")[0] for item in dependencies["analytics"]}
    overlay = next(row for row in data["additive_overlays"] if row["namespace"] == "coleman.problems.tcpci")
    assert literal(overlay["source_paths"][0], "__all__") == overlay["public_exports"]
    assert overlay["guide"] == "tcpci-problems.md"
