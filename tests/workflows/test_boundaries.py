"""Storage, resource, import and exact-byte registration boundaries."""

import ast
import io
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from coleman.application.baseline_lock import BaselineBusyError, baseline_lock
from coleman.application.workflow_registry import inspect_history, owned_fixture
from coleman.application.workflow_store import WorkflowStore
from coleman.application.workflows import WorkflowService
from coleman.workflows.contracts import PlanningRequest, WorkflowLimits
from coleman.workflows.lifecycle import FilePin, HostPolicy


def fixture_bytes(tmp_path):
    manifest = owned_fixture(tmp_path)
    return (tmp_path / "datasets" / f"{manifest.sha256[7:]}.parquet").read_bytes()


@pytest.mark.parametrize(
    "column,value",
    [
        ("Duration", None),
        ("Duration", -1.0),
        ("Duration", float("nan")),
        ("Duration", float("inf")),
        ("Duration", 1e-100),
        ("Duration", 1e100),
        ("Name", ""),
        ("Verdict", 2),
    ],
)
def test_source_rejects_silent_semantic_changes(tmp_path, column, value):
    frame = pl.read_parquet(fixture_bytes(tmp_path)).with_columns(pl.lit(value).alias(column))
    output = io.BytesIO()
    frame.write_parquet(output)
    encoded_history = output.getvalue()
    with pytest.raises(ValueError):
        inspect_history(encoded_history)


def test_zero_duration_is_observed_and_exact_bytes_stay_bound(tmp_path):
    frame = pl.read_parquet(fixture_bytes(tmp_path)).with_columns(pl.lit(0.0).alias("Duration"))
    output = io.BytesIO()
    frame.write_parquet(output)
    assert inspect_history(output.getvalue())["Duration"].sum() == 0
    service = WorkflowService(tmp_path)
    manifest = owned_fixture(tmp_path)
    first = service.register(manifest)
    second = service.register(manifest)
    assert first == manifest
    assert second == first
    changed = manifest.model_copy(update={"assumptions": ("different source declaration",)})
    with pytest.raises(ValueError, match="replaced"):
        service.register(changed)
    source = tmp_path / "datasets" / f"{manifest.sha256[7:]}.parquet"
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="manifest"):
        service.register(manifest)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "nested/../file", "a\\b", "a//b", "a/./b"])
def test_unsafe_evidence_names_reject(name):
    with pytest.raises(ValueError):
        FilePin(name=name, sha256="sha256:" + "a" * 64, size_bytes=0)


def test_store_reopen_policy_identity_and_corrupt_versions(tmp_path):
    store = WorkflowStore(tmp_path)
    assert WorkflowStore(tmp_path).policy == store.policy
    changed_policy = HostPolicy(schema_version=1, max_jobs=1)
    with pytest.raises(ValueError, match="replaced"):
        WorkflowStore(tmp_path, changed_policy)

    def interrupted_transaction():
        """Fail after a real write so the transaction must roll it back."""
        with store.transaction() as connection:
            connection.execute("INSERT INTO settings VALUES ('rolled-back','value')")
            raise RuntimeError("disk-independent rollback boundary")

    with pytest.raises(RuntimeError):
        interrupted_transaction()
    with store.transaction() as connection:
        assert connection.execute("SELECT 1 FROM settings WHERE name='rolled-back'").fetchone() is None
        connection.execute("DROP TABLE imports")
    with pytest.raises(ValueError, match="schema"):
        WorkflowStore(tmp_path)


def test_unversioned_foreign_database_is_not_migrated(tmp_path):
    with sqlite3.connect(tmp_path / "workspace.sqlite3") as connection:
        connection.execute("CREATE TABLE foreign_data (value TEXT)")
    with pytest.raises(ValueError, match="schema"):
        WorkflowStore(tmp_path)


def test_host_limits_and_missing_owned_inputs(tmp_path):
    service = WorkflowService(tmp_path, policy=HostPolicy(schema_version=1, max_plans=1, max_datasets=1))
    unregistered_request = PlanningRequest(schema_version=1, dataset_id="unregistered")
    with pytest.raises(KeyError):
        service.plan(unregistered_request)
    service.register(owned_fixture(tmp_path))
    over_cap_request = PlanningRequest(
        schema_version=1, dataset_id="owned-ci-three-build-v1", limits=WorkflowLimits(attempt_seconds=301.0)
    )
    with pytest.raises(ValueError, match="host cap"):
        service.plan(over_cap_request)
    request = PlanningRequest(schema_version=1, dataset_id="owned-ci-three-build-v1")
    first = service.plan(request)
    assert service.plan(request) == first
    another_seed_request = request.model_copy(update={"seeds": (0,)})
    with pytest.raises(ValueError, match="slots"):
        service.plan(another_seed_request)
    with pytest.raises(ValueError, match="offset"):
        service.inspect("missing", offset=True)
    with pytest.raises(KeyError):
        service.status("missing")


def test_lock_handoff_retains_real_child_ownership(tmp_path):
    path = tmp_path / "execution.lock"
    child = None
    try:
        with baseline_lock(path, handoff=True) as descriptor:
            child = subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
                pass_fds=(descriptor,),
                stdin=subprocess.PIPE,
            )
        with pytest.raises(BaselineBusyError), baseline_lock(path):
            pytest.fail("Inherited live ownership must not be acquired by another coordinator")
    finally:
        if child is not None:
            child.communicate(timeout=10)
    with baseline_lock(path):
        pass


def test_pure_workflow_imports_have_no_transports_or_model_sdks():
    root = Path(__file__).parents[2] / "coleman" / "workflows"
    forbidden = ("coleman.cli", "coleman.transports", "mcp", "openai", "anthropic")
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(name == prefix or name.startswith(prefix + ".") for name in names for prefix in forbidden)
    program = """
import sys
import coleman.workflows
import coleman.workflows.repetitions
forbidden = ("coleman.cli", "coleman.transports", "mcp", "openai", "anthropic")
assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden)
"""
    subprocess.run([sys.executable, "-c", program], check=True)


def test_registration_cli_and_versioned_discovery(tmp_path):
    command = [sys.executable, "-m", "coleman.cli", "workflow", "--workspace", str(tmp_path)]
    setup = subprocess.run(command + ["init-fixture"], capture_output=True, text=True, check=True)
    assert json.loads(setup.stdout)["rows"] == 9
    discovery = subprocess.run(command + ["discover"], capture_output=True, text=True, check=True)
    result = json.loads(discovery.stdout)
    assert result["contract"] == "coleman-workflows-v1"
    assert result["datasets"][0]["duration_unit"] == "s"
    assert result["planning_schema"]["required"] == ["schema_version", "dataset_id"]
