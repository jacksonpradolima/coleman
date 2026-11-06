"""Contract, crash/replay and integration regressions for domain result artifacts."""

from __future__ import annotations

import ast
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from coleman.artifacts import ArtifactWriter
from coleman.manifest import generate_manifest
from coleman.results.domain import (
    RESERVED_METRICS,
    DomainRecord,
    Measurement,
    fingerprint,
)
from coleman.results.domain_artifacts import DomainArtifacts
from coleman.results.duckdb_sink import DuckDBSink
from coleman.results.parquet_sink import ParquetSink
from coleman.results.sink_base import NullSink
from coleman.results.writer import ResultsWriter

FIXTURES = Path(__file__).parents[1] / "fixtures" / "domain-results"
ROOT = Path(__file__).parents[2]


def example_record():
    return DomainRecord.from_json((FIXTURES / "record-v1.json").read_bytes())


def altered_record(**updates):
    payload = json.loads(example_record().to_json())
    payload.update(updates)
    return DomainRecord.from_json(json.dumps(payload))


@pytest.mark.parametrize("sink_cls", [DuckDBSink, ParquetSink])
def test_round_trip_replay_query_and_conflict(tmp_path, sink_cls):
    record = example_record()
    sink = sink_cls(out_dir=str(tmp_path))
    try:
        sink.write_domain_record(record)
        sink.flush()
        sink.write_domain_record(DomainRecord.from_json(record.to_json()))
        result = sink.read_domain_record(*record.logical_key[1:], namespace=record.namespace)
        assert result == record
        assert len(list(sink.domain_records())) == 1
        assert result.measurement("net_value").value == -2.0
        assert result.learning_reward.value == 0.25
        conflict = altered_record(available_at="2026-01-03T12:00:00Z")
        with pytest.raises(ValueError, match="Conflicting"):
            sink.write_domain_record(conflict)
    finally:
        sink.close()
    reopened = sink_cls(out_dir=str(tmp_path))
    try:
        reopened.write_domain_record(record)
        assert list(reopened.domain_records()) == [record]
    finally:
        reopened.close()


def test_versioned_golden_bytes_digests_and_roundtrip():
    record = example_record()
    assert record.to_json() + "\n" == (FIXTURES / "record-v1.json").read_text()
    expected = json.loads((FIXTURES / "digests-v1.json").read_text())
    assert record.digest == expected["content"]
    assert record.identity_digest == expected["identity"]
    assert DomainRecord.from_json(record.to_json()) == record


@pytest.mark.parametrize("version", [0, 2, 999, "1", True, None])
def test_unknown_versions_fail_closed(version):
    with pytest.raises(ValidationError):
        altered_record(schema_version=version)


def test_omitted_version_is_not_assumed_legacy_v1():
    payload = json.loads(example_record().to_json())
    del payload["schema_version"]
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError):
        DomainRecord.from_json(serialized)


@pytest.mark.parametrize("field", ["data_fingerprint", "available_at", "provenance", "measurements"])
def test_required_provenance_is_not_invented(field):
    payload = json.loads(example_record().to_json())
    del payload[field]
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError):
        DomainRecord.from_json(serialized)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "3.0"])
def test_nonfinite_and_coerced_values_rejected(value):
    payload = json.loads(example_record().to_json())
    payload["measurements"][0]["value"] = value
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError):
        DomainRecord.from_json(serialized)
    payload = json.loads(example_record().to_json())
    payload["learning_reward"]["value"] = value
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError):
        DomainRecord.from_json(serialized)


def test_zero_null_unavailable_not_applicable_and_omitted():
    base = {"name": "outcome", "unit": "credit", "meaning": "net-v1", "direction": "maximize"}
    zero = Measurement.model_validate({**base, "value": 0.0, "status": "observed"})
    negative = Measurement.model_validate({**base, "value": -2.0, "status": "observed"})
    assert zero.value == 0.0
    assert negative.value == -2.0
    values = [
        Measurement.model_validate({**base, "value": None, "status": status})
        for status in ("unknown", "unavailable", "not_applicable")
    ]
    assert len({item.model_dump_json() for item in values}) == 3
    with pytest.raises(ValidationError):
        Measurement.model_validate({**base, "status": "unknown"})
    for status, value in (("observed", None), ("unknown", 0.0), ("unavailable", -2.0)):
        with pytest.raises(ValidationError):
            Measurement.model_validate({**base, "value": value, "status": status})
    below_minimum = {**base, "value": -2.0, "status": "observed", "minimum": 0.0}
    with pytest.raises(ValidationError, match="minimum"):
        Measurement.model_validate(below_minimum)
    record = example_record()
    with pytest.raises(KeyError):
        record.measurement("omitted")


def test_optional_omission_and_explicit_null_remain_distinct():
    payload = json.loads(example_record().to_json())
    del payload["learning_reward"]
    absent = DomainRecord.from_json(json.dumps(payload))
    explicit = altered_record(learning_reward=None)
    assert "learning_reward" not in json.loads(absent.to_json())
    assert json.loads(explicit.to_json())["learning_reward"] is None
    assert absent.digest != explicit.digest
    assert absent.identity_digest == explicit.identity_digest


@pytest.mark.parametrize("name", sorted(RESERVED_METRICS))
def test_native_columns_cannot_be_overwritten(name):
    with pytest.raises(ValidationError, match="Reserved"):
        Measurement(name=name, value=0.0, unit="1", meaning="fake-v1", direction="maximize", status="observed")


@pytest.mark.parametrize("namespace", ["coleman", "coleman.results"])
def test_native_namespace_reserved(namespace):
    with pytest.raises(ValidationError, match="Reserved"):
        altered_record(namespace=namespace)


@pytest.mark.parametrize(
    "change",
    [
        {"unit": "ms"},
        {"meaning": "other-v2"},
        {"direction": "maximize"},
        {"minimum": -10.0},
    ],
)
def test_incompatible_measurements_rejected(change):
    original = example_record()
    payload = json.loads(original.to_json())
    next(item for item in payload["measurements"] if item["name"] == "duration").update(change)
    other = DomainRecord.from_json(json.dumps(payload))
    with pytest.raises(ValueError, match="Incompatible"):
        original.require_comparable(other, "duration")


def test_comparison_requires_shared_dataset_evaluation_namespace_and_observation():
    original = example_record()
    original.require_comparable(altered_record(run_id="other-run"), "duration")
    for updates in ({"data_fingerprint": fingerprint([])}, {"namespace": "other"}):
        incompatible = altered_record(**updates)
        with pytest.raises(ValueError, match="Incompatible"):
            original.require_comparable(incompatible, "duration")
    payload = json.loads(original.to_json())
    payload["provenance"]["evaluation_fingerprint"] = fingerprint({"other": True})
    incompatible = DomainRecord.from_json(json.dumps(payload))
    with pytest.raises(ValueError, match="Incompatible"):
        original.require_comparable(incompatible, "duration")
    with pytest.raises(ValueError, match="observed"):
        original.require_comparable(original, "unknown_cost")


def test_metamorphic_declaration_order_timezone_and_json_key_order():
    original = example_record()
    payload = json.loads(original.to_json())
    for order in itertools.permutations(payload["measurements"]):
        payload["measurements"] = list(order)
        payload["available_at"] = "2026-01-02T09:00:00-03:00"
        record = DomainRecord.from_json(json.dumps(dict(reversed(list(payload.items())))))
        assert record.to_json() == original.to_json()
        assert record.digest == original.digest
    assert fingerprint({"b": [0, None, -2], "a": "ação"}) == fingerprint({"a": "ação", "b": [0, None, -2]})
    assert fingerprint([1, 2]) != fingerprint([2, 1])


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-01-02T12:00:00",
        "not-a-time",
        1767355200,
        "1767355200",
        "1767355200000",
        "1767355200.0",
        "-1767355200",
        "+1767355200",
        "1.7673552e9",
        "20260102",
    ],
)
def test_naive_or_invalid_timestamp_rejected(timestamp):
    with pytest.raises(ValidationError):
        altered_record(available_at=timestamp)


@pytest.mark.parametrize("payload", [{1: "bad"}, {"x": object()}, [float("nan")], (1, 2)])
def test_fingerprints_reject_ambiguous_or_nonjson_inputs(payload):
    with pytest.raises((ValueError, TypeError)):
        fingerprint(payload)


def test_duplicate_names_keys_unknown_fields_and_mutation_rejected(tmp_path):
    original = example_record()
    payload = json.loads(original.to_json())
    payload["measurements"].append(payload["measurements"][0])
    serialized = json.dumps(payload)
    with pytest.raises(ValidationError, match="unique"):
        DomainRecord.from_json(serialized)
    with pytest.raises(ValidationError):
        altered_record(measurements=[])
    with pytest.raises(ValidationError):
        altered_record(uncontracted={})
    with pytest.raises(ValueError, match="Duplicate"):
        DomainRecord.from_json('{"schema_version":1,"schema_version":2}')
    with pytest.raises(ValueError, match="object"):
        DomainRecord.from_json("[]")
    with pytest.raises(ValidationError):
        original.run_id = "mutated"
    with pytest.raises(ValidationError):
        original.measurements[0].value = 100.0
    unchecked = original.model_copy(update={"schema_version": 42})
    artifacts = DomainArtifacts(tmp_path)
    with pytest.raises(ValidationError):
        artifacts.write(unchecked)


@pytest.mark.parametrize("sink_cls", [DuckDBSink, ParquetSink])
def test_legacy_fixture_retains_native_output(tmp_path, sink_cls):
    legacy = json.loads((FIXTURES / "native-v3.2.json").read_text())
    sink = sink_cls(out_dir=str(tmp_path), top_k=3)
    sink.write_row(legacy)
    sink.close()
    if sink_cls is DuckDBSink:
        import duckdb

        connection = duckdb.connect(str(tmp_path / "results.duckdb"))
        before = connection.execute("SELECT * FROM coleman_results").fetchall()
        columns = [column[0] for column in connection.description]
        assert before[0][columns.index("fitness")] == legacy["fitness"]
        assert before[0][columns.index("cost")] == legacy["cost"]
        connection.close()
    else:
        artifact = next(tmp_path.rglob("*.parquet"))
        before = artifact.read_bytes()
        row = pq.read_table(artifact).to_pylist()[0]
        assert row["fitness"] == legacy["fitness"]
        assert row["cost"] == legacy["cost"]
    reopened = sink_cls(out_dir=str(tmp_path))
    assert list(reopened.domain_records()) == []
    assert reopened.read_domain_record("old", "execution", "decision", namespace="ci") is None
    reopened.write_domain_record(example_record())
    reopened.close()
    if sink_cls is DuckDBSink:
        connection = duckdb.connect(str(tmp_path / "results.duckdb"))
        assert connection.execute("SELECT * FROM coleman_results").fetchall() == before
        connection.close()
    else:
        assert artifact.read_bytes() == before


def _process_write(out_dir, payload):
    sink = ParquetSink(out_dir=out_dir)
    sink.write_domain_record(DomainRecord.from_json(payload))
    sink.close()


def test_concurrent_process_replay_uses_one_logical_artifact(tmp_path):
    record = example_record()
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_process_write, str(tmp_path), record.to_json()) for _ in range(9)]
        for future in futures:
            future.result(timeout=60)
    assert list(DomainArtifacts(tmp_path).records()) == [record]
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_concurrent_conflicts_never_overwrite(tmp_path):
    records = [example_record(), altered_record(available_at="2026-01-03T12:00:00Z")]

    def commit(record):
        try:
            DomainArtifacts(tmp_path).write(record)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(commit, records))
    assert sorted(results) == [False, True]
    assert list(DomainArtifacts(tmp_path).records())[0] == records[results.index(True)]


@pytest.mark.parametrize("stage", ["before", "after"])
def test_actual_process_exit_before_after_artifact_commit(tmp_path, stage):
    script = """
import os, sys
from coleman.results.domain import DomainRecord
from coleman.results.domain_artifacts import DomainArtifacts
record = DomainRecord.from_json(sys.argv[1])
original = os.link
def interrupt(source, destination):
    if sys.argv[3] == "after":
        original(source, destination)
    os._exit(71)
os.link = interrupt
DomainArtifacts(sys.argv[2]).write(record)
"""
    record = example_record()
    result = subprocess.run([sys.executable, "-c", script, record.to_json(), str(tmp_path), stage], check=False)
    assert result.returncode == 71
    artifacts = DomainArtifacts(tmp_path)
    assert list(artifacts.records()) == ([record] if stage == "after" else [])
    assert list(tmp_path.rglob("*.tmp"))  # orphan scratch files are ignored
    manifest = json.loads(generate_manifest(tmp_path, run_id=record.run_id).read_text())
    assert len([item for item in manifest["artifacts"] if item["type"] == "domain_records"]) == (stage == "after")
    artifacts.write(record)
    assert list(artifacts.records()) == [record]


def test_io_error_does_not_publish_and_synchronous_writer_errors_propagate(tmp_path, monkeypatch):
    def fail(*_args):
        raise OSError("disk error")

    monkeypatch.setattr(os, "link", fail)
    sink = ParquetSink(out_dir=str(tmp_path))
    writer = ResultsWriter(sink)
    record = example_record()
    with pytest.raises(OSError, match="disk error"):
        writer.write_domain_record(record)
    assert list(sink.domain_records()) == []
    assert list(tmp_path.rglob("*.tmp")) == []
    writer.stop()
    sink.close()


def test_immutable_writer_replay_and_conflict(tmp_path):
    writer = ArtifactWriter(tmp_path)
    path = tmp_path / "record.json"
    writer.write_immutable(path, "{}\n")
    writer.write_immutable(path, "{}\n")
    with pytest.raises(ValueError, match="Conflicting"):
        writer.write_immutable(path, "[]\n")
    assert path.read_text() == "{}\n"


def test_unknown_or_miskeyed_persisted_artifact_fails_without_migration(tmp_path):
    artifacts = DomainArtifacts(tmp_path)
    record = example_record()
    artifacts.write(record)
    path = next(artifacts.root.glob("*.json"))
    payload = json.loads(path.read_text())
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload))
    for read in (
        lambda: list(artifacts.records()),
        lambda: artifacts.read(*record.logical_key[1:], namespace=record.namespace),
    ):
        with pytest.raises(ValidationError):
            read()
    payload["schema_version"] = 1
    payload["decision_id"] = "wrong-key"
    path.write_text(json.dumps(payload))
    for read in (
        lambda: list(artifacts.records()),
        lambda: artifacts.read(*record.logical_key[1:], namespace=record.namespace),
    ):
        with pytest.raises(ValueError, match="logical key"):
            read()


def test_unsupported_sink_capabilities_fail_explicitly():
    sink = NullSink()
    for operation in (
        lambda: sink.write_domain_record(example_record()),
        lambda: sink.read_domain_record("r", "e", "d", namespace="ci"),
        lambda: sink.domain_records(),
    ):
        with pytest.raises(NotImplementedError):
            operation()


def test_import_boundary_no_cli_mcp_or_runner_dependency():
    for module in ("domain", "domain_artifacts", "sink_base"):
        tree = ast.parse((ROOT / "coleman" / "results" / f"{module}.py").read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert not any(name and any(part in name.lower() for part in ("cli", "mcp", "runner")) for name in imports)
    script = (
        "from coleman.results.domain import DomainRecord; import sys; "
        "assert 'coleman.cli' not in sys.modules; assert 'coleman.runner' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_real_tcpci_checkpoint_resume_and_replay_ahead_of_checkpoint(tmp_path):
    import runpy

    from coleman.environment import Environment
    from coleman.evaluation import NAPFDVerdictMetric
    from coleman.policy import GreedyPolicy
    from coleman.results.domain import ResultProvenance
    from coleman.reward import RNFailReward
    from coleman.runner import create_agents, get_scenario_provider

    example = runpy.run_path(str(ROOT / "examples" / "domain-results" / "tcpci.py"))
    monitor_cls = example["ObservationMonitor"]

    class InterruptedEnvironment(Environment):
        def save_periodically(self, restore, t, experiment, bandit, interval=None):
            if t == 2:
                raise RuntimeError("interrupted before checkpoint 2")
            return super().save_periodically(restore, t, experiment, bandit, interval)

    def build_environment(environment_cls):
        provider = get_scenario_provider(
            datasets_dir=str(ROOT / "examples"),
            dataset="fakedata",
            sched_time_ratio=0.5,
            use_hcs=False,
            use_context=False,
            context_config={},
            feature_groups={},
        )
        env = environment_cls(
            create_agents(GreedyPolicy(), RNFailReward(), [], seed=42),
            provider,
            NAPFDVerdictMetric(),
            results_config={"enabled": True, "sink": "parquet", "out_dir": str(tmp_path / "results")},
            checkpoint_config={"enabled": True, "base_dir": str(tmp_path / "checkpoints"), "interval": 1},
            runtime_metadata={"execution_id": "physical-attempt-1"},
        )
        env.monitor = monitor_cls(
            env.monitor.sink,
            "checkpoint-example",
            ResultProvenance(
                producer="tcpci-checkpoint-test",
                producer_version="1",
                spec_fingerprint=fingerprint({"scenario": "fakedata", "policy": "Greedy", "seed": 42}),
                evaluation_fingerprint=fingerprint({"evaluation": "napfd-verdict-v1", "ratio": 0.5}),
            ),
            fingerprint({"fixture": "public-fakedata-v3.2"}),
        )
        return env

    interrupted = build_environment(InterruptedEnvironment)
    with pytest.raises(RuntimeError, match="interrupted"):
        interrupted.run_single(experiment=1, trials=3, forecast_next=False)
    interrupted.monitor.close()
    committed = list(interrupted.monitor.sink.domain_records())
    assert len(committed) == 2
    checkpoint = interrupted.checkpoint_store.load(str(interrupted.scenario_provider), 1)
    assert checkpoint is not None
    assert checkpoint.step == 1

    resumed = build_environment(Environment)
    resumed.set_runtime_metadata({"execution_id": "physical-attempt-2"})
    resumed.run_single(experiment=1, trials=3, forecast_next=False)
    resumed.monitor.close()
    records = list(resumed.monitor.sink.domain_records())
    assert len(records) == 3
    assert set(record.digest for record in committed) <= {record.digest for record in records}
    assert len({record.logical_key for record in records}) == 3
    checkpoint = resumed.checkpoint_store.load(str(resumed.scenario_provider), 1)
    assert checkpoint is not None
    assert checkpoint.step == 3


def test_installed_wheel_outside_repository_ci_capability(tmp_path):
    """Exercise complete examples through an isolated, non-editable base install."""
    import shutil

    wheel_dir = tmp_path / "wheel"
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(wheel_dir)], cwd=ROOT, check=True)
    wheel = next(wheel_dir.glob("*.whl"))
    env_dir = tmp_path / "installed"
    subprocess.run(["uv", "venv", str(env_dir), "--python", sys.executable], check=True)
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel) + "[ci]"], check=True)
    external = tmp_path / "external"
    external.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for example in ("tcpci", "allocation"):
        script = external / f"{example}.py"
        shutil.copy(ROOT / "examples" / "domain-results" / f"{example}.py", script)
        result = subprocess.run(
            [str(python), str(script), str(external / example)],
            cwd=external,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if example == "tcpci":
            assert json.loads(result.stdout.splitlines()[-1])["spec_traced"] is True
        else:
            record = DomainRecord.from_json(result.stdout)
            assert record.measurement("net_value").value == -2.0
    forbidden = subprocess.run(
        [str(python), str(external / "allocation.py"), "../escaped-example"],
        cwd=external,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert forbidden.returncode != 0
    assert "within the current working directory" in forbidden.stderr
    assert not (tmp_path / "escaped-example").exists()
    parallel_root = external / "tcpci-parallel"
    parallel = subprocess.run(
        [str(python), str(external / "tcpci.py"), str(parallel_root), "2", "2"],
        cwd=external,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    summary = json.loads(parallel.stdout.splitlines()[-1])
    assert summary["records"] == 6
    assert summary["spec_traced"] is True
    records = list(DomainArtifacts(parallel_root / "runs" / summary["run_id"] / "results").records())
    assert len({record.logical_key for record in records}) == 6
    assert len({record.execution_id for record in records}) == 2
    script = (
        "import coleman, sys; from coleman.results.domain import DomainRecord; "
        "assert 'coleman.cli' not in sys.modules; assert 'coleman.runner' not in sys.modules; "
        "assert 'clickhouse_connect' not in sys.modules; "
        f"assert coleman.__file__.startswith({str(env_dir)!r})"
    )
    subprocess.run([str(python), "-c", script], cwd=external, env=environment, check=True)


def test_existing_parquet_artifact_remains_queryable_and_byte_identical(tmp_path):
    import shutil

    from coleman.results.duckdb_catalog import DuckDBCatalog

    artifact = tmp_path / "native-v3.2.parquet"
    shutil.copy(FIXTURES / artifact.name, artifact)
    before = artifact.read_bytes()
    legacy = json.loads((FIXTURES / "native-v3.2.json").read_text())
    with_native = ParquetSink(out_dir=str(tmp_path))
    with_native.write_domain_record(example_record())
    with_native.close()
    catalog = DuckDBCatalog(str(tmp_path))
    try:
        result = catalog.query("SELECT fitness, cost, rewards, ttf_duration FROM results").row(0, named=True)
        assert result == {key: legacy[key] for key in result}
    finally:
        catalog.close()
    assert artifact.read_bytes() == before


def test_domain_failure_prevents_checkpoint_advancing(tmp_path, monkeypatch):
    import runpy

    from coleman.environment import Environment
    from coleman.evaluation import NAPFDVerdictMetric
    from coleman.policy import GreedyPolicy
    from coleman.results.domain import ResultProvenance
    from coleman.reward import RNFailReward
    from coleman.runner import create_agents, get_scenario_provider

    provider = get_scenario_provider(
        datasets_dir=str(ROOT / "examples"),
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
    )
    env = Environment(
        create_agents(GreedyPolicy(), RNFailReward(), [], seed=42),
        provider,
        NAPFDVerdictMetric(),
        results_config={"enabled": True, "out_dir": str(tmp_path / "results")},
        checkpoint_config={"enabled": True, "base_dir": str(tmp_path / "checkpoints"), "interval": 1},
    )
    monitor_cls = runpy.run_path(str(ROOT / "examples" / "domain-results" / "tcpci.py"))["ObservationMonitor"]
    env.monitor = monitor_cls(
        env.monitor.sink,
        "r",
        ResultProvenance(
            producer="ci",
            producer_version="1",
            spec_fingerprint=fingerprint({}),
            evaluation_fingerprint=fingerprint({}),
        ),
        fingerprint([]),
    )

    def fail(*_args):
        raise OSError("interrupted domain write")

    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(OSError, match="interrupted domain write"):
        env.run_single(experiment=1, trials=2, forecast_next=False)
    assert env.checkpoint_store.load(str(provider), 1) is None
    assert list(env.monitor.sink.domain_records()) == []
    assert env.monitor.rows_collected == 0
    env.monitor.close()


@pytest.mark.parametrize("field,status", [("minimum", "observed"), ("value", "unknown")])
def test_unchecked_nonfinite_instances_rejected_before_json_coercion(tmp_path, field, status):
    record = example_record()
    measurement = record.measurement("duration").model_copy(update={field: float("nan"), "status": status})
    unchecked = record.model_copy(update={"measurements": (measurement,)})
    with pytest.raises(ValidationError):
        unchecked.to_json()
    artifacts = DomainArtifacts(tmp_path)
    with pytest.raises(ValidationError):
        artifacts.write(unchecked)
    assert list(DomainArtifacts(tmp_path).records()) == []


@pytest.mark.parametrize(
    "base_name", ["../escaped", "/tmp/escaped", r"..\escaped", "C:escaped", "", ".", "..", "bad\0name"]
)
def test_duckdb_basename_cannot_escape_artifact_directory(tmp_path, base_name):
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="base_name"):
        DuckDBSink(out_dir=str(output), base_name=base_name)
    assert not output.exists()
    assert not list(tmp_path.rglob("*.duckdb"))


def test_duckdb_supports_unicode_basename_with_domain_records(tmp_path):
    sink = DuckDBSink(out_dir=str(tmp_path), base_name="resultados-ação")
    try:
        sink.write_domain_record(example_record())
        assert list(sink.domain_records()) == [example_record()]
    finally:
        sink.close()
    assert (tmp_path / "resultados-ação.duckdb").exists()
