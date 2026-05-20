"""Tests for the Pydantic v2 spec models."""

import json

import pytest
from pydantic import ValidationError

from coleman.spec.models import (
    AlgorithmSpec,
    CheckpointSpec,
    ExecutionSpec,
    ExperimentSpec,
    ResultsSpec,
    RunSpec,
    TelemetrySpec,
)


class TestExecutionSpec:
    def test_defaults(self):
        spec = ExecutionSpec()
        assert spec.parallel_pool_size == 10
        assert spec.independent_executions == 10
        assert spec.verbose is False
        assert spec.force_sequential_under_scalene is True

    def test_custom_values(self):
        spec = ExecutionSpec(
            parallel_pool_size=4,
            independent_executions=5,
            verbose=True,
            force_sequential_under_scalene=False,
        )
        assert spec.parallel_pool_size == 4
        assert spec.independent_executions == 5
        assert spec.verbose is True
        assert spec.force_sequential_under_scalene is False

    def test_from_dict(self):
        spec = ExecutionSpec.model_validate({"parallel_pool_size": 2})
        assert spec.parallel_pool_size == 2
        assert spec.independent_executions == 10  # default


class TestExperimentSpec:
    def test_defaults(self):
        spec = ExperimentSpec()
        assert spec.scheduled_time_ratio == [0.1, 0.5, 0.8]
        assert spec.datasets_dir == "examples"
        assert spec.datasets == ["alibaba@druid"]
        assert "Random" in spec.policies
        assert "RNFail" in spec.rewards

    def test_custom_policies(self):
        spec = ExperimentSpec(policies=["LinUCB"], rewards=["TimeRank"])
        assert spec.policies == ["LinUCB"]
        assert spec.rewards == ["TimeRank"]


class TestAlgorithmSpec:
    def test_extra_allowed(self):
        spec = AlgorithmSpec.model_validate({"frrmab": {"window_sizes": [100]}})
        assert spec.model_dump()["frrmab"] == {"window_sizes": [100]}

    def test_empty(self):
        spec = AlgorithmSpec()
        assert spec.model_dump() == {}


class TestResultsSpec:
    def test_defaults(self):
        spec = ResultsSpec()
        assert spec.enabled is True
        assert spec.sink == "parquet"
        assert spec.out_dir == "./runs"
        assert spec.batch_size == 1000


class TestCheckpointSpec:
    def test_defaults(self):
        spec = CheckpointSpec()
        assert spec.enabled is True
        assert spec.interval == 50000
        assert spec.base_dir == "checkpoints"


class TestTelemetrySpec:
    def test_defaults(self):
        spec = TelemetrySpec()
        assert spec.enabled is False
        assert spec.service_name == "coleman"
        assert spec.resource_attributes == {}

    def test_resource_attributes(self):
        spec = TelemetrySpec(resource_attributes={"run_id": "abc123", "execution_id": "test|exp=1"})
        assert spec.resource_attributes == {"run_id": "abc123", "execution_id": "test|exp=1"}

    def test_resource_attributes_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            TelemetrySpec.model_validate({"unknown_field": "bad"})


class TestRunSpec:
    def test_defaults(self):
        spec = RunSpec()
        assert spec.execution.parallel_pool_size == 10
        assert spec.experiment.datasets_dir == "examples"
        assert spec.results.sink == "parquet"
        assert spec.telemetry.enabled is False
        assert spec.checkpoint.enabled is True

    def test_from_dict(self, tmp_path):
        out_dir = str(tmp_path / "test_runs")
        data = {
            "execution": {"parallel_pool_size": 2, "verbose": True},
            "experiment": {"datasets": ["org@proj"]},
            "results": {"sink": "parquet", "out_dir": out_dir},
            "telemetry": {"enabled": True},
        }
        spec = RunSpec.model_validate(data)
        assert spec.execution.parallel_pool_size == 2
        assert spec.execution.verbose is True
        assert spec.experiment.datasets == ["org@proj"]
        assert spec.results.out_dir == out_dir
        assert spec.telemetry.enabled is True

    def test_roundtrip_json(self):
        spec = RunSpec()
        data = spec.model_dump()
        spec2 = RunSpec.model_validate(data)
        assert spec == spec2

    def test_model_dump_json_is_valid(self):
        spec = RunSpec()
        raw = spec.model_dump_json()
        parsed = json.loads(raw)
        spec2 = RunSpec.model_validate(parsed)
        assert spec == spec2

    def test_algorithm_freeform(self):
        spec = RunSpec(algorithm=AlgorithmSpec.model_validate({"ucb": {"timerank": {"c": 0.5}}}))
        dumped = spec.algorithm.model_dump()
        assert dumped["ucb"]["timerank"]["c"] == pytest.approx(0.5)

    def test_invalid_type_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RunSpec.model_validate({"execution": {"parallel_pool_size": "not_an_int"}})

    def test_extra_fields_rejected_on_execution(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="extra_forbidden"):
            # Intentional typo: "indepedent" instead of "independent"
            ExecutionSpec.model_validate({"indepedent_executions": 5})  # noqa: RUF001

    def test_extra_fields_rejected_on_runspec(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="extra_forbidden"):
            RunSpec.model_validate({"unknown_section": {}})

    def test_extra_fields_allowed_on_algorithm(self):
        spec = AlgorithmSpec.model_validate({"custom_key": 42})
        assert spec.model_dump()["custom_key"] == 42
