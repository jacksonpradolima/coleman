"""Tests for deterministic run_id computation."""

from coleman.spec.models import (
    AlgorithmSpec,
    CheckpointSpec,
    ExecutionSpec,
    ExperimentSpec,
    ResultsSpec,
    RunSpec,
    TelemetrySpec,
)
from coleman.spec.run_id import _canonical_json, compute_run_id


class TestCanonicalJson:
    def test_deterministic(self):
        spec = RunSpec()
        assert _canonical_json(spec) == _canonical_json(spec)

    def test_sorted_keys(self):
        spec = RunSpec()
        cj = _canonical_json(spec)
        assert '"algorithm"' in cj
        # Keys should appear in sorted order
        idx_algo = cj.index('"algorithm"')
        idx_exec = cj.index('"execution"')
        assert idx_algo < idx_exec


class TestComputeRunId:
    def test_length(self):
        spec = RunSpec()
        rid = compute_run_id(spec)
        assert len(rid) == 12

    def test_hex_chars(self):
        spec = RunSpec()
        rid = compute_run_id(spec)
        assert all(c in "0123456789abcdef" for c in rid)

    def test_same_spec_same_id(self):
        spec1 = RunSpec()
        spec2 = RunSpec()
        assert compute_run_id(spec1) == compute_run_id(spec2)

    def test_different_spec_different_id(self):
        spec1 = RunSpec()
        spec2 = RunSpec(execution=ExecutionSpec(parallel_pool_size=999))
        assert compute_run_id(spec1) != compute_run_id(spec2)

    def test_algorithm_params_affect_id(self):
        spec1 = RunSpec(algorithm=AlgorithmSpec.model_validate({"ucb": {"timerank": {"c": 0.5}}}))
        spec2 = RunSpec(algorithm=AlgorithmSpec.model_validate({"ucb": {"timerank": {"c": 0.3}}}))
        assert compute_run_id(spec1) != compute_run_id(spec2)

    def test_scalene_default_flag_does_not_change_id(self):
        base = RunSpec(execution=ExecutionSpec(parallel_pool_size=2))
        explicit_default = RunSpec(execution=ExecutionSpec(parallel_pool_size=2, force_sequential_under_scalene=True))
        assert compute_run_id(base) == compute_run_id(explicit_default)

    def test_scalene_override_flag_changes_id(self):
        base = RunSpec(execution=ExecutionSpec(parallel_pool_size=2))
        override = RunSpec(execution=ExecutionSpec(parallel_pool_size=2, force_sequential_under_scalene=False))
        assert compute_run_id(base) != compute_run_id(override)

    def test_default_hooks_and_extensions_do_not_change_id(self):
        base = RunSpec()
        explicit_defaults = RunSpec.model_validate({"hooks": {"fail_fast": True, "plugins": []}, "extensions": {}})
        assert compute_run_id(base) == compute_run_id(explicit_defaults)

    def test_non_default_hooks_and_extensions_change_id(self):
        base = RunSpec()
        with_hooks = RunSpec.model_validate(
            {
                "hooks": {"fail_fast": False, "plugins": ["my_project.hooks.ForecastHook"]},
                "extensions": {"my_domain": {"k": "v"}},
            }
        )
        assert compute_run_id(base) != compute_run_id(with_hooks)

    def test_golden_determinism(self):
        """Same spec must always produce the same run_id (golden test).

        If this fails, the determinism contract is broken.
        """
        spec = RunSpec(
            execution=ExecutionSpec(parallel_pool_size=1, independent_executions=1, verbose=False),
            experiment=ExperimentSpec(datasets=["test@proj"], rewards=["RNFail"], policies=["UCB"]),
            algorithm=AlgorithmSpec.model_validate({"ucb": {"rnfail": {"c": 0.3}}}),
            results=ResultsSpec(enabled=False),
            checkpoint=CheckpointSpec(enabled=False),
            telemetry=TelemetrySpec(enabled=False),
        )
        # This pre-computed value must never change.
        golden_rid = "33a651e6197b"
        assert compute_run_id(spec) == golden_rid
        # Two independent constructions must match.
        spec_copy = RunSpec.model_validate(spec.model_dump())
        assert compute_run_id(spec_copy) == golden_rid
