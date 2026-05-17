"""Tests for the CLI module."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import yaml

from coleman.cli import _cmd_sweep, _cmd_validate, _parse_kv, main


class TestParseKv:
    def test_simple_values(self):
        key, vals = _parse_kv("alpha=0.1,0.3,0.5")
        assert key == "alpha"
        assert vals == [0.1, 0.3, 0.5]

    def test_integer_values(self):
        key, vals = _parse_kv("seed=1,2,3")
        assert key == "seed"
        assert vals == [1, 2, 3]

    def test_range_expression(self):
        key, vals = _parse_kv("seed=range(0,5)")
        assert key == "seed"
        assert vals == [0, 1, 2, 3, 4]

    def test_range_with_step(self):
        key, vals = _parse_kv("seed=range(0,10,2)")
        assert key == "seed"
        assert vals == [0, 2, 4, 6, 8]

    def test_string_values(self):
        key, vals = _parse_kv("policy=Random,Greedy")
        assert key == "policy"
        assert vals == ["Random", "Greedy"]

    def test_missing_equals_raises(self):
        """Lines 41-42: ValueError when format has no '='."""
        import pytest

        with pytest.raises(ValueError, match="Expected key=values format"):
            _parse_kv("no_equals_here")


class TestCLIValidate:
    def test_validate_valid_config(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump({"execution": {"verbose": True}}, fh)
            path = fh.name
        try:
            main(["validate", "--config", path])
            captured = capsys.readouterr()
            assert "VALID" in captured.out
            assert "run_id=" in captured.out
        finally:
            os.unlink(path)

    def test_validate_with_resolve(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump({"execution": {"parallel_pool_size": 3}}, fh)
            path = fh.name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = os.path.join(tmpdir, "resolved.json")
                main(["validate", "--config", path, "--resolve", out_path])
                assert os.path.exists(out_path)
        finally:
            os.unlink(path)


class TestCLIRun:
    def test_run_command(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "run.yaml")
            with open(cfg_path, "w") as fh:
                yaml.dump(
                    {
                        "execution": {
                            "parallel_pool_size": 1,
                            "independent_executions": 1,
                            "verbose": False,
                        },
                        "experiment": {
                            "scheduled_time_ratio": [0.1],
                            "datasets_dir": "examples",
                            "datasets": ["fakedata"],
                            "rewards": ["RNFail"],
                            "policies": ["Random"],
                        },
                        "results": {"out_dir": tmpdir},
                    },
                    fh,
                )
            main(["run", "--config", cfg_path])
            captured = capsys.readouterr()
            assert "run_id:" in captured.out


class TestCLISweep:
    def test_sweep_dry_run(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "base.yaml")
            with open(cfg_path, "w") as fh:
                yaml.dump(
                    {
                        "execution": {
                            "parallel_pool_size": 1,
                            "independent_executions": 1,
                            "verbose": False,
                        },
                        "experiment": {
                            "scheduled_time_ratio": [0.1],
                            "datasets_dir": "examples",
                            "datasets": ["fakedata"],
                            "rewards": ["RNFail"],
                            "policies": ["Random"],
                        },
                        "results": {"out_dir": tmpdir},
                    },
                    fh,
                )
            main(
                [
                    "sweep",
                    "--config",
                    cfg_path,
                    "--grid",
                    "execution.parallel_pool_size=1,2,4",
                    "--dry-run",
                ]
            )
            captured = capsys.readouterr()
            assert "Generated 3 specs" in captured.out
            assert "run_id=" in captured.out

    def test_sweep_multiple_grid_flags(self, capsys):
        """Multiple --grid flags accumulate all sweep dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "base.yaml")
            with open(cfg_path, "w") as fh:
                yaml.dump(
                    {
                        "execution": {
                            "parallel_pool_size": 1,
                            "independent_executions": 1,
                        },
                        "experiment": {
                            "scheduled_time_ratio": [0.1],
                            "datasets_dir": "examples",
                            "datasets": ["fakedata"],
                            "rewards": ["RNFail"],
                            "policies": ["Random"],
                        },
                        "results": {"out_dir": tmpdir},
                    },
                    fh,
                )
            main(
                [
                    "sweep",
                    "--config",
                    cfg_path,
                    "--grid",
                    "execution.seed=1,2",
                    "--grid",
                    "execution.parallel_pool_size=1,2",
                    "--dry-run",
                ]
            )
            captured = capsys.readouterr()
            # 2 seeds × 2 pool sizes = 4 specs
            assert "Generated 4 specs" in captured.out

    def test_sweep_non_dry_run_prints_run_ids(self, capsys):
        """Lines 105-107: non-dry-run sweep prints run_ids."""
        from coleman.api import RunResult

        fake_result = MagicMock(spec=RunResult)
        fake_result.run_id = "abc123"

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "base.yaml")
            with open(cfg_path, "w") as fh:
                yaml.dump(
                    {
                        "execution": {"parallel_pool_size": 1, "independent_executions": 1},
                        "experiment": {
                            "scheduled_time_ratio": [0.1],
                            "datasets_dir": "examples",
                            "datasets": ["fakedata"],
                            "rewards": ["RNFail"],
                            "policies": ["Random"],
                        },
                        "results": {"out_dir": tmpdir},
                    },
                    fh,
                )
            with patch("coleman.cli.run_many", return_value=[fake_result]):
                main(["sweep", "--config", cfg_path, "--grid", "execution.seed=1,2"])
            captured = capsys.readouterr()
            assert "run_id=abc123" in captured.out


class TestCLIRunArtifacts:
    def test_run_prints_artifacts_dir_when_set(self, capsys, tmp_path):
        """Lines 95-98: prints artifact dir when artifacts_dir is non-empty."""
        from coleman.api import RunResult
        from coleman.spec.models import RunSpec

        fake_spec = RunSpec()
        artifacts_dir = str(tmp_path / "artifacts")
        fake_result = RunResult(run_id="test-run-id", spec=fake_spec, artifacts_dir=artifacts_dir)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump({}, fh)
            path = fh.name
        try:
            with patch("coleman.cli.run", return_value=fake_result):
                main(["run", "--config", path])
            captured = capsys.readouterr()
            assert "run_id: test-run-id" in captured.out
            assert f"artifacts: {artifacts_dir}" in captured.out
        finally:
            os.unlink(path)


class TestCLIValidateResolveMessage:
    def test_validate_with_resolve_prints_message(self, capsys):
        """Line 156: prints 'Resolved spec written to' when --resolve is given."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump({}, fh)
            path = fh.name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = os.path.join(tmpdir, "resolved.json")
                main(["validate", "--config", path, "--resolve", out_path])
                captured = capsys.readouterr()
                assert "Resolved spec written to" in captured.out
        finally:
            os.unlink(path)


class TestCLIPrivateHandlers:
    def test_cmd_sweep_non_dry_run_direct_handler(self, capsys):
        """Cover direct _cmd_sweep result-print loop lines 105-107."""
        import argparse

        args = argparse.Namespace(
            config="unused.yaml",
            packs_dir=None,
            grid=[["execution.seed=1,2"]],
            dry_run=False,
            workers=2,
        )

        fake_spec = MagicMock()
        fake_spec.execution.parallel_pool_size = 1
        fake_result = MagicMock()
        fake_result.run_id = "rid-1"

        with (
            patch("coleman.cli.load_spec", return_value=fake_spec),
            patch("coleman.cli.sweep", return_value=[fake_spec]),
            patch("coleman.cli.run_many", return_value=[fake_result]),
        ):
            _cmd_sweep(args)

        captured = capsys.readouterr()
        assert "run_id=rid-1" in captured.out

    def test_cmd_validate_resolve_direct_handler(self, capsys, tmp_path):
        """Cover direct _cmd_validate resolve print line 156."""
        import argparse

        out_path = tmp_path / "resolved.json"
        args = argparse.Namespace(config="unused.yaml", packs_dir=None, resolve=str(out_path))
        fake_spec = MagicMock()

        with (
            patch("coleman.cli.load_spec", return_value=fake_spec),
            patch("coleman.cli.compute_run_id", return_value="rid-x"),
            patch("coleman.cli.save_resolved", return_value=out_path),
        ):
            _cmd_validate(args)

        captured = capsys.readouterr()
        assert "Resolved spec written to" in captured.out


class TestCLIErrorAndMainBlock:
    def test_validate_invalid_config_exits(self):
        """Cover _cmd_validate error path lines 105-107."""
        import argparse

        args = argparse.Namespace(config="bad.yaml", packs_dir=None, resolve=None)
        with patch("coleman.cli.load_spec", side_effect=ValueError("bad spec")):
            import pytest

            with pytest.raises(SystemExit) as exc:
                _cmd_validate(args)
        assert exc.value.code == 1

    def test_cli_module_main_block(self):
        """Cover module __main__ guard line 156 via runpy."""
        import runpy
        import sys

        argv = sys.argv[:]
        try:
            sys.argv = ["coleman", "--help"]
            with patch("sys.exit", side_effect=SystemExit):
                import pytest

                with pytest.raises(SystemExit):
                    runpy.run_module("coleman.cli", run_name="__main__")
        finally:
            sys.argv = argv
