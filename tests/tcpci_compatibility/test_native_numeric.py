"""Compare actual installed native execution with measured original v4 output."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from coleman.corpus.baseline_contracts import FAILURE_MEANING, BaselineBuild, BaselineTest
from coleman.corpus.baseline_reference import baseline_reference
from coleman.pilot.contracts import SECONDS
from coleman.results.domain import Measurement
from tests.installed_support import build_install_stdlib_wheel

ROOT = Path(__file__).parents[2]
GOLDEN = ROOT / "tests/fixtures/tcpci-compatibility/v1/native-numeric.json"
INPUTS = ROOT / "tests/fixtures/native-semantics/v1/legacy-ci-inputs.json"
CASES = [
    (policy, mode, value)
    for policy in ("Random", "UCB")
    for mode, value in (("ratio", 0.5), ("ratio", 1.0), ("fixed_time", 3.0), ("subset_size", 2.0))
]


def _digest(rows):
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _fixture():
    assert GOLDEN.stat().st_size < 2 * 1024**2
    data = json.loads(GOLDEN.read_text())
    assert data["schema_version"] == 1
    assert data["contract"] == "coleman-native-numeric-compatibility-v1"
    assert data["reference_source_revision"] == "1edbee874129aa85e26e592d9649826e85e1ac6b"
    assert data["reference_SDK_resources"] == 358
    assert data["input_sha256"] == hashlib.sha256(INPUTS.read_bytes()).hexdigest()
    return data


def _reference_build(step):
    inputs = json.loads(INPUTS.read_text())
    return BaselineBuild(
        step=step,
        member_id=f"authored-native-build-{step}",
        source_metadata=None,
        decided_at=None,
        original_order=None,
        tests=tuple(
            BaselineTest(
                test_id=row["Name"],
                duration=Measurement(
                    name="authored_reported_test_seconds",
                    value=float(row["Duration"]),
                    unit=SECONDS.unit,
                    meaning=SECONDS.meaning,
                    direction="minimize",
                    status="observed",
                    minimum=0,
                ),
                failure=Measurement(
                    name="authored_binary_failure",
                    value=float(row["Verdict"]),
                    unit="1",
                    meaning=FAILURE_MEANING,
                    direction="none",
                    status="observed",
                    minimum=0,
                ),
            )
            for row in inputs["rows"]
            if row["BuildId"] == step
        ),
    )


@pytest.mark.parametrize("policy,mode,value", CASES)
def test_frozen_original_numeric_rows_have_independent_full_universe_metric_evidence(policy, mode, value):
    """Use the existing rational oracle, never the native metric or scheduler."""
    fixture = _fixture()
    case = next(
        c for c in fixture["cases"] if (c["policy"], c["budget_mode"], c["budget_value"]) == (policy, mode, value)
    )
    assert case["projection_sha256"] == _digest(case["rows"])
    assert len(case["rows"]) == 4
    for row in case["rows"]:
        build = _reference_build(row["step"])
        order = tuple(json.loads(row["prioritization_order_top_k"]))
        assert hashlib.sha256(json.dumps(list(order)).encode()).hexdigest() == row["prioritization_order_hash"]
        costs = {test.test_id: test.duration.value for test in build.tests}
        assert all(cost is not None and cost > 0 for cost in costs.values())
        if mode == "ratio":
            seconds = float(row["total_build_duration"]) * value
        elif mode == "fixed_time":
            seconds = value
        else:
            # Only this positive integer-cost fixture uses time as an equivalent
            # reference for count selection. This is not a general conversion.
            seconds = sum(float(costs[name]) for name in order[: int(value)])
        reference = baseline_reference(build, order, seconds)
        assert row["fitness"] == reference.napfd
        assert row["cost"] == reference.apfdc
        assert row["tests_ran"] == len(reference.selected)
        assert row["detected"] == reference.detected
        assert row["missed"] == reference.failures - reference.detected
        assert row["total_build_duration"] == reference.projected_full_work
        if reference.first_failure is None:
            assert row["ttf_duration"] == reference.projected_full_work
        else:
            assert row["ttf_duration"] == reference.first_failure


@pytest.fixture(scope="module")
def installed_native_compatibility(tmp_path_factory):
    """Install the current wheel with its frozen CI closure outside the checkout."""
    owned = tmp_path_factory.mktemp("installed-native-compatibility")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(
        UV_OFFLINE="1",
        UV_NO_INDEX="1",
        UV_LINK_MODE="copy",
        POLARS_MAX_THREADS="1",
        COLEMAN_POLICY_SEED="20261106",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    virtual = owned / "native"
    wheel = build_install_stdlib_wheel(uv, owned / "dist", virtual, environment, extras=("ci",))
    outside = owned / "outside"
    outside.mkdir()
    assert not outside.is_relative_to(ROOT)
    shutil.copyfile(Path(__file__).with_name("native_driver.py"), outside / "native_driver.py")
    shutil.copyfile(INPUTS, outside / "legacy-ci-inputs.json")
    (owned / "installation.json").write_text(
        json.dumps(
            {
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "driver_sha256": hashlib.sha256((outside / "native_driver.py").read_bytes()).hexdigest(),
                "input_sha256": hashlib.sha256((outside / "legacy-ci-inputs.json").read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return virtual / "bin/python", outside, environment


@pytest.mark.parametrize("policy,mode,value", CASES)
def test_installed_current_native_api_matches_original_frozen_numeric_projection(
    installed_native_compatibility, policy, mode, value
):
    """Compare complete per-build values and rank digests without a native mock."""
    python, outside, environment = installed_native_compatibility
    case_name = f"{policy}-{mode}-{value}"
    work = outside / case_name
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-B",
            str(outside / "native_driver.py"),
            str(outside / "legacy-ci-inputs.json"),
            str(work),
            policy,
            mode,
            str(value),
        ],
        cwd=outside,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads((work / "receipt.json").read_text())
    fixture = _fixture()
    expected = next(
        c for c in fixture["cases"] if (c["policy"], c["budget_mode"], c["budget_value"]) == (policy, mode, value)
    )
    assert actual["numeric_versions"] == fixture["numeric_versions"]
    assert actual["input_sha256"] == fixture["input_sha256"]
    assert actual["profile"] == fixture["profile"]
    assert actual["api_metrics_empty"] is True
    assert actual["projected_fields"] == expected["projected_fields"]
    assert actual["rows"] == expected["rows"]
    assert actual["projection_sha256"] == expected["projection_sha256"] == _digest(actual["rows"])
