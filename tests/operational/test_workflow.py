"""Execute the actual checked-in CI reporting snippets under controlled files."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
UPLOADER = "codecov/codecov-action@57e3a136b779b570ffcdbf80b3bdc90e7fab3de2"


def steps(job="test"):
    return yaml.safe_load((ROOT / ".github/workflows/codecov.yml").read_text())["jobs"][job]["steps"]


def snippet(name, job="test"):
    step = next(s for s in steps(job) if s.get("name") == name)
    body = step["run"].split("python - <<'PY'\n", 1)[1]
    return body.rsplit("\nPY", 1)[0]


@pytest.mark.parametrize("percentage,accepted", ((94.94, False), (94.999999999, False), (95.0, True), (95.01, True)))
def test_raw_gate_and_display_apply_the_same_unchanged95_policy(tmp_path, percentage, accepted):
    (tmp_path / "coverage-final.json").write_text(json.dumps({"totals": {"percent_covered": percentage}}))
    (tmp_path / "pyproject.toml").write_text("[tool.coverage.report]\nfail_under=95\nprecision=8\n")
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", snippet("Complete installed corpus tests and enforce combined coverage")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert (result.returncode == 0) is accepted
    assert f"Raw combined coverage: {percentage!r}%" in result.stdout
    assert f"Unrounded coverage policy: {'PASS' if accepted else 'FAIL'}" in result.stdout
    assert "fail_under=95; precision=8" in result.stdout


@pytest.mark.parametrize("present", ((), ("junit.xml",), ("junit.xml", "junit-installed-corpus.xml")))
def test_real_report_snippet_preserves_partial_failed_run_and_missing_phase_reporting(tmp_path, present):
    for name in present:
        (tmp_path / name).write_text('<testsuite tests="1" failures="1"/>')
    output = tmp_path / "github-output"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", snippet("Report retained JUnit files")],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == "files=" + ",".join(present) + "\n"
    for expected in ("junit.xml", "junit-installed-corpus.xml"):
        assert (f"::warning::Missing JUnit report: {expected}" in result.stdout) is (expected not in present)


def test_current_supported_schema_pins_and_original_inventory_resources_are_preserved():
    current = steps()
    junit = next(s for s in current if s.get("name") == "Upload test results to Codecov")
    coverage = next(s for s in current if s.get("name") == "Upload coverage to Codecov")
    reporting = next(s for s in current if s.get("id") == "junit_reports")
    assert coverage["uses"] == junit["uses"] == UPLOADER
    assert junit["with"]["report_type"] == "test_results"
    assert junit["with"]["disable_search"] is True
    assert "!cancelled()" in reporting["if"] and "!cancelled()" in junit["if"]
    assert "outputs.files != ''" in junit["if"]
    parallel = next(s["run"] for s in current if s.get("name") == "Run tests")
    assert "-n 4 --dist=loadfile" in parallel
    assert parallel.count("--ignore=") == 2
    serial = next(s["run"] for s in current if s.get("name", "").startswith("Complete installed corpus"))
    for path in ("tests/corpus/test_installed_public_ci.py", "tests/corpus/test_installed_westermo.py"):
        assert path in parallel and path in serial
    assert "--cov-append --cov-fail-under=95" in serial
    policy = yaml.safe_load((ROOT / ".github/workflows/codecov.yml").read_text())
    assert policy["permissions"] == {"contents": "read"}
    assert policy["jobs"]["test"]["timeout-minutes"] == 120


def test_real_ci_probe_reuses_uploader_and_preserves_normal_required_check_identity():
    policy = yaml.safe_load((ROOT / ".github/workflows/codecov.yml").read_text())
    normal, probe = policy["jobs"]["test"], policy["jobs"]["uploader_qualification"]
    assert "github.event_name != 'push'" in normal["if"]
    assert "format('Run {0} tests and collect coverage', github.event_name)" in normal["name"]
    assert "product suite intentionally not executed" in normal["name"]
    for name in ("success", "failure"):
        reference = "refs/heads/qualification/v5-372-uploader-" + name
        assert reference in normal["if"] and reference in probe["if"]
    assert "github.event_name == 'push'" in probe["if"]
    assert probe["timeout-minutes"] == 5
    assert probe["steps"][-1] is normal["steps"][-1]
    assert probe["steps"][6] is normal["steps"][-2]
    assert "qualification/v5-372-uploader-failure" in probe["steps"][-1]["with"]["fail_ci_if_error"]
    assert all(not step.get("continue-on-error") for step in probe["steps"])


@pytest.mark.parametrize("expectation", ("success", "failure"))
def test_probe_runs_actual_pytest_and_reports_missing_serial_after_real_failure(tmp_path, expectation):
    environment = {
        **os.environ,
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_ENV": str(tmp_path / "env"),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "GITHUB_SHA": "unit-source-fixture",
        "COLEMAN_PROBE_EXPECTATION": expectation,
        "COLEMAN_PROBE_PYTHON": sys.executable,
    }
    prepare = snippet("Prepare a real tiny pytest case outside product collection", "uploader_qualification")
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", prepare],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    key, value = (tmp_path / "env").read_text().strip().split("=", 1)
    environment[key] = value
    source = steps("uploader_qualification")
    primary = next(s["run"] for s in source if s.get("id") == "probe_primary")
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", primary],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == (1 if expectation == "failure" else 0), result.stdout + result.stderr
    environment["COLEMAN_PRIMARY_OUTCOME"] = "failure" if result.returncode else "success"
    if result.returncode == 0:
        secondary = next(
            s["run"] for s in source if s.get("name") == "Run real second phase only after successful primary"
        )
        result = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", secondary],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    reporter = snippet("Report retained JUnit files", "uploader_qualification")
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", reporter],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert ((tmp_path / "junit-installed-corpus.xml").exists()) is (expectation == "success")
    validation = snippet("Record actual probe reports and expected outcome", "uploader_qualification")
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", validation],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["product_suite_executed"] is False
    assert receipt["expected_negative"] is (expectation == "failure")
    # This qualifies real local JUnit production/reporting only. A network
    # upload is separate actual-CI evidence; it is never simulated here.
