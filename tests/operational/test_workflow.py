"""Execute the actual checked-in CI reporting snippets under controlled files."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
UPLOADER = "codecov/codecov-action@fb8b3582c8e4def4969c97caa2f19720cb33a72f"
COMPOSITE = "./.github/actions/report-junit"


def steps(job="test"):
    filename = "uploader-qualification.yml" if job == "uploader_qualification" else "codecov.yml"
    return yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())["jobs"][job]["steps"]


def action():
    return yaml.safe_load((ROOT / ".github/actions/report-junit/action.yml").read_text())


def snippet(name, job="test"):
    source = action()["runs"]["steps"] if name == "Report retained JUnit files" else steps(job)
    step = next(s for s in source if s.get("name") == name)
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
    shared = action()
    junit = next(s for s in shared["runs"]["steps"] if s.get("name") == "Upload test results to Codecov")
    coverage = next(s for s in current if s.get("name") == "Upload coverage to Codecov")
    reporting = next(s for s in shared["runs"]["steps"] if s.get("id") == "junit_reports")
    caller = next(s for s in current if s.get("uses") == COMPOSITE)
    assert coverage["uses"] == junit["uses"] == UPLOADER
    assert junit["with"]["report_type"] == "test_results"
    assert junit["with"]["disable_search"] is True
    assert "!cancelled()" in reporting["if"]
    assert "!cancelled()" in junit["if"]
    assert "!cancelled()" in caller["if"]
    assert "outputs.files != ''" in junit["if"]
    assert reporting["shell"] == "bash"
    assert shared["runs"]["using"] == "composite"
    assert shared["outputs"]["files"]["value"] == "${{ steps.junit_reports.outputs.files }}"
    assert junit["with"]["token"] == "${{ inputs.token }}"
    assert junit["with"]["fail_ci_if_error"] is True
    assert coverage["with"]["fail_ci_if_error"] is True
    assert coverage["with"]["files"] == "coverage.xml"
    assert coverage["with"]["disable_search"] is True
    assert not junit["with"].get("skip_validation", False)
    assert not coverage["with"].get("skip_validation", False)
    assert "fail-ci-if-error" not in shared["inputs"]
    assert not junit.get("continue-on-error", False)
    assert caller["continue-on-error"] is True
    assert coverage["continue-on-error"] is True
    assert caller["with"]["token"] == "${{ secrets.CODECOV_TOKEN }}"
    assert "fail-ci-if-error" not in caller["with"]
    parallel = next(s["run"] for s in current if s.get("name") == "Run tests")
    assert "-n 4 --dist=loadfile" in parallel
    assert parallel.count("--ignore=") == 2
    serial = next(s["run"] for s in current if s.get("name", "").startswith("Complete installed corpus"))
    for path in ("tests/corpus/test_installed_public_ci.py", "tests/corpus/test_installed_westermo.py"):
        assert path in parallel
        assert path in serial
    assert "--cov-append --cov-fail-under=95" in serial
    for step in current:
        if step.get("name") == "Run tests" or step.get("name", "").startswith("Complete installed corpus"):
            assert not step.get("continue-on-error", False)
    policy = yaml.safe_load((ROOT / ".github/workflows/codecov.yml").read_text())
    assert policy["permissions"] == {"contents": "read"}
    assert policy["jobs"]["test"]["timeout-minutes"] == 120


def test_real_ci_probe_reuses_uploader_and_preserves_normal_required_check_identity():
    policy = yaml.safe_load((ROOT / ".github/workflows/codecov.yml").read_text())
    qualification = yaml.safe_load((ROOT / ".github/workflows/uploader-qualification.yml").read_text())
    normal, probe = policy["jobs"]["test"], qualification["jobs"]["uploader_qualification"]
    references = ["qualification/v5-372-uploader-" + name for name in ("success", "failure")]
    assert "if" not in normal
    assert normal["name"] == "Run ${{ github.event_name }} tests and collect coverage"
    assert set(policy["jobs"]) == {"test"}
    assert policy[True]["push"] == {"branches-ignore": references, "tags": ["**"]}
    assert "pull_request" in policy[True]
    assert policy[True]["pull_request"] is None
    assert qualification[True] == {"push": {"branches": references}}
    assert qualification["permissions"] == {"contents": "read"}
    assert probe["timeout-minutes"] == 5
    assert probe["steps"][-1]["uses"] == normal["steps"][-1]["uses"] == COMPOSITE
    assert "fail-ci-if-error" not in probe["steps"][-1]["with"]
    assert "uploader-qualification-" in probe["steps"][-1]["with"]["upload-name"]
    assert "!cancelled()" in probe["steps"][-1]["if"]
    assert all(not step.get("continue-on-error") for step in probe["steps"])
    checkout = next(step for step in normal["steps"] if step.get("name") == "Checkout")
    assert "ref" not in checkout.get("with", {})


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
