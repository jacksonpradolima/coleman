"""Additive optional command identity never invents historical provenance."""

import hashlib
import json
from pathlib import Path

import pytest

from coleman.spec.selected_runtime import SelectedInstalledRuntime
from tests.operational.test_load_contracts import hand_task_report
from tests.operational.test_measurement import hand_report


@pytest.fixture(params=("measurement", "bootstrap", "tasks"))
def report_case(request, measurement, bootstrap_measurement, task_measurement):
    """Reuse independent v1 hand fixtures, keeping absent CLI identity explicit."""
    if request.param == "measurement":
        value = hand_report(measurement)
        return value, value.runtime
    if request.param == "tasks":
        value = hand_task_report(task_measurement)
        return value, value.runtime
    source = (Path(__file__).with_name("fixtures") / "bootstrap-v1.json").read_bytes()
    value = bootstrap_measurement.BootstrapReport.from_json(source)
    return value, value.cells[0].runtime


def canonical(values):
    """Use the independent standard-library v1 byte convention for comparison."""
    return json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def driver_for(runtime):
    """Declare a fixture CLI closure; this is reader evidence, not a capture."""
    values = runtime.model_dump(mode="json")
    return {
        **values,
        "schema_version": 2,
        "inventory_profile": "selected-extras-v1",
        "selected_extras": ["cli"],
        "distributions": sorted(
            [*values["distributions"], {"name": "click", "version": "8.1.8"}], key=lambda item: item["name"]
        ),
    }


def test_v1_absent_driver_identity_preserves_exact_legacy_bytes_and_digest(report_case):
    """Missing provenance stays unavailable; neither reading nor writing fits it."""
    report, _ = report_case
    legacy_values = report.model_dump(mode="json")
    assert "driver_runtime" not in legacy_values
    legacy = canonical(legacy_values)
    restored = type(report).from_json(legacy)
    assert restored.driver_runtime is None
    assert restored.to_json() == legacy
    assert restored.fingerprint == "sha256:" + hashlib.sha256(legacy.encode()).hexdigest()
    # Explicit unavailable optional metadata also canonicalizes to the old wire.
    legacy_values["driver_runtime"] = None
    assert type(report).model_validate(legacy_values).to_json() == legacy


def test_existing_bootstrap_golden_retains_independent_v1_canonical_bytes(bootstrap_measurement):
    """A pre-CLI fixture checks old keys/units and canonical bytes independently."""
    source = json.loads((Path(__file__).with_name("fixtures") / "bootstrap-v1.json").read_text())
    expected = canonical(source)
    report = bootstrap_measurement.BootstrapReport.from_json(expected)
    assert report.to_json() == expected
    assert report.driver_runtime is None


def test_selected_cli_identity_round_trips_without_changing_base_or_telemetry(report_case):
    """The recorded tooling closure is separate from measured SDK dependencies."""
    report, runtime = report_case
    values = report.model_dump(mode="json")
    driver = driver_for(runtime)
    values["driver_runtime"] = driver
    restored = type(report).from_json(canonical(values))
    assert restored.driver_runtime == SelectedInstalledRuntime.model_validate(driver)
    assert type(report).from_json(restored.to_json()).to_json() == restored.to_json()
    assert restored.driver_runtime.selected_extras == ("cli",)
    assert (
        report.to_json()
        == type(report).model_validate({**restored.model_dump(mode="json"), "driver_runtime": None}).to_json()
    )


@pytest.mark.parametrize("damage", ("future", "extras", "missing_click", "click_version", "package", "python", "base"))
def test_driver_identity_rejects_incompatible_versions_and_wrong_base_binding(report_case, damage):
    """One purported runtime cannot contain conflicting base/driver versions."""
    report, runtime = report_case
    values = report.model_dump(mode="json")
    driver = driver_for(runtime)
    if damage == "future":
        driver["schema_version"] = 3
    elif damage == "extras":
        driver["selected_extras"] = ["telemetry"]
    elif damage == "missing_click":
        driver["distributions"] = [item for item in driver["distributions"] if item["name"] != "click"]
    elif damage == "click_version":
        next(item for item in driver["distributions"] if item["name"] == "click")["version"] = "0.0.0"
    elif damage == "package":
        driver["package_fingerprint"] = "sha256:" + "f" * 64
    elif damage == "python":
        driver["python_version"] = "3.14.99"
    else:
        next(item for item in driver["distributions"] if item["name"] == "coleman")["version"] = "wrong-base"
    values["driver_runtime"] = driver
    encoded = canonical(values)
    with pytest.raises(ValueError):
        type(report).from_json(encoded)
