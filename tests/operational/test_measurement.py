"""Independent complete-grid and immutable publication refusal boundaries."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.selected_runtime import SelectedInstalledRuntime


def hand_report(module):
    # Hand-authored zero-duration diagnostic fixture, never performance evidence.
    runtime = InstalledRuntime.from_json((Path(__file__).parents[1] / "spec/fixtures/runtime-v1.json").read_bytes())
    selected = SelectedInstalledRuntime.model_validate(
        {
            **runtime.model_dump(),
            "schema_version": 2,
            "inventory_profile": "selected-extras-v1",
            "selected_extras": ("telemetry",),
            "distributions": (
                {"name": "coleman", "version": "0.0.0"},
                {"name": "opentelemetry-api", "version": "unit-fixture"},
                {"name": "opentelemetry-sdk", "version": "unit-fixture"},
                {"name": "opentelemetry-exporter-otlp-proto-http", "version": "unit-fixture"},
            ),
        }
    )
    rows = tuple(
        module.Observation(
            schema_version=1,
            experts=2,
            panels=2,
            trajectories=1,
            mode=mode,
            elapsed_seconds=0.0,
            cpu_seconds=0.0,
            peak_rss_mib=None,
            current_rss_mib=None,
            parent_elapsed_seconds=0.0,
            accepted_updates=2,
            updates_per_second=None,
            mathematical_output="sha256:" + "2" * 64,
            snapshot_bytes=0,
        )
        for mode in ("off", "noop", "memory")
    )
    return module.Report(
        schema_version=1,
        complete=True,
        profile="quick",
        runtime=runtime,
        instrumentation_runtime=selected,
        hardware="authored-unit-fixture",
        native_threads=(),
        driver_sha256="0" * 64,
        example_sha256="1" * 64,
        measurements=rows,
    )


@pytest.mark.parametrize(
    "damage", ("missing", "duplicate", "changed_math", "wrong_case", "future", "wrong_sdk", "wrong_base_version")
)
def test_no_complete_report_for_missing_cells_changed_math_or_unsupported_version(measurement, damage):
    report = hand_report(measurement)
    values = report.model_dump()
    if damage == "missing":
        values["measurements"] = values["measurements"][:-1]
    elif damage == "duplicate":
        values["measurements"] = values["measurements"][:1] * 3
    elif damage == "changed_math":
        values["measurements"][1]["mathematical_output"] = "sha256:" + "3" * 64
    elif damage == "wrong_case":
        values["measurements"][1]["experts"] = 3
    elif damage == "future":
        values["schema_version"] = 2
    elif damage == "wrong_sdk":
        values["instrumentation_runtime"]["selected_extras"] = ("ci",)
    else:
        values["instrumentation_runtime"]["distributions"][0]["version"] = "0.0.1"
    with pytest.raises(ValueError):
        measurement.Report.from_json(json.dumps(values))


def test_report_round_trip_old_runtime_reader_and_concurrent_identical_publication(measurement, tmp_path):
    report = hand_report(measurement)
    assert measurement.Report.from_json(report.to_json()).to_json() == report.to_json()
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _: measurement.publish(report, tmp_path), range(4)))
    assert len(set(paths)) == 1
    assert measurement.Report.from_json(paths[0].read_bytes()).to_json() == report.to_json()
    assert json.loads(paths[0].read_text())["complete"] is True


def test_interrupted_real_immutable_writer_cannot_publish_false_completion(measurement, tmp_path, monkeypatch):
    report = hand_report(measurement)

    def interruption(*_args, **_kwargs):
        raise InterruptedError("controlled interruption before final atomic publication")

    with monkeypatch.context() as patch:
        patch.setattr("coleman.artifacts.os.link", interruption)
        with pytest.raises(InterruptedError):
            measurement.publish(report, tmp_path)
    assert list(tmp_path.glob("*.json")) == []
    final = measurement.publish(report, tmp_path)
    assert final.is_file()


def test_unchecked_report_rejects_before_any_publication(measurement, tmp_path):
    damaged = hand_report(measurement).model_copy(update={"measurements": ()})
    with pytest.raises(ValueError):
        measurement.publish(damaged, tmp_path)
    assert not list(tmp_path.iterdir())
