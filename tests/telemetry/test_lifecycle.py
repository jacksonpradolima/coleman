"""Independent metric arithmetic, strict wire semantics and finite dimensions."""

import hashlib
import json
from datetime import UTC, datetime
from itertools import permutations
from pathlib import Path

import pytest

from coleman.telemetry.export import NoOpLifecycleTelemetry
from coleman.telemetry.lifecycle import LifecycleSnapshot, MetricPoint, metric_catalog, point
from coleman.telemetry.resources import ResourceSnapshot
from coleman.telemetry.views import resource_points

GOLDEN = Path(__file__).with_name("fixtures") / "lifecycle-v1.json"


def sample():
    return LifecycleSnapshot.from_json(GOLDEN.read_bytes())


def test_hand_authored_versioned_golden_round_trip_and_independent_digest():
    # 0.75**2 +0.25**2 =0.625, no production concentration helper involved.
    expected = json.dumps(json.loads(GOLDEN.read_text()), sort_keys=True, separators=(",", ":"))
    current = sample()
    assert current.to_json() == expected
    assert current.fingerprint == "sha256:" + hashlib.sha256(expected.encode()).hexdigest()
    assert LifecycleSnapshot.from_json(current.to_json()) == current


@pytest.mark.parametrize("version", (0, 2, 999, True, 1.0, "1"))
def test_unknown_or_coerced_versions_reject(version):
    document = json.loads(GOLDEN.read_text())
    document["schema_version"] = version
    with pytest.raises(ValueError):
        LifecycleSnapshot.from_json(json.dumps(document))


@pytest.mark.parametrize("value", (float("inf"), float("-inf"), float("nan"), -1, True, "0"))
def test_nonfinite_negative_and_coerced_operational_values_reject(value):
    with pytest.raises(ValueError):
        point("pending", value)


@pytest.mark.parametrize("name,value", (("pending", 0.5), ("task_progress", 1.1), ("concentration", 1.01)))
def test_count_and_ratio_envelopes_are_not_clipped(name, value):
    with pytest.raises(ValueError):
        point(name, value)


def test_zero_unknown_unavailable_and_not_applicable_remain_distinct():
    points = (
        point("rss", 0),
        point("rss", None),
        point("rss", None, missing="unknown"),
        point("rss", None, missing="not_applicable"),
    )
    assert [p.status for p in points] == ["observed", "unavailable", "unknown", "not_applicable"]
    assert [p.value for p in points] == [0, None, None, None]
    assert len({p.model_dump_json() for p in points}) == 4
    with pytest.raises(ValueError):
        MetricPoint(name="rss", unit="s", status="observed", value=0)
    with pytest.raises(ValueError):
        MetricPoint(name="rss", unit="MiB", status="unavailable", value=0)
    with pytest.raises(ValueError):
        point("rss", None, missing="observed")


def test_catalog_is_closed_and_every_metric_has_owner_units_and_response():
    definitions = metric_catalog()
    assert len(definitions) == 19
    assert [d.name for d in definitions] == sorted(d.name for d in definitions)
    assert all(d.unit and d.interpretation and d.owner and d.response for d in definitions)
    for spec in definitions:
        assert point(spec.name, 0).unit == spec.unit
    assert point("cpu_utilization", 250).value == 250
    assert point("output_usage", 1.25).value == 1.25


def test_lifetime_counts_preserve_int64_beyond_binary64_exact_integer_range():
    for value in (2**53 + 1, 2**63 - 1):
        current = point("updates", value)
        assert type(current.value) is int
        assert current.value == value
        assert MetricPoint.model_validate_json(current.model_dump_json()).value == value
    with pytest.raises(ValueError):
        point("updates", 2**63)
    with pytest.raises(ValueError):
        point("updates", 0.0)
    with pytest.raises(ValueError):
        MetricPoint.model_validate_json('{"name":"updates","unit":"1","status":"observed","value":9007199254740993.0}')


def test_permutations_preserve_exact_identity_and_duplicate_coordinates_reject():
    current = sample()
    for ordering in permutations(current.points):
        changed = LifecycleSnapshot.model_validate({**current.model_dump(), "points": ordering})
        assert changed.to_json() == current.to_json()
        assert changed.fingerprint == current.fingerprint
    duplicated = {**current.model_dump(), "points": (current.points[0],) * 2}
    with pytest.raises(ValueError):
        LifecycleSnapshot.model_validate(duplicated)


def test_unknown_fields_duplicate_json_keys_and_naive_clock_reject():
    duplicate_keys = GOLDEN.read_text().replace('"schema_version": 1', '"schema_version": 1,"schema_version": 1')
    with pytest.raises(ValueError):
        LifecycleSnapshot.from_json(duplicate_keys)
    unknown_fields = {**sample().model_dump(), "undocumented": True}
    with pytest.raises(ValueError):
        LifecycleSnapshot.model_validate(unknown_fields)
    naive_clock = {**sample().model_dump(), "observed_at": datetime(2026, 1, 1)}
    with pytest.raises(ValueError):
        LifecycleSnapshot.model_validate(naive_clock)
    assert LifecycleSnapshot.model_validate({**sample().model_dump(), "observed_at": datetime(2026, 1, 1, tzinfo=UTC)})


def test_resource_adapter_preserves_actual_tracker_attribution_and_missingness():
    sample = ResourceSnapshot(None, 32.0, 0.0, 2.0, 250.0)
    values = {p.name: p for p in resource_points(sample)}
    assert values["rss"].status == "unavailable"
    assert values["cpu_time"].value == 0
    assert values["cpu_utilization"].value == 250
    with pytest.raises(ValueError):
        resource_points(ResourceSnapshot(-1, None, 0, 0, None))


def test_noop_revalidates_unchecked_instances_and_retains_no_mutable_state():
    observer = NoOpLifecycleTelemetry()
    before = sample().to_json()
    observer.record(sample())
    observer.record(sample())
    assert vars(observer) == {}
    assert sample().to_json() == before
    damaged = sample().model_copy(update={"schema_version": 2})
    with pytest.raises(ValueError):
        observer.record(damaged)
