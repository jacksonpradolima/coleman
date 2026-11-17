"""Real optional SDK gauges: bounded labels, explicit availability and no math."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from coleman.telemetry.export import LifecycleTelemetry
from coleman.telemetry.lifecycle import LifecycleSnapshot, point
from tests.telemetry.test_lifecycle import sample


@contextmanager
def real_reader():
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=(reader,))
    try:
        yield LifecycleTelemetry(provider.get_meter("qualification")), reader
    finally:
        provider.shutdown()


def collected(reader):
    from opentelemetry.sdk.metrics.export import Gauge

    data = reader.get_metrics_data()
    assert data is not None
    result = {}
    for resource in data.resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                assert isinstance(metric.data, Gauge)
                result[metric.name] = tuple(metric.data.data_points)
    return result


def test_actual_sdk_records_closed_labels_and_versions_remain_artifact_annotations():
    current = sample()
    with real_reader() as (observer, reader):
        observer.record(current)
        observer.record(current)
        metrics = collected(reader)
        value = metrics["coleman.lifecycle.concentration"]
        assert len(value) == 1
        assert value[0].value == 0.625
        assert dict(value[0].attributes or {}) == {"route": "replay"}
        assert "coleman.lifecycle.rss" not in metrics
        available = metrics["coleman.lifecycle.availability"]
        assert len(available) == len(current.points) * 4
        assert sum(p.value for p in available) == len(current.points)
        assert all(set(p.attributes or {}) == {"route", "metric", "status"} for p in available)
        assert all("ci-rule-mixture" not in str(p.attributes) for p in available)


def test_availability_clears_previous_status_and_never_overwrites_missing_as_zero():
    original = sample()
    with real_reader() as (observer, reader):
        for value in (32, None, 0):
            observer.record(
                LifecycleSnapshot.model_validate({**original.model_dump(), "points": (point("rss", value),)})
            )
            rows = collected(reader)
            availability = rows["coleman.lifecycle.availability"]
            ones = [(p.attributes or {})["status"] for p in availability if p.value == 1]
            assert ones == ["unavailable" if value is None else "observed"]
            assert rows["coleman.lifecycle.rss"][0].value == (32 if value is None else value)


def test_invalid_complete_snapshot_is_rejected_before_any_actual_export():
    with real_reader() as (observer, reader):
        damaged = sample().model_copy(update={"schema_version": 2})
        with pytest.raises(ValueError):
            observer.record(damaged)
        assert reader.get_metrics_data() is None


def test_concurrent_real_gauge_writers_leave_one_current_availability_status():
    current = sample()
    observations = [
        LifecycleSnapshot.model_validate(
            {**current.model_dump(), "points": (point("rss", None if index % 2 else index),)}
        )
        for index in range(40)
    ]
    with real_reader() as (observer, reader):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(observer.record, observations))
        availability = collected(reader)["coleman.lifecycle.availability"]
        assert sum(p.value for p in availability) == 1
        assert len(availability) == 4


def test_actual_sdk_retains_integer_count_instead_of_projecting_to_binary64():
    current = sample()
    observation = LifecycleSnapshot.model_validate({**current.model_dump(), "points": (point("updates", 2**53 + 1),)})
    with real_reader() as (observer, reader):
        observer.record(observation)
        value = collected(reader)["coleman.lifecycle.updates"][0].value
        assert type(value) is int
        assert value == 2**53 + 1
