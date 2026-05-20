"""
coleman.results - Results (Fact Data) Pipeline.

This subpackage provides a pluggable results-sink architecture for persisting
experiment facts (NAPFD, APFDc, step metadata, etc.) in a scalable,
framework-first way.

Default: **Partitioned Parquet** (zstd compressed, Hive-style).
Optional: DuckDB catalog views, ClickHouse sink.

When results collection is disabled the ``NullSink`` is used, which discards
all rows with near-zero overhead.
"""

from coleman.results.sink_base import NullSink, ResultsSink

__all__ = ["ResultsSink", "NullSink"]
