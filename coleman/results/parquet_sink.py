"""
coleman.results.parquet_sink - Default Parquet Results Sink.

Implements the ``ResultsSink`` interface backed by **partitioned Parquet**
files with *zstd* compression.  Rows are buffered in memory up to a
configurable batch size and then flushed as a new Parquet file inside a
Hive-style partition tree.

Features
--------
* Bounded memory usage (batch_size controls max rows in RAM).
* Crash-safe: at most the current unflushed buffer is lost.
* ``prioritization_order`` is hashed by default to avoid storage explosion;
  an optional ``top_k`` retains the first *k* entries.
* Thread-safe: all public methods are guarded by a lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from coleman.results.sink_base import ResultsSink

# Arrow schema matching MonitorCollector columns.
# Keep this in sync with the monitor output so DuckDB can scan a stable layout.
_RESULT_SCHEMA = pa.schema(
    [
        ("scenario", pa.utf8()),
        ("experiment", pa.int64()),
        ("step", pa.int64()),
        ("execution_id", pa.utf8()),
        ("worker_id", pa.utf8()),
        ("parallel_mode", pa.utf8()),
        ("policy", pa.utf8()),
        ("reward_function", pa.utf8()),
        ("budget_mode", pa.utf8()),
        ("budget_value", pa.float64()),
        ("total_build_duration", pa.float64()),
        ("prioritization_time", pa.float64()),
        ("process_memory_rss_mib", pa.float64()),
        ("process_memory_peak_rss_mib", pa.float64()),
        ("process_cpu_utilization_percent", pa.float64()),
        ("process_cpu_time_seconds", pa.float64()),
        ("wall_time_seconds", pa.float64()),
        ("detected", pa.int64()),
        ("missed", pa.int64()),
        ("tests_ran", pa.int64()),
        ("tests_not_ran", pa.int64()),
        ("ttf", pa.float64()),
        ("ttf_duration", pa.float64()),
        ("time_reduction", pa.float64()),
        ("fitness", pa.float64()),
        ("cost", pa.float64()),
        ("rewards", pa.float64()),
        ("avg_precision", pa.float64()),
        ("prioritization_order_hash", pa.utf8()),
        ("prioritization_order_top_k", pa.utf8()),
        ("variant", pa.utf8()),
    ]
)


def _hash_order(order: Any) -> str:
    """Return a stable SHA-256 hex-digest for a prioritization order.

    Parameters
    ----------
    order : Any
        The prioritization order (typically a list).

    Returns
    -------
    str
        Hex-digest string.
    """
    raw = json.dumps(order, sort_keys=False, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _top_k(order: Any, k: int | None) -> str:
    """Return a JSON string of the first *k* items, or empty string.

    Parameters
    ----------
    order : Any
        Prioritization order.
    k : int or None
        Number of leading items to keep.  ``None`` means skip.

    Returns
    -------
    str
        JSON-encoded top-k list, or ``""``.
    """
    if k is None or not isinstance(order, list):
        return ""
    return json.dumps(order[:k], default=str)


class ParquetSink(ResultsSink):
    """Partitioned Parquet results sink (zstd compressed).

    Parameters
    ----------
    out_dir : str
        Root directory for Hive-style partitioned output.
    batch_size : int
        Maximum rows buffered in memory before an automatic flush.
    top_k : int or None
        If set, store the first *k* entries of ``prioritization_order`` in
        addition to the hash.  Default ``None`` (hash only).
    partition_cols : list[str] or None
        Columns used for Hive partitioning.  Defaults to
        ``["scenario", "policy", "reward_function", "budget_mode", "budget_value"]``.

    Attributes
    ----------
    out_dir : str
        Output directory.
    batch_size : int
        Flush threshold.
    top_k : int or None
        Top-k retention size.
    """

    def __init__(
        self,
        out_dir: str = "./runs",
        batch_size: int = 1000,
        top_k: int | None = None,
        partition_cols: list[str] | None = None,
    ) -> None:
        """Initialise the sink and create *out_dir* if it does not exist.

        Parameters
        ----------
        out_dir : str
            Root directory for Hive-style partitioned output.
        batch_size : int
            Maximum rows buffered in memory before an automatic flush.
        top_k : int or None
            If set, store the first *k* entries of ``prioritization_order``.
        partition_cols : list[str] or None
            Hive partition columns.  Defaults to
            ``["scenario", "policy", "reward_function", "budget_mode", "budget_value"]``.
        """
        self.out_dir = out_dir
        self.batch_size = batch_size
        self.top_k = top_k
        self._partition_cols = partition_cols or [
            "scenario",
            "policy",
            "reward_function",
            "budget_mode",
            "budget_value",
        ]
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._file_counter = 0
        os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Pickling support (threading.Lock is not picklable)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state (exclude the lock)."""
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state and recreate the lock."""
        for key, value in state.items():
            setattr(self, key, value)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_row(self, row: dict[str, Any]) -> None:
        """Buffer one result row, flushing when *batch_size* is reached.

        Parameters
        ----------
        row : dict[str, Any]
            Column-name → value mapping.
        """
        with self._lock:
            processed = self._process_row(row)
            self._buffer.append(processed)
            if len(self._buffer) >= self.batch_size:
                self._flush_locked()

    def flush(self) -> None:
        """Force-write the current buffer to Parquet."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Flush remaining data and release resources."""
        self.flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Transform a raw row dict into the Parquet schema layout.

        Parameters
        ----------
        row : dict[str, Any]
            Raw input row.

        Returns
        -------
        dict[str, Any]
            Row with ``prioritization_order`` replaced by hash + top-k.
        """
        out = dict(row)
        order = out.pop("prioritization_order", None)
        out["prioritization_order_hash"] = _hash_order(order)
        out["prioritization_order_top_k"] = _top_k(order, self.top_k)
        return out

    def _flush_locked(self) -> None:
        """Write buffered rows to a new Parquet file (caller holds lock)."""
        if not self._buffer:
            return

        table = pa.Table.from_pylist(self._buffer, schema=_RESULT_SCHEMA)
        sort_keys = [col for col in self._partition_cols if col in table.column_names]
        if sort_keys:
            table = table.sort_by([(col, "ascending") for col in sort_keys])

        self._file_counter += 1
        pid = os.getpid()
        basename = f"part-{self._file_counter:06d}-{pid}-{int(time.time())}-{{i}}.parquet"

        pq.write_to_dataset(
            table,
            root_path=self.out_dir,
            partition_cols=self._partition_cols,
            compression="zstd",
            basename_template=basename,
        )
        self._buffer.clear()
