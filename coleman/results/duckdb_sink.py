"""coleman.results.duckdb_sink - Consolidated DuckDB Results Sink.

Implements ``ResultsSink`` backed by one or more DuckDB database files.
Rows are buffered in memory and inserted in batches into a ``coleman_results``
table. By default all rows are consolidated into a single file.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import duckdb

from coleman.results.sink_base import ResultsSink

_TABLE_NAME = "coleman_results"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
    scenario VARCHAR,
    experiment BIGINT,
    step BIGINT,
    execution_id VARCHAR,
    worker_id VARCHAR,
    parallel_mode VARCHAR,
    policy VARCHAR,
    reward_function VARCHAR,
    sched_time DOUBLE,
    sched_time_duration DOUBLE,
    total_build_duration DOUBLE,
    prioritization_time DOUBLE,
    process_memory_rss_mib DOUBLE,
    process_memory_peak_rss_mib DOUBLE,
    process_cpu_utilization_percent DOUBLE,
    process_cpu_time_seconds DOUBLE,
    wall_time_seconds DOUBLE,
    detected BIGINT,
    missed BIGINT,
    tests_ran BIGINT,
    tests_not_ran BIGINT,
    ttf DOUBLE,
    ttf_duration DOUBLE,
    time_reduction DOUBLE,
    fitness DOUBLE,
    cost DOUBLE,
    rewards DOUBLE,
    avg_precision DOUBLE,
    prioritization_order_hash VARCHAR,
    prioritization_order_top_k VARCHAR,
    variant VARCHAR
)
"""

_INSERT_COLS = [
    "scenario",
    "experiment",
    "step",
    "execution_id",
    "worker_id",
    "parallel_mode",
    "policy",
    "reward_function",
    "sched_time",
    "sched_time_duration",
    "total_build_duration",
    "prioritization_time",
    "process_memory_rss_mib",
    "process_memory_peak_rss_mib",
    "process_cpu_utilization_percent",
    "process_cpu_time_seconds",
    "wall_time_seconds",
    "detected",
    "missed",
    "tests_ran",
    "tests_not_ran",
    "ttf",
    "ttf_duration",
    "time_reduction",
    "fitness",
    "cost",
    "rewards",
    "avg_precision",
    "prioritization_order_hash",
    "prioritization_order_top_k",
    "variant",
]


def _hash_order(order: Any) -> str:
    """Return a stable SHA-256 hex digest for prioritization order."""
    raw = json.dumps(order, sort_keys=False, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _top_k(order: Any, k: int | None) -> str:
    """Return JSON top-k list or empty string when disabled."""
    if k is None or not isinstance(order, list):
        return ""
    return json.dumps(order[:k], default=str)


class DuckDBSink(ResultsSink):
    """DuckDB results sink.

    Parameters
    ----------
    out_dir : str
        Output directory where DuckDB files are stored.
    batch_size : int
        Number of rows buffered before automatic flush.
    file_count : int
        Number of DuckDB files to shard writes across. Default 1.
    base_name : str
        Base filename used for generated DuckDB files.
    top_k : int | None
        Optional number of prioritization items to persist in addition to hash.
    shard_key : str
        Row key used to shard writes when ``file_count > 1``.
    """

    def __init__(
        self,
        out_dir: str = "./runs",
        batch_size: int = 1000,
        file_count: int = 1,
        base_name: str = "results",
        top_k: int | None = None,
        shard_key: str = "execution_id",
    ) -> None:
        """Initialise DuckDB result storage and create target table(s).

        Parameters
        ----------
        out_dir : str
            Output directory where DuckDB files are stored.
        batch_size : int
            Number of buffered rows before automatic flush.
        file_count : int
            Number of DuckDB files used for sharded writes.
        base_name : str
            Base filename for generated DuckDB files.
        top_k : int | None
            Optional number of prioritization items persisted besides hash.
        shard_key : str
            Row key used to distribute writes when ``file_count > 1``.
        """
        self.out_dir = out_dir
        self.batch_size = batch_size
        self.file_count = max(1, int(file_count))
        self.base_name = base_name
        self.top_k = top_k
        self.shard_key = shard_key

        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)

        self._buffers: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(self.file_count)}
        self._conns: dict[int, duckdb.DuckDBPyConnection] = {}
        self._lock = threading.Lock()

        for idx in range(self.file_count):
            conn = duckdb.connect(str(self._db_path(idx)))
            conn.execute(_CREATE_TABLE_SQL)
            self._conns[idx] = conn

    def _db_path(self, idx: int) -> Path:
        """Return DuckDB file path for a shard index.

        Parameters
        ----------
        idx : int
            Shard index.

        Returns
        -------
        Path
            Absolute output path for the shard database file.
        """
        if self.file_count == 1:
            return Path(self.out_dir) / f"{self.base_name}.duckdb"
        return Path(self.out_dir) / f"{self.base_name}_{idx:03d}.duckdb"

    def _target_idx(self, row: dict[str, Any]) -> int:
        """Map one row to a shard index.

        Parameters
        ----------
        row : dict[str, Any]
            Result row.

        Returns
        -------
        int
            Target shard index in ``[0, file_count)``.
        """
        if self.file_count == 1:
            return 0

        value = row.get(self.shard_key)
        if value is None:
            value = row.get("execution_id") or row.get("scenario") or "default"
        digest = hashlib.sha256(str(value).encode()).hexdigest()
        return int(digest, 16) % self.file_count

    def _process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Normalize a raw row to the sink table layout.

        Parameters
        ----------
        row : dict[str, Any]
            Raw result row.

        Returns
        -------
        dict[str, Any]
            Processed row with prioritization hash and optional top-k string.
        """
        out = dict(row)
        order = out.pop("prioritization_order", None)
        out["prioritization_order_hash"] = _hash_order(order)
        out["prioritization_order_top_k"] = _top_k(order, self.top_k)
        return out

    def write_row(self, row: dict[str, Any]) -> None:
        """Buffer one result row and auto-flush per shard batch size."""
        with self._lock:
            processed = self._process_row(row)
            idx = self._target_idx(processed)
            bucket = self._buffers[idx]
            bucket.append(processed)
            if len(bucket) >= self.batch_size:
                self._flush_bucket_locked(idx)

    def _flush_bucket_locked(self, idx: int) -> None:
        """Flush one shard buffer to DuckDB.

        Parameters
        ----------
        idx : int
            Shard index whose buffered rows should be inserted.

        Notes
        -----
        Caller must hold ``self._lock``.
        """
        bucket = self._buffers[idx]
        if not bucket:
            return
        conn = self._conns[idx]
        values = [[row.get(col) for col in _INSERT_COLS] for row in bucket]
        placeholders = ", ".join(["?"] * len(_INSERT_COLS))
        conn.executemany(
            f"INSERT INTO {_TABLE_NAME} ({', '.join(_INSERT_COLS)}) VALUES ({placeholders})",
            values,
        )
        bucket.clear()

    def flush(self) -> None:
        """Force-write all buffered rows to DuckDB files."""
        with self._lock:
            for idx in range(self.file_count):
                self._flush_bucket_locked(idx)

    def close(self) -> None:
        """Flush and close all DuckDB connections."""
        with self._lock:
            for idx in range(self.file_count):
                self._flush_bucket_locked(idx)
            for conn in self._conns.values():
                conn.close()
