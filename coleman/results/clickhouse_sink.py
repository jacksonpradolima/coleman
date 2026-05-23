"""
coleman.results.clickhouse_sink - Optional ClickHouse Results Sink.

Requires ``pip install coleman[clickhouse]`` which pulls in the
``clickhouse-connect`` driver.  This sink inserts result rows into a
ClickHouse table in batches.

When the extra is **not** installed, importing this module will raise
``ImportError`` at construction time only — the rest of the framework
remains unaffected.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any

from coleman.results.sink_base import ResultsSink

_CLICKHOUSE_TABLE = "coleman_results"
_NULLABLE_STRING = "Nullable(String)"
_NULLABLE_FLOAT64 = "Nullable(Float64)"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_CLICKHOUSE_TABLE} (
    scenario           String,
    experiment         Int64,
    step               Int64,
    execution_id       {_NULLABLE_STRING},
    worker_id          {_NULLABLE_STRING},
    parallel_mode      {_NULLABLE_STRING},
    policy             String,
    reward_function    String,
    budget_mode        {_NULLABLE_STRING},
    budget_value       {_NULLABLE_FLOAT64},
    total_build_duration Float64,
    prioritization_time Float64,
    process_memory_rss_mib {_NULLABLE_FLOAT64},
    process_memory_peak_rss_mib {_NULLABLE_FLOAT64},
    process_cpu_utilization_percent {_NULLABLE_FLOAT64},
    process_cpu_time_seconds {_NULLABLE_FLOAT64},
    wall_time_seconds {_NULLABLE_FLOAT64},
    detected           Int64,
    missed             Int64,
    tests_ran          Int64,
    tests_not_ran      Int64,
    ttf                Float64,
    ttf_duration       Float64,
    time_reduction     Float64,
    fitness            Float64,
    cost               Float64,
    rewards            Float64,
    avg_precision      Float64,
    prioritization_order String,
    variant            {_NULLABLE_STRING}
) ENGINE = MergeTree()
ORDER BY (scenario, policy, reward_function, experiment, step)
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
    "budget_mode",
    "budget_value",
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
    "prioritization_order",
    "variant",
]


class ClickHouseSink(ResultsSink):
    """ClickHouse results sink (optional).

    Parameters
    ----------
    host : str
        ClickHouse server hostname.
    port : int
        ClickHouse HTTP interface port.
    database : str
        Target database name.
    batch_size : int
        Rows buffered before an automatic insert.

    Raises
    ------
    ImportError
        If ``clickhouse-connect`` is not installed.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        database: str = "default",
        batch_size: int = 1000,
        *,
        username: str | None = None,
        password: str | None = None,
        secure: bool = True,
        verify: bool = True,
        connect_timeout: int | None = 10,
        send_receive_timeout: int | None = 30,
    ) -> None:
        """Initialise the ClickHouse client and ensure the results table exists.

        Parameters
        ----------
        host : str
            ClickHouse server hostname.
        port : int
            ClickHouse HTTP interface port.
        database : str
            Target database name.
        batch_size : int
            Rows buffered before an automatic insert.
        username : str | None
            ClickHouse username. If omitted, client defaults apply.
        password : str | None
            ClickHouse password. If omitted, client defaults apply.
        secure : bool
            Whether to use TLS for transport.
        verify : bool
            Whether to verify TLS certificates.
        connect_timeout : int | None
            Connection timeout in seconds.
        send_receive_timeout : int | None
            Network read/write timeout in seconds.

        Raises
        ------
        ImportError
            If ``clickhouse-connect`` is not installed.
        """
        try:
            clickhouse_connect = importlib.import_module("clickhouse_connect")
        except ImportError as exc:
            raise ImportError(
                "clickhouse-connect is required for ClickHouseSink. Install it with: pip install coleman[clickhouse]"
            ) from exc

        client_kwargs: dict[str, Any] = {
            "host": host,
            "port": port,
            "database": database,
            "secure": secure,
            "verify": verify,
        }
        if username is not None:
            client_kwargs["username"] = username
        if password is not None:
            client_kwargs["password"] = password
        if connect_timeout is not None:
            client_kwargs["connect_timeout"] = connect_timeout
        if send_receive_timeout is not None:
            client_kwargs["send_receive_timeout"] = send_receive_timeout

        self._client = clickhouse_connect.get_client(**client_kwargs)
        self._client.command(_CREATE_TABLE_SQL)
        self._ensure_schema()
        self.batch_size = batch_size
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _ensure_schema(self) -> None:
        """Add newly introduced columns for existing tables without requiring manual migration."""
        for column_name, column_type in [
            ("execution_id", _NULLABLE_STRING),
            ("worker_id", _NULLABLE_STRING),
            ("parallel_mode", _NULLABLE_STRING),
            ("process_memory_rss_mib", _NULLABLE_FLOAT64),
            ("process_memory_peak_rss_mib", _NULLABLE_FLOAT64),
            ("process_cpu_utilization_percent", _NULLABLE_FLOAT64),
            ("process_cpu_time_seconds", _NULLABLE_FLOAT64),
            ("wall_time_seconds", _NULLABLE_FLOAT64),
            ("variant", _NULLABLE_STRING),
            ("budget_mode", _NULLABLE_STRING),
            ("budget_value", _NULLABLE_FLOAT64),
        ]:
            self._client.command(
                f"ALTER TABLE {_CLICKHOUSE_TABLE} ADD COLUMN IF NOT EXISTS {column_name} {column_type}"
            )

    def write_row(self, row: dict[str, Any]) -> None:
        """Buffer one result row.

        Parameters
        ----------
        row : dict[str, Any]
            Column-name → value mapping.
        """
        with self._lock:
            # Ensure prioritization_order is a string for ClickHouse
            r = dict(row)
            if isinstance(r.get("prioritization_order"), list):
                r["prioritization_order"] = str(r["prioritization_order"])
            self._buffer.append(r)
            if len(self._buffer) >= self.batch_size:
                self._flush_locked()

    def flush(self) -> None:
        """Force-write the current buffer to ClickHouse."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Flush remaining data and close the connection."""
        self.flush()
        self._client.close()

    def _flush_locked(self) -> None:
        """Insert buffered rows into ClickHouse (caller holds lock)."""
        if not self._buffer:
            return
        data = [[row.get(col) for col in _INSERT_COLS] for row in self._buffer]
        self._client.insert(_CLICKHOUSE_TABLE, data, column_names=_INSERT_COLS)
        self._buffer.clear()
