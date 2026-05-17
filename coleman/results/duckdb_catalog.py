"""
coleman.results.duckdb_catalog - Optional DuckDB Views over Parquet.

Provides a thin helper that creates analytical DuckDB views on top of the
Hive-partitioned Parquet dataset produced by ``ParquetSink``.  This allows
users to run ad-hoc SQL queries over experiment results without loading data
into RAM.

Usage
-----
>>> from coleman.results.duckdb_catalog import DuckDBCatalog
>>> cat = DuckDBCatalog("./runs")
>>> df = cat.query("SELECT scenario, AVG(fitness) FROM results GROUP BY 1")
"""

from __future__ import annotations

import re

import duckdb
import polars as pl

_WRITE_OR_DDL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE|ATTACH|DETACH|COPY|INSTALL|LOAD|CALL|EXPORT)\b",
    re.IGNORECASE,
)


class DuckDBCatalog:
    """Read-only DuckDB view layer over a Parquet results dataset.

    Parameters
    ----------
    parquet_root : str
        Root directory of the Hive-partitioned Parquet dataset.
    db_path : str
        DuckDB database path.  Default ``:memory:`` (in-process, ephemeral).

    Attributes
    ----------
    parquet_root : str
        Parquet root directory.
    conn : duckdb.DuckDBPyConnection
        DuckDB connection.
    """

    def __init__(self, parquet_root: str, db_path: str = ":memory:", read_only: bool = True) -> None:
        """Initialise the catalog and create the ``results`` view.

        Parameters
        ----------
        parquet_root : str
            Root directory of the Hive-partitioned Parquet dataset.
        db_path : str
            DuckDB database path.  Default ``:memory:`` (ephemeral).
        read_only : bool
            If True, block mutating/DDL SQL in :meth:`query`.
        """
        self.parquet_root = parquet_root
        self.conn = duckdb.connect(db_path)
        self._read_only = read_only
        self._create_view()

    def _create_view(self) -> None:
        """Create the ``results`` view pointing at the Parquet dataset."""
        glob_path = f"{self.parquet_root}/**/*.parquet"
        escaped = glob_path.replace("'", "''")
        self.conn.execute(
            f"CREATE OR REPLACE VIEW results AS SELECT * FROM read_parquet('{escaped}', hive_partitioning=1)",
        )

    def query(self, sql: str) -> pl.DataFrame:
        """Execute an SQL query against the results view.

        Parameters
        ----------
        sql : str
            SQL statement.

        Returns
        -------
        polars.DataFrame
            Query result as a DataFrame.
        """
        self._validate_query(sql)
        return self.conn.execute(sql).pl()

    def _validate_query(self, sql: str) -> None:
        """Validate a query string before execution.

        In read-only mode, block multi-statement SQL and mutating/DDL commands.
        """
        if not self._read_only:
            return

        normalized = sql.strip()
        if not normalized:
            msg = "SQL query cannot be empty"
            raise ValueError(msg)

        # Only allow a single statement in read-only mode.
        if ";" in normalized.rstrip(";"):
            msg = "Multiple SQL statements are not allowed in read-only mode"
            raise ValueError(msg)

        if _WRITE_OR_DDL_PATTERN.search(normalized):
            msg = "Mutating/DDL SQL is not allowed in read-only mode"
            raise ValueError(msg)

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()
