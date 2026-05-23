"""Tests for the results sink interfaces and implementations."""

import json
import threading
from unittest.mock import Mock

import pyarrow.parquet as pq
import pytest

from coleman.results.duckdb_sink import DuckDBSink
from coleman.results.parquet_sink import ParquetSink, _hash_order, _top_k
from coleman.results.sink_base import NullSink, ResultsSink
from coleman.results.writer import _SENTINEL, ResultsWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(**overrides):
    """Build a minimal result row dict with sensible defaults."""
    row = {
        "scenario": "TestScenario",
        "experiment": 1,
        "step": 1,
        "execution_id": "exec-1",
        "worker_id": "1",
        "parallel_mode": "process",
        "policy": "UCB",
        "reward_function": "RNFail",
        "sched_time": 0.5,
        "sched_time_duration": 50.0,
        "total_build_duration": 100.0,
        "prioritization_time": 1.5,
        "process_memory_rss_mib": 64.0,
        "process_memory_peak_rss_mib": 96.0,
        "process_cpu_utilization_percent": 75.0,
        "process_cpu_time_seconds": 2.5,
        "wall_time_seconds": 3.0,
        "detected": 5,
        "missed": 2,
        "tests_ran": 10,
        "tests_not_ran": 3,
        "ttf": 2.0,
        "ttf_duration": 10.0,
        "time_reduction": 90.0,
        "fitness": 0.8,
        "cost": 0.6,
        "rewards": 0.9,
        "avg_precision": 0.75,
        "prioritization_order": ["tc1", "tc2", "tc3"],
        "variant": None,
    }
    row.update(overrides)
    return row


# ============================================================================
# NullSink
# ============================================================================


class TestNullSink:
    def test_is_results_sink(self):
        assert isinstance(NullSink(), ResultsSink)

    def test_write_row_is_noop(self):
        sink = NullSink()
        sink.write_row({"a": 1})  # should not raise

    def test_flush_is_noop(self):
        NullSink().flush()

    def test_close_is_noop(self):
        NullSink().close()


# ============================================================================
# _hash_order / _top_k helpers
# ============================================================================


class TestHashOrder:
    def test_deterministic(self):
        order = ["a", "b", "c"]
        assert _hash_order(order) == _hash_order(order)

    def test_different_orders_differ(self):
        assert _hash_order(["a", "b"]) != _hash_order(["b", "a"])

    def test_is_sha256_hex(self):
        h = _hash_order([1, 2, 3])
        assert len(h) == 64
        int(h, 16)  # valid hex

    def test_none_order(self):
        h = _hash_order(None)
        assert isinstance(h, str)
        assert len(h) == 64


class TestTopK:
    def test_none_k_returns_empty(self):
        assert _top_k(["a", "b", "c"], None) == ""

    def test_non_list_returns_empty(self):
        assert _top_k("not_a_list", 2) == ""

    def test_returns_json_list(self):
        result = _top_k(["a", "b", "c", "d"], 2)
        assert json.loads(result) == ["a", "b"]

    def test_k_larger_than_list(self):
        result = _top_k(["a"], 5)
        assert json.loads(result) == ["a"]


# ============================================================================
# ParquetSink
# ============================================================================


class TestParquetSink:
    def test_write_and_flush(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=5)
        for i in range(3):
            sink.write_row(_make_row(step=i))
        sink.flush()

        # Should produce at least one parquet file
        parquet_files = list(tmp_path.rglob("*.parquet"))
        assert len(parquet_files) >= 1

        table = pq.read_table(parquet_files[0])
        assert table.num_rows == 3

    def test_auto_flush_on_batch_size(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=2)
        sink.write_row(_make_row(step=1))
        sink.write_row(_make_row(step=2))  # triggers flush
        # Buffer should be empty after auto-flush
        assert len(sink._buffer) == 0

    def test_prioritization_order_hashed(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=10)
        sink.write_row(_make_row())
        sink.flush()

        parquet_files = list(tmp_path.rglob("*.parquet"))
        table = pq.read_table(parquet_files[0])
        assert "prioritization_order_hash" in table.column_names
        assert "prioritization_order" not in table.column_names

    def test_top_k_stored_when_set(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=10, top_k=2)
        sink.write_row(_make_row(prioritization_order=["a", "b", "c", "d"]))
        sink.flush()

        parquet_files = list(tmp_path.rglob("*.parquet"))
        table = pq.read_table(parquet_files[0])
        top_k_val = table.column("prioritization_order_top_k")[0].as_py()
        assert json.loads(top_k_val) == ["a", "b"]

    def test_close_flushes(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        sink.write_row(_make_row())
        sink.close()
        parquet_files = list(tmp_path.rglob("*.parquet"))
        assert len(parquet_files) >= 1

    def test_thread_safety(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=50)
        errors = []

        def writer(start):
            try:
                for i in range(20):
                    sink.write_row(_make_row(step=start + i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 20,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        sink.flush()
        assert not errors
        parquet_files = list(tmp_path.rglob("*.parquet"))
        total = sum(pq.read_table(f).num_rows for f in parquet_files)
        assert total == 100

    def test_picklable(self, tmp_path):
        """ParquetSink must survive pickle round-trip (needed for multiprocessing.Pool)."""
        import pickle

        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100, top_k=3)
        sink.write_row(_make_row(step=1))

        data = pickle.dumps(sink)
        restored = pickle.loads(data)  # noqa: S301

        assert restored.out_dir == sink.out_dir
        assert restored.batch_size == sink.batch_size
        assert restored.top_k == sink.top_k
        # Verify restored sink is functional
        restored.write_row(_make_row(step=2))
        restored.flush()


# ============================================================================
# ResultsWriter (thread-safe queue)
# ============================================================================


class TestResultsWriter:
    def test_enqueue_and_stop(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        writer = ResultsWriter(sink, max_queue_size=50)
        for i in range(10):
            writer.enqueue(_make_row(step=i))
        writer.stop()

        parquet_files = list(tmp_path.rglob("*.parquet"))
        total = sum(pq.read_table(f).num_rows for f in parquet_files)
        assert total == 10

    def test_auto_start(self, tmp_path):
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        writer = ResultsWriter(sink, max_queue_size=50)
        assert not writer._started
        writer.enqueue(_make_row())
        assert writer._started
        writer.stop()

    def test_null_sink_writer(self):
        sink = NullSink()
        writer = ResultsWriter(sink, max_queue_size=50)
        for i in range(10):
            writer.enqueue(_make_row(step=i))
        writer.stop()  # should not raise

    def test_drain_writes_remaining_items_after_stop_signal(self):
        """Cover branch where _drain sees stop marker before remaining queue items."""

        class RecordingSink(NullSink):
            def __init__(self):
                self.rows = []
                self.flushed = False

            def write_row(self, row):
                self.rows.append(row)

            def flush(self):
                self.flushed = True

        sink = RecordingSink()
        writer = ResultsWriter(sink, max_queue_size=10)

        # Drive _drain deterministically without background thread.
        writer._queue.put(_SENTINEL)  # pylint: disable=protected-access
        writer._queue.put(_make_row(step=99))  # pylint: disable=protected-access
        writer._drain()  # pylint: disable=protected-access

        assert len(sink.rows) == 1
        assert sink.rows[0]["step"] == 99
        assert sink.flushed is True


# ============================================================================
# DuckDBSink
# ============================================================================


class TestDuckDBSink:
    def test_write_and_flush_single_file(self, tmp_path):
        sink = DuckDBSink(out_dir=str(tmp_path / "runs"), batch_size=10, file_count=1)
        for i in range(3):
            sink.write_row(_make_row(step=i))
        sink.flush()

        db_file = tmp_path / "runs" / "results.duckdb"
        assert db_file.exists()

        import duckdb

        conn = duckdb.connect(str(db_file))
        count_row = conn.execute("SELECT COUNT(*) FROM coleman_results").fetchone()
        conn.close()
        sink.close()
        assert count_row is not None
        count = count_row[0]
        assert count == 3

    def test_target_idx_falls_back_when_shard_key_missing(self, tmp_path):
        sink = DuckDBSink(out_dir=str(tmp_path / "runs"), batch_size=10, file_count=2, shard_key="missing_key")
        idx = sink._target_idx(_make_row(execution_id=None, scenario="scenario-x"))
        assert idx in {0, 1}
        sink.close()

    def test_auto_flush_on_batch_size(self, tmp_path):
        sink = DuckDBSink(out_dir=str(tmp_path / "runs"), batch_size=1, file_count=1)
        sink.write_row(_make_row(step=1))
        assert sink._buffers[0] == []
        sink.close()

    def test_multiple_files_respected(self, tmp_path):
        sink = DuckDBSink(out_dir=str(tmp_path / "runs"), batch_size=100, file_count=2)
        for i in range(20):
            sink.write_row(_make_row(step=i, execution_id=f"exec-{i}"))
        sink.close()

        files = sorted((tmp_path / "runs").glob("results_*.duckdb"))
        assert len(files) == 2

        import duckdb

        total = 0
        for file in files:
            conn = duckdb.connect(str(file))
            count_row = conn.execute("SELECT COUNT(*) FROM coleman_results").fetchone()
            conn.close()
            assert count_row is not None
            total += count_row[0]
        assert total == 20

    def test_prioritization_order_hash_columns(self, tmp_path):
        sink = DuckDBSink(out_dir=str(tmp_path / "runs"), batch_size=10, top_k=2)
        sink.write_row(_make_row(prioritization_order=["a", "b", "c"]))
        sink.close()

        import duckdb

        conn = duckdb.connect(str(tmp_path / "runs" / "results.duckdb"))
        row = conn.execute(
            "SELECT prioritization_order_hash, prioritization_order_top_k FROM coleman_results"
        ).fetchone()
        conn.close()

        assert row is not None
        assert isinstance(row[0], str)
        assert json.loads(row[1]) == ["a", "b"]

    def test_close_attempts_all_shards_even_if_one_flush_fails(self, tmp_path):
        sink = DuckDBSink(out_dir=str(tmp_path / "runs"), batch_size=10, file_count=2)
        sink._flush_bucket_locked = Mock(side_effect=[RuntimeError("boom"), None])  # type: ignore[attr-defined]

        with pytest.raises(RuntimeError, match="boom"):
            sink.close()


# ============================================================================
# DuckDBCatalog
# ============================================================================


class TestDuckDBCatalog:
    def test_sql_scrubbing_helpers_handle_comments_and_literals(self):
        from coleman.results.duckdb_catalog import _count_sql_statements, _strip_sql_literals_and_comments

        sql = "SELECT 'a''b' AS s -- comment with DROP\n/* block comment INSERT */ SELECT \"x; y\" AS c"
        scrubbed = _strip_sql_literals_and_comments(sql)

        assert "DROP" not in scrubbed
        assert "INSERT" not in scrubbed
        assert _count_sql_statements("SELECT 1; -- trailing comment\nSELECT 2") == 2
        assert _count_sql_statements("SELECT ';' AS marker") == 1

    def test_create_view_and_query(self, tmp_path):
        """DuckDBCatalog should create a view over Parquet and return query results."""
        from coleman.results.duckdb_catalog import DuckDBCatalog

        # Write some rows via ParquetSink first
        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        for i in range(5):
            sink.write_row(_make_row(step=i, fitness=0.5 + i * 0.1))
        sink.close()

        cat = DuckDBCatalog(str(tmp_path / "runs"))
        df = cat.query("SELECT COUNT(*) AS cnt FROM results")
        assert df["cnt"][0] == 5
        cat.close()

    def test_query_aggregation(self, tmp_path):
        """DuckDBCatalog should support aggregation queries over the view."""
        from coleman.results.duckdb_catalog import DuckDBCatalog

        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        for i in range(3):
            sink.write_row(_make_row(step=i, fitness=float(i)))
        sink.close()

        cat = DuckDBCatalog(str(tmp_path / "runs"))
        df = cat.query("SELECT AVG(fitness) AS avg_fitness FROM results")
        assert abs(df["avg_fitness"][0] - 1.0) < 1e-6
        cat.close()

    def test_close(self, tmp_path):
        """DuckDBCatalog.close() should not raise."""
        from coleman.results.duckdb_catalog import DuckDBCatalog

        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        sink.write_row(_make_row())
        sink.close()

        cat = DuckDBCatalog(str(tmp_path / "runs"))
        cat.close()  # should not raise

    def test_query_blocks_mutating_sql_in_read_only_mode(self, tmp_path):
        """DuckDBCatalog should reject mutating SQL by default."""
        from coleman.results.duckdb_catalog import DuckDBCatalog

        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        sink.write_row(_make_row())
        sink.close()

        cat = DuckDBCatalog(str(tmp_path / "runs"))
        with pytest.raises(ValueError, match="read-only mode"):
            cat.query("DELETE FROM results")
        cat.close()

    def test_query_blocks_empty_sql(self, tmp_path):
        """DuckDBCatalog should reject empty SQL strings in read-only mode."""
        from coleman.results.duckdb_catalog import DuckDBCatalog

        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        sink.write_row(_make_row())
        sink.close()

        cat = DuckDBCatalog(str(tmp_path / "runs"))
        with pytest.raises(ValueError, match="cannot be empty"):
            cat.query("   ")
        cat.close()

    def test_query_blocks_multi_statement_in_read_only_mode(self, tmp_path):
        """DuckDBCatalog should reject multiple statements by default."""
        from coleman.results.duckdb_catalog import DuckDBCatalog

        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        sink.write_row(_make_row())
        sink.close()

        cat = DuckDBCatalog(str(tmp_path / "runs"))
        with pytest.raises(ValueError, match="Multiple SQL statements"):
            cat.query("SELECT 1; SELECT 2")
        cat.close()

    def test_query_allows_mutating_sql_when_read_only_disabled(self, tmp_path):
        """DuckDBCatalog can optionally allow mutating SQL when explicitly requested."""
        from coleman.results.duckdb_catalog import DuckDBCatalog

        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        sink.write_row(_make_row())
        sink.close()

        cat = DuckDBCatalog(str(tmp_path / "runs"), read_only=False)
        df = cat.query("SELECT COUNT(*) AS cnt FROM results")
        assert df["cnt"][0] == 1
        cat.close()

    def test_query_allows_semicolon_inside_string_literal(self, tmp_path):
        """Semicolons inside string literals should not be treated as multi-statement SQL."""
        from coleman.results.duckdb_catalog import DuckDBCatalog

        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        sink.write_row(_make_row())
        sink.close()

        cat = DuckDBCatalog(str(tmp_path / "runs"))
        df = cat.query("SELECT ';' AS marker")
        assert df["marker"][0] == ";"
        cat.close()

    def test_query_allows_write_keyword_inside_string_literal(self, tmp_path):
        """Write keywords inside literals should not trigger read-only mutation guard."""
        from coleman.results.duckdb_catalog import DuckDBCatalog

        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        sink.write_row(_make_row())
        sink.close()

        cat = DuckDBCatalog(str(tmp_path / "runs"))
        df = cat.query("SELECT 'DROP TABLE results' AS statement_text")
        assert df["statement_text"][0] == "DROP TABLE results"
        cat.close()

    def test_query_blocks_mutating_keyword_in_block_comment(self, tmp_path):
        """Mutating keywords inside comments should still be ignored by the read-only guard."""
        from coleman.results.duckdb_catalog import DuckDBCatalog

        sink = ParquetSink(out_dir=str(tmp_path / "runs"), batch_size=100)
        sink.write_row(_make_row())
        sink.close()

        cat = DuckDBCatalog(str(tmp_path / "runs"))
        df = cat.query("SELECT 1 /* DROP TABLE results */ AS one")
        assert df["one"][0] == 1
        cat.close()
