"""Unit tests for ClickHouseSink using mocked clickhouse-connect client."""

# ruff: noqa: I001
from unittest.mock import MagicMock, patch

import pytest

from coleman.results.clickhouse_sink import ClickHouseSink, _CLICKHOUSE_TABLE, _INSERT_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client() -> MagicMock:
    """Return a fully mocked ClickHouse client."""
    client = MagicMock()
    client.command.return_value = None
    client.insert.return_value = None
    client.close.return_value = None
    return client


def _make_mock_cc_module(client: MagicMock) -> MagicMock:
    """Return a lightweight fake ``clickhouse_connect`` module."""
    mod = MagicMock()
    mod.get_client.return_value = client
    return mod


def _row(**overrides) -> dict:
    """Build a minimal result row."""
    row = {
        "scenario": "S1",
        "experiment": 1,
        "step": 1,
        "execution_id": "exec-1",
        "worker_id": "1",
        "parallel_mode": "sequential",
        "policy": "Random",
        "reward_function": "RNFail",
        "sched_time": 0.5,
        "sched_time_duration": 50.0,
        "total_build_duration": 100.0,
        "prioritization_time": 0.1,
        "process_memory_rss_mib": None,
        "process_memory_peak_rss_mib": None,
        "process_cpu_utilization_percent": None,
        "process_cpu_time_seconds": None,
        "wall_time_seconds": None,
        "detected": 3,
        "missed": 1,
        "tests_ran": 5,
        "tests_not_ran": 2,
        "ttf": 1.0,
        "ttf_duration": 5.0,
        "time_reduction": 80.0,
        "fitness": 0.75,
        "cost": 0.5,
        "rewards": 0.8,
        "avg_precision": 0.7,
        "prioritization_order": ["tc1", "tc2"],
        "variant": None,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Import-error path
# ---------------------------------------------------------------------------


class TestClickHouseSinkImportError:
    def test_raises_import_error_when_driver_missing(self):
        """ClickHouseSink must raise ImportError if clickhouse-connect is absent."""
        with (
            patch("coleman.results.clickhouse_sink.importlib.import_module", side_effect=ImportError("no module")),
            pytest.raises(ImportError, match="clickhouse-connect"),
        ):
            ClickHouseSink(host="localhost", port=8123)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestClickHouseSinkInit:
    def test_creates_table_and_ensures_schema(self):
        client = _make_mock_client()
        mod = _make_mock_cc_module(client)

        with patch("coleman.results.clickhouse_sink.importlib.import_module", return_value=mod):
            sink = ClickHouseSink(host="h", port=9000, database="db", batch_size=50)

        # get_client called with the supplied host/port/database + secure defaults
        mod.get_client.assert_called_once_with(
            host="h",
            port=9000,
            database="db",
            secure=True,
            verify=True,
            connect_timeout=10,
            send_receive_timeout=30,
        )
        # First command call is CREATE TABLE
        first_call_args = client.command.call_args_list[0]
        assert "CREATE TABLE IF NOT EXISTS" in first_call_args[0][0]
        # _ensure_schema calls ALTER TABLE for each optional column
        alter_calls = [c for c in client.command.call_args_list if "ALTER TABLE" in str(c)]
        assert len(alter_calls) > 0
        assert sink.batch_size == 50
        assert sink._buffer == []

    def test_defaults(self):
        client = _make_mock_client()
        mod = _make_mock_cc_module(client)

        with patch("coleman.results.clickhouse_sink.importlib.import_module", return_value=mod):
            sink = ClickHouseSink()

        mod.get_client.assert_called_once_with(
            host="localhost",
            port=8123,
            database="default",
            secure=True,
            verify=True,
            connect_timeout=10,
            send_receive_timeout=30,
        )
        assert sink.batch_size == 1000

    def test_allows_overriding_connection_security_and_auth(self):
        client = _make_mock_client()
        mod = _make_mock_cc_module(client)

        with patch("coleman.results.clickhouse_sink.importlib.import_module", return_value=mod):
            ClickHouseSink(
                host="h",
                port=9440,
                database="db",
                username="user",
                password="secret",
                secure=False,
                verify=False,
                connect_timeout=3,
                send_receive_timeout=7,
            )

        mod.get_client.assert_called_once_with(
            host="h",
            port=9440,
            database="db",
            username="user",
            password="secret",
            secure=False,
            verify=False,
            connect_timeout=3,
            send_receive_timeout=7,
        )


# ---------------------------------------------------------------------------
# write_row
# ---------------------------------------------------------------------------


class TestClickHouseSinkWriteRow:
    def _make_sink(self, batch_size: int = 100) -> tuple[ClickHouseSink, MagicMock]:
        client = _make_mock_client()
        mod = _make_mock_cc_module(client)
        with patch("coleman.results.clickhouse_sink.importlib.import_module", return_value=mod):
            sink = ClickHouseSink(batch_size=batch_size)
        return sink, client

    def test_row_goes_into_buffer(self):
        sink, client = self._make_sink(batch_size=10)
        sink.write_row(_row())
        assert len(sink._buffer) == 1
        client.insert.assert_not_called()

    def test_auto_flush_on_batch_size(self):
        sink, client = self._make_sink(batch_size=2)
        sink.write_row(_row(step=1))
        sink.write_row(_row(step=2))
        # Should have auto-flushed after the second row
        client.insert.assert_called_once()
        assert sink._buffer == []

    def test_prioritization_order_is_stringified(self):
        """Lists in prioritization_order must be converted to strings before insert."""
        sink, client = self._make_sink(batch_size=1)
        sink.write_row(_row(prioritization_order=["tc1", "tc2"]))
        insert_call = client.insert.call_args
        data = insert_call[0][1]  # second positional arg is the data
        order_idx = _INSERT_COLS.index("prioritization_order")
        assert isinstance(data[0][order_idx], str)

    def test_non_list_prioritization_order_passed_through(self):
        """Non-list prioritization_order should not be modified."""
        sink, client = self._make_sink(batch_size=1)
        sink.write_row(_row(prioritization_order="already_a_string"))
        insert_call = client.insert.call_args
        data = insert_call[0][1]
        order_idx = _INSERT_COLS.index("prioritization_order")
        assert data[0][order_idx] == "already_a_string"


# ---------------------------------------------------------------------------
# flush
# ---------------------------------------------------------------------------


class TestClickHouseSinkFlush:
    def _make_sink(self) -> tuple[ClickHouseSink, MagicMock]:
        client = _make_mock_client()
        mod = _make_mock_cc_module(client)
        with patch("coleman.results.clickhouse_sink.importlib.import_module", return_value=mod):
            sink = ClickHouseSink(batch_size=100)
        return sink, client

    def test_flush_empty_buffer_is_noop(self):
        sink, client = self._make_sink()
        sink.flush()
        client.insert.assert_not_called()

    def test_flush_sends_buffered_rows(self):
        sink, client = self._make_sink()
        sink.write_row(_row(step=1))
        sink.write_row(_row(step=2))
        assert len(sink._buffer) == 2

        sink.flush()
        client.insert.assert_called_once()
        assert sink._buffer == []

    def test_flush_passes_correct_table_and_columns(self):
        sink, client = self._make_sink()
        sink.write_row(_row())
        sink.flush()
        insert_args, insert_kwargs = client.insert.call_args
        assert insert_args[0] == _CLICKHOUSE_TABLE
        assert insert_kwargs["column_names"] == _INSERT_COLS


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClickHouseSinkClose:
    def test_close_flushes_then_closes_client(self):
        client = _make_mock_client()
        mod = _make_mock_cc_module(client)
        with patch("coleman.results.clickhouse_sink.importlib.import_module", return_value=mod):
            sink = ClickHouseSink(batch_size=100)

        sink.write_row(_row())
        client.insert.assert_not_called()

        sink.close()
        client.insert.assert_called_once()
        client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestClickHouseSinkThreadSafety:
    def test_concurrent_writes_do_not_corrupt_buffer(self):
        import threading

        client = _make_mock_client()
        mod = _make_mock_cc_module(client)
        with patch("coleman.results.clickhouse_sink.importlib.import_module", return_value=mod):
            sink = ClickHouseSink(batch_size=1000)

        results: list[Exception] = []

        def writer():
            try:
                for i in range(20):
                    sink.write_row(_row(step=i))
            except Exception as exc:  # noqa: BLE001
                results.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not results, f"Thread raised: {results}"
        # All 100 rows should be in buffer (batch_size=1000)
        assert len(sink._buffer) == 100
