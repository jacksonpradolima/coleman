"""
coleman.results.writer - Thread-Safe Bounded-Queue Writer.

Provides a ``ResultsWriter`` that accepts row dicts through a bounded queue
and flushes them to a ``ResultsSink`` via a dedicated background thread.

This decouples the hot path (``monitor.collect``) from I/O latency and
guarantees bounded memory usage regardless of experiment length.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

from coleman.results.sink_base import ResultsSink

logger = logging.getLogger(__name__)


class _StopSignal:
    """Marker object used to stop the writer thread."""


_SENTINEL = _StopSignal()
_QueueItem = dict[str, Any] | _StopSignal


class ResultsWriter:
    """Thread-safe writer that drains a bounded queue into a ``ResultsSink``.

    Parameters
    ----------
    sink : ResultsSink
        The target results sink.
    max_queue_size : int
        Maximum number of rows that can be queued.  When the queue is full
        the calling thread blocks until space is available.

    Attributes
    ----------
    sink : ResultsSink
        Target sink.
    """

    def __init__(self, sink: ResultsSink, max_queue_size: int = 10_000) -> None:
        """Initialise the writer with a target sink and queue capacity.

        Parameters
        ----------
        sink : ResultsSink
            The target results sink.
        max_queue_size : int
            Maximum number of rows that can be queued before the caller blocks.
        """
        self.sink = sink
        self._queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=max_queue_size)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background writer thread.

        A new thread is created each time so the writer can be
        stopped and restarted safely.  Protected by a lock so
        concurrent callers cannot create duplicate threads.
        """
        with self._lock:
            if not self._started:
                self._thread = threading.Thread(target=self._drain, daemon=True, name="ResultsWriter")
                self._thread.start()
                self._started = True

    def enqueue(self, row: dict[str, Any]) -> None:
        """Add a row to the write queue.

        Parameters
        ----------
        row : dict[str, Any]
            Result row to persist.
        """
        if not self._started:
            self.start()
        self._queue.put(row)

    def stop(self) -> None:
        """Signal the writer thread to finish and wait for it to drain."""
        with self._lock:
            if self._started:
                self._queue.put(_SENTINEL)
                if self._thread is not None:
                    self._thread.join()
                self._started = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _drain(self) -> None:
        """Background loop: read from queue, write to sink."""
        while True:
            row = self._queue.get()
            if isinstance(row, _StopSignal):
                # Drain remaining items
                while not self._queue.empty():
                    remaining = self._queue.get_nowait()
                    if not isinstance(remaining, _StopSignal):
                        self.sink.write_row(remaining)
                self.sink.flush()
                break
            self.sink.write_row(row)
