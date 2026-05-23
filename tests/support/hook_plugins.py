"""Test hook plugins used by hooks module tests."""

from __future__ import annotations

EVENTS: list[tuple[str, str | None]] = []


class RecordingHook:
    def on_run_start(self, context):
        EVENTS.append(("on_run_start", context.run_id))

    def on_run_end(self, context, run_result):
        del run_result
        EVENTS.append(("on_run_end", context.run_id))


def functional_hook(event_name, context, payload=None):
    del payload
    EVENTS.append((event_name, context.run_id))


class NeedsArgsHook:
    def __init__(self, required: str):
        self.required = required
