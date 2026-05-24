"""Tests for lifecycle hook infrastructure."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from coleman.hooks import HookContext, dispatch_hook_event, load_hook_plugins


def test_load_hook_plugins_supports_class_and_function():
    hooks = load_hook_plugins(
        [
            "tests.support.hook_plugins.RecordingHook",
            "tests.support.hook_plugins.functional_hook",
        ]
    )
    assert len(hooks) == 2


def test_dispatch_hook_event_calls_method_hooks():
    hook = Mock()
    ctx = HookContext(run_id="rid-1")

    dispatch_hook_event([hook], "on_run_start", ctx)

    hook.on_run_start.assert_called_once_with(ctx)


def test_dispatch_hook_event_fail_fast_false_continues():
    bad = Mock()
    bad.on_run_start.side_effect = RuntimeError("boom")
    good = Mock()
    ctx = HookContext(run_id="rid-1")

    dispatch_hook_event([bad, good], "on_run_start", ctx, fail_fast=False)

    good.on_run_start.assert_called_once_with(ctx)


def test_dispatch_hook_event_fail_fast_true_raises():
    bad = Mock()
    bad.on_run_start.side_effect = RuntimeError("boom")
    ctx = HookContext(run_id="rid-1")

    with pytest.raises(RuntimeError, match="boom"):
        dispatch_hook_event([bad], "on_run_start", ctx, fail_fast=True)


def test_load_hook_plugin_class_requires_no_args_message():
    with pytest.raises(ValueError, match="must be instantiable without arguments"):
        load_hook_plugins(["tests.support.hook_plugins.NeedsArgsHook"])


def test_load_hook_plugin_invalid_path_and_missing_symbol():
    from coleman.hooks import _load_hook_plugin

    with pytest.raises(ValueError, match="expected dotted path"):
        _load_hook_plugin("NotADottedPath")

    with pytest.raises(ValueError, match="not found in module"):
        _load_hook_plugin("tests.support.hook_plugins.DoesNotExist")


def test_load_hook_plugin_rejects_non_callable_symbol(monkeypatch):
    import tests.support.hook_plugins as hook_plugins
    from coleman.hooks import _load_hook_plugin

    monkeypatch.setattr(hook_plugins, "NOT_A_HOOK", 123, raising=False)

    with pytest.raises(ValueError, match="neither a class nor a callable"):
        _load_hook_plugin("tests.support.hook_plugins.NOT_A_HOOK")


def test_dispatch_hook_event_uses_function_adapter():
    hooks = load_hook_plugins(["tests.support.hook_plugins.functional_hook"])
    ctx = HookContext(run_id="rid-2")

    dispatch_hook_event(hooks, "on_run_start", ctx)

    from tests.support.hook_plugins import EVENTS

    assert ("on_run_start", "rid-2") in EVENTS


def test_dispatch_hook_event_uses_handle_fallback_when_method_missing():
    from coleman.hooks import FunctionHookAdapter

    calls = []

    class HandleOnlyHook:
        def handle(self, event_name, context, payload=None):
            calls.append((event_name, context.run_id, payload))

    def record_call(event_name, context, payload=None):
        calls.append((event_name, context.run_id, payload))

    ctx = HookContext(run_id="rid-3")
    dispatch_hook_event([HandleOnlyHook()], "on_run_end", ctx, payload={"status": "ok"})
    dispatch_hook_event([FunctionHookAdapter(record_call)], "on_dataset_end", ctx)

    assert ("on_run_end", "rid-3", {"status": "ok"}) in calls
    assert ("on_dataset_end", "rid-3", None) in calls
