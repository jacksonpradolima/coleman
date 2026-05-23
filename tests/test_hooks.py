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
