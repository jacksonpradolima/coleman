"""coleman.hooks - Runner lifecycle hook support.

This module provides a lightweight plugin system for runner lifecycle events
with deterministic ordering and configurable error handling.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from coleman.budget import BudgetMode


@dataclass(frozen=True)
class HookContext:
    """Immutable context passed to hooks."""

    run_id: str | None = None
    dataset_id: str | None = None
    execution_id: str | None = None
    worker_id: str | None = None
    parallel_mode: str | None = None
    iteration: int | None = None
    trials: int | None = None
    budget_mode: BudgetMode | None = None
    budget_value: float | None = None
    extensions: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExecutionResult:
    """Execution-level hook payload."""

    status: str
    duration_seconds: float


@dataclass(frozen=True)
class DatasetResult:
    """Dataset-level hook payload."""

    status: str
    executions: int
    duration_seconds: float


@dataclass(frozen=True)
class RunResult:
    """Run-level hook payload."""

    status: str
    datasets: int
    duration_seconds: float


class RunnerHook(Protocol):
    """Minimal hook protocol used by the dispatcher."""

    def handle(self, event_name: str, context: HookContext, payload: Any | None = None) -> None:
        """Handle generic event fallback when typed methods are not implemented."""
        ...


@dataclass(frozen=True)
class FunctionHookAdapter:
    """Adapter for functional hooks.

    The wrapped function must accept ``(event_name, context, payload=None)``.
    """

    func: Callable[[str, HookContext, Any | None], Any]

    def handle(self, event_name: str, context: HookContext, payload: Any | None = None) -> None:
        """Delegate event handling to wrapped functional hook."""
        self.func(event_name, context, payload)


def _load_hook_plugin(path: str) -> RunnerHook:
    """Load one hook plugin by dotted path.

    Dotted path examples:
    - ``my_project.hooks.MyHook`` (class)
    - ``my_project.hooks.my_hook`` (function)
    """
    if "." not in path:
        msg = f"Invalid hook path {path!r}; expected dotted path 'module.symbol'."
        raise ValueError(msg)

    module_name, symbol_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)

    if not hasattr(module, symbol_name):
        msg = f"Hook symbol {symbol_name!r} not found in module {module_name!r}."
        raise ValueError(msg)

    symbol = getattr(module, symbol_name)

    if inspect.isclass(symbol):
        try:
            return cast(RunnerHook, symbol())
        except TypeError as exc:
            msg = f"Hook class {path!r} must be instantiable without arguments (constructor signature is incompatible)."
            raise ValueError(msg) from exc

    if callable(symbol):
        return FunctionHookAdapter(func=cast(Callable[[str, HookContext, Any | None], Any], symbol))

    msg = f"Hook symbol {path!r} is neither a class nor a callable function."
    raise ValueError(msg)


def load_hook_plugins(paths: list[str]) -> list[RunnerHook]:
    """Load hook plugins in deterministic order."""
    return [_load_hook_plugin(path) for path in paths]


def dispatch_hook_event(
    hooks: list[RunnerHook],
    event_name: str,
    context: HookContext,
    payload: Any | None = None,
    *,
    fail_fast: bool = True,
) -> None:
    """Dispatch one lifecycle event to all hooks.

    If ``fail_fast`` is ``False``, hook errors are logged and execution
    continues.
    """
    for hook in hooks:
        try:
            method = getattr(hook, event_name, None)
            if callable(method):
                if payload is None:
                    method(context)
                else:
                    method(context, payload)
                continue

            handle = getattr(hook, "handle", None)
            if callable(handle):
                handle(event_name, context, payload)
        except Exception:  # noqa: BLE001
            logging.exception(
                ("Hook failure event=%s run_id=%s dataset_id=%s execution_id=%s worker_id=%s"),
                event_name,
                context.run_id,
                context.dataset_id,
                context.execution_id,
                context.worker_id,
            )
            if fail_fast:
                raise
