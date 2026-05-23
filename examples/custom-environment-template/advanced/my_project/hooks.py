"""Example hook plugins for custom artifact generation."""

from __future__ import annotations

from dataclasses import asdict

from coleman.artifacts import ArtifactWriter
from coleman.hooks import ExecutionResult, HookContext


class ForecastHook:
    """Persist one execution artifact with deterministic path layout."""

    def on_execution_end(self, context: HookContext, payload: ExecutionResult) -> None:
        if context.run_id is None:
            return

        writer = ArtifactWriter("./runs")
        artifact_path = writer.path_for(
            run_id=context.run_id,
            dataset_id=context.dataset_id,
            budget_mode=context.budget_mode.value if context.budget_mode else None,
            budget_value=context.budget_value,
            execution_id=context.execution_id,
            artifact_type="forecast",
            ext="json",
            stem="execution-summary",
        )

        payload_dict = {
            "context": asdict(context),
            "result": asdict(payload),
        }
        writer.write_json_atomic(artifact_path, payload_dict)


def audit_hook(event_name: str, context: HookContext, payload=None) -> None:
    """Minimal function-style hook for audit trails."""
    if context.run_id is None:
        return

    writer = ArtifactWriter("./runs")
    path = writer.path_for(
        run_id=context.run_id,
        dataset_id=context.dataset_id,
        budget_mode=context.budget_mode.value if context.budget_mode else None,
        budget_value=context.budget_value,
        execution_id=context.execution_id,
        artifact_type="audit",
        ext="csv",
        stem="events",
    )

    rows = [
        {
            "event_name": event_name,
            "run_id": context.run_id,
            "dataset_id": context.dataset_id,
            "execution_id": context.execution_id,
            "worker_id": context.worker_id,
            "parallel_mode": context.parallel_mode,
        }
    ]

    # Keep example simple and append-like by rewriting one-row CSV per event.
    # Real projects can batch rows in memory and flush on on_run_end.
    writer.write_csv_atomic(path, rows)
