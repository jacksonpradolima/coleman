"""Bounded serialization edge shared by the opt-in task CLI and MCP tools."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal, get_args

from pydantic import ValidationError

from coleman.application.tasks import TaskService
from coleman.core.reading import bounded_json_bytes
from coleman.tasks._base import TaskContract
from coleman.tasks.requests import (
    TaskCleanup,
    TaskExecuteRequest,
    TaskFailure,
    TaskJobCommand,
    TaskJobList,
    TaskPlanningRequest,
)
from coleman.workflows.contracts import WorkflowError

TaskOperation = Literal["discover", "plan", "execute", "status", "cancel", "resume", "report", "reconnect", "cleanup"]
OPERATIONS: tuple[TaskOperation, ...] = get_args(TaskOperation)
MAXIMUM_REQUEST_BYTES = 1024 * 1024
MAXIMUM_REPLY_BYTES = 1024 * 1024


@dataclass(frozen=True)
class TaskCommands:
    """Decode concrete known requests and invoke the same neutral library facade.

    Parameters
    ----------
    service : TaskService
        Host-owned service with trusted principal and independently supplied
        bootstrap approval. Neither is accepted from command JSON.

    Notes
    -----
    Wire requests/replies have a separate 1 MiB interactive ceiling. The library
    retains its explicit larger artifact ceilings. No transport discovers code,
    runs shell commands or interprets an imported artifact as approval.
    """

    service: TaskService

    def invoke(self, operation: TaskOperation, payload: str | None = None) -> str:
        """Return one known result or typed expected failure as canonical JSON.

        Parameters
        ----------
        operation : str
            Exact declared operation; unknown names reject as invalid input.
        payload : str or None, optional
            Complete strict versioned request, at most 1 MiB. Metadata-only
            operations accept no payload rather than ignoring supplied fields.

        Returns
        -------
        str
            Bounded concrete known reply or TaskFailure, with no exception trace,
            provider import pointer or arbitrary server path.

        Notes
        -----
        Unexpected programming errors propagate. Only declared boundary/storage
        failures are translated. Oversized replies refuse instead of truncating
        evidence or presenting a partial body as complete.
        """
        try:
            value = self._invoke(operation, payload)
            body = value.to_json()
            bounded_json_bytes(body, MAXIMUM_REPLY_BYTES)
            return body
        except ValidationError:
            error = "invalid_input"
        except KeyError:
            error = "not_found"
        except WorkflowError:
            error = "conflict_or_limit"
        except ValueError:
            error = "invalid_input"
        except (OSError, sqlite3.Error):
            error = "storage_unavailable"
        return TaskFailure(
            schema_version=1, contract="coleman-tasks-v1", artifact_type="failure", error=error
        ).to_json()

    def _invoke(self, operation: TaskOperation, payload: str | None) -> TaskContract:
        """Route only known concrete values at this bounded serialization edge."""
        if operation not in OPERATIONS:
            raise ValueError("Unknown task operation")
        if operation in ("discover", "reconnect", "cleanup"):
            if payload is not None:
                raise ValueError("This metadata operation accepts no request payload")
            return self._metadata(operation)
        if payload is None:
            raise ValueError("A versioned task request is required")
        bounded_json_bytes(payload, MAXIMUM_REQUEST_BYTES)
        if operation == "plan":
            return self.service.plan(TaskPlanningRequest.from_json(payload))
        if operation == "execute":
            return self.service.execute(TaskExecuteRequest.from_json(payload))
        request = TaskJobCommand.from_json(payload)
        operations = {
            "status": self.service.status,
            "cancel": self.service.cancel,
            "resume": self.service.resume,
            "report": self.service.report,
        }
        return operations[operation](request.job_id)

    def _metadata(self, operation: TaskOperation) -> TaskContract:
        """Keep known metadata replies typed without provider or command factories."""
        if operation == "discover":
            return self.service.discover()
        if operation == "reconnect":
            return TaskJobList(
                schema_version=1, contract="coleman-tasks-v1", artifact_type="job_list", jobs=self.service.reconnect()
            )
        return TaskCleanup(
            schema_version=1, contract="coleman-tasks-v1", artifact_type="cleanup", expired_jobs=self.service.cleanup()
        )
