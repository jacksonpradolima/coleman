"""Known metadata unions for the one workflow store, without provider loading."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

from coleman.core.primitives import decode_json
from coleman.core.reading import bounded_json_bytes
from coleman.tasks._base import MAXIMUM_BYTES
from coleman.tasks.contracts import TaskJobRecord, TaskPlan
from coleman.workflows.contracts import ExperimentPlan, WorkflowContract

if TYPE_CHECKING:
    from coleman.application.workflow_store import JobRecord

ROUTING_MAXIMUM_BYTES = 64 * 1024**2


def _route(payload: str | bytes) -> tuple[dict[str, Any], bool]:
    """Bound shared routing before choosing one exact known namespace/version."""
    wire = bounded_json_bytes(payload, ROUTING_MAXIMUM_BYTES)
    value = decode_json(wire, exact_decimals=True)
    if not isinstance(value, dict):
        raise ValueError("Saved execution metadata requires an object")
    version = value.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise ValueError("Saved execution metadata requires integer schema_version 1")
    namespace = value.get("contract", "coleman-workflows-v1")
    if not isinstance(namespace, str) or namespace not in {"coleman-workflows-v1", "coleman-tasks-v1"}:
        raise ValueError("Unknown saved execution metadata namespace")
    task = namespace == "coleman-tasks-v1"
    if task and len(wire) > MAXIMUM_BYTES:
        raise ValueError("Task metadata payload exceeds its 16 MiB typed-reader ceiling")
    return value, task


def read_plan_payload(payload: str | bytes) -> ExperimentPlan | TaskPlan:
    """Read a complete known saved-plan arm without selecting executable code.

    Parameters
    ----------
    payload : str or bytes
        Complete metadata under the new shared 64 MiB/depth64 routing ceiling.
        The task arm additionally requires at most16 MiB before model decoding.

    Returns
    -------
    ExperimentPlan or TaskPlan
        Validated native or task metadata. Opaque task bodies remain subject to
        separately approved concrete codecs before execution.

    Raises
    ------
    ValueError, TypeError, UnicodeError
        Invalid bounds, duplicate keys, unknown namespace/version or plan shape.

    Notes
    -----
    No capture, provider import, factory, source access or restore occurs.
    Retained native facade readers keep their original unbounded wire behavior;
    this explicitly bounded mixed route does not migrate or rewrite them.
    """
    value, task = _route(payload)
    return TaskPlan.model_validate(value) if task else ExperimentPlan.model_validate(value)


def read_job_payload[N: WorkflowContract](payload: str | bytes, *, native_type: type[N]) -> N | TaskJobRecord:
    """Read one record using a known task arm or the trusted native model type.

    Parameters
    ----------
    payload : str or bytes
        Complete metadata with shared64 MiB/depth64 and task16 MiB ceilings.
    native_type : type of WorkflowContract
        The store's actual retained JobRecord class, supplied by trusted code
        to avoid moving that class or introducing an import cycle.

    Returns
    -------
    WorkflowContract or TaskJobRecord
        Known operational record only; no full plan or provider body is loaded.

    Raises
    ------
    ValueError, TypeError, UnicodeError
        Unknown/malformed namespace, version, bounds or record fields.
    """
    value, task = _route(payload)
    return TaskJobRecord.model_validate(value) if task else native_type.model_validate(value)


def read_plan(connection: sqlite3.Connection, plan_id: str, owner: str) -> ExperimentPlan | TaskPlan:
    """Load one owned bounded plan and verify its complete content handle.

    Parameters
    ----------
    connection : sqlite3.Connection
        Caller-owned consistent metadata connection; no transaction is started.
    plan_id, owner : str
        Full64-hex content handle and trusted owner, not an import selector.

    Returns
    -------
    ExperimentPlan or TaskPlan
        Exact retained known plan with its indexed identity checked.

    Raises
    ------
    KeyError, ValueError, TypeError, sqlite3.Error
        Missing/nonowned plan, corrupt bytes/key or unsupported metadata.
    """
    row = connection.execute("SELECT payload FROM plans WHERE id=? AND owner=?", (plan_id, owner)).fetchone()
    if row is None:
        raise KeyError("No owned workflow plan")
    plan = read_plan_payload(row[0])
    if plan.fingerprint.removeprefix("sha256:") != plan_id:
        raise ValueError("Saved plan identity mismatch")
    return plan


def require_record_plan(job: JobRecord | TaskJobRecord, plan: ExperimentPlan | TaskPlan) -> None:
    """Check a selected record/plan family and complete task frontier joins.

    Parameters
    ----------
    job : JobRecord or TaskJobRecord
        Known revalidated operational record, not arbitrary provider state.
    plan : ExperimentPlan or TaskPlan
        Actual retained selected plan. Call at claim/transition boundaries;
        hot record scans deliberately do not load every complete plan.

    Raises
    ------
    ValueError
        Mixed families, wrong plan identity or task stage/catalogue/frontier/caps.
        This pure check does not authenticate storage or approve a provider.
    """
    if isinstance(job, TaskJobRecord):
        if not isinstance(plan, TaskPlan):
            raise ValueError("Task record requires the task plan family")
        job.validate_for(plan)
    elif not isinstance(plan, ExperimentPlan):
        raise ValueError("Native record requires the native plan family")
    elif job.plan_id != plan.fingerprint.removeprefix("sha256:"):
        raise ValueError("Native record differs from its retained plan")


def write_record(
    connection: sqlite3.Connection, job: JobRecord | TaskJobRecord, *, create: bool, native_type: type[JobRecord]
) -> None:
    """Write one validated arm with the original indexed identity and SQL path.

    Parameters
    ----------
    connection : sqlite3.Connection
        Active caller-owned write transaction. Its body is never retried here.
    job : JobRecord or TaskJobRecord
        Complete validated record; the caller joins a task to its selected plan
        before this private writer. Existing rows must retain the same family.
    create : bool
        Insert a new owner/request identity or update the exact retained one.
    native_type : type of JobRecord
        Actual retained native class supplied by the store, without a back-edge
        import or namespace-selected executable lookup.

    Raises
    ------
    ValueError, TypeError, sqlite3.Error
        Missing/changed identity, wrong retained family or corrupt metadata.
        Known native record reading retains its original wire-size behavior.
    """
    identity = (job.job_id, job.owner, job.request_id, job.plan_id)
    if create:
        connection.execute("INSERT INTO jobs VALUES (?,?,?,?,?)", (*identity, job.to_json()))
        return
    condition = "WHERE id=? AND owner=? AND request=? AND plan=?"
    row = connection.execute("SELECT payload FROM jobs " + condition, identity).fetchone()
    if row is None:
        raise ValueError("Cannot replace workflow job identity")
    current = TaskJobRecord.from_json(row[0]) if isinstance(job, TaskJobRecord) else native_type.from_json(row[0])
    if (current.job_id, current.owner, current.request_id, current.plan_id) != identity:
        raise ValueError("Job metadata identity mismatch")
    cursor = connection.execute("UPDATE jobs SET payload=? " + condition, (job.to_json(), *identity))
    if cursor.rowcount != 1:
        raise ValueError("Cannot replace workflow job identity")
