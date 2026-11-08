"""Current operator approval and persisted equality constraints for task hosts."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Literal

from pydantic import Field

from coleman.application.workflow_store import WorkflowStore
from coleman.objectives.bindings import DefinitionBinding
from coleman.tasks._base import TaskContract
from coleman.tasks.contracts import TaskJobRecord, TaskPlan
from coleman.tasks.identity import HostBootstrapProfile
from coleman.tasks.requests import TaskHostPolicy

_PROFILE = "task_bootstrap_v1"
_POLICY = "task_policy_v1"


class _ProfileObservation(TaskContract):
    """Bind one scheduler queue observation to its actually supplied approval."""

    artifact_type: Literal["profile_observation"]
    generation: Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
    bootstrap: DefinitionBinding
    _maximum_bytes = 16 * 1024


def acknowledge_profile(connection: sqlite3.Connection, current: HostBootstrapProfile, generation: int) -> None:
    """Retain a current approval observation beside the unchanged v1 counters.

    Parameters
    ----------
    connection : sqlite3.Connection
        Scheduler's existing queue-observation transaction.
    current : HostBootstrapProfile
        Actual supplied profile after queued-task validation, never SQL authority.
    generation : int
        Actual requested scheduler generation, unrelated to learner work.

    Notes
    -----
    This is metadata acknowledgement only. The inherited descriptor separately
    establishes ownership, and workers independently verify the installed image.
    """
    value = _ProfileObservation(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="profile_observation",
        generation=generation,
        bootstrap=current.binding,
    )
    connection.execute(
        "INSERT INTO settings(name,payload) VALUES('task_profile_observed_v1',?) "
        "ON CONFLICT(name) DO UPDATE SET payload=excluded.payload",
        (value.to_json(),),
    )


def profile_observed(connection: sqlite3.Connection, current: HostBootstrapProfile, generation: int) -> bool:
    """Check transactional observation of the exact current approval generation.

    Parameters
    ----------
    connection : sqlite3.Connection
        Caller-owned consistent metadata read connection.
    current : HostBootstrapProfile
        Actual requesting operator approval, not a retained replacement.
    generation : int
        Requested queue generation from the existing bounded handshake.

    Returns
    -------
    bool
        True only for this profile and an observation at least this generation.
        A busy lock, old observation or another profile is not acknowledgement.
    """
    row = connection.execute("SELECT payload FROM settings WHERE name='task_profile_observed_v1'").fetchone()
    if row is None:
        return False
    value = _ProfileObservation.from_json(row[0])
    return value.bootstrap == current.binding and value.generation >= generation


def stored_profile(connection: sqlite3.Connection) -> HostBootstrapProfile | None:
    """Read an equality constraint without treating stored data as approval.

    Parameters
    ----------
    connection : sqlite3.Connection
        Caller-owned consistent metadata connection, with no provider callback.

    Returns
    -------
    HostBootstrapProfile or None
        Pure bounded retained declaration, or no generic registration yet.

    Notes
    -----
    Only an independently supplied current profile can authorize loading. This
    function performs no provider imports or implementation capture.
    """
    row = connection.execute("SELECT payload FROM settings WHERE name=?", (_PROFILE,)).fetchone()
    return None if row is None else HostBootstrapProfile.from_json(row[0])


def stored_policy(connection: sqlite3.Connection) -> TaskHostPolicy | None:
    """Read bounded generic caps separately from unchanged native HostPolicy.

    Parameters
    ----------
    connection : sqlite3.Connection
        Caller-owned consistent metadata connection.

    Returns
    -------
    TaskHostPolicy or None
        Immutable generic ceilings, or no generic host initialization yet.
    """
    row = connection.execute("SELECT payload FROM settings WHERE name=?", (_POLICY,)).fetchone()
    return None if row is None else TaskHostPolicy.from_json(row[0])


def require_current_profile(connection: sqlite3.Connection, current: HostBootstrapProfile | None) -> None:
    """Require actual supplied approval before scheduling any queued task.

    Parameters
    ----------
    connection : sqlite3.Connection
        Caller-owned metadata transaction; no provider is loaded here.
    current : HostBootstrapProfile or None
        Independently supplied current operator profile, never restored from SQL.

    Raises
    ------
    ValueError
        Missing registration, absent current approval or mismatched profile.
        Persisted declaration equality constrains approval; it cannot grant it.
    """
    retained = stored_profile(connection)
    if current is None or retained is None or HostBootstrapProfile.model_validate(current) != retained:
        raise ValueError("Generic scheduling requires the matching explicit current host bootstrap profile")


def bind_host(
    store: WorkflowStore, current: HostBootstrapProfile | None, requested: TaskHostPolicy | None
) -> TaskHostPolicy:
    """Serialize first generic-host equality registration without loading code.

    Parameters
    ----------
    store : WorkflowStore
        Existing five-table metadata store; no new database/schema is created.
    current : HostBootstrapProfile or None
        Explicit current operator approval, or provider-free inspection mode.
    requested : TaskHostPolicy or None
        Initial generic caps; conflicting explicit reopen policy rejects.

    Returns
    -------
    TaskHostPolicy
        Retained or initial caps. Provider-free access never installs authority.

    Raises
    ------
    ValueError
        Concurrent/conflicting approval or changed immutable policy. Both
        settings are committed together; a first-profile race has one winner.

    Notes
    -----
    Registration is equality metadata only. Capture and concrete bootstrap
    validation occur separately outside SQL before planning, admission or work.
    A failed subsequent capture does not silently replace an immutable profile.
    """
    profile = None if current is None else HostBootstrapProfile.model_validate(current)
    policy = None if requested is None else TaskHostPolicy.model_validate(requested)
    with store.transaction() as connection:
        retained_profile, retained_policy = stored_profile(connection), stored_policy(connection)
        if (retained_profile is None) != (retained_policy is None):
            raise ValueError("Incomplete generic host equality metadata; no implicit repair")
        if retained_policy is not None:
            if policy is not None and policy != retained_policy:
                raise ValueError("Existing generic host policy cannot be silently replaced")
            if profile is not None and profile != retained_profile:
                raise ValueError("Existing generic bootstrap profile cannot be silently replaced")
            return retained_policy
        selected = policy or TaskHostPolicy(schema_version=1, contract="coleman-tasks-v1", artifact_type="host_policy")
        if profile is not None:
            connection.execute("INSERT INTO settings VALUES (?,?)", (_PROFILE, profile.to_json()))
            connection.execute("INSERT INTO settings VALUES (?,?)", (_POLICY, selected.to_json()))
        return selected


def validate_task_caps(connection: sqlite3.Connection, plan: TaskPlan) -> None:
    """Check immutable generic ceilings using only known pure metadata.

    Parameters
    ----------
    connection : sqlite3.Connection
        Caller-owned consistent metadata transaction.
    plan : TaskPlan
        Complete already validated task with explicit named work semantics.

    Raises
    ------
    ValueError
        Missing host policy or a request exceeds any generic cap. Native build
        capacities retain their original separate interpretation.
    """
    policy = stored_policy(connection)
    if policy is None:
        raise ValueError("Generic task host policy is not registered")
    for name, value in plan.limits.model_dump().items():
        if value > getattr(policy.caps, name):
            raise ValueError(f"Task planning limit exceeds immutable generic host cap: {name}")


def validate_queued_tasks(
    store: WorkflowStore, connection: sqlite3.Connection, current: HostBootstrapProfile | None
) -> None:
    """Check all queued task approvals before the scheduler claims any work.

    Parameters
    ----------
    store, connection : WorkflowStore, sqlite3.Connection
        Existing common metadata owner and serialized queue transaction.
    current : HostBootstrapProfile or None
        Actual operator approval carried through fixed internal startup argv.

    Raises
    ------
    ValueError
        Any queued task lacks matching current approval, plan binding or caps.
        No provider bootstrap, source read or worker spawn occurs in this check.
    """
    for job in store.records(connection):
        if isinstance(job, TaskJobRecord) and job.state == "queued":
            require_current_profile(connection, current)
            plan = store.read_plan(connection, job.plan_id, job.owner)
            if not isinstance(plan, TaskPlan):
                raise ValueError("Queued task requires its exact task plan family")
            job.validate_for(plan)
            if current is None:
                raise ValueError("Queued task has no current bootstrap approval")
            plan.validate_bootstrap(current)
            validate_task_caps(connection, plan)
