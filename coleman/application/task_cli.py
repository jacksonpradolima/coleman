"""Opt-in generic task console host; native command grammars are unchanged."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coleman.application.task_commands import MAXIMUM_REQUEST_BYTES, OPERATIONS, TaskCommands
from coleman.application.tasks import TaskService
from coleman.tasks.identity import HostBootstrapProfile


def _startup_path(value: Path, *, directory: Path) -> Path:
    """Resolve a startup path inside the independently inherited host directory.

    Parameters
    ----------
    value : pathlib.Path
        Absolute or relative host argument; existing symlinks are resolved.
    directory : pathlib.Path
        Canonical inherited current directory, never another command argument.

    Returns
    -------
    pathlib.Path
        Canonical path at or below the host directory, including missing leaves.

    Raises
    ------
    ValueError
        The canonical path escapes the inherited directory namespace.
    OSError
        The path cannot be resolved in the trusted local filesystem.

    Notes
    -----
    This namespace guard assumes the documented trusted filesystem. It does not
    provide race-free confinement against concurrent hostile filesystem changes.
    """
    selected = value.resolve()
    if not selected.is_relative_to(directory):
        raise ValueError("Task host startup paths must remain within the inherited current directory")
    return selected


def open_task_service(workspace: Path, *, bootstrap: Path | None, owner: str) -> TaskService:
    """Open explicit operator setup before accepting task command JSON.

    Parameters
    ----------
    workspace : pathlib.Path
        Host workspace within the inherited current directory after canonical
        resolution, never selected by tool JSON.
    bootstrap : pathlib.Path or None
        Reviewed small approval file within that same startup namespace, or
        provider-free inspection mode. Contents are bounded at 16 KiB before
        decoding, never loaded from workspace settings or command JSON.
    owner : str
        Host-supplied local security principal, independent of per-request fields.

    Returns
    -------
    TaskService
        Explicit common facade. No package installation, source acquisition,
        learner construction, preparation or mathematical update occurs.

    Raises
    ------
    ValueError, OSError
        Invalid/conflicting current approval, unsafe storage or unavailable file.

    Notes
    -----
    Sets the host process's no-bytecode mode before capture and provider loading.
    The operator still selects a trusted immutable directory installation with
    optimization zero and standard import context, as the capture profile checks.
    Start with Python -B or PYTHONDONTWRITEBYTECODE=1: setting this flag here
    cannot undo caches already created by earlier console startup imports.
    Existing native console commands do not change their interpreter settings.
    Both paths resolve against the independently inherited current directory;
    absolute outsiders and symlinks escaping it refuse before reading approval
    contents or creating the workspace. Library TaskService roots retain their
    existing trusted-operator contract.
    """
    sys.dont_write_bytecode = True

    directory = Path.cwd().resolve()
    workspace = _startup_path(workspace, directory=directory)
    bootstrap = None if bootstrap is None else _startup_path(bootstrap, directory=directory)
    profile = None
    if bootstrap is not None:
        with bootstrap.open("rb") as stream:
            profile = HostBootstrapProfile.from_json(stream.read(16 * 1024 + 1))
    return TaskService(workspace, owner=owner, bootstrap_profile=profile)


def main() -> None:
    """Run one bounded task command; generic execution stays in shared services."""
    parser = argparse.ArgumentParser(description="COLEMAN opt-in trusted generic task host")
    parser.add_argument("--workspace", required=True, type=Path, help="Trusted persistent local workspace")
    parser.add_argument("--bootstrap", type=Path, help="Trusted operator approval file; never a tool argument")
    parser.add_argument("--owner", default="local-workspace", help="Trusted local process principal")
    parser.add_argument(
        "operation",
        choices=OPERATIONS,
    )
    parser.add_argument("--request", help="Complete versioned JSON, or '-' to read at most 1 MiB from stdin")
    args = parser.parse_args()
    service = open_task_service(args.workspace, bootstrap=args.bootstrap, owner=args.owner)
    payload = args.request
    if payload == "-":
        payload = sys.stdin.buffer.read(MAXIMUM_REQUEST_BYTES + 1).decode("utf-8")
    print(TaskCommands(service).invoke(args.operation, payload))


if __name__ == "__main__":
    main()
