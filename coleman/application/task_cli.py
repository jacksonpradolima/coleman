"""Opt-in generic task console host; native command grammars are unchanged."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coleman.application.task_commands import MAXIMUM_REQUEST_BYTES, OPERATIONS, TaskCommands
from coleman.application.tasks import TaskService
from coleman.tasks.identity import HostBootstrapProfile


def open_task_service(workspace: Path, *, bootstrap: Path | None, owner: str) -> TaskService:
    """Open explicit operator setup before accepting task command JSON.

    Parameters
    ----------
    workspace : pathlib.Path
        Trusted host-owned local workspace, never selected by tool JSON.
    bootstrap : pathlib.Path or None
        Trusted reviewed small approval file, or provider-free inspection mode.
        File contents are bounded at 16 KiB before decoding; they are not loaded
        from workspace settings and never contain a command supplied by a client.
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
    """
    sys.dont_write_bytecode = True

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
