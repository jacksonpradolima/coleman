"""Optional generic-only MCP host over the same bounded task command service."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from coleman.application.task_cli import open_task_service
from coleman.application.task_commands import TaskCommands
from coleman.application.tasks import TaskService

if TYPE_CHECKING:
    from mcp.server import MCPServer


def create_task_server(service: TaskService) -> MCPServer:
    """Bind opt-in generic tools without importing native CI or choosing a model.

    Parameters
    ----------
    service : TaskService
        Independently configured host-owned facade and principal. Current
        bootstrap approval is supplied by the host, never tool JSON.

    Returns
    -------
    mcp.server.MCPServer
        Optional pinned SDK server with only the nine distinct task tools. The
        existing application-v1/workflow-v1 tool catalogues remain unchanged.

    Raises
    ------
    ImportError
        The optional MCP SDK is missing. Base generic library/CLI use does not
        require MCP, Polars, the CI extra, an LLM or any remote service.

    Notes
    -----
    Loads the fixed optional SDK only when explicitly requested. This local
    stdio host is one trusted process principal, not a multiuser security service.
    All tools use the same strict 1 MiB command/reply edge as the task CLI.
    """
    try:
        sdk = import_module("mcp.server")
    except ModuleNotFoundError as error:
        if error.name != "mcp":
            raise
        raise ImportError("Install the reviewed COLEMAN wheel with its [mcp] extra") from error
    server = sdk.MCPServer("COLEMAN adaptive tasks")
    commands = TaskCommands(service)

    @server.tool()
    def task_discover() -> str:
        """List current operator-approved task recipes/codecs and finite generic caps."""
        return commands.invoke("discover")

    @server.tool()
    def task_plan(request_json: str) -> str:
        """Validate complete planning_request JSON under an already approved recipe."""
        return commands.invoke("plan", request_json)

    @server.tool()
    def task_execute(request_json: str) -> str:
        """Admit execute_request JSON with saved plan_id and retained request_id."""
        return commands.invoke("execute", request_json)

    @server.tool()
    def task_status(command_json: str) -> str:
        """Inspect job_command JSON containing an owned job_id; no provider loads."""
        return commands.invoke("status", command_json)

    @server.tool()
    def task_cancel(command_json: str) -> str:
        """Cancel an owned job; repeated terminal cancellation preserves its result."""
        return commands.invoke("cancel", command_json)

    @server.tool()
    def task_resume(command_json: str) -> str:
        """Explicitly continue an interrupted owned job after current image checks."""
        return commands.invoke("resume", command_json)

    @server.tool()
    def task_report(command_json: str) -> str:
        """Read complete typed outer evidence; a partial directory is not a result."""
        return commands.invoke("report", command_json)

    @server.tool()
    def task_reconnect() -> str:
        """Reconcile owned task snapshots and ensure the current approved coordinator."""
        return commands.invoke("reconnect")

    @server.tool()
    def task_cleanup() -> str:
        """Expire eligible owned terminal bytes; retained request keys are not recycled."""
        return commands.invoke("cleanup")

    return server


def main() -> None:
    """Start generic local stdio; only MCP protocol traffic is written to stdout."""
    parser = argparse.ArgumentParser(description="COLEMAN optional generic local MCP stdio host")
    parser.add_argument("--workspace", required=True, type=Path, help="Trusted persistent local workspace")
    parser.add_argument("--bootstrap", type=Path, help="Trusted operator approval file, never a tool argument")
    parser.add_argument("--owner", default="local-workspace", help="Trusted local process principal")
    args = parser.parse_args()
    service = open_task_service(args.workspace, bootstrap=args.bootstrap, owner=args.owner)
    create_task_server(service).run(transport="stdio")


if __name__ == "__main__":
    main()
