"""Optional MCP 2.3 local stdio adapter; no HTTP hosting or model dependency."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from coleman.application import ApplicationService, Caller, Submit, default_registry
from coleman.application.commands import ApplicationCommands
from coleman.results.domain import canonical_json

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from coleman.application.workflows import WorkflowService


def create_server(
    service: ApplicationService, *, caller: Caller, workflows: WorkflowService | None = None
) -> MCPServer:
    """Bind seven local tools to the shared service and trusted process-session identity.

    Parameters
    ----------
    service : ApplicationService
        Existing ephemeral application-v1 service; its lifetime remains unchanged.
    caller : Caller
        Trusted host identity for the existing seven tools.
    workflows : WorkflowService or None
        Optional persistent local workflow service with its own trusted principal.
        None preserves the original exact seven-tool interface.

    Returns
    -------
    mcp.server.MCPServer
        Optional SDK server with typed durable tools only when explicitly enabled.

    Raises
    ------
    ImportError
        Optional pinned MCP SDK is not installed.

    Notes
    -----
    Imports the optional pinned SDK only here. Caller identity is host-supplied;
    tools accept no owner, paths, URLs, plugin imports or shell commands. A single
    stdio process is one local security principal, not a multi-user hosted server.
    """
    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise ImportError(
            "Install the reviewed COLEMAN wheel with its [mcp] extra; see docs/application-api.md"
        ) from exc

    server = MCPServer("COLEMAN CI Intelligence")
    commands = ApplicationCommands(service, caller=caller)

    @server.tool()
    def discover() -> str:
        """List approved CI decision configurations, limits and the versioned submit JSON schema."""
        return canonical_json(
            {**service.capabilities().model_dump(mode="json"), "submit_schema": Submit.model_json_schema()}
        )

    @server.tool()
    def submit(request_json: str) -> str:
        """Submit versioned JSON using a discovered spec_id/request_id and optional typed parameters/limits."""
        return commands.invoke("submit", request_json)

    @server.tool()
    def status(command_json: str) -> str:
        """Inspect JSON {schema_version:1, job_id:...}; poll until a terminal state and result handle."""
        return commands.invoke("status", command_json)

    @server.tool()
    def cancel(command_json: str) -> str:
        """Cancel caller-owned queued/running work; terminal cancellation retries are idempotent."""
        return commands.invoke("cancel", command_json)

    @server.tool()
    def result(handle_json: str) -> str:
        """Retrieve the completed native result using its versioned job_id/digest handle."""
        return commands.invoke("result", handle_json)

    @server.tool()
    def export_result(handle_json: str) -> str:
        """Export a validated portable result bundle; copy the response data as portable JSON."""
        return commands.invoke("export", handle_json)

    @server.tool()
    def import_result(bundle_json: str) -> str:
        """Import bounded portable evidence and issue a new caller-owned handle; never execute imported content."""
        return commands.invoke("import", bundle_json)

    if workflows is not None:
        from coleman.transports.workflow_tools import register_workflow_tools

        register_workflow_tools(server, workflows)
    return server


def main() -> None:
    """Start local stdio tools; stdout is exclusively MCP protocol traffic."""
    parser = argparse.ArgumentParser(description="COLEMAN optional local MCP stdio tools")
    parser.add_argument(
        "--workspace", type=Path, help="Trusted persistent workflow workspace; initialize using coleman workflow"
    )
    args = parser.parse_args()
    workflows = None
    if args.workspace is not None:
        from coleman.application.workflows import WorkflowService

        workflows = WorkflowService(args.workspace)
        workflows.reconnect()
    with (
        tempfile.TemporaryDirectory(prefix="coleman-mcp-") as directory,
        ApplicationService(default_registry(), root=Path(directory)) as service,
    ):
        try:
            create_server(service, caller=Caller(subject="local-stdio"), workflows=workflows).run(transport="stdio")
        except ImportError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
