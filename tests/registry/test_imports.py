"""Static transport/domain boundary complements the installed cold-import proof."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PURE = {
    "__init__",
    "definitions",
    "limits",
    "lifecycle",
    "_transitions",
    "envelopes",
    "ownership",
    "composition",
    "_checkpoint",
}
FORBIDDEN = (
    "coleman.cli",
    "coleman.transports",
    "coleman.application",
    "coleman.workflows",
    "coleman.runner",
    "coleman.native",
    "coleman.agent",
    "coleman.policy",
    "coleman.telemetry",
    "mcp",
    "openai",
    "anthropic",
)
OUTER = (
    "coleman.state.components",
    "coleman.state.local",
    "coleman.replay",
    "coleman.spec",
    "coleman.artifacts",
    "sqlite3",
)


@pytest.mark.parametrize("source", sorted((ROOT / "coleman/registry").glob("*.py")), ids=lambda path: path.name)
def test_registry_has_no_cli_mcp_native_or_provider_dependency_and_pure_leaves_avoid_outer_state(source):
    imported = []
    for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "Make each dependency boundary explicit"
            imported.append(node.module or "")
    blocked = FORBIDDEN + (OUTER if source.stem in PURE else ())
    assert not [name for name in imported if any(name == item or name.startswith(item + ".") for item in blocked)]
