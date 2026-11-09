"""Fixture-only offline bootstrap, run by the parent under one600-second fence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tests.installed_support import build_install_stdlib_wheel, seed_locked_dependencies

ROOT = Path(__file__).parents[2]


def _command(arguments: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    """Bootstrap one bounded owned artifact; never install into the root env."""
    result = subprocess.run(
        arguments, cwd=cwd, env=environment, capture_output=True, text=True, timeout=180, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def main() -> None:
    """Build exactly three actual wheels and one immutable base-only consumer."""
    owned = Path(sys.argv[1]).resolve()
    mode = sys.argv[2]
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(
        UV_OFFLINE="1", UV_NO_INDEX="1", UV_NO_PROGRESS="1", UV_LINK_MODE="copy", UV_PYTHON_DOWNLOADS="never"
    )
    if mode != "base":
        _optional(owned, uv, environment, mode)
        return
    virtual = owned / "base"
    core = build_install_stdlib_wheel(uv, owned / "dist", virtual, environment)
    extra = []
    for name in ("provider", "support"):
        distributions = owned / (name + "-dist")
        _command(
            [
                uv,
                "build",
                "--offline",
                "--no-build-isolation",
                "--wheel",
                "--python",
                sys.executable,
                "--out-dir",
                str(distributions),
            ],
            cwd=ROOT / "examples/typed_executor" / name,
            environment=environment,
        )
        wheels = tuple(distributions.glob("*.whl"))
        assert len(wheels) == 1
        extra.append(wheels[0])
    python = str(virtual / "bin/python")
    _command(
        [
            uv,
            "pip",
            "install",
            "--offline",
            "--no-index",
            "--no-deps",
            "--python",
            python,
            *(str(path) for path in extra),
        ],
        cwd=owned,
        environment=environment,
    )
    _command([uv, "pip", "check", "--python", python], cwd=owned, environment=environment)
    (owned / "built.json").write_text(
        json.dumps(
            {
                "uv": uv,
                "python": python,
                "virtual": str(virtual),
                "wheels": [str(core), *(str(path) for path in extra)],
            },
            sort_keys=True,
        )
    )


def _optional(owned: Path, uv: str, environment: dict[str, str], mode: str) -> None:
    """Reuse the SAME three built wheels; add MCP first and CI only afterward."""
    built = json.loads((owned / "built.json").read_text())
    virtual = owned / "optional"
    if mode == "optional-mcp":
        _command(
            [sys.executable, "-I", "-B", "-m", "venv", "--without-pip", str(virtual)], cwd=ROOT, environment=environment
        )
        extras = ("mcp",)
    elif mode == "optional-ci":
        extras = ("ci", "mcp")
    else:
        raise ValueError("Unknown fixed optional bootstrap phase")
    seed_locked_dependencies(uv, virtual, environment, extras=extras)
    python = str(virtual / "bin/python")
    if mode == "optional-mcp":
        _command(
            [uv, "pip", "install", "--offline", "--no-index", "--no-deps", "--python", python, *built["wheels"]],
            cwd=owned,
            environment=environment,
        )
    _command([uv, "pip", "check", "--python", python], cwd=owned, environment=environment)


if __name__ == "__main__":
    main()
