#!/usr/bin/env python3
"""Generate llms.txt as a consolidated, LLM-friendly project reference.

This script is deterministic and intended to run in pre-commit and CI.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "llms.txt"


def _resolve_repo_path(rel: str) -> Path:
    """Resolve one repository-relative path and prevent path traversal."""
    rel_path = Path(rel)
    if rel_path.is_absolute():
        msg = f"Absolute source path is not allowed: {rel}"
        raise ValueError(msg)

    resolved = (ROOT / rel_path).resolve()
    root_resolved = ROOT.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        msg = f"Source path escapes repository root: {rel}"
        raise ValueError(msg) from exc
    return resolved


def _discover_sources() -> list[str]:
    """Return deterministic source list for llms.txt generation.

    Includes README plus every markdown file under docs/, recursively.
    """
    sources = ["README.md"]
    docs_root = ROOT / "docs"
    docs_md_files = sorted(path.relative_to(ROOT).as_posix() for path in docs_root.rglob("*.md"))
    sources.extend(docs_md_files)
    return sources


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _build_header() -> str:
    return "\n".join(
        [
            "# Coleman llms.txt",
            "",
            "> Generated file. Do not edit manually.",
            "> Source: scripts/generate_llms_txt.py",
            "",
            "## Project",
            "",
            "Coleman is a framework for test-case prioritization in CI using multi-armed bandits,",
            "with typed YAML configuration, deterministic run identifiers, sweep orchestration,",
            "parallel execution, and extensibility via hooks and namespaced custom config.",
            "",
            "## Canonical Documentation Sources",
            "",
        ]
    )


def _build_source_index() -> str:
    sources = _discover_sources()
    lines = []
    for rel in sources:
        lines.append(f"- {rel}")
    return "\n".join(lines)


def _build_consolidated() -> str:
    sources = _discover_sources()
    chunks: list[str] = []
    chunks.append("\n## Consolidated Documentation\n")

    for rel in sources:
        path = _resolve_repo_path(rel)
        if not path.exists():
            chunks.append(f"\n### FILE: {rel}\n")
            chunks.append("(missing)\n")
            continue

        chunks.append(f"\n### FILE: {rel}\n")
        chunks.append(_read_text(path))
        chunks.append("\n")

    return "\n".join(chunks)


def generate() -> str:
    parts = [
        _build_header(),
        _build_source_index(),
        _build_consolidated(),
        "\n## Notes\n\nThis file is intended for LLM ingestion and broad repository understanding.\n",
    ]
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    content = generate()
    OUT.write_text(content, encoding="utf-8")
    print(f"Updated {OUT}")  # noqa: T201


if __name__ == "__main__":
    main()
