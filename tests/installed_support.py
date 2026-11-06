"""Seed selected installed-wheel test environments from the project lock offline."""

import subprocess
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).parents[1]


def seed_locked_dependencies(
    uv: str, virtual: Path, environment: Mapping[str, str], *, extras: tuple[str, ...] = ()
) -> None:
    """Install one locked dependency profile without resolving registry metadata.

    Parameters
    ----------
    uv : str
        Executable used by the calling installed-wheel test.
    virtual : pathlib.Path
        Already created scratch environment, with its interpreter at ``bin/python``.
    environment : Mapping[str, str]
        Child environment, including any owned cache configuration. It is copied.
    extras : tuple[str, ...], optional
        Exact optional capabilities required by the test; empty selects base only.

    Raises
    ------
    AssertionError
        The locked bootstrap failed; its standard output and error are retained.
    subprocess.TimeoutExpired
        The dependency bootstrap exceeded 240 seconds.
    OSError
        The executable could not be launched.

    Notes
    -----
    The outer test environment must have cached the selected locked artifacts.
    A lock-based sync needs those artifacts, while a fresh offline pip resolution
    can additionally need uncached registry metadata. This sync forbids indexes
    and targets only the scratch environment. The caller subsequently installs
    its noneditable wheel with ``--offline --no-index --no-deps``.
    """
    arguments = [
        uv,
        "sync",
        "--frozen",
        "--offline",
        "--no-index",
        "--no-default-groups",
        "--no-dev",
        "--no-install-project",
        "--inexact",
        "--python",
        str(virtual / "bin/python"),
    ]
    for extra in extras:
        arguments.extend(("--extra", extra))
    result = subprocess.run(
        arguments,
        cwd=ROOT,
        env={**environment, "UV_PROJECT_ENVIRONMENT": str(virtual)},
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
