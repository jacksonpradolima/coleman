"""Build offline wheels and seed selected scratch environments from the project lock."""

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _run(arguments: list[str], environment: Mapping[str, str]) -> None:
    """Run one bounded offline bootstrap step and retain independent failures."""
    result = subprocess.run(
        arguments,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
    _run(arguments, {**environment, "UV_PROJECT_ENVIRONMENT": str(virtual)})


def build_install_offline_wheel(
    uv: str,
    distributions: Path,
    virtual: Path,
    environment: Mapping[str, str],
    *,
    extras: tuple[str, ...] = (),
) -> Path:
    """Build a real wheel and install one locked profile in a scratch environment.

    Parameters
    ----------
    uv : str
        Executable used by the calling installed-wheel test.
    distributions : pathlib.Path
        Owned build directory; it must contain exactly one wheel after building.
    virtual : pathlib.Path
        Scratch environment to create with the current test interpreter.
    environment : Mapping[str, str]
        Child environment, including any owned cache configuration. It is copied.
    extras : tuple[str, ...], optional
        Exact selected capabilities; empty retains the base-only installation.

    Returns
    -------
    pathlib.Path
        Actual noneditable wheel installed in the newly created environment.

    Raises
    ------
    AssertionError
        Build/install/check failed, with standard output/error retained, or the
        owned output did not contain exactly one wheel.
    subprocess.TimeoutExpired
        An individual bootstrap command exceeded 240 seconds.
    OSError
        An executable or local build output could not be accessed.

    Notes
    -----
    Uses the current interpreter's already installed build backend, without
    build isolation or registry access. The outer qualification environment must
    supply that backend and cache the selected locked dependency artifacts.
    Creates only owned build/scratch directories and cache entries. The caller's
    environment and root project environment are not installed or synchronized.
    Dependency resolution stays frozen; local wheel installation does not
    resolve and is followed by a real dependency check. Normal consumer resolver
    qualification deliberately uses a separate test path.
    """
    offline = {**environment, "UV_OFFLINE": "1", "UV_NO_INDEX": "1", "UV_PYTHON_DOWNLOADS": "never"}
    _run(
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
        offline,
    )
    wheels = tuple(distributions.glob("*.whl"))
    assert len(wheels) == 1, "Offline build must produce exactly one wheel"
    wheel = wheels[0]
    _run([uv, "venv", str(virtual), "--python", sys.executable], offline)
    seed_locked_dependencies(uv, virtual, offline, extras=extras)
    python = str(virtual / "bin/python")
    requirement = str(wheel) + ("[" + ",".join(extras) + "]" if extras else "")
    _run([uv, "pip", "install", "--offline", "--no-index", "--no-deps", "--python", python, requirement], offline)
    _run([uv, "pip", "check", "--python", python], offline)
    return wheel
