"""Measure an empty uv dependency cache and its offline warm reuse.

This is trusted operator installation tooling, never a task request. Copy it
outside the checkout and run with the reviewed wheel's interpreter. Supply the
explicit frozen project, reviewed wheel and independently verified uv0.13.1.
The existing interpreter, OS and remote index/CDN caches are not cleared.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.core.base import CoreContract
from coleman.core.numeric import Binary64
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from coleman.telemetry.resources import ProcessResourceTracker

LOCK_FILENAME = "uv.lock"


class CacheCell(CoreContract):
    """Record one fresh environment, cache occupancy and installation seconds.

    Notes
    -----
    CPU/RSS belong to this host, not uv or its children. Wall seconds include
    actual venv creation, dependency acquisition, wheel install and checks.
    Cache bytes are retained files, not network transfer bytes.
    """

    mode: Literal["empty-online", "warm-offline"]
    cache_before_files: Annotated[int, Field(strict=True, ge=0, le=16384)]
    cache_before_bytes: Annotated[int, Field(strict=True, ge=0, le=256 * 1024**2)]
    cache_after_files: Annotated[int, Field(strict=True, ge=0, le=16384)]
    cache_after_bytes: Annotated[int, Field(strict=True, ge=0, le=256 * 1024**2)]
    elapsed_seconds: Annotated[Binary64, Field(ge=0)]
    host_cpu_seconds: Annotated[Binary64, Field(ge=0)]
    host_peak_rss_mib: Annotated[Binary64, Field(ge=0)] | None
    runtime: InstalledRuntime


class BootstrapReport(CoreContract):
    """Require both actual profiles and one unchanged installed runtime.

    Notes
    -----
    V1 is a finite installation observation, not a universal bootstrap budget
    or source attestation. Unknown versions and partial/inconsistent reports
    reject. Host timestamps and paths are deliberately outside identity.
    An absent ``driver_runtime`` means the old artifact did not record CLI
    tooling; omitted-when-null metadata preserves its v1 bytes/fingerprint.
    The measured cells retain their actual minimal base dependency identities.
    """

    complete: Literal[True]
    driver_runtime: SelectedInstalledRuntime | None = Field(default=None, exclude_if=lambda value: value is None)
    hardware: Annotated[str, Field(strict=True, min_length=1, max_length=1024)]
    uv_version: Literal["0.13.1"]
    uv_sha256: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    lock_sha256: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    project_sha256: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    wheel_sha256: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    driver_sha256: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    cells: tuple[CacheCell, CacheCell]

    @model_validator(mode="after")
    def _paired(self):
        """Refuse a populated cold cache, changed frontier or changed runtime."""
        if self.driver_runtime is not None:
            self.driver_runtime.require_base(self.cells[0].runtime, extras=("cli",), pins=(("click", "8.1.8"),))
        cold, warm = self.cells
        if (cold.mode, warm.mode) != ("empty-online", "warm-offline"):
            raise ValueError("Both ordered empty-online and warm-offline profiles are required")
        if cold.cache_before_files or cold.cache_before_bytes:
            raise ValueError("The online cell must start with an actually empty uv cache")
        if not cold.cache_after_files or not cold.cache_after_bytes:
            raise ValueError("Dependency acquisition must populate the selected cache")
        if (warm.cache_before_files, warm.cache_before_bytes) != (cold.cache_after_files, cold.cache_after_bytes):
            raise ValueError("The warm cell must reuse the exact prior cache frontier")
        if cold.runtime.to_json() != warm.runtime.to_json():
            raise ValueError("Both fresh environments must contain the same exact base runtime")
        return self


def _occupancy(cache: Path) -> tuple[int, int]:
    """Bound a trusted uv-owned cache inventory without following symlinks."""
    count = size = 0
    for directory, subdirectories, files in os.walk(cache, followlinks=False):
        subdirectories[:] = [name for name in subdirectories if not (Path(directory) / name).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            count += 1
            size += path.stat().st_size
            if count > 16384 or size > 256 * 1024**2:
                raise ValueError("Finite bootstrap cache inventory exceeded its explicit observation cap")
    return count, size


def _run(command: list[str], root: Path, environment: dict[str, str], log: Path) -> str:
    """Preserve actual installer output under the existing240-second fence."""
    result = subprocess.run(
        command, cwd=root, env=environment, capture_output=True, text=True, timeout=240, check=False
    )
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Bootstrap command failed with exit{result.returncode}; inspect {log.name}")
    return result.stdout


@dataclass(frozen=True)
class _Setup:
    """Keep explicit trusted installer inputs separate from measurements."""

    project: Path
    wheel: Path
    uv: Path
    output: Path
    environment: dict[str, str]


def _cell(mode: Literal["empty-online", "warm-offline"], setup: _Setup) -> CacheCell:
    """Observe one real installation using only its newly owned cache."""
    virtual = setup.output / mode
    selected = {**setup.environment, "UV_PROJECT_ENVIRONMENT": str(virtual)}
    if mode == "warm-offline":
        selected["UV_OFFLINE"] = "1"
    offline = ["--offline"] if mode == "warm-offline" else []
    before = _occupancy(setup.output / "cache")
    tracker = ProcessResourceTracker()
    started = time.perf_counter()
    _run(
        [str(setup.uv), "venv", str(virtual), "--python", sys.executable],
        setup.output,
        selected,
        setup.output / f"{mode}-venv.txt",
    )
    python = virtual / "bin/python"
    _run(
        [
            str(setup.uv),
            "sync",
            "--frozen",
            "--no-default-groups",
            "--no-dev",
            "--no-install-project",
            "--no-build",
            "--python",
            str(python),
            *offline,
        ],
        setup.project,
        selected,
        setup.output / f"{mode}-sync.txt",
    )
    _run(
        [
            str(setup.uv),
            "pip",
            "install",
            "--offline",
            "--no-index",
            "--no-deps",
            "--no-build",
            "--python",
            str(python),
            str(setup.wheel),
        ],
        setup.output,
        selected,
        setup.output / f"{mode}-wheel.txt",
    )
    _run(
        [str(setup.uv), "pip", "check", "--python", str(python)],
        setup.output,
        selected,
        setup.output / f"{mode}-check.txt",
    )
    body = (
        "from coleman.telemetry.lifecycle import LifecycleSnapshot\n"
        "from coleman.spec.runtime_identity import InstalledRuntime\n"
        "print(InstalledRuntime.capture().to_json())\n"
    )
    runtime = InstalledRuntime.from_json(
        _run([str(python), "-I", "-B", "-c", body], setup.output, selected, setup.output / f"{mode}-runtime.json")
    )
    elapsed, resources = time.perf_counter() - started, tracker.sample()
    after = _occupancy(setup.output / "cache")
    return CacheCell(
        schema_version=1,
        mode=mode,
        cache_before_files=before[0],
        cache_before_bytes=before[1],
        cache_after_files=after[0],
        cache_after_bytes=after[1],
        elapsed_seconds=elapsed,
        host_cpu_seconds=resources.cpu_time_seconds,
        host_peak_rss_mib=resources.peak_rss_mib,
        runtime=runtime,
    )


def measure(
    project: Path,
    wheel: Path,
    uv: Path,
    output: Path,
    *,
    driver_runtime: SelectedInstalledRuntime | None = None,
) -> BootstrapReport:
    """Install the same wheel twice using an empty cache and offline warm cache.

    Parameters
    ----------
    project : pathlib.Path
        Explicit trusted project containing the exact pyproject.toml/uv.lock.
        Frozen resolution reads it without installing or modifying its source.
    wheel : pathlib.Path
        Independently reviewed noneditable COLEMAN wheel; no dependency build.
    uv : pathlib.Path
        Explicit independently verified uv0.13.1 executable, not downloaded here.
    output : pathlib.Path
        New private measurement directory. Existing paths reject before work;
        partial command logs remain on failure, without a completed report.
    driver_runtime : SelectedInstalledRuntime or None, optional
        Actual selected CLI closure supplied by ``main`` before publication.
        None records unavailable driver identity without inferring Click usage;
        direct programmatic installation measurement does not require Click.

    Returns
    -------
    BootstrapReport
        Complete matched runtime/cache/hardware/source observations. No claim
        that OS, Python-distribution, proxy, DNS or CDN caches were cold.

    Raises
    ------
    ValueError, OSError, RuntimeError, subprocess.TimeoutExpired
        Invalid/reused paths, changed source, unsupported uv, failed bounded
        installation/check, missing public index or incomplete observations.

    Notes
    -----
    Deliberately acquires base dependencies from public PyPI only in the first
    cell. Warm resolution and local-wheel installation explicitly use offline
    mode. Separate fresh environments share only this newly owned uv cache.
    No global environment/cache, scheduler or resource fence is changed.
    """
    project, wheel, uv, output = (path.absolute() for path in (project, wheel, uv, output))
    if wheel.suffix != ".whl" or not wheel.is_file() or not uv.is_file():
        raise ValueError("Bootstrap requires an existing reviewed wheel and verified uv executable")
    sources = {path: path.read_bytes() for path in (project / "pyproject.toml", project / LOCK_FILENAME, uv, wheel)}
    packages = tomllib.loads(sources[project / LOCK_FILENAME].decode())["package"]
    if any(item["source"] != {"registry": "https://pypi.org/simple"} for item in packages if item["name"] != "coleman"):
        raise ValueError("This bootstrap profile admits only the reviewed public PyPI registry closure")
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "cache"
    cache.mkdir()
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("UV_", "PIP_")) and key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    environment.update(
        UV_CACHE_DIR=str(cache),
        UV_DEFAULT_INDEX="https://pypi.org/simple",
        UV_PYTHON_DOWNLOADS="never",
        UV_LINK_MODE="copy",
        UV_NO_PROGRESS="1",
        UV_NO_CONFIG="1",
    )
    version = _run([str(uv), "--version"], output, environment, output / "uv-version.txt")
    if not version.startswith("uv 0.13.1 "):
        raise ValueError("This frozen bootstrap profile requires independently verified uv0.13.1")
    setup = _Setup(project, wheel, uv, output, environment)
    cells = (_cell("empty-online", setup), _cell("warm-offline", setup))
    if any(path.read_bytes() != content for path, content in sources.items()):
        raise ValueError("Frozen project changed during the actual bootstrap observations")
    report = BootstrapReport(
        schema_version=1,
        complete=True,
        driver_runtime=driver_runtime,
        hardware=f"{platform.platform()}; processor={platform.processor()}; logical_cpus={os.cpu_count()}",
        uv_version="0.13.1",
        uv_sha256=hashlib.sha256(sources[uv]).hexdigest(),
        lock_sha256=hashlib.sha256(sources[project / LOCK_FILENAME]).hexdigest(),
        project_sha256=hashlib.sha256(sources[project / "pyproject.toml"]).hexdigest(),
        wheel_sha256=hashlib.sha256(sources[wheel]).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cells=(cells[0], cells[1]),
    )
    ArtifactWriter(output).write_immutable(output / "complete.json", report.to_json())
    return report


def main() -> None:
    """Run trusted installation qualification through the optional Click command.

    Raises
    ------
    SystemExit
        Help, malformed flags or a missing CLI extra exit before creating any
        output/cache; missing tooling includes installation guidance.

    Notes
    -----
    Click loads only for command invocation. All four paths remain required
    trusted operator inputs. The selected CLI closure belongs to this driver;
    the actual empty/warm cells still install and measure the minimal package.
    The original240-second installer fences remain unchanged.
    """
    try:
        click = import_module("click")
    except ImportError as error:
        raise SystemExit("Install the reviewed COLEMAN wheel with its [cli] extra") from error

    @click.option("--project", type=click.Path(path_type=Path), required=True)
    @click.option("--wheel", type=click.Path(path_type=Path), required=True)
    @click.option("--uv", type=click.Path(path_type=Path), required=True)
    @click.option("--output", type=click.Path(path_type=Path), required=True)
    def _command(project: Path, wheel: Path, uv: Path, output: Path) -> None:
        """Measure the original empty-cache and offline warm-cache pair."""
        report = measure(project, wheel, uv, output, driver_runtime=SelectedInstalledRuntime.capture(extras=("cli",)))
        print(report.to_json())

    command = click.command(help=__doc__)(_command)
    command.main()


if __name__ == "__main__":
    main()
