"""
Provenance tracking for experiment runs.

Captures environmental metadata so that a run can be reproduced or
audited after the fact.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from coleman.spec.redaction import redact_sensitive_data


def _git_info() -> dict[str, Any]:
    """Collect git commit and dirty-flag if available.

    Returns
    -------
    dict
        Keys: ``commit``, ``dirty``.  Values are ``None`` when git is
        not available.
    """
    try:
        commit = (
            subprocess.check_output(  # noqa: S603, S607
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.check_output(  # noqa: S603, S607
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
            != ""
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"commit": commit, "dirty": dirty}


def _lock_hash() -> str | None:
    """Return the SHA-256 of ``uv.lock`` if it exists.

    Returns
    -------
    str or None
        Hex digest, or ``None`` if the file is absent.
    """
    import hashlib

    lock = Path("uv.lock")
    if lock.exists():
        return hashlib.sha256(lock.read_bytes()).hexdigest()
    return None


def build_provenance() -> dict[str, Any]:
    """Build a provenance dictionary for the current environment.

    Returns
    -------
    dict
        Provenance metadata including Python version, platform, git
        info, and lock-file hash.
    """
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "git": _git_info(),
        "uv_lock_hash": _lock_hash(),
    }


def save_provenance(directory: str | Path, *, redact_sensitive: bool = True) -> Path:
    """Write ``provenance.json`` into *directory*.

    Parameters
    ----------
    directory : str | Path
        Target directory (created if missing).
    redact_sensitive : bool
        If ``True``, redact likely sensitive values before persisting.

    Returns
    -------
    Path
        Path to the written file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "provenance.json"
    payload = build_provenance()
    if redact_sensitive:
        payload = redact_sensitive_data(payload)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return out
