"""
Spec I/O — load from YAML, save resolved canonical JSON.

Functions
---------
load_spec
    Load and validate a ``RunSpec`` from a YAML file.
save_resolved
    Write the resolved spec as deterministic JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from coleman.spec.models import RunSpec
from coleman.spec.packs import resolve_packs
from coleman.spec.redaction import redact_sensitive_data


def load_spec(
    path: str | Path,
    *,
    packs_dir: str | Path | None = None,
) -> RunSpec:
    """Load a :class:`RunSpec` from a YAML config file.

    Pack references (``packs:`` key) are resolved and deep-merged
    before validation.

    Parameters
    ----------
    path : str | Path
        Path to the YAML config file.
    packs_dir : str | Path | None
        Root directory for config packs.  When ``None`` (the default),
        the directory is derived as ``<config_dir>/packs`` so that
        configs remain relocatable regardless of the working directory.

    Returns
    -------
    RunSpec
        Validated run specification.

    Raises
    ------
    FileNotFoundError
        If *path* or a referenced pack does not exist.
    pydantic.ValidationError
        If the resolved dict fails schema validation.
    """
    path = Path(path).resolve()
    if packs_dir is None:
        packs_dir = path.parent / "packs"

    with open(path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    resolved = resolve_packs(raw, packs_dir=packs_dir)
    return RunSpec.model_validate(resolved)


def save_resolved(spec: RunSpec, path: str | Path, *, redact_sensitive: bool = True) -> Path:
    """Persist *spec* as canonical JSON.

    Parameters
    ----------
    spec : RunSpec
        Resolved run specification.
    path : str | Path
        Destination file path.
    redact_sensitive : bool
        If ``True``, redact likely sensitive values before persisting.

    Returns
    -------
    Path
        The written file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = spec.model_dump()
    if redact_sensitive:
        payload = redact_sensitive_data(payload)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2)
    return path
