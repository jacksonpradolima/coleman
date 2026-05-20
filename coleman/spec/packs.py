"""
Config pack resolution.

*Packs* are small YAML fragments stored under ``packs/<category>/``
that can be referenced from a user config via the ``packs`` key.

The resolver deep-merges packs into a base dict and then applies
inline overrides, producing a single flat dict ready for
:class:`RunSpec` validation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a copy of *base*.

    Parameters
    ----------
    base : dict
        Base dictionary.
    override : dict
        Overriding dictionary (values win on conflict).

    Returns
    -------
    dict
        Merged result (new object — inputs are not mutated).
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_packs(
    raw: dict[str, Any],
    *,
    packs_dir: str | Path = "packs",
) -> dict[str, Any]:
    """Resolve pack references and merge into a single config dict.

    The ``packs`` key (if present) is a list of strings like
    ``"policy/linucb"`` which map to ``packs/policy/linucb.yaml``.
    Packs are merged left-to-right, then the remaining keys in *raw*
    are applied as overrides.

    Parameters
    ----------
    raw : dict
        User-supplied config dict (may contain a ``packs`` key).
    packs_dir : str | Path
        Root directory for pack files.

    Returns
    -------
    dict
        Fully resolved config dict without the ``packs`` key.

    Raises
    ------
    FileNotFoundError
        If a referenced pack file does not exist.
    TypeError
        If the ``packs`` key is not a list, or if any item is not a string.
    ValueError
        If any pack reference is an empty string.
    """
    packs_dir = Path(packs_dir)
    raw = dict(raw)  # Shallow copy to avoid mutating the caller's dict
    pack_refs_raw = raw.pop("packs", [])

    if not isinstance(pack_refs_raw, list):
        msg = f"The 'packs' key must be a list of strings, got {type(pack_refs_raw).__name__}."
        raise TypeError(msg)

    pack_refs: list[str] = []
    for index, ref in enumerate(pack_refs_raw):
        if not isinstance(ref, str):
            msg = f"Each item in 'packs' must be a string, but item at index {index} is {type(ref).__name__}."
            raise TypeError(msg)
        if not ref:
            msg = f"Each item in 'packs' must be a non-empty string, but item at index {index} is empty."
            raise ValueError(msg)
        pack_refs.append(ref)

    merged: dict[str, Any] = {}
    for ref in pack_refs:
        pack_path = packs_dir / f"{ref}.yaml"
        if not pack_path.exists():
            msg = f"Pack file not found: {pack_path}"
            raise FileNotFoundError(msg)
        with open(pack_path, encoding="utf-8") as fh:
            pack_data = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, pack_data)

    merged = _deep_merge(merged, raw)
    return merged
