"""
Deterministic run identifier derivation.

The ``run_id`` is the first 12 hex characters of the SHA-256 digest of
the *canonical* JSON serialisation of a resolved :class:`RunSpec`.
Canonical means: sorted keys and no whitespace padding — so the same
logical spec always yields the same id.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coleman.spec.models import RunSpec


def _canonical_json(spec: RunSpec) -> str:
    """Return a deterministic JSON string for *spec*.

    Parameters
    ----------
    spec : RunSpec
        Resolved run specification.

    Returns
    -------
    str
        Canonical JSON with sorted keys and compact separators.
    """
    payload = spec.model_dump()

    # Backward compatibility: the new execution-level Scalene safeguard
    # should not change run_id when left at its default behavior (True).
    execution = payload.get("execution")
    if isinstance(execution, dict) and execution.get("force_sequential_under_scalene", True):
        execution.pop("force_sequential_under_scalene", None)

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compute_run_id(spec: RunSpec) -> str:
    """Derive a deterministic run identifier from *spec*.

    Parameters
    ----------
    spec : RunSpec
        Resolved run specification.

    Returns
    -------
    str
        12-character hex string (SHA-256 prefix).
    """
    digest = hashlib.sha256(_canonical_json(spec).encode("utf-8")).hexdigest()
    return digest[:12]
