"""Selection helpers for resolving requested policy/reward names."""

from __future__ import annotations


def resolve_requested_names(
    requested: list[str],
    available: list[str],
    *,
    wildcard_aliases: tuple[str, ...] = ("*", "all"),
) -> tuple[list[str], list[str]]:
    """Resolve user-requested names against available canonical names.

    Resolution is case-insensitive and deterministic.

    Parameters
    ----------
    requested : list[str]
        User requested names.
    available : list[str]
        Canonical available names.
    wildcard_aliases : tuple[str, ...]
        Tokens that map to all available values.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple ``(resolved_names, unknown_names)``.
    """
    available_by_lower = {name.lower(): name for name in available}
    wildcard_set = {alias.lower() for alias in wildcard_aliases}

    if any(name.lower() in wildcard_set for name in requested):
        resolved = list(available)
    else:
        resolved = []
        seen: set[str] = set()
        for name in requested:
            canonical = available_by_lower.get(name.lower())
            if canonical is None:
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            resolved.append(canonical)

    unknown = sorted(
        {name for name in requested if name.lower() not in available_by_lower and name.lower() not in wildcard_set}
    )
    return resolved, unknown
