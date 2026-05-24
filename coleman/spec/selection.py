"""Selection helpers for resolving requested policy/reward names."""

from __future__ import annotations


def _to_lower_set(values: tuple[str, ...]) -> set[str]:
    """Normalize alias tuple to lowercase set."""
    return {value.lower() for value in values}


def _build_available_lookup(available: list[str]) -> dict[str, str]:
    """Build lowercase-to-canonical lookup map."""
    return {name.lower(): name for name in available}


def _has_wildcard_request(requested: list[str], wildcard_set: set[str]) -> bool:
    """Return whether any requested token maps to wildcard semantics."""
    return any(name.lower() in wildcard_set for name in requested)


def _resolve_known_names(requested: list[str], available_by_lower: dict[str, str]) -> list[str]:
    """Resolve canonical names, preserving request order and removing duplicates."""
    resolved: list[str] = []
    seen: set[str] = set()
    for name in requested:
        canonical = available_by_lower.get(name.lower())
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        resolved.append(canonical)
    return resolved


def _collect_unknown_names(
    requested: list[str],
    available_by_lower: dict[str, str],
    wildcard_set: set[str],
) -> list[str]:
    """Collect requested names that are neither available nor wildcard aliases."""
    return sorted(
        {name for name in requested if name.lower() not in available_by_lower and name.lower() not in wildcard_set}
    )


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
    available_by_lower = _build_available_lookup(available)
    wildcard_set = _to_lower_set(wildcard_aliases)

    if _has_wildcard_request(requested, wildcard_set):
        resolved = list(available)
    else:
        resolved = _resolve_known_names(requested, available_by_lower)

    unknown = _collect_unknown_names(requested, available_by_lower, wildcard_set)
    return resolved, unknown
