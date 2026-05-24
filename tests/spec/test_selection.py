"""Tests for spec name-resolution helpers."""

from coleman.spec.selection import resolve_requested_names


def test_resolve_requested_names_case_insensitive_and_deterministic():
    resolved, unknown = resolve_requested_names(
        requested=["ucb1", "RANDOM", "UCB1"],
        available=["Random", "UCB1", "Greedy"],
    )

    assert resolved == ["UCB1", "Random"]
    assert unknown == []


def test_resolve_requested_names_wildcard_maps_to_all():
    resolved, unknown = resolve_requested_names(
        requested=["all"],
        available=["Random", "UCB1", "Greedy"],
    )

    assert resolved == ["Random", "UCB1", "Greedy"]
    assert unknown == []


def test_resolve_requested_names_reports_unknowns():
    resolved, unknown = resolve_requested_names(
        requested=["Random", "Nope", "Missing"],
        available=["Random", "UCB1"],
    )

    assert resolved == ["Random"]
    assert unknown == ["Missing", "Nope"]
