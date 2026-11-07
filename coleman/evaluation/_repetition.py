"""Pure repetition identities and complete-panel means shared across domains."""

from fractions import Fraction

from coleman.core.primitives import fingerprint


def repetition_execution_id(candidate_id: str, seed: int) -> str:
    """Identify a candidate/seed cell with the unchanged legacy v1 profile.

    Parameters
    ----------
    candidate_id : str
        Nonempty trimmed candidate identity without newline characters.
    seed : int
        Unsigned 64-bit repetition coordinate. Booleans are not accepted.

    Returns
    -------
    str
        Identified SHA-256 using the retained
        ``coleman.corpus.repetition-execution.v1`` namespace. The record's
        separate decision identity names the selected source member.

    Raises
    ------
    ValueError
        Invalid candidate or seed. Legacy validation and digest bytes remain
        unchanged; no implicit identity migration occurs.

    Notes
    -----
    The historical namespace is retained so existing CI producer records need
    no relabeling. The pure operation uses no RNG, registry, I/O or shared mutable
    state. A seed coordinate is not an independent scientific observation or a
    physical attempt; retries retain separate operational evidence.
    """
    if (
        not isinstance(candidate_id, str)
        or not candidate_id
        or candidate_id.strip() != candidate_id
        or "\n" in candidate_id
        or "\r" in candidate_id
    ):
        raise ValueError("Repetition candidate requires a nonempty trimmed identifier without newlines")
    if type(seed) is not int or not 0 <= seed <= 2**64 - 1:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Repetition seed requires an unsigned 64-bit integer")
    return fingerprint(["coleman.corpus.repetition-execution.v1", candidate_id, seed])


def complete_mean(builds: tuple[tuple[float | None, ...], ...]) -> float | None:
    """Average equal repetitions within members, then equal member means.

    Parameters
    ----------
    builds : tuple of tuple of float or None
        Complete predeclared members and their repetition values. Both tuple
        levels must be nonempty and observed values must be finite binary64.
        The name retains the legacy private CI helper's keyword compatibility.

    Returns
    -------
    float or None
        Mean in the original measurement unit, rounded once from exact rational
        arithmetic over the supplied binary64 values. Any null required value
        makes the whole mean unavailable; its raw status belongs to the caller.

    Raises
    ------
    ZeroDivisionError
        An empty panel or member was supplied. Callers validate complete grids
        before this arithmetic; the legacy helper's behavior is preserved.
    ValueError, OverflowError
        A value cannot be represented as a finite rational or the final result
        cannot be represented as binary64.

    Notes
    -----
    This internal helper performs no validation, imputation, clipping, inference,
    source access or I/O. Equal member weighting is a named estimand, not a
    universal scientific assumption. Fraction arithmetic avoids intermediate
    floating-point sum overflow and order dependence; it does not recover exact
    decimal or physical measurements. Repetitions never create independent units.
    """
    means = []
    for values in builds:
        if any(value is None for value in values):
            return None
        exact = sum((Fraction(value) for value in values if value is not None), start=Fraction(0))
        means.append(exact / len(values))
    return float(sum(means, start=Fraction(0)) / len(means))
