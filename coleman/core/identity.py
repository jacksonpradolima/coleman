"""Versioned local references and canonical bounded identity declarations."""

from coleman.core.base import CoreImmutable
from coleman.core.primitives import MAX_MEMBERS, Identifier


class Reference(CoreImmutable):
    """Identify a logical member and its exact selected semantic version.

    Attributes
    ----------
    id : str
        Caller-declared stable task/action/expert/dimension identity.
    version : str
        Explicit semantic version, never inferred from a method name.

    Notes
    -----
    Requires no global registry or artifact store. Definitions are identified,
    not authenticated; consumers check refs against their supplied universe.
    """

    id: Identifier
    version: Identifier


def ordered_references(values: tuple[Reference, ...]) -> tuple[Reference, ...]:
    """Validate and sort a bounded set without hiding duplicate identities.

    Parameters
    ----------
    values : tuple of Reference
        At most 4096 local refs. Empty catalogues are valid unless the consuming
        capability requires a member.

    Returns
    -------
    tuple of Reference
        Revalidated references in ascending logical-ID order.

    Raises
    ------
    ValueError
        Mutable shape, exceeded cap, invalid ref, duplicate ID or conflicting
        version. No duplicates are discarded.

    Notes
    -----
    Only set declarations are canonicalized. Ranking/slate order must remain
    significant and must not use this helper. No state or I/O is involved.
    """
    if not isinstance(values, tuple) or len(values) > MAX_MEMBERS:
        raise ValueError("References require a tuple of at most 4096 members")
    checked = tuple(Reference.model_validate(value) for value in values)
    ids = [value.id for value in checked]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate reference ID or conflicting semantic version")
    return tuple(sorted(checked, key=lambda value: value.id))
