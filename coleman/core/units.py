"""Opaque shared unit and meaning declarations, without domain imports."""

from coleman.core.base import Immutable
from coleman.core.primitives import Identifier


class UnitSemantics(Immutable):
    """Declare case-sensitive units and semantic identity without conversion.

    Attributes
    ----------
    unit : str
        Explicit unit symbol, including dimensionless ``1`` when appropriate.
    meaning : str
        Versioned meaning defining the quantity beyond its dimensional unit.

    Notes
    -----
    Same retained class exposed by finite solvers. Equal unit symbols do not
    make different meanings comparable. No registry lookup or I/O occurs.
    """

    unit: Identifier
    meaning: Identifier
