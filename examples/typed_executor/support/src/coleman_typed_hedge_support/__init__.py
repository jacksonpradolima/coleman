"""Select an actual next panel; this dependency contains no learning rule."""


def panel_at[T](panels: tuple[T, ...], position: int) -> T:
    """Read the next original panel using an accepted-panel logical position.

    Parameters
    ----------
    panels : tuple
        Complete finite ordered source trajectory, selected by the provider.
    position : int
        Strict nonnegative zero-based accepted-panel count.

    Returns
    -------
    object
        Original panel at that position, without copying or transforming it.

    Raises
    ------
    ValueError
        The position is not a strict integer inside the supplied trajectory.
    """
    if type(position) not in (int,) or not 0 <= position < len(panels):
        raise ValueError("Panel position must name an original trajectory item")
    return panels[position]
