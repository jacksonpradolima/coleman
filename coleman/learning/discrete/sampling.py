"""Finite integer-ticket sampling, never a floating cumulative-probability draw."""

import math
from fractions import Fraction

from coleman.core.numeric import binary64
from coleman.learning.discrete.contracts import GRID_SIZE


def _proposal_values(log_weights: tuple[float, ...], gamma: float) -> tuple[tuple[float, ...], float]:
    """Validate bounded callback-free numerical values before exponentiation."""
    if type(log_weights) is not tuple or not 1 <= len(log_weights) <= 256:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("The finite proposal requires a tuple of one through 256 log weights")
    checked = tuple(binary64(value) for value in log_weights)
    exploration = binary64(gamma)
    if any(not -700 <= value <= 0 for value in checked):
        raise ValueError("Unsupported finite log-weight envelope")
    if not len(checked) / GRID_SIZE <= exploration <= 1:
        raise ValueError("Gamma cannot represent all exploration support")
    return checked, exploration


def exp3_tickets(log_weights: tuple[float, ...], gamma: float) -> tuple[int, ...]:
    """Apportion the declared Exp3 proposal into exact dyadic53 action masses.

    Parameters
    ----------
    log_weights : tuple of float
        One through 256 finite centered binary64 log weights within [-700, 0].
    gamma : float
        Exploration coefficient in [K/2**53, 1], dimensionless.

    Returns
    -------
    tuple of int
        Positive integer tickets summing exactly to 2**53. Largest remainders
        win, with canonical coordinate order resolving exact ties. Each mass
        differs from its rationalized proposal by strictly less than 2**-53.

    Raises
    ------
    ValueError
        Unsupported finite envelope or lost exploration support.

    Notes
    -----
    Exponentials use the host binary64 math library; their exact float values
    are rationalized before normalization and mixing. This numerical profile
    differs from real-arithmetic Exp3 and claims no inherited regret theorem.
    No RNG is consulted and no clipping or clip-and-renormalize is performed.
    """
    log_weights, gamma = _proposal_values(log_weights, gamma)
    size = len(log_weights)
    weights = tuple(Fraction.from_float(math.exp(value)) for value in log_weights)
    total = sum(weights, Fraction())
    exploration = Fraction.from_float(gamma)
    quotas = tuple(GRID_SIZE * ((1 - exploration) * weight / total + exploration / size) for weight in weights)
    counts = [quota.numerator // quota.denominator for quota in quotas]
    remainder = GRID_SIZE - sum(counts)
    order = sorted(range(size), key=lambda index: (-(quotas[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    if any(count <= 0 for count in counts):
        raise ValueError("Finite sampler lost required exploration support")
    return tuple(counts)


def ticket_index(tickets: tuple[int, ...], draw: int) -> int:
    """Resolve one exact integer draw against a complete finite distribution.

    Parameters
    ----------
    tickets : tuple of int
        Nonnegative masses summing to 2**53, with at least one coordinate.
    draw : int
        Owned uniform integer in [0, 2**53), rather than a seed or float draw.

    Returns
    -------
    int
        Chosen canonical coordinate. Zero-mass coordinates are never selected.

    Raises
    ------
    ValueError
        Invalid mass or draw. No approximation or fallback occurs.
    """
    # Exact built-in integers exclude bool and callback-bearing numeric subclasses.
    # pylint: disable=unidiomatic-typecheck
    if (
        not tickets
        or any(type(count) is not int or count < 0 for count in tickets)
        or sum(tickets) != GRID_SIZE
        or type(draw) is not int
        or not 0 <= draw < GRID_SIZE
    ):
        raise ValueError("Sampling requires a complete integer distribution and 53-bit draw")
    cumulative = 0
    for index, count in enumerate(tickets):
        cumulative += count
        if draw < cumulative:
            return index
    raise ValueError("Integer sampler exhausted a malformed distribution")
