"""Bounded scalar binary64 operations shared by pure mixture consumers."""

from __future__ import annotations

from math import exp, fsum, isfinite, log

from coleman.core.actions import Weight
from coleman.core.identity import ordered_references
from coleman.core.numeric import binary64, validate_simplex
from coleman.learning.contracts import LearningNumericalError


def _weights(weights: tuple[Weight, ...]) -> tuple[Weight, ...]:
    """Validate finite coordinate shape before checking its numerical simplex."""
    if type(weights) is not tuple or not 1 <= len(weights) <= 4096:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Exponential weights require a bounded immutable coordinate tuple")
    if any(type(item) is not Weight for item in weights):  # pylint: disable=unidiomatic-typecheck
        raise TypeError("Exponential coordinates require concrete Weight values")
    checked = tuple(Weight.model_validate(item) for item in weights)
    ordered_references(tuple(item.member for item in checked))
    validate_simplex(tuple(item.value for item in checked))
    return checked


def _output(weights: tuple[Weight, ...], values: tuple[float, ...], positive: tuple[bool, ...]) -> tuple[Weight, ...]:
    """Refuse unrepresentable positive coefficients, without epsilon repair."""
    if any(not isfinite(value) for value in values):
        raise LearningNumericalError("numeric_range", "Mixture output is outside finite binary64")
    if any(required and value == 0 for value, required in zip(values, positive, strict=True)):
        raise LearningNumericalError("positive_underflow", "A positive mathematical coefficient underflows binary64")
    try:
        validate_simplex(values)
    except ValueError as error:
        raise LearningNumericalError("normalization", "Mixture output violates binary64-simplex-v1") from error
    return tuple(Weight(member=item.member, value=value) for item, value in zip(weights, values, strict=True))


def normalize_positive_prior(weights: tuple[Weight, ...]) -> tuple[Weight, ...]:
    """Emit a binary64-normalized positive stored prior without changing its law.

    Parameters
    ----------
    weights : tuple of Weight
        Complete positive binary64-simplex-v1 raw prior. Its exact mathematical
        interpretation is a/exact_sum(a), independently of rounded emission.

    Returns
    -------
    tuple of Weight
        Each raw value divided by math.fsum's finite mass. Singletons emit1.

    Raises
    ------
    ValueError, TypeError
        Invalid/zero-support prior or malformed coordinates.
    LearningNumericalError
        A positive output cannot satisfy the finite numerical envelope.
    """
    checked = _weights(weights)
    if any(item.value <= 0 for item in checked):
        raise ValueError("Reference initialization requires a strictly positive prior")
    mass = fsum(item.value for item in checked)
    values = tuple(item.value / mass for item in checked)
    return _output(checked, values, (True,) * len(checked))


def _losses(losses: tuple[float, ...], count: int, eta: float) -> tuple[tuple[float, ...], float]:
    """Admit bounded nonnegative block losses and one finite nonnegative rate."""
    if type(losses) is not tuple or len(losses) != count:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Exponential losses require the exact finite coordinate tuple")
    values = tuple(binary64(value) for value in losses)
    rate = binary64(eta)
    if rate < 0 or any(value < 0 for value in values):
        raise ValueError("Exponential losses and learning rate must be nonnegative")
    return values, rate


def _scores(
    weights: tuple[Weight, ...], values: tuple[float, ...], rate: float, center: float, active: tuple[int, ...]
) -> tuple[float, ...]:
    """Center before scaling, retaining every positive supported log score."""
    result = []
    for index in active:
        penalty = rate * (values[index] - center)
        if not isfinite(penalty):
            raise LearningNumericalError("positive_underflow", "A positive exponential coefficient is unrepresentable")
        result.append(log(weights[index].value) - penalty)
    return tuple(result)


def exp_update(weights: tuple[Weight, ...], losses: tuple[float, ...], eta: float) -> tuple[Weight, ...]:
    """Apply normalized exponential loss reweighting with common loss centering.

    Parameters
    ----------
    weights : tuple of Weight
        Actual current emitted coefficients, including legitimate structural
        zeros. Coordinate enumeration is retained; duplicate members reject.
    losses : tuple of float
        One finite nonnegative loss or sum-loss per coordinate. Internal block
        sums may exceed1; this function does not label them observations.
    eta : float
        Finite nonnegative learning rate, in reciprocal loss units.

    Returns
    -------
    tuple of Weight
        log-space normalized coefficients. Exact structural zeros remain zero.
        Eta0 or equal active-support losses retain the original emitted vector,
        including its admitted sum residual; no hidden hold normalization runs.

    Raises
    ------
    ValueError, TypeError
        Invalid input, shape, simplex, loss or learning rate.
    LearningNumericalError
        Positive-coefficient underflow or failed finite/simplex qualification.

    Notes
    -----
    Subtract the minimum active-support loss before multiplying by eta, then
    subtract the largest log score. Equal huge losses cannot erase prior-ratio
    information through eta*loss cancellation. This finite binary64 convention
    is separate from an ideal-arithmetic reference theorem. No state mutates.
    """
    checked = _weights(weights)
    values, rate = _losses(losses, len(checked), eta)
    active = tuple(index for index, item in enumerate(checked) if item.value > 0)
    center = min(values[index] for index in active)
    if rate == 0 or all(values[index] == center for index in active):
        return checked
    scores = _scores(checked, values, rate, center, active)
    pivot = max(scores)
    unnormalized = tuple(exp(score - pivot) for score in scores)
    mass = fsum(unnormalized)
    normalized = iter(value / mass for value in unnormalized)
    result = tuple(next(normalized) if item.value > 0 else 0.0 for item in checked)
    return _output(checked, result, tuple(item.value > 0 for item in checked))


def _add(total: float, correction: float, value: float) -> tuple[float, float]:
    """Advance a finite Neumaier accumulator, retaining its correction."""
    updated = total + value
    residual = (total - updated) + value if abs(total) >= abs(value) else (value - updated) + total
    corrected = correction + residual
    if not isfinite(updated) or not isfinite(corrected):
        raise LearningNumericalError("numeric_range", "Cumulative loss arithmetic exceeds finite binary64")
    return updated, corrected


def accumulate_losses(
    totals: tuple[float, ...], corrections: tuple[float, ...], losses: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Add one complete bounded panel to checkpointed compensated statistics.

    Parameters
    ----------
    totals, corrections : tuple of float
        Complete finite Neumaier state in the fixed member order.
    losses : tuple of float
        Already admitted complete loss vector in[0,1].

    Returns
    -------
    tuple of tuple
        Updated totals and corrections; neither input is modified.

    Raises
    ------
    ValueError
        Mismatched finite vector cardinalities.
    LearningNumericalError
        The resulting sufficient statistics cannot remain finite.
    """
    if not 1 <= len(totals) <= 4096 or len(totals) != len(corrections) or len(totals) != len(losses):
        raise ValueError("Cumulative losses require complete matched bounded vectors")
    pairs = tuple(
        _add(total, correction, value) for total, correction, value in zip(totals, corrections, losses, strict=True)
    )
    return tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)


def _prefix(values: tuple[float, ...]) -> tuple[float, ...]:
    """Compute stable running positive sums in O(n), without subtraction."""
    result = [0.0]
    total, correction = 0.0, 0.0
    for value in values:
        total, correction = _add(total, correction, value)
        result.append(fsum((total, correction)))
    return tuple(result)


def excluding_origin_share(weights: tuple[Weight, ...], alpha: float) -> tuple[Weight, ...]:
    """Redistribute normalized loss weights to the other n-1 coordinates.

    Parameters
    ----------
    weights : tuple of Weight
        Actual normalized exponential output q in binary64-simplex-v1.
    alpha : float
        Finite fraction[0,1]. Alpha0 retains q; alpha1 redistributes every
        origin's mass to other coordinates. Singleton output is exactly1.

    Returns
    -------
    tuple of Weight
        (1-alpha)*q_i + alpha*sum(j!=i,q_j)/(n-1), in O(n) time/storage.

    Raises
    ------
    ValueError, TypeError
        Invalid coordinates or share fraction.
    LearningNumericalError
        A positive output underflows or fails the finite simplex envelope.

    Notes
    -----
    Prefix/suffix compensated sums avoid catastrophic ``1-q_i`` cancellation
    when the largest rounded coefficient equals1 beside tiny positive mass.
    No all-experts target share, post-share renormalization or state mutation
    is hidden in this excluding-origin reference convention.
    """
    checked = _weights(weights)
    fraction = binary64(alpha)
    if not 0 <= fraction <= 1:
        raise ValueError("Fixed Share alpha must lie in[0,1]")
    if len(checked) == 1:
        return (Weight(member=checked[0].member, value=1.0),)
    if fraction == 0:
        return checked
    values = tuple(item.value for item in checked)
    before, after = _prefix(values), _prefix(tuple(reversed(values)))
    complements = tuple(fsum((before[index], after[len(values) - index - 1])) for index in range(len(values)))
    result = tuple(
        fsum(((1 - fraction) * value, fraction * (complement / (len(values) - 1))))
        for value, complement in zip(values, complements, strict=True)
    )
    positive = tuple(
        ((fraction < 1 and value > 0) or (fraction > 0 and complement > 0))
        for value, complement in zip(values, complements, strict=True)
    )
    return _output(checked, result, positive)
