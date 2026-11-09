"""Public installed reference provider; importing the recipe creates no owner."""

from importlib import import_module
from typing import TYPE_CHECKING

from .recipe import recipe

if TYPE_CHECKING:
    from .executor import HedgeExecutor

__all__ = ["bootstrap", "recipe"]


def bootstrap() -> HedgeExecutor:
    """Load the approved provider body after host installation checks.

    Returns
    -------
    HedgeExecutor
        Structural typed executor wrapping the shipped constant Hedge rule.

    Notes
    -----
    Calling this explicit entry point imports optional local state machinery.
    It does not capture installation, prepare a job or construct a learner.
    """
    return import_module(".executor", __name__).HedgeExecutor()
