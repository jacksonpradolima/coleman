"""Public authored synthetic funded-portfolio composition and comparison."""

from importlib import import_module
from typing import TYPE_CHECKING

from .recipe import recipe

if TYPE_CHECKING:
    from .executor import PortfolioExecutor

__all__ = ["bootstrap", "recipe"]


def bootstrap() -> PortfolioExecutor:
    """Load the trusted optional task adapter after installed-image approval.

    Returns
    -------
    PortfolioExecutor
        Concrete adapter around exactly the same direct experiment. No capture,
        preparation, learner construction or execution occurs during bootstrap.
    """
    return import_module(".executor", __name__).PortfolioExecutor()
