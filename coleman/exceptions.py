"""Custom exceptions for the Coleman framework.

Classes
-------
QException
    Exception raised for Q-value computation errors in contextual
    multi-armed bandit (CMAB) policies.
"""


class QException(Exception):
    """Exception raised for Q-value computation errors in CMAB policies."""
