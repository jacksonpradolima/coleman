"""
Unit tests for the QException class.

These tests check the behavior of the custom exception class QException,
including its inheritance from the base Exception class, and its behavior
when raised with or without a custom message.
"""

import pytest

from coleman.exceptions import QException


def test_qexception_inheritance():
    """
    Checks if QException correctly inherits from the Exception class.
    """
    assert issubclass(QException, Exception)


def test_qexception_raised_without_message():
    """
    Tests if QException can be raised without a custom message.
    """
    with pytest.raises(QException) as exc_info:
        raise QException()
    assert str(exc_info.value) == ""  # No message provided


def test_qexception_raised_with_message():
    """
    Tests if QException can be raised with a custom message and if the message is retained.
    """
    message = "This is a sample error"
    with pytest.raises(QException) as exc_info:
        raise QException(message)
    assert str(exc_info.value) == message
