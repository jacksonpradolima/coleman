"""Bound the new task namespace without changing retained workflow codecs."""

from typing import Any, Literal, Self

from coleman.core._wire import BoundedContract

MAXIMUM_BYTES = 16 * 1024**2


class TaskContract(BoundedContract):
    """Require an explicit task generation and bounded complete JSON envelope.

    Attributes
    ----------
    contract : {'coleman-tasks-v1'}
        Required namespace, distinct from retained native workflow artifacts.

    Notes
    -----
    Inherits the pure shared byte/depth/graph guards with a 16 MiB task ceiling.
    Performs no capture, codec lookup, import, clock read or filesystem access.
    Concrete codecs separately validate encoded bodies before runtime callbacks.
    """

    contract: Literal["coleman-tasks-v1"]
    _maximum_bytes = MAXIMUM_BYTES

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Self:
        """Guard mutable input length before delegating to the shared reader.

        Parameters
        ----------
        json_data : str or bytes or bytearray
            Complete JSON, at most this contract's selected byte ceiling.
        **kwargs : object
            Existing strict core reader options, passed through unchanged.

        Returns
        -------
        Self
            Concrete task value with shared graph/depth/version validation.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Invalid bounds, JSON, typed content or a relaxed validation policy.
        """
        if type(json_data) is bytearray and len(json_data) > cls._maximum_bytes:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("Task bytearray exceeds its byte ceiling before copying")
        return super().model_validate_json(json_data, **kwargs)
