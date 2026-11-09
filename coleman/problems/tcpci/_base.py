"""Bound new TCPCI mapping records without changing retained native readers."""

from typing import Any, Self

from coleman.core._wire import BoundedContract

MAXIMUM_BYTES = 16 * 1024**2


class TcpciContract(BoundedContract):
    """Keep strict v1 fields within a complete 16MiB, depth64 envelope.

    Notes
    -----
    The shared core reader supplies graph, unknown-field and duplicate-key
    guards. This namespace adds no execution, source lookup or migration.
    """

    _maximum_bytes = MAXIMUM_BYTES

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Self:
        """Bound mutable input before the shared reader copies or decodes it.

        Parameters
        ----------
        json_data : str or bytes or bytearray
            Complete built-in UTF-8 JSON, at most16MiB and lexical depth64.
            Callback-bearing subclasses are not supported input kinds.
        **kwargs : object
            Existing strict core reader options, passed through unchanged.

        Returns
        -------
        Self
            Revalidated concrete known-version record.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Invalid bounds, content, version or relaxed validation policy.
        """
        if type(json_data) not in {str, bytes, bytearray}:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("TCPCI JSON requires built-in str, bytes or bytearray")
        if type(json_data) is bytearray and len(json_data) > cls._maximum_bytes:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("TCPCI bytearray exceeds its byte ceiling before copying")
        return super().model_validate_json(json_data, **kwargs)
