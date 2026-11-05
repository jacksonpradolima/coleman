"""Explicit zoned timestamps shared without importing evaluation or domain adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import AfterValidator, AwareDatetime, BeforeValidator


def _iso_time(value: Any) -> Any:
    """Accept ISO strings or datetimes, without numeric epoch/scale inference."""
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        raise ValueError("An explicitly zoned ISO timestamp is required")
    return value


def _utc(value: datetime) -> datetime:
    """Normalize aware instants to UTC; AwareDatetime rejects naive inputs first."""
    return value.astimezone(UTC)


Timestamp = Annotated[AwareDatetime, BeforeValidator(_iso_time), AfterValidator(_utc)]
