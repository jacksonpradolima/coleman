"""Sensitive data redaction helpers for persisted artifacts."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_REDACTED = "<redacted>"

_SENSITIVE_KEY_TOKENS = {
    "access_key",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "bearer",
    "client_secret",
    "cookie",
    "credential",
    "jwt",
    "password",
    "passwd",
    "private_key",
    "secret",
    "secret_key",
    "session",
    "token",
}

_SPLIT_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _is_sensitive_key(key: str) -> bool:
    """Return ``True`` when *key* looks like a sensitive credential field."""
    normalized = _SPLIT_NON_ALNUM.sub("_", key.lower()).strip("_")
    if not normalized:
        return False
    if normalized in _SENSITIVE_KEY_TOKENS:
        return True
    parts = [p for p in normalized.split("_") if p]
    return any(part in _SENSITIVE_KEY_TOKENS for part in parts)


def _redact_url_secret_parts(value: str) -> str:
    """Redact URL userinfo and sensitive query-string values."""
    if "://" not in value:
        return value

    try:
        split = urlsplit(value)
    except ValueError:
        return value

    netloc = split.netloc
    if "@" in netloc:
        userinfo, host = netloc.rsplit("@", 1)
        userinfo = "<redacted>:<redacted>" if ":" in userinfo else _REDACTED
        netloc = f"{userinfo}@{host}"

    query_pairs = parse_qsl(split.query, keep_blank_values=True)
    if query_pairs:
        redacted_pairs = []
        for key, val in query_pairs:
            if _is_sensitive_key(key):
                redacted_pairs.append((key, _REDACTED))
            else:
                redacted_pairs.append((key, val))
        query = urlencode(redacted_pairs, doseq=True)
    else:
        query = split.query

    return urlunsplit((split.scheme, netloc, split.path, query, split.fragment))


def redact_sensitive_data(data: Any) -> Any:
    """Recursively redact likely secrets from nested structures."""
    if isinstance(data, dict):
        out: dict[Any, Any] = {}
        for key, value in data.items():
            key_str = str(key)
            if _is_sensitive_key(key_str):
                out[key] = _REDACTED
            else:
                out[key] = redact_sensitive_data(value)
        return out

    if isinstance(data, list):
        return [redact_sensitive_data(item) for item in data]

    if isinstance(data, tuple):
        return tuple(redact_sensitive_data(item) for item in data)

    if isinstance(data, str):
        return _redact_url_secret_parts(data)

    return data
