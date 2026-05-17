"""Tests for redaction helpers."""

from unittest.mock import patch

from coleman.spec.redaction import _is_sensitive_key, redact_sensitive_data


class TestSensitiveKeyDetection:
    """Validate sensitive-key detection heuristics."""

    def test_detects_normalized_compound_key(self):
        """Compound keys like ``api_key`` should be treated as sensitive."""
        assert _is_sensitive_key("api_key") is True

    def test_rejects_non_sensitive_key(self):
        """Non-secret semantic keys should not be redacted."""
        assert _is_sensitive_key("scenario") is False


class TestRedactSensitiveData:
    """Validate recursive redaction behavior over supported types."""

    def test_redacts_nested_structures_including_tuple(self):
        """Redaction should recurse into dicts/lists/tuples and URL query params."""
        data = {
            "token": "abc",
            "items": (
                {"password": "secret"},
                "https://user:pw@example.com/path?api_key=xyz",
            ),
        }

        redacted = redact_sensitive_data(data)

        assert redacted["token"] == "<redacted>"
        assert redacted["items"][0]["password"] == "<redacted>"
        assert "<redacted>:<redacted>@" in redacted["items"][1]
        assert "api_key=%3Credacted%3E" in redacted["items"][1]

    def test_returns_original_string_when_urlsplit_fails(self):
        """Malformed URLs should be returned unchanged when parsing fails."""
        raw = "https://example.com/path?token=abc"
        with patch("coleman.spec.redaction.urlsplit", side_effect=ValueError("bad url")):
            assert redact_sensitive_data(raw) == raw
