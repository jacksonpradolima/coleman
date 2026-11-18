"""Small lexical projection regressions with independent historical/tree oracles.

Expected outcomes use the existing literal pre-optimization DFA and a Python
tree, never a quote-erasure helper, regex match stream or optimized token list.
These cases supplement the unchanged 314 existing lexical cases.
"""

import json

import pytest

from coleman.core._wire import BoundedContract
from coleman.core.reading import _depth, bounded_json_bytes
from tests.core.test_lexical_depth import historical_depth_refuses, tree_depth


class _StringEnvelope(BoundedContract):
    """Keep quoted user data through the actual complete selected JSON reader."""

    value: str


def _assert_exact_lexical_outcome(text, refuses):
    """Check both labels/input kinds while retaining the original UTF-8 bytes."""
    assert historical_depth_refuses(text) is refuses
    content = text.encode("utf-8")
    maximum = len(content)
    for label in ("State", "Replay"):
        expected_error = f"{label} JSON container depth is invalid or exceeds 64"
        if refuses:
            with pytest.raises(ValueError) as private_error:
                _depth(text, label)
            assert str(private_error.value) == expected_error
            for payload in (text, content):
                with pytest.raises(ValueError) as public_error:
                    bounded_json_bytes(payload, maximum, label=label)
                assert str(public_error.value) == expected_error
        else:
            assert _depth(text, label) is None
            for payload in (text, content):
                actual = bounded_json_bytes(payload, maximum, label=label)
                assert actual == content
                assert actual.decode("utf-8") == text


@pytest.mark.parametrize(
    "text,refuses",
    [
        pytest.param('""[]""{}""', False, id="empty_spans_between_containers"),
        pytest.param('["a""b"]"gap"{}', False, id="adjacent_quoted_spans"),
        pytest.param('["x"]"between"}', True, id="negative_close_after_span"),
        pytest.param('"prefix"]"tail"[]', True, id="negative_prefix_before_more_spans"),
        pytest.param('"q"'.join("[" * 64 + "]" * 64), False, id="quoted_gaps_at_depth64"),
        pytest.param('"q"'.join("[" * 65 + "]" * 65), True, id="quoted_gaps_at_depth65"),
        pytest.param(r'["escaped\"[]{}"]"gap"[{}]', False, id="escaped_quote_hides_brackets"),
        pytest.param('"λ😀\n"["[]{}"]"tail"{}', False, id="literal_newline_and_utf8_spans"),
    ],
)
def test_quote_projection_preserves_delimiter_order_and_prefix_refusals(text, refuses):
    """Quoted spans must not reorder or hide outside delimiters."""
    _assert_exact_lexical_outcome(text, refuses)


@pytest.mark.parametrize(
    "text,refuses",
    [
        pytest.param("[" * 64 + '"unfinished' + chr(92), False, id="depth64_dangling_escape"),
        pytest.param("[" * 65 + '"unfinished', True, id="depth65_before_unfinished_span"),
        pytest.param(']"unfinished' + chr(92), True, id="negative_before_dangling_span"),
        pytest.param('"λ\n' + chr(92) + '"][]{}', False, id="escaped_final_quote_keeps_suffix_inside"),
    ],
)
def test_unfinished_span_preserves_original_eof_and_prefix_semantics(text, refuses):
    """An unmatched quote consumes EOF, including its final escape character."""
    _assert_exact_lexical_outcome(text, refuses)


@pytest.mark.parametrize("levels", [0, 64, 65], ids=["scalar", "depth64", "depth65"])
def test_quoted_content_substitution_preserves_independent_outside_tree_depth(levels):
    """Quoted bracket insertions and escaped quotes cannot create containers."""
    variants = ("anchor", "anchor" + "[]{}" * 32, 'anchor"\\\nλ😀' + "[}" * 32)
    for leaf in variants:
        tree: object = leaf
        for _ in range(levels):
            tree = [tree]
        expected_depth = tree_depth(tree)
        assert expected_depth == levels
        text = json.dumps(tree, ensure_ascii=False, allow_nan=False)
        assert json.loads(text) == tree
        _assert_exact_lexical_outcome(text, expected_depth > 64)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("[]{}", id="quoted_brackets"),
        pytest.param('λ😀\n"\\[]{}', id="utf8_quotes_escape_newline"),
        pytest.param("", id="empty_quoted_value"),
    ],
)
def test_public_boundary_retains_original_bytes_for_the_actual_selected_reader(value):
    """Depth-only projection must never become the selected reader's input."""
    text = " \n" + json.dumps({"schema_version": 1, "value": value}, ensure_ascii=False) + "\t"
    content = text.encode("utf-8")
    for payload in (text, content):
        bounded = bounded_json_bytes(payload, len(content))
        assert bounded == content
        parsed = _StringEnvelope.from_json(bounded)
        assert parsed.value == value
        assert parsed.schema_version == 1


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('{"schema_version":1,"value":"first","value":"second"}', id="duplicate_key"),
        pytest.param('{"schema_version":1,"value":"x""y"}', id="adjacent_values"),
        pytest.param(r'{"schema_version":1,"value":"bad\q[]{}"}', id="invalid_escape"),
        pytest.param('{"schema_version":1,"value":"λ\n😀"}', id="literal_newline"),
        pytest.param('{"schema_version":1,"value":"unfinished' + chr(92), id="dangling_escape_eof"),
    ],
)
def test_actual_selected_reader_still_rejects_original_malformed_or_duplicate_json(text):
    """Lexical acceptance never substitutes for complete guarded JSON parsing."""
    _assert_exact_lexical_outcome(text, False)
    content = text.encode("utf-8")
    with pytest.raises(ValueError):
        _StringEnvelope.from_json(content)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b'["complete"]"tail"\xff', id="invalid_utf8_after_spans"),
        pytest.param(b'"\xf0\x9f"[]', id="incomplete_utf8_inside_span"),
    ],
)
def test_quoted_spans_cannot_hide_invalid_original_utf8(content):
    """The public boundary must decode every original byte before projection."""
    maximum = len(content)
    for label in ("State", "Replay"):
        with pytest.raises(UnicodeError):
            bounded_json_bytes(content, maximum, label=label)
