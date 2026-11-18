"""Independent historical DFA and JSON-tree depth oracles for bounded readers."""

import itertools
import json
import random
from typing import cast

import pytest

from coleman.core._wire import BoundedContract
from coleman.core.reading import ReaderLabel, _depth, bounded_json_bytes


def historical_depth_refuses(text):
    """Retain the pre-optimization literal character DFA, without product helpers.

    This is an exact historical compatibility oracle for malformed text as well
    as valid JSON; the separate Python-tree oracle supplies independent expected
    container depth for valid JSON rather than deriving it from this scanner.
    """
    level = 0
    inside_string = consume_next = False
    for character in text:
        if inside_string:
            if consume_next:
                consume_next = False
            elif character == chr(92):
                consume_next = True
            elif character == chr(34):
                inside_string = False
        else:
            if character == chr(34):
                inside_string = True
            elif character in "[{":
                level += 1
            elif character in "]}":
                level -= 1
            if level < 0 or level > 64:
                return True
    return False


def expect_lexical(text, label, refuses):
    """Check the exact legacy error text without asking the actual JSON parser."""
    if refuses:
        with pytest.raises(ValueError) as error:
            _depth(text, label)
        assert str(error.value) == f"{label} JSON container depth is invalid or exceeds 64"
    else:
        assert _depth(text, label) is None


@pytest.mark.parametrize("length", range(6))
@pytest.mark.parametrize("label", ["Replay", "State"])
def test_bounded_exhaustive_structural_alphabet_matches_literal_historical_dfa(length, label):
    """Enumerate all short delimiter, quote, escape and neutral-character words."""
    for characters in itertools.product('[]{}"\\x', repeat=length):
        text = "".join(characters)
        expect_lexical(text, label, historical_depth_refuses(text))


def tree_depth(value):
    """Compute container nesting from a Python tree, independently of JSON text."""
    if isinstance(value, dict):
        return 1 + max((tree_depth(child) for child in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((tree_depth(child) for child in value), default=0)
    return 0


def nested(value, levels):
    """Add one real alternating object/array container on every iteration."""
    for index in range(levels):
        value = [value] if index % 2 else {'escaped"key\\λ': value}
    return value


@pytest.mark.parametrize("levels", [0, 1, 63, 64, 65, 66])
@pytest.mark.parametrize("leaf", [None, 'ordinary\\"[]{}\nλ😀'])
@pytest.mark.parametrize("label", ["Replay", "State"])
def test_standard_json_tree_oracle_preserves_exact_64_65_boundary(levels, leaf, label):
    value = nested(leaf, levels)
    assert tree_depth(value) == levels
    text = json.dumps(value, ensure_ascii=False)
    encoded = text.encode("utf-8")
    if levels > 64:
        with pytest.raises(ValueError, match=f"{label} JSON container depth"):
            bounded_json_bytes(text, len(encoded), label=label)
    else:
        assert bounded_json_bytes(text, len(encoded), label=label) == encoded


@pytest.mark.parametrize("gap", ["", "x", "xx", "λ", "😀", "\n", "ordinary" * 128])
@pytest.mark.parametrize("another_backslash", [False, True])
@pytest.mark.parametrize("label", ["Replay", "State"])
def test_ordinary_and_unicode_gaps_consume_a_pending_escape_before_the_next_token(gap, another_backslash, label):
    """A neutral gap changes whether the following quote ends the string."""
    quote, slash = chr(34), chr(92)
    text = quote + slash + gap + (slash if another_backslash else "") + quote + "[" * 65
    # With no gap, adjacent backslashes pair; with a gap the first character
    # consumes the prior escape before a possible second backslash is visited.
    refuses = another_backslash if not gap else not another_backslash
    expect_lexical(text, label, refuses)
    assert historical_depth_refuses(text) == refuses


@pytest.mark.parametrize("slashes", range(12))
@pytest.mark.parametrize("label", ["Replay", "State"])
def test_adjacent_odd_even_backslashes_preserve_quoted_bracket_semantics(slashes, label):
    text = chr(34) + chr(92) * slashes + chr(34) + "{" * 65
    expect_lexical(text, label, slashes % 2 == 0)


@pytest.mark.parametrize("label", ["Replay", "State"])
def test_many_brackets_inside_a_real_json_string_do_not_change_container_depth(label):
    value = {"quoted": '[{]}\\"λ😀' * 1024}
    assert tree_depth(value) == 1
    text = json.dumps(value, ensure_ascii=False)
    assert bounded_json_bytes(text, len(text.encode()), label=label) == text.encode()


@pytest.mark.parametrize("payload", [b"\xff", b"\x80", b"\xc0\xaf", b"\xf0\x9f", b"\xed\xa0\x80", '"\ud800"'])
@pytest.mark.parametrize("label", ["Replay", "State"])
def test_malformed_utf8_and_lone_surrogates_still_fail_before_lexical_scanning(payload, label):
    with pytest.raises(UnicodeError):
        bounded_json_bytes(payload, 128, label=label)


@pytest.mark.parametrize("label", ["Replay", "State"])
def test_encoded_byte_capacity_remains_distinct_from_character_count(label):
    text = json.dumps({"λ": "😀"}, ensure_ascii=False)
    size = len(text.encode())
    assert len(text) < size
    assert bounded_json_bytes(text, size, label=label) == text.encode()
    for capacity in (len(text), size - 1):
        with pytest.raises(ValueError, match="maximum_bytes"):
            bounded_json_bytes(text, capacity, label=label)


@pytest.mark.parametrize("capacity", [True, False, 0, -1, 2.0, "2"])
def test_strict_positive_capacity_still_refuses_coercion(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        bounded_json_bytes("{}", capacity)


@pytest.mark.parametrize("label", [None, "Unknown", 1])
def test_reader_label_remains_explicit(label):
    with pytest.raises(ValueError, match="diagnostic label"):
        bounded_json_bytes("{}", 2, label=label)


def test_input_subclass_callbacks_are_never_used_by_the_public_boundary():
    class HostileString(str):
        def __len__(self):
            raise AssertionError("A string subclass must refuse before callback execution")

        def encode(self, *arguments, **keywords):
            raise AssertionError("A string subclass must refuse before callback execution")

    class HostileBytes(bytes):
        def __len__(self):
            raise AssertionError("A byte subclass must refuse before callback execution")

    for value in (HostileString("{}"), HostileBytes(b"{}"), bytearray(b"{}"), None, True):
        with pytest.raises(TypeError, match="built-in str or bytes"):
            bounded_json_bytes(cast(str | bytes, value), 16)
    with pytest.raises(ValueError, match="diagnostic label"):
        bounded_json_bytes("{}", 2, label=cast(ReaderLabel, HostileString("State")))


@pytest.mark.parametrize("text", ["[", "{]", '"unterminated', '"trailing\\', "not JSON"])
def test_lexical_guard_continues_leaving_full_grammar_to_the_selected_reader(text):
    expect_lexical(text, "State", False)
    assert bounded_json_bytes(text, len(text.encode())) == text.encode()
    with pytest.raises(ValueError):
        BoundedContract.from_json(text)


@pytest.mark.parametrize(
    "payload", ['{"schema_version":1,"schema_version":1}', '{"schema_version":true}', '{"schema_version":2}']
)
def test_actual_selected_reader_still_refuses_duplicate_keys_and_unsupported_versions(payload):
    with pytest.raises(ValueError):
        BoundedContract.from_json(payload)


def generated_tree(generator, remaining):
    """Construct finite JSON trees without a product helper or added dependency."""
    scalars = (None, False, True, -100, 0, 100, "", 'aλ😀"\\[]{}\n', "\x00\x1f\x7f", "\u2028\u2029", "界")
    choice = generator.randrange(3) if remaining else 0
    if choice == 0:
        return generator.choice(scalars)
    if choice == 1:
        return [generated_tree(generator, remaining - 1) for _ in range(generator.randrange(4))]
    keys = ("", "a", "λ", "😀", '"', "\\", "[]{}", "\n")
    return {
        generator.choice(keys) + str(index): generated_tree(generator, remaining - 1)
        for index in range(generator.randrange(4))
    }


@pytest.mark.parametrize("seed", range(8))
def test_generated_json_and_neutral_padding_follow_independent_tree_depth(seed):
    """Wrapping changes depth; neutral padding leaves it unchanged.

    Seeds construct reproducible unit-test inputs only. They do not represent
    independent empirical observations or a statistical performance study.
    """
    generator = random.Random(seed)
    for _ in range(4):
        original = generated_tree(generator, 4)
        for levels in range(67):
            value = nested(original, levels)
            expected = tree_depth(value)
            for padding in ("", " \t\r\n"):
                text = padding + json.dumps(value, ensure_ascii=False, allow_nan=False) + padding
                content = text.encode()
                assert json.loads(text) == value
                for label in ("Replay", "State"):
                    expect_lexical(text, label, expected > 64)
                    if expected > 64:
                        with pytest.raises(ValueError, match="container depth"):
                            bounded_json_bytes(content, len(content), label=label)
                    else:
                        assert bounded_json_bytes(content, len(content), label=label) == content


@pytest.mark.parametrize("repetitions", [8191, 8192, 8193, 32768])
@pytest.mark.parametrize("chunk", ["ordinary", "λ😀\n", '[]{}\\"'])
@pytest.mark.parametrize("levels", [0, 64, 65])
@pytest.mark.parametrize("label", ["Replay", "State"])
def test_long_quoted_chunks_follow_the_independent_tree_depth(repetitions, chunk, levels, label):
    """Long neutral, Unicode and escaped spans retain the exact depth boundary."""
    value = nested(chunk * repetitions, levels)
    assert tree_depth(value) == levels
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    content = text.encode("utf-8")
    assert len(content) < 1024 * 1024
    assert json.loads(text) == value
    expect_lexical(text, label, levels > 64)
    if levels > 64:
        with pytest.raises(ValueError, match=f"{label} JSON container depth"):
            bounded_json_bytes(content, len(content), label=label)
    else:
        assert bounded_json_bytes(content, len(content), label=label) == content


@pytest.mark.parametrize("slashes", [0, 1, 2, 3, 31, 32, 8191, 8192])
@pytest.mark.parametrize("suffix", ["]", "[" * 65])
@pytest.mark.parametrize("label", ["Replay", "State"])
def test_long_odd_even_escape_runs_leave_the_following_brackets_inside_or_outside(slashes, suffix, label):
    """The quote closes only after an even run, regardless of ordinary span size."""
    text = chr(34) + "λx\n" * 4096 + chr(92) * slashes + chr(34) + suffix
    refuses = slashes % 2 == 0
    assert historical_depth_refuses(text) is refuses
    expect_lexical(text, label, refuses)


@pytest.mark.parametrize("slashes", [0, 1, 2, 3, 31, 32, 8191, 8192])
@pytest.mark.parametrize("label", ["Replay", "State"])
def test_long_unterminated_quotes_and_dangling_escape_remain_grammar_errors(slashes, label):
    """EOF inside a string does not invent a lexical close or a depth failure."""
    text = "[" * 64 + chr(34) + "λ😀[]{}\n" * 2048 + chr(92) * slashes
    assert historical_depth_refuses(text) is False
    expect_lexical(text, label, False)
    content = text.encode("utf-8")
    assert len(content) < 1024 * 1024
    assert bounded_json_bytes(content, len(content), label=label) == content
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)
    with pytest.raises(ValueError):
        BoundedContract.from_json(text)


@pytest.mark.parametrize("gap", ["x" * 8193, "\n" * 8193, "λ" * 8193])
@pytest.mark.parametrize("slashes", [1, 2, 31, 32])
@pytest.mark.parametrize("label", ["Replay", "State"])
def test_long_ordinary_gaps_consume_pending_escapes_before_a_closing_quote(gap, slashes, label):
    """Any nonempty ordinary gap clears parity before the real closing quote."""
    text = chr(34) + chr(92) * slashes + gap + chr(34) + "]"
    assert historical_depth_refuses(text) is True
    expect_lexical(text, label, True)


@pytest.mark.parametrize("count", [1, 4096, 32768])
@pytest.mark.parametrize("suffix", ["", "]", "[" * 65])
@pytest.mark.parametrize("label", ["Replay", "State"])
def test_many_complete_quoted_spans_keep_later_depth_checks(count, suffix, label):
    """Concatenated lexical strings never hide a subsequent outside bracket."""
    text = '"x"' * count + suffix
    refuses = bool(suffix)
    assert historical_depth_refuses(text) is refuses
    expect_lexical(text, label, refuses)


@pytest.mark.parametrize("chunk", ["x" * 8193, 'λ😀\n[]{}"\\' * 4096])
def test_real_string_contract_retains_full_json_grammar_after_long_spans(chunk):
    """The bounded lexical guard admits bytes; the actual reader checks syntax."""

    class StringPayload(BoundedContract):
        """Carry one actual string through the complete selected JSON reader."""

        value: str

    text = json.dumps({"schema_version": 1, "value": chunk}, ensure_ascii=False, allow_nan=False)
    content = text.encode("utf-8")
    assert len(content) < 1024 * 1024
    assert StringPayload.from_json(text).value == chunk
    assert bounded_json_bytes(content, len(content)) == content
    prefix = '{"schema_version":1,"value":"'
    malformed = (
        text[:-2],
        text[:-2] + chr(92),
        prefix + "x" * 8193 + "\n" + '"}',
        prefix + chr(92) + "q" + "x" * 8193 + '"}',
    )
    for invalid in malformed:
        assert historical_depth_refuses(invalid) is False
        expect_lexical(invalid, "State", False)
        encoded = invalid.encode("utf-8")
        assert bounded_json_bytes(encoded, len(encoded)) == encoded
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid)
        with pytest.raises(ValueError):
            StringPayload.from_json(invalid)


@pytest.mark.parametrize("prefix", ["]", "[" * 65, "{" * 65, "[" * 64 + "]" * 65])
@pytest.mark.parametrize("label", ["Replay", "State"])
def test_early_invalid_depth_is_not_hidden_by_a_later_long_unterminated_string(prefix, label):
    """A later quoted suffix cannot undo a prefix depth refusal."""
    text = prefix + chr(34) + "ordinary" * 8193 + chr(92)
    assert historical_depth_refuses(text) is True
    expect_lexical(text, label, True)


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("label", ["Replay", "State"])
def test_long_quoted_unicode_still_charges_every_encoded_byte(as_bytes, label):
    """The byte ceiling remains exact for both original built-in input kinds."""
    text = json.dumps({"λ": "😀λ\n" * 8193}, ensure_ascii=False, allow_nan=False)
    content = text.encode("utf-8")
    assert len(content) < 1024 * 1024
    assert len(content) > len(text)
    payload = content if as_bytes else text
    assert bounded_json_bytes(payload, len(content), label=label) == content
    with pytest.raises(ValueError, match="maximum_bytes"):
        bounded_json_bytes(payload, len(content) - 1, label=label)


@pytest.mark.parametrize("label", ["Replay", "State"])
@pytest.mark.parametrize(
    "text,refuses",
    [
        ('"closed"]tail', True),
        ('"closed"' + "[" * 65 + "tail", True),
        ('"closed"' + "[" * 64 + "tail", False),
        ('"unfinished' + "[" * 65 + "tail", False),
    ],
)
def test_eof_anchor_applies_only_to_the_unfinished_quoted_branch(text, refuses, label):
    """Outside brackets remain unanchored even when ordinary text follows."""
    assert historical_depth_refuses(text) is refuses
    expect_lexical(text, label, refuses)
