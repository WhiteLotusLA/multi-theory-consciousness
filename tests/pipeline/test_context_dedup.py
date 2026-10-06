"""Repeated-sentence removal and the reply repetition check."""

import copy
from pathlib import Path

from mtc.pipeline import context_dedup
from mtc.pipeline.context_dedup import (
    DEFAULT_FILLER_PREFIXES,
    check_response_repetition,
    deduplicate_context,
    normalize_sentence,
    split_into_sentences,
)

REPEATED = "The plan for the garden project starts with the soil test."
OTHER = "Then the seed order goes out at the end of the week, as agreed."


def _turn(role, content, speaker=None):
    entry = {"role": role, "content": content}
    if speaker:
        entry["speaker"] = speaker
    return entry


def test_third_repeat_removed_from_assistant_turns_only():
    history = [
        _turn("user", REPEATED, "Alice"),
        _turn("assistant", f"{REPEATED} {OTHER}"),
        _turn("user", REPEATED, "Alice"),
        _turn("assistant", REPEATED),
        _turn("user", REPEATED, "Alice"),
        _turn("assistant", f"Well, {REPEATED} {OTHER}"),
        _turn("user", REPEATED, "Alice"),
    ]
    result = deduplicate_context(history)
    # user turns are untouched, all four of them
    assert [m["content"] for m in result if m["role"] == "user"] == [REPEATED] * 4
    assistant = [m["content"] for m in result if m["role"] == "assistant"]
    # the first two assistant copies stay; the third (with a filler prefix,
    # which normalizes to the same sentence) is removed, the rest of that
    # turn survives
    assert assistant[0] == f"{REPEATED} {OTHER}"
    assert assistant[1] == REPEATED
    assert assistant[2] == OTHER
    # an assistant message whose every sentence repeats is dropped
    more = history + [_turn("assistant", REPEATED)]
    result = deduplicate_context(more)
    assert len(result) == len(history)


def test_min_length_respected():
    short = "Sounds good to me."
    assert len(normalize_sentence(short)) < 40
    history = [_turn("assistant", short) for _ in range(5)]
    assert deduplicate_context(history) == history
    # a lower threshold counts it
    result = deduplicate_context(history, min_length=5)
    assert len(result) == 2


def test_blocklist_empty_by_default_and_extensible():
    phrase = "a line the operator saw loop many times over"
    sentence = f"Here is {phrase} in a reply."
    history = [_turn("assistant", f"{sentence} {OTHER}")]
    assert deduplicate_context(history) == history
    result = deduplicate_context(history, blocklist=(phrase.upper(),))
    assert result == [_turn("assistant", OTHER)]
    # user turns are never filtered by the blocklist
    user = [_turn("user", sentence, "Alice")]
    assert deduplicate_context(user, blocklist=(phrase,)) == user


def test_splitter_handles_dashes_without_literals():
    em, en = chr(0x2014), chr(0x2013)
    text = (
        f"The first clause runs here{em}the second clause follows on. "
        f"A third clause {en} a fourth one.\n\nA new paragraph: with a colon; "
        "and a semicolon."
    )
    parts = split_into_sentences(text)
    assert parts[0] == f"The first clause runs here{em}"
    assert parts[1] == "the second clause follows on."
    assert parts[2] == f"A third clause {en}"
    assert parts[3] == "a fourth one."
    assert parts[4:] == ["A new paragraph:", "with a colon;", "and a semicolon."]
    # the module source holds no dash characters (it builds them from code
    # points), and no other non-ASCII character
    source = Path(context_dedup.__file__).read_text(encoding="utf-8")
    assert source.isascii()


def test_response_repetition_detected():
    history = [_turn("assistant", REPEATED), _turn("assistant", f"So, {REPEATED}")]
    found = check_response_repetition(f"{OTHER} {REPEATED}", history)
    assert found == normalize_sentence(REPEATED)
    # one earlier copy is below the default threshold of two
    assert check_response_repetition(REPEATED, history[:1]) is None
    assert check_response_repetition(REPEATED, history[:1], max_occurrences=1)
    # user copies do not count
    users = [_turn("user", REPEATED, "Alice")] * 3
    assert check_response_repetition(REPEATED, users) is None
    assert check_response_repetition("", history) is None


def test_repetition_log_carries_no_text(caplog):
    history = [_turn("assistant", REPEATED)] * 2
    with caplog.at_level("WARNING", logger="mtc.pipeline.context_dedup"):
        assert check_response_repetition(REPEATED, history)
    assert caplog.records
    assert all("garden" not in record.getMessage() for record in caplog.records)


def test_non_destructive():
    history = [
        _turn("assistant", REPEATED),
        _turn("assistant", REPEATED),
        _turn("assistant", f"{REPEATED} {OTHER}"),
    ]
    before = copy.deepcopy(history)
    result = deduplicate_context(history)
    assert history == before
    assert result is not history
    assert all(a is not b for a, b in zip(result, history))
    assert result[-1]["content"] == OTHER


def test_filler_prefixes_are_generic():
    assert "well" in DEFAULT_FILLER_PREFIXES
    assert normalize_sentence("Well, okay, *the* plan holds.") == "the plan holds."
    assert normalize_sentence('"The plan holds."') == "the plan holds."
