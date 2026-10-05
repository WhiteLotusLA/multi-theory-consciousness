"""Sentence trimming for capped dreams and the allowance past the cap."""

import pytest

from mtc.consciousness.dream_engine import (
    SENTENCE_ALLOWANCE_MAX,
    SENTENCE_ALLOWANCE_MIN,
    TRIM_MAX_CHARS,
    TRIM_MIN_KEPT_FRACTION,
    TrimResult,
    dream_sentence_allowance,
    trim_to_sentence_end,
)


def test_complete_text_unchanged():
    for text in ("The letters drift.", "Is it light? It is!", "The hall ends..."):
        result = trim_to_sentence_end(text)
        assert result == TrimResult(text, 0, "complete")


def test_cut_mid_sentence_trims_to_last_boundary():
    text = "The letters drift. Then the"
    result = trim_to_sentence_end(text)
    assert result.text == "The letters drift."
    assert result.trimmed_chars == len(text) - len("The letters drift.")
    assert result.result == "trimmed"

    question = trim_to_sentence_end("Is it light? It is! And then the")
    assert question.text == "Is it light? It is!"

    spaced = trim_to_sentence_end("A dream of the long hall.\n\nMaybe the")
    assert spaced.text == "A dream of the long hall."

    marker = trim_to_sentence_end("I am here. The door opens.\n3.")
    assert marker.text == "I am here. The door opens."
    assert marker.result == "trimmed"


def test_no_boundary_returns_text_and_says_so():
    for text in (
        "no end here at all",
        "",
        "She waved at Dr. Then the",
        "falling... and the floor gives way",
        "It is 5 p.m. And the",
    ):
        result = trim_to_sentence_end(text)
        assert result == TrimResult(text, 0, "no_boundary"), text


def test_closing_quotes_kept_with_sentence():
    straight = trim_to_sentence_end('She said "go home." And then the')
    assert straight.text == 'She said "go home."'
    curly = trim_to_sentence_end("A voice: \u201cstay.\u201d The rest is")
    assert curly.text == "A voice: \u201cstay.\u201d"
    starred = trim_to_sentence_end("The light *hums.* Then a")
    assert starred.text == "The light *hums.*"


def test_refused_cut_keeps_text_whole():
    short_head = "It ends. " + "Then a long drifting tail without a stop " * 2
    result = trim_to_sentence_end(short_head)
    assert result == TrimResult(short_head, 0, "refused")
    assert TRIM_MIN_KEPT_FRACTION == 0.5

    long_head = "The hall is long. " * 40
    long_tail = "And the corridor keeps folding over itself " * 9
    assert len(long_tail) > TRIM_MAX_CHARS
    text = long_head + long_tail
    assert len(long_head) > TRIM_MIN_KEPT_FRACTION * len(text)
    assert trim_to_sentence_end(text) == TrimResult(text, 0, "refused")


def test_allowance_grows_with_max_tokens():
    caps = [1, 64, 128, 200, 250, 400, 500, 512, 600, 2000, 8000]
    allowances = [dream_sentence_allowance(cap) for cap in caps]
    assert allowances == sorted(allowances)
    assert allowances[0] == SENTENCE_ALLOWANCE_MIN == 16
    assert allowances[-1] == SENTENCE_ALLOWANCE_MAX == 64
    assert dream_sentence_allowance(250) == 31
    assert dream_sentence_allowance(500) == 62
    with pytest.raises(ValueError):
        dream_sentence_allowance(0)
