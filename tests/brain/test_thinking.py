"""Reasoning split, family tag normalization and the canonical reply form."""

import pytest

from mtc.brain.thinking import (
    THINK_CLOSE,
    THINK_OPEN,
    StreamTagNormalizer,
    canonical,
    normalize_reasoning,
    split_thinking,
)


def test_canonical_tags():
    assert (THINK_OPEN, THINK_CLOSE) == ("<think>", "</think>")


def test_split_greedy_first_open_to_last_close():
    text = "<think>weigh the options</think>The answer is four."
    assert split_thinking(text) == ("weigh the options", "The answer is four.")
    lead = "Intro. <think>a</think> middle <think>b</think> Outro."
    reasoning, spoken = split_thinking(lead)
    assert reasoning == "a</think> middle <think>b"
    assert spoken == "Intro.  Outro."


def test_unclosed_block_runs_to_end():
    assert split_thinking("<think>still going and never closed") == (
        "still going and never closed",
        "",
    )
    assert split_thinking("Words first. <think>then reasoning") == (
        "then reasoning",
        "Words first.",
    )


def test_no_block_returns_none_and_text():
    assert split_thinking("Plain reply with no reasoning.") == (
        None,
        "Plain reply with no reasoning.",
    )
    assert split_thinking("  spaced  ") == (None, "  spaced  ")
    assert split_thinking("") == (None, "")


def test_stray_close_tag_inside_reasoning():
    text = "<think>first pass</think> second pass</think>Final words."
    assert split_thinking(text) == ("first pass</think> second pass", "Final words.")


def test_reply_that_starts_inside_reasoning():
    # The chat template opened the block, so only the close tag is emitted.
    assert split_thinking("considering the request</think>Here it is.") == (
        "considering the request",
        "Here it is.",
    )
    assert split_thinking("a</think>b<think>c") == ("a\nc", "b")


def test_empty_block_is_reasoning_not_none():
    assert split_thinking("<think></think>Answer.") == ("", "Answer.")


def test_normalize_channel_tags():
    text = "<|channel>thought\nchecking the facts<channel|>Here is the answer."
    assert normalize_reasoning(text, "gemma") == (
        "checking the facts",
        "Here is the answer.",
    )
    # Canonical tags are understood by every family.
    for family in ("generic", "qwen", "gemma"):
        assert normalize_reasoning("<think>r</think>c", family) == ("r", "c")
    # A family that does not use channel tags leaves them as text.
    assert normalize_reasoning(text, "generic")[0] is None
    with pytest.raises(ValueError):
        normalize_reasoning("x", "other")


def test_canonical_form():
    assert canonical("weighing", "Answer.") == "<think>weighing</think>\n\nAnswer."
    assert canonical(None, "Answer.") == "Answer."
    assert canonical("", "Answer.") == "Answer."
    assert canonical("  ", "Answer.") == "Answer."
    reasoning, content = split_thinking(canonical("weighing", "Answer."))
    assert (reasoning, content) == ("weighing", "Answer.")


def test_stream_normalizer_translates_split_family_tags():
    normalizer = StreamTagNormalizer("gemma")
    chunks = ["<|chan", "nel>thought\nstep one", " step two<chan", "nel|>Answer", "."]
    out = "".join(normalizer.feed(chunk) for chunk in chunks) + normalizer.flush()
    assert out == "<think>\nstep one step two</think>Answer."
    plain = StreamTagNormalizer("generic")
    assert plain.feed("<think>x</think>y") + plain.flush() == "<think>x</think>y"
    held = StreamTagNormalizer("gemma")
    assert held.feed("ends with <|chan") == "ends with "
    assert held.flush() == "<|chan"
