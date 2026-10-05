"""Thinking-loop detection inside reasoning, for whole blocks and streams."""

import pytest

from mtc.brain.thinking_loop_detector import (
    DEFAULT_LOOP_MARKERS,
    ThinkingLoopDetected,
    ThinkingLoopDetector,
)

# 100 characters, so the n-gram windows (every 20 characters) line up.
PHRASE = (
    "The question asks for the boiling point of water, so I must recall the "
    "standard value at sea level. "
)
VARIED = (
    "Let me read the request carefully. The person wants a short summary of the "
    "article. I should identify the main claim first, then the supporting "
    "evidence. The second paragraph introduces a counterexample worth noting. "
    "A concise answer of three sentences fits the request. I will avoid jargon "
    "and keep the tone neutral. Finally I check that nothing important is lost."
)


def _feed_all(detector, text, size=30):
    detector.feed("<think>")
    for start in range(0, len(text), size):
        detector.feed(text[start : start + size])
    detector.feed("</think>")


def test_exception_carries_details_without_the_text_in_its_message():
    error = ThinkingLoopDetected(
        pattern="private words", strategy="ngram", occurrences=4
    )
    assert (error.pattern, error.strategy, error.occurrences) == (
        "private words",
        "ngram",
        4,
    )
    assert "private words" not in str(error)
    assert isinstance(error, Exception)


def test_loop_detected_on_repeated_ngrams():
    with pytest.raises(ThinkingLoopDetected) as info:
        ThinkingLoopDetector().check(PHRASE * 4)
    assert info.value.strategy == "ngram"
    assert info.value.occurrences >= 3


def test_sentence_level_repeat_detected():
    sentence = "The value I keep returning to is one hundred degrees. "
    text = (
        f"Start with the basics. {sentence}Reconsider the units. {sentence}"
        f"Check once more. {sentence}"
    )
    with pytest.raises(ThinkingLoopDetected) as info:
        ThinkingLoopDetector().check(text)
    assert info.value.strategy == "sentence"
    assert info.value.occurrences >= 3


def test_paragraph_breaks_split_sentences():
    paragraph = "This is a draft reply that keeps being rewritten with tiny changes"
    text = f"{paragraph}\n\n{paragraph}\n\n{paragraph}\n\nSomething different."
    with pytest.raises(ThinkingLoopDetected):
        ThinkingLoopDetector().check(text)


def test_markers_configurable_per_family():
    assert set(DEFAULT_LOOP_MARKERS) == {"generic", "qwen", "gemma"}
    assert all(DEFAULT_LOOP_MARKERS.values())
    draft = "A short draft of the reply that says the value is one hundred"
    marked = (
        f"*Option: {draft}\n*Wait, reconsider\n*Option: {draft}\n"
        f"*Okay, again\n*Draft: {draft}\n"
    )
    with pytest.raises(ThinkingLoopDetected):
        ThinkingLoopDetector().check(marked)
    custom = "".join(f"*Attempt: {draft}{' ' * n}\n" for n in range(3))
    # The default markers do not split these lines, so they read as one
    # sentence; a marker set that names the word splits and catches them.
    ThinkingLoopDetector().check(custom)
    with pytest.raises(ThinkingLoopDetected):
        ThinkingLoopDetector(markers=("Attempt",)).check(custom)
    detector = ThinkingLoopDetector.for_family("qwen")
    assert detector.markers == DEFAULT_LOOP_MARKERS["qwen"]
    with pytest.raises(ValueError):
        ThinkingLoopDetector.for_family("other")
    with pytest.raises(TypeError):
        ThinkingLoopDetector(markers="Wait")


def test_feed_streaming_matches_check():
    looping = PHRASE * 5
    with pytest.raises(ThinkingLoopDetected):
        ThinkingLoopDetector().check(looping)
    with pytest.raises(ThinkingLoopDetected):
        _feed_all(ThinkingLoopDetector(), looping)
    # A short loop that only completes at the close tag is still caught there.
    short = PHRASE * 3
    with pytest.raises(ThinkingLoopDetected):
        ThinkingLoopDetector().check(short)
    with pytest.raises(ThinkingLoopDetected):
        _feed_all(ThinkingLoopDetector(), short, size=7)
    ThinkingLoopDetector().check(VARIED)
    _feed_all(ThinkingLoopDetector(), VARIED)


def test_feed_checks_only_inside_reasoning():
    detector = ThinkingLoopDetector()
    # Spoken words outside a think block are never checked.
    for start in range(0, len(PHRASE * 6), 25):
        detector.feed((PHRASE * 6)[start : start + 25])
    assert detector.in_reasoning is False
    # Tags split across chunks are still recognised.
    split = ThinkingLoopDetector()
    split.feed("<thi")
    split.feed("nk>")
    assert split.in_reasoning is True
    with pytest.raises(ThinkingLoopDetected):
        for start in range(0, len(PHRASE * 6), 25):
            split.feed((PHRASE * 6)[start : start + 25])


def test_reply_that_starts_inside_reasoning_is_checked():
    detector = ThinkingLoopDetector(starts_inside=True)
    assert detector.in_reasoning is True
    with pytest.raises(ThinkingLoopDetected):
        for start in range(0, len(PHRASE * 6), 25):
            detector.feed((PHRASE * 6)[start : start + 25])
    orphan = ThinkingLoopDetector()
    for start in range(0, len(PHRASE * 3), 40):
        orphan.feed((PHRASE * 3)[start : start + 40])
    with pytest.raises(ThinkingLoopDetected):
        orphan.feed("</think>The answer.")


def test_varied_text_not_flagged():
    detector = ThinkingLoopDetector()
    detector.check(VARIED)
    numbered = "".join(
        f"{i}. Point number {i} covers a different aspect of the topic {i * 7}.\n"
        for i in range(1, 10)
    )
    detector.check(numbered)
    detector.check("short text " * 5)
    detector.check("")
    detector.check("   \n  ")
    tiny = "I see. " * 10 + "Hmm. " * 10 + "A single longer closing thought here."
    detector.check(tiny)
    abbreviations = (
        "Consider e.g. the case where a model repeats itself. "
        "The value is approximately 3.14 or so. "
        "Each sentence here is unique and different from the others. "
        "No repetition should be detected in this block of text."
    )
    detector.check(abbreviations)


def test_reset_clears_state():
    detector = ThinkingLoopDetector()
    detector.feed("<think>")
    detector.feed(PHRASE * 2)
    detector.reset()
    assert detector.in_reasoning is False
    detector.feed("<think>")
    detector.feed(PHRASE * 2)
    detector.feed("</think>")  # two repeats since the reset: below threshold
