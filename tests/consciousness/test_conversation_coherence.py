"""The conversation summary keeps its words apart: the overflow note after the
first five topics is separated from the last topic by a space."""

from mtc.consciousness.conversation_coherence import ConversationCoherence

HISTORY = [
    {"role": "user", "speaker": "Alice", "content": "Shall we plan a walk?"},
    {"role": "user", "speaker": "Guest", "content": "Yes, if the weather holds."},
]

TOPICS = ["walk", "weather", "river", "lunch", "map", "boots", "rain"]


def test_summary_separates_the_overflow_note():
    summary = ConversationCoherence().build_conversation_summary(
        HISTORY, TOPICS, max_length=1000
    )
    expected = ", ".join(TOPICS[:5]) + " (and 2 more topics)"
    assert expected in summary
    assert "map(and" not in summary


def test_summary_with_five_topics_has_no_overflow_note():
    summary = ConversationCoherence().build_conversation_summary(
        HISTORY, TOPICS[:5], max_length=1000
    )
    assert "Main topics: " + ", ".join(TOPICS[:5]) in summary
    assert "more topics" not in summary
