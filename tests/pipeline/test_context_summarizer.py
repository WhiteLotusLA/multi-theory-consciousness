"""Running summary: thresholds, batches, caps, quotes and honest misses."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mtc.brain.protocol import BrainReply, FakeBrain, NoAnswerError
from mtc.pipeline.context_summarizer import (
    QUOTES_HEADING,
    SUMMARY_INSTRUCTIONS,
    SUMMARY_SYSTEM_PROMPT,
    ContextSummarizer,
    render_summary,
)
from mtc.pipeline.session_state import Session

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
CFG = SimpleNamespace(PROTECTED_NAMES="", DEBUG=False)


def _session(entries):
    return Session(key="k", conversation_id="c", history=list(entries), started_at=T0)


def _entries(count, text="A plain line about the weather report."):
    rows = []
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        speaker = "Alice" if role == "user" else "Testagent"
        rows.append({"role": role, "speaker": speaker, "content": f"{text} ({index})"})
    return rows


@pytest.mark.asyncio
async def test_below_threshold_not_needed():
    brain = FakeBrain(["unused"])
    summarizer = ContextSummarizer(brain, threshold=10, settings=CFG)
    session = _session(_entries(9))
    result = await summarizer.maybe_summarize(session)
    assert result.status == "not_needed" and result.summarized == 0
    assert brain.calls == []
    assert len(session.history) == 9


@pytest.mark.asyncio
async def test_oldest_batch_summarized_content_never_truncated():
    long_text = "word " * 600 + "and the final clause matters."
    entries = _entries(12)
    entries[1]["content"] = long_text
    brain = FakeBrain(["Alice asked about the report; it was explained."])
    summarizer = ContextSummarizer(brain, threshold=10, batch_size=4, settings=CFG)
    session = _session(entries)
    oldest = [dict(e) for e in entries[:4]]
    result = await summarizer.maybe_summarize(session)
    assert result.status == "summarized" and result.summarized == 4
    assert result.summary == "Alice asked about the report; it was explained."
    assert session.summary == result.summary
    prompt = brain.calls[0]["prompt"]
    # every message of the batch is in the prompt in full, with its speaker
    for entry in oldest:
        assert f"{entry['speaker']}: {entry['content']}" in prompt
    assert "and the final clause matters." in prompt
    # exactly the batch left the history, the rest is untouched
    assert session.history == entries[4:]


@pytest.mark.asyncio
async def test_batch_shrinks_to_char_bound():
    entries = _entries(12, text="x" * 90)
    brain = FakeBrain(["short summary."])
    summarizer = ContextSummarizer(
        brain, threshold=10, batch_size=10, max_batch_chars=300, settings=CFG
    )
    session = _session(entries)
    result = await summarizer.maybe_summarize(session)
    # each entry is about 94 characters, so three fit in 300
    assert result.summarized == 3
    assert len(session.history) == 9
    # a first entry over the bound still goes whole
    huge = _entries(12)
    huge[0]["content"] = "y" * 1000
    session = _session(huge)
    result = await ContextSummarizer(
        FakeBrain(["s."]), threshold=10, max_batch_chars=300, settings=CFG
    ).maybe_summarize(session)
    assert result.summarized == 1


@pytest.mark.asyncio
async def test_running_summary_capped():
    replies = [
        f"Sentence number {i} of the running summary is here." for i in range(10)
    ]
    summarizer = ContextSummarizer(
        FakeBrain(replies), threshold=4, batch_size=2, summary_cap=160, settings=CFG
    )
    session = _session(_entries(40))
    for _ in range(10):
        await summarizer.maybe_summarize(session)
    assert len(session.summary) <= 160
    # the newest sentence survives, the oldest were dropped whole
    assert session.summary.endswith("Sentence number 9 of the running summary is here.")
    assert "Sentence number 0 " not in session.summary
    assert session.summary.startswith("Sentence number")


@pytest.mark.asyncio
async def test_pinned_quotes_kept_verbatim():
    entries = _entries(12)
    entries[0]["content"] = "The bus was late. I promise to send the notes tonight."
    entries[2]["content"] = "I'm worried about the deadline."
    summarizer = ContextSummarizer(
        FakeBrain(["Notes and a deadline were discussed."]),
        threshold=10,
        batch_size=4,
        summary_cap=20,
        settings=CFG,
    )
    session = _session(entries)
    result = await summarizer.maybe_summarize(session)
    assert result.quotes == (
        'Alice: "I promise to send the notes tonight."',
        'Alice: "I\'m worried about the deadline."',
    )
    assert session.pinned_quotes == list(result.quotes)
    # the summary cap never eats a quote: quotes live beside the summary
    rendered = render_summary(session.summary, session.pinned_quotes)
    assert QUOTES_HEADING in rendered
    assert 'Alice: "I promise to send the notes tonight."' in rendered
    assert render_summary(None, []) is None
    # the quote ledger has its own cap and drops the oldest first
    later = _entries(12)
    later[0]["content"] = "I promise to send the notes tonight."
    later[4]["content"] = "I'm worried about the deadline."
    small = ContextSummarizer(
        FakeBrain(["s.", "t."]), threshold=4, batch_size=4, quote_cap=50, settings=CFG
    )
    session = _session(later)
    first = await small.maybe_summarize(session)
    assert first.quotes == ('Alice: "I promise to send the notes tonight."',)
    second = await small.maybe_summarize(session)
    assert second.quotes == ('Alice: "I\'m worried about the deadline."',)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        NoAnswerError("nothing", reason="empty"),
        NoAnswerError("loop", reason="loop"),
        "",
        "<think>only reasoning here</think>",
        TimeoutError("slow"),
    ],
    ids=["no_answer", "loop", "empty", "reasoning_only", "timeout"],
)
async def test_no_answer_keeps_history_and_counts(outcome):
    entries = _entries(12)
    summarizer = ContextSummarizer(
        FakeBrain([outcome]), threshold=10, batch_size=4, settings=CFG
    )
    session = _session(entries)
    session.summary = "Earlier summary."
    result = await summarizer.maybe_summarize(session)
    assert result.status == "no_answer" and result.summarized == 0
    assert session.history == entries
    assert session.summary == "Earlier summary."
    assert session.pinned_quotes == []
    failures = summarizer.status()["failures"]
    assert sum(failures.values()) == 1


@pytest.mark.asyncio
async def test_brain_error_raises_under_debug():
    debug = SimpleNamespace(PROTECTED_NAMES="", DEBUG=True)
    summarizer = ContextSummarizer(
        FakeBrain([RuntimeError("boom")]), threshold=2, settings=debug
    )
    with pytest.raises(RuntimeError):
        await summarizer.maybe_summarize(_session(_entries(4)))


@pytest.mark.asyncio
async def test_reasoning_stripped():
    summarizer = ContextSummarizer(
        FakeBrain(
            [BrainReply(text="<think>private steps</think>The report was read.")]
        ),
        threshold=10,
        settings=CFG,
    )
    session = _session(_entries(12))
    result = await summarizer.maybe_summarize(session)
    assert result.summary == "The report was read."
    assert "private steps" not in session.summary


@pytest.mark.asyncio
async def test_prompt_neutral_third_person():
    brain = FakeBrain(["A summary."])
    summarizer = ContextSummarizer(brain, threshold=10, settings=CFG)
    session = _session(_entries(12))
    session.summary = "Earlier, a plan was agreed."
    await summarizer.maybe_summarize(session)
    call = brain.calls[0]
    assert call["system_prompt"] == SUMMARY_SYSTEM_PROMPT
    assert call["thinking_mode"] is False
    text = (call["system_prompt"] + " " + call["prompt"]).lower()
    for wanted in ("third person", "facts", "decisions", "commitments", "questions"):
        assert wanted in text
    assert "Summary so far:\nEarlier, a plan was agreed." in call["prompt"]
    instructions = (SUMMARY_SYSTEM_PROMPT + " " + SUMMARY_INSTRUCTIONS).lower()
    # no relationship tuning and no second-person persona voice
    for unwanted in (
        "family",
        "love",
        "relationship",
        "you are",
        "your ",
        "warm",
        "child",
        "parent",
    ):
        assert unwanted not in instructions, unwanted


@pytest.mark.asyncio
async def test_clock_notes_roll_off_unspoken():
    entries = _entries(12)
    entries.insert(
        1,
        {
            "role": "user",
            "speaker": "clock",
            "content": "a clock note",
            "is_clock_note": True,
        },
    )
    brain = FakeBrain(["s."])
    summarizer = ContextSummarizer(brain, threshold=10, batch_size=3, settings=CFG)
    session = _session(entries)
    result = await summarizer.maybe_summarize(session)
    assert result.summarized == 3
    assert "a clock note" not in brain.calls[0]["prompt"]


def test_brain_required():
    with pytest.raises(ValueError, match="brain"):
        ContextSummarizer(None)
