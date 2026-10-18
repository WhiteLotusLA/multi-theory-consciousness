"""LLMThoughtEnhancer: honest misses and the prompts it sends."""

import asyncio
import logging
from datetime import datetime, timezone

import pytest

from mtc.brain.protocol import (
    BrainReply,
    BrainUnavailableError,
    FakeBrain,
    NoAnswerError,
)
from mtc.consciousness.continuous.config import CTMConfig
from mtc.consciousness.continuous.enhancer import (
    CATEGORY_TO_TYPE,
    LLMThoughtEnhancer,
)
from mtc.consciousness.continuous.persona import CATEGORIES, CTMPersona
from mtc.consciousness.continuous.schemas import ThoughtIntensity, ThoughtType
from mtc.core.config import Settings

NOW = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def persona():
    return CTMPersona(
        name="Testagent", system_prompt=("You are {name}. Write one brief thought.")
    )


def _enhancer(brain, persona, config=None, **kwargs):
    kwargs.setdefault("now", lambda: NOW)
    kwargs.setdefault("narrate_mood", False)
    return LLMThoughtEnhancer(brain, persona, config or CTMConfig(), **kwargs)


def _run(enhancer, category="reflection", context=None, state="awake"):
    return asyncio.run(enhancer.generate(category, context, state))


def test_category_map_covers_five_categories():
    assert set(CATEGORY_TO_TYPE) == set(CATEGORIES)
    assert CATEGORY_TO_TYPE["memory"] is ThoughtType.MEMORY_REPLAY
    assert CATEGORY_TO_TYPE["emotion"] is ThoughtType.EMOTION


def test_success_builds_a_record(persona):
    brain = FakeBrain(["I wonder how quiet a morning can be."])
    enh = _enhancer(brain, persona)
    record = _run(enh, "reflection", state="drowsy")
    assert record.content == "I wonder how quiet a morning can be."
    assert record.thought_type is ThoughtType.REFLECTION
    assert record.consciousness_state == "drowsy"
    assert record.timestamp == NOW and record.triggered_by == "llm_reflection"
    assert record.should_express is False  # the novelty gate decides later
    assert record.salience_score == 0.8
    # the 0.8 is a design constant, and the record says so and names its form
    assert record.processing_context["salience_basis"] == "llm_design_constant"
    assert record.processing_context["form"] == "model"
    assert isinstance(record.intensity, ThoughtIntensity)
    assert enh.successes == 1 and enh.misses == 0 and enh.last_miss_reason is None
    call = brain.calls[0]
    assert call["max_tokens"] == 150 and call["temperature"] == 0.8
    assert call["thinking_mode"] is False
    assert call["system_prompt"] == "You are Testagent. Write one brief thought."


def test_memory_prompt_receives_recent_memory(persona):
    # The reference read a key its callers never sent; this one reads the key
    # the machine sends, and a missing key is a named miss.
    brain = FakeBrain(["A thought about the walk."])
    enh = _enhancer(brain, persona)
    record = _run(enh, "memory", {"recent_memory": "a walk by the river"})
    assert record is not None and record.thought_type is ThoughtType.MEMORY_REPLAY
    assert "a walk by the river" in brain.calls[0]["prompt"]
    assert _run(enh, "memory", {}) is None
    assert enh.last_miss_reason == "missing_context"
    assert len(brain.calls) == 1  # the brain is not asked without the memory


def test_question_prompt_receives_recent_topic(persona):
    brain = FakeBrain(["What stays?", "What else stays?"])
    enh = _enhancer(brain, persona)
    assert _run(enh, "question", {"recent_topic": "gardens"}) is not None
    assert "gardens" in brain.calls[0]["prompt"]
    # The topic is optional: no topic, no topic line, still a thought.
    assert _run(enh, "question", {}) is not None
    assert "topic" not in brain.calls[1]["prompt"].lower()


def test_emotion_prompt_and_record_carry_the_emotions(persona):
    brain = FakeBrain(["Something settled."])
    enh = _enhancer(brain, persona)
    emotions = {"calm": 0.7, "joy": 0.2, "bad": "text", "flag": True}
    record = _run(enh, "emotion", {"emotions": emotions})
    assert "calm" in brain.calls[0]["prompt"]
    assert record.emotional_context == {"calm": 0.7, "joy": 0.2}
    assert _run(enh, "emotion", {}) is None
    assert enh.last_miss_reason == "missing_context"


def test_non_finite_emotion_scores_are_not_stored(persona):
    enh = _enhancer(FakeBrain(["Something settled."]), persona)
    emotions = {
        "calm": 0.7,
        "odd": float("nan"),
        "huge": float("inf"),
        "low": float("-inf"),
    }
    record = _run(enh, "emotion", {"emotions": emotions})
    assert record.emotional_context == {"calm": 0.7}


def test_empty_reply_is_miss_nothing_returned(persona):
    brain = FakeBrain(["", "   \n ", BrainReply(text="")])
    enh = _enhancer(brain, persona)
    for _ in range(3):
        assert _run(enh) is None
    assert enh.misses == 3 and enh.successes == 0
    assert enh.last_miss_reason == "empty_reply"
    assert enh.miss_reasons == {"empty_reply": 3}


def test_placeholder_prefix_from_config_is_miss(persona):
    config = CTMConfig()
    config.llm.placeholder_prefixes = ["[placeholder"]
    brain = FakeBrain(["[placeholder text here]", "A real thought."])
    enh = _enhancer(brain, persona, config)
    assert _run(enh) is None
    assert enh.last_miss_reason == "placeholder_reply"
    assert _run(enh).content == "A real thought."
    # With no prefixes configured nothing is treated as a placeholder.
    plain = _enhancer(FakeBrain(["[placeholder text here]"]), persona)
    assert _run(plain).content == "[placeholder text here]"


def test_no_answer_error_is_miss(persona):
    brain = FakeBrain(
        [
            NoAnswerError("nothing", reason="empty"),
            NoAnswerError("only reasoning", reason="reasoning_only"),
            NoAnswerError("loop", reason="loop"),
        ]
    )
    enh = _enhancer(brain, persona)
    for _ in range(3):
        assert _run(enh) is None
    assert enh.miss_reasons == {
        "no_answer_empty": 1,
        "no_answer_reasoning_only": 1,
        "no_answer_loop": 1,
    }


def test_reasoning_stripped(persona):
    brain = FakeBrain(
        [
            "<think>let me plan the thought</think>\n\nThe light is changing.",
            "<think>only reasoning here</think>",
            "<think>never closed and no words",
        ]
    )
    enh = _enhancer(brain, persona)
    record = _run(enh)
    assert record.content == "The light is changing."
    assert "plan" not in record.content
    assert _run(enh) is None and enh.last_miss_reason == "reasoning_only"
    assert _run(enh) is None and enh.last_miss_reason == "reasoning_only"


def test_never_raises(persona):
    errors = [
        BrainUnavailableError("down"),
        asyncio.TimeoutError(),
        RuntimeError("boom"),
        ValueError("bad"),
    ]
    enh = _enhancer(FakeBrain(errors), persona)
    for _ in errors:
        assert _run(enh) is None
    assert enh.misses == 4 and enh.successes == 0
    assert enh.miss_reasons == {
        "error_BrainUnavailableError": 1,
        "error_TimeoutError": 1,
        "error_RuntimeError": 1,
        "error_ValueError": 1,
    }
    # Bad inputs are misses too.
    enh2 = _enhancer(FakeBrain(), persona)
    assert _run(enh2, "dream") is None and enh2.last_miss_reason == "unknown_category"
    assert _run(enh2, "reflection", state="napping") is None
    assert enh2.last_miss_reason == "unknown_state"
    assert _run(enh2, "reflection", {"emotions": object()}) is not None
    none_brain = LLMThoughtEnhancer(None, persona, CTMConfig())
    assert asyncio.run(none_brain.generate("reflection")) is None
    assert none_brain.last_miss_reason == "no_brain"


def test_unexpected_error_raises_under_debug(persona):
    loud = _enhancer(
        FakeBrain([RuntimeError("boom"), BrainUnavailableError("down")]),
        persona,
        settings=Settings(DEBUG=True),
    )
    with pytest.raises(RuntimeError):
        _run(loud)
    with pytest.raises(BrainUnavailableError):
        _run(loud)
    # counted first, then raised
    assert loud.miss_reasons == {
        "error_RuntimeError": 1,
        "error_BrainUnavailableError": 1,
    }
    assert loud.misses == 2 and loud.last_miss_reason == "error_BrainUnavailableError"
    # a NoAnswerError stays a plain miss in every mode
    quiet = _enhancer(
        FakeBrain([NoAnswerError("nothing", reason="empty")]),
        persona,
        settings=Settings(DEBUG=True),
    )
    assert _run(quiet) is None
    assert quiet.miss_reasons == {"no_answer_empty": 1}
    # and with DEBUG off the same error is a miss
    soft = _enhancer(
        FakeBrain([RuntimeError("boom")]), persona, settings=Settings(DEBUG=False)
    )
    assert _run(soft) is None
    assert soft.miss_reasons == {"error_RuntimeError": 1}


def test_cancellation_is_not_swallowed(persona):
    enh = _enhancer(FakeBrain([asyncio.CancelledError()]), persona)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(enh.generate("reflection"))


def test_clock_injected_only(persona):
    plain = FakeBrain(["One.", "Two."])
    enh = _enhancer(plain, persona)
    _run(enh)
    assert plain.calls[0]["system_prompt"] == persona.render_system()

    brain = FakeBrain(["One.", "Two.", "Three.", "Four."])
    calls = []

    def clock():
        calls.append(1)
        return "It is a fixed test moment."

    enh = _enhancer(brain, persona, clock=clock)
    _run(enh, state="awake")
    _run(enh, state="drowsy")
    _run(enh, state="dreaming")
    _run(enh, state="deep_sleep")
    prompts = [c["system_prompt"] for c in brain.calls]
    assert prompts[0].endswith("It is a fixed test moment.")
    assert prompts[1].endswith("It is a fixed test moment.")
    assert "fixed test moment" not in prompts[2] + prompts[3]  # dreams stay unclocked
    assert len(calls) == 2


def test_broken_clock_costs_no_thought(persona, caplog):
    def clock():
        raise RuntimeError("no clock")

    brain = FakeBrain(["One.", "Two."])
    enh = _enhancer(brain, persona, clock=clock)
    with caplog.at_level(logging.WARNING):
        assert _run(enh) is not None and _run(enh) is not None
    assert brain.calls[0]["system_prompt"] == persona.render_system()
    warnings = [r for r in caplog.records if "clock unavailable" in r.getMessage()]
    assert len(warnings) == 1  # warned once


def test_mood_not_narrated_by_default(persona, monkeypatch):
    emotions = {"calm": 0.9}
    brain = FakeBrain(["One.", "Two.", "Three.", "Four."])
    # Default flags (every NARRATE_ key false): non-emotion prompts carry no mood.
    monkeypatch.setattr(
        "mtc.core.narration.get_settings", lambda: Settings(NARRATE_ALL=False)
    )
    enh = LLMThoughtEnhancer(brain, persona, CTMConfig(), now=lambda: NOW)
    for category in ("reflection", "creative", "question"):
        record = _run(enh, category, {"emotions": emotions, "recent_topic": "x"})
        assert record.emotional_context == {"calm": 0.9}  # stored as data
    for call in brain.calls:
        assert "calm" not in call["prompt"] and "emotional" not in call["prompt"]
    # With the flag on, the mood reaches the prompt.
    monkeypatch.setattr(
        "mtc.core.narration.get_settings", lambda: Settings(NARRATE_THOUGHT_MOOD=True)
    )
    _run(enh, "reflection", {"emotions": emotions})
    assert "calm" in brain.calls[-1]["prompt"]
    # An explicit argument beats the setting.
    quiet = _enhancer(FakeBrain(["x."]), persona, narrate_mood=False)
    _run(quiet, "creative", {"emotions": emotions})
    assert "calm" not in quiet.brain.calls[0]["prompt"]


def test_misses_log_first_and_every_hundredth(persona, caplog):
    enh = _enhancer(FakeBrain(), persona)
    enh.brain = FakeBrain([""] * 101)
    with caplog.at_level(logging.WARNING):
        for _ in range(101):
            _run(enh)
    lines = [r.getMessage() for r in caplog.records if "thought miss" in r.getMessage()]
    assert len(lines) == 2 and "#1:" in lines[0] and "#100:" in lines[1]
    assert not any("text of" in line for line in lines)


def test_status_reports_counters(persona):
    enh = _enhancer(FakeBrain(["A thought.", ""]), persona)
    _run(enh)
    _run(enh)
    status = enh.status()
    assert status["successes"] == 1 and status["misses"] == 1
    assert status["last_miss_reason"] == "empty_reply"
    assert status["has_clock"] is False
