"""ThoughtBuffer: continuity, bounds, themes, statistics and state."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from mtc.consciousness.continuous import buffer as buffer_module
from mtc.consciousness.continuous.buffer import (
    DECAY_INTERVAL,
    ThoughtBuffer,
    ThoughtStream,
)
from mtc.consciousness.continuous.config import CTMConfig
from mtc.consciousness.continuous.schemas import (
    ThoughtIntensity,
    ThoughtRecord,
    ThoughtType,
)

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _record(
    thought_id,
    seconds=0,
    text=None,
    kind=ThoughtType.REFLECTION,
    intensity=ThoughtIntensity.MODERATE,
    **extra,
):
    return ThoughtRecord(
        thought_id=thought_id,
        timestamp=T0 + timedelta(seconds=seconds),
        thought_type=kind,
        content=text or f"text of {thought_id}",
        intensity=intensity,
        salience_score=0.5,
        **extra,
    )


async def _add(buf, *records):
    return [await buf.add_thought(r) for r in records]


def test_continuity_links():
    buf = ThoughtBuffer()
    a, b, c = _record("a", 0), _record("b", 1), _record("c", 2)
    returned = asyncio.run(_add(buf, a, b, c))
    assert returned == [None, "a", "b"]
    assert a.previous_thought_id is None and a.leads_to_thought_id == "b"
    assert b.previous_thought_id == "a" and b.leads_to_thought_id == "c"
    assert c.previous_thought_id == "b" and c.leads_to_thought_id is None
    assert buf.stream_state.current_thought_id == "c"
    assert buf.stream_state.total_thoughts_today == 3
    # A caller-set parent is kept and returned.
    d = _record("d", 3, previous_thought_id="a")
    assert asyncio.run(buf.add_thought(d)) == "a"
    assert d.previous_thought_id == "a"


def test_bounded_buffer():
    buf = ThoughtBuffer(max_buffer_size=5)
    asyncio.run(_add(buf, *[_record(f"t{i}", i) for i in range(12)]))
    assert len(buf.current_thoughts) == 5 and buf.max_buffer_size == 5
    assert [t.thought_id for t in buf.get_recent_thoughts(10)] == [
        "t11",
        "t10",
        "t9",
        "t8",
        "t7",
    ]
    assert [t.thought_id for t in buf.get_recent_thoughts(count=2)] == ["t11", "t10"]
    assert buf.get_recent_thoughts(0) == []
    # Counters keep counting past the bound.
    assert buf.stream_state.total_thoughts_today == 12
    with pytest.raises(ValueError):
        ThoughtBuffer(max_buffer_size=0)


def test_by_type_newest_first():
    buf = ThoughtBuffer()
    asyncio.run(
        _add(
            buf,
            _record("q1", 0, kind=ThoughtType.QUESTION),
            _record("r1", 1),
            _record("q2", 2, kind=ThoughtType.QUESTION),
        )
    )
    assert [t.thought_id for t in buf.get_thoughts_by_type(ThoughtType.QUESTION)] == [
        "q2",
        "q1",
    ]
    assert [t.thought_id for t in buf.get_thoughts_by_type("question", 1)] == ["q2"]
    with pytest.raises(ValueError):
        buf.get_thoughts_by_type("nonesuch")


def test_theme_decay_halves_every_500():
    assert DECAY_INTERVAL == 500
    buf = ThoughtBuffer()
    records = [_record(f"t{i}", i, text="a memory of the past") for i in range(499)]
    asyncio.run(_add(buf, *records))
    assert buf.theme_counter["memory"] == 499
    assert buf.type_counter["reflection"] == 499
    # The 500th thought triggers the halving: (499 + 1) // 2.
    asyncio.run(buf.add_thought(_record("t499", 499, text="a memory of the past")))
    assert buf.theme_counter["memory"] == 250
    assert buf.type_counter["reflection"] == 250
    assert buf.stream_state.thoughts_per_type == {"reflection": 250}
    # A theme that halves to zero is dropped.
    buf2 = ThoughtBuffer()
    asyncio.run(buf2.add_thought(_record("x", 0, text="one time wonder")))
    for i in range(498):
        asyncio.run(buf2.add_thought(_record(f"y{i}", i + 1, text="plain words")))
    asyncio.run(buf2.add_thought(_record("z", 600, text="plain words")))
    assert "time" not in buf2.theme_counter and "curiosity" not in buf2.theme_counter


def test_theme_keywords_configurable_and_generic():
    default = ThoughtBuffer()
    assert default.theme_keywords == CTMConfig().theme_keywords
    assert len(default.theme_keywords) == 8
    assert default.extract_themes("I remember the past, and wonder") == [
        "memory",
        "curiosity",
    ]
    # Whole-word-start matching: no hit inside another word.
    assert default.extract_themes("sometimes and everywhere") == []
    assert default.extract_themes("Remembering, FEELINGS") == ["memory", "emotion"]
    custom = ThoughtBuffer(theme_keywords={"weather": ["rain", "sun"], "empty": []})
    assert custom.extract_themes("rainy and sunny") == ["weather"]
    assert custom.theme_keywords == {"weather": ["rain", "sun"], "empty": []}
    assert ThoughtBuffer(theme_keywords={}).extract_themes("anything") == []
    asyncio.run(custom.add_thought(_record("w", 0, text="rain again")))
    assert custom.get_recent_themes() == ["weather"]


def test_dominant_themes_top_five_most_common_first():
    buf = ThoughtBuffer()
    texts = (
        ["a memory"] * 5
        + ["a feeling"] * 4
        + ["the future"] * 3
        + ["to learn"] * 2
        + ["to wonder", "to grow", "together"]
    )
    asyncio.run(_add(buf, *[_record(f"t{i}", i, text=t) for i, t in enumerate(texts)]))
    themes = buf.get_recent_themes()
    assert len(themes) == 5
    assert themes[:4] == ["memory", "emotion", "time", "learning"]


def test_intensity_and_frequency_none_until_measurable():
    buf = ThoughtBuffer()
    assert buf.stream_intensity() is None and buf.thought_frequency() is None
    summary = buf.get_stream_summary()
    assert summary["current_intensity"] is None
    assert summary["thought_frequency"] is None
    assert summary["last_thought_time"] is None
    asyncio.run(buf.add_thought(_record("a", 0, intensity=ThoughtIntensity.WHISPER)))
    assert buf.stream_intensity() == pytest.approx(0.1)
    assert buf.thought_frequency() is None  # one thought has no span
    asyncio.run(buf.add_thought(_record("b", 0, intensity=ThoughtIntensity.URGENT)))
    assert buf.thought_frequency() is None  # a zero span
    asyncio.run(buf.add_thought(_record("c", 20, intensity=ThoughtIntensity.STRONG)))
    assert buf.stream_intensity() == pytest.approx((0.1 + 0.9 + 0.7) / 3)
    assert buf.thought_frequency() == pytest.approx(2 / 20)
    assert buf.stream_state.current_intensity == pytest.approx(1.7 / 3)


def test_unmeasured_stream_state_persists_as_none():
    buf = ThoughtBuffer()
    assert buf.stream_state.thought_frequency is None
    assert buf.stream_state.current_intensity is None
    state = buf.to_state_dict()
    assert state["scalars"]["thought_frequency"] is None
    assert state["scalars"]["current_intensity"] is None
    restored = ThoughtBuffer()
    restored.from_state_dict(json.loads(json.dumps(state)))
    assert restored.stream_state.thought_frequency is None
    assert restored.stream_state.current_intensity is None
    assert restored.to_state_dict() == state
    # A saved value that is not a finite number restores as unmeasured.
    for bad in ("fast", float("nan"), float("inf"), True, [0.1], {"v": 1}):
        state["scalars"]["thought_frequency"] = bad
        state["scalars"]["current_intensity"] = bad
        restored.from_state_dict(state)
        assert restored.stream_state.thought_frequency is None, bad
        assert restored.stream_state.current_intensity is None, bad
    # A measured value survives the round trip.
    state["scalars"]["thought_frequency"] = 0.05
    state["scalars"]["current_intensity"] = 0.5
    restored.from_state_dict(state)
    assert restored.stream_state.thought_frequency == 0.05
    assert restored.stream_state.current_intensity == 0.5
    # Once the buffer cannot measure them, the state says so again.
    summary = restored.get_stream_summary()
    assert summary["thought_frequency"] is None
    assert summary["current_intensity"] is None
    assert restored.stream_state.thought_frequency is None
    assert restored.stream_state.current_intensity is None
    assert restored.stream_state.to_dict()["thought_frequency"] is None


def test_saved_thought_without_state_restores_none():
    buf = ThoughtBuffer()
    asyncio.run(buf.add_thought(_record("a", 0, consciousness_state="drowsy")))
    asyncio.run(buf.add_thought(_record("b", 1)))
    state = buf.to_state_dict()
    saved = state["histories"]["current_thoughts"]
    assert saved[0]["consciousness_state"] == "drowsy"
    assert saved[1]["consciousness_state"] is None
    del saved[0]["consciousness_state"]
    restored = ThoughtBuffer()
    restored.from_state_dict(state)
    assert [t.consciousness_state for t in restored.current_thoughts] == [None, None]
    assert restored.skipped_on_restore == 0


def test_stream_summary_and_emotional_tone():
    buf = ThoughtBuffer()
    asyncio.run(
        _add(
            buf,
            _record("a", 0, emotional_context={"calm": 0.6}),
            _record("b", 10, kind=ThoughtType.QUESTION),
        )
    )
    summary = buf.get_stream_summary()
    assert summary["total_thoughts_today"] == 2 and summary["buffer_size"] == 2
    assert summary["thoughts_per_type"] == {"reflection": 1, "question": 1}
    assert summary["emotional_tone"] == {"calm": 0.6}  # the last non-empty context
    assert summary["last_thought_time"] == (T0 + timedelta(seconds=10)).isoformat()
    json.dumps(summary)


def test_reset_daily_stats_keeps_themes():
    buf = ThoughtBuffer()
    asyncio.run(buf.add_thought(_record("a", 0, text="a memory")))
    buf.reset_daily_stats()
    assert buf.stream_state.total_thoughts_today == 0
    assert buf.stream_state.thoughts_per_type == {} and not buf.type_counter
    assert buf.theme_counter["memory"] == 1
    assert len(buf.current_thoughts) == 1


def test_state_dict_round_trip():
    buf = ThoughtBuffer(max_buffer_size=10)
    records = [
        _record(
            f"t{i}",
            i * 5,
            text=f"a memory number {i}",
            kind=ThoughtType.MEMORY_REPLAY if i % 2 else ThoughtType.QUESTION,
            emotional_context={"calm": 0.1 * i},
            associated_memories=[f"m{i}"],
            processing_context={"strategy": "memory", "n": i},
            should_express=bool(i % 2),
            stream_id=f"burst-{i // 2}",
            expressed_at=T0 + timedelta(seconds=i) if i == 1 else None,
        )
        for i in range(4)
    ]
    asyncio.run(_add(buf, *records))
    state = buf.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}
    restored = ThoughtBuffer(max_buffer_size=10)
    restored.from_state_dict(json.loads(json.dumps(state)))
    assert list(restored.current_thoughts) == list(buf.current_thoughts)
    assert restored.to_state_dict() == state
    assert restored.stream_state.current_thought_id == "t3"
    assert restored.stream_state.last_thought_time == buf.stream_state.last_thought_time
    assert restored.stream_state.total_thoughts_today == 4
    assert restored.theme_counter == buf.theme_counter
    # The restored buffer keeps its bound.
    assert restored.max_buffer_size == 10
    # And continues the chain from the restored current thought.
    nxt = _record("t4", 100)
    assert asyncio.run(restored.add_thought(nxt)) == "t3"


def test_restore_skips_unreadable_thoughts_without_defaults():
    buf = ThoughtBuffer()
    asyncio.run(_add(buf, _record("a", 0), _record("b", 1)))
    state = buf.to_state_dict()
    state["histories"]["current_thoughts"][0]["thought_type"] = "nonesuch"
    state["histories"]["current_thoughts"][1]["intensity"] = None
    state["histories"]["current_thoughts"].append({"thought_id": "c"})
    restored = ThoughtBuffer()
    restored.from_state_dict(state)
    assert len(restored.current_thoughts) == 0
    assert restored.skipped_on_restore == 3
    restored.from_state_dict({})  # an empty state restores an empty buffer
    assert restored.stream_state.current_thought_id is None


def test_alias_thought_stream():
    assert ThoughtStream is ThoughtBuffer
    assert buffer_module.ThoughtStream is ThoughtBuffer


def test_buffer_holds_no_store_or_privacy_field():
    buf = ThoughtBuffer()
    assert not hasattr(buf, "mongodb") and not hasattr(buf, "mark_private_by_stream_id")
    assert not any("private" in name for name in vars(buf))
