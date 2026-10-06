"""Surfacing: which thoughts and gists the prompt may see, as plain data."""

from datetime import datetime, timedelta, timezone

import pytest

from mtc.brain.prompt import PromptAssembler, PromptInputs
from mtc.consciousness.continuous.schemas import (
    ThoughtIntensity,
    ThoughtRecord,
    ThoughtType,
)
from mtc.consciousness.continuous.store import InMemoryThoughtStore
from mtc.consciousness.continuous.surfacing import (
    SurfacedThought,
    gist_items,
    get_gists,
    get_surfaced_thoughts,
    thought_items,
)
from mtc.memory.inmemory import InMemoryThoughtGistStore
from mtc.memory.thought_gist import ThoughtGist

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _record(
    thought_id, minutes=0, express=True, kind=ThoughtType.REFLECTION, text=None, **extra
):
    return ThoughtRecord(
        thought_id=thought_id,
        timestamp=T0 + timedelta(minutes=minutes),
        thought_type=kind,
        content=text or f"text of {thought_id}",
        intensity=ThoughtIntensity.QUIET,
        salience_score=0.5,
        should_express=express,
        **extra,
    )


def _gist(gist_id, retention=1.0, importance=0.5, minutes=0, count=3):
    return ThoughtGist(
        gist_id=gist_id,
        created_at=T0 + timedelta(minutes=minutes),
        summary=f"summary {gist_id}",
        source_count=count,
        source_thought_ids=["a", "b"],
        dominant_emotion="calm",
        importance_score=importance,
        retention_strength=retention,
    )


@pytest.mark.asyncio
async def test_dream_types_excluded():
    store = InMemoryThoughtStore()
    await store.insert(_record("w", 0))
    await store.insert(_record("d", 1, kind=ThoughtType.DREAM))
    surfaced = await get_surfaced_thoughts(store)
    assert [t.thought_id for t in surfaced] == ["w"]
    # The prefix list is the caller's choice.
    both = await get_surfaced_thoughts(store, exclude_type_prefixes=())
    assert [t.thought_id for t in both] == ["d", "w"]


@pytest.mark.asyncio
async def test_only_expressed():
    store = InMemoryThoughtStore()
    await store.insert(_record("quiet", 0, express=False))
    await store.insert(_record("loud", 1, express=True))
    surfaced = await get_surfaced_thoughts(store, limit=5)
    assert [t.thought_id for t in surfaced] == ["loud"]
    assert await get_surfaced_thoughts(store, limit=0) == []


@pytest.mark.asyncio
async def test_dedup_identical_texts():
    store = InMemoryThoughtStore()
    await store.insert(_record("first", 0, text="same words"))
    await store.insert(_record("second", 1, text="same words"))
    await store.insert(_record("other", 2, text="other words"))
    surfaced = await get_surfaced_thoughts(store)
    assert [t.thought_id for t in surfaced] == ["other", "second"]


@pytest.mark.asyncio
async def test_surfaced_thought_fields():
    store = InMemoryThoughtStore()
    await store.insert(
        _record(
            "t",
            0,
            emotional_context={"calm": 0.3, "joy": 0.8},
            stream_id="burst-1",
            kind=ThoughtType.QUESTION,
        )
    )
    await store.insert(_record("n", 1, text="no emotion"))
    by_id = {t.thought_id: t for t in await get_surfaced_thoughts(store)}
    assert by_id["t"] == SurfacedThought(
        thought_id="t",
        content="text of t",
        timestamp=T0,
        thought_type="question",
        emotional_tone="joy",
        stream_id="burst-1",
    )
    assert by_id["n"].emotional_tone is None


@pytest.mark.asyncio
async def test_gists_by_importance_with_min_retention():
    store = InMemoryThoughtGistStore()
    await store.add_gist(_gist("low", importance=0.2, retention=0.9, minutes=0))
    await store.add_gist(_gist("top", importance=0.9, retention=0.6, minutes=1))
    await store.add_gist(_gist("faded", importance=1.0, retention=0.4, minutes=2))
    await store.add_gist(_gist("mid", importance=0.5, retention=0.5, minutes=3))
    gists = await get_gists(store, limit=5, min_retention=0.5)
    assert [g.gist_id for g in gists] == ["top", "mid", "low"]  # faded is below 0.5
    assert [g.gist_id for g in await get_gists(store, limit=1)] == ["top"]
    assert await get_gists(store, limit=0) == []
    assert [g.gist_id for g in await get_gists(store, min_retention=0.0)][0] == "faded"


def test_gist_count_key_is_source_count():
    items = gist_items([_gist("g", count=4)])
    assert items[0]["source_count"] == 4
    assert "thought_count" not in items[0] and "count" not in items[0]
    assert items[0]["summary"] == "summary g"
    assert items[0]["dominant_emotion"] == "calm"


def test_no_phi_in_items():
    items = thought_items([SurfacedThought("t", "c", T0, "reflection", None, "s")])
    items += gist_items([_gist("g")])
    for item in items:
        assert not any("phi" in key.lower() for key in item)


def test_item_keys_match_prompt_assembler():
    thoughts = thought_items(
        [
            SurfacedThought("t1", "a thought text", T0, "reflection", "calm", "s1"),
            SurfacedThought("t2", "another thought", T0, "question", None, "s2"),
        ]
    )
    gists = gist_items([_gist("g1", count=1), _gist("g2", count=3)])
    inputs = PromptInputs(ctm_thoughts=thoughts, thought_gists=gists)
    text = PromptAssembler().render(inputs).text
    assert "a thought text" in text and "(reflection)" in text
    assert "another thought" in text
    assert "summary g1 (from 1 thought)" in text
    assert "summary g2 (from 3 thoughts)" in text
    # Mood stamps stay off: the tone travels as data and is not narrated.
    assert "[mood:" not in text
    loud = PromptAssembler(narrate_thought_mood=True).render(inputs).text
    assert "a thought text [mood: calm]" in loud
    assert "(from 1 thought) [mood: calm]" in loud
