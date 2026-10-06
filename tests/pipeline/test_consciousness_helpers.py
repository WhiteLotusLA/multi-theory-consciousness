"""Thought and gist reads for the prompt, competition intensity, level cache."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mtc.consciousness.continuous.schemas import (
    ThoughtIntensity,
    ThoughtRecord,
    ThoughtType,
)
from mtc.consciousness.continuous.store import InMemoryThoughtStore
from mtc.core.config import Settings
from mtc.memory.inmemory import InMemoryThoughtGistStore
from mtc.memory.protocols import StoreUnavailable
from mtc.memory.thought_gist import ThoughtGist
from mtc.pipeline.consciousness_helpers import (
    GIST_MIN_RETENTION,
    ConsciousnessLevelCache,
    get_last_competition_intensity,
    get_recent_ctm_thoughts,
    get_thought_gists,
)

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _thought(index, content, thought_type=ThoughtType.REFLECTION, express=True):
    return ThoughtRecord(
        thought_id=f"t{index}",
        timestamp=T0 + timedelta(minutes=index),
        thought_type=thought_type,
        content=content,
        intensity=ThoughtIntensity.MODERATE,
        salience_score=0.5,
        emotional_context={"curiosity": 0.7, "calm": 0.2},
        should_express=express,
    )


@pytest.mark.asyncio
async def test_ctm_thoughts_via_surfacing_dream_excluded():
    store = InMemoryThoughtStore()
    await store.insert(_thought(1, "a reflection on the garden"))
    await store.insert(_thought(2, "a dream of the sea", ThoughtType.DREAM))
    await store.insert(_thought(3, "not marked expressible", express=False))
    await store.insert(_thought(4, "a question about tides", ThoughtType.QUESTION))
    await store.insert(_thought(5, "a reflection on the garden"))
    items = await get_recent_ctm_thoughts(store, limit=5)
    assert [i["content"] for i in items] == [
        "a reflection on the garden",
        "a question about tides",
    ]
    assert items[0]["thought_id"] == "t5"
    assert items[0]["emotional_tone"] == "curiosity"
    assert items[1]["thought_type"] == "question"
    assert set(items[0]) >= {"content", "timestamp", "thought_type", "emotional_tone"}
    assert await get_recent_ctm_thoughts(None) == []
    assert await get_recent_ctm_thoughts(store, limit=0) == []
    assert len(await get_recent_ctm_thoughts(store, limit=1)) == 1


@pytest.mark.asyncio
async def test_gists_min_retention_half():
    store = InMemoryThoughtGistStore()
    for index, (retention, importance) in enumerate(
        [(0.9, 0.3), (0.5, 0.8), (0.49, 0.99), (0.7, 0.5)]
    ):
        await store.add_gist(
            ThoughtGist(
                gist_id=f"g{index}",
                created_at=T0 + timedelta(hours=index),
                summary=f"gist {index}",
                source_count=3 + index,
                source_thought_ids=[f"t{index}"],
                importance_score=importance,
                retention_strength=retention,
            )
        )
    assert GIST_MIN_RETENTION == 0.5
    items = await get_thought_gists(store, limit=5)
    assert [i["gist_id"] for i in items] == ["g1", "g3", "g0"]
    assert items[0]["source_count"] == 4
    assert items[0]["dominant_emotion"] is None
    assert await get_thought_gists(None) == []
    assert len(await get_thought_gists(store, limit=2)) == 2


@pytest.mark.asyncio
async def test_store_failure_propagates():
    class Down:
        async def expressible(self, *args, **kwargs):
            raise StoreUnavailable("synthetic outage")

    with pytest.raises(StoreUnavailable):
        await get_recent_ctm_thoughts(Down())


def _event(candidates, winners):
    return SimpleNamespace(candidates_count=candidates, winners_count=winners)


def test_competition_intensity_none_without_history():
    workspace = SimpleNamespace(bottleneck=SimpleNamespace(competition_history=[]))
    assert get_last_competition_intensity(workspace) is None
    assert get_last_competition_intensity(None) is None
    assert get_last_competition_intensity(SimpleNamespace()) is None
    workspace.bottleneck.competition_history.append(_event(0, 0))
    assert get_last_competition_intensity(workspace) is None
    workspace.bottleneck.competition_history.append(_event(8, 2))
    assert get_last_competition_intensity(workspace) == pytest.approx(0.75)


def test_competition_intensity_reads_the_real_workspace():
    from mtc.consciousness.enhanced_global_workspace import EnhancedGlobalWorkspace

    workspace = EnhancedGlobalWorkspace()
    assert get_last_competition_intensity(workspace) is None


class _Monotonic:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.mark.asyncio
async def test_level_cache_none_when_insufficient():
    cache = ConsciousnessLevelCache(lambda: None, clock=_Monotonic())
    assert await cache.level() is None
    assert cache.status()["value"] is None

    async def failing():
        raise RuntimeError("assessment failed")

    clock = _Monotonic()
    broken = ConsciousnessLevelCache(
        failing, min_interval_s=0.0, clock=clock, settings=Settings(DEBUG=False)
    )
    assert await broken.level() is None
    assert broken.status()["failures"] == 1
    with pytest.raises(ValueError):
        ConsciousnessLevelCache(None)


@pytest.mark.asyncio
async def test_level_cache_failure_raises_under_debug():
    async def failing():
        raise RuntimeError("assessment failed")

    loud = ConsciousnessLevelCache(
        failing, min_interval_s=0.0, clock=_Monotonic(), settings=Settings(DEBUG=True)
    )
    with pytest.raises(RuntimeError, match="assessment failed"):
        await loud.level()
    assert loud.status()["failures"] == 1


@pytest.mark.asyncio
async def test_level_cache_throttles():
    values = iter([0.4, 0.6, 0.8])
    calls = []

    async def source():
        calls.append(1)
        return next(values)

    clock = _Monotonic()
    cache = ConsciousnessLevelCache(source, min_interval_s=300.0, clock=clock)
    assert await cache.level() == 0.4
    clock.now += 100.0
    assert await cache.level() == 0.4
    assert len(calls) == 1
    clock.now += 250.0
    assert await cache.level() == 0.6
    assert len(calls) == 2
    # a sync source works too
    sync = ConsciousnessLevelCache(lambda: 0.5, clock=clock)
    assert await sync.level() == 0.5
