"""Episodic store contract, run over every backend.

The in-memory store always runs. The PostgreSQL store runs under the ``db``
marker on a throwaway database (``MTC_TEST_STORES=1``) and skips otherwise.
Both stores take the same injected clock, so every case uses fixed instants on
2000-01-01.
"""

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from mtc.memory import protocols
from mtc.memory.episodic_store import InMemoryEpisodicStore, PostgresEpisodicStore

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    """A settable clock; every store reads its time from it."""

    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture(
    params=[
        pytest.param("inmemory", id="inmemory"),
        pytest.param("postgres", marks=pytest.mark.db, id="postgres"),
    ]
)
def raw_store(request, clock):
    if request.param == "inmemory":
        return InMemoryEpisodicStore(clock=clock)
    pool = request.getfixturevalue("throwaway_postgres")
    return PostgresEpisodicStore(pool, clock=clock)


@pytest_asyncio.fixture
async def store(raw_store):
    await raw_store.ensure_schema()
    return raw_store


async def _add_at(store, clock, moment, content, **kwargs):
    clock.now = moment
    return await store.add_episodic(content, **kwargs)


@pytest.mark.asyncio
async def test_satisfies_protocol(store):
    assert isinstance(store, protocols.EpisodicStore)


@pytest.mark.asyncio
async def test_core_blocks_seed_without_overwrite(store):
    written = await store.seed_core_blocks({"identity": "first", "persona": "p"})
    assert written == {"identity": True, "persona": True}
    written = await store.seed_core_blocks({"identity": "second", "human": "h"})
    assert written == {"identity": False, "human": True}
    blocks = await store.load_core_blocks()
    assert blocks == {"identity": "first", "persona": "p", "human": "h"}
    written = await store.seed_core_blocks({"identity": "third"}, overwrite=True)
    assert written == {"identity": True}
    assert (await store.load_core_blocks())["identity"] == "third"


@pytest.mark.asyncio
async def test_core_block_update(store):
    await store.set_core_block("persona", "old")
    await store.set_core_block("persona", "new")
    blocks = await store.load_core_blocks()
    assert blocks == {"persona": "new"}
    blocks["persona"] = "mutated"
    assert (await store.load_core_blocks())["persona"] == "new"


@pytest.mark.asyncio
async def test_add_and_get_episodic(store, clock):
    memory_id = await _add_at(
        store,
        clock,
        T0,
        "Alice described a garden",
        memory_type="conversation",
        importance=6.5,
        emotional_context={"joy": 0.4},
    )
    assert isinstance(memory_id, int)
    rows = await store.get_episodic([memory_id])
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == memory_id
    assert row["content"] == "Alice described a garden"
    assert row["memory_type"] == "conversation"
    assert row["importance"] == 6.5
    assert row["emotional_context"] == {"joy": 0.4}
    assert row["created_at"] == T0 and row["created_at"].tzinfo is not None
    assert row["accessed_count"] == 0
    assert row["consolidated"] is False
    assert await store.get_episodic([memory_id + 1000]) == []
    # Rows come back in the order the ids were asked for.
    second = await _add_at(store, clock, T0, "second row")
    ordered = await store.get_episodic([second, memory_id])
    assert [r["id"] for r in ordered] == [second, memory_id]


@pytest.mark.asyncio
async def test_default_emotional_context_is_empty_mapping(store):
    memory_id = await store.add_episodic("plain")
    assert (await store.get_episodic([memory_id]))[0]["emotional_context"] == {}


@pytest.mark.asyncio
async def test_text_search_ranks_matches(store, clock):
    twice = await _add_at(store, clock, T0, "garden path and garden gate")
    once_old = await _add_at(store, clock, T0 + timedelta(minutes=1), "a garden")
    once_new = await _add_at(store, clock, T0 + timedelta(minutes=2), "one garden")
    await _add_at(store, clock, T0 + timedelta(minutes=3), "kitchen table")
    found = await store.search_text("garden")
    assert [r["id"] for r in found] == [twice, once_new, once_old]
    assert all(r["relevance"] > 0 for r in found)
    assert found[0]["relevance"] > found[1]["relevance"]
    assert len(await store.search_text("garden", limit=2)) == 2
    assert await store.search_text("orchard") == []


@pytest.mark.asyncio
async def test_text_search_filters_type_and_importance(store, clock):
    low = await _add_at(store, clock, T0, "river notes", importance=2.0)
    high = await _add_at(store, clock, T0, "river story", importance=8.0)
    other = await _add_at(
        store, clock, T0, "river thought", memory_type="reflection", importance=9.0
    )
    ids = {r["id"] for r in await store.search_text("river", min_importance=5.0)}
    assert ids == {high, other}
    typed = await store.search_text("river", memory_type="episodic")
    assert {r["id"] for r in typed} == {low, high}


@pytest.mark.asyncio
async def test_recent_respects_hours_and_order(store, clock):
    now = T0 + timedelta(hours=10)
    old = await _add_at(store, clock, now - timedelta(hours=5), "old")
    mid = await _add_at(store, clock, now - timedelta(minutes=90), "mid", importance=3)
    new = await _add_at(store, clock, now - timedelta(minutes=30), "new")
    clock.now = now
    assert [r["id"] for r in await store.recent(hours=2, limit=10)] == [new, mid]
    assert [r["id"] for r in await store.recent(hours=6, limit=10)] == [new, mid, old]
    assert [r["id"] for r in await store.recent(hours=6, limit=1)] == [new]
    filtered = await store.recent(hours=6, limit=10, min_importance=4.0)
    assert [r["id"] for r in filtered] == [new, old]


@pytest.mark.asyncio
async def test_unconsolidated_then_mark_consolidated(store, clock):
    first = await _add_at(store, clock, T0, "one")
    second = await _add_at(store, clock, T0 + timedelta(minutes=1), "two")
    third = await _add_at(store, clock, T0 + timedelta(minutes=2), "three")
    pending = await store.unconsolidated(10)
    assert [r["id"] for r in pending] == [first, second, third]
    assert [r["id"] for r in await store.unconsolidated(2)] == [first, second]
    assert await store.mark_consolidated([first, second], 0.8) == 2
    assert [r["id"] for r in await store.unconsolidated(10)] == [third]
    row = (await store.get_episodic([first]))[0]
    assert row["consolidated"] is True
    assert row["consolidation_strength"] == pytest.approx(0.8)
    assert row["replay_count"] == 1
    assert await store.mark_consolidated([], 0.5) == 0


@pytest.mark.asyncio
async def test_add_semantic_links_sources(store, clock):
    first = await _add_at(store, clock, T0, "one")
    second = await _add_at(store, clock, T0, "two")
    semantic_id = await store.add_semantic(
        {
            "concept": "joy_pattern",
            "knowledge": "two calm memories",
            "associations": {"frequency": 2},
            "strength": 0.2,
            "confidence": 0.7,
            "emotional_context": {"joy": 1.0},
            "source_memories": [first, second],
        }
    )
    assert isinstance(semantic_id, int)
    rows = await store.get_semantic([semantic_id])
    assert rows[0]["concept"] == "joy_pattern"
    assert rows[0]["source_memories"] == [first, second]
    assert rows[0]["associations"] == {"frequency": 2}
    assert rows[0]["created_at"] == T0
    with pytest.raises(ValueError):
        await store.add_semantic({"knowledge": "no concept"})


@pytest.mark.asyncio
async def test_touch_increments_access(store, clock):
    memory_id = await _add_at(store, clock, T0, "touched")
    clock.now = T0 + timedelta(hours=1)
    await store.touch([memory_id])
    await store.touch([memory_id])
    row = (await store.get_episodic([memory_id]))[0]
    assert row["accessed_count"] == 2
    assert row["last_accessed"] == T0 + timedelta(hours=1)
    await store.touch([])


@pytest.mark.asyncio
async def test_retention_candidates_older_than(store, clock):
    first = await _add_at(store, clock, T0, "first")
    second = await _add_at(store, clock, T0 + timedelta(days=1), "second")
    await _add_at(store, clock, T0 + timedelta(days=9), "young")
    cutoff = T0 + timedelta(days=2)
    candidates = await store.retention_candidates(cutoff, limit=10)
    assert [r["id"] for r in candidates] == [first, second]
    assert [r["id"] for r in await store.retention_candidates(cutoff, 1)] == [first]
    assert await store.retention_candidates(T0, limit=10) == []


@pytest.mark.asyncio
async def test_retention_candidates_page_with_a_cursor(store, clock):
    ids = [
        await _add_at(store, clock, T0 + timedelta(hours=i // 2), f"row {i}")
        for i in range(5)
    ]
    core = await _add_at(store, clock, T0, "formed", memory_type="core")
    cutoff = T0 + timedelta(days=2)
    seen = []
    after = None
    while True:
        page = await store.retention_candidates(
            cutoff, 2, after=after, exclude_types=("core",)
        )
        if not page:
            break
        seen.extend(r["id"] for r in page)
        after = (page[-1]["created_at"], page[-1]["id"])
    assert seen == ids
    everything = await store.retention_candidates(cutoff, 10)
    assert core in [r["id"] for r in everything]
    last = everything[-1]
    tail = await store.retention_candidates(
        cutoff, 10, after=(last["created_at"], last["id"])
    )
    assert tail == []


@pytest.mark.asyncio
async def test_unmeasured_strength_leaves_the_stored_strength(store, clock):
    memory_id = await _add_at(store, clock, T0, "replayed")
    assert await store.mark_consolidated([memory_id], 0.75) == 1
    assert await store.mark_consolidated([memory_id], None) == 1
    row = (await store.get_episodic([memory_id]))[0]
    assert row["consolidated"] is True
    assert row["consolidation_strength"] == pytest.approx(0.75)
    assert row["replay_count"] == 2


@pytest.mark.asyncio
async def test_archive_hides_from_recall(store, clock):
    kept = await _add_at(store, clock, T0, "lantern kept")
    hidden = await _add_at(store, clock, T0, "lantern hidden")
    assert await store.archive([hidden]) == 1
    assert await store.archive([hidden]) == 0
    assert await store.get_episodic([hidden]) == []
    assert [r["id"] for r in await store.search_text("lantern")] == [kept]
    clock.now = T0 + timedelta(minutes=5)
    assert [r["id"] for r in await store.recent(hours=1, limit=10)] == [kept]
    assert [r["id"] for r in await store.unconsolidated(10)] == [kept]
    later = T0 + timedelta(days=1)
    assert [r["id"] for r in await store.retention_candidates(later, 10)] == [kept]
    assert (await store.stats())["archived"] == 1


@pytest.mark.asyncio
async def test_delete_removes_rows(store, clock):
    first = await _add_at(store, clock, T0, "first")
    second = await _add_at(store, clock, T0, "second")
    assert await store.delete([first]) == 1
    assert await store.delete([first]) == 0
    assert await store.get_episodic([first]) == []
    assert [r["id"] for r in await store.get_episodic([second])] == [second]


@pytest.mark.asyncio
async def test_stats_counts(store, clock):
    await store.set_core_block("identity", "x")
    first = await _add_at(store, clock, T0, "one", importance=4.0)
    await _add_at(store, clock, T0, "two", importance=6.0)
    archived = await _add_at(store, clock, T0, "three", importance=9.0)
    await store.archive([archived])
    await store.mark_consolidated([first], 0.5)
    await store.add_semantic({"concept": "c", "source_memories": [first]})
    stats = await store.stats()
    assert stats["core_blocks"] == 1
    assert stats["episodic"] == 2
    assert stats["archived"] == 1
    assert stats["unconsolidated"] == 1
    assert stats["semantic"] == 1
    assert stats["total_importance"] == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_importance_ten_round_trips(store):
    top = await store.add_episodic("peak", importance=10.0)
    floor = await store.add_episodic("floor", importance=0.0)
    rows = {r["id"]: r for r in await store.get_episodic([top, floor])}
    assert rows[top]["importance"] == 10.0
    assert rows[floor]["importance"] == 0.0
    for bad in (10.5, -0.1, float("nan")):
        with pytest.raises(ValueError):
            await store.add_episodic("bad", importance=bad)


@pytest.mark.asyncio
async def test_returned_rows_are_copies(store):
    memory_id = await store.add_episodic("copy", emotional_context={"a": 1})
    row = (await store.get_episodic([memory_id]))[0]
    row["emotional_context"]["a"] = 99
    again = (await store.get_episodic([memory_id]))[0]
    assert again["emotional_context"] == {"a": 1}


@pytest.mark.asyncio
async def test_naive_clock_is_rejected():
    store = InMemoryEpisodicStore(clock=lambda: datetime(2000, 1, 1))
    with pytest.raises(ValueError):
        await store.add_episodic("naive")
