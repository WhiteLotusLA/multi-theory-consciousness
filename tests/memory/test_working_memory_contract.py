"""Contract tests for the WorkingMemory protocol.

Every case runs against the in-memory implementation and, under ``db``,
against ``RedisManager`` on the ``throwaway_redis`` fixture (skipped unless
``MTC_TEST_STORES=1``). Time-dependent expiry is covered for the in-memory
store with an injected clock at the end of the file.
"""

from datetime import datetime, timedelta, timezone

import pytest

from mtc.memory.inmemory import InMemoryWorkingMemory
from mtc.memory.protocols import WorkingMemory

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(
    params=[
        pytest.param("inmemory", id="inmemory"),
        pytest.param("redis", id="redis", marks=pytest.mark.db),
    ]
)
def memory(request):
    """Sync on purpose: an async fixture cannot request another async fixture."""
    if request.param == "inmemory":
        return InMemoryWorkingMemory(capacity=7)
    from mtc.memory.redis_manager import RedisManager

    target = request.getfixturevalue("throwaway_redis")
    return RedisManager(
        client=target.client, settings=target.settings, working_capacity=7
    )


def test_satisfies_the_protocol():
    assert isinstance(InMemoryWorkingMemory(), WorkingMemory)


@pytest.mark.asyncio
async def test_buffer_is_oldest_first_and_capped(memory):
    for number in range(10):
        size = await memory.add_to_working_memory("s1", {"n": number, "role": "user"})
        assert size == min(number + 1, 7)
    messages = await memory.get_working_memory("s1")
    assert [m["n"] for m in messages] == [3, 4, 5, 6, 7, 8, 9]
    assert all(isinstance(m["timestamp"], str) for m in messages)
    assert await memory.get_working_memory("other") == []
    await memory.clear_working_memory("s1")
    assert await memory.get_working_memory("s1") == []
    await memory.clear_working_memory("s1")  # clearing twice is fine


@pytest.mark.asyncio
async def test_given_timestamp_is_kept(memory):
    await memory.add_to_working_memory("s1", {"n": 1, "timestamp": "given"})
    assert (await memory.get_working_memory("s1"))[0]["timestamp"] == "given"


@pytest.mark.asyncio
async def test_emotional_state_round_trip(memory):
    assert await memory.get_emotional_state("s1") is None
    await memory.update_emotional_state("s1", {"valence": 0.25, "label": "calm"})
    state = await memory.get_emotional_state("s1")
    assert state["valence"] == 0.25 and state["label"] == "calm"
    assert "updated_at" in state
    await memory.update_emotional_state("s1", {"valence": -0.5})
    assert (await memory.get_emotional_state("s1"))["valence"] == -0.5


@pytest.mark.asyncio
async def test_session_context_and_cache(memory):
    assert await memory.get_session_context("s1") is None
    await memory.set_session_context("s1", {"mode": "quiet", "n": [1, 2]}, ttl=120)
    assert await memory.get_session_context("s1") == {"mode": "quiet", "n": [1, 2]}
    await memory.cache_set("k", {"hits": [1, 2, 3]}, ttl=60)
    assert await memory.cache_get("k") == {"hits": [1, 2, 3]}
    assert await memory.cache_get("absent") is None


@pytest.mark.asyncio
async def test_delete_session_removes_every_session_key(memory):
    await memory.add_to_working_memory("s1", {"n": 1})
    await memory.add_to_working_memory("s2", {"n": 2})
    await memory.update_emotional_state("s1", {"valence": 0.1})
    await memory.set_session_context("s1", {"k": 1})
    await memory.delete_session("s1")
    assert await memory.get_working_memory("s1") == []
    assert await memory.get_emotional_state("s1") is None
    assert await memory.get_session_context("s1") is None
    assert len(await memory.get_working_memory("s2")) == 1


@pytest.mark.asyncio
async def test_returned_documents_are_copies(memory):
    original = {"n": 1, "nested": {"a": [1]}}
    await memory.add_to_working_memory("s1", original)
    original["nested"]["a"].append(99)
    got = await memory.get_working_memory("s1")
    assert got[0]["nested"] == {"a": [1]}
    got[0]["nested"]["a"].append(7)
    assert (await memory.get_working_memory("s1"))[0]["nested"] == {"a": [1]}
    await memory.set_session_context("s1", {"x": [1]})
    (await memory.get_session_context("s1"))["x"].append(2)
    assert await memory.get_session_context("s1") == {"x": [1]}


@pytest.mark.asyncio
async def test_hit_rate_is_none_before_any_cache_read(memory):
    fresh = await memory.get_metrics()
    assert fresh["cache_hits"] == 0 and fresh["cache_misses"] == 0
    assert fresh["cache_hit_rate_percent"] is None
    await memory.cache_set("k", 1)  # a write is not a cache read
    assert (await memory.get_metrics())["cache_hit_rate_percent"] is None
    await memory.cache_get("absent")
    assert (await memory.get_metrics())["cache_hit_rate_percent"] == pytest.approx(0.0)
    await memory.disconnect()


@pytest.mark.asyncio
async def test_metrics_count_reads_writes_and_hits(memory):
    await memory.cache_set("k", 1)
    await memory.cache_get("k")
    await memory.cache_get("absent")
    metrics = await memory.get_metrics()
    for key in (
        "total_reads",
        "total_writes",
        "cache_hits",
        "cache_misses",
        "cache_hit_rate_percent",
        "errors",
    ):
        assert key in metrics
    assert metrics["cache_hits"] == 1 and metrics["cache_misses"] == 1
    assert metrics["cache_hit_rate_percent"] == pytest.approx(50.0)
    await memory.disconnect()


# ---------------------------------------------------------------------------
# In-memory expiry with an injected clock
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now = self.now + timedelta(seconds=seconds)


@pytest.mark.asyncio
async def test_inmemory_expiry_follows_the_clock():
    clock = _Clock()
    store = InMemoryWorkingMemory(capacity=7, clock=clock)
    await store.add_to_working_memory("s1", {"n": 1})
    await store.update_emotional_state("s1", {"valence": 0.1})
    await store.set_session_context("s1", {"k": 1}, ttl=100)
    await store.set_session_context("s2", {"k": 2})
    await store.cache_set("c", 1)
    clock.advance(31)
    assert await store.cache_get("c") is None  # 30 s default
    clock.advance(70)
    assert await store.get_session_context("s1") is None  # explicit ttl 100 s
    assert await store.get_session_context("s2") == {"k": 2}  # no ttl, no expiry
    assert len(await store.get_working_memory("s1")) == 1
    clock.advance(1700)  # 1801 s since the push
    assert await store.get_working_memory("s1") == []
    assert await store.get_emotional_state("s1") is not None
    clock.advance(14400)
    assert await store.get_emotional_state("s1") is None


@pytest.mark.asyncio
async def test_inmemory_push_refreshes_the_buffer_expiry():
    clock = _Clock()
    store = InMemoryWorkingMemory(capacity=7, clock=clock)
    await store.add_to_working_memory("s1", {"n": 1})
    clock.advance(1700)
    await store.add_to_working_memory("s1", {"n": 2})
    clock.advance(1700)
    assert [m["n"] for m in await store.get_working_memory("s1")] == [1, 2]
    clock.advance(200)
    assert await store.get_working_memory("s1") == []
    stamp = datetime.fromisoformat((await _first_message(T0))["timestamp"])
    assert stamp == T0


async def _first_message(moment):
    store = InMemoryWorkingMemory(capacity=7, clock=lambda: moment)
    await store.add_to_working_memory("s", {"n": 1})
    return (await store.get_working_memory("s"))[0]


def test_inmemory_capacity_clamped_and_read_from_settings(monkeypatch):
    assert InMemoryWorkingMemory(capacity=2).capacity == 5
    assert InMemoryWorkingMemory(capacity=50).capacity == 9
    assert InMemoryWorkingMemory(capacity=6).capacity == 6
    assert InMemoryWorkingMemory().capacity == 7  # the Settings default
    assert InMemoryWorkingMemory().status()["backend"] == "memory"
