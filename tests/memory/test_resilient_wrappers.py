"""Resilient wrappers: buffering, strict reads, replay and breadcrumbs."""

import asyncio
import json
import logging
import os

import pytest

from mtc.core.circuit_breaker import CircuitBreaker
from mtc.core.config import Settings
from mtc.memory.inmemory import InMemoryConversationTurnStore, InMemoryWorkingMemory
from mtc.memory.protocols import (
    ConversationTurnStore,
    StoreUnavailable,
    WorkingMemory,
)
from mtc.memory.resilient_wrappers import (
    BUFFERED,
    ResilientMongoStore,
    ResilientPool,
    ResilientRedisManager,
)


@pytest.fixture
def settings(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(tmp_path))
    return Settings()


def _breaker(timeout: float = 0.0) -> CircuitBreaker:
    return CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=timeout)


class FakePool:
    """Records the statements that reached it; fails while ``down``."""

    def __init__(self):
        self.down = False
        self.log = []
        self.fail_on = None
        self.closed = False

    def _gate(self, query):
        if self.down or query == self.fail_on:
            raise ConnectionError("pool down")

    async def execute(self, query, *args, **kwargs):
        self._gate(query)
        self.log.append(("execute", query, args))
        return "OK"

    async def fetchval(self, query, *args, **kwargs):
        self._gate(query)
        self.log.append(("fetchval", query, args))
        return 42

    async def fetch(self, query, *args, **kwargs):
        self._gate(query)
        return [{"id": 1}]

    async def fetchrow(self, query, *args, **kwargs):
        self._gate(query)
        return {"id": 1}

    def acquire(self, *args, **kwargs):
        self._gate("acquire")
        return "connection-context"

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_healthy_calls_pass_through(settings):
    pool = FakePool()
    wrapped = ResilientPool(pool, breaker=_breaker(), settings=settings)
    assert await wrapped.execute("INSERT a", 1) == "OK"
    assert await wrapped.execute_returning("INSERT b RETURNING id", 2) == 42
    assert await wrapped.fetch("SELECT") == [{"id": 1}]
    assert await wrapped.fetchrow("SELECT") == {"id": 1}
    assert await wrapped.fetchval("SELECT") == 42
    assert wrapped.acquire() == "connection-context"
    assert wrapped.is_available() is True
    assert wrapped.status()["mode"] == "live"


@pytest.mark.asyncio
async def test_open_breaker_buffers_writes_and_replays_once(settings):
    pool = FakePool()
    wrapped = ResilientPool(pool, breaker=_breaker(), settings=settings)
    pool.down = True
    assert await wrapped.execute("W1", "a") is BUFFERED
    assert await wrapped.execute("W2", "b") is BUFFERED
    assert await wrapped.execute("W3", "c") is BUFFERED
    assert pool.log == []
    assert wrapped.circuit_state == "open"
    assert wrapped.status()["buffered"] == 3

    pool.down = False
    assert await wrapped.execute("W4", "d") == "OK"
    assert [entry[1] for entry in pool.log] == ["W1", "W2", "W3", "W4"]
    status = wrapped.status()
    assert status["buffered"] == 0 and status["replayed"] == 3
    assert wrapped.circuit_state == "closed"

    # the listener path: a second replay has nothing left to send
    await wrapped._replay()
    await asyncio.sleep(0)
    assert [entry[1] for entry in pool.log] == ["W1", "W2", "W3", "W4"]

    # a second outage and recovery replays only its own writes
    pool.down = True
    assert await wrapped.execute("W5") is BUFFERED
    pool.down = False
    await wrapped.fetch("SELECT")
    assert [entry[1] for entry in pool.log] == ["W1", "W2", "W3", "W4", "W5"]
    assert wrapped.status()["replayed"] == 4


@pytest.mark.asyncio
async def test_replay_failure_keeps_the_rest_buffered(settings):
    pool = FakePool()
    wrapped = ResilientPool(pool, breaker=_breaker(), settings=settings)
    pool.down = True
    for name in ("W1", "W2", "W3"):
        await wrapped.execute(name)
    pool.down = False
    pool.fail_on = "W2"
    with pytest.raises(StoreUnavailable):
        await wrapped.fetch("SELECT")  # the drain stops at W2
    assert [entry[1] for entry in pool.log] == ["W1"]
    assert wrapped.status()["buffered"] == 2
    pool.fail_on = None
    await asyncio.sleep(0)
    assert await wrapped.fetch("SELECT") == [{"id": 1}]
    assert [entry[1] for entry in pool.log] == ["W1", "W2", "W3"]
    assert wrapped.status()["buffered"] == 0


@pytest.mark.asyncio
async def test_new_writes_never_overtake_buffered_ones(settings):
    pool = FakePool()
    wrapped = ResilientPool(pool, breaker=_breaker(), settings=settings)
    pool.down = True
    await wrapped.execute("OLD")
    pool.down = False
    await wrapped.execute("NEW")
    assert [entry[1] for entry in pool.log] == ["OLD", "NEW"]


@pytest.mark.asyncio
async def test_buffer_overflow_drops_oldest_and_counts(settings, caplog):
    pool = FakePool()
    wrapped = ResilientPool(pool, breaker=_breaker(), buffer_size=3, settings=settings)
    pool.down = True
    with caplog.at_level(logging.ERROR, logger="mtc.memory.resilient_wrappers"):
        for index in range(5):
            await wrapped.execute(f"W{index}")
    status = wrapped.status()
    assert status["buffered"] == 3 and status["dropped"] == 2
    assert any("oldest write was dropped" in r.getMessage() for r in caplog.records)
    pool.down = False
    await wrapped.fetch("SELECT")
    assert [entry[1] for entry in pool.log] == ["W2", "W3", "W4"]
    with pytest.raises(ValueError):
        ResilientPool(pool, buffer_size=0)


@pytest.mark.asyncio
async def test_execute_returning_buffers_and_returns_sentinel(settings):
    pool = FakePool()
    wrapped = ResilientPool(pool, breaker=_breaker(), settings=settings)
    pool.down = True
    result = await wrapped.execute_returning("INSERT RETURNING id", "x")
    assert result is BUFFERED and not result and repr(result) == "BUFFERED"
    pool.down = False
    assert await wrapped.fetchval("SELECT 1") == 42
    assert pool.log[0] == ("fetchval", "INSERT RETURNING id", ("x",))


@pytest.mark.asyncio
async def test_reads_raise_when_unavailable(settings):
    pool = FakePool()
    wrapped = ResilientPool(pool, breaker=_breaker(60.0), settings=settings)
    pool.down = True
    with pytest.raises(StoreUnavailable):
        await wrapped.fetch("SELECT")  # fails and opens the breaker
    assert wrapped.circuit_state == "open"
    for call in (
        wrapped.fetch("SELECT"),
        wrapped.fetchrow("SELECT"),
        wrapped.fetchval("SELECT"),
    ):
        with pytest.raises(StoreUnavailable):
            await call
    with pytest.raises(StoreUnavailable):
        wrapped.acquire()
    status = wrapped.status()
    assert status["available"] is False and "pool down" in status["last_error"]


@pytest.mark.asyncio
async def test_close_is_awaitable_safe(settings):
    pool = FakePool()
    await ResilientPool(pool, settings=settings).close()
    assert pool.closed is True

    class SyncClose:
        closed = False

        def close(self):
            SyncClose.closed = True

    await ResilientPool(SyncClose(), settings=settings).close()
    assert SyncClose.closed is True


def test_wrappers_refuse_none(settings):
    with pytest.raises(ValueError):
        ResilientPool(None)
    with pytest.raises(ValueError):
        ResilientMongoStore(None)
    with pytest.raises(ValueError):
        ResilientRedisManager(None)


class FlakyTurns(InMemoryConversationTurnStore):
    """In-memory turn store that raises while ``down``."""

    def __init__(self):
        super().__init__()
        self.down = False
        self.calls = []

    def _gate(self):
        if self.down:
            raise ConnectionError("mongo down")

    async def store_conversation_turn(self, **kwargs):
        self._gate()
        self.calls.append(("turn", kwargs["turn_number"]))
        return await super().store_conversation_turn(**kwargs)

    async def start_conversation_session(
        self, conversation_id, primary_participant, **kwargs
    ):
        self._gate()
        self.calls.append(("start", conversation_id))
        return await super().start_conversation_session(
            conversation_id, primary_participant, **kwargs
        )

    async def end_conversation_session(self, conversation_id, **kwargs):
        self._gate()
        self.calls.append(("end", conversation_id))
        return await super().end_conversation_session(conversation_id, **kwargs)

    async def get_conversation_turns(self, *args, **kwargs):
        self._gate()
        return await super().get_conversation_turns(*args, **kwargs)


async def _turn(store, number):
    return await store.store_conversation_turn(
        conversation_id="c1",
        speaker="Alice",
        role="user",
        content=f"turn {number}",
        turn_number=number,
    )


@pytest.mark.asyncio
async def test_mongo_proxy_buffers_turns_and_replays(settings):
    inner = FlakyTurns()
    wrapped = ResilientMongoStore(inner, breaker=_breaker(), settings=settings)
    assert isinstance(wrapped, ConversationTurnStore)
    first = await _turn(wrapped, 1)
    assert isinstance(first, str)
    inner.down = True
    assert await wrapped.start_conversation_session("c2", "Guest") is BUFFERED
    assert await _turn(wrapped, 2) is BUFFERED
    assert await _turn(wrapped, 3) is BUFFERED
    assert await wrapped.end_conversation_session("c2") is None
    assert wrapped.status()["buffered"] == 4
    inner.down = False
    assert isinstance(await _turn(wrapped, 4), str)
    assert [c for c in inner.calls] == [
        ("turn", 1),
        ("start", "c2"),
        ("turn", 2),
        ("turn", 3),
        ("end", "c2"),
        ("turn", 4),
    ]
    turns = await wrapped.get_conversation_turns("c1")
    assert [t["turn_number"] for t in turns] == [1, 2, 3, 4]
    status = wrapped.status()
    assert status["buffered"] == 0 and status["replayed"] == 4
    assert status["backend"] == "mongodb"


@pytest.mark.asyncio
async def test_mongo_proxy_reads_raise_when_unavailable(settings):
    inner = FlakyTurns()
    wrapped = ResilientMongoStore(inner, breaker=_breaker(60.0), settings=settings)
    inner.down = True
    await _turn(wrapped, 1)  # buffered; opens the breaker
    with pytest.raises(StoreUnavailable):
        await wrapped.get_conversation_turns("c1")  # no buffered turn stands in
    assert wrapped.status()["available"] is False


class FlakyMemory:
    """Wraps an in-memory working memory; raises while ``down``."""

    def __init__(self):
        self.inner = InMemoryWorkingMemory(capacity=7)
        self.down = False

    def __getattr__(self, name):
        target = getattr(self.inner, name)

        async def call(*args, **kwargs):
            if self.down:
                raise ConnectionError("redis down")
            return await target(*args, **kwargs)

        return call


@pytest.mark.asyncio
async def test_redis_proxy_serves_only_what_was_written_during_outage(settings):
    flaky = FlakyMemory()
    wrapped = ResilientRedisManager(flaky, breaker=_breaker(), settings=settings)
    assert isinstance(wrapped, WorkingMemory)
    await wrapped.add_to_working_memory("s1", {"n": "before"})
    await wrapped.update_emotional_state("s1", {"valence": 0.9})
    flaky.down = True
    # nothing written during the outage: reads are empty, not the old data
    assert await wrapped.get_working_memory("s1") == []
    assert await wrapped.get_emotional_state("s1") is None
    assert await wrapped.get_session_context("s1") is None
    assert await wrapped.cache_get("k") is None
    await wrapped.add_to_working_memory("s1", {"n": "during"})
    await wrapped.update_emotional_state("s1", {"valence": -0.2})
    await wrapped.set_session_context("s1", {"mode": "quiet"})
    await wrapped.cache_set("k", {"v": 1})
    assert [m["n"] for m in await wrapped.get_working_memory("s1")] == ["during"]
    assert (await wrapped.get_emotional_state("s1"))["valence"] == -0.2
    assert await wrapped.get_session_context("s1") == {"mode": "quiet"}
    assert await wrapped.cache_get("k") == {"v": 1}
    assert await wrapped.get_working_memory("other") == []
    assert wrapped.status()["mode"] == "buffering"
    assert wrapped.status()["served_from_fallback"] > 0
    await wrapped.delete_session("s1")
    assert await wrapped.get_working_memory("s1") == []
    await wrapped.add_to_working_memory("s1", {"n": "kept"})

    flaky.down = False
    # recovery: the outage writes replay into the store, then reads hit it
    messages = await wrapped.get_working_memory("s1")
    assert [m["n"] for m in messages] == ["kept"]  # the outage delete replayed
    assert wrapped.circuit_state == "closed"
    # the emotional state and context written during the outage replayed too,
    # but the session delete in between removed them again
    assert await wrapped.get_emotional_state("s1") is None
    assert await wrapped.get_session_context("s1") is None
    assert wrapped.status()["replayed"] >= 1
    assert wrapped.status()["mode"] == "live"
    await wrapped.disconnect()


@pytest.mark.asyncio
async def test_redis_proxy_fallback_buffer_is_capped_by_capacity(settings):
    flaky = FlakyMemory()
    wrapped = ResilientRedisManager(flaky, breaker=_breaker(60.0), settings=settings)
    flaky.down = True
    for index in range(12):
        await wrapped.add_to_working_memory("s1", {"n": index})
    got = await wrapped.get_working_memory("s1")
    assert [m["n"] for m in got] == list(range(5, 12))  # capacity 7


@pytest.mark.asyncio
async def test_redis_proxy_averages_have_no_fallback(settings):
    class Averages(FlakyMemory):
        async def get_averages(self, session_id):
            if self.down:
                raise ConnectionError("redis down")
            return {"v": 1.0}

    flaky = Averages()
    wrapped = ResilientRedisManager(flaky, breaker=_breaker(60.0), settings=settings)
    assert await wrapped.get_averages("s1") == {"v": 1.0}
    flaky.down = True
    with pytest.raises(StoreUnavailable):
        await wrapped.get_averages("s1")


@pytest.mark.asyncio
async def test_breadcrumbs_off_by_default(settings, tmp_path):
    pool = FakePool()
    wrapped = ResilientPool(pool, breaker=_breaker(), settings=settings)
    pool.down = True
    await wrapped.execute("W1", "secret-argument")
    inner = FlakyTurns()
    inner.down = True
    await _turn(ResilientMongoStore(inner, breaker=_breaker(), settings=settings), 1)
    assert not (tmp_path / "logs").exists()


@pytest.mark.asyncio
async def test_breadcrumbs_under_log_directory_when_enabled(
    settings, monkeypatch, tmp_path
):
    monkeypatch.setenv("MTC_STORE_FALLBACK_LOG_ENABLED", "true")
    monkeypatch.setenv("MTC_LOG_DIRECTORY", "var/log")
    enabled = Settings()
    pool = FakePool()
    wrapped = ResilientPool(pool, breaker=_breaker(), settings=enabled)
    pool.down = True
    await wrapped.execute("INSERT INTO t VALUES ($1)", "arg-1")
    inner = FlakyTurns()
    inner.down = True
    store = ResilientMongoStore(inner, breaker=_breaker(), settings=enabled)
    await _turn(store, 7)
    directory = tmp_path / "var" / "log" / "store_fallback"
    pg_lines = (directory / "postgresql.jsonl").read_text().splitlines()
    record = json.loads(pg_lines[0])
    assert record["query"] == "INSERT INTO t VALUES ($1)"
    assert record["args"] == ["arg-1"]
    assert record["time"].endswith("+00:00")
    mongo_record = json.loads((directory / "mongodb_turns.jsonl").read_text())
    assert mongo_record["turn_number"] == 7 and mongo_record["kind"] == "turn"


@pytest.mark.asyncio
async def test_status_names_mode_and_counts(settings):
    pool = FakePool()
    wrapped = ResilientPool(
        pool, breaker=_breaker(60.0), buffer_size=2, settings=settings
    )
    status = wrapped.status()
    assert status == {
        "backend": "postgresql",
        "available": True,
        "last_error": None,
        "mode": "live",
        "circuit": "closed",
        "buffered": 0,
        "dropped": 0,
        "replayed": 0,
        "rejected": 0,
    }
    pool.down = True
    for name in ("a", "b", "c"):
        await wrapped.execute(name)
    status = wrapped.status()
    assert status["mode"] == "buffering" and status["circuit"] == "open"
    assert status["buffered"] == 2 and status["dropped"] == 1
    assert status["available"] is False
    assert "pool down" in status["last_error"]
    assert wrapped.circuit_stats["total_failures"] >= 1


# -- non-connection errors are the caller's problem, not an outage -----------


class DataError(Exception):
    """A statement refused for a data reason (not a connection failure)."""


class PickyPool(FakePool):
    """Raises a data error for the queries named in ``refuse``."""

    def __init__(self):
        super().__init__()
        self.refuse = set()

    def _gate(self, query):
        if query in self.refuse:
            raise DataError("NUL byte not allowed")
        super()._gate(query)


@pytest.mark.asyncio
async def test_pool_data_error_propagates_and_never_opens_the_breaker(settings):
    pool = PickyPool()
    pool.refuse = {"BAD"}
    wrapped = ResilientPool(pool, breaker=_breaker(60.0), settings=settings)
    with pytest.raises(DataError):
        await wrapped.execute("BAD", 1)
    status = wrapped.status()
    assert status["buffered"] == 0 and status["circuit"] == "closed"
    for index in range(5):
        assert await wrapped.execute(f"GOOD{index}") == "OK"
    assert await wrapped.fetch("SELECT") == [{"id": 1}]
    with pytest.raises(DataError):
        await wrapped.fetch("BAD")
    assert wrapped.is_available() is True


@pytest.mark.asyncio
async def test_pool_replay_drops_a_refused_entry_and_continues(settings, caplog):
    pool = PickyPool()
    wrapped = ResilientPool(pool, breaker=_breaker(), settings=settings)
    pool.down = True
    for name in ("W1", "BAD", "W3"):
        assert await wrapped.execute(name) is BUFFERED
    pool.down = False
    pool.refuse = {"BAD"}
    with caplog.at_level(logging.ERROR, logger="mtc.memory.resilient_wrappers"):
        assert await wrapped.fetch("SELECT") == [{"id": 1}]
    assert [entry[1] for entry in pool.log] == ["W1", "W3"]
    status = wrapped.status()
    assert status["buffered"] == 0 and status["rejected"] == 1
    assert status["replayed"] == 2 and status["circuit"] == "closed"
    assert any("refused a buffered write" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_mongo_data_error_propagates_unbuffered(settings):
    class Picky(FlakyTurns):
        async def store_conversation_turn(self, **kwargs):
            if kwargs["turn_number"] == 99:
                raise DataError("bad document")
            return await super().store_conversation_turn(**kwargs)

    inner = Picky()
    wrapped = ResilientMongoStore(inner, breaker=_breaker(60.0), settings=settings)
    with pytest.raises(DataError):
        await _turn(wrapped, 99)
    assert wrapped.status()["buffered"] == 0 and wrapped.circuit_state == "closed"
    assert isinstance(await _turn(wrapped, 1), str)
    assert (await wrapped.get_conversation_turns("c1"))[0]["turn_number"] == 1


@pytest.mark.asyncio
async def test_mongo_replay_drops_a_refused_entry(settings):
    class Picky(FlakyTurns):
        async def store_conversation_turn(self, **kwargs):
            if kwargs["turn_number"] == 2 and not self.down:
                raise DataError("bad document")
            return await super().store_conversation_turn(**kwargs)

    inner = Picky()
    wrapped = ResilientMongoStore(inner, breaker=_breaker(), settings=settings)
    inner.down = True
    for number in (1, 2, 3):
        assert await _turn(wrapped, number) is BUFFERED
    inner.down = False
    assert isinstance(await _turn(wrapped, 4), str)
    assert [c for c in inner.calls] == [("turn", 1), ("turn", 3), ("turn", 4)]
    status = wrapped.status()
    assert status["rejected"] == 1 and status["buffered"] == 0


@pytest.mark.asyncio
async def test_redis_data_error_propagates_and_stays_live(settings):
    class Picky(FlakyMemory):
        async def add_to_working_memory(self, session_id, message):
            if message.get("bad"):
                raise TypeError("not JSON serializable")
            return await self.inner.add_to_working_memory(session_id, message)

    flaky = Picky()
    wrapped = ResilientRedisManager(flaky, breaker=_breaker(60.0), settings=settings)
    with pytest.raises(TypeError):
        await wrapped.add_to_working_memory("s1", {"bad": True})
    status = wrapped.status()
    assert status["buffered"] == 0 and status["circuit"] == "closed"
    assert await wrapped.add_to_working_memory("s1", {"n": 1}) == 1
    assert [m["n"] for m in await wrapped.get_working_memory("s1")] == [1]


@pytest.mark.asyncio
async def test_redis_refused_command_behind_store_unavailable_is_not_an_outage(
    settings,
):
    class ResponseError(Exception):
        pass

    class Refusing(FlakyMemory):
        async def get_session_context(self, session_id):
            raise StoreUnavailable("Redis get context failed") from ResponseError(
                "WRONGTYPE"
            )

    wrapped = ResilientRedisManager(
        Refusing(), breaker=_breaker(60.0), settings=settings
    )
    with pytest.raises(StoreUnavailable):
        await wrapped.get_session_context("s1")
    assert wrapped.circuit_state == "closed"


@pytest.mark.asyncio
async def test_redis_replay_drops_a_refused_write(settings):
    class Picky(FlakyMemory):
        async def add_to_working_memory(self, session_id, message):
            if self.down:
                raise ConnectionError("redis down")
            if message.get("bad"):
                raise TypeError("not JSON serializable")
            return await self.inner.add_to_working_memory(session_id, message)

    flaky = Picky()
    wrapped = ResilientRedisManager(flaky, breaker=_breaker(), settings=settings)
    flaky.down = True
    await wrapped.add_to_working_memory("s1", {"n": 1})
    await wrapped.add_to_working_memory("s1", {"n": 2, "bad": True})
    await wrapped.add_to_working_memory("s1", {"n": 3})
    flaky.down = False
    assert [m["n"] for m in await wrapped.get_working_memory("s1")] == [1, 3]
    status = wrapped.status()
    assert status["rejected"] == 1 and status["replayed"] == 2
    assert status["buffered"] == 0 and status["circuit"] == "closed"


@pytest.mark.asyncio
async def test_redis_overflow_counts_and_logs_dropped_writes(settings, caplog):
    flaky = FlakyMemory()
    wrapped = ResilientRedisManager(
        flaky, breaker=_breaker(60.0), settings=settings, buffer_size=5
    )
    flaky.down = True
    with caplog.at_level(logging.ERROR, logger="mtc.memory.resilient_wrappers"):
        for index in range(12):
            await wrapped.add_to_working_memory("s1", {"n": index})
    status = wrapped.status()
    assert status["buffered"] == 5 and status["dropped"] == 7
    dropped_logs = [
        r for r in caplog.records if "oldest write was dropped" in r.getMessage()
    ]
    assert len(dropped_logs) == 7
    with pytest.raises(ValueError):
        ResilientRedisManager(flaky, buffer_size=0)
