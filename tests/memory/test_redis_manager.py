"""RedisManager over an in-test async fake client that records every key."""

import json
import os
import sys

import pytest

from mtc.consciousness.emotion_extractor import (
    EmotionAveragesProvider,
    InMemoryEmotionAverages,
)
from mtc.core.config import Settings
from mtc.memory import redis_manager as rm
from mtc.memory.protocols import StoreUnavailable, WorkingMemory
from mtc.memory.redis_manager import (
    KEY_FAMILIES,
    RedisManager,
    get_redis_manager,
)


class FakePipeline:
    def __init__(self, client):
        self._client = client
        self._queued = []

    def _queue(self, name, *args, **kwargs):
        self._queued.append((name, args, kwargs))
        return self

    def rpush(self, *a, **k):
        return self._queue("rpush", *a, **k)

    def ltrim(self, *a, **k):
        return self._queue("ltrim", *a, **k)

    def expire(self, *a, **k):
        return self._queue("expire", *a, **k)

    async def execute(self):
        results = []
        for name, args, kwargs in self._queued:
            results.append(await getattr(self._client, name)(*args, **kwargs))
        self._queued = []
        return results


class FakeRedis:
    """Strings, lists and streams with recorded keys and expiry arguments."""

    def __init__(self):
        self.strings = {}
        self.lists = {}
        self.streams = {}
        self.expiries = {}
        self.touched = []
        self.xadd_maxlens = []
        self.ltrim_calls = []
        self.fail_with = None
        self.closed = False
        self._stream_seq = 0

    def _check(self, *keys):
        if self.fail_with is not None:
            raise self.fail_with
        self.touched.extend(keys)

    async def ping(self):
        self._check()
        return True

    def pipeline(self, transaction=True):
        return FakePipeline(self)

    async def rpush(self, key, value):
        self._check(key)
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    async def ltrim(self, key, start, stop):
        self._check(key)
        self.ltrim_calls.append((key, start, stop))
        items = self.lists.get(key, [])
        self.lists[key] = items[start:] if stop == -1 and start < 0 else items
        return True

    async def expire(self, key, seconds):
        self._check(key)
        self.expiries[key] = seconds
        return True

    async def lrange(self, key, start, stop):
        self._check(key)
        return list(self.lists.get(key, []))

    async def set(self, key, value, ex=None):
        self._check(key)
        self.strings[key] = value
        if ex is not None:
            self.expiries[key] = ex
        else:
            self.expiries.pop(key, None)
        return True

    async def get(self, key):
        self._check(key)
        return self.strings.get(key)

    async def delete(self, *keys):
        self._check(*keys)
        removed = 0
        for key in keys:
            for store in (self.strings, self.lists, self.streams):
                if store.pop(key, None) is not None:
                    removed += 1
        return removed

    async def xadd(self, key, fields, maxlen=None, approximate=True):
        self._check(key)
        self.xadd_maxlens.append((maxlen, approximate))
        self._stream_seq += 1
        entries = self.streams.setdefault(key, [])
        entries.append((f"{self._stream_seq}-0", dict(fields)))
        if maxlen is not None:
            del entries[: max(0, len(entries) - maxlen)]
        return f"{self._stream_seq}-0"

    async def xrevrange(self, key, count=None):
        self._check(key)
        entries = list(reversed(self.streams.get(key, [])))
        return entries[:count] if count is not None else entries

    async def xlen(self, key):
        self._check(key)
        return len(self.streams.get(key, []))

    async def info(self, section):
        self._check()
        return {"used_memory": 2 * 1024 * 1024, "used_memory_peak": 3 * 1024 * 1024}

    async def aclose(self):
        self.closed = True


@pytest.fixture
def settings(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MTC_REDIS_KEY_PREFIX", "probe:")
    return Settings()


@pytest.fixture
def fake():
    return FakeRedis()


def _manager(fake, settings, **kwargs):
    return RedisManager(client=fake, settings=settings, **kwargs)


@pytest.mark.asyncio
async def test_every_key_starts_with_prefix(fake, settings):
    manager = _manager(fake, settings)
    await manager.add_to_working_memory("s1", {"role": "user", "content": "hi"})
    await manager.get_working_memory("s1")
    await manager.update_emotional_state("s1", {"valence": 0.1})
    await manager.get_emotional_state("s1")
    await manager.record("s1", {"valence": 0.1})
    await manager.get_averages("s1")
    await manager.sample_count("s1")
    await manager.set_session_context("s1", {"k": 1}, ttl=60)
    await manager.get_session_context("s1")
    await manager.cache_set("c", {"x": 1})
    await manager.cache_get("c")
    await manager.clear_working_memory("s1")
    await manager.delete_session("s1")
    assert fake.touched
    assert all(key.startswith("probe:") for key in fake.touched), fake.touched


@pytest.mark.asyncio
async def test_prefix_is_read_at_call_time(fake, settings):
    manager = _manager(fake, settings)
    await manager.cache_set("a", 1)
    settings.REDIS_KEY_PREFIX = "other:"
    await manager.cache_set("b", 2)
    assert "probe:cache:a" in fake.strings and "other:cache:b" in fake.strings


@pytest.mark.asyncio
@pytest.mark.parametrize("wanted,expected", [(3, 5), (5, 5), (7, 7), (9, 9), (20, 9)])
async def test_working_memory_capacity_from_settings_clamped_five_to_nine(
    fake, settings, wanted, expected
):
    settings.WORKING_MEMORY_CAPACITY = wanted
    manager = _manager(fake, settings)
    assert manager.capacity == expected
    for index in range(12):
        await manager.add_to_working_memory("s1", {"n": index})
    assert fake.ltrim_calls[-1][1:] == (-expected, -1)
    assert [m["n"] for m in await manager.get_working_memory("s1")] == list(
        range(12 - expected, 12)
    )
    explicit = _manager(fake, settings, working_capacity=100)
    assert explicit.capacity == 9


@pytest.mark.asyncio
async def test_working_memory_round_trips_oldest_first(fake, settings):
    manager = _manager(fake, settings)
    sizes = [
        await manager.add_to_working_memory("s1", {"n": n, "role": "user"})
        for n in range(3)
    ]
    assert sizes == [1, 2, 3]
    messages = await manager.get_working_memory("s1")
    assert [m["n"] for m in messages] == [0, 1, 2]
    assert all("timestamp" in m for m in messages)
    stamp = messages[0]["timestamp"]
    assert stamp.endswith("+00:00")
    original = {"n": 9, "timestamp": "kept"}
    await manager.add_to_working_memory("s2", original)
    assert (await manager.get_working_memory("s2"))[0]["timestamp"] == "kept"
    assert "timestamp" in original and original["timestamp"] == "kept"
    await manager.clear_working_memory("s1")
    assert await manager.get_working_memory("s1") == []


@pytest.mark.asyncio
async def test_working_memory_ttl_refreshed_on_push(fake, settings):
    manager = _manager(fake, settings)
    await manager.add_to_working_memory("s1", {"n": 1})
    fake.expiries.clear()
    await manager.add_to_working_memory("s1", {"n": 2})
    assert fake.expiries["probe:wm:conversation:s1"] == 1800


@pytest.mark.asyncio
async def test_emotion_state_ttl_four_hours(fake, settings):
    manager = _manager(fake, settings)
    state = {"valence": 0.4, "label": "calm"}
    await manager.update_emotional_state("s1", state)
    assert fake.expiries["probe:emotion:current:s1"] == 14400
    stored = await manager.get_emotional_state("s1")
    assert stored["valence"] == 0.4 and "updated_at" in stored
    assert "updated_at" not in state  # the caller's dict is not mutated
    assert await manager.get_emotional_state("absent") is None


@pytest.mark.asyncio
async def test_emotion_history_stream_capped(fake, settings):
    manager = _manager(fake, settings, emotion_history_maxlen=4)
    for index in range(10):
        await manager.record("s1", {"valence": float(index)})
    assert fake.xadd_maxlens[-1] == (4, False)
    assert await manager.sample_count("s1") == 4
    history = await manager.get_emotion_history("s1")
    assert [h["valence"] for h in history] == [6.0, 7.0, 8.0, 9.0]
    assert _manager(fake, settings).emotion_history_maxlen == 1000
    with pytest.raises(ValueError):
        RedisManager(client=fake, settings=settings, emotion_history_maxlen=0)
    no_ttl = [f for f in KEY_FAMILIES if f.pattern.startswith("emotion:history")]
    assert no_ttl[0].ttl_seconds is None
    assert not any(k.startswith("probe:emotion:history") for k in fake.expiries)


@pytest.mark.asyncio
async def test_unmeasured_samples_are_not_recorded(fake, settings):
    manager = _manager(fake, settings)
    await manager.record("s1", {})
    await manager.record("s1", {"valence": None, "arousal": float("nan")})
    assert await manager.sample_count("s1") == 0
    await manager.record("s1", {"valence": 0.5, "arousal": float("inf")})
    assert (await manager.get_emotion_history("s1")) == [{"valence": 0.5}]


@pytest.mark.asyncio
async def test_averages_none_without_samples(fake, settings):
    manager = _manager(fake, settings)
    assert await manager.get_averages("s1") is None
    assert await manager.sample_count("s1") == 0


@pytest.mark.asyncio
async def test_averages_match_inmemory_provider_keys(fake, settings):
    manager = _manager(fake, settings)
    reference = InMemoryEmotionAverages()
    samples = [
        {"valence": 0.1, "arousal": 0.9},
        {"valence": -0.3, "arousal": 0.5, "drive_curiosity": 0.2},
        {"valence": 0.7},
        {"arousal": 0.1, "drive_curiosity": 0.8},
    ]
    for sample in samples:
        await manager.record("s1", sample)
        await reference.record("s1", sample)
    got = await manager.get_averages("s1")
    want = await reference.get_averages("s1")
    assert set(got) == set(want)
    for key in want:
        assert got[key] == pytest.approx(want[key], abs=1e-12), key
    assert await manager.sample_count("s1") == await reference.sample_count("s1")
    single = _manager(fake, settings)
    await single.record("one", {"valence": 0.3})
    assert await single.get_averages("one") == {"valence": 0.3, "valence_std": 0.0}


@pytest.mark.asyncio
async def test_averages_window_uses_newest_samples(fake, settings):
    manager = _manager(fake, settings)
    for value in (10.0, 0.0, 0.0):
        await manager.record("s1", {"v": value})
    averages = await manager.get_averages("s1", window=2)
    assert averages == {"v": 0.0, "v_std": 0.0}


def test_satisfies_emotion_averages_provider(fake, settings):
    manager = _manager(fake, settings)
    assert isinstance(manager, EmotionAveragesProvider)
    assert isinstance(manager, WorkingMemory)


@pytest.mark.asyncio
async def test_delete_session_removes_its_keys(fake, settings):
    manager = _manager(fake, settings)
    for session in ("s1", "s2"):
        await manager.add_to_working_memory(session, {"n": 1})
        await manager.update_emotional_state(session, {"valence": 0.1})
        await manager.record(session, {"valence": 0.1})
        await manager.set_session_context(session, {"k": 1})
    await manager.cache_set("shared", 1)
    await manager.delete_session("s1")
    remaining = set(fake.strings) | set(fake.lists) | set(fake.streams)
    assert not any(key.endswith(":s1") for key in remaining)
    assert any(key.endswith(":s2") for key in remaining)
    assert "probe:cache:shared" in remaining
    assert await manager.get_working_memory("s1") == []


@pytest.mark.asyncio
async def test_session_context_and_cache_round_trip(fake, settings):
    manager = _manager(fake, settings)
    await manager.set_session_context("s1", {"mode": "quiet"})
    assert "probe:session:s1" not in fake.expiries
    await manager.set_session_context("s1", {"mode": "quiet"}, ttl=90)
    assert fake.expiries["probe:session:s1"] == 90
    assert await manager.get_session_context("s1") == {"mode": "quiet"}
    assert await manager.get_session_context("absent") is None
    await manager.cache_set("k", {"results": [1, 2]})
    assert fake.expiries["probe:cache:k"] == 30
    await manager.cache_set("k2", [1], ttl=5)
    assert fake.expiries["probe:cache:k2"] == 5
    assert await manager.cache_get("k") == {"results": [1, 2]}
    assert await manager.cache_get("absent") is None


@pytest.mark.asyncio
async def test_unreachable_server_raises_store_unavailable(fake, settings):
    manager = _manager(fake, settings)
    fake.fail_with = ConnectionError("server down")
    calls = [
        manager.add_to_working_memory("s1", {"n": 1}),
        manager.get_working_memory("s1"),
        manager.update_emotional_state("s1", {"v": 1}),
        manager.get_emotional_state("s1"),
        manager.record("s1", {"v": 1.0}),
        manager.get_averages("s1"),
        manager.set_session_context("s1", {}),
        manager.get_session_context("s1"),
        manager.cache_set("k", 1),
        manager.cache_get("k"),
        manager.delete_session("s1"),
    ]
    for call in calls:
        with pytest.raises(StoreUnavailable) as info:
            await call
        assert isinstance(info.value.__cause__, ConnectionError)
    status = manager.status()
    assert status["backend"] == "redis"
    assert status["available"] is False
    assert "server down" in status["last_error"]
    fake.fail_with = None
    assert await manager.connect() is True
    assert manager.status()["available"] is True


@pytest.mark.asyncio
async def test_metrics_report_hit_rate_and_errors(fake, settings):
    manager = _manager(fake, settings)
    await manager.cache_set("k", 1)
    await manager.cache_get("k")
    await manager.cache_get("k")
    await manager.cache_get("absent")
    fake.fail_with = ConnectionError("down")
    with pytest.raises(StoreUnavailable):
        await manager.cache_get("k")
    fake.fail_with = None
    metrics = await manager.get_metrics()
    assert metrics["total_writes"] == 1
    assert metrics["cache_hits"] == 2 and metrics["cache_misses"] == 1
    assert metrics["cache_hit_rate_percent"] == pytest.approx(200 / 3)
    assert metrics["errors"] == 1
    assert metrics["redis_memory_mb"] == pytest.approx(2.0)
    assert metrics["redis_peak_memory_mb"] == pytest.approx(3.0)
    fake.fail_with = ConnectionError("down")
    unmeasured = await manager.get_metrics()
    assert unmeasured["redis_memory_mb"] is None


@pytest.mark.asyncio
async def test_hit_rate_is_none_before_any_cache_read(fake, settings):
    manager = _manager(fake, settings)
    assert (await manager.get_metrics())["cache_hit_rate_percent"] is None
    await manager.cache_set("k", 1)
    assert (await manager.get_metrics())["cache_hit_rate_percent"] is None
    await manager.cache_get("k")
    assert (await manager.get_metrics())["cache_hit_rate_percent"] == pytest.approx(
        100.0
    )


@pytest.mark.asyncio
async def test_missing_redis_names_the_extra(settings, monkeypatch):
    monkeypatch.setitem(sys.modules, "redis", None)
    monkeypatch.setitem(sys.modules, "redis.asyncio", None)
    manager = RedisManager.from_settings(settings)
    with pytest.raises(ImportError) as info:
        await manager.connect()
    assert "redis" in str(info.value) and "[redis]" in str(info.value)
    with pytest.raises(ImportError):
        await manager.get_working_memory("s1")


@pytest.mark.asyncio
async def test_disconnect_closes_only_an_owned_client(fake, settings):
    given = _manager(fake, settings)
    await given.disconnect()
    assert fake.closed is False
    owned = RedisManager(settings=settings)
    owned._client = fake
    owned._owns_client = True
    await owned.disconnect()
    assert fake.closed is True
    assert owned.status()["connected"] is False


def test_get_redis_manager_builds_without_connecting(settings):
    manager = get_redis_manager(settings)
    assert isinstance(manager, RedisManager)
    assert manager.status()["connected"] is False
    assert manager.status()["key_prefix"] == "probe:"


def test_key_families_document_each_pattern():
    patterns = [f.pattern for f in KEY_FAMILIES]
    assert patterns == [
        "wm:conversation:{session}",
        "emotion:current:{session}",
        "emotion:history:{session}",
        "session:{id}",
        "cache:{key}",
    ]
    ttls = {f.pattern.split(":")[0]: f.ttl_seconds for f in KEY_FAMILIES}
    assert ttls["wm"] == 1800 and ttls["cache"] == 30
    assert rm.EVICTION_RECOMMENDATION == "volatile-lru"


@pytest.mark.asyncio
async def test_payloads_are_json_documents(fake, settings):
    manager = _manager(fake, settings)
    await manager.cache_set("k", {"a": [1, 2]})
    assert json.loads(fake.strings["probe:cache:k"]) == {"a": [1, 2]}
    with pytest.raises(TypeError):
        await manager.cache_set("bad", {"x": object()})
