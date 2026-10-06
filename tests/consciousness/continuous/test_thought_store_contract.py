"""One contract for every thought store: in memory, a fake driver, MongoDB."""

import asyncio
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from mtc.consciousness.continuous.schemas import (
    ThoughtIntensity,
    ThoughtRecord,
    ThoughtType,
)
from mtc.consciousness.continuous.store import (
    InMemoryThoughtStore,
    MongoThoughtStore,
    ThoughtStore,
)
from mtc.core.config import Settings
from mtc.memory.protocols import StoreUnavailable
from mtc.neural.mongodb_schemas import NeuralDataManager
from tests.memory.test_mongo_stores import FakeMongoClient

ROOT = Path(__file__).resolve().parents[3]
T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _record(
    thought_id: str,
    minutes: int = 0,
    text: Optional[str] = None,
    thought_type: ThoughtType = ThoughtType.REFLECTION,
    express: bool = False,
    **extra: Any,
) -> ThoughtRecord:
    return ThoughtRecord(
        thought_id=thought_id,
        timestamp=T0 + timedelta(minutes=minutes),
        thought_type=thought_type,
        content=text or f"text of {thought_id}",
        intensity=ThoughtIntensity.QUIET,
        salience_score=0.5,
        should_express=express,
        **extra,
    )


def _memory_factory(**_: Any) -> Callable[..., Any]:
    return lambda max_items=None, clock=None: InMemoryThoughtStore(
        **({"max_items": max_items} if max_items else {}), clock=clock
    )


def _mongo_factory(database: Any) -> Callable[..., Any]:
    def make(max_items=None, clock=None):
        if max_items:
            pytest.skip("the MongoDB store is bounded by the pruner, not a cap")
        manager = NeuralDataManager(database_name=database.name, client=database.client)
        manager.ensure_indexes()
        return MongoThoughtStore(manager, clock=clock)

    return make


@pytest.fixture(
    params=[
        pytest.param("memory", id="memory"),
        pytest.param("mongo-fake", id="mongo-fake"),
        pytest.param("mongo", marks=pytest.mark.db, id="mongo"),
    ]
)
def make_store(request) -> Callable[..., Any]:
    if request.param == "memory":
        return _memory_factory()
    if request.param == "mongo-fake":
        return _mongo_factory(FakeMongoClient()["contract"])
    return _mongo_factory(request.getfixturevalue("throwaway_mongo"))


def test_stores_satisfy_the_protocol(make_store):
    assert isinstance(make_store(), ThoughtStore)


@pytest.mark.asyncio
async def test_insert_then_recent_newest_first(make_store):
    store = make_store()
    for index, name in enumerate(("a", "b", "c")):
        assert await store.insert(_record(name, minutes=index)) == name
    recent = await store.recent(10)
    assert [r.thought_id for r in recent] == ["c", "b", "a"]
    assert [r.thought_id for r in await store.recent(2)] == ["c", "b"]
    assert await store.recent(0) == []
    assert await store.count() == 3
    # Quiet and loud thoughts of every type come back from recent().
    await store.insert(_record("d", 5, thought_type=ThoughtType.DREAM, express=True))
    assert (await store.recent(1))[0].thought_type is ThoughtType.DREAM


@pytest.mark.asyncio
async def test_bump_access_on_matching_text(make_store):
    stamp = T0 + timedelta(days=1)
    store = make_store(clock=lambda: stamp)
    await store.insert(_record("old", 0, text="the same words"))
    await store.insert(_record("new", 5, text="the same words"))
    await store.insert(_record("other", 9, text="different words"))
    assert await store.bump_access("the same words") == "new"
    assert await store.bump_access("the same words") == "new"
    assert await store.bump_access("never stored") is None
    by_id = {r.thought_id: r for r in await store.recent(10)}
    assert by_id["new"].access_count == 2
    assert by_id["new"].last_seen == stamp
    assert by_id["old"].access_count == 0 and by_id["old"].last_seen is None
    assert by_id["other"].access_count == 0
    assert await store.count() == 3  # a bump stores no copy


@pytest.mark.asyncio
async def test_expressible_filters_and_dedups(make_store):
    store = make_store()
    await store.insert(_record("quiet", 0, express=False))
    await store.insert(_record("a1", 1, text="repeated", express=True))
    await store.insert(_record("b", 2, text="distinct", express=True))
    await store.insert(_record("a2", 3, text="repeated", express=True))
    await store.insert(_record("c", 4, text="newest", express=True))
    chosen = await store.expressible(10)
    # Newest first, quiet thoughts out, one per distinct text (the newest).
    assert [r.thought_id for r in chosen] == ["c", "a2", "b"]
    assert [r.thought_id for r in await store.expressible(2)] == ["c", "a2"]
    assert await store.expressible(0) == []


@pytest.mark.asyncio
async def test_exclude_type_prefixes(make_store):
    store = make_store()
    await store.insert(_record("w", 0, express=True))
    await store.insert(_record("d", 1, thought_type=ThoughtType.DREAM, express=True))
    await store.insert(_record("q", 2, thought_type=ThoughtType.QUESTION, express=True))
    default = await store.expressible(10)
    assert [r.thought_id for r in default] == ["q", "w"]
    assert [r.thought_id for r in await store.expressible(10, ())] == ["q", "d", "w"]
    only_reflect = await store.expressible(10, ("dream", "quest"))
    assert [r.thought_id for r in only_reflect] == ["w"]


@pytest.mark.asyncio
async def test_expressible_reaches_past_a_run_of_excluded_thoughts(make_store):
    store = make_store()
    await store.insert(_record("old", 0, express=True))
    for index in range(1, 80):
        await store.insert(
            _record(
                f"dream{index}",
                index,
                thought_type=ThoughtType.DREAM,
                express=True,
            )
        )
    assert [r.thought_id for r in await store.expressible(1)] == ["old"]


@pytest.mark.asyncio
async def test_link_child_by_thought_id(make_store):
    store = make_store()
    # Three thoughts of one burst: only the thought id names a parent.
    for index, name in enumerate(("p", "q", "r")):
        await store.insert(_record(name, index, stream_id="burst"))
    assert await store.link_child("q", "child-1") is True
    assert await store.link_child("q", "child-1") is True
    assert await store.link_child("burst", "child-2") is False
    assert await store.link_child("missing", "child-3") is False
    by_id = {r.thought_id: r for r in await store.recent(10)}
    assert by_id["q"].leads_to_thought_id == "child-1"
    assert by_id["p"].leads_to_thought_id is None
    assert by_id["r"].leads_to_thought_id is None


@pytest.mark.asyncio
async def test_max_items_drops_oldest(make_store):
    store = make_store(max_items=3)
    for index in range(5):
        await store.insert(_record(f"t{index}", index))
    assert await store.count() == 3
    assert [r.thought_id for r in await store.recent(10)] == ["t4", "t3", "t2"]
    assert store.status()["max_items"] == 3 and store.status()["dropped"] == 2


@pytest.mark.asyncio
async def test_returned_records_are_copies(make_store):
    store = make_store()
    record = _record(
        "x",
        0,
        express=True,
        emotional_context={"calm": 0.5},
        associated_memories=["m1"],
        processing_context={"k": [1]},
    )
    await store.insert(record)
    record.content = "mutated after insert"
    record.emotional_context["calm"] = 0.0
    first = (await store.recent(1))[0]
    assert first.content == "text of x" and first.emotional_context == {"calm": 0.5}
    first.associated_memories.append("m2")
    first.processing_context["k"].append(2)
    first.content = "mutated after read"
    again = (await store.expressible(1))[0]
    assert again.associated_memories == ["m1"]
    assert again.processing_context == {"k": [1]}
    assert again.content == "text of x"


@pytest.mark.asyncio
async def test_round_trip_keeps_every_field(make_store):
    store = make_store()
    record = _record(
        "full",
        3,
        express=True,
        triggered_by="a topic",
        associated_memories=["m1", "m2"],
        emotional_context={"calm": 0.4},
        previous_thought_id="before",
        consciousness_state="drowsy",
        processing_context={"strategy": "memory"},
        stream_id="burst-1",
        importance_score=0.3,
        retention_strength=0.9,
        is_protected=True,
        protection_reason="kept",
    )
    await store.insert(record)
    assert (await store.recent(1))[0] == record


@pytest.mark.asyncio
async def test_status_reports_the_backend(make_store):
    store = make_store()
    await store.insert(_record("s", 0))
    status = store.status()
    assert status["backend"] in {"memory", "mongo"}
    assert status["available"] is True


def test_naive_timestamps_never_reach_a_store():
    with pytest.raises(ValueError):
        _record("n", 0, first_seen=datetime(2000, 1, 1))


# ---- tests that need no store fixture --------------------------------------


@pytest.mark.asyncio
async def test_duplicate_id_is_refused_in_memory():
    store = InMemoryThoughtStore()
    await store.insert(_record("dup"))
    with pytest.raises(ValueError, match="already stored"):
        await store.insert(_record("dup", 1))
    with pytest.raises(ValueError):
        InMemoryThoughtStore(max_items=0)


@pytest.mark.asyncio
async def test_mongo_store_skips_documents_it_cannot_read():
    client = FakeMongoClient()
    manager = NeuralDataManager(database_name="skip", client=client)
    from mtc.neural.mongodb_schemas import ThoughtStream

    # A foreign document: a type of the memory integration, no intensity.
    manager.store_thought(
        ThoughtStream(
            timestamp=T0,
            stream_id="s",
            thought_text="a stored memory",
            thought_type="memory",
            metadata={"importance": 1.0},
        )
    )
    store = MongoThoughtStore(manager)
    await store.insert(_record("mine", 5))
    assert [r.thought_id for r in await store.recent(10)] == ["mine"]
    assert store.status()["skipped_unreadable"] == 1


@pytest.mark.asyncio
async def test_mongo_calls_run_off_loop(monkeypatch):
    """Every driver call goes through asyncio.to_thread and runs on a worker."""
    client = FakeMongoClient()
    manager = NeuralDataManager(database_name="offloop", client=client)
    store = MongoThoughtStore(manager)
    loop_thread = threading.get_ident()
    driver_threads = []
    collection = manager.thought_streams
    for name in (
        "insert_one",
        "find",
        "find_one_and_update",
        "update_one",
        "count_documents",
    ):
        original = getattr(collection, name)

        def recorder(*args, _original=original, **kwargs):
            driver_threads.append(threading.get_ident())
            return _original(*args, **kwargs)

        setattr(collection, name, recorder)

    real = asyncio.to_thread
    offloads = []

    async def recording(func, *args, **kwargs):
        offloads.append(func)
        return await real(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", recording)
    await store.insert(_record("a", 0, express=True))
    await store.recent(5)
    await store.expressible(5)
    await store.bump_access("text of a")
    await store.link_child("a", "b")
    assert await store.count() == 1
    assert len(offloads) == 6  # one offload per store call
    assert len(driver_threads) >= 6
    assert loop_thread not in driver_threads


@pytest.mark.asyncio
async def test_unreachable_server_raises_store_unavailable():
    client = FakeMongoClient()
    manager = NeuralDataManager(database_name="down", client=client)
    store = MongoThoughtStore(manager)

    def refuse(*args, **kwargs):
        raise OSError("refused")

    manager.thought_streams.find = refuse
    manager.thought_streams.insert_one = refuse
    with pytest.raises(StoreUnavailable):
        await store.recent(3)
    assert store.status()["store_error"]
    with pytest.raises(StoreUnavailable):
        await store.insert(_record("b", 0))
    assert store.status()["available"] is False


def test_missing_pymongo_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymongo", None)
    with pytest.raises(ImportError) as caught:
        MongoThoughtStore(settings=Settings())
    assert "mongo" in str(caught.value) and "pymongo" in str(caught.value)


def test_import_builds_no_client_and_loads_no_pymongo():
    code = (
        "import sys, mtc.consciousness.continuous.store;"
        "sys.exit(1 if 'pymongo' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr[-1000:]
