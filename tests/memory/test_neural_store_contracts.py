"""Contract tests for the conversation-turn, thought-gist and retention stores.

Every case runs against each entry of ``BACKENDS``. The in-memory backend is
the default; a later phase adds the MongoDB backend by appending one
``pytest.param`` entry (carrying ``pytest.mark.db``) whose factory builds the
stores on a throwaway database and whose seeders write telemetry and thought
documents directly. No case below needs editing.
"""

import asyncio
import copy
import functools
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, Tuple

import pytest

from mtc.core.config import Settings
from mtc.memory import inmemory
from mtc.memory.memory_consolidation_pipeline import MemoryConsolidationPipeline
from mtc.memory.protocols import (
    TELEMETRY_COLLECTIONS,
    ConversationTurnStore,
    NeuralRetentionStore,
    ThoughtGistStore,
)
from mtc.memory.thought_gist import ThoughtGist
from mtc.neural.mongodb_schemas import MongoConversationStore, NeuralDataManager
from tests.memory.test_mongo_stores import FakeMongoClient

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class NeuralBackend:
    """Factories and seeders for one backend."""

    name: str
    make_turns: Callable[[], Any]
    make_gists_and_retention: Callable[[], Tuple[Any, Any]]
    seed_telemetry: Callable[[Any, str, Dict[str, Any]], Any]
    seed_thought: Callable[[Any, Dict[str, Any]], Any]
    # A store-backed entry names a throwaway fixture; its factories then take
    # that fixture's value first and the ``backend`` fixture binds it.
    fixture: Optional[str] = None


def _memory_gists_and_retention():
    gists = inmemory.InMemoryThoughtGistStore()
    return gists, inmemory.InMemoryNeuralRetentionStore(gists)


async def _memory_seed_telemetry(retention, collection, doc):
    retention.add_telemetry(collection, doc)


async def _memory_seed_thought(retention, doc):
    retention.add_thought(doc)


def _mongo_turns(database: Any) -> Any:
    store = MongoConversationStore(database=database)
    store.ensure_indexes()
    return store


def _mongo_gists_and_retention(database: Any) -> Tuple[Any, Any]:
    manager = NeuralDataManager(database_name=database.name, client=database.client)
    manager.ensure_indexes()
    return manager, manager


async def _mongo_seed_telemetry(manager, collection, doc):
    await asyncio.to_thread(manager.db[collection].insert_one, copy.deepcopy(doc))


async def _mongo_seed_thought(manager, doc):
    await asyncio.to_thread(
        manager.db["thought_streams"].insert_one, copy.deepcopy(doc)
    )


@pytest.fixture
def fake_mongo() -> Any:
    """A fake pymongo database: checks the store logic without a server."""
    return FakeMongoClient()["contract"]


def _mongo_backend(fixture: str) -> NeuralBackend:
    return NeuralBackend(
        "mongo",
        _mongo_turns,
        _mongo_gists_and_retention,
        _mongo_seed_telemetry,
        _mongo_seed_thought,
        fixture=fixture,
    )


# Append a pytest.param(NeuralBackend(...), marks=pytest.mark.db, id="...") entry
# per real store; the cases below do not change. The mongo-fake entry runs them
# over the fake driver of tests/memory/test_mongo_stores.py, which models the
# driver and not a server, so the db entry is what checks real MongoDB.
BACKENDS = [
    pytest.param(
        NeuralBackend(
            "inmemory",
            lambda: inmemory.InMemoryConversationTurnStore(),
            _memory_gists_and_retention,
            _memory_seed_telemetry,
            _memory_seed_thought,
        ),
        id="inmemory",
    ),
    pytest.param(_mongo_backend("fake_mongo"), id="mongo-fake"),
    pytest.param(_mongo_backend("throwaway_mongo"), marks=pytest.mark.db, id="mongo"),
]


@pytest.fixture(params=BACKENDS)
def backend(request) -> NeuralBackend:
    entry = request.param
    if entry.fixture is None:
        return entry
    target = request.getfixturevalue(entry.fixture)
    return replace(
        entry,
        make_turns=functools.partial(entry.make_turns, target),
        make_gists_and_retention=functools.partial(
            entry.make_gists_and_retention, target
        ),
    )


def _gist(
    gist_id: str,
    retention: Optional[float] = 1.0,
    importance: Optional[float] = 0.5,
    protected: bool = False,
    offset_minutes: int = 0,
) -> ThoughtGist:
    return ThoughtGist(
        gist_id=gist_id,
        created_at=T0 + timedelta(minutes=offset_minutes),
        summary=f"summary {gist_id}",
        source_count=2,
        source_thought_ids=["a", "b"],
        dominant_emotion="calm",
        importance_score=importance,
        retention_strength=retention,
        is_protected=protected,
        protection_reason="synthetic" if protected else None,
    )


async def _turn(store, conversation, number, content=None, **extra):
    return await store.store_conversation_turn(
        conversation_id=conversation,
        speaker="Alice",
        role="user",
        content=content or f"turn {number}",
        turn_number=number,
        **extra,
    )


def test_stores_satisfy_their_protocols(backend):
    turns = backend.make_turns()
    gists, retention = backend.make_gists_and_retention()
    assert isinstance(turns, ConversationTurnStore)
    assert isinstance(gists, ThoughtGistStore)
    assert isinstance(retention, NeuralRetentionStore)


@pytest.mark.asyncio
async def test_turns_round_trip_in_turn_order(backend):
    store = backend.make_turns()
    for number in (3, 1, 2):
        turn_id = await _turn(
            store,
            "c1",
            number,
            emotional_context={"valence": 0.25},
            topics=["t"],
            metadata={"k": [1, 2]},
        )
        assert isinstance(turn_id, str) and turn_id
    await _turn(store, "c2", 1)
    turns = await store.get_conversation_turns("c1")
    assert [t["turn_number"] for t in turns] == [1, 2, 3]
    assert turns[0]["content"] == "turn 1"
    assert turns[0]["speaker"] == "Alice" and turns[0]["role"] == "user"
    assert turns[0]["emotional_context"] == {"valence": 0.25}
    assert turns[0]["topics"] == ["t"] and turns[0]["metadata"] == {"k": [1, 2]}
    assert [t["turn_number"] for t in await store.get_conversation_turns("c1", 2)] == [
        2,
        3,
    ]
    assert await store.get_conversation_turns("c1", 0) == []
    assert await store.get_conversation_turns("missing") == []


@pytest.mark.asyncio
async def test_session_start_and_end(backend):
    store = backend.make_turns()
    first = await store.start_conversation_session("c1", "Alice")
    second = await store.start_conversation_session("c2", "Guest")
    assert isinstance(first, str) and first and second and first != second
    await _turn(store, "c1", 1)
    assert await store.end_conversation_session("c1") is None
    assert await store.end_conversation_session("never-started") is None
    assert len(await store.get_conversation_turns("c1")) == 1


@pytest.mark.asyncio
async def test_gists_filter_by_retention(backend):
    gists, _ = backend.make_gists_and_retention()
    await gists.add_gist(_gist("low", retention=0.1, importance=0.9))
    await gists.add_gist(_gist("mid", retention=0.5, importance=0.2))
    await gists.add_gist(_gist("high", retention=0.9, importance=0.6))
    ids = [g.gist_id for g in await gists.active_gists()]
    assert ids == ["high", "mid"]  # most important first, floor 0.2
    assert [g.gist_id for g in await gists.active_gists(min_retention=0.0)] == [
        "low",
        "high",
        "mid",
    ]
    assert [
        g.gist_id for g in await gists.active_gists(min_retention=0.0, limit=1)
    ] == ["low"]
    assert await gists.active_gists(min_retention=0.95) == []


@pytest.mark.asyncio
async def test_gist_without_retention_is_never_listed_or_expired(backend):
    gists, retention = backend.make_gists_and_retention()
    await gists.add_gist(_gist("unknown", retention=None, importance=0.9))
    await gists.add_gist(_gist("known", retention=0.6, importance=0.1))
    # An unknown retention is not at or above any floor, the zero floor included.
    assert [g.gist_id for g in await gists.active_gists()] == ["known"]
    assert [g.gist_id for g in await gists.active_gists(min_retention=0.0)] == ["known"]
    # Nor is it below a floor: the retention pass never expires it.
    assert await retention.expire_gists(1.0, dry_run=True) == 1
    assert await retention.expire_gists(1.0, dry_run=False) == 1
    remaining = [g async for g in gists.iter_gists()]
    assert [g.gist_id for g in remaining] == ["unknown"]
    assert remaining[0].retention_strength is None


@pytest.mark.asyncio
async def test_active_gists_order_an_unknown_importance_last(backend):
    gists, _ = backend.make_gists_and_retention()
    await gists.add_gist(_gist("low", importance=0.2, offset_minutes=1))
    await gists.add_gist(_gist("unknown", importance=None, offset_minutes=2))
    await gists.add_gist(_gist("high", importance=0.8, offset_minutes=3))
    ordered = await gists.active_gists(min_retention=0.0)
    assert [g.gist_id for g in ordered] == ["high", "low", "unknown"]
    assert ordered[-1].importance_score is None


@pytest.mark.asyncio
async def test_iter_gists_yields_every_gist_in_creation_order(backend):
    gists, _ = backend.make_gists_and_retention()
    await gists.add_gist(_gist("c", retention=0.9, offset_minutes=5))
    await gists.add_gist(_gist("a", retention=None, offset_minutes=0))
    await gists.add_gist(_gist("e", retention=0.0, offset_minutes=9))
    await gists.add_gist(_gist("d", retention=0.05, offset_minutes=5))
    await gists.add_gist(_gist("b", retention=None, importance=None, offset_minutes=2))
    seen = [g.gist_id async for g in gists.iter_gists(batch=2)]
    # created_at first, gist_id among equal instants; any retention, null included
    assert seen == ["a", "b", "c", "d", "e"]
    assert [g.gist_id async for g in gists.iter_gists()] == seen
    first = [g async for g in gists.iter_gists()][0]
    first.summary = "changed after read"
    again = [g async for g in gists.iter_gists()][0]
    assert again.summary == "summary a"


@pytest.mark.asyncio
async def test_refresh_computes_retention_for_a_gist_stored_without_one(
    backend, tmp_path
):
    gists, _ = backend.make_gists_and_retention()
    now = T0 + timedelta(days=5)
    await gists.add_gist(
        ThoughtGist(
            gist_id="unknown",
            created_at=now - timedelta(days=28),
            summary="s",
            source_count=1,
            source_thought_ids=[],
        )
    )
    # No floor lists a gist without a retention, so only iter_gists reaches it.
    assert await gists.active_gists(min_retention=0.0) == []
    pipeline = MemoryConsolidationPipeline(
        settings=Settings(PROJECT_ROOT=str(tmp_path), SYSTEM_NAME="Testagent"),
        clock=lambda: now,
        gist_store=gists,
    )
    assert await pipeline.refresh_gist_retention() == 1
    (stored,) = await gists.active_gists(min_retention=0.0)
    assert stored.gist_id == "unknown"
    # Two half-lives (14 days each) after creation.
    assert stored.retention_strength == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_update_gist_reports_existence(backend):
    gists, _ = backend.make_gists_and_retention()
    await gists.add_gist(_gist("g1", retention=0.9))
    assert await gists.update_gist("g1", {"retention_strength": 0.05}) is True
    assert await gists.update_gist("absent", {"retention_strength": 0.05}) is False
    assert await gists.active_gists() == []
    stored = (await gists.active_gists(min_retention=0.0))[0]
    assert stored.retention_strength == 0.05
    with pytest.raises(ValueError):
        await gists.update_gist("g1", {"not_a_field": 1})


@pytest.mark.asyncio
async def test_retention_dry_run_counts_without_deleting(backend):
    gists, retention = backend.make_gists_and_retention()
    cutoff = T0
    old, fresh = T0 - timedelta(days=40), T0 + timedelta(days=1)
    for collection in TELEMETRY_COLLECTIONS:
        await backend.seed_telemetry(retention, collection, {"timestamp": old})
        await backend.seed_telemetry(retention, collection, {"timestamp": old})
        await backend.seed_telemetry(retention, collection, {"timestamp": fresh})
    await backend.seed_thought(
        retention, {"timestamp": old, "coherence_score": 0.2, "child_thoughts": []}
    )
    await backend.seed_thought(
        retention, {"timestamp": old, "coherence_score": 0.9, "child_thoughts": []}
    )
    await backend.seed_thought(
        retention, {"timestamp": old, "coherence_score": 0.2, "child_thoughts": ["x"]}
    )
    await backend.seed_thought(
        retention, {"timestamp": fresh, "coherence_score": 0.2, "child_thoughts": []}
    )
    await gists.add_gist(_gist("weak", retention=0.05))
    await gists.add_gist(_gist("strong", retention=0.9))

    expected_telemetry = {name: 2 for name in TELEMETRY_COLLECTIONS}
    assert await retention.prune_telemetry(cutoff, dry_run=True) == expected_telemetry
    assert await retention.prune_thoughts(cutoff, 0.5, dry_run=True) == 1
    assert await retention.expire_gists(0.1, dry_run=True) == 1
    # a dry run removed nothing: the same counts come back
    assert await retention.prune_telemetry(cutoff, dry_run=True) == expected_telemetry
    assert await retention.prune_thoughts(cutoff, 0.5, dry_run=True) == 1
    assert len(await gists.active_gists(min_retention=0.0)) == 2

    assert await retention.prune_telemetry(cutoff, dry_run=False) == expected_telemetry
    assert await retention.prune_thoughts(cutoff, 0.5, dry_run=False) == 1
    assert await retention.expire_gists(0.1, dry_run=False) == 1
    assert await retention.prune_telemetry(cutoff, dry_run=True) == {
        name: 0 for name in TELEMETRY_COLLECTIONS
    }
    assert await retention.prune_thoughts(cutoff, 0.5, dry_run=True) == 0
    assert [g.gist_id for g in await gists.active_gists(min_retention=0.0)] == [
        "strong"
    ]


@pytest.mark.asyncio
async def test_retention_spares_protected_gists(backend):
    gists, retention = backend.make_gists_and_retention()
    await gists.add_gist(_gist("kept", retention=0.0, protected=True))
    await gists.add_gist(_gist("lost", retention=0.0))
    assert await retention.expire_gists(0.5, dry_run=False) == 1
    remaining = await gists.active_gists(min_retention=0.0)
    assert [g.gist_id for g in remaining] == ["kept"]
    assert await retention.expire_gists(0.5, dry_run=False) == 0


@pytest.mark.asyncio
async def test_returned_documents_are_copies(backend):
    turns = backend.make_turns()
    await _turn(turns, "c1", 1, metadata={"k": [1]}, topics=["t"])
    first = await turns.get_conversation_turns("c1")
    first[0]["metadata"]["k"].append(2)
    first[0]["topics"].append("changed")
    first[0]["content"] = "mutated"
    again = await turns.get_conversation_turns("c1")
    assert again[0]["metadata"] == {"k": [1]}
    assert again[0]["topics"] == ["t"] and again[0]["content"] == "turn 1"

    gists, _ = backend.make_gists_and_retention()
    original = _gist("g1")
    await gists.add_gist(original)
    original.summary = "changed after add"
    original.source_thought_ids.append("z")
    got = (await gists.active_gists())[0]
    assert got.summary == "summary g1" and got.source_thought_ids == ["a", "b"]
    got.summary = "changed after read"
    got.emotional_arc["x"] = 1
    again_gist = (await gists.active_gists())[0]
    assert again_gist.summary == "summary g1" and again_gist.emotional_arc == {}
    snapshot = copy.deepcopy(again_gist)
    assert again_gist == snapshot


@pytest.mark.asyncio
async def test_timestamps_are_aware_utc(backend):
    turns = backend.make_turns()
    await _turn(turns, "c1", 1)
    stamp = (await turns.get_conversation_turns("c1"))[0]["timestamp"]
    assert isinstance(stamp, datetime)
    assert stamp.tzinfo is not None and stamp.utcoffset() == timedelta(0)

    gists, retention = backend.make_gists_and_retention()
    await gists.add_gist(_gist("g1"))
    created = (await gists.active_gists())[0].created_at
    assert created.tzinfo is not None and created.utcoffset() == timedelta(0)
    with pytest.raises(ValueError):
        await retention.prune_telemetry(datetime(2000, 1, 1), dry_run=True)
    with pytest.raises(ValueError):
        await retention.prune_thoughts(datetime(2000, 1, 1), 0.5, dry_run=True)


@pytest.mark.asyncio
async def test_inmemory_sessions_record_start_and_end_times():
    ticks = iter(T0 + timedelta(minutes=n) for n in range(10))
    store = inmemory.InMemoryConversationTurnStore(clock=lambda: next(ticks))
    session_id = await store.start_conversation_session("c1", "Alice", topic="x")
    started = store.get_session("c1")
    assert started["session_id"] == session_id and started["ended_at"] is None
    assert started["primary_participant"] == "Alice" and started["topic"] == "x"
    await store.end_conversation_session("c1", summary="done")
    ended = store.get_session("c1")
    assert ended["ended_at"] > ended["started_at"] and ended["summary"] == "done"
    assert store.get_session("absent") is None
    assert store.status()["backend"] == "memory"
