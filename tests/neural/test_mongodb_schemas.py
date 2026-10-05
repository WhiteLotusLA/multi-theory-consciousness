"""The MongoDB neural data layer over a fake driver; no server is contacted.

A recording fake client (``tests/memory/test_mongo_stores.py``) pins the query
shapes; the contract cases for the conversation, gist and retention methods
live in ``tests/memory/test_neural_store_contracts.py``.
"""

import ast
import asyncio
import inspect
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from mtc.core.config import Settings
from mtc.memory.protocols import TELEMETRY_COLLECTIONS, StoreUnavailable
from mtc.neural import mongodb_schemas
from mtc.neural.mongodb_schemas import (
    NEURAL_COLLECTIONS,
    NEURAL_INDEXES,
    ConversationSession,
    HTMColumn,
    LiquidState,
    MongoConversationStore,
    NeuralDataManager,
    NeuralSpike,
    ThoughtStream,
)
from tests.memory.test_mongo_stores import FakeMongoClient

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def client() -> FakeMongoClient:
    return FakeMongoClient()


@pytest.fixture
def manager(client) -> NeuralDataManager:
    return NeuralDataManager(database_name="fake_neural", client=client)


def _spike(stamp=T0, **extra) -> NeuralSpike:
    values = dict(
        timestamp=stamp, neuron_id=3, spike_time=1.5, emotion_type="joy", intensity=0.8
    )
    values.update(extra)
    return NeuralSpike(**values)


def _thought(stamp=T0, **extra) -> ThoughtStream:
    values = dict(
        timestamp=stamp,
        stream_id="s1",
        thought_text="a synthetic thought",
        thought_type="reflection",
        emotional_state={"joy": 0.7},
        coherence_score=0.6,
    )
    values.update(extra)
    return ThoughtStream(**values)


def test_construction_connects_nowhere(client):
    NeuralDataManager(client=client)
    MongoConversationStore(database=client["x"])
    assert client.calls == []


def test_database_from_settings_at_call_time(client, monkeypatch):
    first = NeuralDataManager(client=client, settings=Settings(MONGODB_DB="first_db"))
    second = NeuralDataManager(client=client, settings=Settings(MONGODB_DB="other_db"))
    assert first.db.name == "first_db" and second.db.name == "other_db"
    named = NeuralDataManager(
        database_name="explicit_db", client=client, settings=Settings()
    )
    assert named.db.name == "explicit_db"
    derived = NeuralDataManager(client=client, settings=Settings(DB_PREFIX="proj"))
    assert derived.db.name == "proj_neural"


def test_indexes_match_artifact(client, manager):
    applied = manager.ensure_indexes()
    assert applied == sum(len(specs) for specs in NEURAL_INDEXES.values())
    for name, specs in NEURAL_INDEXES.items():
        built = [
            (tuple(index["keys"]), index["unique"])
            for index in manager.db[name].indexes
        ]
        assert built == [(tuple(s.keys), s.unique) for s in specs], name
    assert set(NEURAL_INDEXES) == set(NEURAL_COLLECTIONS)
    assert len(NEURAL_COLLECTIONS) == 8
    thought_keys = [s.keys for s in NEURAL_INDEXES["thought_streams"]]
    assert (("thought_text", "hashed"),) in thought_keys
    assert (("timestamp", -1),) in [
        s.keys for s in NEURAL_INDEXES["consciousness_snapshots"]
    ]
    assert (("stream_id", 1),) in thought_keys
    assert (("thought_type", 1), ("timestamp", -1)) in thought_keys
    assert [s.unique for s in NEURAL_INDEXES["thought_gists"]] == [
        False,
        False,
        False,
        True,
    ]
    session_keys = [s.keys for s in NEURAL_INDEXES["conversation_sessions"]]
    assert (("phase", 1), ("started_at", -1)) in session_keys
    # idempotent
    snapshot = {n: list(manager.db[n].indexes) for n in NEURAL_COLLECTIONS}
    manager.ensure_indexes()
    assert snapshot == {n: list(manager.db[n].indexes) for n in NEURAL_COLLECTIONS}
    assert "thoughts" not in client["fake_neural"].collections


def _constants(source: str):
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if node.body and isinstance(node.body[0], ast.Expr):
                docstrings.add(id(node.body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_no_private_field_or_filter(manager):
    for text in _constants(inspect.getsource(mongodb_schemas)):
        assert "private" not in text.lower(), text
    for name in ("ThoughtStream", "ConversationTurn"):
        names = {f for f in getattr(mongodb_schemas, name).__dataclass_fields__}
        assert not any("private" in field for field in names)
    manager.store_thought(_thought())
    manager.get_recent_thoughts()
    stored = manager.thought_streams.docs[0]
    assert not any("private" in key for key in stored)
    for _, method, args, _kw in manager.db.client.calls:
        assert "private" not in str(args)


def test_default_participants_from_system_name(client):
    store = MongoConversationStore(
        database=client["c"], settings=Settings(SYSTEM_NAME="TestSystem")
    )
    asyncio.run(store.start_conversation_session("c1", "Alice"))
    asyncio.run(
        store.start_conversation_session("c2", "Alice", participants=["A", "B"])
    )
    sessions = client["c"]["conversation_sessions"].docs
    by_id = {s["conversation_id"]: s for s in sessions}
    assert by_id["c1"]["participants"] == ["Alice", "TestSystem"]
    assert by_id["c2"]["participants"] == ["A", "B"]
    assert "phase" in by_id["c1"] and "developmental_phase" not in by_id["c1"]


def test_session_keyword_extras_go_to_metadata_and_restart_replaces(client):
    store = MongoConversationStore(database=client["c"])
    first = asyncio.run(
        store.start_conversation_session("c1", "Alice", topic="x", phase="warmup")
    )
    again = asyncio.run(store.start_conversation_session("c1", "Alice"))
    docs = client["c"]["conversation_sessions"].docs
    assert len(docs) == 1 and first != again
    assert docs[0]["metadata"] is None and docs[0]["phase"] is None
    asyncio.run(store.start_conversation_session("c2", "Guest", topic="x", phase="p"))
    c2 = [d for d in docs if d["conversation_id"] == "c2"][0]
    assert c2["metadata"] == {"topic": "x"} and c2["phase"] == "p"


def test_end_session_counts_turns_and_keeps_given_fields(client):
    store = MongoConversationStore(database=client["c"])

    async def scenario():
        await store.start_conversation_session("c1", "Alice")
        for number in (1, 2):
            await store.store_conversation_turn(
                conversation_id="c1",
                speaker="Alice",
                role="user",
                content=f"turn {number}",
                turn_number=number,
            )
        await store.end_conversation_session("c1", summary="done")
        await store.end_conversation_session("ghost")

    asyncio.run(scenario())
    session = client["c"]["conversation_sessions"].docs[0]
    assert session["turn_count"] == 2 and session["summary"] == "done"
    assert session["ended_at"] >= session["started_at"]  # millisecond stamps
    assert session["topics"] is None  # not overwritten with an invented value


def test_timestamps_aware_utc(manager, client):
    manager.store_spike(_spike())
    manager.store_liquid_state(
        LiquidState(
            timestamp=T0,
            reservoir_size=2,
            neuron_states=[0.1, 0.2],
            input_pattern=[1.0],
        )
    )
    manager.store_htm_column(
        HTMColumn(
            timestamp=T0,
            column_id=1,
            active_cells=[1],
            predictive_cells=[2],
            anomaly_score=0.1,
        )
    )
    manager.store_thought(_thought())
    for collection in (
        "neural_spikes",
        "liquid_state",
        "htm_columns",
        "thought_streams",
    ):
        stamp = manager.db[collection].docs[0]["timestamp"]
        assert stamp.tzinfo is not None and stamp.utcoffset() == timedelta(0)
    with pytest.raises(ValueError):
        _spike(stamp=datetime(2000, 1, 1))
    with pytest.raises(ValueError):
        ConversationSession(conversation_id="c", started_at=datetime(2000, 1, 1))
    with pytest.raises(ValueError):
        manager.get_spikes_by_timerange(datetime(2000, 1, 1), T0)
    shifted = datetime(2000, 1, 1, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    assert _spike(stamp=shifted).timestamp == T0


def test_to_utc_reads_a_naive_time_as_local_clock_time():
    local = datetime(2000, 6, 1, 8, 30)
    result = mongodb_schemas.to_utc(local)
    assert result == local.astimezone(timezone.utc) and result.tzinfo is not None


def test_telemetry_reads_come_back_with_id_and_aware_times(manager):
    manager.store_spike_batch([_spike(), _spike(neuron_id=4, intensity=0.2)])
    rows = manager.get_spikes_by_emotion("joy", min_intensity=0.5)
    assert len(rows) == 1 and rows[0]["intensity"] == 0.8
    assert "_id" not in rows[0] and isinstance(rows[0]["id"], str)
    assert rows[0]["timestamp"].tzinfo is not None
    assert manager.store_spike_batch([]) == []
    assert len(manager.get_spikes_by_timerange(T0, T0)) == 2
    assert manager.get_spikes_by_emotion("joy", limit=0) == []


def test_unreachable_server_fails_fast_and_disables_until_retry(client):
    clock = Clock()
    manager = NeuralDataManager(
        client=client, retry_after_s=60.0, clock=clock, database_name="d"
    )
    client.unreachable = True
    assert manager.store_spike(_spike()) is None  # the one failure
    status = manager.status()
    assert status["available"] is False
    assert "ServerSelectionTimeoutError" in status["last_error"]
    assert status["failures"] == 1 and status["skipped_writes"] == 0
    attempts = len(client.calls)

    for _ in range(3):
        assert manager.store_spike(_spike()) is None
    assert manager.store_spike_batch([_spike()]) == []
    assert manager.store_thought(_thought()) is None
    assert len(client.calls) == attempts  # nothing reached the driver again
    assert manager.status()["skipped_writes"] == 5
    with pytest.raises(StoreUnavailable):
        manager.get_recent_thoughts()
    assert len(client.calls) == attempts

    clock.now += 61.0
    client.unreachable = False
    assert manager.store_spike(_spike()) is not None
    status = manager.status()
    assert status["available"] is True and status["last_error"] is None


def test_second_failure_after_the_window_disables_again(client):
    clock = Clock()
    manager = NeuralDataManager(client=client, retry_after_s=10.0, clock=clock)
    client.unreachable = True
    manager.store_spike(_spike())
    clock.now += 11.0
    manager.store_spike(_spike())
    assert manager.status()["failures"] == 2
    assert manager.status()["retry_in_s"] == pytest.approx(10.0)


def test_data_errors_propagate_and_do_not_disable(manager):
    manager.ensure_indexes()
    manager.store_gist({"gist_id": "g", "created_at": T0, "retention_strength": 1.0})
    manager.store_gist({"gist_id": "g", "created_at": T0, "retention_strength": 0.5})
    assert len(manager.thought_gists.docs) == 1  # replaced on the unique key
    assert manager.status()["available"] is True


@pytest.mark.asyncio
async def test_store_consciousness_snapshot_is_awaitable(manager):
    result = manager.store_consciousness_snapshot({"timestamp": T0.isoformat(), "x": 1})
    assert inspect.isawaitable(result)
    snapshot_id = await result
    assert isinstance(snapshot_id, str) and snapshot_id
    stored = manager.consciousness_snapshots.docs[0]
    assert stored["timestamp"] == T0 and stored["x"] == 1
    # a snapshot whose timestamp cannot be read is stored as handed in
    await manager.store_consciousness_snapshot({"timestamp": "not a time"})
    assert manager.consciousness_snapshots.docs[1]["timestamp"] == "not a time"
    # an unreachable server raises, since the snapshot is not telemetry
    manager.db.client.unreachable = True
    with pytest.raises(StoreUnavailable):
        await manager.store_consciousness_snapshot({"x": 2})


@pytest.mark.asyncio
async def test_close_is_awaitable(client):
    owned = NeuralDataManager(client=client)
    pending = owned.close()
    assert inspect.isawaitable(pending)
    await pending
    shared = NeuralDataManager(client=client)
    await shared.close()
    assert client.closed is False  # a client handed in belongs to its owner
    with pytest.raises(StoreUnavailable):
        shared.get_recent_thoughts()
    assert shared.status()["closed"] is True
    await shared.close()  # idempotent


def test_default_manager_builds_its_client_with_connect_false(monkeypatch):
    seen = []

    def factory(settings, **options):
        seen.append(options)
        return FakeMongoClient()

    monkeypatch.setattr(mongodb_schemas, "create_mongo_client", factory)
    NeuralDataManager()
    assert seen == [{"connect": False}]


def test_real_driver_default_manager_opens_no_socket(monkeypatch):
    """pymongo connects in the background unless told not to; none may open."""
    pytest.importorskip("pymongo")
    import socket
    import time

    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise OSError("no sockets in tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)  # the lookup comes first
    manager = NeuralDataManager(settings=Settings(MONGODB_HOST="mongo.invalid"))
    time.sleep(0.5)
    assert manager.status()["closed"] is False
    assert attempts == []
    asyncio.run(manager.close())


def test_closing_a_client_it_built_closes_it(monkeypatch):
    built = FakeMongoClient()
    monkeypatch.setattr(mongodb_schemas, "create_mongo_client", lambda s, **kw: built)
    manager = NeuralDataManager()
    asyncio.run(manager.close())
    assert built.closed is True


def test_collection_stats_cover_all_eight(manager):
    manager.store_spike(_spike())
    stats = manager.get_collection_stats()
    assert set(stats) == set(NEURAL_COLLECTIONS) and len(stats) == 8
    assert stats["neural_spikes"]["count"] == 1
    assert stats["thought_gists"]["count"] == 0
    for row in stats.values():
        assert row["size_mb"] is None or row["size_mb"] >= 0.0


def test_cleanup_uses_aware_cutoff(manager):
    old = datetime.now(timezone.utc) - timedelta(days=40)
    fresh = datetime.now(timezone.utc)
    manager.store_spike(_spike(stamp=old))
    manager.store_spike(_spike(stamp=fresh))
    manager.store_thought(_thought(stamp=old))
    result = manager.cleanup_old_data(days_to_keep=30)
    assert result["spikes_deleted"] == 1 and result["thought_streams_deleted"] == 1
    assert result["liquid_states_deleted"] == 0 and result["htm_columns_deleted"] == 0
    deletes = [c for c in manager.db.client.calls if c[1] == "delete_many"]
    assert len(deletes) == 4
    for call in deletes:
        cutoff = call[2][0]["timestamp"]["$lt"]
        assert cutoff.tzinfo is not None and cutoff.utcoffset() == timedelta(0)


def test_increment_access_updates_declared_field(manager):
    manager.store_thought(_thought(stamp=T0, thought_text="again"))
    manager.store_thought(_thought(stamp=T0 + timedelta(hours=1), thought_text="again"))
    assert manager.increment_thought_access("again") is True
    assert manager.increment_thought_access("never stored") is False
    counts = sorted(d["access_count"] for d in manager.thought_streams.docs)
    assert counts == [0, 1]
    newest = max(manager.thought_streams.docs, key=lambda d: d["timestamp"])
    assert newest["access_count"] == 1  # the newest match is the one bumped
    assert "access_count" in ThoughtStream.__dataclass_fields__
    for name in (
        "importance_score",
        "retention_strength",
        "is_protected",
        "protection_reason",
        "consolidated_into",
    ):
        assert name in ThoughtStream.__dataclass_fields__


def test_unscored_thought_fields_are_none_not_numbers(manager):
    manager.store_thought(_thought())
    doc = manager.thought_streams.docs[0]
    assert doc["importance_score"] is None and doc["retention_strength"] is None
    assert doc["is_protected"] is False and doc["consolidated_into"] is None


def test_thought_chain_children_and_lookups(manager):
    manager.store_thought(_thought(stream_id="p1", thought_text="parent"))
    manager.store_thought(
        _thought(
            stamp=T0 + timedelta(minutes=1),
            stream_id="c1",
            thought_text="child",
            parent_thought_id="p1",
            emotional_state={"joy": 0.9, "fear": 0.1},
        )
    )
    manager.update_thought_children("p1", "c1")
    manager.update_thought_children("p1", "c1")
    parent = manager.get_thought_by_stream_id("p1")
    assert parent["child_thoughts"] == ["c1"]
    assert [t["stream_id"] for t in manager.get_thought_chain("p1")] == ["c1"]
    assert [t["stream_id"] for t in manager.get_recent_thoughts()] == ["c1", "p1"]
    assert len(manager.get_recent_thoughts("reflection", limit=1)) == 1
    assert [t["stream_id"] for t in manager.get_thoughts_by_emotion("joy", 0.8)] == [
        "c1"
    ]
    assert manager.get_thought_by_stream_id("missing") is None
    with pytest.raises(ValueError):
        manager.get_thoughts_by_emotion("a.b")


def test_emotion_statistics_group_by_emotion(manager):
    manager.store_spike_batch(
        [_spike(intensity=0.2), _spike(intensity=0.6), _spike(emotion_type="fear")]
    )
    rows = manager.get_emotion_statistics(
        T0 - timedelta(hours=1), T0 + timedelta(hours=1)
    )
    joy = [r for r in rows if r["_id"] == "joy"][0]
    assert joy["count"] == 2 and joy["avg_intensity"] == pytest.approx(0.4)
    assert joy["max_intensity"] == pytest.approx(0.6)


def test_liquid_state_readout_fields_stay_none_when_unmeasured(manager):
    manager.store_liquid_state(
        LiquidState(
            timestamp=T0, reservoir_size=2, neuron_states=[0.1], input_pattern=[1.0]
        )
    )
    doc = manager.get_latest_liquid_state()
    for name in ("readout_values", "coherence", "emotional_valence", "arousal_level"):
        assert doc[name] is None
    assert manager.get_liquid_states_by_coherence(0.0) == []


def test_spike_membrane_potential_defaults_to_none_and_summary_has_no_neuron():
    spike = NeuralSpike(
        timestamp=T0, neuron_id=None, spike_time=0.0, emotion_type="joy", intensity=0.5
    )
    document = spike.to_dict()
    assert document["membrane_potential"] is None and document["neuron_id"] is None


def test_retention_counts_by_name_and_dry_run(manager):
    old = T0 - timedelta(days=40)
    for collection in TELEMETRY_COLLECTIONS:
        manager.db[collection].insert_one({"timestamp": old})
    counts = asyncio.run(manager.prune_telemetry(T0, dry_run=True))
    assert counts == {name: 1 for name in TELEMETRY_COLLECTIONS}
    assert [c for c in manager.db.client.calls if c[1] == "delete_many"] == []
    asyncio.run(manager.prune_telemetry(T0, dry_run=False))
    assert [c[0] for c in manager.db.client.calls if c[1] == "delete_many"] == list(
        TELEMETRY_COLLECTIONS
    )


def test_prune_thought_query_shape(manager):
    asyncio.run(manager.prune_thoughts(T0, 0.5, dry_run=False))
    shape = [c for c in manager.db.client.calls if c[1] == "delete_many"][0][2][0]
    assert set(shape) == {"timestamp", "coherence_score", "child_thoughts.0"}
    assert shape["child_thoughts.0"] == {"$exists": False}
    asyncio.run(manager.expire_gists(0.2, dry_run=False))
    gists = [c for c in manager.db.client.calls if c[1] == "delete_many"][-1][2][0]
    assert gists == {"retention_strength": {"$lt": 0.2}, "is_protected": {"$ne": True}}


def test_missing_pymongo_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymongo", None)
    with pytest.raises(ImportError) as caught:
        NeuralDataManager()
    assert "mongo" in str(caught.value)
    with pytest.raises(ImportError):
        MongoConversationStore()


def test_module_import_loads_no_pymongo():
    code = (
        "import sys, mtc.neural.mongodb_schemas;"
        "sys.exit(1 if 'pymongo' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-1000:]


def test_stores_satisfy_their_protocols(manager, client):
    from mtc.memory import protocols

    assert isinstance(manager, protocols.ThoughtGistStore)
    assert isinstance(manager, protocols.NeuralRetentionStore)
    assert isinstance(
        MongoConversationStore(database=client["c"]), protocols.ConversationTurnStore
    )
    assert manager.status()["backend"] == "mongodb"
    assert MongoConversationStore(database=client["c"]).status()["backend"] == "mongodb"
