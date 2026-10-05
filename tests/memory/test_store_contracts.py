"""Store contract tests, parametrized over backends.

Every case runs against each entry of ``BACKENDS``. The in-memory backend is
the default. A later phase adds a real store by appending one ``pytest.param``
entry (carrying ``pytest.mark.db``) whose factory builds the named store kind;
no test case below needs editing. A backend that does not support a capacity
cap sets ``supports_cap=False`` and the cap test skips for it.
"""

import asyncio
import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import pytest

from mtc.memory import inmemory, protocols
from mtc.memory.protocols import SnapshotEvent, SnapshotRecord

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

KINDS = ("snapshot", "learning", "speaker", "goal", "grab", "dream")

PROTOCOLS = {
    "snapshot": protocols.SnapshotStore,
    "learning": protocols.LearningStore,
    "speaker": protocols.SpeakerModelStore,
    "goal": protocols.GoalStore,
    "grab": protocols.AttentionGrabStore,
    "dream": protocols.DreamStore,
}


@dataclass(frozen=True)
class Backend:
    """A named factory building a store of each kind."""

    name: str
    factory: Callable[..., Any]
    supports_cap: bool = True


def _inmemory_factory(kind: str, max_items: Optional[int] = None) -> Any:
    classes = {
        "snapshot": inmemory.InMemorySnapshotStore,
        "learning": inmemory.InMemoryLearningStore,
        "speaker": inmemory.InMemorySpeakerModelStore,
        "goal": inmemory.InMemoryGoalStore,
        "grab": inmemory.InMemoryAttentionGrabStore,
        "dream": inmemory.InMemoryDreamStore,
    }
    if max_items is None:
        return classes[kind]()
    return classes[kind](max_items=max_items)


# Append a pytest.param(Backend(...), marks=pytest.mark.db, id="...") entry per
# real store; the cases below do not change.
BACKENDS = [
    pytest.param(Backend("inmemory", _inmemory_factory), id="inmemory"),
]


@pytest.fixture(params=BACKENDS)
def backend(request) -> Backend:
    return request.param


@pytest.fixture
def make_store(backend):
    def build(kind: str, **kwargs: Any) -> Any:
        return backend.factory(kind, **kwargs)

    return build


def _snapshot(offset_minutes: int, score: Optional[float] = 0.5, trigger="manual"):
    return SnapshotRecord(
        timestamp=T0 + timedelta(minutes=offset_minutes),
        trigger=trigger,
        overall_score=score,
        passing_count=3,
        phi_value=None,
        theory_scores={"gwt": 0.4},
        indicator_results={"GWT-1": {"status": "measured", "score": 0.4}},
        processing_time_ms=12.5,
        status="measured",
        coverage=0.9,
        metadata={"k": [1, 2]},
    )


def _event(offset_minutes: int, event_type="drop", snapshot_id=None):
    return SnapshotEvent(
        timestamp=T0 + timedelta(minutes=offset_minutes),
        event_type=event_type,
        description="synthetic",
        snapshot_id=snapshot_id,
        old_value=0.5,
        new_value=0.4,
    )


@pytest.mark.asyncio
async def test_round_trip_per_store(make_store):
    snaps = make_store("snapshot")
    sid = await snaps.add_snapshot(_snapshot(0))
    eid = await snaps.add_event(_event(0, snapshot_id=sid))
    latest = await snaps.latest_snapshot()
    assert latest.id == sid and latest.overall_score == 0.5
    assert latest.indicator_results["GWT-1"]["score"] == 0.4
    events = await snaps.events()
    assert events[0].id == eid and events[0].snapshot_id == sid

    learning = make_store("learning")
    await learning.add_experience({"speaker": "Alice", "outcome": "positive"})
    await learning.upsert_belief("b1", {"belief_id": "b1", "confidence": 0.6})
    assert (await learning.recent_experiences(5))[0]["speaker"] == "Alice"
    assert (await learning.load_beliefs(5))[0]["confidence"] == 0.6

    speakers = make_store("speaker")
    await speakers.upsert_model("Alice", {"speaker_name": "Alice", "trust": 0.5})
    assert (await speakers.load_models())[0]["speaker_name"] == "Alice"

    goals = make_store("goal")
    await goals.upsert_goal("g1", {"goal_id": "g1", "status": "active"})
    assert (await goals.load_goals(["active"]))[0]["goal_id"] == "g1"

    grabs = make_store("grab")
    await grabs.add_grab({"grab_id": "x1", "note": "n"})
    assert (await grabs.recent_grabs(5))[0]["grab_id"] == "x1"

    dreams = make_store("dream")
    await dreams.add_dream({"text": "d"})
    assert (await dreams.recent_dreams(5))[0]["text"] == "d"


@pytest.mark.asyncio
async def test_lists_newest_first(make_store):
    snaps = make_store("snapshot")
    for minute in (0, 30, 10):
        await snaps.add_snapshot(_snapshot(minute))
    stamps = [r.timestamp for r in await snaps.snapshots()]
    assert stamps == sorted(stamps, reverse=True)
    assert (await snaps.latest_snapshot()).timestamp == T0 + timedelta(minutes=30)
    for minute in (0, 30, 10):
        await snaps.add_event(_event(minute))
    stamps = [e.timestamp for e in await snaps.events()]
    assert stamps == sorted(stamps, reverse=True)

    learning = make_store("learning")
    for index in range(3):
        await learning.add_experience({"n": index})
        await learning.upsert_belief(f"b{index}", {"n": index})
    assert [d["n"] for d in await learning.recent_experiences(10)] == [2, 1, 0]
    assert [d["n"] for d in await learning.load_beliefs(10)] == [2, 1, 0]

    speakers = make_store("speaker")
    for index in range(3):
        await speakers.upsert_model(f"S{index}", {"n": index})
    assert [d["n"] for d in await speakers.load_models()] == [2, 1, 0]

    grabs = make_store("grab")
    dreams = make_store("dream")
    for index in range(3):
        await grabs.add_grab({"n": index})
        await dreams.add_dream({"n": index})
    assert [d["n"] for d in await grabs.recent_grabs(10)] == [2, 1, 0]
    assert [d["n"] for d in await dreams.recent_dreams(10)] == [2, 1, 0]


@pytest.mark.asyncio
async def test_limit_respected(make_store):
    snaps = make_store("snapshot")
    learning = make_store("learning")
    grabs = make_store("grab")
    for index in range(6):
        await snaps.add_snapshot(_snapshot(index))
        await snaps.add_event(_event(index))
        await learning.add_experience({"n": index})
        await learning.upsert_belief(f"b{index}", {"n": index})
        await grabs.add_grab({"n": index})
    assert len(await snaps.snapshots(limit=4)) == 4
    assert len(await snaps.events(limit=2)) == 2
    assert len(await learning.recent_experiences(3)) == 3
    assert len(await learning.load_beliefs(5)) == 5
    assert [d["n"] for d in await grabs.recent_grabs(2)] == [5, 4]
    since = T0 + timedelta(minutes=4)
    assert len(await snaps.snapshots(since=since, limit=100)) == 2


@pytest.mark.asyncio
async def test_upsert_replaces_and_returns_copies(make_store):
    learning = make_store("learning")
    await learning.upsert_belief("b1", {"belief_id": "b1", "confidence": 0.2})
    await learning.upsert_belief("b1", {"belief_id": "b1", "confidence": 0.9})
    beliefs = await learning.load_beliefs(10)
    assert len(beliefs) == 1 and beliefs[0]["confidence"] == 0.9

    # Mutating a returned document never reaches the store.
    beliefs[0]["confidence"] = -1.0
    assert (await learning.load_beliefs(10))[0]["confidence"] == 0.9

    # Mutating a document after writing it never reaches the store.
    doc = {"belief_id": "b2", "nested": {"values": [1, 2]}}
    await learning.upsert_belief("b2", doc)
    original = copy.deepcopy(doc)
    doc["nested"]["values"].append(3)
    stored = [d for d in await learning.load_beliefs(10) if d["belief_id"] == "b2"]
    assert stored[0] == original

    speakers = make_store("speaker")
    await speakers.upsert_model("Alice", {"trust": 0.5})
    await speakers.upsert_model("Alice", {"trust": 0.7})
    models = await speakers.load_models()
    assert len(models) == 1 and models[0]["trust"] == 0.7

    snaps = make_store("snapshot")
    await snaps.add_snapshot(_snapshot(0))
    first = await snaps.latest_snapshot()
    first.metadata["k"].append(99)
    first.theory_scores["gwt"] = 9.0
    again = await snaps.latest_snapshot()
    assert again.metadata == {"k": [1, 2]} and again.theory_scores == {"gwt": 0.4}


@pytest.mark.asyncio
async def test_load_goals_filters_status(make_store):
    goals = make_store("goal")
    await goals.upsert_goal("a", {"goal_id": "a", "status": "active"})
    await goals.upsert_goal("p", {"goal_id": "p", "status": "paused"})
    await goals.upsert_goal("d", {"goal_id": "d", "status": "completed"})
    ids = {g["goal_id"] for g in await goals.load_goals(["active", "paused"])}
    assert ids == {"a", "p"}
    assert await goals.load_goals(["abandoned"]) == []
    assert await goals.load_goals([]) == []
    await goals.upsert_goal("a", {"goal_id": "a", "status": "completed"})
    ids = {g["goal_id"] for g in await goals.load_goals(["active", "paused"])}
    assert ids == {"p"}


@pytest.mark.asyncio
async def test_delete_model_reports_existence(make_store):
    speakers = make_store("speaker")
    await speakers.upsert_model("Alice", {"trust": 0.5})
    assert await speakers.delete_model("Alice") is True
    assert await speakers.delete_model("Alice") is False
    assert await speakers.delete_model("Nobody") is False
    assert await speakers.load_models() == []


@pytest.mark.asyncio
async def test_snapshot_before_finds_older_record(make_store):
    snaps = make_store("snapshot")
    assert await snaps.snapshot_before(T0) is None
    for minute in (0, 10, 20):
        await snaps.add_snapshot(_snapshot(minute, score=minute / 100))
    found = await snaps.snapshot_before(T0 + timedelta(minutes=15))
    assert found.timestamp == T0 + timedelta(minutes=10)
    assert await snaps.snapshot_before(T0) is None
    # strictly before: a record at the moment itself is not before it
    found = await snaps.snapshot_before(T0 + timedelta(minutes=10))
    assert found.timestamp == T0


@pytest.mark.asyncio
async def test_events_filter_by_type(make_store):
    snaps = make_store("snapshot")
    await snaps.add_event(_event(0, "drop"))
    await snaps.add_event(_event(1, "rise"))
    await snaps.add_event(_event(2, "drop"))
    drops = await snaps.events(event_type="drop")
    assert [e.event_type for e in drops] == ["drop", "drop"]
    assert len(await snaps.events()) == 3
    assert await snaps.events(event_type="absent") == []
    since = T0 + timedelta(minutes=2)
    assert len(await snaps.events(since=since)) == 1


@pytest.mark.asyncio
async def test_max_items_drops_oldest(make_store, backend):
    if not backend.supports_cap:
        pytest.skip("backend has no capacity cap")
    learning = make_store("learning", max_items=3)
    for index in range(5):
        await learning.add_experience({"n": index})
        await learning.upsert_belief(f"b{index}", {"n": index})
    assert [d["n"] for d in await learning.recent_experiences(10)] == [4, 3, 2]
    assert [d["n"] for d in await learning.load_beliefs(10)] == [4, 3, 2]
    snaps = make_store("snapshot", max_items=2)
    for minute in range(4):
        await snaps.add_snapshot(_snapshot(minute))
        await snaps.add_event(_event(minute))
    assert len(await snaps.snapshots()) == 2
    assert len(await snaps.events()) == 2
    assert (await snaps.latest_snapshot()).timestamp == T0 + timedelta(minutes=3)


@pytest.mark.parametrize("kind", KINDS)
def test_implementations_satisfy_protocols(make_store, kind):
    assert isinstance(make_store(kind), PROTOCOLS[kind])


def test_naive_timestamps_are_rejected():
    store = inmemory.InMemorySnapshotStore()
    naive = _snapshot(0)
    naive.timestamp = datetime(2000, 1, 1, 12, 0, 0)
    with pytest.raises(ValueError):
        asyncio.run(store.add_snapshot(naive))
