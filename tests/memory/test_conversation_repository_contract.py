"""Conversation repository contract, run over every backend.

The in-memory repository always runs. The PostgreSQL repository runs under
the ``db`` marker on a throwaway database (``MTC_TEST_STORES=1``) and skips
otherwise. Both take the same injected clock, so every case uses fixed
instants on 2000-01-01. Speaker names are the synthetic fixture names.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from mtc.memory import protocols
from mtc.memory.conversation_store import PostgresConversationRepository
from mtc.memory.inmemory import InMemoryConversationRepository

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
SYSTEM = "Testagent"


class Clock:
    """A settable clock the repository reads its windows from."""

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
def raw_repository(request, clock):
    if request.param == "inmemory":
        return InMemoryConversationRepository(clock=clock)
    pool = request.getfixturevalue("throwaway_postgres")
    return PostgresConversationRepository(pool, clock=clock)


@pytest_asyncio.fixture
async def repo(raw_repository):
    await raw_repository.ensure_schema()
    return raw_repository


def _cid() -> str:
    return str(uuid.uuid4())


async def _conversation(repo, *participants, at=T0):
    cid = _cid()
    await repo.add_conversation(cid, [SYSTEM, *participants], at)
    return cid


@pytest.mark.asyncio
async def test_satisfies_protocol(repo):
    assert isinstance(repo, protocols.ConversationRepository)


@pytest.mark.asyncio
async def test_conversation_round_trip(repo):
    cid = _cid()
    returned = await repo.add_conversation(cid.upper(), [SYSTEM, "Alice"], T0)
    assert returned == cid
    row = await repo.get_conversation(cid)
    assert row == {
        "id": cid,
        "participants": [SYSTEM, "Alice"],
        "summary": None,
        "started_at": T0,
        "ended_at": None,
    }
    ended = T0 + timedelta(hours=1)
    assert await repo.end_conversation(cid, ended, summary="facts and decisions")
    row = await repo.get_conversation(cid)
    assert row["ended_at"] == ended and row["summary"] == "facts and decisions"
    # a None summary keeps the stored one
    assert await repo.end_conversation(cid, ended + timedelta(minutes=1))
    assert (await repo.get_conversation(cid))["summary"] == "facts and decisions"
    assert not await repo.end_conversation(_cid(), ended)
    assert await repo.get_conversation(_cid()) is None
    with pytest.raises(ValueError):
        await repo.add_conversation(cid, [SYSTEM], T0)
    with pytest.raises(ValueError, match="UUID"):
        await repo.add_conversation("not-a-uuid", [SYSTEM], T0)
    with pytest.raises(ValueError):
        await repo.add_message(_cid(), "Alice", "user", "hello", T0)


@pytest.mark.asyncio
async def test_message_fields_checked(repo):
    cid = await _conversation(repo, "Alice")
    with pytest.raises(ValueError, match="role"):
        await repo.add_message(cid, "Alice", "speaker", "hello", T0)
    with pytest.raises(ValueError, match="speaker"):
        await repo.add_message(cid, "", "user", "hello", T0)
    with pytest.raises(ValueError, match="speaker"):
        await repo.add_message(cid, "x" * 201, "user", "hello", T0)


@pytest.mark.asyncio
async def test_recent_messages_oldest_first_by_participant(repo, clock):
    alice = await _conversation(repo, "Alice")
    guest = await _conversation(repo, "Guest")
    for minute in (1, 2, 3):
        await repo.add_message(
            alice, "Alice", "user", f"a{minute}", T0 + timedelta(minutes=minute)
        )
        await repo.add_message(
            alice,
            SYSTEM,
            "assistant",
            f"r{minute}",
            T0 + timedelta(minutes=minute, seconds=30),
        )
    await repo.add_message(guest, "Guest", "user", "g1", T0 + timedelta(minutes=2))
    clock.now = T0 + timedelta(hours=1)

    rows = await repo.recent_messages("Alice", hours=24, limit=10)
    assert [r["content"] for r in rows] == ["a1", "r1", "a2", "r2", "a3", "r3"]
    assert [r["role"] for r in rows][:2] == ["user", "assistant"]
    assert set(rows[0]) == {
        "id",
        "conversation_id",
        "speaker",
        "role",
        "content",
        "timestamp",
        "to_speaker",
    }
    assert all(r["conversation_id"] == alice for r in rows)
    # the newest rows inside the limit, still oldest first
    rows = await repo.recent_messages("Alice", hours=24, limit=3)
    assert [r["content"] for r in rows] == ["r2", "a3", "r3"]
    # the window is on the injected clock and starts strictly after the
    # cutoff (r2 sits exactly on it)
    clock.now = T0 + timedelta(minutes=4)
    rows = await repo.recent_messages("Alice", hours=1.5 / 60, limit=10)
    assert [r["content"] for r in rows] == ["a3", "r3"]
    assert await repo.recent_messages("Alice", hours=24, limit=0) == []
    assert await repo.recent_messages("Operator", hours=24, limit=10) == []
    clock.now = T0 + timedelta(hours=1)
    timeline = await repo.timeline(hours=24, limit=20)
    assert [r["content"] for r in timeline] == [
        "a1",
        "r1",
        "a2",
        "g1",
        "r2",
        "a3",
        "r3",
    ]


@pytest.mark.asyncio
async def test_to_speaker_stored_and_filtered(repo, clock):
    cid = await _conversation(repo, "Alice")
    await repo.add_message(cid, "Alice", "user", "to the system", T0)
    await repo.add_message(
        cid,
        "Alice",
        "user",
        "to someone else",
        T0 + timedelta(seconds=1),
        to_speaker="Operator",
    )
    await repo.add_message(
        cid,
        "Alice",
        "user",
        "named the system",
        T0 + timedelta(seconds=2),
        to_speaker=SYSTEM,
    )
    clock.now = T0 + timedelta(minutes=5)
    default = await repo.recent_messages("Alice", hours=1, limit=10)
    assert [r["content"] for r in default] == ["to the system"]
    for_system = await repo.recent_messages(
        "Alice", hours=1, limit=10, addressee=SYSTEM
    )
    assert [r["content"] for r in for_system] == ["to the system", "named the system"]
    assert for_system[1]["to_speaker"] == SYSTEM
    for_operator = await repo.timeline(hours=1, limit=10, addressee="Operator")
    assert [r["content"] for r in for_operator] == ["to the system", "to someone else"]
    assert for_operator[1]["to_speaker"] == "Operator"
    # the latest-time read applies the same filter
    assert await repo.latest_message_time("Alice") == T0
    assert await repo.latest_message_time("Alice", addressee=SYSTEM) == T0 + timedelta(
        seconds=2
    )


@pytest.mark.asyncio
async def test_latest_message_time(repo):
    assert await repo.latest_message_time() is None
    alice = await _conversation(repo, "Alice")
    guest = await _conversation(repo, "Guest")
    assert await repo.latest_message_time("Alice") is None
    await repo.add_message(alice, "Alice", "user", "x", T0 + timedelta(minutes=1))
    await repo.add_message(guest, "Guest", "user", "y", T0 + timedelta(minutes=9))
    await repo.add_message(alice, SYSTEM, "assistant", "z", T0 + timedelta(minutes=3))
    assert await repo.latest_message_time("Alice") == T0 + timedelta(minutes=3)
    assert await repo.latest_message_time("Guest") == T0 + timedelta(minutes=9)
    assert await repo.latest_message_time() == T0 + timedelta(minutes=9)
    assert await repo.latest_message_time("Operator") is None


@pytest.mark.asyncio
async def test_emotional_rows(repo, clock):
    first = await repo.add_emotional_state(
        {
            "timestamp": T0,
            "primary_emotion": "curiosity",
            "intensity": 0.6,
            "valence": 0.2,
            "arousal": 0.4,
            "dominance": 0.1,
            "trigger_context": "synthetic trigger",
        }
    )
    clock.now = T0 + timedelta(minutes=1)
    second = await repo.add_emotional_state({"valence": -0.1, "arousal": 0.3})
    assert isinstance(first, int) and isinstance(second, int) and second != first
    rows = await repo.recent_emotional_states(limit=10)
    assert [r["id"] for r in rows] == [second, first]
    newest = rows[0]
    assert newest["timestamp"] == T0 + timedelta(minutes=1)
    # unmeasured coordinates and labels stay None
    assert newest["primary_emotion"] is None
    assert newest["intensity"] is None and newest["dominance"] is None
    assert newest["valence"] == pytest.approx(-0.1)
    assert rows[1]["primary_emotion"] == "curiosity"
    assert rows[1]["trigger_context"] == "synthetic trigger"
    assert await repo.recent_emotional_states(limit=1) == rows[:1]
    with pytest.raises(ValueError, match="unknown"):
        await repo.add_emotional_state({"mood": "fine"})
    with pytest.raises(ValueError, match="finite"):
        await repo.add_emotional_state({"valence": float("nan")})


@pytest.mark.asyncio
async def test_returned_rows_are_copies(repo, clock):
    cid = await _conversation(repo, "Alice")
    await repo.add_message(cid, "Alice", "user", "original", T0)
    await repo.add_emotional_state({"timestamp": T0, "valence": 0.5})
    clock.now = T0 + timedelta(minutes=1)
    rows = await repo.recent_messages("Alice", hours=1, limit=5)
    rows[0]["content"] = "mutated"
    conversation = await repo.get_conversation(cid)
    conversation["participants"].append("Guest")
    emotions = await repo.recent_emotional_states()
    emotions[0]["valence"] = -1.0
    assert (await repo.recent_messages("Alice", hours=1, limit=5))[0][
        "content"
    ] == "original"
    assert (await repo.get_conversation(cid))["participants"] == [SYSTEM, "Alice"]
    assert (await repo.recent_emotional_states())[0]["valence"] == 0.5


@pytest.mark.asyncio
async def test_timestamps_aware_utc(repo, clock):
    berlin = timezone(timedelta(hours=1))
    local = datetime(2000, 1, 1, 13, 0, 0, tzinfo=berlin)
    cid = _cid()
    await repo.add_conversation(cid, [SYSTEM, "Alice"], local)
    await repo.add_message(cid, "Alice", "user", "hi", local)
    clock.now = T0 + timedelta(minutes=1)
    row = (await repo.recent_messages("Alice", hours=1, limit=1))[0]
    assert row["timestamp"] == T0
    assert row["timestamp"].utcoffset() == timedelta(0)
    started = (await repo.get_conversation(cid))["started_at"]
    assert started.utcoffset() == timedelta(0) and started == T0
    assert (await repo.latest_message_time("Alice")).utcoffset() == timedelta(0)
    naive = datetime(2000, 1, 1, 12, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        await repo.add_conversation(_cid(), [SYSTEM], naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        await repo.add_message(cid, "Alice", "user", "hi", naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        await repo.end_conversation(cid, naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        await repo.add_emotional_state({"timestamp": naive})


def test_inmemory_cap_drops_oldest():
    import asyncio

    async def scenario():
        repo = InMemoryConversationRepository(max_items=2, clock=Clock())
        cid = await _conversation(repo, "Alice")
        for index in range(3):
            await repo.add_message(
                cid, "Alice", "user", f"m{index}", T0 - timedelta(minutes=3 - index)
            )
        return await repo.recent_messages("Alice", hours=1, limit=10)

    rows = asyncio.run(scenario())
    assert [r["content"] for r in rows] == ["m1", "m2"]
    assert InMemoryConversationRepository().status()["backend"] == "memory"


def test_inmemory_evicted_conversation_cascades_its_messages():
    """Dropping a conversation drops its messages, as the PostgreSQL foreign
    key's ON DELETE CASCADE does, so every read agrees."""
    import asyncio

    async def scenario():
        clock = Clock(T0 + timedelta(hours=1))
        repo = InMemoryConversationRepository(max_items=2, clock=clock)
        alice = await _conversation(repo, "Alice")
        await repo.add_message(alice, "Alice", "user", "a1", T0 + timedelta(minutes=5))
        guest = await _conversation(repo, "Guest")
        await repo.add_message(guest, "Guest", "user", "g1", T0 + timedelta(minutes=1))
        # The third conversation evicts the oldest one, and its message with it
        await _conversation(repo, "Operator")
        return repo, alice

    repo, alice = asyncio.run(scenario())
    assert asyncio.run(repo.get_conversation(alice)) is None
    status = repo.status()
    assert status["conversations"] == 2 and status["messages"] == 1
    timeline = asyncio.run(repo.timeline(hours=24, limit=10))
    assert [row["content"] for row in timeline] == ["g1"]
    # The newest message overall is the surviving one, as with a participant
    assert asyncio.run(repo.latest_message_time()) == T0 + timedelta(minutes=1)
    assert asyncio.run(repo.latest_message_time("Guest")) == T0 + timedelta(minutes=1)
    assert asyncio.run(repo.latest_message_time("Alice")) is None
    assert asyncio.run(repo.recent_messages("Alice", hours=24, limit=10)) == []
