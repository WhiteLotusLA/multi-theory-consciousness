"""Three isolated writes per turn, the provenance snapshot, emotional rows."""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mtc.consciousness.enhanced_global_workspace import ConsciousnessState
from mtc.consciousness.pad_affect import PADState
from mtc.memory.inmemory import (
    InMemoryConversationRepository,
    InMemoryConversationTurnStore,
    InMemoryWorkingMemory,
)
from mtc.memory.protocols import StoreUnavailable
from mtc.memory.resilient_wrappers import BUFFERED
from mtc.pipeline.context_types import ModulatedContext, TheorySections
from mtc.pipeline.message_persistence import (
    INTEGRATION_PROXY_LABEL,
    MessagePersistence,
    PersistResult,
    build_turn_snapshot,
    pad_coordinate,
)

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
CFG = SimpleNamespace(SYSTEM_NAME="Testagent", DEBUG=False, WORKING_MEMORY_CAPACITY=7)


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


class _FailingTurnStore(InMemoryConversationTurnStore):
    async def store_conversation_turn(self, **kwargs):
        raise StoreUnavailable("synthetic turn store outage")


class _FailingWorkingMemory(InMemoryWorkingMemory):
    async def add_to_working_memory(self, session_id, message):
        raise StoreUnavailable("synthetic working memory outage")

    async def clear_working_memory(self, session_id):
        raise StoreUnavailable("synthetic working memory outage")


def _persistence(clock=None, **stores):
    clock = clock or Clock()
    repository = stores.pop("repository", InMemoryConversationRepository(clock=clock))
    turn_store = stores.pop("turn_store", InMemoryConversationTurnStore(clock=clock))
    working = stores.pop(
        "working_memory", InMemoryWorkingMemory(capacity=7, clock=clock)
    )
    return MessagePersistence(
        repository,
        turn_store,
        working,
        settings=stores.pop("settings", CFG),
        clock=clock,
    )


def _state(**overrides):
    values = dict(
        is_conscious=True,
        primary_content=None,
        workspace_contents=[],
        ignition_events=1,
        integration_level=0.6,
        broadcast_coverage=0.5,
        attention_focus="message",
        attention_distribution={"message": 7.0},
        stream_position=3,
    )
    values.update(overrides)
    return ConsciousnessState(**values)


@pytest.mark.asyncio
async def test_three_writes_each_isolated():
    clock = Clock()
    repository = InMemoryConversationRepository(clock=clock)
    working = InMemoryWorkingMemory(capacity=7, clock=clock)
    persistence = _persistence(
        clock,
        repository=repository,
        turn_store=_FailingTurnStore(clock=clock),
        working_memory=working,
    )
    cid = str(uuid.uuid4())
    start = await persistence.store_conversation_start(cid, "verified:Alice")
    assert start.written == ("repository", "turn_store")
    result = await persistence.store_turn(cid, "Alice", "user", "hello there", 1)
    assert isinstance(result, PersistResult)
    assert result.written == ("repository", "working_memory")
    assert set(result.failed) == {"turn_store"}
    assert result.failed["turn_store"].startswith("StoreUnavailable")
    assert not result.ok
    # the repository and working-memory writes are in place
    clock.now = T0 + timedelta(minutes=1)
    rows = await repository.recent_messages("verified:Alice", hours=1, limit=5)
    assert [r["content"] for r in rows] == ["hello there"]
    assert [m["content"] for m in await working.get_working_memory(cid)] == [
        "hello there"
    ]
    assert persistence.status()["failures"] == {"turn_store": 1}


@pytest.mark.asyncio
async def test_failure_raises_under_debug():
    debug = SimpleNamespace(
        SYSTEM_NAME="Testagent", DEBUG=True, WORKING_MEMORY_CAPACITY=7
    )
    clock = Clock()
    persistence = _persistence(
        clock, turn_store=_FailingTurnStore(clock=clock), settings=debug
    )
    cid = str(uuid.uuid4())
    await persistence.store_conversation_start(cid, "verified:Alice")
    with pytest.raises(StoreUnavailable):
        await persistence.store_turn(cid, "Alice", "user", "hello", 1)


@pytest.mark.asyncio
async def test_participants_from_system_name_and_speaker():
    clock = Clock()
    repository = InMemoryConversationRepository(clock=clock)
    turns = InMemoryConversationTurnStore(clock=clock)
    persistence = _persistence(clock, repository=repository, turn_store=turns)
    cid = str(uuid.uuid4())
    await persistence.store_conversation_start(cid, "verified:Alice")
    row = await repository.get_conversation(cid)
    assert row["participants"] == ["Testagent", "verified:Alice"]
    assert row["started_at"] == T0
    assert turns.get_session(cid)["primary_participant"] == "verified:Alice"
    other = str(uuid.uuid4())
    await persistence.store_conversation_start(other, "Testagent")
    assert (await repository.get_conversation(other))["participants"] == ["Testagent"]


@pytest.mark.asyncio
async def test_snapshot_in_turn_metadata():
    clock = Clock()
    turns = InMemoryConversationTurnStore(clock=clock)
    persistence = _persistence(clock, turn_store=turns)
    state = _state(
        pad_pleasure=0.2,
        pad_arousal=0.45,
        pad_dominance=0.6,
        pad_label="content",
        architectural_phi=0.31,
        phi_warming_up=False,
    )
    modulated = ModulatedContext(
        prompt_context="The workspace focused on the message.",
        theory_sections=TheorySections(
            attention_guidance="attention report",
            global_workspace_context={"focus": "message", "ignition": 1},
        ),
        generation_params={"temperature": 0.7},
        behavioral_directives=["You may ask a follow-up question."],
        consciousness_self_awareness="self-awareness line",
        consciousness_state=state,
        workspace_winner="message",
        winner_is_message=True,
        cycle_ok=True,
        absent_signals={"rpt_deep_recurrence": "cadence_not_reached"},
        pad_sources_measured=("somatic_valence", "ignition_rate"),
    )
    snapshot = build_turn_snapshot(modulated, provider_id="mock")
    cid = str(uuid.uuid4())
    await persistence.store_conversation_start(cid, "verified:Alice")
    await persistence.store_turn(
        cid, "Testagent", "assistant", "[mock] reply", 2, snapshot=snapshot
    )
    turn = (await turns.get_conversation_turns(cid))[0]
    stored = turn["metadata"]["consciousness_snapshot"]
    assert stored["narrative"] == "The workspace focused on the message."
    assert stored["theory_sections"] == {
        "attention_guidance": "attention report",
        "global_workspace_context": {"focus": "message", "ignition": 1},
    }
    assert stored["directives"] == ["You may ask a follow-up question."]
    assert stored["self_awareness"] == "self-awareness line"
    assert stored["generation_params"] == {"temperature": 0.7}
    computed = stored["computed_state"]
    assert computed["pad"] == {
        "pleasure": 0.2,
        "arousal": 0.45,
        "dominance": 0.6,
        "label": "content",
    }
    assert computed["integration_proxy"] == {
        "label": INTEGRATION_PROXY_LABEL,
        "value": 0.31,
        "warming_up": False,
    }
    assert stored["pad_sources_measured"] == ["somatic_valence", "ignition_rate"]
    assert stored["absent_signals"] == {"rpt_deep_recurrence": "cadence_not_reached"}
    assert stored["provider_id"] == "mock"
    assert stored["cycle_ok"] is True
    # a turn stored without a snapshot carries no snapshot metadata
    await persistence.store_turn(cid, "Alice", "user", "plain turn", 3)
    plain = (await turns.get_conversation_turns(cid))[-1]
    assert plain["content"] == "plain turn" and plain["metadata"] is None
    # an unavailable cycle snapshots honestly: nothing narrated, no state
    empty = build_turn_snapshot(ModulatedContext.unavailable("boom", 0.7), "mock")
    assert empty["narrative"] == "" and empty["computed_state"] is None
    assert empty["failure"] == "boom" and empty["cycle_ok"] is False


@pytest.mark.asyncio
async def test_emotional_history_arousal_from_pad():
    clock = Clock()
    repository = InMemoryConversationRepository(clock=clock)
    persistence = _persistence(clock, repository=repository)
    # the attention distribution carries a win count; the arousal comes from PAD
    state = _state(pad_pleasure=-0.1, pad_arousal=0.35, pad_dominance=0.5)
    row_id = await persistence.store_emotional_state(
        state, "curious", 0.62, "synthetic trigger"
    )
    assert isinstance(row_id, int)
    row = (await repository.recent_emotional_states(1))[0]
    assert row["arousal"] == pytest.approx(0.35)
    assert row["valence"] == pytest.approx(-0.1)
    assert row["dominance"] == pytest.approx(0.5)
    assert row["primary_emotion"] == "curious"
    assert row["intensity"] == pytest.approx(0.62)
    assert row["trigger_context"] == "synthetic trigger"
    assert row["timestamp"] == T0
    # a PADState and a mapping read the same way
    await persistence.store_emotional_state(
        PADState(pleasure=0.3, arousal=0.7, dominance=0.4), None, None, None
    )
    row = (await repository.recent_emotional_states(1))[0]
    assert row["arousal"] == pytest.approx(0.7)
    assert row["primary_emotion"] is None and row["intensity"] is None
    assert pad_coordinate({"valence": 0.1, "arousal": 0.2}) == {
        "valence": 0.1,
        "arousal": 0.2,
        "dominance": None,
    }
    # a state with no PAD recorded leaves the coordinate unmeasured
    await persistence.store_emotional_state(_state(), "calm", 0.4, None)
    row = (await repository.recent_emotional_states(1))[0]
    assert row["arousal"] is None and row["valence"] is None


@pytest.mark.asyncio
async def test_working_memory_failure_caught():
    clock = Clock()
    repository = InMemoryConversationRepository(clock=clock)
    turns = InMemoryConversationTurnStore(clock=clock)
    persistence = _persistence(
        clock,
        repository=repository,
        turn_store=turns,
        working_memory=_FailingWorkingMemory(capacity=7, clock=clock),
    )
    cid = str(uuid.uuid4())
    await persistence.store_conversation_start(cid, "verified:Alice")
    result = await persistence.store_turn(cid, "Alice", "user", "hello", 1)
    assert result.written == ("repository", "turn_store")
    assert set(result.failed) == {"working_memory"}
    end = await persistence.store_conversation_end(cid)
    assert set(end.failed) == {"working_memory"}
    assert "repository" in end.written


@pytest.mark.asyncio
async def test_conversation_end_clears_working_memory():
    clock = Clock()
    working = InMemoryWorkingMemory(capacity=7, clock=clock)
    repository = InMemoryConversationRepository(clock=clock)
    persistence = _persistence(clock, repository=repository, working_memory=working)
    cid = str(uuid.uuid4())
    await persistence.store_conversation_start(cid, "verified:Alice")
    await persistence.store_turn(cid, "Alice", "user", "hello", 1)
    assert await working.get_working_memory(cid)
    clock.now = T0 + timedelta(minutes=10)
    result = await persistence.store_conversation_end(cid, summary="A short visit.")
    assert result.written == ("repository", "turn_store", "working_memory")
    assert await working.get_working_memory(cid) == []
    row = await repository.get_conversation(cid)
    assert row["ended_at"] == T0 + timedelta(minutes=10)
    assert row["summary"] == "A short visit."
    # ending an unknown conversation is reported, not assumed
    unknown = await persistence.store_conversation_end(str(uuid.uuid4()))
    assert unknown.failed["repository"] == "conversation end not confirmed"


@pytest.mark.asyncio
async def test_no_default_summary_text():
    clock = Clock()
    repository = InMemoryConversationRepository(clock=clock)
    turns = InMemoryConversationTurnStore(clock=clock)
    persistence = _persistence(clock, repository=repository, turn_store=turns)
    cid = str(uuid.uuid4())
    await persistence.store_conversation_start(cid, "verified:Alice")
    await persistence.store_conversation_end(cid)
    assert (await repository.get_conversation(cid))["summary"] is None
    session = turns.get_session(cid)
    assert "summary" not in session
    assert all("ended" not in str(value) for value in session.values())


@pytest.mark.asyncio
async def test_buffered_writes_reported_as_buffered():
    clock = Clock()

    class BufferingTurns(InMemoryConversationTurnStore):
        async def store_conversation_turn(self, **kwargs):
            return BUFFERED

    class BufferingRepository(InMemoryConversationRepository):
        async def add_message(self, *args, **kwargs):
            return BUFFERED

    persistence = _persistence(
        clock,
        repository=BufferingRepository(clock=clock),
        turn_store=BufferingTurns(clock=clock),
    )
    cid = str(uuid.uuid4())
    await persistence.store_conversation_start(cid, "verified:Alice")
    result = await persistence.store_turn(cid, "Alice", "user", "hello", 1)
    assert result.buffered == ("repository", "turn_store")
    assert result.written == ("working_memory",)
    assert result.ok
