"""Context assembly: concurrent isolated sources and the brain window."""

import asyncio
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from mtc.consciousness.conversation_coherence import ConversationCoherence
from mtc.core.config import Settings
from mtc.pipeline.context_assembly import (
    BrainWindow,
    GatheredContext,
    build_brain_window,
    gather_context,
)
from mtc.pipeline.context_budget import CLOCK_SPEAKER
from mtc.pipeline.session_state import Session
from mtc.pipeline.speaker_trust import SpeakerEntry, SpeakerTrust

T0 = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)
TOKEN_ENV = "MTC_TEST_CONTEXT_ASSEMBLY_TOKEN"


def _settings(**overrides) -> Settings:
    values = {"SYSTEM_NAME": "Testagent", "CREATOR_NAME": "Operator", "DEBUG": False}
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def cfg():
    return _settings()


@pytest.fixture
def trust(monkeypatch, cfg):
    token = secrets.token_hex(16)
    monkeypatch.setenv(TOKEN_ENV, token)
    source = SpeakerTrust(
        [SpeakerEntry("Operator", "creator", 0.9, token_env=TOKEN_ENV)], settings=cfg
    )
    return source, token


def _entry(role: str, content: str, speaker: str = None, at: datetime = T0, **extra):
    entry: Dict[str, Any] = {"role": role, "content": content, "timestamp": at}
    if speaker is not None:
        entry["speaker"] = speaker
    entry.update(extra)
    return entry


def _session(history: List[Dict[str, Any]]) -> Session:
    return Session(key="claimed:Alice", conversation_id="c1", history=history)


class FakeMemoryManager:
    def __init__(self, blocks=None, memories=None, recent=None, fail=False):
        self.blocks = blocks or {}
        self.memories = memories or []
        self.recent = recent or []
        self.fail = fail
        self.calls: List[str] = []

    async def get_core_memory(self, name):
        self.calls.append(f"core:{name}")
        if self.fail:
            raise RuntimeError("store down")
        return self.blocks.get(name)

    async def semantic_recall(self, query, top_k=None, min_score=0.5):
        self.calls.append("recall")
        if self.fail:
            raise RuntimeError("recall down")
        return {
            "memories": list(self.memories),
            "concepts": [{"name": "gardens"}],
            "search_stats": {"method": "text"},
        }

    async def get_recent_important_memories(self, hours, min_importance, limit):
        return list(self.recent)


class FakeThoughtStore:
    def __init__(self, records=(), fail=False):
        self.records = list(records)
        self.fail = fail

    async def expressible(self, limit, exclude_type_prefixes=()):
        if self.fail:
            raise RuntimeError("thought store down")
        return list(self.records)[:limit]


class FakeGistStore:
    def __init__(self, fail=False):
        self.fail = fail

    async def active_gists(self, min_retention=0.5, limit=5):
        if self.fail:
            raise RuntimeError("gist store down")
        return []


class BarrierSource:
    """Two sources that only finish once both have started."""

    def __init__(self):
        self.started = 0
        self.both = asyncio.Event()

    async def arrive(self):
        self.started += 1
        if self.started >= 2:
            self.both.set()
        await asyncio.wait_for(self.both.wait(), timeout=2.0)


@pytest.mark.asyncio
async def test_gather_runs_sources_concurrently_and_isolates_failures(cfg):
    barrier = BarrierSource()

    class SlowManager(FakeMemoryManager):
        async def get_core_memory(self, name):
            await barrier.arrive()
            return "block"

    class SlowThoughts(FakeThoughtStore):
        async def expressible(self, limit, exclude_type_prefixes=()):
            await barrier.arrive()
            return []

    gathered = await gather_context(
        "hello",
        _session([]),
        memory_manager=SlowManager(),
        thought_store=SlowThoughts(),
        gist_store=FakeGistStore(fail=True),
        settings=cfg,
    )
    assert isinstance(gathered, GatheredContext)
    assert gathered.core_memory == {
        n: "block" for n in ("identity", "persona", "human")
    }
    assert gathered.ctm_thoughts == []
    assert gathered.gists == []
    assert "gists" in gathered.errors
    assert gathered.errors["gists"].startswith("RuntimeError")
    assert set(gathered.errors) == {"gists"}


@pytest.mark.asyncio
async def test_core_memory_blocks_loaded(cfg):
    manager = FakeMemoryManager(
        blocks={"identity": "I", "persona": "P", "human": "H"},
        memories=[
            {"id": 1, "content": "the garden has roses", "similarity_score": 0.9}
        ],
    )
    gathered = await gather_context(
        "roses?", _session([]), memory_manager=manager, settings=cfg
    )
    assert gathered.core_memory == {"identity": "I", "persona": "P", "human": "H"}
    assert [m["id"] for m in gathered.memories] == [1]
    assert gathered.concepts == [{"name": "gardens"}]
    assert gathered.recall_method == "text"
    assert gathered.errors == {}


@pytest.mark.asyncio
async def test_echo_suppression_applied(cfg):
    history = [_entry("user", "Alice said: the garden has roses and tulips")]
    manager = FakeMemoryManager(
        memories=[
            {"id": 1, "content": "the garden has roses and tulips"},
            {"id": 2, "content": "a completely different memory about rivers"},
        ]
    )
    gathered = await gather_context(
        "roses?", _session(history), memory_manager=manager, settings=cfg
    )
    assert [m["id"] for m in gathered.memories] == [2]


@pytest.mark.asyncio
async def test_coherence_from_existing_module(cfg):
    history = [
        _entry("user", "Tell me about the garden roses", speaker="[Alice]"),
        _entry("assistant", "The roses bloom in June."),
    ]
    gathered = await gather_context(
        "And the tulips?",
        _session(history),
        memory_manager=FakeMemoryManager(),
        coherence=ConversationCoherence(),
        settings=cfg,
    )
    assert "topics" in gathered.coherence
    assert "coherence_instructions" in gathered.coherence
    assert gathered.coherence["message_count"] == 2
    assert gathered.errors == {}


@pytest.mark.asyncio
async def test_gather_without_memory_manager_names_it(cfg):
    gathered = await gather_context(
        "hello", _session([]), memory_manager=None, settings=cfg
    )
    assert gathered.memories == []
    assert gathered.core_memory == {}
    assert "memory_manager" in gathered.errors
    assert gathered.recall_method == "none"


def test_window_labels_from_speaker_trust(cfg, trust):
    source, token = trust
    claimed = source.verify("Alice", None)
    creator = source.verify("Operator", token)
    history = [
        _entry("user", "first", speaker="Alice"),
        _entry("assistant", "reply"),
        _entry("user", "second"),
    ]
    window = build_brain_window(
        _session([dict(e) for e in history]), claimed, source, settings=cfg, now=T0
    )
    assert isinstance(window, BrainWindow)
    assert window.history[0]["speaker"] == "[Alice]"
    assert "speaker" not in window.history[1]
    assert window.current_prompt == "second"
    creator_window = build_brain_window(
        _session([dict(e) for e in history]), creator, source, settings=cfg, now=T0
    )
    assert "speaker" not in creator_window.history[0]


def test_current_prompt_split(cfg, trust):
    source, _token = trust
    identity = source.verify("Alice", None)
    later = T0 + timedelta(minutes=5)
    history = [
        _entry("user", "earlier question", at=T0),
        _entry("assistant", "earlier answer", at=T0 + timedelta(minutes=1)),
        _entry("user", "the current message", at=later),
    ]
    session = _session(history)
    window = build_brain_window(session, identity, source, settings=cfg, now=later)
    assert window.current_prompt == "the current message"
    assert [e["content"] for e in window.history] == [
        "earlier question",
        "earlier answer",
    ]
    assert window.last_turn_at == T0 + timedelta(minutes=1)
    # the session's own history is untouched
    assert len(session.history) == 3


def test_dedup_assistant_turns(cfg, trust):
    source, _token = trust
    identity = source.verify("Alice", None)
    repeated = "This exact sentence is repeated in several assistant turns here."
    history = []
    for index in range(3):
        history.append(_entry("user", f"question {index}"))
        history.append(_entry("assistant", f"{repeated} Extra {index}."))
    history.append(_entry("user", "now"))
    window = build_brain_window(_session(history), identity, source, settings=cfg)
    assistant = [e["content"] for e in window.history if e["role"] == "assistant"]
    assert sum(repeated in text for text in assistant) == 2
    assert assistant[2] == "Extra 2."
    assert any("repeated" in note for note in window.window_notes)


def test_night_note_inserted_only_when_enabled(trust):
    source, _token = trust
    identity = source.verify("Alice", None)
    evening = datetime(2000, 1, 1, 21, 0, tzinfo=timezone.utc)
    morning = datetime(2000, 1, 2, 9, 0, tzinfo=timezone.utc)
    history = [
        _entry("user", "good evening", at=evening),
        _entry("assistant", "good evening to you", at=evening + timedelta(minutes=1)),
        _entry("user", "good morning", at=morning),
    ]
    on = _settings(NIGHT_SEAM_MARKER_ENABLED=True)
    window = build_brain_window(
        _session([dict(e) for e in history]), identity, source, settings=on, now=morning
    )
    notes = [e for e in window.history if e.get("is_clock_note")]
    assert len(notes) == 1
    assert notes[0]["speaker"] == CLOCK_SPEAKER
    off = _settings(NIGHT_SEAM_MARKER_ENABLED=False)
    window = build_brain_window(
        _session([dict(e) for e in history]),
        identity,
        source,
        settings=off,
        now=morning,
    )
    assert not [e for e in window.history if e.get("is_clock_note")]


def test_window_never_trims_tokens(cfg, trust):
    source, _token = trust
    identity = source.verify("Alice", None)
    history = []
    for index in range(300):
        history.append(_entry("user", f"question {index} " + ("w" * 500)))
        history.append(_entry("assistant", f"answer {index} " + ("x" * 500)))
    history.append(_entry("user", "current"))
    window = build_brain_window(_session(history), identity, source, settings=cfg)
    assert len(window.history) == 600


def test_tool_turns_become_labelled_user_turns(cfg, trust):
    source, _token = trust
    identity = source.verify("Alice", None)
    history = [
        _entry("user", "look it up"),
        _entry("tool", "three results", speaker="search"),
        _entry("user", "and now?"),
    ]
    window = build_brain_window(_session(history), identity, source, settings=cfg)
    tool_turn = window.history[1]
    assert tool_turn["role"] == "user"
    assert tool_turn["speaker"] == "tool search"
    assert tool_turn["content"] == "three results"
    assert window.current_prompt == "and now?"
    assert all(e["role"] in ("user", "assistant") for e in window.history)


def test_tool_turn_as_newest_entry_is_not_the_current_prompt(cfg, trust):
    """A turn plugin's tool turn is appended after the user's message; the
    user's message stays the current prompt and the tool turn joins history."""
    source, _token = trust
    identity = source.verify("Alice", None)
    history = [
        _entry("user", "What time is it?"),
        _entry("tool", "12:00", speaker="clock"),
    ]
    window = build_brain_window(_session(history), identity, source, settings=cfg)
    assert window.current_prompt == "What time is it?"
    assert len(window.history) == 1
    tool_turn = window.history[0]
    assert tool_turn["role"] == "user"
    assert tool_turn["speaker"] == "tool clock"
    assert tool_turn["content"] == "12:00"
    assert tool_turn["is_tool_turn"] is True


def test_newest_tool_turn_after_earlier_turns_keeps_the_latest_message(cfg, trust):
    """With earlier turns before it, a converted tool turn that is the newest
    entry never becomes the prompt: the latest speech is the current prompt,
    the earlier user turn keeps the trust label, and the tool turn stays last
    in the history under its own label, unlabelled by the speaker's name."""
    source, _token = trust
    identity = source.verify("Alice", None)
    history = [
        _entry("user", "Is the garden watered?", at=T0),
        _entry("assistant", "It was watered at noon.", at=T0 + timedelta(minutes=1)),
        _entry("user", "What time is it now?", at=T0 + timedelta(minutes=2)),
        _entry("tool", "12:03", speaker="clock", at=T0 + timedelta(minutes=2)),
    ]
    session = _session([dict(e) for e in history])
    window = build_brain_window(session, identity, source, settings=cfg)
    assert window.current_prompt == "What time is it now?"
    assert [e["content"] for e in window.history] == [
        "Is the garden watered?",
        "It was watered at noon.",
        "12:03",
    ]
    assert window.history[0]["speaker"] == "[Alice]"
    newest = window.history[-1]
    assert newest["role"] == "user"
    assert newest["speaker"] == "tool clock"
    assert newest["is_tool_turn"] is True
    assert any("tool result" in note for note in window.window_notes)
    # the session keeps the tool entry as stored
    assert session.history[-1]["role"] == "tool"
    assert session.history[-1]["speaker"] == "clock"


def test_last_turn_at_ignores_entries_of_this_turn(cfg, trust):
    source, _token = trust
    identity = source.verify("Alice", None)
    now = T0 + timedelta(minutes=5)
    history = [
        _entry("user", "earlier question", at=T0),
        _entry("assistant", "earlier answer", at=T0 + timedelta(minutes=1)),
        _entry("user", "the current message", at=now),
        # a turn plugin's result, appended after the message with its time
        _entry("tool", "12:05", speaker="clock_tool", at=now),
    ]
    window = build_brain_window(
        _session(history), identity, source, settings=cfg, now=now
    )
    assert window.current_prompt == "the current message"
    assert window.last_turn_at == T0 + timedelta(minutes=1)
    # nothing earlier than now: the session's own record is the fallback
    fresh = _session([_entry("user", "first words", at=now)])
    fresh.last_turn_at = T0
    window = build_brain_window(fresh, identity, source, settings=cfg, now=now)
    assert window.last_turn_at == T0
    # without now the newest entry of the window stands, as before
    window = build_brain_window(_session(history[:3]), identity, source, settings=cfg)
    assert window.last_turn_at == T0 + timedelta(minutes=1)
