"""Memory recall, echo suppression, rehydration and concept capture."""

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtc.core.embeddings import HashEmbedder
from mtc.memory.enhanced_memory_manager import EnhancedMemoryManager
from mtc.memory.episodic_store import InMemoryEpisodicStore
from mtc.memory.inmemory import (
    InMemoryConversationRepository,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from mtc.memory.protocols import StoreUnavailable
from mtc.pipeline import memory_retrieval
from mtc.pipeline.memory_retrieval import (
    CONCEPT_DESCRIPTION,
    CONCEPT_STOP_WORDS,
    RecallResult,
    concept_candidates,
    extract_and_store_concepts,
    latest_turn_at,
    rehydrate_recent_history,
    retrieve_relevant_memories,
    suppress_rag_echoes,
)

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
CFG = SimpleNamespace(RAG_TOP_K=10, RAG_MIN_SCORE=0.0, SYSTEM_NAME="Testagent")


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


def _manager(vector=False, episodic=None, clock=None):
    clock = clock or Clock()
    store = episodic if episodic is not None else InMemoryEpisodicStore(clock=clock)
    vectors = (
        InMemoryVectorStore(HashEmbedder(), allow_non_semantic=True) if vector else None
    )
    return EnhancedMemoryManager(
        episodic_store=store,
        vector_store=vectors,
        persona={"name": "Testagent"},
        clock=clock,
    )


class _BrokenEpisodic(InMemoryEpisodicStore):
    async def search_text(self, *args, **kwargs):
        raise StoreUnavailable("synthetic outage")

    async def recent(self, *args, **kwargs):
        raise StoreUnavailable("synthetic outage")


@pytest.mark.asyncio
async def test_recall_labels_method():
    text = _manager()
    await text.store_memory("the garden soil test came back fine", importance=3.0)
    result = await retrieve_relevant_memories(text, "garden soil", settings=CFG)
    assert isinstance(result, RecallResult)
    assert result.method == "text"
    assert "vector_error" in result.errors
    assert [m["content"] for m in result.memories] == [
        "the garden soil test came back fine"
    ]
    assert result.memories[0]["similarity_score"] is None

    vector = _manager(vector=True)
    await vector.store_memory("the garden soil test came back fine", importance=3.0)
    result = await retrieve_relevant_memories(vector, "garden soil test", settings=CFG)
    assert result.method == "vector"
    assert result.memories and result.memories[0]["match"] == "vector"

    broken = _manager(episodic=_BrokenEpisodic(clock=Clock()))
    result = await retrieve_relevant_memories(broken, "garden", settings=CFG)
    assert result.method == "none"
    assert result.memories == []
    assert "text_error" in result.errors and "recency" in result.errors

    result = await retrieve_relevant_memories(None, "garden", settings=CFG)
    assert result.method == "none" and result.errors == {
        "manager": "memory manager absent"
    }


@pytest.mark.asyncio
async def test_recall_reads_settings_at_call_time():
    seen = {}

    class Manager:
        async def semantic_recall(self, query, top_k=None, min_score=0.5):
            seen.update(top_k=top_k, min_score=min_score)
            return {"memories": [], "concepts": [], "search_stats": {"method": "text"}}

    cfg = SimpleNamespace(RAG_TOP_K=7, RAG_MIN_SCORE=0.25, SYSTEM_NAME="Testagent")
    await retrieve_relevant_memories(Manager(), "q", settings=cfg)
    assert seen == {"top_k": 7, "min_score": 0.25}
    await retrieve_relevant_memories(Manager(), "q", top_k=2, min_score=0.9)
    assert seen == {"top_k": 2, "min_score": 0.9}


@pytest.mark.asyncio
async def test_recency_anchor_first_and_deduped():
    manager = _manager()
    important = await manager.store_memory(
        "the garden budget was approved", importance=9.0
    )
    await manager.store_memory("the garden fence is blue", importance=2.0)
    result = await retrieve_relevant_memories(manager, "garden", settings=CFG)
    contents = [m["content"] for m in result.memories]
    # the recent important memory comes first, and only once although text
    # search found it too
    assert contents[0] == "the garden budget was approved"
    assert result.memories[0].get("recency_anchored") is True
    assert contents.count("the garden budget was approved") == 1
    assert "the garden fence is blue" in contents
    assert important.postgres_id == result.memories[0]["id"]


def test_echo_suppression():
    history = [
        {"role": "user", "content": "Did the garden soil test come back yet?"},
        {"role": "assistant", "content": "It did, and it looked fine."},
    ]
    memories = [
        {"id": 1, "content": "Alice said: Did the garden soil test come back yet?"},
        {"id": 2, "content": "The fence was painted blue last spring."},
        {"id": 3, "content": ""},
    ]
    kept = suppress_rag_echoes(memories, history)
    assert [m["id"] for m in kept] == [2, 3]
    # a strict threshold keeps a partial overlap
    partial = [{"id": 4, "content": "garden soil fence paint ladder rake"}]
    assert suppress_rag_echoes(partial, history) == partial
    assert suppress_rag_echoes(partial, history, threshold=0.3) == []
    assert suppress_rag_echoes(memories, []) == memories


async def _repository_with_rows(clock):
    repo = InMemoryConversationRepository(clock=clock)
    cid = str(uuid.uuid4())
    await repo.add_conversation(cid, ["Testagent", "auth:Alice"], T0)
    rows = [
        ("Alice", "user", "first question", None),
        ("Testagent", "assistant", "first answer", None),
        ("Alice", "user", "aside to someone else", "Operator"),
        ("Testagent", "assistant", "named reply", "Testagent"),
    ]
    for offset, (speaker, role, content, to) in enumerate(rows):
        await repo.add_message(
            cid, speaker, role, content, T0 + timedelta(minutes=offset), to_speaker=to
        )
    return repo, cid


@pytest.mark.asyncio
async def test_rehydrate_uses_stored_role_not_name():
    clock = Clock(T0 + timedelta(hours=1))
    repo, cid = await _repository_with_rows(clock)
    # a speaker whose name matches the system's still keeps the stored role
    await repo.add_message(
        cid,
        "Testagent",
        "user",
        "a user who shares the name",
        T0 + timedelta(minutes=9),
    )
    history = await rehydrate_recent_history(repo, "auth:Alice", settings=CFG)
    assert [(h["speaker"], h["role"]) for h in history] == [
        ("Alice", "user"),
        ("Testagent", "assistant"),
        ("Testagent", "assistant"),
        ("Testagent", "user"),
    ]
    assert set(history[0]) == {
        "role",
        "speaker",
        "content",
        "timestamp",
        "conversation_id",
    }
    assert history[0]["timestamp"] == T0
    # a name without the session key reads nothing
    assert await rehydrate_recent_history(repo, "Alice", settings=CFG) == []
    assert await rehydrate_recent_history(None, "auth:Alice", settings=CFG) == []


@pytest.mark.asyncio
async def test_rehydrate_excludes_messages_addressed_elsewhere():
    clock = Clock(T0 + timedelta(hours=1))
    repo, _ = await _repository_with_rows(clock)
    history = await rehydrate_recent_history(repo, "auth:Alice", settings=CFG)
    contents = [h["content"] for h in history]
    assert "aside to someone else" not in contents
    assert contents == ["first question", "first answer", "named reply"]
    # the window bounds the rows
    clock.now = T0 + timedelta(hours=30)
    assert await rehydrate_recent_history(repo, "auth:Alice", settings=CFG) == []


@pytest.mark.asyncio
async def test_rehydrate_propagates_store_failure():
    class Down:
        async def recent_messages(self, *args, **kwargs):
            raise StoreUnavailable("synthetic outage")

    with pytest.raises(StoreUnavailable):
        await rehydrate_recent_history(Down(), "auth:Alice", settings=CFG)


@pytest.mark.asyncio
async def test_latest_turn_at_none_when_empty():
    repo = InMemoryConversationRepository(clock=Clock())
    assert await latest_turn_at(repo, settings=CFG) is None
    assert await latest_turn_at(None, settings=CFG) is None
    repo, _ = await _repository_with_rows(Clock(T0 + timedelta(hours=1)))
    latest = await latest_turn_at(repo, "auth:Alice", settings=CFG)
    # the aside at minute two is addressed elsewhere; the newest addressed to
    # the system is the named reply at minute three
    assert latest == T0 + timedelta(minutes=3)
    assert latest.utcoffset() == timedelta(0)


@pytest.mark.asyncio
async def test_concepts_through_graph_seam():
    graph = InMemoryGraphStore()
    calls = []

    class Recording:
        def merge_concept(self, name, description="", category="conversation"):
            calls.append(("merge", name, description, category))
            graph.merge_concept(name, description, category)

        def link_concept_to_conversation(self, name, conversation_id, speaker):
            calls.append(("link", name, conversation_id, speaker))
            graph.link_concept_to_conversation(name, conversation_id, speaker)

    text = "The telescope arrived. The telescope mirror needs cleaning before winter."
    stored = await extract_and_store_concepts(
        Recording(), text, "conv-1", "Alice", max_concepts=4
    )
    assert stored == 4
    merged = [c[1] for c in calls if c[0] == "merge"]
    assert merged[0] == "telescope"
    assert set(merged) == {"telescope", "arrived", "mirror", "cleaning"}
    assert all(c[3] == "conversation" for c in calls if c[0] == "merge")
    assert [c[2:] for c in calls if c[0] == "link"][0] == ("conv-1", "Alice")
    assert "telescope" in graph.list_concept_names()
    assert await extract_and_store_concepts(None, text, "conv-1", "Alice") == 0
    assert await extract_and_store_concepts(Recording(), text, "c", "A", 1) == 1
    # the module holds no graph or SQL query language
    source = Path(memory_retrieval.__file__).read_text(encoding="utf-8")
    for fragment in ("MERGE (", "MATCH (", "SELECT ", "session.run", "INSERT "):
        assert fragment not in source


def test_stop_words_generic():
    assert "about" in CONCEPT_STOP_WORDS and "would" in CONCEPT_STOP_WORDS
    assert all(word == word.lower() and word.isalpha() for word in CONCEPT_STOP_WORDS)
    assert concept_candidates("About the weather, would you think so?") == ["weather"]
    # speaker and excluded names never become concepts
    assert concept_candidates("Alice watered the orchids", exclude=["Alice"]) == [
        "watered",
        "orchids",
    ]
    # a bracketed label excludes the bare name as well as the label
    assert concept_candidates("Alice watered the orchids", exclude=["[Alice]"]) == [
        "watered",
        "orchids",
    ]
    # every word of a name with several words is excluded
    assert concept_candidates(
        "Guest Operator watered orchids", exclude=["[Guest Operator]"]
    ) == ["watered", "orchids"]


@pytest.mark.asyncio
async def test_concept_description_names_no_speaker():
    calls = []

    class Recording:
        def merge_concept(self, name, description="", category="conversation"):
            calls.append(description)

        def link_concept_to_conversation(self, name, conversation_id, speaker):
            pass

    await extract_and_store_concepts(
        Recording(), "Operator mentioned the orchard harvest", "conv-2", "Operator"
    )
    assert calls and all(d == CONCEPT_DESCRIPTION for d in calls)
    assert all("Operator" not in d for d in calls)


@pytest.mark.asyncio
async def test_recent_memory_surfaces_when_recall_misses_it():
    manager = _manager()
    await manager.store_memory("the orchard plan was approved", importance=9.0)
    result = await retrieve_relevant_memories(manager, "unrelated words", settings=CFG)
    assert [m["content"] for m in result.memories] == ["the orchard plan was approved"]
    assert result.memories[0]["recency_anchored"] is True


@pytest.mark.asyncio
async def test_recalled_memories_keep_their_date_and_no_age_label():
    manager = _manager()
    await manager.store_memory("the garden soil test came back fine", importance=3.0)
    result = await retrieve_relevant_memories(manager, "garden soil", settings=CFG)
    memory = result.memories[0]
    # the stored date travels with the memory; the content is not prefixed
    # with a relative-age phrase
    assert memory["content"] == "the garden soil test came back fine"
    created = memory["created_at"]
    if isinstance(created, str):
        created = datetime.fromisoformat(created)
    assert created == T0
    assert not any("ago" in str(key) or "age" == key for key in memory)
