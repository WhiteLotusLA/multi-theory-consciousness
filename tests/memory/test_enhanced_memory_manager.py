"""Memory manager over seams: in-memory stores and fakes only, no database."""

import inspect
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from mtc.consciousness.sleep_scheduler import Consolidator
from mtc.core.circuit_breaker import CircuitBreaker
from mtc.core.config import Settings
from mtc.core.embeddings import EmbeddingUnavailableError, HashEmbedder
from mtc.memory import enhanced_memory_manager as manager_module
from mtc.memory.core_blocks import render_core_blocks
from mtc.memory.enhanced_memory_manager import EnhancedMemoryManager, StoreResult
from mtc.memory.episodic_store import InMemoryEpisodicStore, PostgresEpisodicStore
from mtc.memory.inmemory import InMemoryGraphStore, InMemoryVectorStore
from mtc.memory.protocols import (
    EpisodeConsolidator,
    StoreUnavailable,
    point_id_for_memory,
)
from mtc.memory.resilient_wrappers import ResilientPool

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class Recorder:
    """Shared call log so tests can check the order across stores."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Any]] = []


class RecordingEpisodic(InMemoryEpisodicStore):
    def __init__(self, recorder: Recorder, clock: Clock) -> None:
        super().__init__(clock=clock)
        self.recorder = recorder
        self.get_calls: List[List[int]] = []

    async def add_episodic(self, content, *args, **kwargs):
        memory_id = await super().add_episodic(content, *args, **kwargs)
        self.recorder.calls.append(("episodic", memory_id))
        return memory_id

    async def get_episodic(self, ids):
        self.get_calls.append(list(ids))
        return await super().get_episodic(ids)


class FlakyEpisodic(InMemoryEpisodicStore):
    """Raises StoreUnavailable on every call while ``down`` is set."""

    def __init__(self, clock: Clock) -> None:
        super().__init__(clock=clock)
        self.down = False
        self.closed = 0

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if (
            name in RAISING
            and super().__getattribute__("down")
            and inspect.iscoroutinefunction(attr)
        ):

            async def failing(*args: Any, **kwargs: Any) -> Any:
                raise StoreUnavailable("synthetic outage")

            return failing
        return attr


RAISING = {
    "ensure_schema",
    "load_core_blocks",
    "set_core_block",
    "seed_core_blocks",
    "add_episodic",
    "get_episodic",
    "search_text",
    "recent",
    "unconsolidated",
    "touch",
    "stats",
}


class BufferingEpisodic(InMemoryEpisodicStore):
    """Behaves like a store over a buffering pool: inserts return a marker."""

    MARKER = object()

    async def add_episodic(self, *args: Any, **kwargs: Any) -> Any:
        return self.MARKER


class RecordingVectors:
    def __init__(
        self,
        recorder: Optional[Recorder] = None,
        hits: Optional[List[Tuple[str, float, Dict[str, Any]]]] = None,
        store_error: Optional[BaseException] = None,
        search_error: Optional[BaseException] = None,
    ) -> None:
        self.recorder = recorder or Recorder()
        self.hits = hits or []
        self.store_error = store_error
        self.search_error = search_error
        self.points: Dict[str, Dict[str, Any]] = {}
        self.stored: List[Dict[str, Any]] = []

    def store_memory(
        self, text, metadata=None, importance=1.0, emotional_weight=0.0, point_id=None
    ):
        if self.store_error is not None:
            raise self.store_error
        self.recorder.calls.append(("vector", point_id))
        self.stored.append(
            {
                "text": text,
                "metadata": dict(metadata or {}),
                "importance": importance,
                "emotional_weight": emotional_weight,
                "point_id": point_id,
            }
        )
        self.points[point_id] = {"text": text, **(metadata or {})}
        return point_id

    def search_similar(self, query, top_k=None, min_score=0.7, filters=None):
        if self.search_error is not None:
            raise self.search_error
        return list(self.hits)

    def get_memory_by_id(self, point_id):
        return self.points.get(point_id)

    def delete_memories(self, point_ids):
        return sum(1 for p in point_ids if self.points.pop(p, None) is not None)

    def status(self):
        return {"backend": "fake", "available": True, "last_error": None}


class RecordingGraph(InMemoryGraphStore):
    def __init__(self, recorder: Recorder, fail: bool = False) -> None:
        super().__init__()
        self.recorder = recorder
        self.fail = fail
        self.closed = 0

    def link_memory_to_concepts(self, memory_id, memory_content, concept_names, **kw):
        if self.fail:
            raise RuntimeError("graph offline")
        self.recorder.calls.append(("graph", memory_id))
        return super().link_memory_to_concepts(
            memory_id, memory_content, concept_names, **kw
        )

    def close(self):
        self.closed += 1


class WordMatcher:
    """Whole-word matcher over the known concept names (relevance 0.8)."""

    def match_concepts(self, text, known_concepts, fuzzy=True):
        words = set(re.findall(r"[a-z]+", text.lower()))
        return [(name, 0.8) for name in known_concepts if name.lower() in words]


class FakeConsolidator:
    def __init__(self) -> None:
        self.episodes: List[Dict[str, Any]] = []
        self.patterns: List[Any] = []
        self.consolidate_calls: List[Dict[str, Any]] = []

    def add_episodic_memory(
        self,
        memory_type,
        emotional_state,
        subconscious_pattern=None,
        context=None,
        consciousness_level=None,
    ):
        episode = {
            "memory_type": memory_type,
            "emotional_state": emotional_state,
            "context": context,
        }
        self.episodes.append(episode)
        return episode

    def consolidate_memories(self, memories=None, sleep_duration="short"):
        self.consolidate_calls.append(
            {"memories": list(memories or []), "sleep_duration": sleep_duration}
        )
        self.patterns.append(
            SimpleNamespace(
                pattern_id=f"p{len(self.patterns)}",
                pattern_type="emotional_sequence",
                description="calm sequence",
                frequency=3,
                confidence=0.6,
                source_memories=["e0", "e1"],
                emotional_signature={"calm": 0.7},
            )
        )
        return {
            "cycles": 10,
            "avg_initial_anomaly": 0.6,
            "avg_final_anomaly": 0.2,
            "patterns_discovered": len(self.patterns),
        }

    def get_semantic_patterns(self, pattern_type=None, min_frequency=1):
        return [p for p in self.patterns if p.frequency >= min_frequency]


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        PROJECT_ROOT=str(tmp_path),
        SYSTEM_NAME="Unit",
        CREATOR_NAME="Alice",
        VECTOR_SEARCH_LIMIT=5,
    )


def _manager(settings, clock, **kwargs) -> EnhancedMemoryManager:
    kwargs.setdefault("episodic_store", InMemoryEpisodicStore(clock=clock))
    return EnhancedMemoryManager(settings=settings, clock=clock, **kwargs)


@pytest.mark.asyncio
async def test_initialize_seeds_neutral_blocks_when_empty(settings, clock):
    store = InMemoryEpisodicStore(clock=clock)
    manager = _manager(settings, clock, episodic_store=store)
    report = await manager.initialize()
    assert report["status"] == "ready"
    expected = render_core_blocks(settings=settings)
    assert await store.load_core_blocks() == expected
    assert report["seeded"] == {"identity": True, "persona": True, "human": True}
    assert await manager.get_core_memory("identity") == expected["identity"]


@pytest.mark.asyncio
async def test_initialize_seeds_from_persona_file(settings, clock, tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "persona.yaml").write_text(
        "name: Probe\ncreator:\n  name: Operator\n", encoding="utf-8"
    )
    manager = _manager(settings, clock)
    await manager.initialize()
    assert "Probe" in await manager.get_core_memory("identity")
    assert "Operator" in await manager.get_core_memory("human")


@pytest.mark.asyncio
async def test_initialize_keeps_existing_blocks(settings, clock):
    store = InMemoryEpisodicStore(clock=clock)
    await store.set_core_block("identity", "custom identity")
    manager = _manager(settings, clock, episodic_store=store)
    report = await manager.initialize()
    assert report["seeded"] == {"identity": False, "persona": True, "human": True}
    assert await manager.get_core_memory("identity") == "custom identity"


@pytest.mark.asyncio
async def test_seed_with_overwrite_replaces(settings, clock):
    store = InMemoryEpisodicStore(clock=clock)
    await store.set_core_block("identity", "old")
    manager = _manager(settings, clock, episodic_store=store)
    await manager.initialize()
    written = await manager.seed_core_memory({"identity": "new"}, overwrite=True)
    assert written == {"identity": True}
    assert await manager.get_core_memory("identity") == "new"
    written = await manager.seed_core_memory(overwrite=False)
    assert written == {"identity": False, "persona": False, "human": False}


@pytest.mark.asyncio
async def test_update_core_memory_writes_through(settings, clock):
    store = InMemoryEpisodicStore(clock=clock)
    manager = _manager(settings, clock, episodic_store=store)
    await manager.initialize()
    await manager.update_core_memory("persona", "measured and curious")
    assert (await store.load_core_blocks())["persona"] == "measured and curious"
    assert await manager.get_core_memory("persona") == "measured and curious"


@pytest.mark.asyncio
async def test_store_writes_episodic_then_vector_then_graph(settings, clock):
    recorder = Recorder()
    store = RecordingEpisodic(recorder, clock)
    graph = RecordingGraph(recorder)
    graph.merge_concept("garden")
    vectors = RecordingVectors(recorder)
    manager = _manager(
        settings,
        clock,
        episodic_store=store,
        vector_store=vectors,
        graph=graph,
        concept_matcher=WordMatcher(),
    )
    await manager.initialize()
    result = await manager.store_memory(
        "Alice planted a garden", importance=8.0, emotional_context={"joy": 0.6}
    )
    assert isinstance(result, StoreResult)
    assert [name for name, _ in recorder.calls] == ["episodic", "vector", "graph"]
    assert result.postgres_id == recorder.calls[0][1]
    assert result.concepts_linked == 1
    assert result.buffered is False and result.errors == {}
    stored = vectors.stored[0]
    assert stored["metadata"]["postgres_id"] == result.postgres_id
    assert stored["metadata"]["memory_type"] == "episodic"
    assert stored["metadata"]["created_at"] == T0.isoformat()
    assert stored["emotional_weight"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_vector_weight_is_the_measured_emotion_not_the_importance(
    settings, clock
):
    vectors = RecordingVectors()
    manager = _manager(settings, clock, vector_store=vectors)
    await manager.initialize()
    await manager.store_memory("a plain note", importance=3.0)
    await manager.store_memory(
        "a bright moment", importance=3.0, emotional_context={"joy": 0.8}
    )
    await manager.store_memory(
        "two feelings",
        importance=9.0,
        emotional_context={"calm": 0.2, "awe": 0.5, "label": "text", "flag": True},
    )
    await manager.store_memory("over the scale", emotional_context={"joy": 1.7})
    weights = [entry["emotional_weight"] for entry in vectors.stored]
    assert weights[0] is None
    assert weights[1] == pytest.approx(0.8)
    assert weights[2] == pytest.approx(0.5)
    assert weights[3] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_non_finite_emotion_is_not_a_measurement(settings, clock):
    vectors = RecordingVectors()
    store = InMemoryEpisodicStore(clock=clock)
    manager = _manager(settings, clock, episodic_store=store, vector_store=vectors)
    await manager.initialize()
    await manager.store_memory("nan alone", emotional_context={"joy": float("nan")})
    await manager.store_memory(
        "nan beside a number",
        emotional_context={"joy": float("nan"), "awe": 0.3},
    )
    await manager.store_memory(
        "infinite beside a number",
        emotional_context={"joy": float("inf"), "awe": 0.4},
    )
    weights = [entry["emotional_weight"] for entry in vectors.stored]
    # A non-finite value is unmeasurable: it never reads as the full weight.
    assert weights[0] is None
    assert weights[1] == pytest.approx(0.3)
    assert weights[2] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_simple_consolidation_mean_skips_non_finite_values(settings, clock):
    store = InMemoryEpisodicStore(clock=clock)
    manager = _manager(settings, clock, episodic_store=store)
    await manager.initialize()
    await manager.store_memory("joy one", emotional_context={"joy": 0.4})
    await manager.store_memory("joy two", emotional_context={"joy": 0.8})
    await manager.store_memory(
        "joy unknown", emotional_context={"joy": float("nan"), "calm": 0.1}
    )
    report = await manager.consolidate_memories()
    assert report["method"] == "simple"
    semantics = await store.get_semantic([1, 2])
    joy = [row for row in semantics if row["concept"] == "joy_pattern"]
    assert len(joy) == 1
    assert joy[0]["emotional_context"] == {"joy": pytest.approx(0.6)}
    assert joy[0]["source_memories"] == [1, 2]


@pytest.mark.asyncio
async def test_vector_point_id_derives_from_postgres_id(settings, clock):
    vectors = RecordingVectors()
    manager = _manager(settings, clock, vector_store=vectors)
    await manager.initialize()
    result = await manager.store_memory("a quiet morning")
    assert result.point_id == point_id_for_memory(result.postgres_id)
    assert vectors.stored[0]["point_id"] == result.point_id


@pytest.mark.asyncio
async def test_buffered_store_skips_indexing_and_counts_it(settings, clock):
    vectors = RecordingVectors()
    manager = _manager(
        settings,
        clock,
        episodic_store=BufferingEpisodic(clock=clock),
        vector_store=vectors,
    )
    await manager.initialize()
    result = await manager.store_memory("written during an outage")
    assert result.buffered is True
    assert result.postgres_id is None and result.point_id is None
    assert vectors.stored == []
    assert manager.status()["counts"]["unindexed"] == 1


@pytest.mark.asyncio
async def test_unavailable_embedder_stores_nothing_and_says_so(settings, clock):
    store = InMemoryEpisodicStore(clock=clock)
    vectors = RecordingVectors(
        store_error=EmbeddingUnavailableError("model could not be loaded")
    )
    manager = _manager(settings, clock, episodic_store=store, vector_store=vectors)
    await manager.initialize()
    result = await manager.store_memory("the lake at dusk")
    assert result.postgres_id is not None
    assert len(await store.get_episodic([result.postgres_id])) == 1
    assert result.point_id is None
    assert vectors.points == {}
    assert "EmbeddingUnavailableError" in result.errors["vector"]
    status = manager.status()
    assert status["counts"]["vector_failures"] == 1
    assert status["counts"]["unindexed"] == 1
    assert "model could not be loaded" in status["errors"]["vector"]


@pytest.mark.asyncio
async def test_non_semantic_embedder_needs_opt_in_to_index(settings, clock):
    with pytest.raises(EmbeddingUnavailableError):
        InMemoryVectorStore(HashEmbedder(dim=16))
    vectors = InMemoryVectorStore(HashEmbedder(dim=16), allow_non_semantic=True)
    manager = _manager(settings, clock, vector_store=vectors)
    await manager.initialize()
    result = await manager.store_memory("indexed with an explicit opt-in")
    assert vectors.get_memory_by_id(result.point_id)["postgres_id"] == (
        result.postgres_id
    )


@pytest.mark.asyncio
async def test_graph_failure_keeps_episodic_and_vector(settings, clock):
    store = InMemoryEpisodicStore(clock=clock)
    vectors = RecordingVectors()
    graph = RecordingGraph(Recorder(), fail=True)
    graph.merge_concept("lake")
    manager = _manager(
        settings,
        clock,
        episodic_store=store,
        vector_store=vectors,
        graph=graph,
        concept_matcher=WordMatcher(),
    )
    await manager.initialize()
    result = await manager.store_memory("the lake was still")
    assert result.postgres_id is not None and result.point_id is not None
    assert result.concepts_linked == 0
    assert "graph offline" in result.errors["graph"]
    assert len(await store.get_episodic([result.postgres_id])) == 1
    assert manager.status()["counts"]["graph_failures"] == 1


@pytest.mark.asyncio
async def test_recall_merges_vector_hits_with_rows(settings, clock):
    store = RecordingEpisodic(Recorder(), clock)
    manager = _manager(settings, clock, episodic_store=store)
    await manager.initialize()
    first = (await manager.store_memory("river crossing")).postgres_id
    second = (await manager.store_memory("mountain pass")).postgres_id
    archived = (await manager.store_memory("old note")).postgres_id
    await store.archive([archived])
    vectors = RecordingVectors(
        hits=[
            (point_id_for_memory(second), 0.91, {"postgres_id": second}),
            (point_id_for_memory(first), 0.74, {"postgres_id": first}),
            (point_id_for_memory(archived), 0.70, {"postgres_id": archived}),
            ("gist-1", 0.66, {"type": "gist", "text": "a gist summary"}),
        ]
    )
    manager.vector_store = vectors
    store.get_calls.clear()
    recall = await manager.semantic_recall("crossing", min_score=0.5)
    assert recall["search_stats"]["method"] == "vector"
    # One batched read, not one read per hit.
    assert store.get_calls == [[second, first, archived]]
    memories = recall["memories"]
    assert [m["id"] for m in memories] == [second, first, "gist-1"]
    assert memories[0]["content"] == "mountain pass"
    assert memories[0]["similarity_score"] == pytest.approx(0.91)
    assert memories[0]["created_at"] == T0.isoformat()
    assert memories[2]["source"] == "vector_payload"
    assert recall["search_stats"]["orphan_hits"] == 1
    rows = await store.get_episodic([first, second])
    assert all(row["accessed_count"] == 1 for row in rows)


@pytest.mark.asyncio
async def test_recall_falls_back_to_text_search_and_labels_it(settings, clock):
    manager = _manager(
        settings,
        clock,
        vector_store=RecordingVectors(search_error=RuntimeError("qdrant down")),
    )
    await manager.initialize()
    await manager.store_memory("the harbor lights")
    await manager.store_memory("a field of wheat")
    recall = await manager.semantic_recall("harbor")
    stats = recall["search_stats"]
    assert stats["method"] == "text"
    assert "qdrant down" in stats["vector_error"]
    assert [m["content"] for m in recall["memories"]] == ["the harbor lights"]
    assert recall["memories"][0]["similarity_score"] is None
    assert recall["memories"][0]["match"] == "text"

    absent = _manager(settings, clock)
    await absent.initialize()
    await absent.store_memory("the harbor at noon")
    recall = await absent.semantic_recall("harbor")
    assert recall["search_stats"]["method"] == "text"
    assert recall["search_stats"]["vector_error"] == "vector store absent"


@pytest.mark.asyncio
async def test_recall_reports_none_when_every_path_fails(settings, clock):
    store = FlakyEpisodic(clock)
    manager = _manager(settings, clock, episodic_store=store)
    await manager.initialize()
    store.down = True
    recall = await manager.semantic_recall("anything")
    assert recall["memories"] == []
    assert recall["search_stats"]["method"] == "none"
    assert "synthetic outage" in recall["search_stats"]["text_error"]


@pytest.mark.asyncio
async def test_recall_concepts_from_graph(settings, clock):
    graph = InMemoryGraphStore()
    for name in ("garden", "soil", "seeds"):
        graph.merge_concept(name)
    graph.relate_concepts("garden", "soil", strength=0.9)
    graph.relate_concepts("garden", "seeds", strength=0.4)
    manager = _manager(settings, clock, graph=graph, concept_matcher=WordMatcher())
    await manager.initialize()
    recall = await manager.semantic_recall("tell me about the garden")
    concepts = recall["concepts"]
    assert len(concepts) == 1
    assert concepts[0]["concept"] == "garden"
    assert [r["concept"] for r in concepts[0]["related"]] == ["soil", "seeds"]
    no_concepts = await manager.semantic_recall("garden", include_concepts=False)
    assert no_concepts["concepts"] == []


@pytest.mark.asyncio
async def test_graph_without_matcher_is_named_in_status(settings, clock):
    manager = _manager(settings, clock, graph=InMemoryGraphStore())
    await manager.initialize()
    result = await manager.store_memory("text")
    assert result.concepts_linked == 0
    parts = manager.status()["parts"]
    assert parts["concept_matcher"]["available"] is False
    assert parts["vector"]["available"] is False


@pytest.mark.asyncio
async def test_recent_important_has_recency_anchor(settings, clock):
    manager = _manager(settings, clock)
    await manager.initialize()
    clock.now = T0
    await manager.store_memory("older important", importance=9.0)
    clock.now = T0 + timedelta(hours=1)
    await manager.store_memory("minor detail", importance=2.0)
    clock.now = T0 + timedelta(hours=2)
    await manager.store_memory("newest important", importance=7.0)
    clock.now = T0 + timedelta(hours=3)
    found = await manager.get_recent_important_memories(hours=24, min_importance=5.0)
    assert [m["content"] for m in found] == ["newest important", "older important"]
    assert all(m["recency_anchored"] is True for m in found)
    assert all(m["similarity_score"] is None for m in found)
    assert found[0]["created_at"] == (T0 + timedelta(hours=2)).isoformat()
    assert len(await manager.get_recent_important_memories(limit=1)) == 1


@pytest.mark.asyncio
async def test_lifecycle_boilerplate_filtered(settings, clock):
    manager = _manager(settings, clock)
    await manager.initialize()
    await manager.store_memory("a real memory about tides", importance=6.0)
    await manager.store_memory(
        "Conversation with Alice ended. 12 messages exchanged.", importance=6.0
    )
    recent = await manager.get_recent_important_memories()
    assert [m["content"] for m in recent] == ["a real memory about tides"]
    assert [m["content"] for m in await manager.get_recent_memories()] == [
        "a real memory about tides"
    ]

    custom = _manager(settings, clock, boilerplate_patterns=[r"^session closed"])
    await custom.initialize()
    await custom.store_memory("session closed by the operator", importance=6.0)
    await custom.store_memory(
        "Conversation with Alice ended. 3 messages exchanged.", importance=6.0
    )
    assert [m["content"] for m in await custom.get_recent_important_memories()] == [
        "Conversation with Alice ended. 3 messages exchanged."
    ]


@pytest.mark.asyncio
async def test_search_memories_shapes_rows_and_touches(settings, clock):
    store = InMemoryEpisodicStore(clock=clock)
    manager = _manager(settings, clock, episodic_store=store)
    await manager.initialize()
    memory_id = (
        await manager.store_memory("violin practice", importance=6.0)
    ).postgres_id
    found = await manager.search_memories("violin")
    assert found[0]["id"] == memory_id
    assert found[0]["similarity"] > 0
    assert found[0]["created_at"] == T0.isoformat()
    assert (await store.get_episodic([memory_id]))[0]["accessed_count"] == 1


@pytest.mark.asyncio
async def test_consolidate_uses_injected_consolidator(settings, clock):
    store = InMemoryEpisodicStore(clock=clock)
    consolidator = FakeConsolidator()
    assert isinstance(consolidator, EpisodeConsolidator)
    manager = _manager(settings, clock, episodic_store=store, consolidator=consolidator)
    await manager.initialize()
    ids = [
        (
            await manager.store_memory(f"memory {i}", emotional_context={"calm": 0.5})
        ).postgres_id
        for i in range(3)
    ]
    report = await manager.consolidate_memories()
    assert report["method"] == "htm"
    assert report["memories_analyzed"] == 3
    assert report["episodic_processed"] == 3
    assert report["semantic_created"] == 1
    assert len(consolidator.episodes) == 3
    assert consolidator.episodes[0]["context"] == "memory 0"
    assert len(consolidator.consolidate_calls[0]["memories"]) == 3
    semantic = await store.get_semantic([1])
    assert semantic[0]["concept"] == "emotional_sequence"
    assert semantic[0]["source_memories"] == ids
    assert semantic[0]["knowledge"] == "calm sequence"
    rows = await store.get_episodic(ids)
    assert all(row["consolidated"] for row in rows)
    assert rows[0]["consolidation_strength"] == pytest.approx(0.8)
    # A second pass finds nothing new and writes no duplicate pattern.
    again = await manager.consolidate_memories()
    assert again["memories_analyzed"] == 0 and again["semantic_created"] == 0


@pytest.mark.asyncio
async def test_consolidate_simple_path_is_labelled(settings, clock):
    store = InMemoryEpisodicStore(clock=clock)
    manager = _manager(settings, clock, episodic_store=store)
    await manager.initialize()
    await manager.store_memory("calm one", emotional_context={"calm": 0.9, "joy": 0.1})
    await manager.store_memory("calm two", emotional_context={"calm": 0.7})
    await manager.store_memory("joy one", emotional_context={"joy": 0.8})
    await manager.store_memory("plain")
    report = await manager.consolidate_memories()
    assert report["method"] == "simple"
    assert report["memories_analyzed"] == 4
    assert report["episodic_processed"] == 4
    assert report["patterns_found"] == 1
    assert report["semantic_created"] == 1
    semantic = await store.get_semantic([1])
    assert semantic[0]["concept"] == "calm_pattern"
    assert semantic[0]["source_memories"] == [1, 2]
    assert (await store.stats())["unconsolidated"] == 0


@pytest.mark.asyncio
async def test_simple_consolidation_writes_no_constant(settings, clock):
    store = InMemoryEpisodicStore(clock=clock)
    manager = _manager(settings, clock, episodic_store=store)
    await manager.initialize()
    first = await manager.store_memory("joy one", emotional_context={"joy": 0.4})
    second = await manager.store_memory(
        "joy two", emotional_context={"joy": 0.8, "calm": 0.1, "note": "text"}
    )
    report = await manager.consolidate_memories()
    assert report["method"] == "simple"
    assert "consolidation_strength" in report
    assert report["consolidation_strength"] is None
    semantic = (await store.get_semantic([1]))[0]
    assert semantic["concept"] == "joy_pattern"
    assert semantic["emotional_context"] == {"joy": pytest.approx(0.6)}
    assert semantic["confidence"] is None
    # The count-based strength of the semantic row stays (two members).
    assert semantic["strength"] == pytest.approx(0.2)
    rows = await store.get_episodic([first.postgres_id, second.postgres_id])
    assert [row["consolidated"] for row in rows] == [True, True]
    assert [row["consolidation_strength"] for row in rows] == [None, None]
    assert not hasattr(manager_module, "SIMPLE_STRENGTH")
    assert not hasattr(manager_module, "SIMPLE_CONFIDENCE")


@pytest.mark.asyncio
async def test_simple_path_invents_no_emotion_for_unlabelled_rows(settings, clock):
    store = InMemoryEpisodicStore(clock=clock)
    manager = _manager(settings, clock, episodic_store=store)
    await manager.initialize()
    await manager.store_memory("plain one")
    await manager.store_memory("plain two", emotional_context={"note": "text"})
    report = await manager.consolidate_memories()
    assert report["semantic_created"] == 1
    semantic = (await store.get_semantic([1]))[0]
    assert semantic["concept"] == "unlabelled_pattern"
    assert semantic["emotional_context"] == {}
    assert "neutral" not in str(semantic)


@pytest.mark.asyncio
async def test_unmeasured_htm_strength_is_not_written(settings, clock):
    class SilentConsolidator(FakeConsolidator):
        def consolidate_memories(self, memories=None, sleep_duration="short"):
            report = super().consolidate_memories(memories, sleep_duration)
            report.pop("avg_final_anomaly")
            return report

    store = InMemoryEpisodicStore(clock=clock)
    manager = _manager(
        settings, clock, episodic_store=store, consolidator=SilentConsolidator()
    )
    await manager.initialize()
    memory_id = (await manager.store_memory("one")).postgres_id
    await store.mark_consolidated([memory_id], 0.4)
    store._rows[memory_id]["consolidated"] = False
    report = await manager.consolidate_memories()
    assert report["consolidation_strength"] is None
    assert report["episodic_processed"] == 1
    row = (await store.get_episodic([memory_id]))[0]
    assert row["consolidated"] is True
    assert row["consolidation_strength"] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_consolidate_seam_returns_counts(settings, clock):
    manager = _manager(settings, clock)
    assert isinstance(manager, Consolidator)
    await manager.initialize()
    await manager.store_memory("one", emotional_context={"calm": 0.5})
    await manager.store_memory("two", emotional_context={"calm": 0.6})
    counts = await manager.consolidate()
    assert counts["memories_analyzed"] == 2
    assert counts["semantic_created"] == 1
    assert all(isinstance(value, int) for value in counts.values())


@pytest.mark.asyncio
async def test_unavailable_episodic_store_degrades_visibly(settings, clock):
    store = FlakyEpisodic(clock)
    manager = _manager(settings, clock, episodic_store=store)
    first = await manager.initialize()
    assert first["status"] == "ready"
    loaded = await manager.get_core_memory("identity")
    store.down = True
    report = await manager.initialize()
    assert report["status"] == "degraded"
    assert "synthetic outage" in report["errors"]["episodic"]
    assert await manager.get_core_memory("identity") == loaded
    status = manager.status()
    assert status["degraded"] is True
    assert status["core_memory"]["stale"] is True
    result = await manager.store_memory("written while the store is down")
    assert result.postgres_id is None and result.buffered is False
    assert "synthetic outage" in result.errors["episodic"]
    assert manager.status()["counts"]["episodic_failures"] == 1
    with pytest.raises(StoreUnavailable):
        await manager.update_core_memory("identity", "changed")
    assert await manager.get_core_memory("identity") == loaded
    store.down = False
    assert (await manager.initialize())["status"] == "ready"
    assert manager.status()["core_memory"]["stale"] is False


@pytest.mark.asyncio
async def test_degraded_start_recovers_after_backoff(settings, clock):
    store = FlakyEpisodic(clock)
    store.down = True
    manager = _manager(settings, clock, episodic_store=store)
    assert (await manager.initialize())["status"] == "degraded"
    rendered = await manager.get_core_memory("identity")
    store.down = False
    await store.set_core_block("identity", "an operator-edited identity block")
    # Inside the backoff the manager keeps serving the stale rendered block.
    assert await manager.get_core_memory("identity") == rendered
    assert manager.status()["degraded"] is True
    clock.now = T0 + timedelta(seconds=manager_module.RECOVERY_BACKOFF_SECONDS)
    assert (
        await manager.get_core_memory("identity") == "an operator-edited identity block"
    )
    status = manager.status()
    assert status["degraded"] is False and status["available"] is True
    assert status["last_error"] is None
    assert status["core_memory"] == {
        "blocks": ["human", "identity", "persona"],
        "stale": False,
        "source": "store",
    }


@pytest.mark.asyncio
async def test_failed_retry_backs_off_further(settings, clock):
    store = FlakyEpisodic(clock)
    store.down = True
    manager = _manager(settings, clock, episodic_store=store)
    await manager.initialize()
    base = manager_module.RECOVERY_BACKOFF_SECONDS
    clock.now = T0 + timedelta(seconds=base)
    await manager.get_core_memory("identity")
    assert manager.status()["degraded"] is True
    store.down = False
    # The second wait is twice the first.
    clock.now = T0 + timedelta(seconds=base * 2.5)
    await manager.get_core_memory("identity")
    assert manager.status()["degraded"] is True
    clock.now = T0 + timedelta(seconds=base * 3)
    await manager.get_core_memory("identity")
    assert manager.status()["degraded"] is False


@pytest.mark.asyncio
async def test_successful_write_while_degraded_reloads_core_memory(settings, clock):
    store = FlakyEpisodic(clock)
    store.down = True
    manager = _manager(settings, clock, episodic_store=store)
    await manager.initialize()
    store.down = False
    await store.set_core_block("identity", "stored identity")
    result = await manager.store_memory("the store is back")
    assert isinstance(result.postgres_id, int) and not result.errors
    assert await manager.get_core_memory("identity") == "stored identity"
    assert manager.status()["degraded"] is False
    assert manager.status()["core_memory"]["source"] == "store"


@pytest.mark.asyncio
async def test_first_initialize_without_store_serves_rendered_blocks(settings, clock):
    store = FlakyEpisodic(clock)
    store.down = True
    manager = _manager(settings, clock, episodic_store=store)
    report = await manager.initialize()
    assert report["status"] == "degraded"
    expected = render_core_blocks(settings=settings)
    assert await manager.get_core_memory("identity") == expected["identity"]
    core = manager.status()["core_memory"]
    assert core["stale"] is True and core["source"] == "rendered"


@pytest.mark.asyncio
async def test_memory_stats(settings, clock):
    manager = _manager(settings, clock)
    await manager.initialize()
    await manager.store_memory("one", importance=4.0)
    await manager.store_memory("two", importance=6.0)
    stats = await manager.get_memory_stats()
    assert stats["core_memory_blocks"] == 3
    assert stats["episodic_memories"] == 2
    assert stats["avg_importance"] == pytest.approx(5.0)
    assert stats["unindexed_memories"] == 0


@pytest.mark.asyncio
async def test_shutdown_awaitable_safe(settings, clock):
    class AsyncClose:
        def __init__(self) -> None:
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1

    class SyncClose:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    store = FlakyEpisodic(clock)
    graph = RecordingGraph(Recorder())
    vectors = SyncClose()
    manager = _manager(settings, clock, episodic_store=store, graph=graph)
    manager.vector_store = vectors
    await manager.initialize()
    await manager.shutdown()
    assert graph.closed == 1 and vectors.closed == 1
    assert manager.status()["initialized"] is False

    awaited = AsyncClose()
    other = _manager(settings, clock)
    other.graph = awaited
    await other.shutdown()
    assert awaited.closed == 1


def test_no_random_vector_anywhere():
    source = Path(manager_module.__file__).read_text(encoding="utf-8")
    assert "np.random" not in source
    assert "random." not in source
    assert "import random" not in source
    assert "hashlib" not in source


class _FakeConnection:
    def __init__(self, pool: "FakePool") -> None:
        self.pool = pool

    def transaction(self) -> "_FakeContext":
        return _FakeContext(None)

    async def execute(self, query: str, *args: Any) -> str:
        self.pool.calls.append(("execute", query, args))
        return "CREATE"


class _FakeContext:
    def __init__(self, value: Any) -> None:
        self.value = value

    async def __aenter__(self) -> Any:
        return self.value

    async def __aexit__(self, *exc: Any) -> None:
        return None


class FakePool:
    """An asyncpg-shaped pool that records statements; no database."""

    def __init__(self, next_id: int = 42) -> None:
        self.next_id = next_id
        self.calls: List[Tuple[str, str, Tuple[Any, ...]]] = []

    def acquire(self) -> _FakeContext:
        return _FakeContext(_FakeConnection(self))

    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append(("execute", query, args))
        return "INSERT 0 1"

    async def fetchval(self, query: str, *args: Any) -> int:
        self.calls.append(("fetchval", query, args))
        return self.next_id

    async def fetch(self, query: str, *args: Any) -> List[Any]:
        self.calls.append(("fetch", query, args))
        return []

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_store_memory_returns_id_through_a_live_pool(settings, clock):
    raw = FakePool(next_id=42)
    pool = ResilientPool(raw, breaker=CircuitBreaker(name="pg-live"))
    manager = _manager(settings, clock, episodic_store=PostgresEpisodicStore(pool))
    await manager.initialize()
    result = await manager.store_memory("a real memory", memory_type="conversation")
    assert result.postgres_id == 42 and result.buffered is False
    inserts = [c for c in raw.calls if "INSERT INTO episodic_memory" in c[1]]
    assert len(inserts) == 1 and inserts[0][0] == "fetchval"
    assert inserts[0][2][0] == "a real memory"


@pytest.mark.asyncio
async def test_writes_through_an_open_breaker_are_buffered_not_dropped(settings, clock):
    raw = FakePool()
    breaker = CircuitBreaker(name="pg-open", start_open=True, recovery_timeout=3600)
    pool = ResilientPool(raw, breaker=breaker)
    manager = _manager(settings, clock, episodic_store=PostgresEpisodicStore(pool))
    report = await manager.initialize()
    assert report["status"] == "degraded"
    result = await manager.store_memory("a memory during an outage")
    assert result.buffered is True and result.postgres_id is None
    assert pool.status()["buffered"] == 1
    # Identity writes ride the same buffered path instead of raising.
    await manager.update_core_memory("identity", "Unit identity text")
    assert pool.status()["buffered"] == 2
    assert await manager.get_core_memory("identity") == "Unit identity text"
    assert raw.calls == []
