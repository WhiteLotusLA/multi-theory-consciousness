"""The memory factory: one try per store, honest component status."""

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Sequence

import numpy as np
import pytest

from mtc.core.config import Settings
from mtc.knowledge.concept_extraction import ConceptExtractor
from mtc.memory.episodic_store import InMemoryEpisodicStore, PostgresEpisodicStore
from mtc.memory.factory import MemoryStack, build_memory_manager
from mtc.memory.inmemory import InMemoryGraphStore, InMemoryVectorStore
from mtc.memory.memory_pruner import MemoryPruner
from mtc.memory.protocols import StoreUnavailable, point_id_for_memory
from mtc.memory.resilient_wrappers import ResilientPool

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
VOCABULARY = ("river", "bridge", "garden", "stone", "lantern", "budget")


class Clock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class LookupEmbedder:
    """Each vocabulary word owns one axis (a semantic-flagged test stand-in)."""

    semantic = True

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for row, text in enumerate(texts):
            for word in text.lower().split():
                word = word.strip(".,;")
                axis = VOCABULARY.index(word) if word in VOCABULARY else self.dim - 1
                out[row, axis] += 1.0
            out[row] /= np.linalg.norm(out[row]) or 1.0
        return out


class FakeConnection:
    def __init__(self, log: List[str]) -> None:
        self.log = log

    def transaction(self):
        connection = self

        class Transaction:
            async def __aenter__(self):
                connection.log.append("begin")

            async def __aexit__(self, *exc):
                connection.log.append("commit")
                return False

        return Transaction()

    async def execute(self, statement, *args):
        self.log.append("ddl")
        return "OK"


class FakePool:
    """Answers like an empty asyncpg pool; inserts return increasing ids."""

    def __init__(self) -> None:
        self.log: List[str] = []
        self.next_id = 0
        self.closed = False

    async def execute(self, query, *args):
        self.log.append("execute")
        return "INSERT 0 1"

    async def fetch(self, query, *args):
        self.log.append("fetch")
        return []

    async def fetchrow(self, query, *args):
        return None

    async def fetchval(self, query, *args):
        self.log.append("fetchval")
        self.next_id += 1
        return self.next_id

    def acquire(self):
        pool = self

        class Acquired:
            async def __aenter__(self):
                return FakeConnection(pool.log)

            async def __aexit__(self, *exc):
                return False

        return Acquired()

    async def close(self):
        self.closed = True


@pytest.fixture
def settings(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return Settings()


def _pool_builder(pool: FakePool):
    calls: List[Any] = []

    async def build(active):
        calls.append(active)
        return pool

    build.calls = calls
    return build


def _raises(message: str):
    calls: List[Any] = []

    def build(*args):
        calls.append(args)
        raise StoreUnavailable(message)

    build.calls = calls
    return build


def _recording(store):
    calls: List[Any] = []

    def build(*args):
        calls.append(args)
        return store

    build.calls = calls
    return build


@pytest.mark.asyncio
async def test_partial_infrastructure_keeps_postgres(settings):
    pool = FakePool()
    vectors = _raises("qdrant refused the connection")
    graph = _raises("neo4j refused the connection")
    stack = await build_memory_manager(
        settings,
        graph=True,
        pool_builder=_pool_builder(pool),
        vector_builder=vectors,
        graph_builder=graph,
    )
    assert isinstance(stack, MemoryStack)
    assert isinstance(stack.pool, ResilientPool)
    assert isinstance(stack.manager.episodic, PostgresEpisodicStore)
    assert stack.vector_store is None and stack.graph is None
    assert stack.manager.vector_store is None and stack.manager.graph is None
    assert vectors.calls and graph.calls  # both were tried, each in its own try
    components = stack.components
    assert components["postgres"]["available"] is True
    assert components["vectors"]["available"] is False
    assert "qdrant refused" in components["vectors"]["last_error"]
    assert components["graph"]["available"] is False
    assert "neo4j refused" in components["graph"]["last_error"]

    result = await stack.manager.store_memory("a stone bridge over the river")
    assert result.postgres_id == 1 and result.errors == {}
    status = stack.manager.status()
    assert status["backend"] == "postgres" and status["available"] is True
    assert status["parts"]["vector"]["reason"] == "absent"
    assert status["parts"]["graph"]["reason"] == "absent"
    await stack.close()
    assert pool.closed is True


@pytest.mark.asyncio
async def test_missing_postgres_client_names_the_extra(settings, monkeypatch):
    monkeypatch.setitem(sys.modules, "asyncpg", None)
    vectors = _recording(None)
    with pytest.raises(ImportError) as info:
        await build_memory_manager(settings, vector_builder=vectors)
    assert "[postgres]" in str(info.value) and "'postgres'" in str(info.value)
    assert vectors.calls == []  # nothing else is built once the primary store fails


@pytest.mark.asyncio
async def test_unreachable_postgres_raises_store_unavailable(settings):
    async def down(active):
        raise StoreUnavailable("PostgreSQL did not accept a connection")

    with pytest.raises(StoreUnavailable):
        await build_memory_manager(settings, pool_builder=down, vectors=False)


@pytest.mark.asyncio
async def test_components_report_status(settings):
    vector_store = InMemoryVectorStore(LookupEmbedder())
    graph_store = InMemoryGraphStore()

    class Consolidator:
        def add_episodic_memory(self, *args, **kwargs):
            return None

        def consolidate_memories(self, memories=None, sleep_duration="short"):
            return {}

        def get_semantic_patterns(self, pattern_type=None, min_frequency=1):
            return []

    stack = await build_memory_manager(
        settings,
        graph=True,
        consolidator=Consolidator(),
        pool_builder=_pool_builder(FakePool()),
        vector_builder=_recording(vector_store),
        graph_builder=_recording(graph_store),
    )
    expected = {
        "postgres",
        "episodic",
        "vectors",
        "graph",
        "concept_matcher",
        "consolidator",
    }
    assert set(stack.components) == expected
    for name, report in stack.components.items():
        assert {"backend", "available", "last_error"} <= set(report), name
        assert report["available"] is True, name
    assert stack.components["vectors"]["backend"] == "memory"
    assert stack.components["concept_matcher"]["backend"] == "ConceptExtractor"
    assert isinstance(stack.manager.concept_matcher, ConceptExtractor)
    assert stack.vector_store is vector_store and stack.graph is graph_store


@pytest.mark.asyncio
async def test_graph_off_by_default(settings):
    graph = _recording(InMemoryGraphStore())
    vectors = _recording(InMemoryVectorStore(LookupEmbedder()))
    stack = await build_memory_manager(
        settings,
        postgres=False,
        vectors=False,
        vector_builder=vectors,
        graph_builder=graph,
    )
    assert graph.calls == [] and vectors.calls == []  # skipped stores are never built
    assert stack.graph is None and stack.vector_store is None and stack.pool is None
    assert isinstance(stack.manager.episodic, InMemoryEpisodicStore)
    for name in ("postgres", "vectors", "graph"):
        assert stack.components[name]["skipped"] is True, name
        assert stack.components[name]["available"] is False, name
    assert stack.components["episodic"]["backend"] == "memory"
    assert stack.manager.concept_matcher is None


@pytest.mark.asyncio
async def test_embedder_reaches_the_vector_builder(settings):
    seen: Dict[str, Any] = {}
    embedder = LookupEmbedder()

    def build(active, chosen):
        seen.update(settings=active, embedder=chosen)
        return InMemoryVectorStore(chosen)

    await build_memory_manager(
        settings, postgres=False, vector_builder=build, embedder=embedder
    )
    assert seen == {"settings": settings, "embedder": embedder}


@pytest.mark.asyncio
async def test_memory_manager_end_to_end_local(settings):
    qdrant_client = pytest.importorskip("qdrant_client")
    from mtc.neural.vectors.coordinator import VectorCoordinator

    clock = Clock()
    graph_store = InMemoryGraphStore()
    for name in ("river", "bridge", "lantern"):
        graph_store.merge_concept(name)

    def vectors(active, embedder):
        return VectorCoordinator(
            collection_name="e2e_memories",
            embedder=embedder,
            client=qdrant_client.QdrantClient(":memory:"),
            settings=active,
        )

    stack = await build_memory_manager(
        settings,
        postgres=False,
        graph=True,
        embedder=LookupEmbedder(),
        vector_builder=vectors,
        graph_builder=lambda active: graph_store,
        clock=clock,
    )
    manager = stack.manager
    stored = await manager.store_memory("a stone bridge over the river", importance=0.5)
    assert stored.errors == {} and isinstance(stored.postgres_id, int)
    assert stored.point_id == point_id_for_memory(stored.postgres_id)
    assert stored.concepts_linked == 2
    other = await manager.store_memory("the budget meeting", importance=9.0)

    recall = await manager.semantic_recall("river bridge", min_score=0.3)
    assert recall["search_stats"]["method"] == "vector"
    assert [m["id"] for m in recall["memories"]] == [stored.postgres_id]

    clock.now = T0 + timedelta(days=400)
    pruner = MemoryPruner(
        episodic=manager.episodic,
        vectors=stack.vector_store,
        graph=stack.graph,
        clock=clock,
    )
    result = await pruner.prune(dry_run=False)
    assert result["episodic_deleted"] == 1
    assert result["vectors_deleted"] == 1 and result["graph_nodes_deleted"] == 1
    assert stack.vector_store.get_memory_by_id(stored.point_id) is None
    assert stack.vector_store.get_memory_by_id(other.point_id) is not None
    await stack.close()


@pytest.mark.db
@pytest.mark.asyncio
async def test_memory_manager_end_to_end_stores(
    settings, throwaway_postgres, throwaway_qdrant, throwaway_neo4j
):
    from mtc.knowledge.config import Neo4jConfig
    from mtc.knowledge.graph_manager import KnowledgeGraphManager
    from mtc.neural.vectors.coordinator import VectorCoordinator

    clock = Clock()

    async def pool(active):
        return throwaway_postgres

    def vectors(active, embedder):
        return VectorCoordinator(
            collection_name=throwaway_qdrant.collection,
            embedder=embedder,
            client=throwaway_qdrant.client,
            settings=active,
        )

    def graph(active):
        manager = KnowledgeGraphManager(
            config=Neo4jConfig("unused", "unused", "unused", throwaway_neo4j.database),
            driver=throwaway_neo4j.driver,
            settings=active,
        )
        manager.bootstrap_constraints()
        for name in ("river", "bridge"):
            manager.merge_concept(name)
        return manager

    stack = await build_memory_manager(
        settings,
        graph=True,
        embedder=LookupEmbedder(),
        pool_builder=pool,
        vector_builder=vectors,
        graph_builder=graph,
        clock=clock,
    )
    assert all(report["available"] for report in stack.components.values())
    stored = await stack.manager.store_memory("a stone bridge over the river", 0.5)
    assert stored.errors == {} and stored.concepts_linked == 2
    recall = await stack.manager.semantic_recall("river bridge", min_score=0.3)
    assert recall["search_stats"]["method"] == "vector"
    clock.now = T0 + timedelta(days=400)
    pruner = MemoryPruner(
        episodic=stack.manager.episodic,
        vectors=stack.vector_store,
        graph=stack.graph,
        clock=clock,
    )
    result = await pruner.prune(dry_run=False)
    assert result["vectors_deleted"] == 1 and result["graph_nodes_deleted"] == 1
