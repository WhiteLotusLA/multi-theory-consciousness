"""Contract tests for the vector and graph stores.

Every case runs against each entry of ``VECTOR_BACKENDS`` or
``GRAPH_BACKENDS``. The in-memory backends are the default; the Qdrant
coordinator and the graph manager join by appending one ``pytest.param`` entry
each (carrying ``pytest.mark.db`` where they need a server), so no case below
needs editing. A vector factory takes the embedder and ``allow_non_semantic``.
A backend that needs a throwaway store names its fixture; the factory then
receives the fixture's value first. The Qdrant coordinator also runs here in
local mode (``QdrantClient(":memory:")``, no server and no marker).
"""

import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pytest

from mtc.core.embeddings import EmbeddingUnavailableError, HashEmbedder
from mtc.memory import inmemory
from mtc.memory.protocols import (
    GraphStore,
    VectorStore,
    point_id_for_memory,
)


@dataclass(frozen=True)
class Backend:
    name: str
    factory: Callable[..., Any]
    fixture: Optional[str] = None


class _SemanticHash(HashEmbedder):
    """Hash vectors flagged semantic: a stand-in with predictable overlap."""

    semantic = True


def _memory_vector(embedder=None, allow_non_semantic=False):
    return inmemory.InMemoryVectorStore(
        embedder or _SemanticHash(), allow_non_semantic=allow_non_semantic
    )


def _qdrant_local(embedder=None, allow_non_semantic=False):
    qdrant_client = pytest.importorskip("qdrant_client")
    from mtc.neural.vectors.coordinator import VectorCoordinator

    return VectorCoordinator(
        collection_name="contract_memories",
        embedder=embedder or _SemanticHash(),
        client=qdrant_client.QdrantClient(":memory:"),
        allow_non_semantic=allow_non_semantic,
    )


def _qdrant_server(target, embedder=None, allow_non_semantic=False):
    from mtc.neural.vectors.coordinator import VectorCoordinator

    return VectorCoordinator(
        collection_name=target.collection,
        embedder=embedder or _SemanticHash(),
        client=target.client,
        allow_non_semantic=allow_non_semantic,
    )


def _neo4j_graph(target):
    from mtc.knowledge.config import Neo4jConfig
    from mtc.knowledge.graph_manager import KnowledgeGraphManager

    manager = KnowledgeGraphManager(
        config=Neo4jConfig("unused", "unused", "unused", target.database),
        driver=target.driver,
    )
    manager.bootstrap_constraints()
    return manager


# Append a pytest.param(Backend(...), marks=pytest.mark.db, id="...") entry per
# real store; the cases below do not change.
VECTOR_BACKENDS = [
    pytest.param(Backend("inmemory", _memory_vector), id="inmemory"),
    pytest.param(Backend("qdrant_local", _qdrant_local), id="qdrant_local"),
    pytest.param(
        Backend("qdrant", _qdrant_server, fixture="throwaway_qdrant"),
        marks=pytest.mark.db,
        id="qdrant",
    ),
]
GRAPH_BACKENDS = [
    pytest.param(Backend("inmemory", inmemory.InMemoryGraphStore), id="inmemory"),
    pytest.param(
        Backend("neo4j", _neo4j_graph, fixture="throwaway_neo4j"),
        marks=pytest.mark.db,
        id="neo4j",
    ),
]


def _bound_factory(request):
    backend = request.param
    if backend.fixture is None:
        return backend.factory
    # The throwaway fixtures are synchronous, so they resolve here directly.
    return functools.partial(backend.factory, request.getfixturevalue(backend.fixture))


@pytest.fixture(params=VECTOR_BACKENDS)
def make_vector(request):
    return _bound_factory(request)


@pytest.fixture(params=GRAPH_BACKENDS)
def make_graph(request):
    return _bound_factory(request)


def test_stores_satisfy_their_protocols(make_vector, make_graph):
    assert isinstance(make_vector(), VectorStore)
    assert isinstance(make_graph(), GraphStore)


def test_store_then_search_finds_text(make_vector):
    store = make_vector()
    first = store.store_memory(
        "the garden path winds past the old stone wall",
        metadata={"memory_type": "episodic"},
        importance=2.5,
        emotional_weight=0.25,
    )
    store.store_memory("quarterly budget review for the accounting team")
    assert isinstance(first, str) and first
    hits = store.search_similar(
        "the garden path winds past the old stone wall", min_score=0.5
    )
    assert hits and hits[0][0] == first
    point_id, score, payload = hits[0]
    assert score == pytest.approx(1.0, abs=1e-6)
    assert payload["text"] == "the garden path winds past the old stone wall"
    assert payload["memory_type"] == "episodic"
    assert payload["importance"] == 2.5 and payload["emotional_weight"] == 0.25
    assert store.get_memory_by_id(first)["text"] == payload["text"]
    assert store.get_memory_by_id("absent") is None
    filtered = store.search_similar(
        "the garden path", min_score=0.0, filters={"memory_type": "episodic"}
    )
    assert [h[0] for h in filtered] == [first]
    assert store.search_similar("the garden path", top_k=0, min_score=0.0) == []
    assert store.status()["available"] is True


def test_min_score_filters(make_vector):
    store = make_vector()
    store.store_memory("alpha beta gamma delta")
    store.store_memory("zebra yak xylophone walrus")
    everything = store.search_similar("alpha beta gamma delta", min_score=-1.0)
    assert len(everything) == 2
    scores = [score for _, score, _ in everything]
    assert scores == sorted(scores, reverse=True)
    strict = store.search_similar("alpha beta gamma delta", min_score=0.99)
    assert len(strict) == 1 and strict[0][2]["text"] == "alpha beta gamma delta"
    assert store.search_similar("alpha beta gamma delta", min_score=1.5) == []


def test_point_id_for_memory_is_stable():
    assert point_id_for_memory(7) == point_id_for_memory(7)
    assert point_id_for_memory(7) != point_id_for_memory(8)
    assert point_id_for_memory(7) == point_id_for_memory("7")
    assert len(point_id_for_memory(7)) == 36


def test_upsert_with_a_derived_point_id(make_vector):
    store = make_vector()
    pid = point_id_for_memory(41)
    assert store.store_memory("first text", point_id=pid) == pid
    assert store.store_memory("second text", point_id=pid) == pid
    assert store.get_memory_by_id(pid)["text"] == "second text"
    assert len(store.search_similar("text", min_score=-1.0)) == 1


def test_delete_memories_removes_points(make_vector):
    store = make_vector()
    a = store.store_memory("one")
    b = store.store_memory("two")
    assert store.delete_memories([a, "never-stored"]) == 1
    assert store.get_memory_by_id(a) is None and store.get_memory_by_id(b)
    assert store.delete_memories([a]) == 0
    assert store.delete_memories([]) == 0


def test_non_semantic_embedder_requires_opt_in(make_vector):
    with pytest.raises(EmbeddingUnavailableError):
        make_vector(HashEmbedder())
    allowed = make_vector(HashEmbedder(), allow_non_semantic=True)
    assert allowed.status()["semantic"] is False
    allowed.store_memory("explicitly allowed")
    assert len(allowed.search_similar("explicitly allowed", min_score=0.5)) == 1


def test_embedder_failure_stores_nothing(make_vector):
    class Down(_SemanticHash):
        def encode(self, texts):
            raise EmbeddingUnavailableError("model absent")

    store = make_vector(Down())
    with pytest.raises(EmbeddingUnavailableError):
        store.store_memory("text")
    status = store.status()
    assert status["available"] is False and "model absent" in status["last_error"]
    assert status.get("points", 0) == 0


def test_returned_payloads_are_copies(make_vector):
    store = make_vector()
    pid = store.store_memory("text", metadata={"tags": ["a"]})
    store.get_memory_by_id(pid)["tags"].append("b")
    assert store.get_memory_by_id(pid)["tags"] == ["a"]
    store.search_similar("text", min_score=0.0)[0][2]["tags"].append("c")
    assert store.get_memory_by_id(pid)["tags"] == ["a"]


def test_link_and_list_concepts(make_graph):
    graph = make_graph()
    for name in ("river", "bridge", "stone"):
        graph.merge_concept(name, description=f"about {name}")
    assert graph.list_concept_names() == ["bridge", "river", "stone"]
    assert graph.list_concept_names(limit=2) == ["bridge", "river"]
    linked = graph.link_memory_to_concepts(
        11, "a bridge over the river", ["river", "bridge", "unknown"], relevance=0.6
    )
    assert linked == 2  # the unknown concept is skipped
    assert graph.link_memory_to_concepts(11, "again", ["river"]) == 1
    assert graph.link_memory_to_concepts(12, "no concepts", []) == 0
    graph.link_concept_to_conversation("lantern", "conv-1", "Alice")
    graph.link_concept_to_conversation("lantern", "conv-1", "Guest")
    assert "lantern" in graph.list_concept_names()  # the mention merged it


def test_merge_concept_is_idempotent(make_graph):
    graph = make_graph()
    graph.merge_concept("river", "first description", "nature")
    graph.merge_concept("river", "second description", "other")
    graph.merge_concept("river")
    assert graph.list_concept_names() == ["river"]


def test_related_concepts_shape(make_graph):
    graph = make_graph()
    for name in ("a", "b", "c", "d"):
        graph.merge_concept(name)
    assert graph.relate_concepts("a", "b", "similar_to", 0.9) is True
    assert graph.relate_concepts("c", "a", strength=0.4) is True
    assert graph.relate_concepts("a", "missing") is False
    assert graph.relate_concepts("a", "a") is False
    rows = graph.get_related_concepts("a")
    assert rows == [
        {"concept": "b", "relationship": "similar_to", "strength": 0.9},
        {"concept": "c", "relationship": "relates_to", "strength": 0.4},
    ]
    assert graph.get_related_concepts("a", limit=1) == rows[:1]
    assert graph.get_related_concepts("d") == []
    assert graph.get_related_concepts("missing") == []
    rows[0]["strength"] = 0.0
    assert graph.get_related_concepts("a")[0]["strength"] == 0.9


def test_delete_memory_nodes(make_graph):
    graph = make_graph()
    graph.merge_concept("river")
    graph.link_memory_to_concepts(1, "one", ["river"])
    graph.link_memory_to_concepts(2, "two", [])
    assert graph.delete_memory_nodes([1]) == 1
    assert graph.delete_memory_nodes([1]) == 0
    assert graph.delete_memory_nodes([2, 3]) == 1
    assert graph.delete_memory_nodes([]) == 0
    assert graph.list_concept_names() == ["river"]  # concepts survive
    graph.close()
    assert graph.status()["available"] is True
