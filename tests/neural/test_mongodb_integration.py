"""The integrated memory system: one write to a vector store and a thought
collection, linked by the point id.

The vector side runs on the in-memory store and, when ``qdrant_client`` is
installed, on the coordinator over Qdrant local mode (no server). The thought
side runs on the data manager over the fake MongoDB driver of
``tests/memory/test_mongo_stores.py``. A lookup embedder maps known texts to
fixed unit vectors and counts its calls.
"""

import ast
import asyncio
import inspect
from typing import Dict, List

import numpy as np
import pytest

from mtc.memory.inmemory import InMemoryVectorStore
from mtc.memory.protocols import StoreUnavailable
from mtc.neural.mongodb_schemas import NeuralDataManager
from mtc.neural.vectors import mongodb_integration
from mtc.neural.vectors.mongodb_integration import IntegratedMemorySystem
from tests.memory.test_mongo_stores import FakeMongoClient

DIM = 8


def _unit(*weights: float) -> np.ndarray:
    vector = np.zeros(DIM)
    vector[: len(weights)] = weights
    return vector / np.linalg.norm(vector)


TABLE: Dict[str, np.ndarray] = {
    "alpha memory": _unit(1.0, 0.05),
    "beta memory": _unit(0.0, 1.0),
    "alpha again": _unit(1.0, 0.1),
    "gamma memory": _unit(0.0, 0.0, 1.0),
    "alpha query": _unit(1.0, 0.0),
    "beta query": _unit(0.0, 1.0, 0.0),
}


class LookupEmbedder:
    """Fixed vectors for known texts; any other text is an error."""

    semantic = True
    dim = DIM

    def __init__(self) -> None:
        self.calls: List[str] = []

    def encode(self, texts):
        self.calls.extend(texts)
        return np.stack([TABLE[text] for text in texts])


class Ticker:
    """A clock that advances by a fixed step on every read."""

    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def _coordinator(embedder):
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient

    from mtc.neural.vectors.coordinator import VectorCoordinator

    return VectorCoordinator(
        collection_name="integration_test",
        vector_size=DIM,
        embedder=embedder,
        client=QdrantClient(":memory:"),
    )


@pytest.fixture(params=["inmemory", "coordinator"])
def embedder_and_store(request):
    embedder = LookupEmbedder()
    if request.param == "inmemory":
        return embedder, InMemoryVectorStore(embedder=embedder)
    return embedder, _coordinator(embedder)


@pytest.fixture
def client() -> FakeMongoClient:
    return FakeMongoClient()


@pytest.fixture
def manager(client) -> NeuralDataManager:
    return NeuralDataManager(database_name="integration", client=client)


@pytest.fixture
def system(embedder_and_store, manager) -> IntegratedMemorySystem:
    return IntegratedMemorySystem(embedder_and_store[1], manager)


@pytest.mark.asyncio
async def test_dual_write_links_point_and_thought(system, manager, embedder_and_store):
    result = await system.store_memory(
        "alpha memory",
        emotional_state={"joy": 0.8, "fear": 0.2},
        thought_type="reflection",
        importance=0.7,
        triggers=["a trigger"],
        parent_thought_id="parent-1",
    )
    point_id = result["point_id"]
    assert result["stream_id"] == f"thought_{point_id}"
    assert result["thought_stored"] is True and isinstance(result["thought_id"], str)
    payload = embedder_and_store[1].get_memory_by_id(point_id)
    assert payload["text"] == "alpha memory"
    assert payload["emotional_state"] == {"joy": 0.8, "fear": 0.2}
    assert payload["emotional_weight"] == pytest.approx(0.5)
    assert payload["importance"] == pytest.approx(0.7)
    thought = manager.thought_streams.docs[0]
    assert thought["associated_memories"] == [point_id]
    assert thought["stream_id"] == result["stream_id"]
    assert thought["thought_text"] == "alpha memory"
    assert thought["thought_type"] == "reflection"
    assert thought["triggers"] == ["a trigger"]
    assert thought["parent_thought_id"] == "parent-1"
    assert thought["metadata"]["vector_point_id"] == point_id
    linked = await system.get_memory_with_neural_context(point_id)
    assert linked["vector_data"]["text"] == "alpha memory"
    assert linked["neural_context"]["thought_type"] == "reflection"
    assert linked["point_id"] == point_id
    assert await system.get_memory_with_neural_context("no-such-point") is None


@pytest.mark.asyncio
async def test_no_emotional_state_means_no_weight_not_a_neutral_default(system):
    result = await system.store_memory("alpha memory")
    payload = system.vector_store.get_memory_by_id(result["point_id"])
    assert payload["emotional_state"] == {} and payload["emotional_weight"] is None
    assert system.data_manager.thought_streams.docs[0]["emotional_state"] == {}


@pytest.mark.asyncio
async def test_embeds_once_per_store(system, embedder_and_store):
    embedder = embedder_and_store[0]
    await system.store_memory("alpha memory", {"joy": 0.5})
    assert embedder.calls == ["alpha memory"]  # the reference embedded twice
    await system.search_memories("alpha query", top_k=3, min_score=0.0)
    assert embedder.calls == ["alpha memory", "alpha query"]


@pytest.mark.asyncio
async def test_search_applies_emotional_filter(system):
    await system.store_memory("alpha memory", {"joy": 0.9})
    await system.store_memory("alpha again", {"fear": 0.9})
    await system.store_memory("beta memory", {"joy": 0.9})
    # Every floor here (-1.0) sits below every score in this test: "beta memory"
    # scores exactly 0.0 against the query, and a floor of exactly 0.0 keeps it
    # in the in-memory store but not in Qdrant local mode from client 1.19 (see
    # search_similar).
    everything = await system.search_memories("alpha query", top_k=5, min_score=-1.0)
    assert len(everything) == 3
    joyful = await system.search_memories(
        "alpha query", top_k=5, emotional_filter="joy", min_score=-1.0
    )
    assert [hit[2]["text"] for hit in joyful] == ["alpha memory", "beta memory"]
    assert all(hit[2]["emotional_state"]["joy"] >= 0.5 for hit in joyful)
    strict = await system.search_memories(
        "alpha query",
        top_k=5,
        emotional_filter="joy",
        min_score=-1.0,
        emotion_threshold=0.95,
    )
    assert strict == []
    fearful = await system.search_memories(
        "alpha query", top_k=1, emotional_filter="fear", min_score=-1.0
    )
    assert [hit[2]["text"] for hit in fearful] == ["alpha again"]
    capped = await system.search_memories("alpha query", top_k=1, min_score=-1.0)
    assert len(capped) == 1
    assert await system.search_memories("alpha query", top_k=0, min_score=-1.0) == []


@pytest.mark.asyncio
async def test_in_memory_store_keeps_a_score_equal_to_the_floor(manager):
    """The in-memory store includes a hit scoring exactly the floor; the
    coordinator's docstring says Qdrant's rule at the floor is the client's."""
    embedder = LookupEmbedder()
    system = IntegratedMemorySystem(InMemoryVectorStore(embedder=embedder), manager)
    await system.store_memory("alpha memory", {"joy": 0.9})
    await system.store_memory("beta memory", {"joy": 0.9})
    hits = await system.search_memories("alpha query", top_k=5, min_score=0.0)
    assert [hit[2]["text"] for hit in hits] == ["alpha memory", "beta memory"]
    assert hits[-1][1] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_search_uses_the_store_default_score_floor(system):
    await system.store_memory("alpha memory", {"joy": 0.9})
    await system.store_memory("beta memory", {"joy": 0.9})
    hits = await system.search_memories("alpha query", top_k=5)
    assert [hit[2]["text"] for hit in hits] == ["alpha memory"]  # 0.7 default floor


@pytest.mark.asyncio
async def test_recent_memories_with_emotion(system):
    for text, emotions in (
        ("alpha memory", {"joy": 0.9}),
        ("beta memory", {"joy": 0.2, "fear": 0.9}),
        ("gamma memory", {"joy": 0.8}),
    ):
        await system.store_memory(text, emotions)
        await asyncio.sleep(0.005)  # thought times are milliseconds apart
    found = await system.get_recent_memories_with_emotion("joy", threshold=0.5)
    assert [m["vector_data"]["text"] for m in found] == [
        "gamma memory",
        "alpha memory",
    ]
    assert all(m["neural_context"] is not None for m in found)
    limited = await system.get_recent_memories_with_emotion("joy", 0.5, limit=1)
    assert len(limited) == 1
    # a thought whose vector point was deleted is left out
    gone = found[0]["point_id"]
    system.vector_store.delete_memories([gone])
    after = await system.get_recent_memories_with_emotion("joy", threshold=0.5)
    assert [m["vector_data"]["text"] for m in after] == ["alpha memory"]


def test_no_private_filter(system, client):
    for node in ast.walk(ast.parse(inspect.getsource(mongodb_integration))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) < 200:
                assert "private" not in node.value.lower(), node.value

    async def scenario() -> None:
        await system.store_memory("alpha memory", {"joy": 0.9})
        await system.get_recent_memories_with_emotion("joy")
        await system.get_memory_with_neural_context(
            system.data_manager.thought_streams.docs[0]["associated_memories"][0]
        )

    asyncio.run(scenario())
    for call in client.calls:
        assert "private" not in str(call[2])


@pytest.mark.asyncio
async def test_metrics_report_measured_latency(embedder_and_store, manager):
    clock = Ticker(step=0.010)  # every clock read is 10 ms later
    system = IntegratedMemorySystem(
        embedder_and_store[1], manager, latency_target_ms=50.0, clock=clock
    )
    empty = system.get_metrics()
    assert empty["total_writes"] == 0
    for name in (
        "avg_latency_ms",
        "avg_vector_ms",
        "avg_thought_ms",
        "max_latency_ms",
        "min_latency_ms",
        "target_met_rate",
    ):
        assert empty[name] is None, name
    first = await system.store_memory("alpha memory", {"joy": 0.5})
    assert first["vector_ms"] == pytest.approx(10.0)
    assert first["thought_ms"] == pytest.approx(10.0)
    assert first["latency_ms"] == pytest.approx(20.0) and first["target_met"] is True
    slow = IntegratedMemorySystem(
        embedder_and_store[1], manager, latency_target_ms=15.0, clock=Ticker(0.010)
    )
    second = await slow.store_memory("beta memory", {"joy": 0.5})
    assert second["target_met"] is False
    metrics = system.get_metrics()
    assert metrics["total_writes"] == 1 and metrics["avg_latency_ms"] == pytest.approx(
        20.0
    )
    assert metrics["max_latency_ms"] == metrics["min_latency_ms"] == pytest.approx(20.0)
    assert metrics["target_met_count"] == 1 and metrics["target_missed_count"] == 0
    assert metrics["target_met_rate"] == 1.0 and metrics["latency_target_ms"] == 50.0
    assert slow.get_metrics()["target_missed_count"] == 1


@pytest.mark.asyncio
async def test_missing_store_is_named(embedder_and_store, manager):
    vector_store = embedder_and_store[1]
    no_vectors = IntegratedMemorySystem(None, manager)
    with pytest.raises(StoreUnavailable) as caught:
        await no_vectors.store_memory("alpha memory")
    assert "vector store" in str(caught.value)
    with pytest.raises(StoreUnavailable):
        await no_vectors.search_memories("alpha query")
    no_manager = IntegratedMemorySystem(vector_store, None)
    with pytest.raises(StoreUnavailable) as caught:
        await no_manager.store_memory("alpha memory")
    assert "data manager" in str(caught.value)
    assert manager.thought_streams.docs == []
    # nothing was written to the vector store either: the check comes first
    assert vector_store.search_similar("alpha query", top_k=5, min_score=0.0) == []
    status = IntegratedMemorySystem(None, None).status()
    assert status["available"] is False and status["vector_store"] is None


@pytest.mark.asyncio
async def test_vector_failure_writes_nothing_to_mongodb(manager, embedder_and_store):
    class Broken:
        def store_memory(self, *args, **kwargs):
            raise RuntimeError("embedder unavailable")

    system = IntegratedMemorySystem(Broken(), manager)
    with pytest.raises(RuntimeError):
        await system.store_memory("alpha memory")
    assert manager.thought_streams.docs == []
    assert system.get_metrics()["total_writes"] == 0


@pytest.mark.asyncio
async def test_unreachable_mongodb_keeps_the_vector_and_says_so(
    system, client, embedder_and_store
):
    client.unreachable = True
    result = await system.store_memory("alpha memory", {"joy": 0.5})
    assert result["thought_stored"] is False and result["thought_id"] is None
    assert embedder_and_store[1].get_memory_by_id(result["point_id"]) is not None
    assert system.get_metrics()["thoughts_not_stored"] == 1
    with pytest.raises(StoreUnavailable):
        await system.get_recent_memories_with_emotion("joy")
    status = system.status()
    assert status["available"] is False and status["last_error"]


@pytest.mark.asyncio
async def test_status_reports_both_stores(system):
    await system.store_memory("alpha memory", {"joy": 0.5})
    status = system.status()
    assert status["backend"] == "integrated" and status["available"] is True
    assert status["last_error"] is None
    assert status["data_manager"]["backend"] == "mongodb"
    assert status["vector_store"]["backend"] in ("memory", "qdrant")


def test_status_unknown_until_the_data_manager_made_contact(system):
    status = system.status()
    assert status["data_manager"]["available"] is None
    assert status["available"] is None
    assert status["last_error"] is None


def test_module_import_loads_no_store_client():
    import subprocess
    import sys

    code = (
        "import sys, mtc.neural.vectors.mongodb_integration;"
        "bad=[m for m in ('pymongo','qdrant_client','motor') if m in sys.modules];"
        "sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-1000:]


def test_types_are_what_the_signature_promises(system):
    parameters = inspect.signature(IntegratedMemorySystem.store_memory).parameters
    assert list(parameters)[1:] == [
        "text",
        "emotional_state",
        "thought_type",
        "importance",
        "triggers",
        "parent_thought_id",
    ]
    assert parameters["thought_type"].default == "memory"
    search = inspect.signature(IntegratedMemorySystem.search_memories).parameters
    assert search["top_k"].default == 5 and search["emotional_filter"].default is None
    recent = inspect.signature(
        IntegratedMemorySystem.get_recent_memories_with_emotion
    ).parameters
    assert recent["threshold"].default == 0.5 and recent["limit"].default == 10
    assert IntegratedMemorySystem(None, None).latency_target_ms == 50.0
    assert isinstance(system, IntegratedMemorySystem)
