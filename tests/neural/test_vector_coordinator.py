"""Vector coordinator at 384 native dimensions over Qdrant local mode."""

import json
import os
import subprocess
import sys
import types
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pytest

qdrant_client = pytest.importorskip("qdrant_client")
from qdrant_client import models  # noqa: E402

from mtc.core.config import Settings  # noqa: E402
from mtc.core.embeddings import EmbeddingUnavailableError, HashEmbedder  # noqa: E402
from mtc.memory.protocols import StoreUnavailable, VectorStore  # noqa: E402
from mtc.neural.vectors import coordinator as coordinator_module  # noqa: E402
from mtc.neural.vectors.coordinator import (  # noqa: E402
    COLLECTION_PARAMS,
    PAYLOAD_INDEXES,
    VectorCoordinator,
    VectorDimensionMismatch,
    collection_params,
)

ROOT = Path(__file__).resolve().parents[2]
VOCABULARY = ("river", "bridge", "garden", "stone", "budget", "team", "lantern")


class LookupEmbedder:
    """Each vocabulary word owns one axis; other words share the last axis.

    Flagged semantic so the store accepts it: a stand-in with predictable
    overlaps, not a model of meaning.
    """

    semantic = True

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.calls: List[List[str]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=float)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                word = word.strip(".,;:!?")
                axis = VOCABULARY.index(word) if word in VOCABULARY else self.dim - 1
                out[row, axis] += 1.0
            out[row] /= np.linalg.norm(out[row]) or 1.0
        return out


class DownEmbedder(LookupEmbedder):
    def encode(self, texts):
        raise EmbeddingUnavailableError("model absent")


class SpyClient:
    """Delegates to a real client and records every method call."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def recorded(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            return attr(*args, **kwargs)

        return recorded

    def named(self, name: str) -> List[Dict[str, Any]]:
        return [kwargs for n, _a, kwargs in self.calls if n == name]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _coordinator(settings=None, **kwargs):
    client = kwargs.pop("client", None) or SpyClient(
        qdrant_client.QdrantClient(":memory:")
    )
    embedder = kwargs.pop("embedder", None) or LookupEmbedder()
    coordinator = VectorCoordinator(
        collection_name=kwargs.pop("collection_name", "unit_memories"),
        embedder=embedder,
        client=client,
        settings=settings or Settings(),
        **kwargs,
    )
    return coordinator, client, embedder


def test_satisfies_vector_store(clean_env):
    coordinator, _, _ = _coordinator()
    assert isinstance(coordinator, VectorStore)


def test_collection_created_with_documented_params(clean_env):
    assert COLLECTION_PARAMS == {
        "size": 384,
        "distance": "cosine",
        "hnsw_config": {
            "m": 32,
            "ef_construct": 400,
            "full_scan_threshold": 20000,
            "on_disk": False,
        },
    }
    assert PAYLOAD_INDEXES == (
        ("postgres_id", "integer"),
        ("memory_type", "keyword"),
        ("importance", "float"),
    )
    coordinator, client, _ = _coordinator()
    (create,) = client.named("create_collection")
    assert create["collection_name"] == "unit_memories"
    assert create["vectors_config"].size == 384
    assert create["vectors_config"].distance == models.Distance.COSINE
    hnsw = create["hnsw_config"]
    assert (hnsw.m, hnsw.ef_construct, hnsw.full_scan_threshold, hnsw.on_disk) == (
        32,
        400,
        20000,
        False,
    )
    for absent in ("quantization_config", "optimizers_config"):
        assert create.get(absent) is None
    indexed = {
        k["field_name"]: k["field_schema"] for k in client.named("create_payload_index")
    }
    assert indexed == {
        "postgres_id": models.PayloadSchemaType.INTEGER,
        "memory_type": models.PayloadSchemaType.KEYWORD,
        "importance": models.PayloadSchemaType.FLOAT,
    }
    info = client.get_collection("unit_memories")
    assert info.config.params.vectors.size == 384
    # building a second coordinator on the same collection creates nothing
    again, _, _ = _coordinator(client=client)
    assert len(client.named("create_collection")) == 1
    assert again.ensure_collection() is False


def test_distance_follows_the_metric_setting(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_VECTOR_SIMILARITY_METRIC", "dot")
    assert collection_params(Settings())["distance"] == "dot"
    _, client, _ = _coordinator(Settings())
    assert client.named("create_collection")[0]["vectors_config"].distance == (
        models.Distance.DOT
    )
    monkeypatch.setenv("MTC_VECTOR_SIMILARITY_METRIC", "manhattan")
    with pytest.raises(ValueError) as info:
        collection_params(Settings())
    assert "VECTOR_SIMILARITY_METRIC" in str(info.value)


def test_existing_collection_with_other_dimension_raises(clean_env):
    inner = qdrant_client.QdrantClient(":memory:")
    inner.create_collection(
        "unit_memories",
        vectors_config=models.VectorParams(size=8192, distance=models.Distance.COSINE),
    )
    with pytest.raises(VectorDimensionMismatch) as info:
        _coordinator(client=inner)
    message = str(info.value)
    assert "8192" in message and "384" in message and "re-embed" in message


def test_embedder_width_must_match(clean_env):
    with pytest.raises(VectorDimensionMismatch):
        _coordinator(embedder=LookupEmbedder(dim=16))


def test_non_semantic_embedder_needs_opt_in(clean_env):
    with pytest.raises(EmbeddingUnavailableError):
        _coordinator(embedder=HashEmbedder())
    allowed, _, _ = _coordinator(embedder=HashEmbedder(), allow_non_semantic=True)
    assert allowed.status()["semantic"] is False


def test_settings_supply_host_port_collection_at_call_time(clean_env, monkeypatch):
    built: List[Dict[str, Any]] = []

    class RecordingClient:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "qdrant_client",
        types.SimpleNamespace(QdrantClient=RecordingClient),
    )
    monkeypatch.setenv("MTC_QDRANT_HOST", "vectors.example")
    monkeypatch.setenv("MTC_QDRANT_PORT", "7333")
    monkeypatch.setenv("MTC_QDRANT_COLLECTION", "unit_collection")
    first = Settings()
    monkeypatch.setattr(coordinator_module, "get_settings", lambda: first)
    coordinator = VectorCoordinator(embedder=LookupEmbedder(), create_collection=False)
    assert built[-1] == {"host": "vectors.example", "port": 7333}
    assert coordinator.collection_name == "unit_collection"
    monkeypatch.setenv("MTC_QDRANT_COLLECTION", "later_collection")
    later = Settings()
    monkeypatch.setattr(coordinator_module, "get_settings", lambda: later)
    assert (
        VectorCoordinator(
            embedder=LookupEmbedder(), create_collection=False
        ).collection_name
        == "later_collection"
    )
    explicit = VectorCoordinator(
        host="other.example",
        port=1,
        collection_name="named",
        embedder=LookupEmbedder(),
        create_collection=False,
    )
    assert built[-1] == {"host": "other.example", "port": 1}
    assert explicit.collection_name == "named"


def test_embeddings_are_native_dimension(clean_env):
    coordinator, _, _ = _coordinator()
    vector = coordinator.embed_memory("river and bridge")
    assert vector.shape == (384,) and vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)
    assert not hasattr(coordinator, "projection_matrix")
    assert coordinator.vector_size == 384


def test_coordinator_has_no_mock_path():
    source = Path(coordinator_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("hash(", "np.random", "use_mock", "random.seed"):
        assert forbidden not in source, forbidden


def test_unavailable_embedder_raises_and_writes_nothing(clean_env):
    coordinator, client, _ = _coordinator(embedder=DownEmbedder())
    with pytest.raises(EmbeddingUnavailableError):
        coordinator.store_memory("river")
    with pytest.raises(EmbeddingUnavailableError):
        coordinator.store_memories(["river", "bridge"])
    with pytest.raises(EmbeddingUnavailableError):
        coordinator.search_similar("river")
    assert client.named("upsert") == []
    assert client.count("unit_memories").count == 0
    status = coordinator.status()
    assert status["available"] is False and "model absent" in status["last_error"]
    metrics = coordinator.get_metrics()
    assert metrics["total_embeddings"] == 0 and metrics["failed_embeddings"] == 4


def test_point_ids_are_uuids(clean_env):
    coordinator, _, _ = _coordinator()
    first = coordinator.store_memory("river")
    second = coordinator.store_memory("river")
    assert uuid.UUID(first).version == 4 and first != second
    fixed = str(uuid.UUID(int=99))
    assert coordinator.store_memory("bridge", point_id=fixed) == fixed
    with pytest.raises(ValueError):
        coordinator.store_memory("bridge", point_id="not-a-uuid")
    assert coordinator.get_memory_by_id("not-a-uuid") is None


def test_store_memories_upserts_once(clean_env):
    coordinator, client, embedder = _coordinator()
    ids = coordinator.store_memories(
        ["river walk", "stone bridge", "garden path"],
        metadatas=[{"memory_type": "episodic"}, None, {"tag": "x"}],
        importances=[1.0, 2.0, 3.0],
    )
    assert len(client.named("upsert")) == 1
    assert len(embedder.calls) == 1 and len(embedder.calls[0]) == 3
    assert len(set(ids)) == 3 and all(uuid.UUID(i) for i in ids)
    payload = coordinator.get_memory_by_id(ids[2])
    assert payload["tag"] == "x" and payload["importance"] == 3.0
    assert payload["text"] == "garden path" and "stored_at" in payload
    assert coordinator.store_memories([]) == []
    assert len(client.named("upsert")) == 1
    with pytest.raises(ValueError):
        coordinator.store_memories(["a", "b"], importances=[1.0])


def test_search_uses_vector_search_limit_default(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_VECTOR_SEARCH_LIMIT", "2")
    limited = Settings()
    monkeypatch.setattr(coordinator_module, "get_settings", lambda: limited)
    client = SpyClient(qdrant_client.QdrantClient(":memory:"))
    coordinator = VectorCoordinator(
        collection_name="unit_memories", embedder=LookupEmbedder(), client=client
    )  # no settings argument: the limit is read through get_settings at call time
    coordinator.store_memories(["river", "river bridge", "river stone", "garden"])
    hits = coordinator.search_similar("river", min_score=-1.0)
    assert len(hits) == 2
    assert client.named("query_points")[-1]["limit"] == 2
    assert len(coordinator.search_similar("river", top_k=3, min_score=-1.0)) == 3
    assert coordinator.search_similar("river", top_k=0) == []


def test_filters_and_scores(clean_env):
    coordinator, _, _ = _coordinator()
    a = coordinator.store_memory("river", metadata={"memory_type": "episodic"})
    coordinator.store_memory("river", metadata={"memory_type": "gist"})
    hits = coordinator.search_similar(
        "river", min_score=0.5, filters={"memory_type": "episodic"}
    )
    assert [h[0] for h in hits] == [a]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)
    coordinator.update_memory_importance(a, 7.5)
    assert coordinator.get_memory_by_id(a)["importance"] == 7.5
    stats = coordinator.get_collection_stats()
    assert stats["points_count"] == 2 and stats["config"]["vector_size"] == 384


def test_metrics_report_latency_without_target_claims(clean_env):
    coordinator, _, _ = _coordinator()
    before = coordinator.get_metrics()
    assert before["avg_embedding_time_ms"] is None  # nothing measured yet
    coordinator.store_memory("river")
    coordinator.search_similar("river")
    metrics = coordinator.get_metrics()
    assert metrics["total_embeddings"] == 2 and metrics["total_searches"] == 1
    assert metrics["avg_embedding_time_ms"] >= 0.0
    assert metrics["avg_search_time_ms"] >= 0.0
    assert metrics["last_search_ms"] >= 0.0
    for claim in ("target_latency_ms", "meeting_target", "speedup"):
        assert claim not in metrics


def test_unreachable_server_is_store_unavailable(clean_env):
    class ResponseHandlingException(Exception):
        pass

    class Unreachable:
        def __getattr__(self, name):
            def fail(*args, **kwargs):
                raise ResponseHandlingException("connection refused")

            return fail

    with pytest.raises(StoreUnavailable):
        _coordinator(client=Unreachable())
    coordinator, _, _ = _coordinator(client=Unreachable(), create_collection=False)
    with pytest.raises(StoreUnavailable):
        coordinator.store_memory("river")
    with pytest.raises(StoreUnavailable):
        coordinator.get_memory_by_id(str(uuid.uuid4()))  # never an invented None
    assert coordinator.status()["available"] is False


def test_module_import_loads_no_qdrant():
    code = (
        "import json, sys\n"
        "import mtc.neural.vectors.coordinator\n"
        "import mtc.neural.vectors\n"
        "print(json.dumps([m for m in ('qdrant_client', 'sentence_transformers')"
        " if m in sys.modules]))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYPHI_WELCOME_OFF": "yes"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
