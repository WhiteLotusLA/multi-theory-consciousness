"""GPU vector coordinator and the GPU factory: honest paths and counts."""

import os
import uuid
from typing import Any, Dict, List, Sequence

import numpy as np
import pytest

torch = pytest.importorskip("torch")
qdrant_client = pytest.importorskip("qdrant_client")

from mtc.core import device as device_module  # noqa: E402
from mtc.core.config import Settings  # noqa: E402
from mtc.core.embeddings import EmbeddingUnavailableError  # noqa: E402
from mtc.neural.gpu_integration import (  # noqa: E402
    GPUComponentFactory,
    GPUIntegrationConfig,
)
from mtc.neural.liquid.lsm_core import LiquidStateMachine  # noqa: E402
from mtc.neural.liquid.mps_lsm import MPSLiquidStateMachine  # noqa: E402
from mtc.neural.mps_utils import MPSManager  # noqa: E402
from mtc.neural.vectors.coordinator import VectorCoordinator  # noqa: E402
from mtc.neural.vectors.mps_coordinator import MPSVectorCoordinator  # noqa: E402

VOCABULARY = ("river", "bridge", "garden", "stone", "lantern")


class LookupEmbedder:
    """Each vocabulary word owns one axis (a semantic-flagged test stand-in)."""

    semantic = True

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for row, text in enumerate(texts):
            for word in text.lower().split():
                axis = VOCABULARY.index(word) if word in VOCABULARY else self.dim - 1
                out[row, axis] += 1.0
            out[row] /= np.linalg.norm(out[row]) or 1.0
        return out


class DownEmbedder(LookupEmbedder):
    def encode(self, texts):
        raise EmbeddingUnavailableError("model absent")


class PretendAccelerator(MPSManager):
    """Reports an accelerated device while computing on the CPU."""

    def __init__(self) -> None:
        super().__init__(device="cpu")
        self.synced = 0

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def enabled(self) -> bool:
        return True

    def synchronize(self) -> None:
        self.synced += 1

    def get_memory_stats(self) -> Dict[str, Any]:
        return {"device": "pretend", "allocated_mb": 1.0, "driver_mb": 2.0}


class SpyClient:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: List[str] = []

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def recorded(*args, **kwargs):
            self.calls.append(name)
            return attr(*args, **kwargs)

        return recorded


@pytest.fixture
def cpu_settings(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MTC_DEVICE", "cpu")
    device_module._LOGGED.clear()
    return Settings()


def _coordinator(settings, manager=None, embedder=None, client=None, **kwargs):
    return MPSVectorCoordinator(
        collection_name="gpu_memories",
        embedder=embedder or LookupEmbedder(),
        client=client or qdrant_client.QdrantClient(":memory:"),
        settings=settings,
        manager=manager or MPSManager(settings=settings),
        **kwargs,
    )


def test_batch_embedding_shape_and_device(cpu_settings):
    coordinator = _coordinator(cpu_settings)
    assert coordinator.use_gpu is False  # the CPU setting wins over use_gpu=True
    batch = coordinator.embed_memory_batch_gpu(["river", "stone bridge", "garden"])
    assert tuple(batch.shape) == (3, 384) and batch.dtype == torch.float32
    assert batch.device.type == "cpu"
    assert torch.allclose(batch.norm(dim=1), torch.ones(3))
    empty = coordinator.embed_memory_batch_gpu([])
    assert tuple(empty.shape) == (0, 384)
    assert coordinator.gpu_metrics["cpu_embeddings"] == 3
    assert coordinator.gpu_metrics["gpu_embeddings"] == 0


@pytest.mark.gpu
def test_batch_embedding_on_the_accelerator(tmp_path):
    if torch.backends.mps.is_available():
        kind = "mps"
    elif torch.cuda.is_available():
        kind = "cuda"
    else:
        pytest.skip("needs MPS or CUDA")
    coordinator = _coordinator(
        Settings(), manager=MPSManager(device=kind), gpu_batch_threshold=2
    )
    batch = coordinator.embed_memory_batch_gpu(["river", "stone"])
    assert batch.device.type == kind
    single = coordinator.embed_memory_batch_gpu(["river"])
    assert single.device.type == "cpu"  # under the batch threshold
    assert coordinator.gpu_metrics["gpu_embeddings"] == 2


def test_gpu_search_honors_flag(cpu_settings):
    manager = PretendAccelerator()
    coordinator = _coordinator(cpu_settings, manager=manager)
    assert coordinator.use_gpu is True
    point = coordinator.store_memory("river")
    off = coordinator.search_similar_gpu("river", use_gpu_search=False)
    assert coordinator.gpu_metrics["cpu_searches"] == 1
    assert coordinator.gpu_metrics["gpu_searches"] == 0
    assert coordinator.gpu_metrics["last_search_path"] == "cpu"
    synced = manager.synced
    on = coordinator.search_similar_gpu("river", use_gpu_search=True)
    assert coordinator.gpu_metrics["gpu_searches"] == 1
    assert coordinator.gpu_metrics["last_search_path"] == "gpu"
    assert manager.synced > synced  # the accelerated path really ran
    assert [h[0] for h in off] == [h[0] for h in on] == [point]
    assert coordinator.search_similar_gpu("river", top_k=0) == []

    plain = _coordinator(cpu_settings)
    plain.store_memory("river")
    plain.search_similar_gpu("river", use_gpu_search=True)
    assert plain.gpu_metrics["last_search_path"] == "cpu"  # no accelerator to use


def test_batch_store_upserts_once_with_uuid_ids(cpu_settings):
    client = SpyClient(qdrant_client.QdrantClient(":memory:"))
    coordinator = _coordinator(cpu_settings, client=client)
    ids = coordinator.store_memories_batch(
        ["river", "stone bridge", "lantern"],
        metadatas=[{"memory_type": "episodic"}, None, {"tag": "t"}],
        importances=[1.0, 4.0, 9.0],
    )
    assert client.calls.count("upsert") == 1
    assert len(set(ids)) == 3 and all(uuid.UUID(i).version == 4 for i in ids)
    assert coordinator.get_memory_by_id(ids[1])["importance"] == 4.0
    assert coordinator.get_memory_by_id(ids[2])["tag"] == "t"
    assert coordinator.store_memories_batch([]) == []
    assert client.calls.count("upsert") == 1
    with pytest.raises(ValueError):
        coordinator.store_memories_batch(["a"], importances=[1.0, 2.0])


def test_metrics_never_count_failed_embeddings(cpu_settings):
    client = SpyClient(qdrant_client.QdrantClient(":memory:"))
    coordinator = _coordinator(
        cpu_settings,
        manager=PretendAccelerator(),
        embedder=DownEmbedder(),
        client=client,
    )
    for call in (
        lambda: coordinator.embed_memory_batch_gpu(["river", "stone"]),
        lambda: coordinator.store_memories_batch(["river"]),
        lambda: coordinator.search_similar_gpu("river"),
    ):
        with pytest.raises(EmbeddingUnavailableError):
            call()
    gpu = coordinator.get_gpu_metrics()
    stats = gpu["gpu_stats"]
    assert stats["gpu_embeddings"] == stats["cpu_embeddings"] == 0
    assert stats["gpu_searches"] == stats["cpu_searches"] == 0
    assert gpu["total_embeddings"] == 0 and gpu["failed_embeddings"] == 4
    assert "upsert" not in client.calls
    assert stats["gpu_percentage"] is None  # nothing measured yet
    assert not any("saved" in key for key in stats)
    assert gpu["device_memory"]["device"] == "pretend"


def test_raw_topk_on_device(cpu_settings):
    coordinator = _coordinator(cpu_settings)
    memories = torch.eye(5, 384)
    queries = torch.eye(2, 384)
    scores, indices = coordinator.search_similar_vectors_gpu(queries, memories, top_k=3)
    assert tuple(scores.shape) == (2, 3) and indices[:, 0].tolist() == [0, 1]
    scores, _ = coordinator.search_similar_vectors_gpu(queries, memories, top_k=99)
    assert tuple(scores.shape) == (2, 5)


def test_factory_returns_gpu_variant_when_enabled(cpu_settings):
    pretend = GPUComponentFactory(GPUIntegrationConfig(manager=PretendAccelerator()))
    built = pretend.create_vector_coordinator(
        client=qdrant_client.QdrantClient(":memory:"),
        embedder=LookupEmbedder(),
        settings=cpu_settings,
    )
    assert isinstance(built, MPSVectorCoordinator) and built.use_gpu is True
    lsm = pretend.create_lsm(n_input=3, n_reservoir=20, n_output=2, seed=1)
    assert isinstance(lsm, MPSLiquidStateMachine) and lsm.use_gpu is True
    assert pretend.config.describe()["vectors"] == "gpu"

    cpu = GPUComponentFactory(
        GPUIntegrationConfig(manager=MPSManager(settings=cpu_settings))
    )
    plain = cpu.create_vector_coordinator(
        client=qdrant_client.QdrantClient(":memory:"),
        embedder=LookupEmbedder(),
        settings=cpu_settings,
    )
    assert type(plain) is VectorCoordinator
    assert isinstance(
        cpu.create_lsm(n_input=3, n_reservoir=20, n_output=2), LiquidStateMachine
    )
    off = GPUComponentFactory(
        GPUIntegrationConfig(enable_gpu=False, manager=PretendAccelerator())
    )
    assert off.config.gpu_vectors is False  # the master switch wins


def test_gpu_coordinator_status_reports_device(cpu_settings):
    report = _coordinator(cpu_settings).status()
    assert report["backend"] == "qdrant" and report["available"] is True
    device = report["device"]
    assert device["backend"] == "torch" and device["available"] is True
    assert device["device"] == "cpu" and device["use_gpu"] is False
    assert device["last_error"] is None
