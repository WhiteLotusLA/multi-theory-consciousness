"""The device manager follows MTC_DEVICE; the GPU helpers use what it resolves."""

import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mtc.core import device as device_module  # noqa: E402
from mtc.core.config import Settings  # noqa: E402
from mtc.core.device import DeviceUnavailableError  # noqa: E402
from mtc.neural import gpu_integration, memory_pool, mps_utils  # noqa: E402
from mtc.neural.gpu_integration import (  # noqa: E402
    GPUComponentFactory,
    GPUIntegrationConfig,
)
from mtc.neural.memory_pool import GPUMemoryPool, ThoughtEmbeddingCache  # noqa: E402
from mtc.neural.mps_utils import MPSManager  # noqa: E402
from tests.core.test_no_hardcoded_endpoints import (  # noqa: E402
    find_endpoint_literals,
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    device_module._LOGGED.clear()
    yield tmp_path
    device_module._LOGGED.clear()


def _settings(monkeypatch, value: str) -> Settings:
    monkeypatch.setenv("MTC_DEVICE", value)
    return Settings()


def _availability(monkeypatch, mps: bool, cuda: bool) -> None:
    monkeypatch.setattr(device_module, "_mps_available", lambda t: mps)
    monkeypatch.setattr(device_module, "_cuda_available", lambda t: cuda)
    # The availability is faked, so are the memory readings a resolved manager
    # logs: no case here queries the real backend of this machine's accelerator.
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 0)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 0)


def test_manager_follows_device_setting(clean_env, monkeypatch):
    _availability(monkeypatch, mps=False, cuda=True)
    cuda = MPSManager(settings=_settings(monkeypatch, "cuda"))
    assert cuda.device.type == "cuda" and cuda.enabled is True
    auto = MPSManager(settings=_settings(monkeypatch, "auto"))
    assert auto.device.type == "cuda"  # auto keeps the MPS, CUDA, CPU order
    _availability(monkeypatch, mps=True, cuda=True)
    assert MPSManager(settings=_settings(monkeypatch, "auto")).device.type == "mps"
    pinned = MPSManager(device="cpu", settings=_settings(monkeypatch, "cuda"))
    assert pinned.device.type == "cpu"  # an argument overrides the setting
    _availability(monkeypatch, mps=False, cuda=False)
    missing = MPSManager(settings=_settings(monkeypatch, "cuda"))
    with pytest.raises(DeviceUnavailableError) as info:
        missing.device
    assert "MTC_DEVICE" in str(info.value)  # no silent fallback to the CPU
    assert "historical" in (MPSManager.__doc__ or "")


def test_module_manager_resolves_lazily(clean_env, monkeypatch):
    built = MPSManager()
    assert built._device is None  # construction resolves nothing
    _availability(monkeypatch, mps=False, cuda=False)
    monkeypatch.setattr(mps_utils, "get_settings", lambda: Settings())
    assert built.device.type == "cpu"
    assert isinstance(mps_utils.mps_manager, MPSManager)


def test_cpu_setting_disables_acceleration(clean_env, monkeypatch):
    _availability(monkeypatch, mps=True, cuda=True)
    manager = MPSManager(settings=_settings(monkeypatch, "cpu"))
    assert manager.device.type == "cpu" and manager.enabled is False
    assert manager.should_use_gpu("matmul", 10**9, is_batch=True) is False
    forced = MPSManager(force_cpu=True, settings=_settings(monkeypatch, "mps"))
    assert forced.device.type == "cpu" and forced.enabled is False
    stats = manager.get_memory_stats()
    assert stats["device"] == "cpu"
    assert stats["allocated_mb"] is None and stats["driver_mb"] is None  # unmeasured
    manager.empty_cache()
    manager.synchronize()  # no accelerator call on the CPU


def test_cuda_branch_for_cache_sync_and_memory_stats(clean_env, monkeypatch):
    _availability(monkeypatch, mps=False, cuda=True)
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *a, **k: calls.append("synchronize")
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 3 * 1024**2)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 8 * 1024**2)

    def not_mps(*args, **kwargs):
        raise AssertionError("the MPS backend was called on a CUDA device")

    monkeypatch.setattr(torch.mps, "synchronize", not_mps)
    monkeypatch.setattr(torch.mps, "empty_cache", not_mps)
    manager = MPSManager(settings=_settings(monkeypatch, "cuda"))
    manager.empty_cache()
    manager.synchronize()
    assert calls == ["empty", "synchronize"]
    stats = manager.get_memory_stats()
    assert stats["device"] == "cuda"
    assert stats["allocated_mb"] == pytest.approx(3.0)
    assert stats["driver_mb"] == pytest.approx(8.0)
    mps_utils.synchronize_device(torch.device("cuda"))
    assert calls[-1] == "synchronize"


def test_to_numpy_handles_any_accelerator(clean_env, monkeypatch):
    manager = MPSManager(settings=_settings(monkeypatch, "cpu"))

    class AcceleratorTensor:
        device = SimpleNamespace(type="cuda")

        def __init__(self):
            self.moved = False

        def detach(self):
            return self

        def cpu(self):
            self.moved = True
            return torch.tensor([1.0, 2.0])

    fake = AcceleratorTensor()
    assert manager.to_numpy(fake).tolist() == [1.0, 2.0] and fake.moved
    with_grad = torch.ones(2, requires_grad=True) * 2
    assert manager.to_numpy(with_grad).tolist() == [2.0, 2.0]


def test_thought_cache_dim_follows_vector_dimensions(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_VECTOR_DIMENSIONS", "16")
    narrow = Settings()
    monkeypatch.setattr(memory_pool, "get_settings", lambda: narrow)
    cache = ThoughtEmbeddingCache(use_gpu=False)
    assert cache.embedding_dim == 16
    assert tuple(cache.embedding_buffer.shape) == (50, 16)
    cache.add("a thought", torch.ones(16))
    assert len(cache) == 1 and cache.get_all_texts() == ["a thought"]
    assert tuple(ThoughtEmbeddingCache(4, 8, use_gpu=False).embedding_buffer.shape) == (
        4,
        8,
    )


def test_pool_synchronizes_through_the_manager(clean_env):
    class Manager:
        enabled = True
        device = torch.device("cpu")

        def __init__(self):
            self.synced = 0

        def synchronize(self):
            self.synced += 1

    manager = Manager()
    pool = GPUMemoryPool(max_size_gb=0.01, max_entries=4, manager=manager)
    pool.prefetch(["a", "b"], [torch.ones(2), torch.zeros(3)])
    assert manager.synced == 1
    assert pool.get("a") is not None and pool.get("missing") is None
    stats = pool.get_stats()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert not any("saved" in key for key in stats)  # no estimated saving
    source = Path(memory_pool.__file__).read_text(encoding="utf-8")
    assert "torch.mps.synchronize" not in source


def test_factory_collection_from_settings(clean_env, monkeypatch):
    qdrant_client = pytest.importorskip("qdrant_client")
    from mtc.neural.vectors import coordinator as coordinator_module
    from mtc.neural.vectors.coordinator import VectorCoordinator

    class Embedder:
        semantic = True
        dim = 384

        def encode(self, texts):
            out = np.zeros((len(texts), 384))
            out[:, 0] = 1.0
            return out

    monkeypatch.setenv("MTC_QDRANT_COLLECTION", "unit_collection")
    named = Settings()
    monkeypatch.setattr(coordinator_module, "get_settings", lambda: named)
    factory = GPUComponentFactory(GPUIntegrationConfig(enable_gpu=False))
    built = factory.create_vector_coordinator(
        client=qdrant_client.QdrantClient(":memory:"), embedder=Embedder()
    )
    assert type(built) is VectorCoordinator
    assert built.collection_name == "unit_collection"


def test_factory_has_no_endpoint_defaults():
    signature = inspect.signature(GPUComponentFactory.create_vector_coordinator)
    for name in ("host", "port", "collection_name"):
        assert signature.parameters[name].default is None, name
    source = Path(gpu_integration.__file__).read_text(encoding="utf-8")
    assert find_endpoint_literals(source) == []
    for claim in ("18.55", "6.8x", "1.51x", "0.53x", "faster"):
        assert claim not in source, claim


def test_manager_status_reports_device_and_unavailable_setting(clean_env, monkeypatch):
    _availability(monkeypatch, mps=False, cuda=True)
    report = MPSManager(settings=_settings(monkeypatch, "cuda")).status()
    assert report["backend"] == "torch" and report["available"] is True
    assert report["device"] == "cuda" and report["accelerated"] is True
    assert report["last_error"] is None and report["requested"] == "cuda"
    _availability(monkeypatch, mps=False, cuda=False)
    missing = MPSManager(settings=_settings(monkeypatch, "cuda"))
    report = missing.status()  # never raises
    assert report["available"] is False and report["device"] is None
    assert "MTC_DEVICE" in report["last_error"]


def test_pool_and_cache_status(clean_env, monkeypatch):
    _availability(monkeypatch, mps=False, cuda=False)
    cpu = MPSManager(settings=_settings(monkeypatch, "cpu"))
    cache = ThoughtEmbeddingCache(max_recent=4, embedding_dim=8, manager=cpu)
    report = cache.status()
    assert report["backend"] == "torch" and report["available"] is True
    assert report["device"] == "cpu" and report["last_error"] is None
    assert report["embedding_dim"] == 8 and report["size"] == 0
    pool = GPUMemoryPool(max_size_gb=0.01, manager=cpu)
    assert pool.status()["available"] is True and pool.status()["entries"] == 0
    missing = MPSManager(settings=_settings(monkeypatch, "mps"))
    report = GPUMemoryPool(manager=missing).status()
    assert report["available"] is False and "MTC_DEVICE" in report["last_error"]
