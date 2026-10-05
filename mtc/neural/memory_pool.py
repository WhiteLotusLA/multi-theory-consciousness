"""
Tensor Memory Pool
==================

Keeps frequently used tensors resident on the device the device manager
resolves (``mtc.neural.mps_utils``, which follows ``MTC_DEVICE``), so a tensor
that is reused is moved there once instead of on every use.

- ``GPUMemoryPool``: a least-recently-used cache bounded by total bytes and by
  entry count, with hit, miss and eviction counts. ``prefetch`` moves a batch
  of tensors and synchronizes once through the manager. The device is resolved
  on first use, so the module instance ``global_memory_pool`` costs nothing at
  import. The pool counts transfers it avoided; it does not estimate the time
  they would have taken.
- ``ThoughtEmbeddingCache``: a preallocated circular buffer of the most recent
  thought embeddings. Its width defaults to ``VECTOR_DIMENSIONS`` read when the
  cache is built. The cached texts are private thought content at runtime and
  live only in process memory.

Config keys: ``VECTOR_DIMENSIONS`` (the thought cache width) and, through the
device manager, ``MTC_DEVICE``. No persisted state.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import torch

from mtc.core.config import get_settings
from mtc.neural.mps_utils import device_status, mps_manager

logger = logging.getLogger(__name__)

_GB = 1024**3
_MB = 1024**2


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class GPUCacheEntry:
    """One cached tensor and its use counts."""

    tensor: torch.Tensor
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    key: str


class GPUMemoryPool:
    """LRU cache of tensors kept on the resolved device."""

    def __init__(
        self,
        max_size_gb: float = 1.0,
        max_entries: int = 1000,
        use_gpu: bool = True,
        manager: Any = None,
    ) -> None:
        self.max_size_bytes = int(max_size_gb * _GB)
        self.max_entries = int(max_entries)
        self._want_gpu = bool(use_gpu)
        self._manager = manager if manager is not None else mps_manager
        self.cache: "OrderedDict[str, GPUCacheEntry]" = OrderedDict()
        self.stats: Dict[str, int] = {
            "total_size_bytes": 0,
            "total_entries": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "transfers_avoided": 0,
        }

    @property
    def use_gpu(self) -> bool:
        """True when caching on an accelerator (requested and available)."""
        return self._want_gpu and bool(self._manager.enabled)

    @property
    def device(self) -> torch.device:
        """The device cached tensors live on."""
        return self._manager.device if self.use_gpu else torch.device("cpu")

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Return a cached tensor (marking it most recently used), or None."""
        entry = self.cache.pop(key, None)
        if entry is None:
            self.stats["misses"] += 1
            return None
        entry.last_accessed = _utcnow()
        entry.access_count += 1
        self.cache[key] = entry
        self.stats["hits"] += 1
        self.stats["transfers_avoided"] += 1
        return entry.tensor

    def put(
        self, key: str, tensor: torch.Tensor, move_to_gpu: bool = True
    ) -> torch.Tensor:
        """Cache a tensor (moved to the device when asked); return the cached one."""
        if self.use_gpu and move_to_gpu and tensor.device != self.device:
            tensor = tensor.to(self.device)
        size_bytes = tensor.element_size() * tensor.numel()
        old = self.cache.pop(key, None)
        if old is not None:
            self.stats["total_size_bytes"] -= old.size_bytes
        while self.cache and (
            self.stats["total_size_bytes"] + size_bytes > self.max_size_bytes
            or len(self.cache) >= self.max_entries
        ):
            self._evict_lru()
        now = _utcnow()
        self.cache[key] = GPUCacheEntry(
            tensor=tensor,
            created_at=now,
            last_accessed=now,
            access_count=0,
            size_bytes=size_bytes,
            key=key,
        )
        self.stats["total_size_bytes"] += size_bytes
        self.stats["total_entries"] = len(self.cache)
        return tensor

    def _evict_lru(self) -> None:
        if not self.cache:
            return
        _key, entry = self.cache.popitem(last=False)
        self.stats["total_size_bytes"] -= entry.size_bytes
        self.stats["evictions"] += 1
        self.stats["total_entries"] = len(self.cache)

    def clear(self) -> None:
        """Drop every cached tensor."""
        self.cache.clear()
        self.stats["total_size_bytes"] = 0
        self.stats["total_entries"] = 0

    def get_or_create(
        self,
        key: str,
        create_fn: Callable[[], torch.Tensor],
        move_to_gpu: bool = True,
    ) -> torch.Tensor:
        """Return the cached tensor, or create, cache and return it."""
        cached = self.get(key)
        if cached is not None:
            return cached
        return self.put(key, create_fn(), move_to_gpu)

    def prefetch(self, keys: List[str], tensors: List[torch.Tensor]) -> None:
        """Move a batch of tensors to the device, synchronize once, cache them."""
        if not self.use_gpu:
            return
        moved = [t if t.device == self.device else t.to(self.device) for t in tensors]
        self._manager.synchronize()
        for key, tensor in zip(keys, moved):
            self.put(key, tensor, move_to_gpu=False)
        logger.debug("prefetched %d tensors", len(keys))

    def get_stats(self) -> Dict[str, Any]:
        """Return the counts, the hit rate and the memory in use."""
        lookups = self.stats["hits"] + self.stats["misses"]
        return {
            **self.stats,
            "hit_rate_percent": (
                (self.stats["hits"] / lookups * 100) if lookups else None
            ),
            "size_mb": self.stats["total_size_bytes"] / _MB,
            "max_size_gb": self.max_size_bytes / _GB,
            "utilization_percent": (
                self.stats["total_size_bytes"] / self.max_size_bytes * 100
                if self.max_size_bytes
                else None
            ),
            "avg_access_per_entry": (
                sum(e.access_count for e in self.cache.values()) / len(self.cache)
                if self.cache
                else None
            ),
        }

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available``, ``last_error`` and the device.

        Never raises: a device the manager cannot provide is reported here.
        """
        return device_status(
            lambda: self.device,
            entries=len(self.cache),
            size_mb=self.stats["total_size_bytes"] / _MB,
        )

    def print_stats(self) -> None:
        """Print the statistics."""
        for key, value in self.get_stats().items():
            print(f"{key}: {value}")

    def __repr__(self) -> str:
        return (
            f"GPUMemoryPool(entries={len(self.cache)}, "
            f"size={self.stats['total_size_bytes'] / _MB:.1f}MB)"
        )


class ThoughtEmbeddingCache:
    """Circular buffer of recent thought embeddings on the resolved device."""

    def __init__(
        self,
        max_recent: int = 50,
        embedding_dim: Optional[int] = None,
        use_gpu: bool = True,
        manager: Any = None,
    ) -> None:
        self.max_recent = int(max_recent)
        self.embedding_dim = int(
            embedding_dim
            if embedding_dim is not None
            else get_settings().VECTOR_DIMENSIONS
        )
        active = manager if manager is not None else mps_manager
        self.use_gpu = bool(use_gpu) and bool(active.enabled)
        self.device = active.device if self.use_gpu else torch.device("cpu")
        self.embedding_buffer = torch.zeros(
            (self.max_recent, self.embedding_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.current_size = 0
        self.write_index = 0
        self.thought_texts: List[str] = []

    def add(self, text: str, embedding: torch.Tensor) -> None:
        """Write one embedding into the next slot (overwriting the oldest)."""
        if embedding.device != self.device:
            embedding = embedding.to(self.device)
        self.embedding_buffer[self.write_index] = embedding
        if self.current_size < self.max_recent:
            self.thought_texts.append(text)
            self.current_size += 1
        else:
            self.thought_texts[self.write_index] = text
        self.write_index = (self.write_index + 1) % self.max_recent

    def get_all_embeddings(self) -> torch.Tensor:
        """Return the filled rows of the buffer (no copy)."""
        return self.embedding_buffer[: self.current_size]

    def get_all_texts(self) -> List[str]:
        """Return the cached texts in slot order."""
        return self.thought_texts[: self.current_size]

    def clear(self) -> None:
        """Empty the cache (the buffer stays allocated)."""
        self.current_size = 0
        self.write_index = 0
        self.thought_texts.clear()

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available``, ``last_error`` and the device."""
        return device_status(
            lambda: self.device,
            embedding_dim=self.embedding_dim,
            size=self.current_size,
            max_recent=self.max_recent,
        )

    def __len__(self) -> int:
        return self.current_size

    def __repr__(self) -> str:
        return (
            f"ThoughtEmbeddingCache(device={self.device}, "
            f"size={self.current_size}/{self.max_recent})"
        )


# The shared pool; it resolves its device on first use.
global_memory_pool = GPUMemoryPool(max_size_gb=1.0, max_entries=1000, use_gpu=True)
