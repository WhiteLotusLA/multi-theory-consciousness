"""
GPU Vector Coordinator
======================

``MPSVectorCoordinator`` extends ``VectorCoordinator`` with tensor paths on the
device the device manager resolves (``mtc.neural.mps_utils``, which follows
``MTC_DEVICE``; the class name is historical and CUDA works as well):

- ``embed_memory_batch_gpu`` embeds a batch in one embedder call and returns a
  tensor; batches of at least ``gpu_batch_threshold`` texts are placed on the
  accelerator when one is active, smaller ones stay on the CPU.
- ``search_similar_gpu`` honours ``use_gpu_search``: when it is true and an
  accelerator is active the query embedding goes through the accelerated
  path, otherwise through the CPU; the nearest-neighbour search itself is
  Qdrant's HNSW index either way.
- ``search_similar_vectors_gpu`` is an exact cosine top-k between tensors the
  caller holds, on the accelerator when the operation is large enough.
- ``store_memories_batch`` embeds a batch and writes it in one upsert with
  UUID point ids.

Everything the base class promises holds: no mock path (an embedder failure
raises and writes nothing), no projection, ``StoreUnavailable`` for an
unreachable server, and the same config keys and schema artifact. The GPU
counters count operations that completed, by the path they took; a failed
embedding is never counted as a real one, and no time saving is estimated.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mtc.core.config import Settings
from mtc.core.embeddings import Embedder
from mtc.neural.mps_utils import cosine_similarity_gpu, device_status, mps_manager
from mtc.neural.vectors.coordinator import VectorCoordinator

logger = logging.getLogger(__name__)


class MPSVectorCoordinator(VectorCoordinator):
    """Vector coordinator with batch embedding and top-k on the device."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        embedder: Optional[Embedder] = None,
        client: Any = None,
        allow_non_semantic: bool = False,
        create_collection: bool = True,
        settings: Optional[Settings] = None,
        use_gpu: bool = True,
        gpu_batch_threshold: int = 5,
        manager: Any = None,
    ) -> None:
        super().__init__(
            host=host,
            port=port,
            collection_name=collection_name,
            vector_size=vector_size,
            embedder=embedder,
            client=client,
            allow_non_semantic=allow_non_semantic,
            create_collection=create_collection,
            settings=settings,
        )
        self._manager = manager if manager is not None else mps_manager
        self.use_gpu = bool(use_gpu) and bool(self._manager.enabled)
        self.gpu_batch_threshold = max(1, int(gpu_batch_threshold))
        self.device = self._manager.device if self.use_gpu else torch.device("cpu")
        self.gpu_metrics: Dict[str, Any] = {
            "gpu_embeddings": 0,
            "cpu_embeddings": 0,
            "gpu_searches": 0,
            "cpu_searches": 0,
            "last_search_path": None,
        }
        logger.info("vector coordinator tensor path on %s", self.device)

    def _place(self, vectors: np.ndarray, accelerated: bool) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32))
        if accelerated:
            tensor = tensor.to(self.device)
            self._manager.synchronize()
        return tensor

    @torch.no_grad()
    def embed_memory_batch_gpu(self, texts: Sequence[str]) -> torch.Tensor:
        """Embed texts in one call; return an ``(n, vector_size)`` tensor.

        Batches of at least ``gpu_batch_threshold`` texts land on the
        accelerator when one is active.
        """
        items = list(texts)
        if not items:
            return torch.zeros((0, self.vector_size), dtype=torch.float32)
        vectors = self._encode(items)
        accelerated = self.use_gpu and len(items) >= self.gpu_batch_threshold
        tensor = self._place(vectors, accelerated)
        key = "gpu_embeddings" if accelerated else "cpu_embeddings"
        self.gpu_metrics[key] += len(items)
        return tensor

    @torch.no_grad()
    def search_similar_gpu(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.7,
        use_gpu_search: bool = True,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search like ``search_similar``; ``use_gpu_search`` picks the path."""
        limit = top_k if top_k is not None else self._active().VECTOR_SEARCH_LIMIT
        if limit <= 0:
            return []
        accelerated = bool(use_gpu_search) and self.use_gpu
        tensor = self._place(self._encode([query]), accelerated)
        vector = self._manager.to_numpy(tensor)[0]
        results = self._query(vector, limit, min_score)
        self.gpu_metrics["gpu_searches" if accelerated else "cpu_searches"] += 1
        self.gpu_metrics["last_search_path"] = "gpu" if accelerated else "cpu"
        return results

    @torch.no_grad()
    def search_similar_vectors_gpu(
        self,
        query_vectors: torch.Tensor,
        memory_vectors: torch.Tensor,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact cosine top-k of each query row over the memory rows.

        Returns ``(scores, indices)`` of shape ``(queries, min(top_k, memories))``.
        """
        size = int(query_vectors.shape[0]) * int(memory_vectors.shape[0])
        if self.use_gpu and self._manager.should_use_gpu("similarity_search", size):
            target = self.device
        else:
            target = torch.device("cpu")
        started = time.perf_counter()
        similarity = cosine_similarity_gpu(
            query_vectors.to(target), memory_vectors.to(target)
        )
        k = min(int(top_k), int(memory_vectors.shape[0]))
        scores, indices = torch.topk(similarity, k=k, dim=1)
        if target.type != "cpu":
            self._manager.synchronize()
        logger.debug(
            "top-%d over %d memories on %s in %.2f ms",
            k,
            int(memory_vectors.shape[0]),
            target,
            (time.perf_counter() - started) * 1000,
        )
        return scores, indices

    def store_memories_batch(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        importances: Optional[Sequence[float]] = None,
    ) -> List[str]:
        """Embed a batch and store it in one upsert; return the UUID point ids."""
        items = list(texts)
        count = len(items)
        if count == 0:
            return []
        metas = self._per_item(metadatas, count, "metadatas")
        weights = self._per_item(importances, count, "importances")
        tensor = self.embed_memory_batch_gpu(items)
        vectors = self._manager.to_numpy(tensor)
        return self._upsert_vectors(
            items, vectors, metas, weights, [None] * count, [None] * count
        )

    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Return the base metrics plus the per-path counts and device memory."""
        stats = dict(self.gpu_metrics)
        accelerated = stats["gpu_embeddings"] + stats["gpu_searches"]
        total = accelerated + stats["cpu_embeddings"] + stats["cpu_searches"]
        stats.update(
            {
                "gpu_enabled": self.use_gpu,
                "device": str(self.device),
                "total_operations": total,
                "gpu_percentage": accelerated / total * 100 if total else None,
            }
        )
        return {
            **self.get_metrics(),
            "gpu_stats": stats,
            "device_memory": self._manager.get_memory_stats() if self.use_gpu else {},
        }

    def status(self) -> Dict[str, Any]:
        """Return the base store status plus the tensor path's device report."""
        report = super().status()
        report["device"] = device_status(lambda: self.device, use_gpu=self.use_gpu)
        return report

    def __repr__(self) -> str:
        return (
            f"MPSVectorCoordinator(collection={self.collection_name!r}, "
            f"device={self.device}, vectors={self.vector_size}D)"
        )
