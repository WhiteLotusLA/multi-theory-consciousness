"""
Integrated Memory System
========================

Writes a memory twice, once as a vector (semantic search) and once as a thought
document (rich neural metadata), and links the two. The vector store is anything
implementing the ``VectorStore`` protocol (``VectorCoordinator``,
``InMemoryVectorStore``); the thought documents go through ``NeuralDataManager``.

- ``store_memory`` embeds the text once, in the vector store, and returns the
  point id and the thought id with the measured latency of each write. The
  thought's ``stream_id`` is ``thought_<point id>`` and its
  ``associated_memories`` lists the point id, so the link works in both
  directions through an indexed field. When the data manager cannot store the
  thought (an unreachable server) the result says so (``thought_id`` None,
  ``thought_stored`` False) and the vector stays. When the vector store fails,
  the call raises and nothing is written to MongoDB.
- ``search_memories`` searches by meaning. ``emotional_filter`` names an emotion
  and keeps only the hits whose stored ``emotional_state`` has that emotion at
  or above ``emotion_threshold``; the search is widened (``OVERFETCH`` times
  ``top_k``) before filtering so the filter does not starve the result.
- ``get_memory_with_neural_context`` and ``get_recent_memories_with_emotion``
  combine a vector payload with its thought document.
- ``get_metrics`` reports measured latencies; every figure is None until a
  write was timed, and ``latency_target_ms`` is only the line a write is
  compared with.

A missing store is named in a ``StoreUnavailable`` raised by the first call that
needs it. Both stores are synchronous; async methods run them through
``asyncio.to_thread``. Config keys: none read here (the stores read their own).
Persisted state: the vector point and the thought document; thought texts are
private conversation content wherever they are stored.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from mtc.memory.protocols import StoreUnavailable, VectorStore
from mtc.neural.mongodb_schemas import NeuralDataManager, ThoughtStream

logger = logging.getLogger(__name__)

# A filtered search asks the vector store for this many times ``top_k`` hits.
OVERFETCH = 5


class IntegratedMemorySystem:
    """Dual write of memories to a vector store and a thought collection.

    Args:
        vector_store: The vector store (``VectorStore`` protocol).
        data_manager: The neural data manager holding the thought documents.
        latency_target_ms: The latency a dual write is compared with; the
            metrics count writes within and beyond it.
        clock: A monotonic clock in seconds (tests inject one).
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore],
        data_manager: Optional[NeuralDataManager],
        latency_target_ms: float = 50.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.vector_store = vector_store
        self.data_manager = data_manager
        self.latency_target_ms = float(latency_target_ms)
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._writes = 0
        self._latency_sum = 0.0
        self._vector_sum = 0.0
        self._thought_sum = 0.0
        self._max_latency: Optional[float] = None
        self._min_latency: Optional[float] = None
        self._met = 0
        self._missed = 0
        self._thoughts_skipped = 0

    # ---- helpers ------------------------------------------------------

    def _vectors(self) -> VectorStore:
        if self.vector_store is None:
            raise StoreUnavailable(
                "the integrated memory system has no vector store configured"
            )
        return self.vector_store

    def _thoughts(self) -> NeuralDataManager:
        if self.data_manager is None:
            raise StoreUnavailable(
                "the integrated memory system has no data manager configured"
            )
        return self.data_manager

    @staticmethod
    def _stream_id(point_id: str) -> str:
        return f"thought_{point_id}"

    # ---- writing ------------------------------------------------------

    async def store_memory(
        self,
        text: str,
        emotional_state: Optional[Dict[str, float]] = None,
        thought_type: str = "memory",
        importance: float = 1.0,
        triggers: Optional[List[str]] = None,
        parent_thought_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a memory as a vector and as a thought and link them.

        ``emotional_weight`` of the vector is the mean intensity of
        ``emotional_state`` (None without one: nothing was measured).
        Returns ``point_id``, ``stream_id``, ``thought_id`` (None when the
        thought was not stored), ``thought_stored``, ``vector_ms``,
        ``thought_ms``, ``latency_ms`` and ``target_met``.
        """
        vectors, thoughts = self._vectors(), self._thoughts()
        emotions = dict(emotional_state or {})
        weight = sum(emotions.values()) / len(emotions) if emotions else None
        started = self._clock()

        point_id = await asyncio.to_thread(
            vectors.store_memory,
            text,
            metadata={"emotional_state": emotions, "thought_type": thought_type},
            importance=importance,
            emotional_weight=weight,
        )
        vector_done = self._clock()

        stream_id = self._stream_id(point_id)
        thought = ThoughtStream(
            timestamp=datetime.now(timezone.utc),
            stream_id=stream_id,
            thought_text=text,
            thought_type=thought_type,
            emotional_state=emotions,
            associated_memories=[point_id],
            triggers=list(triggers or []),
            parent_thought_id=parent_thought_id,
            metadata={"importance": importance, "vector_point_id": point_id},
        )
        thought_id = await asyncio.to_thread(thoughts.store_thought, thought)
        finished = self._clock()

        vector_ms = (vector_done - started) * 1000.0
        thought_ms = (finished - vector_done) * 1000.0
        total_ms = (finished - started) * 1000.0
        met = total_ms < self.latency_target_ms
        self._record(vector_ms, thought_ms, total_ms, met, thought_id is not None)
        return {
            "point_id": point_id,
            "stream_id": stream_id,
            "thought_id": thought_id,
            "thought_stored": thought_id is not None,
            "vector_ms": vector_ms,
            "thought_ms": thought_ms,
            "latency_ms": total_ms,
            "target_met": met,
        }

    def _record(
        self,
        vector_ms: float,
        thought_ms: float,
        total_ms: float,
        met: bool,
        thought_stored: bool,
    ) -> None:
        self._writes += 1
        self._latency_sum += total_ms
        self._vector_sum += vector_ms
        self._thought_sum += thought_ms
        self._max_latency = (
            total_ms if self._max_latency is None else max(self._max_latency, total_ms)
        )
        self._min_latency = (
            total_ms if self._min_latency is None else min(self._min_latency, total_ms)
        )
        if met:
            self._met += 1
        else:
            self._missed += 1
        if not thought_stored:
            self._thoughts_skipped += 1

    # ---- reading ------------------------------------------------------

    async def search_memories(
        self,
        query: str,
        top_k: int = 5,
        emotional_filter: Optional[str] = None,
        *,
        min_score: Optional[float] = None,
        emotion_threshold: float = 0.5,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search by meaning; return ``(point_id, score, payload)`` best first.

        ``emotional_filter`` keeps the hits whose ``emotional_state`` has that
        emotion at or above ``emotion_threshold``. ``min_score`` overrides the
        vector store's default score floor.
        """
        vectors = self._vectors()
        wanted = top_k * OVERFETCH if emotional_filter else top_k
        arguments: Dict[str, Any] = {"top_k": wanted}
        if min_score is not None:
            arguments["min_score"] = min_score
        hits = await asyncio.to_thread(
            lambda: vectors.search_similar(query, **arguments)
        )
        if emotional_filter:
            hits = [
                hit
                for hit in hits
                if float(
                    (hit[2].get("emotional_state") or {}).get(emotional_filter, 0.0)
                )
                >= emotion_threshold
            ]
        return list(hits[: max(top_k, 0)])

    async def get_memory_with_neural_context(
        self, point_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return a vector payload with its thought document, or None when the
        vector store has no such point. ``neural_context`` is None when no
        thought is linked to the point."""
        vectors, thoughts = self._vectors(), self._thoughts()
        payload = await asyncio.to_thread(vectors.get_memory_by_id, point_id)
        if not payload:
            return None
        context = await asyncio.to_thread(
            thoughts.get_thought_by_stream_id, self._stream_id(point_id)
        )
        return {"vector_data": payload, "neural_context": context, "point_id": point_id}

    async def get_recent_memories_with_emotion(
        self, emotion: str, threshold: float = 0.5, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return the newest memories whose thought has an emotion at or above a
        threshold, each with its vector payload (thoughts whose vector point is
        gone are left out)."""
        thoughts = self._thoughts()
        found = await asyncio.to_thread(
            thoughts.get_thoughts_by_emotion, emotion, threshold, limit
        )
        results: List[Dict[str, Any]] = []
        for thought in found:
            linked = thought.get("associated_memories") or []
            if not linked:
                continue
            memory = await self.get_memory_with_neural_context(str(linked[0]))
            if memory is not None:
                results.append(memory)
        return results

    # ---- reporting ------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return the measured write latencies in milliseconds.

        Every figure is None until a write has been timed.
        """
        count = self._writes
        return {
            "total_writes": count,
            "avg_latency_ms": self._latency_sum / count if count else None,
            "avg_vector_ms": self._vector_sum / count if count else None,
            "avg_thought_ms": self._thought_sum / count if count else None,
            "max_latency_ms": self._max_latency,
            "min_latency_ms": self._min_latency,
            "latency_target_ms": self.latency_target_ms,
            "target_met_count": self._met,
            "target_missed_count": self._missed,
            "target_met_rate": self._met / count if count else None,
            "thoughts_not_stored": self._thoughts_skipped,
        }

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available`` and ``last_error`` plus both stores'
        own status (a store that is not configured reads as None).

        ``available`` is False when a store is missing or reports False, None
        when a store does not know yet (the data manager before its first
        contact) and none reports False, and True otherwise.
        """
        parts: Dict[str, Any] = {}
        errors: List[str] = []
        available: Optional[bool] = (
            self.vector_store is not None and self.data_manager is not None
        )
        unknown = False
        for name, store in (
            ("vector_store", self.vector_store),
            ("data_manager", self.data_manager),
        ):
            report = store.status() if hasattr(store, "status") else None
            parts[name] = report
            if isinstance(report, dict):
                reported = report.get("available", True)
                if reported is None:
                    unknown = True
                else:
                    available = available and bool(reported)
                if report.get("last_error"):
                    errors.append(f"{name}: {report['last_error']}")
        if available and unknown:
            available = None
        return {
            "backend": "integrated",
            "available": available,
            "last_error": "; ".join(errors) or None,
            **parts,
        }
