"""
In-Memory Stores
================

Default implementations of the protocols in ``mtc.memory.protocols``. Each
store keeps at most ``max_items`` entries (default 10000) and drops the oldest
first. Documents and records are deep-copied on write and on read, so callers
never share state with the store. Nothing is persisted: a process restart
starts empty. Keyed stores (beliefs, speaker models, goals) move an upserted
key to the newest position, which makes "newest first" mean most recently
written. The documents hold real people's names at runtime and are private
data.

The working-memory, conversation-turn, thought-gist, retention, vector and graph
stores follow the same rules: deep copies on write and read, timezone-aware UTC
timestamps, an injectable clock where time matters, and ``status()`` reporting
``backend: "memory"``. ``InMemoryWorkingMemory`` reads ``WORKING_MEMORY_CAPACITY``
(clamped to five to nine) when no capacity is passed; ``InMemoryVectorStore``
reads ``VECTOR_SEARCH_LIMIT`` for a default ``top_k`` and refuses an embedder
whose ``semantic`` attribute is not true unless ``allow_non_semantic=True``, so
no hashed or random vector passes for a meaning. The episodic store has its own
in-memory implementation next to its schema.

``InMemorySafetyAuditStore``, ``InMemoryAlertStore`` and
``InMemoryPrivacyAuditStore`` take ``(max_items, clock)``; the clock stamps a
document written without a ``timestamp`` and an alert's ``acknowledged_at``.
They read newest first by timestamp and reject a second document with an id
already stored. Their documents hold proposed actions, alert context, user ids
and access records, all private data.
"""

import copy
import logging
import threading
import uuid
from collections import OrderedDict, deque
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from mtc.core.config import get_settings
from mtc.core.embeddings import Embedder, EmbeddingUnavailableError
from mtc.memory.protocols import (
    CACHE_TTL_S,
    EMOTION_STATE_TTL_S,
    TELEMETRY_COLLECTIONS,
    WORKING_MEMORY_TTL_S,
    SnapshotEvent,
    SnapshotRecord,
    clamp_working_capacity,
    normalize_safety_audit,
    severities_at_or_above,
)
from mtc.memory.thought_gist import ThoughtGist

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITEMS = 10000


def _aware(moment: datetime) -> datetime:
    """Return the instant in UTC; a naive datetime raises ValueError."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(timezone.utc)


Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _check_cap(max_items: int) -> int:
    if max_items < 1:
        raise ValueError("max_items must be at least 1")
    return int(max_items)


class _KeyedDocs:
    """Keyed documents, most recently written last, oldest dropped at the cap."""

    def __init__(self, max_items: int) -> None:
        self._max_items = _check_cap(max_items)
        self._docs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def put(self, key: str, doc: Dict[str, Any]) -> None:
        self._docs.pop(key, None)
        self._docs[key] = copy.deepcopy(doc)
        while len(self._docs) > self._max_items:
            self._docs.popitem(last=False)

    def newest_first(self) -> List[Dict[str, Any]]:
        return [copy.deepcopy(doc) for doc in reversed(self._docs.values())]

    def delete(self, key: str) -> bool:
        return self._docs.pop(key, None) is not None


class _DocLog:
    """Append-only documents, newest first on read, oldest dropped at the cap."""

    def __init__(self, max_items: int) -> None:
        self._docs: Deque[Dict[str, Any]] = deque(maxlen=_check_cap(max_items))

    def append(self, doc: Dict[str, Any]) -> None:
        self._docs.append(copy.deepcopy(doc))

    def newest_first(self, limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        items = list(self._docs)[::-1][:limit]
        return [copy.deepcopy(doc) for doc in items]


class InMemorySnapshotStore:
    """Snapshots and events held in process memory."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        cap = _check_cap(max_items)
        self._snapshots: Deque[SnapshotRecord] = deque(maxlen=cap)
        self._events: Deque[SnapshotEvent] = deque(maxlen=cap)
        self._next_snapshot_id = 1
        self._next_event_id = 1

    async def add_snapshot(self, record: SnapshotRecord) -> int:
        stamp = _aware(record.timestamp)
        stored = replace(
            copy.deepcopy(record), timestamp=stamp, id=self._next_snapshot_id
        )
        self._next_snapshot_id += 1
        self._snapshots.append(stored)
        return stored.id

    async def add_event(self, event: SnapshotEvent) -> int:
        stamp = _aware(event.timestamp)
        stored = replace(copy.deepcopy(event), timestamp=stamp, id=self._next_event_id)
        self._next_event_id += 1
        self._events.append(stored)
        return stored.id

    @staticmethod
    def _newest_first(items: Sequence[Any]) -> List[Any]:
        # Stable sort keeps later insertions first among equal timestamps.
        return sorted(reversed(list(items)), key=lambda x: x.timestamp, reverse=True)

    async def latest_snapshot(self) -> Optional[SnapshotRecord]:
        ordered = self._newest_first(self._snapshots)
        return copy.deepcopy(ordered[0]) if ordered else None

    async def snapshots(
        self, since: Optional[datetime] = None, limit: int = 100
    ) -> List[SnapshotRecord]:
        floor = _aware(since) if since is not None else None
        ordered = self._newest_first(self._snapshots)
        if floor is not None:
            ordered = [r for r in ordered if r.timestamp >= floor]
        return [copy.deepcopy(r) for r in ordered[: max(limit, 0)]]

    async def events(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
        event_type: Optional[str] = None,
    ) -> List[SnapshotEvent]:
        floor = _aware(since) if since is not None else None
        ordered = self._newest_first(self._events)
        if floor is not None:
            ordered = [e for e in ordered if e.timestamp >= floor]
        if event_type is not None:
            ordered = [e for e in ordered if e.event_type == event_type]
        return [copy.deepcopy(e) for e in ordered[: max(limit, 0)]]

    async def snapshot_before(self, moment: datetime) -> Optional[SnapshotRecord]:
        limit = _aware(moment)
        for record in self._newest_first(self._snapshots):
            if record.timestamp < limit:
                return copy.deepcopy(record)
        return None


class InMemoryLearningStore:
    """Experiences and beliefs held in process memory."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        self._experiences = _DocLog(max_items)
        self._beliefs = _KeyedDocs(max_items)

    async def add_experience(self, doc: Dict[str, Any]) -> None:
        self._experiences.append(doc)

    async def recent_experiences(self, limit: int) -> List[Dict[str, Any]]:
        return self._experiences.newest_first(limit)

    async def upsert_belief(self, belief_id: str, doc: Dict[str, Any]) -> None:
        self._beliefs.put(belief_id, doc)

    async def load_beliefs(self, limit: int) -> List[Dict[str, Any]]:
        return self._beliefs.newest_first()[: max(limit, 0)]


class InMemorySpeakerModelStore:
    """Speaker models held in process memory, keyed by speaker name."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        self._models = _KeyedDocs(max_items)

    async def upsert_model(self, speaker_name: str, doc: Dict[str, Any]) -> None:
        self._models.put(speaker_name, doc)

    async def load_models(self) -> List[Dict[str, Any]]:
        return self._models.newest_first()

    async def delete_model(self, speaker_name: str) -> bool:
        return self._models.delete(speaker_name)


class InMemoryGoalStore:
    """Goals held in process memory, keyed by goal id."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        self._goals = _KeyedDocs(max_items)

    async def upsert_goal(self, goal_id: str, doc: Dict[str, Any]) -> None:
        self._goals.put(goal_id, doc)

    async def load_goals(self, statuses: Sequence[str]) -> List[Dict[str, Any]]:
        wanted = set(statuses)
        return [g for g in self._goals.newest_first() if g.get("status") in wanted]


class InMemoryAttentionGrabStore:
    """Attention-capture journal entries held in process memory."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        self._grabs = _DocLog(max_items)

    async def add_grab(self, doc: Dict[str, Any]) -> None:
        self._grabs.append(doc)

    async def recent_grabs(self, limit: int) -> List[Dict[str, Any]]:
        return self._grabs.newest_first(limit)


class InMemoryDreamStore:
    """Dream records held in process memory."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        self._dreams = _DocLog(max_items)

    async def add_dream(self, doc: Dict[str, Any]) -> None:
        self._dreams.append(doc)

    async def recent_dreams(self, limit: int) -> List[Dict[str, Any]]:
        return self._dreams.newest_first(limit)


class InMemoryWorkingMemory:
    """Per-session working memory with the Redis key lifetimes, in process.

    The conversation buffer keeps the newest ``capacity`` messages (five to
    nine) and expires 1800 s after the last push; the emotional state expires
    after 14400 s, session context after its optional ttl and cache entries
    after their ttl (30 s default). Expiry follows the injected clock.
    """

    def __init__(
        self, capacity: Optional[int] = None, clock: Optional[Clock] = None
    ) -> None:
        wanted = (
            capacity if capacity is not None else get_settings().WORKING_MEMORY_CAPACITY
        )
        self.capacity = clamp_working_capacity(wanted)
        if self.capacity != wanted:
            logger.warning(
                "working memory capacity %s outside 5-9; using %s",
                wanted,
                self.capacity,
            )
        self._clock: Clock = clock or _utc_now
        self._messages: Dict[str, Tuple[List[Dict[str, Any]], datetime]] = {}
        self._emotions: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        self._contexts: Dict[str, Tuple[Dict[str, Any], Optional[datetime]]] = {}
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._metrics = {
            "total_reads": 0,
            "total_writes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }

    def _expired(self, expires_at: Optional[datetime]) -> bool:
        return expires_at is not None and self._clock() >= expires_at

    def _later(self, seconds: float) -> datetime:
        return self._clock() + timedelta(seconds=seconds)

    def _read(self, hit: bool) -> None:
        self._metrics["total_reads"] += 1
        self._metrics["cache_hits" if hit else "cache_misses"] += 1

    async def add_to_working_memory(
        self, session_id: str, message: Dict[str, Any]
    ) -> int:
        stored = copy.deepcopy(message)
        stored.setdefault("timestamp", self._clock().isoformat())
        entry = self._messages.get(session_id)
        buffer = [] if entry is None or self._expired(entry[1]) else entry[0]
        buffer.append(stored)
        del buffer[: max(0, len(buffer) - self.capacity)]
        self._messages[session_id] = (buffer, self._later(WORKING_MEMORY_TTL_S))
        self._metrics["total_writes"] += 1
        return len(buffer)

    async def get_working_memory(self, session_id: str) -> List[Dict[str, Any]]:
        entry = self._messages.get(session_id)
        if entry is None or self._expired(entry[1]):
            self._messages.pop(session_id, None)
            self._read(False)
            return []
        self._read(bool(entry[0]))
        return copy.deepcopy(entry[0])

    async def clear_working_memory(self, session_id: str) -> None:
        self._messages.pop(session_id, None)

    async def update_emotional_state(
        self, session_id: str, state: Dict[str, Any]
    ) -> None:
        stored = copy.deepcopy(state)
        stored["updated_at"] = self._clock().isoformat()
        self._emotions[session_id] = (stored, self._later(EMOTION_STATE_TTL_S))
        self._metrics["total_writes"] += 1

    async def get_emotional_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        entry = self._emotions.get(session_id)
        if entry is None or self._expired(entry[1]):
            self._emotions.pop(session_id, None)
            self._read(False)
            return None
        self._read(True)
        return copy.deepcopy(entry[0])

    async def set_session_context(
        self, session_id: str, context: Dict[str, Any], ttl: Optional[int] = None
    ) -> None:
        expires = self._later(ttl) if ttl else None
        self._contexts[session_id] = (copy.deepcopy(context), expires)
        self._metrics["total_writes"] += 1

    async def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        entry = self._contexts.get(session_id)
        if entry is None or self._expired(entry[1]):
            self._contexts.pop(session_id, None)
            self._read(False)
            return None
        self._read(True)
        return copy.deepcopy(entry[0])

    async def delete_session(self, session_id: str) -> None:
        self._messages.pop(session_id, None)
        self._emotions.pop(session_id, None)
        self._contexts.pop(session_id, None)

    async def cache_set(self, key: str, value: Any, ttl: int = CACHE_TTL_S) -> None:
        self._cache[key] = (copy.deepcopy(value), self._later(ttl))
        self._metrics["total_writes"] += 1

    async def cache_get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None or self._expired(entry[1]):
            self._cache.pop(key, None)
            self._read(False)
            return None
        self._read(True)
        return copy.deepcopy(entry[0])

    async def get_metrics(self) -> Dict[str, Any]:
        hits = self._metrics["cache_hits"]
        accesses = hits + self._metrics["cache_misses"]
        return {
            **self._metrics,
            "cache_hit_rate_percent": (hits / accesses * 100.0) if accesses else 0.0,
        }

    async def disconnect(self) -> None:
        return None

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "memory",
            "available": True,
            "last_error": None,
            "capacity": self.capacity,
        }


class InMemoryConversationTurnStore:
    """Conversation turns and sessions held in process memory (private)."""

    def __init__(
        self, max_items: int = DEFAULT_MAX_ITEMS, clock: Optional[Clock] = None
    ) -> None:
        self._turns: Deque[Dict[str, Any]] = deque(maxlen=_check_cap(max_items))
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._clock: Clock = clock or _utc_now

    async def store_conversation_turn(
        self,
        *,
        conversation_id: str,
        speaker: str,
        role: str,
        content: str,
        turn_number: int,
        emotional_context: Optional[Dict[str, Any]] = None,
        neural_state_refs: Optional[Dict[str, Any]] = None,
        topics: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        turn_id = uuid.uuid4().hex
        self._turns.append(
            copy.deepcopy(
                {
                    "turn_id": turn_id,
                    "conversation_id": conversation_id,
                    "speaker": speaker,
                    "role": role,
                    "content": content,
                    "turn_number": turn_number,
                    "emotional_context": emotional_context,
                    "neural_state_refs": neural_state_refs,
                    "topics": topics,
                    "metadata": metadata,
                    "timestamp": self._clock(),
                }
            )
        )
        return turn_id

    async def start_conversation_session(
        self, conversation_id: str, primary_participant: str, **kwargs: Any
    ) -> str:
        session_id = uuid.uuid4().hex
        self._sessions[conversation_id] = copy.deepcopy(
            {
                **kwargs,
                "session_id": session_id,
                "conversation_id": conversation_id,
                "primary_participant": primary_participant,
                "started_at": self._clock(),
                "ended_at": None,
            }
        )
        return session_id

    async def end_conversation_session(
        self, conversation_id: str, **kwargs: Any
    ) -> None:
        session = self._sessions.get(conversation_id)
        if session is None:
            return
        session.update(copy.deepcopy(kwargs))
        session["ended_at"] = self._clock()

    async def get_conversation_turns(
        self, conversation_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        turns = [t for t in self._turns if t["conversation_id"] == conversation_id]
        turns.sort(key=lambda t: t["turn_number"])
        return [copy.deepcopy(t) for t in turns[-limit:]]

    def get_session(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Return a copy of the session record (not part of the protocol)."""
        session = self._sessions.get(conversation_id)
        return copy.deepcopy(session) if session is not None else None

    def status(self) -> Dict[str, Any]:
        return {"backend": "memory", "available": True, "last_error": None}


_GIST_FIELDS = {f.name for f in fields(ThoughtGist)} - {"gist_id"}


class InMemoryThoughtGistStore:
    """Thought gists held in process memory, keyed by ``gist_id``."""

    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS) -> None:
        self._max_items = _check_cap(max_items)
        self._gists: "OrderedDict[str, ThoughtGist]" = OrderedDict()

    async def add_gist(self, gist: ThoughtGist) -> str:
        """Store a gist; an existing ``gist_id`` is replaced."""
        self._gists.pop(gist.gist_id, None)
        self._gists[gist.gist_id] = copy.deepcopy(gist)
        while len(self._gists) > self._max_items:
            self._gists.popitem(last=False)
        return gist.gist_id

    async def active_gists(
        self, min_retention: float = 0.2, limit: int = 100
    ) -> List[ThoughtGist]:
        kept = [
            g for g in self._gists.values() if g.retention_strength >= min_retention
        ]
        kept.sort(key=lambda g: (g.importance_score, g.created_at), reverse=True)
        return [copy.deepcopy(g) for g in kept[: max(limit, 0)]]

    async def update_gist(self, gist_id: str, fields: Dict[str, Any]) -> bool:
        unknown = set(fields) - _GIST_FIELDS
        if unknown:
            raise ValueError(f"unknown gist fields: {sorted(unknown)}")
        current = self._gists.get(gist_id)
        if current is None:
            return False
        updated = copy.deepcopy(current)
        for name, value in fields.items():
            setattr(updated, name, copy.deepcopy(value))
        updated.__post_init__()  # re-check the timestamps before storing
        self._gists[gist_id] = updated
        return True

    def expire(self, min_retention: float, dry_run: bool) -> int:
        """Remove unprotected gists below the floor; used by the retention store."""
        doomed = [
            gist_id
            for gist_id, gist in self._gists.items()
            if not gist.is_protected and gist.retention_strength < min_retention
        ]
        if not dry_run:
            for gist_id in doomed:
                del self._gists[gist_id]
        return len(doomed)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "memory",
            "available": True,
            "last_error": None,
            "gists": len(self._gists),
        }


class InMemoryNeuralRetentionStore:
    """Age-based pruning over telemetry and thought documents held in memory.

    Gists are shared with the ``InMemoryThoughtGistStore`` passed in, so
    expiring a gist here removes it from that store. ``add_telemetry`` and
    ``add_thought`` seed documents (they are not part of the protocol; a real
    backend is written to by the substrate code).
    """

    def __init__(self, gists: Optional[InMemoryThoughtGistStore] = None) -> None:
        self.gists = gists if gists is not None else InMemoryThoughtGistStore()
        self._telemetry: Dict[str, List[Dict[str, Any]]] = {
            name: [] for name in TELEMETRY_COLLECTIONS
        }
        self._thoughts: List[Dict[str, Any]] = []

    def add_telemetry(self, collection: str, doc: Dict[str, Any]) -> None:
        if collection not in self._telemetry:
            raise ValueError(f"unknown telemetry collection {collection!r}")
        stored = copy.deepcopy(doc)
        stored["timestamp"] = _aware(stored["timestamp"])
        self._telemetry[collection].append(stored)

    def add_thought(self, doc: Dict[str, Any]) -> None:
        stored = copy.deepcopy(doc)
        stored["timestamp"] = _aware(stored["timestamp"])
        self._thoughts.append(stored)

    async def prune_telemetry(
        self, older_than: datetime, dry_run: bool
    ) -> Dict[str, int]:
        cutoff = _aware(older_than)
        counts: Dict[str, int] = {}
        for name, docs in self._telemetry.items():
            old = [d for d in docs if d["timestamp"] < cutoff]
            counts[name] = len(old)
            if not dry_run and old:
                self._telemetry[name] = [d for d in docs if d["timestamp"] >= cutoff]
        return counts

    async def prune_thoughts(
        self, older_than: datetime, max_coherence: float, dry_run: bool
    ) -> int:
        cutoff = _aware(older_than)

        def doomed(doc: Dict[str, Any]) -> bool:
            coherence = doc.get("coherence_score")
            return (
                doc["timestamp"] < cutoff
                and coherence is not None
                and coherence < max_coherence
                and not doc.get("child_thoughts")
            )

        old = [d for d in self._thoughts if doomed(d)]
        if not dry_run and old:
            self._thoughts = [d for d in self._thoughts if not doomed(d)]
        return len(old)

    async def expire_gists(self, min_retention: float, dry_run: bool) -> int:
        return self.gists.expire(min_retention, dry_run)

    def status(self) -> Dict[str, Any]:
        return {"backend": "memory", "available": True, "last_error": None}


class InMemoryVectorStore:
    """Cosine-similarity search over the injected embedder's vectors.

    The embedder must be semantic; a non-semantic one (the hash embedder) is
    refused with ``EmbeddingUnavailableError`` unless ``allow_non_semantic`` is
    true, which only tests and demonstrations pass. An embedder failure
    propagates and nothing is stored.
    """

    def __init__(
        self,
        embedder: Embedder,
        allow_non_semantic: bool = False,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> None:
        if not getattr(embedder, "semantic", False) and not allow_non_semantic:
            raise EmbeddingUnavailableError(
                "the embedder is not marked semantic; a vector store compares "
                "meanings, so set semantic = True on a trained embedder or pass "
                "allow_non_semantic=True for tests and demonstrations"
            )
        self._embedder = embedder
        self._max_items = _check_cap(max_items)
        self._points: "OrderedDict[str, Tuple[np.ndarray, Dict[str, Any]]]" = (
            OrderedDict()
        )
        self._lock = threading.RLock()
        self._last_error: Optional[str] = None

    def _embed(self, text: str) -> np.ndarray:
        try:
            vector = np.asarray(self._embedder.encode([text]), dtype=float)[0]
        except Exception as exc:
            self._last_error = str(exc)
            raise
        self._last_error = None
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0.0 else vector

    def store_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
        emotional_weight: float = 0.0,
        point_id: Optional[str] = None,
    ) -> str:
        vector = self._embed(text)
        payload = copy.deepcopy(metadata) if metadata else {}
        payload.update(
            {
                "text": text,
                "importance": importance,
                "emotional_weight": emotional_weight,
                "stored_at": _utc_now().isoformat(),
            }
        )
        key = point_id or str(uuid.uuid4())
        with self._lock:
            self._points.pop(key, None)
            self._points[key] = (vector, payload)
            while len(self._points) > self._max_items:
                self._points.popitem(last=False)
        return key

    def search_similar(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        limit = top_k if top_k is not None else get_settings().VECTOR_SEARCH_LIMIT
        if limit <= 0:
            return []
        probe = self._embed(query)
        with self._lock:
            hits: List[Tuple[str, float, Dict[str, Any]]] = []
            for key, (vector, payload) in self._points.items():
                if filters and any(payload.get(k) != v for k, v in filters.items()):
                    continue
                score = float(np.dot(vector, probe))
                if score >= min_score:
                    hits.append((key, score, copy.deepcopy(payload)))
        hits.sort(key=lambda hit: hit[1], reverse=True)
        return hits[:limit]

    def get_memory_by_id(self, point_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._points.get(point_id)
            return copy.deepcopy(entry[1]) if entry is not None else None

    def delete_memories(self, point_ids: Sequence[str]) -> int:
        with self._lock:
            return sum(1 for pid in point_ids if self._points.pop(pid, None))

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "memory",
            "available": self._last_error is None,
            "last_error": self._last_error,
            "points": len(self._points),
            "semantic": bool(getattr(self._embedder, "semantic", False)),
        }


class InMemoryGraphStore:
    """Concepts, memory links, concept relations and mentions in dicts."""

    def __init__(self) -> None:
        self._concepts: Dict[str, Dict[str, str]] = {}
        self._memories: Dict[int, Dict[str, Any]] = {}
        self._relations: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._mentions: Dict[str, Dict[str, Set[str]]] = {}
        self._lock = threading.RLock()

    def link_memory_to_concepts(
        self,
        memory_id: int,
        memory_content: str,
        concept_names: Sequence[str],
        relevance: float = 0.5,
        extraction_method: str = "keyword",
    ) -> int:
        with self._lock:
            node = self._memories.setdefault(
                int(memory_id), {"content_preview": "", "links": {}}
            )
            node["content_preview"] = memory_content[:200]
            linked = 0
            for name in concept_names:
                if name not in self._concepts:
                    logger.warning("concept %r not found; skipping link", name)
                    continue
                node["links"][name] = {
                    "relevance": relevance,
                    "extraction_method": extraction_method,
                }
                linked += 1
            return linked

    def list_concept_names(self, limit: Optional[int] = None) -> List[str]:
        with self._lock:
            names = sorted(self._concepts)
        return names if limit is None else names[: max(limit, 0)]

    def merge_concept(
        self, name: str, description: str = "", category: str = "conversation"
    ) -> None:
        with self._lock:
            self._concepts.setdefault(
                name, {"name": name, "description": description, "category": category}
            )

    def link_concept_to_conversation(
        self, name: str, conversation_id: str, speaker: str
    ) -> None:
        with self._lock:
            self.merge_concept(name)
            self._mentions.setdefault(name, {}).setdefault(conversation_id, set()).add(
                speaker
            )

    def relate_concepts(
        self,
        first: str,
        second: str,
        relationship: str = "relates_to",
        strength: float = 0.5,
    ) -> bool:
        with self._lock:
            if first == second or first not in self._concepts:
                return False
            if second not in self._concepts:
                return False
            key = (first, second) if first < second else (second, first)
            self._relations[key] = {"relationship": relationship, "strength": strength}
            return True

    def get_related_concepts(
        self, concept_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        with self._lock:
            rows = []
            for (a, b), edge in self._relations.items():
                if concept_name not in (a, b):
                    continue
                rows.append(
                    {
                        "concept": b if concept_name == a else a,
                        "relationship": edge["relationship"],
                        "strength": edge["strength"],
                    }
                )
        rows.sort(key=lambda row: (-row["strength"], row["concept"]))
        return rows[: max(limit, 0)]

    def delete_memory_nodes(self, memory_ids: Sequence[int]) -> int:
        with self._lock:
            return sum(1 for mid in memory_ids if self._memories.pop(int(mid), None))

    def close(self) -> None:
        return None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "backend": "memory",
                "available": True,
                "last_error": None,
                "concepts": len(self._concepts),
                "memory_nodes": len(self._memories),
            }


class _TimedDocs:
    """Documents keyed by an id, read newest first by ``timestamp``.

    Among equal timestamps the later write comes first. The oldest write is
    dropped at the cap. Documents are deep-copied on write and on read.
    """

    def __init__(self, max_items: int, clock: Optional[Clock]) -> None:
        self._max_items = _check_cap(max_items)
        self._clock = clock or _utc_now
        self._docs: "OrderedDict[str, Tuple[int, Dict[str, Any]]]" = OrderedDict()
        self._seq = 0
        self._lock = threading.RLock()

    def now(self) -> datetime:
        return _aware(self._clock())

    def stamp(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Copy ``doc`` and give it an aware UTC ``timestamp``."""
        stored = copy.deepcopy(doc)
        moment = stored.get("timestamp")
        stored["timestamp"] = self.now() if moment is None else _aware(moment)
        return stored

    def put(self, key: str, doc: Dict[str, Any]) -> None:
        with self._lock:
            if key in self._docs:
                raise ValueError(f"id {key!r} is already stored")
            self._seq += 1
            self._docs[key] = (self._seq, doc)
            while len(self._docs) > self._max_items:
                self._docs.popitem(last=False)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._docs.get(key)
            return None if entry is None else entry[1]

    def newest_first(
        self,
        limit: int,
        keep: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        with self._lock:
            entries = list(self._docs.values())
        entries.sort(key=lambda entry: (entry[1]["timestamp"], entry[0]), reverse=True)
        docs = [doc for _seq, doc in entries if keep is None or keep(doc)]
        return [copy.deepcopy(doc) for doc in docs[:limit]]

    def __len__(self) -> int:
        with self._lock:
            return len(self._docs)


def _require_keys(doc: Dict[str, Any], keys: Sequence[str], kind: str) -> None:
    missing = [key for key in keys if doc.get(key) in (None, "")]
    if missing:
        raise ValueError(f"{kind} lacks required fields {missing}")


class InMemorySafetyAuditStore:
    """Safety audit rows held in process memory (proposed actions are private)."""

    def __init__(
        self, max_items: int = DEFAULT_MAX_ITEMS, clock: Optional[Clock] = None
    ) -> None:
        self._rows = _TimedDocs(max_items, clock)

    async def add_audit(self, record: Dict[str, Any]) -> str:
        row = self._rows.stamp(normalize_safety_audit(record))
        row_id = str(uuid.uuid4())
        row["id"] = row_id
        self._rows.put(row_id, row)
        return row_id

    async def pending_review(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._rows.newest_first(limit, lambda row: bool(row["requires_review"]))

    async def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._rows.newest_first(limit)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "memory",
            "available": True,
            "last_error": None,
            "rows": len(self._rows),
        }


class InMemoryAlertStore:
    """Alerts held in process memory, keyed by ``alert_id``."""

    def __init__(
        self, max_items: int = DEFAULT_MAX_ITEMS, clock: Optional[Clock] = None
    ) -> None:
        self._alerts = _TimedDocs(max_items, clock)

    async def add_alert(self, alert: Dict[str, Any]) -> None:
        _require_keys(alert, ("alert_id", "alert_type", "severity"), "alert")
        severities_at_or_above(alert["severity"])
        stored = self._alerts.stamp(alert)
        self._alerts.put(str(stored["alert_id"]), stored)

    async def recent_alerts(
        self,
        limit: int = 100,
        alert_type: Optional[str] = None,
        min_severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        allowed = (
            set(severities_at_or_above(min_severity))
            if min_severity is not None
            else None
        )

        def keep(doc: Dict[str, Any]) -> bool:
            if alert_type is not None and doc.get("alert_type") != alert_type:
                return False
            return allowed is None or doc.get("severity") in allowed

        return self._alerts.newest_first(limit, keep)

    async def acknowledge(self, alert_id: str) -> bool:
        doc = self._alerts.get(str(alert_id))
        if doc is None:
            return False
        doc["acknowledged"] = True
        doc["acknowledged_at"] = self._alerts.now()
        return True

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "memory",
            "available": True,
            "last_error": None,
            "alerts": len(self._alerts),
        }


class InMemoryPrivacyAuditStore:
    """Privacy audit events held in process memory (user ids are private)."""

    def __init__(
        self, max_items: int = DEFAULT_MAX_ITEMS, clock: Optional[Clock] = None
    ) -> None:
        self._events = _TimedDocs(max_items, clock)

    async def add_event(self, event: Dict[str, Any]) -> None:
        _require_keys(event, ("event_id", "event_type", "user_id"), "privacy event")
        stored = self._events.stamp(event)
        self._events.put(str(stored["event_id"]), stored)

    async def events(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        floor = _aware(since) if since is not None else None

        def keep(doc: Dict[str, Any]) -> bool:
            if floor is not None and doc["timestamp"] < floor:
                return False
            if user_id is not None and doc.get("user_id") != user_id:
                return False
            return event_type is None or doc.get("event_type") == event_type

        return self._events.newest_first(limit, keep)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "memory",
            "available": True,
            "last_error": None,
            "events": len(self._events),
        }
