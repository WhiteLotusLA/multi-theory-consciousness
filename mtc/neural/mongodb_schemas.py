"""
MongoDB Neural Data Layer
=========================

MongoDB documents and operations for the substrate: spiking telemetry, liquid
state snapshots, HTM column states, thought streams, conversation turns and
sessions, thought gists and consciousness snapshots. ``NeuralDataManager`` is
the data manager the encoder, the bridge, the memory consolidator and the
orchestrator write through; ``MongoConversationStore`` implements
``ConversationTurnStore``.

Collections (eight): ``neural_spikes``, ``liquid_state``, ``htm_columns``,
``thought_streams``, ``conversation_turns``, ``conversation_sessions``,
``thought_gists`` and ``consciousness_snapshots``. No legacy ``thoughts``
collection exists.

Configuration, read when a manager or store is built: ``MONGODB_HOST``,
``MONGODB_PORT``, ``MONGODB_DB`` (the database name; empty derives
``<DB_PREFIX>_neural``), the credential keys,
``MONGODB_SERVER_SELECTION_TIMEOUT_MS`` and ``SYSTEM_NAME`` (the default second
participant of a session). pymongo is the optional ``mongo`` extra and is
imported only through ``mtc.memory.clients`` inside the client factory;
importing this module loads no driver, and building a manager connects
nowhere.

Writes and reads:

- The telemetry writes (``store_spike``, ``store_spike_batch``,
  ``store_liquid_state``, ``store_htm_column``, ``store_thought``) are
  synchronous because the substrate callers are, and best effort: a server that
  cannot be reached is recorded once, the manager is disabled until
  ``retry_after_s`` has passed, and writes in that window return None (or an
  empty list) and are counted in ``status()['skipped_writes']``. The first
  failure waits at most ``MONGODB_SERVER_SELECTION_TIMEOUT_MS``; nothing
  blocks at construction.
- Every other method (reads, snapshots, gists, retention) raises
  ``StoreUnavailable`` for an unreachable server and never answers with a
  stand-in. Every async method runs the same call in ``asyncio.to_thread``.
- Stored timestamps are timezone-aware UTC; a naive timestamp handed to a
  document or a retention call raises ValueError. Documents read back carry
  ``id`` (the document id as text) in place of ``_id``.

Retention is the retention store's job (``prune_telemetry``, ``prune_thoughts``
and ``expire_gists``), not a TTL index, because two mechanisms would disagree.

Schema artifact: ``NEURAL_INDEXES``, applied idempotently by ``ensure_indexes()``
as a safety net; the store setup (``mtc db init``) is canonical. Thought streams
and conversation turns hold private conversation content at runtime and live
only in the operator's MongoDB; the modules log no document content.
"""

import asyncio
import copy
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Tuple

from mtc.core.config import Settings, get_settings
from mtc.memory.clients import IndexSpec, apply_indexes, create_mongo_client
from mtc.memory.mongo_stores import (
    ASCENDING,
    DESCENDING,
    MongoDocumentStore,
    is_mongo_unreachable,
)
from mtc.memory.protocols import TELEMETRY_COLLECTIONS, StoreUnavailable
from mtc.memory.thought_gist import ThoughtGist

logger = logging.getLogger(__name__)

NEURAL_INDEXES: Dict[str, List[IndexSpec]] = {
    "neural_spikes": [
        IndexSpec((("timestamp", DESCENDING),)),
        IndexSpec((("neuron_id", ASCENDING), ("timestamp", DESCENDING))),
        IndexSpec((("emotion_type", ASCENDING), ("timestamp", DESCENDING))),
        IndexSpec((("intensity", DESCENDING),)),
    ],
    "liquid_state": [
        IndexSpec((("timestamp", DESCENDING),)),
        IndexSpec((("coherence", DESCENDING),)),
        IndexSpec((("entropy", DESCENDING),)),
    ],
    "htm_columns": [
        IndexSpec((("timestamp", DESCENDING),)),
        IndexSpec((("column_id", ASCENDING), ("timestamp", DESCENDING))),
        IndexSpec((("anomaly_score", DESCENDING),)),
        IndexSpec((("semantic_label", ASCENDING),)),
    ],
    "thought_streams": [
        IndexSpec((("timestamp", DESCENDING),)),
        IndexSpec((("stream_id", ASCENDING),)),
        # the id child links resolve on; documents written before it existed
        # carry none, so the index is not unique
        IndexSpec((("thought_id", ASCENDING),)),
        IndexSpec((("parent_thought_id", ASCENDING),)),
        IndexSpec((("thought_type", ASCENDING), ("timestamp", DESCENDING))),
        # hashed: long thought texts cannot exceed the index key size limit
        IndexSpec((("thought_text", "hashed"),)),
    ],
    "conversation_turns": [
        IndexSpec((("timestamp", DESCENDING),)),
        IndexSpec((("conversation_id", ASCENDING), ("turn_number", ASCENDING))),
        IndexSpec((("speaker", ASCENDING), ("timestamp", DESCENDING))),
        IndexSpec((("conversation_id", ASCENDING), ("timestamp", ASCENDING))),
    ],
    "conversation_sessions": [
        IndexSpec((("started_at", DESCENDING),)),
        IndexSpec((("conversation_id", ASCENDING),)),
        IndexSpec((("primary_participant", ASCENDING), ("started_at", DESCENDING))),
        IndexSpec((("phase", ASCENDING), ("started_at", DESCENDING))),
        IndexSpec((("ended_at", DESCENDING),)),
    ],
    "thought_gists": [
        IndexSpec((("dominant_emotion", ASCENDING), ("created_at", DESCENDING))),
        IndexSpec((("retention_strength", ASCENDING),)),
        IndexSpec((("importance_score", DESCENDING),)),
        IndexSpec((("gist_id", ASCENDING),), unique=True),
    ],
    "consciousness_snapshots": [IndexSpec((("timestamp", DESCENDING),))],
}

NEURAL_COLLECTIONS = tuple(NEURAL_INDEXES)

_GIST_FIELDS = {f.name for f in fields(ThoughtGist)} - {"gist_id"}
_GIST_TIME_FIELDS = ("created_at", "last_retrieved")


def to_utc(moment: datetime) -> datetime:
    """Return an instant in UTC.

    A naive datetime is read as local clock time (the convention of the
    substrate's callers, which stamp events with ``datetime.now()``).
    """
    return moment.astimezone(timezone.utc)


def _aware(moment: datetime, name: str) -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return moment.astimezone(timezone.utc)


@dataclass
class NeuralSpike:
    """One spike event (or one summary record, which has no neuron).

    ``membrane_potential`` is the value read from the network, or None when the
    network exposes none; it is never an approximation.
    """

    timestamp: datetime
    neuron_id: Optional[int]
    spike_time: float
    emotion_type: str
    intensity: float
    pre_synaptic_ids: List[int] = field(default_factory=list)
    post_synaptic_ids: List[int] = field(default_factory=list)
    membrane_potential: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = _aware(self.timestamp, "timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Return the MongoDB document."""
        return asdict(self)


@dataclass
class LiquidState:
    """A liquid state machine snapshot.

    ``readout_values``, ``coherence``, ``emotional_valence`` and
    ``arousal_level`` come from the readout and are None while the readout is
    untrained or the measurement is absent.
    """

    timestamp: datetime
    reservoir_size: int
    neuron_states: List[float]
    input_pattern: List[float]
    readout_values: Optional[Dict[str, float]] = None
    entropy: Optional[float] = None
    coherence: Optional[float] = None
    emotional_valence: Optional[float] = None
    arousal_level: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = _aware(self.timestamp, "timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Return the MongoDB document."""
        return asdict(self)


@dataclass
class HTMColumn:
    """The state of one HTM step or column."""

    timestamp: datetime
    column_id: int
    active_cells: List[int]
    predictive_cells: List[int]
    anomaly_score: float
    learning_cells: List[int] = field(default_factory=list)
    permanence_values: Dict[str, float] = field(default_factory=dict)
    predicted_next: List[int] = field(default_factory=list)
    prediction_confidence: Optional[float] = None
    semantic_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = _aware(self.timestamp, "timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Return the MongoDB document."""
        return asdict(self)


@dataclass
class ThoughtStream:
    """One thought of the continuous thought machine (private content).

    The memory-management fields (``access_count``, ``importance_score``,
    ``retention_strength``, ``is_protected``, ``protection_reason`` and
    ``consolidated_into``) are what live documents carry; the scores are None
    until something measures them. ``thought_id`` is the thought's own id (a
    burst shares one ``stream_id``); ``parent_thought_id`` and
    ``child_thoughts`` hold thought ids, and ``link_child_thought`` resolves a
    parent by it. It is None on documents written before the field existed.
    """

    timestamp: datetime
    stream_id: str
    thought_text: str
    thought_type: str
    emotional_state: Dict[str, float] = field(default_factory=dict)
    associated_memories: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    coherence_score: Optional[float] = None
    parent_thought_id: Optional[str] = None
    child_thoughts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    importance_score: Optional[float] = None
    retention_strength: Optional[float] = None
    is_protected: bool = False
    protection_reason: Optional[str] = None
    consolidated_into: Optional[str] = None
    thought_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.timestamp = _aware(self.timestamp, "timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Return the MongoDB document."""
        return asdict(self)


@dataclass
class ConversationTurn:
    """One turn of a conversation (private content)."""

    timestamp: datetime
    conversation_id: str
    speaker: str
    role: str
    content: str
    turn_number: int
    turn_id: Optional[str] = None
    neural_state_refs: Optional[Dict[str, Any]] = None
    emotional_context: Optional[Dict[str, Any]] = None
    topics: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.timestamp = _aware(self.timestamp, "timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Return the MongoDB document."""
        return asdict(self)


@dataclass
class ConversationSession:
    """A conversation session; ``phase`` is a free-form label."""

    conversation_id: str
    started_at: datetime
    session_id: Optional[str] = None
    ended_at: Optional[datetime] = None
    participants: Optional[List[str]] = None
    primary_participant: Optional[str] = None
    turn_count: int = 0
    topics: Optional[List[str]] = None
    summary: Optional[str] = None
    dominant_emotions: Optional[Dict[str, float]] = None
    coherence_score: Optional[float] = None
    conversation_type: str = "general"
    phase: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.started_at = _aware(self.started_at, "started_at")
        if self.ended_at is not None:
            self.ended_at = _aware(self.ended_at, "ended_at")

    def to_dict(self) -> Dict[str, Any]:
        """Return the MongoDB document."""
        return asdict(self)


def _read(doc: Dict[str, Any]) -> Dict[str, Any]:
    """A document as handed to callers: ``id`` for ``_id``, aware datetimes."""
    out = dict(doc)
    if "_id" in out:
        out["id"] = str(out.pop("_id"))
    for key, value in out.items():
        if isinstance(value, datetime) and value.tzinfo is None:
            out[key] = value.replace(tzinfo=timezone.utc)
    return out


def _bounded(limit: int) -> int:
    return max(int(limit), 0)


def _check_emotion_name(name: str) -> str:
    if not name or "." in name or name.startswith("$"):
        raise ValueError(f"{name!r} is not a usable emotion name")
    return name


class NeuralDataManager:
    """Reads and writes the substrate's MongoDB collections.

    Args:
        database_name: Database name; defaults to ``MONGODB_DB`` at
            construction.
        client: A pymongo client to use (it is not closed by ``close()``);
            None builds one from Settings, which connects lazily.
        settings: Settings to read instead of the process-wide ones.
        retry_after_s: Seconds the manager stays disabled after a connection
            failure before it tries the server again.
        clock: A monotonic clock in seconds (tests inject one).
    """

    def __init__(
        self,
        database_name: Optional[str] = None,
        client: Any = None,
        settings: Optional[Settings] = None,
        retry_after_s: float = 60.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else create_mongo_client(self._settings, connect=False)
        )
        self.db = self._client[database_name or self._settings.MONGODB_DB]
        self.retry_after_s = float(retry_after_s)
        self._clock: Callable[[], float] = clock or time.monotonic
        self._disabled_until: Optional[float] = None
        self._last_error: Optional[str] = None
        self._failures = 0
        self._skipped_writes = 0
        self._closed = False
        # True once a driver call returned or failed to reach the server
        self._contacted = False

        self.spikes = self.db["neural_spikes"]
        self.liquid_states = self.db["liquid_state"]
        self.htm_columns = self.db["htm_columns"]
        self.thought_streams = self.db["thought_streams"]
        self.conversation_turns = self.db["conversation_turns"]
        self.conversation_sessions = self.db["conversation_sessions"]
        self.thought_gists = self.db["thought_gists"]
        self.consciousness_snapshots = self.db["consciousness_snapshots"]

    # ---- availability -------------------------------------------------

    def _remaining_backoff(self) -> float:
        if self._disabled_until is None:
            return 0.0
        return max(self._disabled_until - self._clock(), 0.0)

    def _mark_failed(self, exc: BaseException) -> None:
        was_available = self._last_error is None
        self._contacted = True
        self._failures += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
        self._disabled_until = self._clock() + self.retry_after_s
        if was_available:
            logger.warning(
                "MongoDB could not be reached (%s); neural data writes are "
                "skipped for %.0f s",
                type(exc).__name__,
                self.retry_after_s,
            )

    def _mark_ok(self) -> None:
        self._contacted = True
        self._last_error = None
        self._disabled_until = None

    def _call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a driver call; an unreachable server raises ``StoreUnavailable``."""
        if self._closed:
            raise StoreUnavailable("the neural data manager is closed")
        remaining = self._remaining_backoff()
        if remaining > 0.0:
            raise StoreUnavailable(
                f"MongoDB is disabled for another {remaining:.0f} s after: "
                f"{self._last_error}"
            )
        try:
            result = func(*args, **kwargs)
        except StoreUnavailable:
            raise
        except Exception as exc:
            if is_mongo_unreachable(exc):
                self._mark_failed(exc)
                raise StoreUnavailable(
                    f"MongoDB could not be reached: {type(exc).__name__}: {exc}"
                ) from exc
            raise
        self._mark_ok()
        return result

    def _telemetry(self, default: Any, func: Callable[..., Any], *args: Any) -> Any:
        """Best-effort write: ``default`` when the server is unavailable."""
        if self._closed or self._remaining_backoff() > 0.0:
            self._skipped_writes += 1
            return default
        try:
            return self._call(func, *args)
        except StoreUnavailable:
            return default  # the failure was recorded by ``_call``

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available`` and ``last_error`` plus the failure,
        skipped-write and retry figures.

        ``available`` is None until a driver call has returned or failed to
        reach the server (nothing is known before the first contact), True
        after a success, and False after a connection failure or ``close()``.
        """
        if self._closed:
            available: Optional[bool] = False
        elif not self._contacted:
            available = None
        else:
            available = self._last_error is None
        return {
            "backend": "mongodb",
            "available": available,
            "last_error": self._last_error,
            "database": getattr(self.db, "name", None),
            "failures": self._failures,
            "skipped_writes": self._skipped_writes,
            "retry_in_s": self._remaining_backoff() or None,
            "closed": self._closed,
        }

    # ---- schema -------------------------------------------------------

    def ensure_indexes(self) -> int:
        """Create every index in ``NEURAL_INDEXES`` (idempotent); return the count.

        A safety net: the store setup (``mtc db init``) is the canonical schema.
        """
        return self._call(apply_indexes, self.db, NEURAL_INDEXES)

    # ---- neural spikes --------------------------------------------------

    def store_spike(self, spike: NeuralSpike) -> Optional[str]:
        """Store one spike; None when the server is unavailable."""
        result = self._telemetry(None, self.spikes.insert_one, spike.to_dict())
        return None if result is None else str(result.inserted_id)

    def store_spike_batch(self, spikes: Sequence[NeuralSpike]) -> List[str]:
        """Store spikes in one write; an empty list when none were stored."""
        documents = [spike.to_dict() for spike in spikes]
        if not documents:
            return []
        result = self._telemetry(None, self.spikes.insert_many, documents)
        return [] if result is None else [str(i) for i in result.inserted_ids]

    def get_spikes_by_emotion(
        self, emotion_type: str, limit: int = 100, min_intensity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Return spikes of one emotion type at or above an intensity, newest first."""
        query = {"emotion_type": emotion_type, "intensity": {"$gte": min_intensity}}
        return self._find(self.spikes, query, "timestamp", DESCENDING, limit)

    def get_spikes_by_timerange(
        self, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Return spikes within an inclusive time range, oldest first."""
        query = {
            "timestamp": {
                "$gte": _aware(start_time, "start_time"),
                "$lte": _aware(end_time, "end_time"),
            }
        }
        return self._find(self.spikes, query, "timestamp", ASCENDING, None)

    # ---- liquid state ---------------------------------------------------

    def store_liquid_state(self, state: LiquidState) -> Optional[str]:
        """Store a liquid state snapshot; None when the server is unavailable."""
        result = self._telemetry(None, self.liquid_states.insert_one, state.to_dict())
        return None if result is None else str(result.inserted_id)

    def get_latest_liquid_state(self) -> Optional[Dict[str, Any]]:
        """Return the newest liquid state, or None."""
        found = self._find(self.liquid_states, {}, "timestamp", DESCENDING, 1)
        return found[0] if found else None

    def get_liquid_states_by_coherence(
        self, min_coherence: float = 0.7, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return states at or above a coherence, newest first."""
        query = {"coherence": {"$gte": min_coherence}}
        return self._find(self.liquid_states, query, "timestamp", DESCENDING, limit)

    # ---- HTM columns ----------------------------------------------------

    def store_htm_column(self, column: HTMColumn) -> Optional[str]:
        """Store an HTM column state; None when the server is unavailable."""
        result = self._telemetry(None, self.htm_columns.insert_one, column.to_dict())
        return None if result is None else str(result.inserted_id)

    def get_htm_anomalies(
        self, min_anomaly_score: float = 0.8, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return columns at or above an anomaly score, newest first."""
        query = {"anomaly_score": {"$gte": min_anomaly_score}}
        return self._find(self.htm_columns, query, "timestamp", DESCENDING, limit)

    def get_htm_by_semantic_label(
        self, label: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return columns with one semantic label, newest first."""
        query = {"semantic_label": label}
        return self._find(self.htm_columns, query, "timestamp", DESCENDING, limit)

    # ---- thought streams ------------------------------------------------

    def store_thought(self, thought: ThoughtStream) -> Optional[str]:
        """Store a thought; None when the server is unavailable."""
        result = self._telemetry(
            None, self.thought_streams.insert_one, thought.to_dict()
        )
        return None if result is None else str(result.inserted_id)

    def increment_thought_access(self, thought_text: str) -> bool:
        """Count one more occurrence of a recurring thought.

        Bumps ``access_count`` of the newest thought with exactly this text
        instead of storing a copy, so frequency is kept. Returns True when a
        matching thought was found.
        """
        found = self._call(
            self.thought_streams.find_one_and_update,
            {"thought_text": thought_text},
            {"$inc": {"access_count": 1}},
            sort=[("timestamp", DESCENDING)],
        )
        return found is not None

    def update_thought_children(
        self, parent_stream_id: str, child_thought_id: str
    ) -> None:
        """Add a child thought id to the newest document with this stream id.

        A stream id names a burst of thoughts, so it cannot single out one
        parent; ``link_child_thought`` matches on the thought id instead.
        """
        self._call(
            self.thought_streams.update_one,
            {"stream_id": parent_stream_id},
            {"$addToSet": {"child_thoughts": child_thought_id}},
        )

    def link_child_thought(self, parent_thought_id: str, child_thought_id: str) -> bool:
        """Add a child thought id to the parent found by its ``thought_id``.

        Returns False when no document carries the parent id (a parent that
        was never stored or was pruned), so the caller can tell a link that
        did not resolve from one that did. Linking twice adds the child once.
        """
        result = self._call(
            self.thought_streams.update_one,
            {"thought_id": parent_thought_id},
            {"$addToSet": {"child_thoughts": child_thought_id}},
        )
        return bool(result.matched_count)

    def touch_thought(
        self, thought_text: str, seen_at: datetime
    ) -> Optional[Dict[str, Any]]:
        """Count one more occurrence of a recurring thought and stamp it seen.

        Like ``increment_thought_access`` (the newest thought with exactly this
        text), and also sets ``metadata.last_seen``. Returns the matched
        document as it was before the update (so its ``thought_id`` is
        available), or None when no thought has this text.
        """
        found = self._call(
            self.thought_streams.find_one_and_update,
            {"thought_text": thought_text},
            {
                "$inc": {"access_count": 1},
                "$set": {"metadata.last_seen": _aware(seen_at, "seen_at")},
            },
            sort=[("timestamp", DESCENDING)],
        )
        return None if found is None else _read(found)

    def find_thoughts(self, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Return thoughts matching an equality query, newest first.

        Raises ``StoreUnavailable`` when the server cannot be reached.
        """
        return self._find(self.thought_streams, query, "timestamp", DESCENDING, limit)

    def get_thought_chain(self, parent_thought_id: str) -> List[Dict[str, Any]]:
        """Return the thoughts that branched from a parent, oldest first."""
        query = {"parent_thought_id": parent_thought_id}
        return self._find(self.thought_streams, query, "timestamp", ASCENDING, None)

    def get_recent_thoughts(
        self, thought_type: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return recent thoughts, optionally of one type, newest first."""
        query: Dict[str, Any] = {"thought_type": thought_type} if thought_type else {}
        return self._find(self.thought_streams, query, "timestamp", DESCENDING, limit)

    def get_thought_by_stream_id(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Return the newest thought with a stream id (indexed), or None."""
        found = self._find(
            self.thought_streams, {"stream_id": stream_id}, "timestamp", DESCENDING, 1
        )
        return found[0] if found else None

    def get_thoughts_by_emotion(
        self, emotion: str, threshold: float = 0.5, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return thoughts whose ``emotional_state`` for an emotion is at or above
        a threshold, newest first."""
        query = {f"emotional_state.{_check_emotion_name(emotion)}": {"$gte": threshold}}
        return self._find(self.thought_streams, query, "timestamp", DESCENDING, limit)

    # ---- thought gists --------------------------------------------------

    def store_gist(self, gist_doc: Dict[str, Any]) -> str:
        """Store a gist document, replacing one with the same ``gist_id``."""
        stored = copy.deepcopy(gist_doc)
        gist_id = stored["gist_id"]
        self._call(self.thought_gists.replace_one, {"gist_id": gist_id}, stored, True)
        return gist_id

    def get_active_gists(
        self, min_retention: float = 0.2, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Return gist documents at or above a retention, most important first.

        A null retention (unknown) never matches the floor, so such a gist is
        never returned; a null importance sorts after every measured one (the
        server orders null below every number).
        """
        size = _bounded(limit)
        if size == 0:
            return []

        def run() -> List[Dict[str, Any]]:
            cursor = self.thought_gists.find(
                {"retention_strength": {"$gte": min_retention}}
            ).sort([("importance_score", DESCENDING), ("created_at", DESCENDING)])
            return [_read(doc) for doc in cursor.limit(size)]

        return self._call(run)

    async def add_gist(self, gist: ThoughtGist) -> str:
        """Store a gist (``ThoughtGistStore``); an existing id is replaced."""
        return await asyncio.to_thread(self.store_gist, gist.to_document())

    async def active_gists(
        self, min_retention: float = 0.2, limit: int = 100
    ) -> List[ThoughtGist]:
        """Return gists with retention at or above the floor, most important first.

        A None retention is never returned; a None importance sorts last.
        """
        docs = await asyncio.to_thread(self.get_active_gists, min_retention, limit)
        for doc in docs:
            doc.pop("id", None)
        return [ThoughtGist.from_document(doc) for doc in docs]

    def _gist_page(
        self, after: Optional[Tuple[datetime, str]], size: int
    ) -> List[Dict[str, Any]]:
        """The next ``size`` gist documents by ``created_at`` then ``gist_id``.

        ``after`` is the key of the last document of the previous page; the
        page starts strictly after it, so equal creation times are not
        skipped or repeated.
        """
        if after is None:
            query: Dict[str, Any] = {}
        else:
            created, gist_id = after
            query = {
                "$or": [
                    {"created_at": {"$gt": created}},
                    {"created_at": created, "gist_id": {"$gt": gist_id}},
                ]
            }
        cursor = (
            self.thought_gists.find(query)
            .sort([("created_at", ASCENDING), ("gist_id", ASCENDING)])
            .limit(size)
        )
        return [_read(doc) for doc in cursor]

    async def iter_gists(self, batch: int = 500) -> AsyncIterator[ThoughtGist]:
        """Yield every stored gist whatever its retention (null included).

        Gists come ordered by ``created_at``, then ``gist_id``, read from the
        server ``batch`` at a time; each page starts after the last key of the
        one before. A batch below 1 raises ValueError; an unreachable server
        raises ``StoreUnavailable``.
        """
        if int(batch) < 1:
            raise ValueError("batch must be at least 1")
        size = int(batch)
        after: Optional[Tuple[datetime, str]] = None
        while True:
            docs = await asyncio.to_thread(self._call, self._gist_page, after, size)
            for doc in docs:
                doc.pop("id", None)
            gists = [ThoughtGist.from_document(doc) for doc in docs]
            for gist in gists:
                yield gist
            if len(docs) < size:
                return
            last = docs[-1]
            after = (last["created_at"], last["gist_id"])

    async def update_gist(self, gist_id: str, fields: Dict[str, Any]) -> bool:
        """Set fields of a stored gist; False when no such gist exists.

        An unknown field name raises ValueError, and so does a naive timestamp.
        """
        unknown = set(fields) - _GIST_FIELDS
        if unknown:
            raise ValueError(f"unknown gist fields: {sorted(unknown)}")
        changes = copy.deepcopy(fields)
        for name in _GIST_TIME_FIELDS:
            if changes.get(name) is not None:
                changes[name] = _aware(changes[name], name)

        def run() -> bool:
            if not changes:
                return self.thought_gists.count_documents({"gist_id": gist_id}) > 0
            result = self.thought_gists.update_one(
                {"gist_id": gist_id}, {"$set": changes}
            )
            return result.matched_count > 0

        return await asyncio.to_thread(self._call, run)

    # ---- consciousness snapshots ---------------------------------------

    async def store_consciousness_snapshot(self, snapshot_doc: Dict[str, Any]) -> str:
        """Store an aggregate assessment snapshot and return its id.

        The integration's snapshot carries an ISO timestamp string; it is
        stored as a datetime in UTC (a naive one is read as local time), so the
        timestamp index serves time-range queries.
        """
        stored = copy.deepcopy(snapshot_doc)
        stamp = stored.get("timestamp")
        if isinstance(stamp, str):
            try:
                stamp = datetime.fromisoformat(stamp)
            except ValueError:
                stamp = None
            if stamp is not None:
                stored["timestamp"] = stamp
        if isinstance(stored.get("timestamp"), datetime):
            stored["timestamp"] = to_utc(stored["timestamp"])
        result = await asyncio.to_thread(
            self._call, self.consciousness_snapshots.insert_one, stored
        )
        return str(result.inserted_id)

    # ---- analytics and housekeeping -----------------------------------

    def get_emotion_statistics(
        self, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Return per emotion spike counts and intensities within a time range."""
        pipeline = [
            {
                "$match": {
                    "timestamp": {
                        "$gte": _aware(start_time, "start_time"),
                        "$lte": _aware(end_time, "end_time"),
                    }
                }
            },
            {
                "$group": {
                    "_id": "$emotion_type",
                    "count": {"$sum": 1},
                    "avg_intensity": {"$avg": "$intensity"},
                    "max_intensity": {"$max": "$intensity"},
                }
            },
            {"$sort": {"count": -1}},
        ]
        return self._call(lambda: list(self.spikes.aggregate(pipeline)))

    def _size_mb(self, name: str) -> Optional[float]:
        try:
            rows = list(self.db[name].aggregate([{"$collStats": {"storageStats": {}}}]))
            return float(rows[0]["storageStats"]["size"]) / (1024 * 1024)
        except Exception as exc:
            if is_mongo_unreachable(exc):
                raise
            return None  # the server cannot size it (for example, no such collection)

    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return ``count`` and ``size_mb`` for all eight collections.

        ``size_mb`` is None for a collection the server cannot size.
        """

        def run() -> Dict[str, Dict[str, Any]]:
            return {
                name: {
                    "count": self.db[name].count_documents({}),
                    "size_mb": self._size_mb(name),
                }
                for name in NEURAL_COLLECTIONS
            }

        return self._call(run)

    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Delete spikes, liquid states, HTM columns and thoughts older than a
        number of days; return the counts. The retention store prunes with
        more control and is the usual route."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        counts = {}
        for key, collection in (
            ("spikes_deleted", self.spikes),
            ("liquid_states_deleted", self.liquid_states),
            ("htm_columns_deleted", self.htm_columns),
            ("thought_streams_deleted", self.thought_streams),
        ):
            result = self._call(collection.delete_many, {"timestamp": {"$lt": cutoff}})
            counts[key] = result.deleted_count
        logger.info("neural data cleanup removed %s", counts)
        return counts

    # ---- retention (NeuralRetentionStore) --------------------------------

    async def prune_telemetry(
        self, older_than: datetime, dry_run: bool
    ) -> Dict[str, int]:
        """Remove telemetry older than the cutoff; return counts per collection.

        The keys are ``TELEMETRY_COLLECTIONS``. A dry run counts and removes
        nothing; a naive cutoff raises ValueError.
        """
        cutoff = _aware(older_than, "older_than")

        def run() -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for name in TELEMETRY_COLLECTIONS:
                collection = self.db[name]
                query = {"timestamp": {"$lt": cutoff}}
                if dry_run:
                    counts[name] = collection.count_documents(query)
                else:
                    counts[name] = collection.delete_many(query).deleted_count
            return counts

        return await asyncio.to_thread(self._call, run)

    async def prune_thoughts(
        self, older_than: datetime, max_coherence: float, dry_run: bool
    ) -> int:
        """Remove thoughts older than the cutoff, below ``max_coherence`` and with
        no child thoughts; return the count."""
        cutoff = _aware(older_than, "older_than")
        query = {
            "timestamp": {"$lt": cutoff},
            "coherence_score": {"$lt": max_coherence},
            "child_thoughts.0": {"$exists": False},
        }

        def run() -> int:
            if dry_run:
                return self.thought_streams.count_documents(query)
            return self.thought_streams.delete_many(query).deleted_count

        return await asyncio.to_thread(self._call, run)

    async def expire_gists(self, min_retention: float, dry_run: bool) -> int:
        """Remove unprotected gists whose stored retention is below the floor.

        A null retention (unknown) is never below a floor, so such a gist is
        never expired; a refresh computes it first.
        """
        query = {
            "retention_strength": {"$lt": min_retention},
            "is_protected": {"$ne": True},
        }

        def run() -> int:
            if dry_run:
                return self.thought_gists.count_documents(query)
            return self.thought_gists.delete_many(query).deleted_count

        return await asyncio.to_thread(self._call, run)

    # ---- lifecycle ------------------------------------------------------

    async def close(self) -> None:
        """Release the client when this manager built it; later calls raise
        ``StoreUnavailable``. A client handed in belongs to its owner."""
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await asyncio.to_thread(self._client.close)

    # ---- internals ------------------------------------------------------

    def _find(
        self,
        collection: Any,
        query: Dict[str, Any],
        sort_field: str,
        direction: int,
        limit: Optional[int],
    ) -> List[Dict[str, Any]]:
        size = None if limit is None else _bounded(limit)
        if size == 0:
            return []

        def run() -> List[Dict[str, Any]]:
            cursor = collection.find(query).sort(sort_field, direction)
            if size is not None:
                cursor = cursor.limit(size)
            return [_read(doc) for doc in cursor]

        return self._call(run)


class MongoConversationStore(MongoDocumentStore):
    """``ConversationTurnStore`` over ``conversation_turns`` and
    ``conversation_sessions`` (private conversation text).

    Turns come back in turn order and a ``limit`` keeps the most recent turns;
    turn and session documents carry a ``turn_id`` or ``session_id`` of their
    own. Starting a session again with the same ``conversation_id`` replaces the
    record. Keyword arguments of ``start_conversation_session`` that are not
    session fields are kept under ``metadata``. The default participants are
    the primary participant and ``SYSTEM_NAME``.
    """

    _SESSION_FIELDS = (
        "participants",
        "topics",
        "summary",
        "dominant_emotions",
        "coherence_score",
        "conversation_type",
        "phase",
        "metadata",
    )

    def __init__(
        self, database: Any = None, settings: Optional[Settings] = None
    ) -> None:
        super().__init__(database=database, settings=settings)

    def _indexes(self) -> Dict[str, List[IndexSpec]]:
        return {
            name: NEURAL_INDEXES[name]
            for name in ("conversation_turns", "conversation_sessions")
        }

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
        """Store one turn and return its id."""
        turn = ConversationTurn(
            timestamp=datetime.now(timezone.utc),
            conversation_id=conversation_id,
            speaker=speaker,
            role=role,
            content=content,
            turn_number=turn_number,
            turn_id=uuid.uuid4().hex,
            neural_state_refs=neural_state_refs,
            emotional_context=emotional_context,
            topics=topics,
            metadata=metadata,
        )
        await self._run(self._database["conversation_turns"].insert_one, turn.to_dict())
        return turn.turn_id or ""

    async def start_conversation_session(
        self, conversation_id: str, primary_participant: str, **kwargs: Any
    ) -> str:
        """Open (or reopen) a session record and return its id."""
        extras = {k: v for k, v in kwargs.items() if k not in self._SESSION_FIELDS}
        known = {k: v for k, v in kwargs.items() if k in self._SESSION_FIELDS}
        metadata = {**(known.pop("metadata", None) or {}), **extras}
        participants = known.pop("participants", None) or [
            primary_participant,
            self._settings.SYSTEM_NAME,
        ]
        session = ConversationSession(
            conversation_id=conversation_id,
            started_at=datetime.now(timezone.utc),
            session_id=uuid.uuid4().hex,
            primary_participant=primary_participant,
            participants=participants,
            metadata=metadata or None,
            **known,
        )
        await self._run(
            self._database["conversation_sessions"].replace_one,
            {"conversation_id": conversation_id},
            session.to_dict(),
            True,
        )
        return session.session_id or ""

    async def end_conversation_session(
        self, conversation_id: str, **kwargs: Any
    ) -> None:
        """Close a session record; closing an unknown session is not an error."""

        def run() -> None:
            count = self._database["conversation_turns"].count_documents(
                {"conversation_id": conversation_id}
            )
            changes = {
                **copy.deepcopy(kwargs),
                "ended_at": datetime.now(timezone.utc),
                "turn_count": count,
            }
            self._database["conversation_sessions"].update_one(
                {"conversation_id": conversation_id}, {"$set": changes}
            )

        await self._run(run)

    async def get_conversation_turns(
        self, conversation_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return the latest ``limit`` turns of a conversation in turn order."""
        size = _bounded(limit)
        if size == 0:
            return []

        def run() -> List[Dict[str, Any]]:
            cursor = (
                self._database["conversation_turns"]
                .find({"conversation_id": conversation_id}, {"_id": 0})
                .sort("turn_number", DESCENDING)
                .limit(size)
            )
            return [_read(doc) for doc in reversed(list(cursor))]

        return await self._run(run)

    async def get_recent_conversations(
        self, participant: str, limit: int = 10, phase: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return a participant's recent session records, newest first."""
        size = _bounded(limit)
        if size == 0:
            return []
        query: Dict[str, Any] = {"primary_participant": participant}
        if phase:
            query["phase"] = phase

        def run() -> List[Dict[str, Any]]:
            cursor = (
                self._database["conversation_sessions"]
                .find(query, {"_id": 0})
                .sort("started_at", DESCENDING)
                .limit(size)
            )
            return [_read(doc) for doc in cursor]

        return await self._run(run)

    async def get_stats(self) -> Dict[str, Any]:
        """Return the number of stored sessions and turns."""

        def run() -> Dict[str, Any]:
            return {
                "total_sessions": self._database[
                    "conversation_sessions"
                ].count_documents({}),
                "total_turns": self._database["conversation_turns"].count_documents({}),
            }

        return await self._run(run)


__all__ = [
    "NEURAL_INDEXES",
    "NEURAL_COLLECTIONS",
    "NeuralSpike",
    "LiquidState",
    "HTMColumn",
    "ThoughtStream",
    "ConversationTurn",
    "ConversationSession",
    "NeuralDataManager",
    "MongoConversationStore",
    "to_utc",
]
