"""
Store Protocols
===============

Domain repositories the consciousness modules persist through. Every protocol
is runtime-checkable and, except where noted below, every method is async. The
protocols speak in records and documents, not queries, so an implementation owns
its own query language. A module given no store uses the in-memory implementation from
``mtc.memory.inmemory``; no module in this package opens a network client.

Contract every implementation must satisfy (pinned by
``tests/memory/test_store_contracts.py``):

- Lists are newest first. For document collections keyed by an id, newest means
  most recently written.
- ``upsert_*`` replaces the whole document stored under the key.
- Documents and records handed in and handed back are copies; mutating one
  never reaches the store.
- Stored timestamps are timezone-aware; a naive timestamp raises ValueError.
- ``limit`` bounds the number of items returned.

Privacy: speaker models, learned beliefs, goals, grab entries and experiences
hold real people's names at runtime and are private data. Backends must keep
them out of logs, exports and version control.

The memory, working-memory, turn, gist, retention, vector and graph protocols
below follow the same rules. The vector, graph, concept-matcher and episode
consolidator protocols are synchronous (their backends are blocking clients);
callers in async code run them through ``asyncio.to_thread``. A backend that
cannot reach its server raises ``StoreUnavailable``; it never answers a read
with an invented or empty stand-in.

Backend layout the later PostgreSQL and MongoDB implementations must match:

- PostgreSQL tables ``consciousness_snapshots`` and ``consciousness_events``
  with the columns of ``SnapshotRecord`` and ``SnapshotEvent`` (including
  ``status`` and ``coverage``), a nullable ``overall_score`` and a timestamp
  index on each table.
- MongoDB collections: ``learning_experiences`` (indexes on timestamp,
  speaker, outcome and topic), ``learning_beliefs`` (unique ``belief_id``),
  ``speaker_models`` (unique ``speaker_name``), ``intrinsic_goals`` (unique
  ``goal_id``, plus an index on ``status``), ``attention_grabs`` (timestamp
  index and unique ``grab_id``) and ``dreams`` (timestamp index).
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from mtc.memory.thought_gist import ThoughtGist

WORKING_MEMORY_MIN = 5
WORKING_MEMORY_MAX = 9
WORKING_MEMORY_TTL_S = 1800
EMOTION_STATE_TTL_S = 14400
CACHE_TTL_S = 30

# Substrate telemetry collections the retention store prunes by age.
TELEMETRY_COLLECTIONS = ("neural_spikes", "liquid_state", "htm_columns")

# Fixed namespace for the deterministic vector point of an episodic row.
MEMORY_POINT_NAMESPACE = uuid.UUID("0e091127-84c3-4950-a834-667edb20ac74")


class StoreUnavailable(RuntimeError):
    """A store or its client could not be reached; the operation did not run."""


def point_id_for_memory(postgres_id: int) -> str:
    """Return the vector point id of an episodic row.

    The id is a uuid5 of the row id under ``MEMORY_POINT_NAMESPACE``, so the
    pruner deletes a row's point without a search and a re-embed backfill
    upserts the same point instead of adding a second one.
    """
    return str(uuid.uuid5(MEMORY_POINT_NAMESPACE, str(int(postgres_id))))


def clamp_working_capacity(value: int) -> int:
    """Clamp a working-memory capacity to the five to nine item range."""
    return max(WORKING_MEMORY_MIN, min(WORKING_MEMORY_MAX, int(value)))


@dataclass
class SnapshotRecord:
    """One stored assessment snapshot.

    ``overall_score`` and ``phi_value`` are None when the assessment could not
    measure them; ``status`` and ``coverage`` carry the assessment's own
    vocabulary (``measured``, ``absent``, ``insufficient_data``).
    """

    timestamp: datetime
    trigger: str
    overall_score: Optional[float]
    passing_count: int
    phi_value: Optional[float]
    theory_scores: Dict[str, Any]
    indicator_results: Dict[str, Any]
    processing_time_ms: Optional[float]
    status: str
    coverage: Optional[float]
    metadata: Dict[str, Any]
    id: Optional[int] = None


@dataclass
class SnapshotEvent:
    """One notable change between snapshots (a rise, a drop, a threshold)."""

    timestamp: datetime
    event_type: str
    description: str
    snapshot_id: Optional[int]
    old_value: Optional[float]
    new_value: Optional[float]
    id: Optional[int] = None


@runtime_checkable
class SnapshotStore(Protocol):
    """Snapshots and their events, newest first."""

    async def add_snapshot(self, record: SnapshotRecord) -> int:
        """Store a snapshot and return its assigned id."""
        ...

    async def add_event(self, event: SnapshotEvent) -> int:
        """Store an event and return its assigned id."""
        ...

    async def latest_snapshot(self) -> Optional[SnapshotRecord]:
        """Return the snapshot with the newest timestamp, or None."""
        ...

    async def snapshots(
        self, since: Optional[datetime] = None, limit: int = 100
    ) -> List[SnapshotRecord]:
        """Return snapshots at or after ``since``, newest first."""
        ...

    async def events(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
        event_type: Optional[str] = None,
    ) -> List[SnapshotEvent]:
        """Return events at or after ``since``, newest first, optionally of one type."""
        ...

    async def snapshot_before(self, moment: datetime) -> Optional[SnapshotRecord]:
        """Return the newest snapshot strictly before ``moment``, or None."""
        ...


@runtime_checkable
class LearningStore(Protocol):
    """Experiences (an append-only log) and learned beliefs (keyed documents)."""

    async def add_experience(self, doc: Dict[str, Any]) -> None:
        """Append one experience document."""
        ...

    async def recent_experiences(self, limit: int) -> List[Dict[str, Any]]:
        """Return the latest experiences, newest first."""
        ...

    async def upsert_belief(self, belief_id: str, doc: Dict[str, Any]) -> None:
        """Store a belief under its id, replacing any previous document."""
        ...

    async def load_beliefs(self, limit: int) -> List[Dict[str, Any]]:
        """Return beliefs, most recently written first."""
        ...


@runtime_checkable
class SpeakerModelStore(Protocol):
    """Per-speaker models keyed by the speaker's name (private data at runtime)."""

    async def upsert_model(self, speaker_name: str, doc: Dict[str, Any]) -> None:
        """Store a model under the speaker name, replacing any previous one."""
        ...

    async def load_models(self) -> List[Dict[str, Any]]:
        """Return all models, most recently written first."""
        ...

    async def delete_model(self, speaker_name: str) -> bool:
        """Delete a model; True when one existed."""
        ...


@runtime_checkable
class GoalStore(Protocol):
    """Intrinsic goals keyed by goal id; each document carries a ``status``."""

    async def upsert_goal(self, goal_id: str, doc: Dict[str, Any]) -> None:
        """Store a goal under its id, replacing any previous document."""
        ...

    async def load_goals(self, statuses: Sequence[str]) -> List[Dict[str, Any]]:
        """Return goals whose ``status`` is in ``statuses``, newest first."""
        ...


@runtime_checkable
class AttentionGrabStore(Protocol):
    """Attention-capture journal entries."""

    async def add_grab(self, doc: Dict[str, Any]) -> None:
        """Append one journal entry."""
        ...

    async def recent_grabs(self, limit: int) -> List[Dict[str, Any]]:
        """Return the latest entries, newest first."""
        ...


@runtime_checkable
class DreamStore(Protocol):
    """Dream records."""

    async def add_dream(self, doc: Dict[str, Any]) -> None:
        """Append one dream document."""
        ...

    async def recent_dreams(self, limit: int) -> List[Dict[str, Any]]:
        """Return the latest dreams, newest first."""
        ...


@runtime_checkable
class EpisodicStore(Protocol):
    """Core memory blocks, episodic and semantic memories (private content).

    The PostgreSQL implementation uses the tables ``core_memory``,
    ``episodic_memory`` and ``semantic_memory``; the in-memory implementation
    scores text search by term overlap in place of full-text ranking. Importance
    is on a 0 to 10 scale and 10 must round-trip. Archiving moves a row to the
    memory type ``archived``, which recall and search exclude. Episodic rows
    hold conversation text and are private data at runtime.
    """

    async def ensure_schema(self) -> None:
        """Create what the store needs when it is missing (idempotent)."""
        ...

    async def load_core_blocks(self) -> Dict[str, str]:
        """Return every core block as ``{name: content}``."""
        ...

    async def set_core_block(self, name: str, content: str) -> None:
        """Insert or replace one core block."""
        ...

    async def seed_core_blocks(
        self, blocks: Mapping[str, str], overwrite: bool = False
    ) -> Dict[str, bool]:
        """Write blocks; return ``{name: written}``.

        Without ``overwrite`` an existing block is kept and reports False.
        """
        ...

    async def add_episodic(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 5.0,
        emotional_context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store one episodic memory and return its assigned id."""
        ...

    async def get_episodic(self, ids: Sequence[int]) -> List[Dict[str, Any]]:
        """Return the rows with the given ids (archived rows excluded)."""
        ...

    async def search_text(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Return rows matching the query, best match first, ties newest first."""
        ...

    async def recent(
        self, hours: float, limit: int, min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Return rows created within ``hours``, newest first."""
        ...

    async def unconsolidated(self, limit: int) -> List[Dict[str, Any]]:
        """Return rows not yet consolidated, oldest first."""
        ...

    async def add_semantic(self, record: Dict[str, Any]) -> int:
        """Store one semantic memory and return its assigned id."""
        ...

    async def mark_consolidated(self, ids: Sequence[int], strength: float) -> int:
        """Mark rows consolidated with a strength; return the rows changed."""
        ...

    async def touch(self, ids: Sequence[int]) -> None:
        """Count one access and refresh the last-accessed time of each row."""
        ...

    async def retention_candidates(
        self, older_than: datetime, limit: int
    ) -> List[Dict[str, Any]]:
        """Return non-archived rows created before ``older_than``, oldest first."""
        ...

    async def archive(self, ids: Sequence[int]) -> int:
        """Set the memory type of rows to ``archived``; return the rows changed."""
        ...

    async def delete(self, ids: Sequence[int]) -> int:
        """Delete rows; return the rows removed."""
        ...

    async def stats(self) -> Dict[str, Any]:
        """Return row counts by kind."""
        ...

    async def close(self) -> None:
        """Release the store's resources."""
        ...


@runtime_checkable
class WorkingMemory(Protocol):
    """Per-session working memory held in fast storage (private content).

    The Redis implementation keeps these under ``REDIS_KEY_PREFIX``. The
    conversation buffer holds five to nine messages (``WORKING_MEMORY_CAPACITY``
    clamped), refreshes a 1800 second expiry on each push and returns messages
    oldest first. The emotional state expires after four hours, the cache after
    30 seconds unless a ttl is given. Reads answer from storage only.
    """

    async def add_to_working_memory(
        self, session_id: str, message: Dict[str, Any]
    ) -> int:
        """Append a message (a timestamp is added when absent); return the size."""
        ...

    async def get_working_memory(self, session_id: str) -> List[Dict[str, Any]]:
        """Return the buffered messages, oldest first."""
        ...

    async def clear_working_memory(self, session_id: str) -> None:
        """Drop the session's conversation buffer."""
        ...

    async def update_emotional_state(
        self, session_id: str, state: Dict[str, Any]
    ) -> None:
        """Store the session's current emotional state."""
        ...

    async def get_emotional_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the current emotional state, or None."""
        ...

    async def set_session_context(
        self, session_id: str, context: Dict[str, Any], ttl: Optional[int] = None
    ) -> None:
        """Store session context, optionally expiring after ``ttl`` seconds."""
        ...

    async def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the session context, or None."""
        ...

    async def delete_session(self, session_id: str) -> None:
        """Drop every key of the session."""
        ...

    async def cache_set(self, key: str, value: Any, ttl: int = 30) -> None:
        """Cache a JSON-compatible value for ``ttl`` seconds."""
        ...

    async def cache_get(self, key: str) -> Optional[Any]:
        """Return a cached value, or None when absent or expired."""
        ...

    async def get_metrics(self) -> Dict[str, Any]:
        """Return read, write, hit, miss and error counts."""
        ...

    async def disconnect(self) -> None:
        """Release the connection."""
        ...


@runtime_checkable
class ConversationTurnStore(Protocol):
    """Conversation turns and sessions (private conversation text).

    The MongoDB implementation keeps turns and sessions in their own
    collections. Turns come back in turn order (ascending ``turn_number``); a
    ``limit`` keeps the most recent turns. Timestamps are timezone-aware UTC.
    """

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
        ...

    async def start_conversation_session(
        self, conversation_id: str, primary_participant: str, **kwargs: Any
    ) -> str:
        """Open a session record and return its id."""
        ...

    async def end_conversation_session(
        self, conversation_id: str, **kwargs: Any
    ) -> None:
        """Close a session record; closing an unknown session is not an error."""
        ...

    async def get_conversation_turns(
        self, conversation_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return the latest ``limit`` turns of a conversation in turn order."""
        ...


@runtime_checkable
class ThoughtGistStore(Protocol):
    """Thought gists (private content), keyed by ``gist_id``."""

    async def add_gist(self, gist: ThoughtGist) -> str:
        """Store a gist and return its id."""
        ...

    async def active_gists(
        self, min_retention: float = 0.2, limit: int = 100
    ) -> List[ThoughtGist]:
        """Return gists with retention at or above the floor, most important first."""
        ...

    async def update_gist(self, gist_id: str, fields: Dict[str, Any]) -> bool:
        """Set fields of a stored gist; False when no such gist exists.

        An unknown field name raises ValueError.
        """
        ...


@runtime_checkable
class NeuralRetentionStore(Protocol):
    """Age-based pruning of substrate telemetry, thought streams and gists.

    Every method takes ``dry_run``: when True it counts what it would remove and
    removes nothing. Protected gists are never expired.
    """

    async def prune_telemetry(
        self, older_than: datetime, dry_run: bool
    ) -> Dict[str, int]:
        """Remove telemetry older than the cutoff; return counts per collection.

        The keys are ``TELEMETRY_COLLECTIONS``.
        """
        ...

    async def prune_thoughts(
        self, older_than: datetime, max_coherence: float, dry_run: bool
    ) -> int:
        """Remove thoughts older than the cutoff, below ``max_coherence`` and
        with no child thoughts; return the count."""
        ...

    async def expire_gists(self, min_retention: float, dry_run: bool) -> int:
        """Remove unprotected gists whose retention is below the floor."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Semantic vector store (the Qdrant implementation uses 384 dimensions).

    Methods are synchronous. A store never invents a vector: when the embedder
    is unavailable the call raises and nothing is written. Payloads hold
    private memory text at runtime.
    """

    def store_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
        emotional_weight: float = 0.0,
        point_id: Optional[str] = None,
    ) -> str:
        """Embed and store ``text``; return the point id (upserting ``point_id``)."""
        ...

    def search_similar(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return ``(point_id, score, payload)`` best first, scores at or above
        ``min_score``; ``filters`` keeps payloads whose fields equal the values."""
        ...

    def get_memory_by_id(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Return a point's payload, or None."""
        ...

    def delete_memories(self, point_ids: Sequence[str]) -> int:
        """Delete points; return the points removed."""
        ...

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available`` and ``last_error`` at least."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Knowledge graph of concepts, memory nodes and conversation mentions.

    Methods are synchronous. Memory nodes are keyed by the episodic row id.
    Mentions with a speaker are private data at runtime.
    """

    def link_memory_to_concepts(
        self,
        memory_id: int,
        memory_content: str,
        concept_names: Sequence[str],
        relevance: float = 0.5,
        extraction_method: str = "keyword",
    ) -> int:
        """Create or refresh the memory node and link it to each named concept
        that exists; names with no concept are skipped. Return the links made."""
        ...

    def list_concept_names(self, limit: Optional[int] = None) -> List[str]:
        """Return concept names in ascending order."""
        ...

    def merge_concept(
        self, name: str, description: str = "", category: str = "conversation"
    ) -> None:
        """Create the concept when absent (idempotent)."""
        ...

    def link_concept_to_conversation(
        self, name: str, conversation_id: str, speaker: str
    ) -> None:
        """Merge the concept and record a mention in a conversation by a speaker."""
        ...

    def relate_concepts(
        self,
        first: str,
        second: str,
        relationship: str = "relates_to",
        strength: float = 0.5,
    ) -> bool:
        """Relate two existing concepts; False when either does not exist."""
        ...

    def get_related_concepts(
        self, concept_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return ``{"concept", "relationship", "strength"}`` rows, strongest first."""
        ...

    def delete_memory_nodes(self, memory_ids: Sequence[int]) -> int:
        """Delete memory nodes and their links; return the nodes removed."""
        ...

    def close(self) -> None:
        """Release the driver."""
        ...

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available`` and ``last_error`` at least."""
        ...


@runtime_checkable
class ConceptMatcher(Protocol):
    """Matches memory text against concept names already in the graph."""

    def match_concepts(
        self, text: str, known_concepts: Sequence[str], fuzzy: bool = True
    ) -> List[Tuple[str, float]]:
        """Return ``(concept, relevance)`` pairs, most relevant first."""
        ...


@runtime_checkable
class EpisodeConsolidator(Protocol):
    """The substrate memory consolidator the orchestrator and the memory
    manager call: episodes in, semantic patterns out."""

    def add_episodic_memory(
        self,
        memory_type: str,
        emotional_state: Any,
        subconscious_pattern: Any = None,
        context: Optional[Dict[str, Any]] = None,
        consciousness_level: Optional[float] = None,
    ) -> Any:
        """Record one episode and return it."""
        ...

    def consolidate_memories(
        self, memories: Optional[Sequence[Any]] = None, sleep_duration: str = "short"
    ) -> Dict[str, Any]:
        """Consolidate episodes into patterns and return a report."""
        ...

    def get_semantic_patterns(
        self, pattern_type: Optional[str] = None, min_frequency: int = 1
    ) -> List[Any]:
        """Return the consolidated patterns, optionally filtered."""
        ...
