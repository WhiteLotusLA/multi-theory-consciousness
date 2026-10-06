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

The safety audit, alert and privacy audit protocols (pinned by
``tests/memory/test_audit_store_contracts.py``) are async and follow the same
rules, ordering by ``timestamp`` with later writes first among equal instants.
Their documents hold proposed actions, alert context, user ids and access
records, all private data at runtime. ``ALERT_SEVERITY_ORDER``,
``severities_at_or_above`` and ``normalize_safety_audit`` are shared by every
backend so the filters and field checks agree.

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
- PostgreSQL table ``safety_audit`` with the columns ``SAFETY_AUDIT_FIELDS``
  and a uuid ``id``; MongoDB collections ``alerts`` (unique ``alert_id``) and
  ``privacy_audit_log`` (unique ``event_id``).
- PostgreSQL tables ``conversations``, ``messages`` (with the ``to_speaker``
  addressee column) and ``emotional_history`` for ``ConversationRepository``,
  whose message reads are oldest first (the replay order of a history). The
  rows hold conversation text and speaker names, private data at runtime.
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


# Message roles a conversation repository stores: the brain's history roles
# plus ``tool``, the role of a plugin's reply stored under the plugin's own
# name (never the system's words). The DDL's role check lists the same four.
MESSAGE_ROLES = ("user", "assistant", "system", "tool")
# Column widths of the conversation schema; both backends check them.
SPEAKER_MAX_CHARS = 200
ROLE_MAX_CHARS = 20
EMOTION_LABEL_MAX_CHARS = 50
EMOTIONAL_STATE_FIELDS = (
    "timestamp",
    "primary_emotion",
    "intensity",
    "valence",
    "arousal",
    "dominance",
    "trigger_context",
)
_EMOTIONAL_NUMBERS = ("intensity", "valence", "arousal", "dominance")


def conversation_uuid(conversation_id: Any) -> str:
    """Return the canonical string form of a conversation id.

    A conversation id is a UUID (the PostgreSQL column type); anything that
    does not parse as one raises ValueError, so both backends refuse the same
    ids.
    """
    try:
        return str(uuid.UUID(str(conversation_id)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("conversation_id must be a UUID string") from exc


def check_message_fields(
    speaker: str, role: str, content: str, to_speaker: Optional[str]
) -> None:
    """Raise ValueError when a message does not fit the conversation schema.

    ``speaker`` must be non-empty and at most ``SPEAKER_MAX_CHARS`` long,
    ``role`` one of ``MESSAGE_ROLES``, ``content`` a string and ``to_speaker``
    None or a non-empty string within the speaker width.
    """
    if not isinstance(speaker, str) or not speaker.strip():
        raise ValueError("speaker must be a non-empty string")
    if len(speaker) > SPEAKER_MAX_CHARS:
        raise ValueError(f"speaker is longer than {SPEAKER_MAX_CHARS} characters")
    if role not in MESSAGE_ROLES:
        raise ValueError(f"role must be one of {MESSAGE_ROLES}")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    if to_speaker is not None:
        if not isinstance(to_speaker, str) or not to_speaker.strip():
            raise ValueError("to_speaker must be None or a non-empty string")
        if len(to_speaker) > SPEAKER_MAX_CHARS:
            raise ValueError(
                f"to_speaker is longer than {SPEAKER_MAX_CHARS} characters"
            )


def normalize_emotional_state(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Check an emotional-history record and return a full copy.

    Every key must be one of ``EMOTIONAL_STATE_FIELDS``; a missing key reads as
    None (unmeasured), never as a neutral value. The four coordinates are None
    or finite numbers, ``primary_emotion`` is None or a label within
    ``EMOTION_LABEL_MAX_CHARS``, and a naive ``timestamp`` raises ValueError (a
    missing one is stamped by the store's clock).
    """
    unknown = sorted(set(record) - set(EMOTIONAL_STATE_FIELDS))
    if unknown:
        raise ValueError(f"unknown emotional-history fields {unknown}")
    row: Dict[str, Any] = {name: record.get(name) for name in EMOTIONAL_STATE_FIELDS}
    for name in _EMOTIONAL_NUMBERS:
        value = row[name]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number or None")
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{name} must be finite")
        row[name] = float(value)
    label = row["primary_emotion"]
    if label is not None:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("primary_emotion must be None or a non-empty string")
        if len(label) > EMOTION_LABEL_MAX_CHARS:
            raise ValueError(
                f"primary_emotion is longer than {EMOTION_LABEL_MAX_CHARS} characters"
            )
    if row["trigger_context"] is not None and not isinstance(
        row["trigger_context"], str
    ):
        raise ValueError("trigger_context must be None or a string")
    moment = row["timestamp"]
    if moment is not None:
        if not isinstance(moment, datetime):
            raise ValueError("timestamp must be a datetime")
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
    return row


@runtime_checkable
class ConversationRepository(Protocol):
    """Conversations, messages and emotional history (private data).

    The rows hold conversation text, speaker names and the trigger context of
    emotional states, all private at runtime. The PostgreSQL implementation
    writes the ``conversations``, ``messages`` and ``emotional_history``
    tables of ``mtc.memory.conversation_store.CONVERSATION_SCHEMA_SQL``.

    Contract (pinned by ``tests/memory/test_conversation_repository_contract.py``):

    - Conversation ids are UUID strings (``conversation_uuid``); ids come back
      in canonical form. A conversation lists its participants; a message's
      ``role`` is one of ``MESSAGE_ROLES`` (``check_message_fields``).
    - A message's ``to_speaker`` is None when it belongs to the conversation
      with the system and names another addressee otherwise. Reads keep a row
      whose ``to_speaker`` is None or equals the ``addressee`` argument, so a
      message addressed elsewhere never enters the system's history.
    - Message reads (``recent_messages`` and ``timeline``) return the newest
      ``limit`` rows inside the window, ordered OLDEST first (the order a
      history is replayed in), unlike the newest-first rule of the other
      stores. A row has ``id``, ``conversation_id``, ``speaker``, ``role``,
      ``content``, ``timestamp`` and ``to_speaker``. The window is the last
      ``hours`` hours on the store's clock (rows strictly after the cutoff).
    - ``recent_emotional_states`` is newest first. A missing coordinate is
      None (``normalize_emotional_state``).
    - Timestamps are timezone-aware and come back in UTC; a naive one raises
      ValueError before anything is written. Rows are copies.
    - A duplicate conversation id and a message for an unknown conversation
      raise ValueError.
    """

    async def ensure_schema(self) -> None:
        """Create the tables when missing (idempotent)."""
        ...

    async def add_conversation(
        self,
        conversation_id: str,
        participants: Sequence[str],
        started_at: datetime,
        summary: Optional[str] = None,
    ) -> str:
        """Open a conversation and return its canonical id."""
        ...

    async def end_conversation(
        self,
        conversation_id: str,
        ended_at: datetime,
        summary: Optional[str] = None,
    ) -> bool:
        """Close a conversation; a None summary keeps the stored one. False
        when no such conversation exists."""
        ...

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Return ``id``, ``participants``, ``summary``, ``started_at`` and
        ``ended_at`` of a conversation, or None."""
        ...

    async def add_message(
        self,
        conversation_id: str,
        speaker: str,
        role: str,
        content: str,
        timestamp: datetime,
        to_speaker: Optional[str] = None,
    ) -> int:
        """Store one message and return its id."""
        ...

    async def add_emotional_state(self, record: Dict[str, Any]) -> int:
        """Store one emotional-history row and return its id."""
        ...

    async def recent_emotional_states(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the latest emotional-history rows, newest first."""
        ...

    async def latest_message_time(
        self, participant: Optional[str] = None, *, addressee: Optional[str] = None
    ) -> Optional[datetime]:
        """Return the newest message timestamp (of conversations the
        participant is in, when given), or None when there is none."""
        ...

    async def recent_messages(
        self,
        participant: str,
        hours: float,
        limit: int,
        *,
        addressee: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the participant's messages of the last ``hours`` hours,
        oldest first."""
        ...

    async def timeline(
        self, hours: float, limit: int, *, addressee: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return every conversation's messages of the last ``hours`` hours,
        oldest first."""
        ...

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available`` and ``last_error``."""
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


# Alert severities from least to most severe. A ``min_severity`` filter keeps
# the named level and every level after it.
ALERT_SEVERITY_ORDER = ("info", "warning", "urgent", "critical")

# Columns of one safety audit row (the ``safety_audit`` table) besides ``id``.
SAFETY_AUDIT_FIELDS = (
    "timestamp",
    "action_proposed",
    "safety_level",
    "checks_performed",
    "decision",
    "reasoning",
    "requires_review",
)
SAFETY_AUDIT_REQUIRED = ("action_proposed", "safety_level", "decision")


def severities_at_or_above(min_severity: str) -> Tuple[str, ...]:
    """Return the severities at or above ``min_severity``, least severe first.

    An unknown level raises ValueError naming ``ALERT_SEVERITY_ORDER``.
    """
    if min_severity not in ALERT_SEVERITY_ORDER:
        raise ValueError(
            f"unknown alert severity {min_severity!r}; "
            f"expected one of {ALERT_SEVERITY_ORDER}"
        )
    return ALERT_SEVERITY_ORDER[ALERT_SEVERITY_ORDER.index(min_severity) :]


def normalize_safety_audit(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Check one safety audit record and fill its optional fields.

    The required fields are ``SAFETY_AUDIT_REQUIRED``; a missing one, or a key
    outside ``SAFETY_AUDIT_FIELDS``, raises ValueError. ``checks_performed``
    defaults to an empty mapping, ``reasoning`` to an empty string and
    ``requires_review`` to False. ``timestamp`` is left to the store (its
    current UTC time when absent). The result is a new dict.
    """
    unknown = sorted(set(record) - set(SAFETY_AUDIT_FIELDS))
    if unknown:
        raise ValueError(f"safety audit record has unknown fields {unknown}")
    missing = [name for name in SAFETY_AUDIT_REQUIRED if record.get(name) is None]
    if missing:
        raise ValueError(f"safety audit record lacks required fields {missing}")
    row = dict(record)
    row["checks_performed"] = dict(row.get("checks_performed") or {})
    row["reasoning"] = row.get("reasoning") or ""
    row["requires_review"] = bool(row.get("requires_review", False))
    return row


@runtime_checkable
class SafetyAuditStore(Protocol):
    """Safety decisions, one row per checked action (private content).

    A record holds ``SAFETY_AUDIT_FIELDS`` (checked by
    ``normalize_safety_audit``); the proposed action can contain conversation
    text. Rows come back with their ``id`` (a string) and an aware
    ``timestamp``, newest first. The PostgreSQL implementation writes the
    ``safety_audit`` table.
    """

    async def add_audit(self, record: Dict[str, Any]) -> str:
        """Store one audit row and return its assigned id.

        A missing ``timestamp`` means the store's current UTC time; a naive
        one, a missing required field or an unknown field raises ValueError.
        """
        ...

    async def pending_review(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return rows whose ``requires_review`` is true, newest first."""
        ...

    async def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the latest rows, newest first."""
        ...


@runtime_checkable
class AlertStore(Protocol):
    """Alerts keyed by ``alert_id`` (they can carry private context).

    An alert document needs ``alert_id``, ``alert_type`` and ``severity`` (one
    of ``ALERT_SEVERITY_ORDER``); other fields are stored as given. A missing
    ``timestamp`` means the store's current UTC time; a naive one raises
    ValueError. The MongoDB implementation writes the ``alerts`` collection.
    """

    async def add_alert(self, alert: Dict[str, Any]) -> None:
        """Store one alert; a missing key field or unknown severity raises
        ValueError."""
        ...

    async def recent_alerts(
        self,
        limit: int = 100,
        alert_type: Optional[str] = None,
        min_severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return alerts newest first, optionally of one type and at or above
        a severity (an unknown severity raises ValueError)."""
        ...

    async def acknowledge(self, alert_id: str) -> bool:
        """Mark an alert acknowledged (``acknowledged`` true and
        ``acknowledged_at`` the store's current UTC time); False when no such
        alert exists."""
        ...


@runtime_checkable
class PrivacyAuditStore(Protocol):
    """Privacy audit events (user ids and access records; private data).

    An event document needs ``event_id``, ``event_type`` and ``user_id``;
    other fields are stored as given. A missing ``timestamp`` means the
    store's current UTC time; a naive one raises ValueError. The MongoDB
    implementation writes the ``privacy_audit_log`` collection.
    """

    async def add_event(self, event: Dict[str, Any]) -> None:
        """Store one event; a missing key field raises ValueError."""
        ...

    async def events(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return events at or after ``since``, newest first, optionally of one
        user and one type."""
        ...
