"""
Store Protocols
===============

Domain repositories the consciousness modules persist through. Every protocol
is runtime-checkable and every method is async. The protocols speak in records
and documents, not queries, so an implementation owns its own query language.
A module given no store uses the in-memory implementation from
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

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable


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
