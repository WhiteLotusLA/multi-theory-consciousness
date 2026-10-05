"""
Thought Gist
============

``ThoughtGist`` is the compressed summary of a cluster of thoughts that the
consolidation pipeline writes and the ``ThoughtGistStore`` protocol persists.
Documents carry the thought text's summary, the ids of the source thoughts and
scores derived from them, so a stored gist is private conversation content at
runtime and lives only in gitignored or operator-controlled stores.

``to_document`` and ``from_document`` are the single serialization pair every
backend uses. The count of source thoughts is stored under one key,
``source_count``. Timestamps are timezone-aware UTC: the constructor rejects a
naive datetime and converts any other offset to UTC, while ``from_document``
reads a naive datetime (what a driver without timezone support returns) as
UTC. No config keys; the schema artifact of the backing collection belongs to
the store that persists it.
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _as_utc(moment: datetime, name: str) -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return moment.astimezone(timezone.utc)


def _read_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Read a stored timestamp; a naive value is UTC."""
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass
class ThoughtGist:
    """A compressed gist of clustered thoughts from the consolidation cycle."""

    gist_id: str
    created_at: datetime
    summary: str
    source_count: int
    source_thought_ids: List[str]
    dominant_emotion: str

    # Temporal pattern data
    emotional_arc: Dict[str, Any] = field(default_factory=dict)
    time_span: Dict[str, Any] = field(default_factory=dict)
    burst_windows: List[Dict[str, Any]] = field(default_factory=list)

    # Scoring
    avg_salience: float = 0.0
    total_access_count: int = 0
    content_variations: List[str] = field(default_factory=list)
    importance_score: float = 0.0
    retention_strength: float = 1.0

    # Protection
    is_protected: bool = False
    protection_reason: Optional[str] = None

    # Retrieval tracking
    retrieval_count: int = 0
    last_retrieved: Optional[datetime] = None

    def __post_init__(self) -> None:
        self.created_at = _as_utc(self.created_at, "created_at")
        if self.last_retrieved is not None:
            self.last_retrieved = _as_utc(self.last_retrieved, "last_retrieved")

    def to_document(self) -> Dict[str, Any]:
        """Return a storable document; nested values are copies."""
        return {
            "gist_id": self.gist_id,
            "created_at": self.created_at,
            "summary": self.summary,
            "source_count": self.source_count,
            "source_thought_ids": list(self.source_thought_ids),
            "dominant_emotion": self.dominant_emotion,
            "emotional_arc": copy.deepcopy(self.emotional_arc),
            "time_span": copy.deepcopy(self.time_span),
            "burst_windows": copy.deepcopy(self.burst_windows),
            "avg_salience": self.avg_salience,
            "total_access_count": self.total_access_count,
            "content_variations": list(self.content_variations),
            "importance_score": self.importance_score,
            "retention_strength": self.retention_strength,
            "is_protected": self.is_protected,
            "protection_reason": self.protection_reason,
            "retrieval_count": self.retrieval_count,
            "last_retrieved": self.last_retrieved,
        }

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> "ThoughtGist":
        """Build a gist from a stored document.

        ``gist_id``, ``created_at``, ``summary`` and ``source_count`` are
        required (a missing one raises KeyError); every other field falls back
        to the dataclass default.
        """
        return cls(
            gist_id=doc["gist_id"],
            created_at=_read_utc(doc["created_at"]),
            summary=doc["summary"],
            source_count=doc["source_count"],
            source_thought_ids=list(doc.get("source_thought_ids", [])),
            dominant_emotion=doc.get("dominant_emotion", "neutral"),
            emotional_arc=copy.deepcopy(doc.get("emotional_arc", {})),
            time_span=copy.deepcopy(doc.get("time_span", {})),
            burst_windows=copy.deepcopy(doc.get("burst_windows", [])),
            avg_salience=doc.get("avg_salience", 0.0),
            total_access_count=doc.get("total_access_count", 0),
            content_variations=list(doc.get("content_variations", [])),
            importance_score=doc.get("importance_score", 0.0),
            retention_strength=doc.get("retention_strength", 1.0),
            is_protected=doc.get("is_protected", False),
            protection_reason=doc.get("protection_reason"),
            retrieval_count=doc.get("retrieval_count", 0),
            last_retrieved=_read_utc(doc.get("last_retrieved")),
        )
