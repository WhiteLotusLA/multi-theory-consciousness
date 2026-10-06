"""Thought records and the stream state of the continuous thought machine.

A ``ThoughtRecord`` is one background thought: its text, type, intensity,
salience, links to the thoughts before and after it, and the memory-lifecycle
fields the consolidation pass and the pruner read. Thought text, emotions,
associated memory ids and processing context are private runtime data; the
module writes nothing itself.

One document shape serves the in-memory and the MongoDB store. ``to_doc()``
emits exactly the thought-stream document of
``mtc.neural.mongodb_schemas.ThoughtStream`` (``thought_text``,
``thought_type``, ``stream_id``, ``parent_thought_id``, ``child_thoughts``,
``emotional_state``, ``associated_memories``, ``triggers``, ``coherence_score``,
``metadata`` holding ``should_express`` and ``intensity``, and the lifecycle
fields) plus ``thought_id``, so the pruner and the consolidation pass read every
writer's documents. ``salience_score`` is stored as ``coherence_score``: the
pruner's coherence floor is therefore a salience floor for these thoughts.
``from_doc()`` reads that shape back and raises ``ValueError`` naming the field
for a document it cannot read honestly (an unknown thought type, or no
``intensity`` or salience), so a foreign document is never given invented
values.

Timestamps are timezone-aware UTC: the constructor rejects a naive datetime
and converts any other offset to UTC, and ``from_doc`` reads a naive value (what
a driver without timezone support returns) as UTC. No config keys.
"""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class ThoughtType(str, Enum):
    """The kinds of background thought."""

    REFLECTION = "reflection"
    QUESTION = "question"
    ASSOCIATION = "association"
    EMOTION = "emotion"
    MEMORY_REPLAY = "memory_replay"
    CREATIVE = "creative"
    PLANNING = "planning"
    SELF_AWARE = "self_aware"
    DREAM = "dream"
    SPONTANEOUS = "spontaneous"


class ThoughtIntensity(str, Enum):
    """How salient a thought is, from barely present to hard to ignore."""

    WHISPER = "whisper"
    QUIET = "quiet"
    MODERATE = "moderate"
    STRONG = "strong"
    URGENT = "urgent"

    @property
    def level(self) -> float:
        """The numeric level, 0.1 (whisper) to 0.9 (urgent)."""
        return INTENSITY_LEVELS[self.value]

    @classmethod
    def from_level(cls, level: float) -> "ThoughtIntensity":
        """The intensity whose level is nearest to a number in [0, 1]."""
        value = float(level)
        if not math.isfinite(value):
            raise ValueError("an intensity level must be a finite number")
        return min(cls, key=lambda member: abs(member.level - value))


INTENSITY_LEVELS: Dict[str, float] = {
    "whisper": 0.1,
    "quiet": 0.3,
    "moderate": 0.5,
    "strong": 0.7,
    "urgent": 0.9,
}

CONSCIOUSNESS_STATES = ("awake", "drowsy", "dreaming", "deep_sleep")


def new_thought_id() -> str:
    """Return a new unique thought id."""
    return f"thought_{uuid.uuid4().hex}"


def _aware(moment: Optional[datetime], name: str) -> Optional[datetime]:
    if moment is None:
        return None
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return moment.astimezone(timezone.utc)


def read_utc(value: Any) -> Optional[datetime]:
    """Read a stored timestamp; a naive value is UTC."""
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise ValueError(f"{value!r} is not a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass
class ThoughtRecord:
    """One background thought and its lifecycle fields.

    ``stream_id`` names the burst the thought belongs to and defaults to the
    thought's own id; ``previous_thought_id`` and ``leads_to_thought_id`` hold
    thought ids, never stream ids. ``importance_score`` and
    ``retention_strength`` are None until the consolidation pass measures them.
    ``consciousness_state`` is the thought mode the thought was made in
    (``CONSCIOUSNESS_STATES``), or None when nothing recorded one; an absent
    state is never read as ``awake``.
    """

    thought_id: str
    timestamp: datetime
    thought_type: ThoughtType
    content: str
    intensity: ThoughtIntensity
    salience_score: float
    triggered_by: Optional[str] = None
    associated_memories: List[str] = field(default_factory=list)
    emotional_context: Dict[str, float] = field(default_factory=dict)
    previous_thought_id: Optional[str] = None
    leads_to_thought_id: Optional[str] = None
    consciousness_state: Optional[str] = None
    processing_context: Dict[str, Any] = field(default_factory=dict)
    should_express: bool = False
    expressed_at: Optional[datetime] = None
    stream_id: Optional[str] = None
    access_count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    importance_score: Optional[float] = None
    retention_strength: Optional[float] = None
    is_protected: bool = False
    protection_reason: Optional[str] = None
    consolidated_into: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.thought_id:
            raise ValueError("thought_id must not be empty")
        self.thought_type = ThoughtType(self.thought_type)
        self.intensity = ThoughtIntensity(self.intensity)
        score = float(self.salience_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("salience_score must be a number in [0, 1]")
        self.salience_score = score
        self.timestamp = _aware(self.timestamp, "timestamp")
        self.expressed_at = _aware(self.expressed_at, "expressed_at")
        self.first_seen = _aware(self.first_seen, "first_seen")
        self.last_seen = _aware(self.last_seen, "last_seen")
        if self.stream_id is None:
            self.stream_id = self.thought_id

    def copy(self) -> "ThoughtRecord":
        """Return an independent copy (nested values are copied too)."""
        return copy.deepcopy(self)

    def to_doc(self) -> Dict[str, Any]:
        """Return the thought-stream document (see the module docstring)."""
        return {
            "thought_id": self.thought_id,
            "timestamp": self.timestamp,
            "stream_id": self.stream_id,
            "thought_text": self.content,
            "thought_type": self.thought_type.value,
            "emotional_state": dict(self.emotional_context),
            "associated_memories": list(self.associated_memories),
            "triggers": [] if self.triggered_by is None else [self.triggered_by],
            "coherence_score": self.salience_score,
            "parent_thought_id": self.previous_thought_id,
            "child_thoughts": (
                [] if self.leads_to_thought_id is None else [self.leads_to_thought_id]
            ),
            "metadata": {
                "should_express": self.should_express,
                "intensity": self.intensity.value,
                "expressed_at": self.expressed_at,
                "consciousness_state": self.consciousness_state,
                "processing_context": copy.deepcopy(self.processing_context),
                "first_seen": self.first_seen,
                "last_seen": self.last_seen,
            },
            "access_count": self.access_count,
            "importance_score": self.importance_score,
            "retention_strength": self.retention_strength,
            "is_protected": self.is_protected,
            "protection_reason": self.protection_reason,
            "consolidated_into": self.consolidated_into,
        }

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any]) -> "ThoughtRecord":
        """Build a record from a thought-stream document.

        Raises ``ValueError`` naming the field when the document has no
        readable thought id, text, type, timestamp, ``metadata.intensity`` or
        salience (``coherence_score``): nothing is invented in their place.
        """
        thought_id = doc.get("thought_id") or doc.get("id")
        if not thought_id:
            raise ValueError("thought document has no thought_id")
        text = doc.get("thought_text")
        if not isinstance(text, str):
            raise ValueError("thought document has no thought_text")
        try:
            thought_type = ThoughtType(doc.get("thought_type"))
        except ValueError as exc:
            raise ValueError(
                f"thought_type {doc.get('thought_type')!r} is not a known type"
            ) from exc
        meta = doc.get("metadata") or {}
        try:
            intensity = ThoughtIntensity(meta.get("intensity"))
        except ValueError as exc:
            raise ValueError(
                "thought document has no readable metadata.intensity"
            ) from exc
        salience = doc.get("coherence_score")
        if salience is None:
            raise ValueError("thought document has no coherence_score (salience)")
        timestamp = read_utc(doc.get("timestamp"))
        if timestamp is None:
            raise ValueError("thought document has no timestamp")
        triggers = list(doc.get("triggers") or [])
        children = list(doc.get("child_thoughts") or [])
        return cls(
            thought_id=str(thought_id),
            timestamp=timestamp,
            thought_type=thought_type,
            content=text,
            intensity=intensity,
            salience_score=salience,
            triggered_by=triggers[0] if triggers else None,
            associated_memories=list(doc.get("associated_memories") or []),
            emotional_context=dict(doc.get("emotional_state") or {}),
            previous_thought_id=doc.get("parent_thought_id"),
            leads_to_thought_id=children[0] if children else None,
            consciousness_state=meta.get("consciousness_state"),
            processing_context=copy.deepcopy(meta.get("processing_context") or {}),
            should_express=bool(meta.get("should_express", False)),
            expressed_at=read_utc(meta.get("expressed_at")),
            stream_id=doc.get("stream_id"),
            access_count=int(doc.get("access_count") or 0),
            first_seen=read_utc(meta.get("first_seen")),
            last_seen=read_utc(meta.get("last_seen")),
            importance_score=doc.get("importance_score"),
            retention_strength=doc.get("retention_strength"),
            is_protected=bool(doc.get("is_protected", False)),
            protection_reason=doc.get("protection_reason"),
            consolidated_into=doc.get("consolidated_into"),
        )


@dataclass
class ThoughtStreamState:
    """The running state of the thought stream.

    ``thought_frequency`` (thoughts per second) and ``current_intensity`` (in
    [0, 1]) are measurements of the buffered thoughts: None until the buffer
    can compute them, and None again whenever it cannot, never an assumed
    value. The counters cover the current day. ``emotional_tone`` is the
    emotional context of the newest thought that carried one.
    """

    current_thought_id: Optional[str] = None
    last_thought_time: Optional[datetime] = None
    thought_frequency: Optional[float] = None
    current_intensity: Optional[float] = None
    dominant_themes: List[str] = field(default_factory=list)
    recurring_patterns: Dict[str, int] = field(default_factory=dict)
    emotional_tone: Dict[str, float] = field(default_factory=dict)
    total_thoughts_today: int = 0
    thoughts_per_type: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict for status and logging."""
        return {
            "current_thought_id": self.current_thought_id,
            "last_thought_time": (
                self.last_thought_time.isoformat() if self.last_thought_time else None
            ),
            "thought_frequency": self.thought_frequency,
            "current_intensity": self.current_intensity,
            "dominant_themes": list(self.dominant_themes),
            "recurring_patterns": dict(self.recurring_patterns),
            "emotional_tone": dict(self.emotional_tone),
            "total_thoughts_today": self.total_thoughts_today,
            "thoughts_per_type": dict(self.thoughts_per_type),
        }
