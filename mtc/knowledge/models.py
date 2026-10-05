"""
Knowledge Graph Models
======================

Pydantic models of the nodes and relationships the graph manager writes to
Neo4j. Node labels: ``Person``, ``Concept``, ``Experience``, ``Emotion``,
``Memory`` (a reference to an episodic row) and ``Topic``. Relationship types:
``TAUGHT`` (Person to Concept), ``LEARNED`` (Person to Concept), ``RELATES_TO``
(Concept to Concept), ``REFERENCES_IN`` (Memory to Concept) and
``MENTIONED_IN`` (Concept to the conversation's Experience, with the speaker).
Only relationships that some module writes are declared here.

``Concept`` carries exactly the properties the MERGE paths of the graph
manager write (``description``, ``mention_count``, ``last_mentioned``, ...);
``understanding_level`` is optional and never assumed. ``Person`` carries no
bond, trust or comfort value, and ``first_met`` is None unless known.
Timestamps default to timezone-aware UTC. The schema examples are synthetic.

``Person`` names and ``MENTIONED_IN`` speakers are real people's names at
runtime and are private data; the graph lives in an operator-controlled store.

``node_to_cypher_properties`` and ``relationship_to_cypher_properties`` turn a
model into a Neo4j property map: UUIDs and datetimes become strings and None
values are left out (Neo4j stores no nulls).
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_EXAMPLE_TIME = "2000-01-01T00:00:00+00:00"

# ============================================================================
# Nodes
# ============================================================================


class Person(BaseModel):
    """Someone the system knows (a real name at runtime)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Alice",
                "relationship": "friend",
                "first_met": _EXAMPLE_TIME,
                "interaction_count": 3,
            }
        }
    )

    id: UUID = Field(default_factory=uuid4)
    name: str
    relationship: str
    first_met: Optional[datetime] = None
    interaction_count: int = Field(default=0, ge=0)


class Concept(BaseModel):
    """An idea in the graph, keyed by its unique name."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "river",
                "category": "nature",
                "description": "A natural stream of flowing water",
                "importance": 0.5,
                "mention_count": 2,
                "learned_at": _EXAMPLE_TIME,
                "last_mentioned": _EXAMPLE_TIME,
            }
        }
    )

    id: UUID = Field(default_factory=uuid4)
    name: str
    category: str
    description: str = ""
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    mention_count: int = Field(default=0, ge=0)
    learned_at: datetime = Field(default_factory=_utcnow)
    last_mentioned: Optional[datetime] = None
    understanding_level: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Experience(BaseModel):
    """An event; a conversation's node is keyed by ``conversation_id``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "A conversation about rivers",
                "experience_type": "conversation",
                "conversation_id": "conv-0001",
                "occurred_at": _EXAMPLE_TIME,
                "emotional_valence": 0.2,
                "arousal_level": 0.4,
                "importance": 0.5,
            }
        }
    )

    id: UUID = Field(default_factory=uuid4)
    description: str = ""
    experience_type: str = "conversation"
    conversation_id: Optional[str] = None
    occurred_at: datetime = Field(default_factory=_utcnow)
    emotional_valence: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    arousal_level: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class Emotion(BaseModel):
    """A named emotional state with its valence and arousal."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "curiosity",
                "valence": 0.6,
                "arousal": 0.7,
                "category": "complex",
            }
        }
    )

    id: UUID = Field(default_factory=uuid4)
    name: str
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    category: str = "basic"


class Memory(BaseModel):
    """A reference node for an episodic row (``postgres_id``)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "00000000-0000-0000-0000-000000000007",
                "postgres_id": 7,
                "memory_type": "episodic",
                "content_preview": "A walk along the river path",
                "created_at": _EXAMPLE_TIME,
                "access_count": 0,
            }
        }
    )

    id: str
    postgres_id: Optional[int] = None
    memory_type: str = "episodic"
    content_preview: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)


class Topic(BaseModel):
    """A recurring conversation theme."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "gardening",
                "frequency": 4,
                "first_mentioned": _EXAMPLE_TIME,
                "last_mentioned": _EXAMPLE_TIME,
            }
        }
    )

    id: UUID = Field(default_factory=uuid4)
    name: str
    frequency: int = Field(default=0, ge=0)
    first_mentioned: datetime = Field(default_factory=_utcnow)
    last_mentioned: datetime = Field(default_factory=_utcnow)


# ============================================================================
# Relationships
# ============================================================================


class Taught(BaseModel):
    """``(Person)-[:TAUGHT]->(Concept)``.

    ``effectiveness`` is None until something measured it; the property
    helpers then leave it off the edge (Neo4j stores no nulls).
    """

    when: datetime = Field(default_factory=_utcnow)
    method: str = "conversation"
    effectiveness: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Learned(BaseModel):
    """``(Person)-[:LEARNED]->(Concept)``.

    ``confidence``, ``depth`` and ``proficiency`` are None until measured and
    are then left off the edge rather than written as a stand-in value.
    """

    when: datetime = Field(default_factory=_utcnow)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    depth: Optional[str] = None
    proficiency: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class RelatesTo(BaseModel):
    """``(Concept)-[:RELATES_TO]-(Concept)``."""

    relationship_type: str = "relates_to"
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    discovered_at: datetime = Field(default_factory=_utcnow)


class ReferencesIn(BaseModel):
    """``(Memory)-[:REFERENCES_IN]->(Concept)``."""

    when: datetime = Field(default_factory=_utcnow)
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_method: str = "manual"


class MentionedIn(BaseModel):
    """``(Concept)-[:MENTIONED_IN]->(Experience)``, one edge per speaker.

    ``speaker`` is a real name at runtime (private data).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"speaker": "Alice", "when": _EXAMPLE_TIME, "mentions": 1}
        }
    )

    speaker: str
    when: datetime = Field(default_factory=_utcnow)
    mentions: int = Field(default=1, ge=1)


# ============================================================================
# Property helpers
# ============================================================================


def _cypher_value(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def node_to_cypher_properties(node: BaseModel) -> Dict[str, Any]:
    """Return a node model as a Neo4j property map (no None values)."""
    return {
        key: _cypher_value(value)
        for key, value in node.model_dump().items()
        if value is not None
    }


def relationship_to_cypher_properties(rel: BaseModel) -> Dict[str, Any]:
    """Return a relationship model as a Neo4j property map (no None values)."""
    return node_to_cypher_properties(rel)
