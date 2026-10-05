"""
Knowledge Graph Manager
=======================

The Neo4j implementation of the ``GraphStore`` protocol (``mtc.memory.protocols``)
plus the reference implementation's node, relationship, query and extraction
methods. Every node write is a ``MERGE`` keyed by a unique property (a concept
or person name, a memory or node id, a conversation id), so running a write
twice never duplicates a node. Every query is a module constant and every value
travels as a parameter; nothing is interpolated into Cypher.

Schema artifact: ``CONSTRAINTS`` (unique ``Concept.name``, a unique ``id`` per
label, unique ``Experience.conversation_id`` and an index on ``Person.name``,
each ``IF NOT EXISTS``). ``bootstrap_constraints()`` applies it as a safety net;
Phase 7's init script is canonical.

Seeds: ``seed_core_entities()`` is what Phase 7's bootstrap calls when the
wizard enables Neo4j. It merges a ``Person`` node for ``SYSTEM_NAME``
(relationship ``self``) and, unless ``include_creator`` is false, one for
``CREATOR_NAME`` with ``CREATOR_RELATIONSHIP`` and ``first_met`` from
``SYSTEM_BIRTH_DATE`` only when that key is set; then the ``SEED_CONCEPTS``.
No bond, trust or comfort value is written: the creator node exists for
display and for the who-taught query, never to grant trust (trust comes from
an injected lookup elsewhere). Seeding twice creates nothing.

Writes the memory layer uses: ``link_memory_to_concepts`` (``Memory`` nodes
keyed by the episodic row id, ``REFERENCES_IN`` links to existing concepts),
``list_concept_names``, ``merge_concept``, ``link_concept_to_conversation``
(``MENTIONED_IN`` from a concept to the conversation's ``Experience`` node, one
edge per speaker), ``relate_concepts`` and ``delete_memory_nodes``.
``process_llm_extraction`` writes what the LLM concept extractor measured; a
``no_answer`` or ``unparseable`` extraction writes nothing.

Connection: the driver is built from ``Neo4jConfig`` (``NEO4J_URI``,
``NEO4J_USER``, ``NEO4J_PASSWORD``, ``NEO4J_DATABASE`` read when the manager is
built) on the first call or by ``connect()``; the ``neo4j`` package is imported
there through ``require_client``. An injected driver belongs to the caller and
is never closed here. A server that cannot be reached raises
``StoreUnavailable`` and is reported by ``status()``; any other driver error
(a syntax or constraint error) propagates unchanged. ``SYSTEM_NAME``,
``CREATOR_NAME``, ``CREATOR_RELATIONSHIP`` and ``SYSTEM_BIRTH_DATE`` are read at
call time. Methods are synchronous; async callers use ``asyncio.to_thread``.
Timestamps are ISO strings from the injectable clock (timezone-aware UTC).

Private data: ``Person`` nodes and ``MENTIONED_IN`` speakers hold real names at
runtime, and memory previews hold conversation text; the graph lives in an
operator-controlled store and never in version control.
"""

import logging
from datetime import date, datetime, time, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from mtc.core.config import Settings, get_settings
from mtc.knowledge.config import Neo4jConfig
from mtc.knowledge.models import (
    Concept,
    Emotion,
    Experience,
    Learned,
    Memory,
    Person,
    ReferencesIn,
    RelatesTo,
    Taught,
    Topic,
    node_to_cypher_properties,
    relationship_to_cypher_properties,
)
from mtc.memory.protocols import StoreUnavailable

logger = logging.getLogger(__name__)

CONSTRAINTS: Tuple[str, ...] = (
    "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS "
    "FOR (n:Concept) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT person_id_unique IF NOT EXISTS "
    "FOR (n:Person) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS "
    "FOR (n:Concept) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS "
    "FOR (n:Memory) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT experience_id_unique IF NOT EXISTS "
    "FOR (n:Experience) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT emotion_id_unique IF NOT EXISTS "
    "FOR (n:Emotion) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS "
    "FOR (n:Topic) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT experience_conversation_unique IF NOT EXISTS "
    "FOR (n:Experience) REQUIRE n.conversation_id IS UNIQUE",
    "CREATE INDEX person_name IF NOT EXISTS FOR (n:Person) ON (n.name)",
)

SEED_CONCEPTS: Tuple[str, ...] = ("consciousness", "emotions", "neural networks")

_SEED_CONCEPT_DETAILS: Dict[str, Tuple[str, str]] = {
    "consciousness": (
        "philosophy",
        "Awareness of one's own existence, states and experiences",
    ),
    "emotions": (
        "psychology",
        "States that combine feeling, appraisal and a readiness to act",
    ),
    "neural networks": (
        "technology",
        "Computational systems built from many simple connected units",
    ),
}

SYSTEM_RELATIONSHIP = "self"
DEFAULT_CATEGORY = "general"
LLM_CATEGORIES = ("person", "place", "emotion", "topic", "activity", "concept")
PREVIEW_CHARS = 200
# Conversation mentions: the reference implementation's starting importance and
# the step each further mention adds while importance is below the ceiling.
MENTION_IMPORTANCE = 0.6
MENTION_STEP = 0.02
ADD_CONCEPT_STEP = 0.01
IMPORTANCE_CEILING = 0.9
NETWORK_MAX_DEPTH = 5

TAUGHT_TYPES = frozenset({"taught", "teaches"})
LEARNED_TYPES = frozenset({"learned", "learns", "learning_about"})

# Driver exception class names that mean the server was not reached. Matched by
# name so this module never imports the driver.
_UNREACHABLE_NAMES = frozenset(
    {
        "ServiceUnavailable",
        "SessionExpired",
        "IncompleteCommit",
        "ConnectionAcquisitionTimeoutError",
    }
)

# ---------------------------------------------------------------------------
# Cypher (every value is a parameter)
# ---------------------------------------------------------------------------

_MERGE_PERSON = """
MERGE (p:Person {name: $name})
ON CREATE SET p += $props
RETURN p.id AS id, p.id = $props.id AS created
"""

_MERGE_CONCEPT = """
MERGE (c:Concept {name: $name})
ON CREATE SET c += $props
RETURN c.id AS id, c.id = $props.id AS created
"""

_MERGE_CONCEPT_COUNTED = """
MERGE (c:Concept {name: $name})
ON CREATE SET c += $props
ON MATCH SET c.mention_count = coalesce(c.mention_count, 0) + 1,
    c.last_mentioned = $now
RETURN c.id AS id, c.id = $props.id AS created
"""

_ADD_CONCEPT = """
MERGE (c:Concept {name: $name})
ON CREATE SET c += $props
ON MATCH SET c.importance = CASE
        WHEN coalesce(c.importance, 0.5) < $ceiling
        THEN coalesce(c.importance, 0.5) + $step
        ELSE c.importance END,
    c.last_mentioned = $now,
    c.mention_count = coalesce(c.mention_count, 0) + 1
"""

_MERGE_EXPERIENCE_BY_CONVERSATION = """
MERGE (e:Experience {conversation_id: $key})
ON CREATE SET e += $props
RETURN e.id AS id
"""

_MERGE_EXPERIENCE_BY_ID = """
MERGE (e:Experience {id: $key})
ON CREATE SET e += $props
RETURN e.id AS id
"""

_MERGE_EMOTION = """
MERGE (e:Emotion {name: $name})
ON CREATE SET e += $props
RETURN e.id AS id
"""

_MERGE_MEMORY = """
MERGE (m:Memory {id: $id})
ON CREATE SET m += $props
RETURN m.id AS id
"""

_MERGE_TOPIC = """
MERGE (t:Topic {name: $name})
ON CREATE SET t += $props
RETURN t.id AS id
"""

_TAUGHT_BY_ID = """
MATCH (teacher:Person {id: $teacher_id})
MATCH (concept:Concept {id: $concept_id})
MERGE (teacher)-[r:TAUGHT]->(concept)
SET r += $props
RETURN type(r) AS kind
"""

_LEARNED_BY_ID = """
MATCH (learner:Person {id: $learner_id})
MATCH (concept:Concept {id: $concept_id})
MERGE (learner)-[r:LEARNED]->(concept)
SET r += $props
RETURN type(r) AS kind
"""

_RELATES_BY_ID = """
MATCH (a:Concept {id: $concept1_id})
MATCH (b:Concept {id: $concept2_id})
MERGE (a)-[r:RELATES_TO]->(b)
SET r += $props
RETURN type(r) AS kind
"""

_REFERENCES_BY_ID = """
MATCH (m:Memory {id: $memory_id})
MATCH (c:Concept {id: $concept_id})
MERGE (m)-[r:REFERENCES_IN]->(c)
SET r += $props
RETURN type(r) AS kind
"""

_TAUGHT_BY_NAME = """
MATCH (teacher:Person {name: $teacher})
MATCH (concept:Concept {name: $concept})
MERGE (teacher)-[r:TAUGHT]->(concept)
SET r += $props
RETURN type(r) AS kind
"""

_LEARNED_BY_NAME = """
MATCH (learner:Person {name: $learner})
MATCH (concept:Concept {name: $concept})
MERGE (learner)-[r:LEARNED]->(concept)
SET r += $props
RETURN type(r) AS kind
"""

_LINK_MEMORY = """
MERGE (m:Memory {id: $memory_id})
ON CREATE SET m += $memory_props
SET m.content_preview = $preview
WITH m
UNWIND $names AS concept_name
MATCH (c:Concept {name: concept_name})
MERGE (m)-[r:REFERENCES_IN]->(c)
SET r += $link_props
RETURN count(r) AS linked, collect(DISTINCT c.name) AS names
"""

_LINK_MEMORY_WEIGHTED = """
MERGE (m:Memory {id: $memory_id})
ON CREATE SET m += $memory_props
SET m.content_preview = $preview
WITH m
UNWIND $links AS link
MATCH (c:Concept {name: link.name})
MERGE (m)-[r:REFERENCES_IN]->(c)
SET r += $link_props, r.relevance = link.relevance
RETURN count(r) AS linked
"""

_LIST_CONCEPTS = """
MATCH (c:Concept)
RETURN c.name AS name
ORDER BY name
"""

_LIST_CONCEPTS_LIMITED = """
MATCH (c:Concept)
RETURN c.name AS name
ORDER BY name
LIMIT $limit
"""

_MERGE_CONCEPT_QUIET = """
MERGE (c:Concept {name: $name})
ON CREATE SET c += $props
"""

_MENTION = """
MERGE (c:Concept {name: $name})
ON CREATE SET c += $props
ON MATCH SET c.mention_count = coalesce(c.mention_count, 0) + 1,
    c.last_mentioned = $now,
    c.importance = CASE
        WHEN coalesce(c.importance, 0.5) < $ceiling
        THEN coalesce(c.importance, 0.5) + $step
        ELSE c.importance END
MERGE (e:Experience {conversation_id: $conversation_id})
ON CREATE SET e += $experience_props
MERGE (c)-[r:MENTIONED_IN {speaker: $speaker}]->(e)
ON CREATE SET r.when = $now, r.mentions = 1
ON MATCH SET r.when = $now, r.mentions = coalesce(r.mentions, 0) + 1
"""

_RELATE = """
MATCH (a:Concept {name: $first})
MATCH (b:Concept {name: $second})
MERGE (a)-[r:RELATES_TO]-(b)
ON CREATE SET r.discovered_at = $now
SET r.relationship_type = $relationship,
    r.strength = coalesce($strength, r.strength)
RETURN r.strength AS strength
"""

_RELATED = """
MATCH (c:Concept {name: $concept_name})-[r:RELATES_TO]-(related:Concept)
RETURN related.name AS concept, r.relationship_type AS relationship,
    r.strength AS strength
ORDER BY strength IS NULL, strength DESC, concept ASC
LIMIT $limit
"""

_DELETE_MEMORIES = """
MATCH (m:Memory)
WHERE m.id IN $ids
WITH m, m.id AS removed_id
DETACH DELETE m
RETURN count(removed_id) AS removed
"""

_PERSON_BY_NAME = """
MATCH (p:Person {name: $name})
RETURN properties(p) AS node
LIMIT 1
"""

_CONCEPT_BY_NAME = """
MATCH (c:Concept {name: $name})
RETURN properties(c) AS node
LIMIT 1
"""

_WHO_TAUGHT = """
MATCH (teacher:Person)-[t:TAUGHT]->(concept:Concept {name: $concept_name})
MATCH (concept)<-[:LEARNED]-(learner:Person {name: $learner})
RETURN teacher.name AS teacher, t.when AS when, t.method AS method,
    t.effectiveness AS effectiveness
"""

_NETWORK_TEMPLATE = """
MATCH path = (start:Concept {name: $concept_name})-[*1..DEPTH]-(connected:Concept)
RETURN path
LIMIT 100
"""

# Neo4j takes no parameter for a variable-length bound, so one statement per
# allowed depth is built here from the constant template; no call-time value
# reaches the text.
NETWORK_QUERIES: Dict[int, str] = {
    depth: _NETWORK_TEMPLATE.replace("DEPTH", str(depth))
    for depth in range(1, NETWORK_MAX_DEPTH + 1)
}

_MEMORIES_FOR_CONCEPT = """
MATCH (memory:Memory)-[r:REFERENCES_IN]->(concept:Concept {name: $concept_name})
RETURN memory.id AS memory_id, memory.postgres_id AS postgres_id,
    memory.content_preview AS content, memory.created_at AS created_at,
    r.relevance AS relevance, r.extraction_method AS extraction_method
ORDER BY relevance DESC, created_at DESC
"""

_CONCEPTS_IN_MEMORY = """
MATCH (memory:Memory {id: $memory_id})-[r:REFERENCES_IN]->(concept:Concept)
RETURN concept.name AS concept_name, concept.category AS category,
    concept.understanding_level AS understanding_level, r.relevance AS relevance
ORDER BY relevance DESC
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _unreachable(exc: BaseException) -> bool:
    if isinstance(exc, OSError):  # includes ConnectionError and TimeoutError
        return True
    return bool({cls.__name__ for cls in type(exc).__mro__} & _UNREACHABLE_NAMES)


def memory_node_id(memory_id: int) -> str:
    """Return the ``Memory`` node id of an episodic row id."""
    return str(UUID(int=int(memory_id)))


def map_category(raw: Any) -> str:
    """Map an extractor category onto the graph's: known ones, else ``general``."""
    text = str(raw or "").strip().lower()
    return text if text in LLM_CATEGORIES else DEFAULT_CATEGORY


def _clamp_unit(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return max(0.0, min(1.0, number))


class KnowledgeGraphManager:
    """Neo4j knowledge graph; a ``GraphStore`` with MERGE writes."""

    def __init__(
        self,
        config: Optional[Neo4jConfig] = None,
        driver: Any = None,
        settings: Optional[Settings] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._settings = settings
        self.config = (
            config if config is not None else Neo4jConfig.from_settings(settings)
        )
        self._driver = driver
        self._owns_driver = driver is None
        self._clock = clock or _utcnow
        self._last_error: Optional[str] = None
        self._verified = False

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    @property
    def settings(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    def _now(self) -> str:
        moment = self._clock()
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("the graph manager clock must return aware datetimes")
        return moment.astimezone(timezone.utc).isoformat()

    def connect(self) -> None:
        """Build the driver when needed and verify that the server answers."""
        if self._driver is None:
            self._driver = self.config.create_driver()
            self._owns_driver = True
        try:
            self._driver.verify_connectivity()
        except Exception as exc:
            self._last_error = _describe(exc)
            logger.warning(
                "Neo4j at %s is unreachable: %s", self.config.uri, self._last_error
            )
            if self._owns_driver:
                try:
                    self._driver.close()
                finally:
                    self._driver = None
            raise StoreUnavailable(
                f"Neo4j at {self.config.uri} did not answer: {exc}"
            ) from exc
        self._last_error = None
        self._verified = True
        logger.info("connected to the knowledge graph %s", self.config.masked())

    def _run(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        if self._driver is None:
            self.connect()
        try:
            with self._driver.session(database=self.config.database) as session:
                result = session.run(query, params or {})
                records = [dict(record) for record in result]
        except Exception as exc:
            self._last_error = _describe(exc)
            if _unreachable(exc):
                logger.warning("Neo4j call failed: %s", self._last_error)
                raise StoreUnavailable(
                    f"Neo4j is unavailable: {self._last_error}"
                ) from exc
            raise
        self._last_error = None
        self._verified = True
        return records

    def _first(self, query: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        records = self._run(query, params)
        return records[0] if records else None

    def close(self) -> None:
        """Release a driver this manager built; an injected one stays open."""
        if self._driver is not None and self._owns_driver:
            self._driver.close()
            self._driver = None
            logger.info("disconnected from the knowledge graph")

    def __enter__(self) -> "KnowledgeGraphManager":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available``, ``last_error`` and the target."""
        return {
            "backend": "neo4j",
            "available": self._verified and self._last_error is None,
            "last_error": self._last_error,
            "connected": self._driver is not None,
            "uri": self.config.uri,
            "database": self.config.database,
        }

    # ------------------------------------------------------------------
    # Schema and seeds
    # ------------------------------------------------------------------

    def bootstrap_constraints(self) -> int:
        """Apply ``CONSTRAINTS`` (idempotent); return how many statements ran."""
        for statement in CONSTRAINTS:
            self._run(statement)
        return len(CONSTRAINTS)

    def seed_core_entities(self, include_creator: bool = True) -> Dict[str, int]:
        """Merge the system and creator persons and ``SEED_CONCEPTS``.

        Returns the nodes created; a second run returns zeros.
        """
        active = self.settings
        persons: List[Person] = [
            Person(name=active.SYSTEM_NAME, relationship=SYSTEM_RELATIONSHIP)
        ]
        if include_creator:
            first_met: Optional[datetime] = None
            if active.SYSTEM_BIRTH_DATE:
                first_met = datetime.combine(
                    date.fromisoformat(active.SYSTEM_BIRTH_DATE),
                    time(0, 0),
                    tzinfo=timezone.utc,
                )
            persons.append(
                Person(
                    name=active.CREATOR_NAME,
                    relationship=active.CREATOR_RELATIONSHIP,
                    first_met=first_met,
                )
            )
        persons_created = 0
        for person in persons:
            record = self._first(
                _MERGE_PERSON,
                {"name": person.name, "props": node_to_cypher_properties(person)},
            )
            persons_created += int(bool(record and record.get("created")))
        concepts_created = 0
        now = self._clock().astimezone(timezone.utc)
        for name in SEED_CONCEPTS:
            category, description = _SEED_CONCEPT_DETAILS[name]
            concept = Concept(
                name=name, category=category, description=description, learned_at=now
            )
            record = self._first(
                _MERGE_CONCEPT,
                {"name": name, "props": node_to_cypher_properties(concept)},
            )
            concepts_created += int(bool(record and record.get("created")))
        logger.info(
            "graph seed: %d person and %d concept nodes created",
            persons_created,
            concepts_created,
        )
        return {
            "persons_created": persons_created,
            "concepts_created": concepts_created,
        }

    # ------------------------------------------------------------------
    # Node writes (MERGE)
    # ------------------------------------------------------------------

    @staticmethod
    def _id_of(record: Optional[Dict[str, Any]], fallback: UUID) -> UUID:
        if record and record.get("id"):
            return UUID(str(record["id"]))
        return fallback

    def create_person(self, person: Person) -> UUID:
        """Merge a ``Person`` by name; return the stored node's id."""
        record = self._first(
            _MERGE_PERSON,
            {"name": person.name, "props": node_to_cypher_properties(person)},
        )
        return self._id_of(record, person.id)

    def create_concept(self, concept: Concept) -> UUID:
        """Merge a ``Concept`` by name; return the stored node's id."""
        record = self._first(
            _MERGE_CONCEPT,
            {"name": concept.name, "props": node_to_cypher_properties(concept)},
        )
        return self._id_of(record, concept.id)

    def create_experience(self, experience: Experience) -> UUID:
        """Merge an ``Experience`` by conversation id when it has one, else id."""
        props = node_to_cypher_properties(experience)
        if experience.conversation_id:
            query, key = _MERGE_EXPERIENCE_BY_CONVERSATION, experience.conversation_id
        else:
            query, key = _MERGE_EXPERIENCE_BY_ID, props["id"]
        return self._id_of(
            self._first(query, {"key": key, "props": props}), experience.id
        )

    def create_emotion(self, emotion: Emotion) -> UUID:
        """Merge an ``Emotion`` by name; return the stored node's id."""
        record = self._first(
            _MERGE_EMOTION,
            {"name": emotion.name, "props": node_to_cypher_properties(emotion)},
        )
        return self._id_of(record, emotion.id)

    def create_memory(self, memory: Memory) -> UUID:
        """Merge a ``Memory`` reference node by id; return its id."""
        record = self._first(
            _MERGE_MEMORY, {"id": memory.id, "props": node_to_cypher_properties(memory)}
        )
        return self._id_of(record, UUID(memory.id))

    def create_topic(self, topic: Topic) -> UUID:
        """Merge a ``Topic`` by name; return the stored node's id."""
        record = self._first(
            _MERGE_TOPIC,
            {"name": topic.name, "props": node_to_cypher_properties(topic)},
        )
        return self._id_of(record, topic.id)

    def add_concept(
        self,
        name: str,
        source: str = "web",
        learned_at: Optional[datetime] = None,
        description: str = "",
    ) -> None:
        """Merge a concept by name: created with ``source`` as its category, or
        counted as one more mention (importance rises by 0.01 below 0.9).

        Synchronous like every method here; async callers run it through
        ``asyncio.to_thread`` (the reference implementation awaited it).
        """
        now = self._now()
        created_at = (
            learned_at.astimezone(timezone.utc)
            if learned_at is not None and learned_at.tzinfo is not None
            else self._clock().astimezone(timezone.utc)
        )
        concept = Concept(
            name=name,
            category=source,
            description=description,
            mention_count=1,
            learned_at=created_at,
            last_mentioned=datetime.fromisoformat(now),
        )
        self._run(
            _ADD_CONCEPT,
            {
                "name": name,
                "props": node_to_cypher_properties(concept),
                "now": now,
                "ceiling": IMPORTANCE_CEILING,
                "step": ADD_CONCEPT_STEP,
            },
        )
        logger.debug("merged concept from %s", source)

    # ------------------------------------------------------------------
    # Relationship writes (MERGE)
    # ------------------------------------------------------------------

    def create_taught_relationship(
        self, teacher_id: UUID, concept_id: UUID, taught: Taught
    ) -> bool:
        """``(Person)-[:TAUGHT]->(Concept)`` by node ids; False when either is absent."""
        return (
            self._first(
                _TAUGHT_BY_ID,
                {
                    "teacher_id": str(teacher_id),
                    "concept_id": str(concept_id),
                    "props": relationship_to_cypher_properties(taught),
                },
            )
            is not None
        )

    def create_learned_relationship(
        self, learner_id: UUID, concept_id: UUID, learned: Learned
    ) -> bool:
        """``(Person)-[:LEARNED]->(Concept)`` by node ids; False when either is absent."""
        return (
            self._first(
                _LEARNED_BY_ID,
                {
                    "learner_id": str(learner_id),
                    "concept_id": str(concept_id),
                    "props": relationship_to_cypher_properties(learned),
                },
            )
            is not None
        )

    def create_relates_to_relationship(
        self, concept1_id: UUID, concept2_id: UUID, relates_to: RelatesTo
    ) -> bool:
        """``(Concept)-[:RELATES_TO]->(Concept)`` by node ids."""
        return (
            self._first(
                _RELATES_BY_ID,
                {
                    "concept1_id": str(concept1_id),
                    "concept2_id": str(concept2_id),
                    "props": relationship_to_cypher_properties(relates_to),
                },
            )
            is not None
        )

    def create_references_in_relationship(
        self, memory_id: str, concept_id: UUID, references_in: ReferencesIn
    ) -> bool:
        """``(Memory)-[:REFERENCES_IN]->(Concept)`` by node ids."""
        return (
            self._first(
                _REFERENCES_BY_ID,
                {
                    "memory_id": str(memory_id),
                    "concept_id": str(concept_id),
                    "props": relationship_to_cypher_properties(references_in),
                },
            )
            is not None
        )

    # ------------------------------------------------------------------
    # GraphStore
    # ------------------------------------------------------------------

    def _memory_props(self, memory_id: int, preview: str) -> Dict[str, Any]:
        memory = Memory(
            id=memory_node_id(memory_id),
            postgres_id=int(memory_id),
            memory_type="episodic",
            content_preview=preview,
            created_at=self._clock().astimezone(timezone.utc),
        )
        return node_to_cypher_properties(memory)

    def link_memory_to_concepts(
        self,
        memory_id: int,
        memory_content: str,
        concept_names: Sequence[str],
        relevance: float = 0.5,
        extraction_method: str = "keyword",
    ) -> int:
        """Merge the memory node and link it to each existing concept named."""
        preview = (memory_content or "")[:PREVIEW_CHARS]
        names = [str(name) for name in concept_names]
        link = ReferencesIn(
            when=self._clock().astimezone(timezone.utc),
            relevance=relevance,
            extraction_method=extraction_method,
        )
        record = self._first(
            _LINK_MEMORY,
            {
                "memory_id": memory_node_id(memory_id),
                "memory_props": self._memory_props(memory_id, preview),
                "preview": preview,
                "names": names,
                "link_props": relationship_to_cypher_properties(link),
            },
        )
        linked = int((record or {}).get("linked") or 0)
        if record is not None and "names" in record:
            found = set(record.get("names") or [])
            skipped = [name for name in names if name not in found]
        else:
            skipped = []
        if skipped:
            logger.warning(
                "%d concept names not in the graph; links skipped", len(skipped)
            )
        return linked

    def list_concept_names(self, limit: Optional[int] = None) -> List[str]:
        """Return concept names in ascending order."""
        if limit is None:
            records = self._run(_LIST_CONCEPTS)
        else:
            records = self._run(_LIST_CONCEPTS_LIMITED, {"limit": max(int(limit), 0)})
        return [str(record["name"]) for record in records]

    def merge_concept(
        self, name: str, description: str = "", category: str = "conversation"
    ) -> None:
        """Create the concept when absent; an existing one is left unchanged."""
        concept = Concept(
            name=name,
            category=category,
            description=description,
            learned_at=self._clock().astimezone(timezone.utc),
        )
        self._run(
            _MERGE_CONCEPT_QUIET,
            {"name": name, "props": node_to_cypher_properties(concept)},
        )

    def link_concept_to_conversation(
        self, name: str, conversation_id: str, speaker: str
    ) -> None:
        """Merge the concept (one more mention when it exists) and record that
        ``speaker`` mentioned it in the conversation (private data)."""
        now = self._now()
        moment = datetime.fromisoformat(now)
        concept = Concept(
            name=name,
            category="conversation",
            importance=MENTION_IMPORTANCE,
            mention_count=1,
            learned_at=moment,
            last_mentioned=moment,
        )
        experience = Experience(
            experience_type="conversation",
            conversation_id=conversation_id,
            occurred_at=moment,
        )
        self._run(
            _MENTION,
            {
                "name": name,
                "props": node_to_cypher_properties(concept),
                "now": now,
                "ceiling": IMPORTANCE_CEILING,
                "step": MENTION_STEP,
                "conversation_id": conversation_id,
                "experience_props": node_to_cypher_properties(experience),
                "speaker": speaker,
            },
        )

    def relate_concepts(
        self,
        first: str,
        second: str,
        relationship: str = "relates_to",
        strength: Optional[float] = 0.5,
    ) -> bool:
        """Relate two existing concepts; False when either is missing or equal.

        A None ``strength`` means unmeasured: a new edge gets no strength
        property and an existing edge keeps the strength it has.
        """
        if first == second:
            return False
        record = self._first(
            _RELATE,
            {
                "first": first,
                "second": second,
                "relationship": relationship,
                "strength": None if strength is None else float(strength),
                "now": self._now(),
            },
        )
        return record is not None

    def get_related_concepts(
        self, concept_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return ``{"concept", "relationship", "strength"}`` rows, strongest first.

        An edge with no measured strength reports None and sorts last.
        """
        records = self._run(
            _RELATED, {"concept_name": concept_name, "limit": max(int(limit), 0)}
        )
        return [
            {
                "concept": record.get("concept"),
                "relationship": record.get("relationship"),
                "strength": record.get("strength"),
            }
            for record in records
        ]

    def delete_memory_nodes(self, memory_ids: Sequence[int]) -> int:
        """Detach-delete memory nodes; concepts stay. Return the nodes removed."""
        ids = [memory_node_id(memory_id) for memory_id in memory_ids]
        if not ids:
            return 0
        record = self._first(_DELETE_MEMORIES, {"ids": ids})
        return int((record or {}).get("removed") or 0)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_person_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Return a person's properties, or None."""
        record = self._first(_PERSON_BY_NAME, {"name": name})
        return dict(record["node"]) if record else None

    def get_concept_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Return a concept's properties, or None."""
        record = self._first(_CONCEPT_BY_NAME, {"name": name})
        return dict(record["node"]) if record else None

    def who_taught_concept(
        self, concept_name: str, learner: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return who taught ``learner`` (default ``SYSTEM_NAME``) a concept."""
        who = learner if learner is not None else self.settings.SYSTEM_NAME
        return self._run(_WHO_TAUGHT, {"concept_name": concept_name, "learner": who})

    def get_knowledge_network(
        self, concept_name: str, depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Return paths of up to ``depth`` hops (clamped to 1 to 5) from a concept."""
        bounded = max(1, min(int(depth), NETWORK_MAX_DEPTH))
        return self._run(NETWORK_QUERIES[bounded], {"concept_name": concept_name})

    def get_memories_referencing_concept(
        self, concept_name: str
    ) -> List[Dict[str, Any]]:
        """Return the memory nodes that reference a concept, most relevant first."""
        return self._run(_MEMORIES_FOR_CONCEPT, {"concept_name": concept_name})

    def get_concepts_in_memory(self, memory_id: int) -> List[Dict[str, Any]]:
        """Return the concepts an episodic row's memory node references."""
        return self._run(_CONCEPTS_IN_MEMORY, {"memory_id": memory_node_id(memory_id)})

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    def process_llm_extraction(
        self,
        extraction: Any,
        memory_id: int,
        learner: Optional[str] = None,
        memory_content: str = "",
    ) -> Dict[str, Any]:
        """Write a measured concept extraction for an episodic row.

        ``extraction`` is a ``ConceptExtraction`` or a mapping with
        ``concepts`` (``name``, ``category``, ``relevance``) and optional
        ``relationships`` (``from``, ``to``, ``type``, ``strength``). Concepts
        are merged with their category mapped onto ``LLM_CATEGORIES`` (anything
        else is ``general``) and linked from the memory node with method
        ``llm``. A ``taught`` relationship needs an existing ``Person`` named by
        ``from`` and also records that ``learner`` (default ``SYSTEM_NAME``)
        learned the concept; a ``learned`` relationship links ``learner`` to the
        concept; any other type relates two existing concepts. A name the model
        wrote never creates a ``Person``. Nothing is written unless the
        extraction's status is ``measured``.
        """
        status, concepts, relationships = self._unpack(extraction)
        stats: Dict[str, Any] = {
            "status": status,
            "concepts_created": 0,
            "concepts_found": 0,
            "relationships_created": 0,
            "relationships_skipped": 0,
            "memory_linked": False,
        }
        if status != "measured":
            return stats
        now = self._now()
        moment = datetime.fromisoformat(now)
        links: List[Dict[str, Any]] = []
        for item in concepts:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            relevance = _clamp_unit(item.get("relevance"))
            fields: Dict[str, Any] = {
                "name": name,
                "category": map_category(item.get("category")),
                "mention_count": 1,
                "learned_at": moment,
                "last_mentioned": moment,
            }
            if relevance is not None:
                fields["importance"] = relevance
            record = self._first(
                _MERGE_CONCEPT_COUNTED,
                {
                    "name": name,
                    "props": node_to_cypher_properties(Concept(**fields)),
                    "now": now,
                },
            )
            if record and record.get("created"):
                stats["concepts_created"] += 1
            else:
                stats["concepts_found"] += 1
            links.append({"name": name, "relevance": relevance})
        if links:
            preview = (memory_content or "")[:PREVIEW_CHARS]
            link = ReferencesIn(when=moment, extraction_method="llm")
            link_props = relationship_to_cypher_properties(link)
            link_props.pop("relevance", None)  # set per concept, or left unset
            record = self._first(
                _LINK_MEMORY_WEIGHTED,
                {
                    "memory_id": memory_node_id(memory_id),
                    "memory_props": self._memory_props(memory_id, preview),
                    "preview": preview,
                    "links": links,
                    "link_props": link_props,
                },
            )
            stats["memory_linked"] = bool(record and (record.get("linked") or 0) > 0)
        who = learner if learner is not None else self.settings.SYSTEM_NAME
        for relation in relationships:
            if self._write_relationship(relation, who, moment):
                stats["relationships_created"] += 1
            else:
                stats["relationships_skipped"] += 1
        logger.info(
            "llm extraction written: %d new concepts, %d existing, %d relationships",
            stats["concepts_created"],
            stats["concepts_found"],
            stats["relationships_created"],
        )
        return stats

    @staticmethod
    def _unpack(extraction: Any) -> Tuple[str, List[Mapping[str, Any]], List[Any]]:
        if isinstance(extraction, Mapping):
            status = str(extraction.get("status", "measured"))
            concepts = [dict(c) for c in extraction.get("concepts") or []]
            relationships = list(extraction.get("relationships") or [])
            return status, concepts, relationships
        concepts = [
            {
                "name": getattr(c, "name", None),
                "category": getattr(c, "category", None),
                "relevance": getattr(c, "relevance", None),
            }
            for c in getattr(extraction, "concepts", []) or []
        ]
        return str(getattr(extraction, "status", "measured")), concepts, []

    def _write_relationship(
        self, relation: Any, learner: str, moment: datetime
    ) -> bool:
        if not isinstance(relation, Mapping):
            return False
        source = str(relation.get("from") or "").strip()
        target = str(relation.get("to") or "").strip()
        if not source or not target:
            return False
        kind = str(relation.get("type") or "relates_to").strip().lower()
        strength = _clamp_unit(relation.get("strength"))
        if kind in TAUGHT_TYPES:
            # An unmeasured strength stays None and is left off the edge.
            taught = Taught(when=moment, method="conversation", effectiveness=strength)
            made = self._first(
                _TAUGHT_BY_NAME,
                {
                    "teacher": source,
                    "concept": target,
                    "props": relationship_to_cypher_properties(taught),
                },
            )
            if made is None:
                return False
            self._first(
                _LEARNED_BY_NAME,
                {
                    "learner": learner,
                    "concept": target,
                    "props": relationship_to_cypher_properties(Learned(when=moment)),
                },
            )
            return True
        if kind in LEARNED_TYPES:
            learned = Learned(when=moment, confidence=strength, proficiency=strength)
            return (
                self._first(
                    _LEARNED_BY_NAME,
                    {
                        "learner": learner,
                        "concept": target,
                        "props": relationship_to_cypher_properties(learned),
                    },
                )
                is not None
            )
        return self.relate_concepts(
            source, target, relationship=kind, strength=strength
        )
