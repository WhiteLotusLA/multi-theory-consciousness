# Knowledge graph

The concept graph records the concepts conversations mention, how they relate,
which memories reference them and who taught or learned them. With
`NEO4J_ENABLED` true it lives in Neo4j through
`mtc/knowledge/graph_manager.py` (`KnowledgeGraphManager`); otherwise the
runtime uses the in-memory graph store with the same protocol
(`GraphStore` in `mtc/memory/protocols.py`), so concept capture, recall and
the knowledge routes always have a graph.

A test (`tests/docs/test_knowledge_graph_doc.py`) compares the labels,
relationship types and constraints on this page with
`mtc/knowledge/models.py` and `CONSTRAINTS` in the graph manager.

## Node labels

Properties come from the pydantic models in `mtc/knowledge/models.py`. None
values are never stored (Neo4j stores no nulls), UUIDs and datetimes are
stored as strings, and timestamps are timezone-aware UTC.

| Label | Properties | Notes |
|---|---|---|
| `Person` | `id`, `name`, `relationship`, `first_met`, `interaction_count` | no bond, trust or comfort value; `first_met` is None unless known |
| `Concept` | `id`, `name`, `category`, `description`, `importance`, `mention_count`, `learned_at`, `last_mentioned`, `understanding_level` | `understanding_level` is optional and never assumed |
| `Experience` | `id`, `description`, `experience_type`, `conversation_id`, `occurred_at`, `emotional_valence`, `arousal_level`, `importance` | one node per conversation, keyed by `conversation_id` |
| `Emotion` | `id`, `name`, `valence`, `arousal`, `category` | |
| `Memory` | `id`, `postgres_id`, `memory_type`, `content_preview`, `created_at`, `importance`, `access_count` | a reference to an episodic row; the preview holds conversation text |
| `Topic` | `id`, `name`, `frequency`, `first_mentioned`, `last_mentioned` | |

## Relationships and their writers

| Type | From and to | Properties | Written by |
|---|---|---|---|
| `TAUGHT` | `Person` to `Concept` | `when`, `method`, `effectiveness` | `create_taught_relationship`; `process_llm_extraction` for a taught relation |
| `LEARNED` | `Person` to `Concept` | `when`, `confidence`, `depth`, `proficiency` | `create_learned_relationship`; `process_llm_extraction` for a learned relation |
| `RELATES_TO` | `Concept` to `Concept` | `relationship_type`, `strength`, `discovered_at` | `create_relates_to_relationship`, `relate_concepts` |
| `REFERENCES_IN` | `Memory` to `Concept` | `when`, `relevance`, `extraction_method` | `link_memory_to_concepts` (the memory manager links each stored memory to concepts that already exist), `create_references_in_relationship` |
| `MENTIONED_IN` | `Concept` to `Experience` | `speaker`, `when`, `mentions` | `link_concept_to_conversation` (the post-turn `concepts` hook, one edge per concept, conversation and speaker) |

The turn's concept capture (`extract_and_store_concepts` in
`mtc/pipeline/memory_retrieval.py`) merges up to 5 content words (its default
`max_concepts`) of the message and of the spoken reply as `Concept` nodes with a neutral
description, then records the mention. The speaker's name, bare or bracketed,
and the system's name never become concepts, and reasoning is never
extracted.

## Constraints

`CONSTRAINTS` in `mtc/knowledge/graph_manager.py`, each `IF NOT EXISTS`:

| Name | Rule |
|---|---|
| `concept_name_unique` | `Concept.name` is unique |
| `person_id_unique` | `Person.id` is unique |
| `concept_id_unique` | `Concept.id` is unique |
| `memory_id_unique` | `Memory.id` is unique |
| `experience_id_unique` | `Experience.id` is unique |
| `emotion_id_unique` | `Emotion.id` is unique |
| `topic_id_unique` | `Topic.id` is unique |
| `experience_conversation_unique` | `Experience.conversation_id` is unique |
| `person_name` | an index on `Person.name` |

Every node write is a `MERGE` keyed by one of these properties, so running a
write twice never duplicates a node. `mtc db init` applies the constraints
through `KnowledgeGraphManager.bootstrap_constraints()`; `mtc db check`
matches them by label, properties and type.

## Seeds

`seed_core_entities()` (called by the bootstrap when Neo4j is enabled) merges:

- a `Person` for `SYSTEM_NAME` with relationship `self`;
- a `Person` for `CREATOR_NAME` with `CREATOR_RELATIONSHIP`, and `first_met`
  from `SYSTEM_BIRTH_DATE` only when that key is set;
- the `SEED_CONCEPTS`: `consciousness` (philosophy), `emotions` (psychology)
  and `neural networks` (technology), each with a neutral description.

No bond, trust or comfort value is written: the creator node exists for
display and for the who-taught query, never to grant trust. Seeding twice
creates nothing.

## Cypher examples

Every query the manager runs is a module constant with parameters; nothing is
interpolated. These read-only examples use neutral entities:

```cypher
// Concepts related to "memory", strongest first
MATCH (c:Concept {name: "memory"})-[r:RELATES_TO]-(related:Concept)
RETURN related.name, r.strength
ORDER BY r.strength DESC
LIMIT 10;

// The memories that reference "attention"
MATCH (m:Memory)-[r:REFERENCES_IN]->(c:Concept {name: "attention"})
RETURN m.postgres_id, m.created_at, r.relevance
ORDER BY m.created_at DESC;

// How often each concept was mentioned in one conversation
MATCH (c:Concept)-[r:MENTIONED_IN]->(e:Experience {conversation_id: $conversation_id})
RETURN c.name, sum(r.mentions) AS mentions
ORDER BY mentions DESC;

// The most mentioned concepts overall
MATCH (c:Concept)
RETURN c.name, c.mention_count
ORDER BY c.mention_count DESC
LIMIT 20;
```

The API exposes `GET /api/knowledge/concepts` (concept names) and
`POST /api/knowledge/concepts` (related concepts); see [API.md](API.md).

## The community edition holds one graph

Neo4j Community Edition serves one user database. MTC writes into the database
named by `NEO4J_DATABASE` (default `neo4j`) and keys every node by a unique
property, but it does not namespace nodes by `DB_PREFIX`: two systems that
share one community server share one graph. Give each install its own Neo4j
server (or an edition with several databases), and never point a test or a
second system at a graph that already serves another one. The throwaway test
fixtures skip Neo4j on the community edition for this reason.

## Privacy

`Person` nodes and `MENTIONED_IN` speakers hold real names at runtime, and
`Memory` previews hold conversation text. The graph lives in an
operator-controlled store and never in version control; back it up and
protect it like the other stores (see [DATABASES.md](DATABASES.md) and
[SECURITY.md](SECURITY.md)).
