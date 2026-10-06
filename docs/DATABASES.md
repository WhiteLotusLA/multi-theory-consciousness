# Databases

MTC can persist into five stores. Every one is optional and off by default:
with all five enable keys false the runtime keeps everything in process
memory behind the same protocols (`mtc/memory/protocols.py`), and nothing
survives a restart except the persistence hub's files. A store is used only
when its enable key is true and its client extra is installed.

Every schema artifact is a module constant registered in
`mtc/core/schema_registry.py` (`SCHEMA_ARTIFACTS`). The PostgreSQL
migrations are rendered from that registry, the bootstrap applies the same
artifacts, and the verifier derives its expectations from them, so there is
one source for every table, index, collection, payload index, constraint and
key family. A test (`tests/docs/test_databases_doc.py`) fails when this page
misses an artifact the registry defines.

## Overview

| Store | Enable key | Extra | Needed for | Default port | Holds |
|---|---|---|---|---|---|
| PostgreSQL | `POSTGRES_ENABLED` | `postgres` | persistence of memories and conversations | 5432 | episodic, semantic and core memory; conversations, messages and emotional history; snapshots and events; the safety audit |
| MongoDB | `MONGODB_ENABLED` | `mongo` | recommended | 27017 | conversation turns and sessions, thoughts and gists, neural telemetry, learning, speaker models, goals, dreams, alerts, the privacy audit log |
| Qdrant | `QDRANT_ENABLED` | `qdrant` | recommended (vector recall) | 6333 | 384-dimension memory vectors |
| Neo4j | `NEO4J_ENABLED` | `neo4j` | optional | 7687 (bolt) | the concept graph |
| Redis | `REDIS_ENABLED` | `redis` | optional | 6379 | working memory and short-lived state |

To run the stores, see [SETUP_DOCKER.md](SETUP_DOCKER.md) (a Docker Compose
file that was reviewed against official documentation and not run on the
development machine) or [SETUP_MACOS.md](SETUP_MACOS.md) (host services); the
path run on the development machine is the in-memory install with the mock
provider.

`pip install ".[databases]"` installs every store client. Store names derive
from `DB_PREFIX` (`mtc`): PostgreSQL database `mtc_consciousness` and role
`mtc`, MongoDB database `mtc_neural`, Qdrant collection `mtc_memories`, Redis
key prefix `mtc:`; an explicit `MTC_POSTGRES_DB`, `MTC_POSTGRES_USER`,
`MTC_MONGODB_DB`, `MTC_QDRANT_COLLECTION` or `MTC_REDIS_KEY_PREFIX` is kept.
Neo4j writes into `NEO4J_DATABASE` (`neo4j`).

## Bootstrap and verification

```bash
mtc db init                    # every enabled store; records the install date; seeds
mtc db init --only postgres    # one store
mtc db check                   # verify every enabled store against the registry
mtc db check --json --strict   # machine-readable; warnings count as failures
```

`mtc db init` (`mtc.db.bootstrap`) runs, in `STORE_ORDER` (PostgreSQL,
MongoDB, Qdrant, Neo4j, Redis), each enabled store's initialization in its own
try, so one failure does not stop the next; Redis has no schema and gets its
checks only. It then writes `MTC_SYSTEM_BIRTH_DATE` into `.env` once (the
calendar day in `TIMEZONE`; an existing date is kept, never overwritten) and
seeds the stores: core memory in PostgreSQL (the identity block gains one
sentence, "This system was first initialized on YYYY-MM-DD.", which records
the install date without claiming a birthday) and the graph's system,
creator and seed-concept nodes in Neo4j. With no store enabled nothing is
seeded and core memory is rendered from the persona at run time. Every step
is idempotent: a second run creates nothing. Nothing in any init path drops, truncates or recreates anything.

`scripts/db/` holds thin wrappers over the same package modules
(`bootstrap.py`, `verify.py`, `init_mongo.py`, `init_qdrant.py`,
`init_neo4j.py`, `reembed_postgres_to_qdrant.py`) and the PostgreSQL role
script `init_postgres.sh`.

### Statuses and exit codes

Every check reports one status (`mtc.core.checks.CheckStatus`):

| Status | Meaning |
|---|---|
| `ok` | measured and as expected |
| `warning` | measured; an advisory difference (an HNSW parameter, an eviction policy) |
| `failed` | measured and wrong (a missing table, a refused login, a dimension mismatch) |
| `unreachable` | the server refused the connection or timed out |
| `skipped` | the store is disabled |
| `insufficient_data` | the check could not be performed (a missing extra, a refused `CONFIG GET`), with the reason |

A check that did not run is never `ok`. Exit codes, shared by every `mtc`
subcommand and every wrapper: 0 success; 1 a required check or the command's
own work failed; 2 a usage or configuration error (including a missing extra);
3 a required service unreachable; 130 interrupted. Advisory checks never
change the exit code except under `--strict`, which counts `warning` and
`insufficient_data` as failures.

## PostgreSQL

The base store. `migrations/postgres/` holds three files rendered by
`python3 -m mtc.db.migrations --write` from the registry; each statement is
`CREATE ... IF NOT EXISTS`, and PostgreSQL 14 or later needs no extension
(`gen_random_uuid()` is built in). No migration drops anything, sets a
password or inserts a row.

| File | Source constants |
|---|---|
| `001_core_schema.sql` | `mtc.memory.episodic_store.EPISODIC_SCHEMA_SQL`, then `mtc.memory.conversation_store.CONVERSATION_SCHEMA_SQL` |
| `002_research_schema.sql` | `mtc.memory.postgres_stores.RESEARCH_SCHEMA_SQL` |
| `003_safety_schema.sql` | `mtc.safety.stores.SAFETY_SCHEMA_SQL` |

Tables and the indexes the queries need:

| Table | Holds | Indexes and why |
|---|---|---|
| `core_memory` | the `identity`, `persona` and `human` blocks (primary key `block_name`) | none beyond the key |
| `episodic_memory` | every episodic memory with importance, emotional context, consolidation state; `tsv_content` is a generated full-text vector | `episodic_memory_tsv_idx` (GIN, full-text recall when there is no vector store), `episodic_memory_created_at_idx` (recent memories and the pruner's cursor), `episodic_memory_type_idx` (excluding archived and core rows), `episodic_memory_importance_idx` (recent important memories), `episodic_memory_unconsolidated_idx` (partial, the consolidation pass) |
| `semantic_memory` | concepts and knowledge formed by consolidation | none beyond the key |
| `conversations` | one row per conversation (participants, summary, start and end) | none beyond the key |
| `messages` | every message with its stored `role` and `to_speaker` addressee | `messages_conversation_timestamp_idx` (a conversation's history in order), `messages_timestamp_idx` (rehydrating recent history) |
| `emotional_history` | the emotion label, intensity and PAD coordinate per turn | `emotional_history_timestamp_idx` (recent states) |
| `consciousness_snapshots` | assessment snapshots (overall score, passing count, Phi, theory and indicator results, status, coverage) | `consciousness_snapshots_timestamp_idx` (history by time) |
| `consciousness_events` | events derived from snapshots, linked to their snapshot | `consciousness_events_timestamp_idx`, `consciousness_events_type_timestamp_idx` (events by type, newest first) |
| `safety_audit` | every safety decision with its checks and reasoning | `safety_audit_timestamp_idx` (newest first, with a sequence tie-break), `safety_audit_review_idx` (partial, decisions that need review) |

**The role script.** `scripts/db/init_postgres.sh` creates the application
role and database for a Homebrew or apt install (the Compose image creates
both from its environment). It reads only `MTC_DB_PREFIX` and the
`MTC_POSTGRES_*` lines of `.env` with a line parser, connects as the
administrator through the standard `PG*` variables, creates the role and the
database when they do not exist, never changes an existing role's password and
never drops anything, then applies the migrations as the new role (its
password in `PGPASSWORD` for that step only, never on the command line).
`--dry-run` prints the statements without the password; `--no-migrate` stops
after the role and database.

**Safety nets.** Each store module also has an `ensure_*` method that applies
its own DDL at first use. They are secondary: the rendered migrations and
`mtc db init` are canonical.

## MongoDB

Collections and indexes, from the four registry tables. Every index is
created with `create_index`, which is idempotent; `mtc db check` compares
keys and uniqueness. There is no legacy thoughts collection and no TTL index:
retention is the memory pruner's job (below). Every stored time is
timezone-aware UTC.

| Registry table | Collection | Indexes (keys; unique where marked) |
|---|---|---|
| `mtc.neural.mongodb_schemas.NEURAL_INDEXES` | `neural_spikes` | `timestamp`; `neuron_id` + `timestamp`; `emotion_type` + `timestamp`; `intensity` |
| | `liquid_state` | `timestamp`; `coherence`; `entropy` |
| | `htm_columns` | `timestamp`; `column_id` + `timestamp`; `anomaly_score`; `semantic_label` |
| | `thought_streams` | `timestamp`; `stream_id`; `thought_id`; `parent_thought_id`; `thought_type` + `timestamp`; `thought_text` (hashed) |
| | `conversation_turns` | `timestamp`; `conversation_id` + `turn_number`; `speaker` + `timestamp`; `conversation_id` + `timestamp` |
| | `conversation_sessions` | `started_at`; `conversation_id`; `primary_participant` + `started_at`; `phase` + `started_at`; `ended_at` |
| | `thought_gists` | `dominant_emotion` + `created_at`; `retention_strength`; `importance_score`; `gist_id` (unique) |
| | `consciousness_snapshots` | `timestamp` |
| `mtc.memory.mongo_stores.DOCUMENT_INDEXES` | `learning_experiences` | `timestamp`; `speaker`; `outcome`; `topic` |
| | `learning_beliefs` | `belief_id` (unique) |
| | `speaker_models` | `speaker_name` (unique) |
| | `intrinsic_goals` | `goal_id` (unique); `status` |
| | `attention_grabs` | `timestamp`; `grab_id` (unique) |
| | `dreams` | `timestamp` |
| `mtc.privacy.stores.PRIVACY_INDEXES` | `privacy_audit_log` | `event_id` (unique); `timestamp`; `user_id`; `event_type` |
| `mtc.monitoring.stores.ALERT_INDEXES` | `alerts` | `alert_id` (unique); `timestamp`; `alert_type`; `severity` |

The live runtime keeps the privacy audit log in process memory; the
`privacy_audit_log` collection is created for callers that pass a MongoDB
audit store. A refused MongoDB login reports `failed` with a hint naming
`MONGODB_AUTH_SOURCE` (empty uses `MONGODB_DB`, the database `MONGODB_URL`
names; set it to the database where the user is defined).

## Qdrant

One collection, `QDRANT_COLLECTION`, from
`mtc.neural.vectors.coordinator.COLLECTION_PARAMS` and `PAYLOAD_INDEXES`:

| Parameter | Value |
|---|---|
| vector size | 384 (`VECTOR_DIMENSIONS`) |
| distance | cosine |
| HNSW `m` | 32 |
| HNSW `ef_construct` | 400 |
| HNSW `full_scan_threshold` | 20000 |
| HNSW `on_disk` | false |

Payload indexes: `postgres_id` (integer), `memory_type` (keyword) and
`importance` (float). Each point's id is `point_id_for_memory` of the
PostgreSQL row id, so a point can be found and deleted without a search.

`mtc db init` creates the collection when it is missing and adds any missing
payload index. An existing collection with another size or distance is
`failed` and never recreated: point `MTC_QDRANT_COLLECTION` at a new name and
run the re-embed backfill. An 8192-dimension collection from an older
deployment is handled the same way. A different HNSW parameter or a missing
payload index is an advisory `warning`.

**The embedding model.** Vectors come from `EMBEDDING_MODEL`
(`sentence-transformers/all-MiniLM-L6-v2`, the `embeddings` extra). It is the
memory layer's one network dependency: the model is downloaded from the model
hub on first use. `mtc doctor` reports whether it is in the local cache and
`mtc doctor --fetch-models` downloads it once; after that, set the hub's
offline variable (`HF_HUB_OFFLINE=1`) so nothing is fetched again. Without
the model nothing is embedded and recall falls back to full-text search,
labelled `text`.

**The re-embed backfill.** `python3 -m mtc.db.reembed` (or
`scripts/db/reembed_postgres_to_qdrant.py`) reads `episodic_memory` by keyset
pagination on `id`, skips archived rows, asks Qdrant which points already
exist, and embeds and upserts only the missing ones with the payload
`postgres_id`, `memory_type`, `importance` and `created_at`. `--dry-run`
counts without writing; `--all` re-embeds every row (after a model change). A
non-semantic embedder is refused unless explicitly allowed, and an
unavailable embedder writes nothing and exits non-zero.

## Neo4j

Constraints (`mtc.knowledge.graph_manager.CONSTRAINTS`): unique
`Concept.name`, a unique `id` per label (`Person`, `Concept`, `Memory`,
`Experience`, `Emotion`, `Topic`), unique `Experience.conversation_id`, and an
index on `Person.name`; the names are `concept_name_unique`,
`person_id_unique`, `concept_id_unique`, `memory_id_unique`,
`experience_id_unique`, `emotion_id_unique`, `topic_id_unique`,
`experience_conversation_unique` and `person_name`. `mtc db init` applies them
through `KnowledgeGraphManager.bootstrap_constraints()`; `mtc db check`
matches them by label, properties and type.

Seeds (`mtc.knowledge.graph_manager.SEED_CONCEPTS`): `consciousness`,
`emotions` and `neural networks`, plus the system and creator `Person` nodes,
merged once by the bootstrap. The community edition holds one graph and MTC
does not namespace nodes by prefix, so give each install its own server. See
[KNOWLEDGE_GRAPH.md](KNOWLEDGE_GRAPH.md).

## Redis

Redis has no schema; `mtc db check` pings it (required), reads the eviction
policy (advisory) and counts keys per family under `REDIS_KEY_PREFIX`
(advisory). It never runs `CONFIG SET` and never flushes.

Key families (`mtc.memory.redis_manager.KEY_FAMILIES`), each under the key
prefix:

| Pattern | Type | Expiry | Holds |
|---|---|---|---|
| `wm:conversation:{session}` | list | 1800 seconds, refreshed on each push | the newest conversation messages |
| `emotion:current:{session}` | string (JSON) | 14400 seconds | the session's current emotional state |
| `emotion:history:{session}` | stream | none (capped by length) | raw emotion-dimension samples |
| `session:{id}` | string (JSON) | only when the caller passes one | session context |
| `cache:{key}` | string (JSON) | 30 seconds | short-lived cached values |

Because some families carry no expiry, the recommended eviction policy is
`volatile-lru` (only keys with an expiry are evicted under memory pressure);
`allkeys-lru` also reads `ok`, any other policy is a `warning`. The Compose
file starts Redis with `--maxmemory-policy volatile-lru`.

## Snapshot roles

Two stores hold snapshots, with different roles:

- PostgreSQL `consciousness_snapshots` and `consciousness_events` are the
  canonical aggregate record: the snapshot service
  (`mtc/consciousness/snapshot_service.py`) writes each periodic or
  event-driven assessment snapshot there, and `GET
  /api/consciousness/snapshots` reads them. The service also writes a JSON
  backup per snapshot under `<RESULTS_DIR>/consciousness_snapshots`.
- MongoDB `consciousness_snapshots` holds module-level snapshots written
  through the neural data manager.

## Data flow of one turn

See the store table in [RUNTIME_PIPELINE.md](RUNTIME_PIPELINE.md): the
message and the reply go to `messages` and `conversation_turns` (with the
provenance snapshot) and to the working-memory list; recall reads
`core_memory`, `episodic_memory` and the vector collection; the post-turn
hooks write one emotional-history row, one episodic memory with its vector,
the learning record, the speaker model and the concepts.

## Retention

The memory pruner (`mtc/memory/memory_pruner.py`), run by the sleep
scheduler, decays episodic retention with true half-lives, archives or deletes
weak episodic rows (never `core` rows or the core memory blocks) together with
their vector point and memory node, removes old neural telemetry, old
low-coherence childless thoughts and expired unprotected gists. MongoDB
collections have no TTL index; this pass is their retention.

## Backup and restore

`scripts/ops/backup.sh` dumps each enabled store with its native tool into a
timestamped directory under `BACKUP_LOCATION` with a manifest of sizes,
SHA-256 sums and counts; `restore.sh` restores one store after confirmation;
`restore_drill.sh` restores the newest backup into `_restore_test` targets and
compares counts. Redis is not restored (working memory expires by design) and
Neo4j is dumped only offline on request. See [OPERATIONS.md](OPERATIONS.md).

## Privacy of persisted content

The stores hold conversation text, names, memories, thoughts, beliefs and
emotional records. Keep them on loopback or behind authentication, keep their
backups private (see [SECURITY.md](SECURITY.md)), and never commit an export.

## Persistence outside the stores

| Path (Settings key, default) | What | Conversation text |
|---|---|---|
| `STATE_DIR` (`agent_space/consciousness_state`) | the persistence hub's module state, saved every 300 seconds and at shutdown | fragments (topic words, recent content) |
| `AGENT_SPACE_DIR` (`agent_space`) | the identity statement and working files | no |
| `RESULTS_DIR` (`results`) | assessment results, snapshot backups, model-transition probes | the transition probes hold the system's own answers |
| `SHADOW_LOG_DIR` (`./logs/shadow_signals`) | shadow signal logs | may hold conversation-derived text |
| `LOG_DIRECTORY` (`./logs`) | `mtc.log` and the safety, metrics and emotional JSON-lines logs | the specialty logs hold private records |
| `PROVIDER_STATE_PATH` (`./data/provider_state.json`) | the active provider id | no |
| `DCM_CACHE_DIR` (`data/dcm_cache`) | the DCM scheme and evaluator caches | no |
| `<DATA_DIRECTORY>/pyphi` (`./data/pyphi`) | PyPhi's home: its configuration copy, cache and log | no |

All of these resolve under `PROJECT_ROOT` and are gitignored.
