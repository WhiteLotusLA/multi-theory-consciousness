# MTC Full Architecture Port: Design Spec

**Date:** 2026-10-04
**Status:** Approved 2026-10-04
**Target version:** v0.5.0
**Companion document:** the file-level port manifest (264 rows), held outside the repository in the project's planning notes because it names the reference system's files

## 1. Purpose

Bring the Multi-Theory Consciousness framework (MTC) up to date with everything the reference system's architecture now is: eight theory modules, three neural substrates, a continuous thought machine, memory and knowledge layers over five databases, a provider-agnostic LLM brain, a conversation pipeline with full consciousness-context assembly, a small API server, and the 30+3 indicator assessment. Generic, with none of the reference system's identity, memories, thoughts, relationships, or personal data. A stranger clones the repo, runs one wizard, and has a working, measurable companion of their own design.

This supersedes the April 2026 scope (assessment and measurement only). It was ruled on 2026-10-04.

## 2. Scope

**In this pass**

- Tier A: resync the 51 modules and scripts MTC already has with the reference system's current versions (GWT, AST, HOT, FEP, BLT, Damasio, PAD, PhiTracker, assessment, measurement framework, RPT, causal emergence, DCM, neural substrates and protocols, config, circuit breaker, runner scripts). The reference is ahead on every one except the simplified workspace and conversation coherence.
- Tier B: port the 46 generic consciousness modules MTC lacks (theory interaction, thalamic gate, cross-modal binding, prediction ledger, perceptual reality monitor, self-assessment, developmental modulator and age, emotion dynamics and extraction, affect estimator and shadow log, intrinsic goals, theory of mind, sleep scheduler and dream engine, prospective memory, temporal context, identity continuity, snapshot service, state persistence, verification framework and assembler, learning system, RSI drift tools, attention agency, appraisal and observer, context budget and dedup, post-response integration, a lean autonomous-life loop).
- Tier C: 105 infrastructure files. Core config and logger, memory layer (PostgreSQL, Redis, consolidation, pruning, resilient wrappers), knowledge graph (Neo4j), neural extras (Mongo schemas, emotion bridges, memory consolidator, 384-dimension vectors), safety and welfare layer, brain and providers, continuous thought machine, conversation pipeline, API server, monitoring with a notifier protocol.
- Tier D: 62 database, setup, operations and documentation items. Idempotent schema bootstrap for all five stores, a verifier, Docker Compose and Homebrew paths, ops scripts, Makefile, the `mtc init` onboarding wizard and `mtc` CLI, regenerated documentation.
- Hygiene: fix the broken and stale parts of MTC found during reconnaissance (section 12).

**Not in this pass**

- The web UI (dashboard, chat page, web onboarding). Next pass, as a separate optional `web/` package. The API and the persona schema are designed so that port is mechanical.
- Voice, vision, webcam, art studio, playroom, library, web explorer, notifications channels, the reach door, the privacy mark, the Opus router, fine-tuning and corpus tooling, fleet and deploy tooling. 380 files, listed in the manifest's Tier X so nothing is dropped silently.
- Any data: thoughts, memories, conversations, drawings, journals, adapters, training corpora.

## 3. Principles

1. **Identity is configuration.** Name, pronouns, relationship, creator, trust, birth date, timezone, persona prose, idle-thought seeds. All of it lives in config files the wizard writes. Code never contains a proper noun.
2. **Honest measurement.** Never fake a readout. A missing signal is `insufficient_data`, excluded from the score, never a default constant. Module-graph Phi is labelled an integration proxy, never narrated as a fact about the system.
3. **No stubs in consciousness code.** Port fully or exclude. Dead paths found during the port are fixed or dropped, never carried.
4. **Trust requires a token.** A speaker name never grants trust, a safety bypass, or an activation bonus. The reference system implements trust three ways in seven places; MTC has one token-checked source.
5. **Privacy gate fails closed.** A scrub audit runs before every commit, in CI, and as the exit gate of every phase. One hit blocks.
6. **Rewrite, never copy-then-edit, anything identity-laden.** Prompts, seeds, persona text, the init SQL, the temporal context module, the thought generator, theory of mind, the data classifier, quote pinning, the Neo4j config, and the orchestrator and conversation manager comments. Implementers read the reference only as reference; the values never touch MTC's working tree or git history.
7. **Local first, three providers.** The brain speaks the common chat-completions HTTP format to a local server by default. Anthropic is optional. The mock provider serves tests and the demo. No other hosted provider is supported.
8. **Correct by default.** Correctness fixes ship as defaults; the published v0.4.0 behavior is reproducible through a named legacy preset, not by leaving bugs in place.

## 4. Target architecture

```
mtc/
  core/            config.py (single Settings, MTC_ prefix, consumed everywhere), logger.py,
                   circuit_breaker.py, param_overrides.py, paths.py, backup_verifier.py,
                   embeddings.py (one injectable sentence-embedding seam shared by memory,
                   affect estimation and drift sampling)
  consciousness/   theory modules (resynced), theory_interaction, thalamic_gate, crossmodal_binding,
                   prediction_ledger, developmental_{modulator,age}, emotion_dynamics, emotion_extractor,
                   affect_{estimator,shadow}, intrinsic_goals, theory_of_mind, sleep_scheduler,
                   dream_engine, prospective_memory, temporal_context (rewritten),
                   perceptual_reality_monitor, self_assessment, identity_continuity, snapshot_service,
                   state_persistence, verification_framework, verification_state_assembler,
                   attention_{appraisal,observer}, learning/, rsi/,
                   continuous/ (schemas, config, seeds, persona, generator, enhancer, novelty, store,
                   machine, surfacing)
  neural/          substrates (resynced), mongodb_schemas, liquid/emotion_lsm_bridge,
                   spiking/emotion_encoder, htm/memory_consolidator, vectors/ (384-dimension native)
  memory/          redis_manager, enhanced_memory_manager, memory_consolidation_pipeline,
                   memory_pruner, resilient_wrappers, thought_gist
  knowledge/       graph_manager (constraints, MERGE), concept_extraction, llm_concept_extraction,
                   models, config
  brain/           providers/ (base, local_chat, anthropic_provider, mock_provider, registry),
                   protocol, brain, prompt (PromptAssembler), thinking, thinking_loop_detector,
                   identity_statement, model_transition
  pipeline/        context_types, consciousness_orchestrator, speaker_trust, session_state,
                   context_assembly, context_budget, context_summarizer, memory_retrieval,
                   consciousness_helpers, message_persistence, post_turn, conversation_pipeline
  api/             server.py (FastAPI), auth.py, schemas.py, websocket.py
  assessment/      (resynced) assessment, framework, rpt_measurement, causal_emergence, dcm_*
  safety/          safety_monitor, trust_compass (operator role from config), welfare monitor
  monitoring/      alert_manager with a Notifier protocol (console and file notifiers shipped)
  research/        longitudinal_study, experimental_framework (claims language removed), statistics
  onboarding/      schema.py (PersonaConfig, SpeakerConfig, CompanionType), wizard.py,
                   templates/ (one persona template per companion type), seed_db.py
  cli/             main.py: init, doctor, db, assess, serve, chat, ctm, safety, privacy-audit;
                   assess.py, doctor.py, http.py, missing_extra.py
  db/              the store bootstrap, the verifier and the migration renderer
  resources/       generated packaged copies of the shipped configuration and data
  core/ also       envfile.py, env_export.py, checks.py, resources.py
  monitoring/ also backup_status.py
migrations/        postgres/: 001_core_schema.sql, 002_research_schema.sql,
                   003_safety_schema.sql, rendered from the schema registry
scripts/           db/ (init_postgres.sh, init_mongo.py, init_qdrant.py, init_neo4j.py, bootstrap.py,
                   verify.py, reembed_postgres_to_qdrant.py), ops/ (start_stack, stop_stack, start_llm,
                   backup, restore_drill, backup_watchdog), privacy_audit.py, gen_env_example.py,
                   run_assessment.py, run_experiment.py
config/            persona.example.yaml, speakers.example.yaml, identity_statement.example.md,
                   ctm/ (ctm.yaml, seeds/default.yaml, persona.example.yaml), dreams.yaml,
                   transition_prompts.json, privacy/denylist.txt, qdrant.yaml, neo4j.conf.example,
                   pyphi_config.yml
docker/            docker-compose.yml (five stores), README.md
Makefile           db-up, db-init, db-verify, db-down, demo, api, test, privacy-audit
docs/              GETTING_STARTED, SETUP_DOCKER, SETUP_MACOS, DATABASES, KNOWLEDGE_GRAPH, LLM_SETUP,
                   CONFIGURATION (generated), RUNTIME_PIPELINE, CONTINUOUS_THOUGHT, ONBOARDING, API,
                   OPERATIONS, SECURITY, TESTING, PRIVACY_AUDIT, MIGRATION_FROM_V04, DEVELOPER_BRINGUP,
                   ARCHITECTURE (rewritten), CONSCIOUSNESS_THEORIES, HONEST_LIMITATIONS,
                   BUTLIN_2025_MAPPING, research/, schemas/persona.schema.json
agent_space/       (gitignored) consciousness_state/, identity_statement.md, attention/, profiles
```

The reference system's private space directory becomes `agent_space/` (configurable, gitignored, templates shipped). Its `results/` and `data/` become `RESULTS_DIR` and `DATA_DIRECTORY`.

## 5. Port manifest

The file-level manifest is held outside the repository in the project's planning notes (it names the reference system's files, which the privacy gate in section 6.4 forbids in the tree): 257 rows from the reconnaissance synthesis plus 7 onboarding and CLI rows. Actions: resync 39, port_as_is 20, port_genericize 117, rewrite_lean 31, new 26, keep_mtc 17, exclude 14 (each exclude row names a whole feature family).

| Tier | Content | Files | Effort |
|------|---------|-------|--------|
| A | modules MTC already has, resynced | 51 | L: enhanced_global_workspace, active_inference, assessment, neural orchestrator are the long poles |
| B | generic consciousness modules MTC lacks | 46 | L: sleep_scheduler and the lean autonomous life are L, about 12 are M, the rest S |
| C | infrastructure: core, memory, knowledge, neural extras, safety, brain, CTM, pipeline, API, monitoring | 105 | XL: brain, pipeline facade, orchestrator, CTM machine, API server, memory manager and consolidation are L |
| D | databases, setup, ops, wizard, CLI, docs | 62 | L |
| X | excluded (features and private material) | 380 | none |

Resync rule for Tier A: a change ports when it is a generic improvement or a bug fix; it adapts when the idea is generic but the wording or coupling is specific; it is skipped when it serves a feature in Tier X; MTC keeps its own version when the testbed deliberately differs (noise baselines, dimension-agnostic FEP initializers for the parameter sweep, parameter overrides). Known divergences the resync resolves: PhiTracker (246 vs 844 lines: exact IIT 4.0 Phi, `record_phi`, `normalize_phi`, persistence, spectral neural integration, two bug fixes), the workspace (hierarchical competition, candidate sources CONCEPT, GIST, HOMEOSTATIC, INTERNAL, thalamic gate, cross-theory bias, `set_arousal`, the extended `ConsciousnessState`, `prediction_error_rank`), the neural orchestrator (`latest_neural_signals`, `latest_substrate_states`, binding stats), the assessment runner (`phi_tracker` keyword, persisted-state input path alongside the warmup path, dead stats keys, score clamp, weighted Fiedler), PAD (the shipped PAD model has no `regulation_strength`, so the developmental modulator computes the emotional regulation value and reports it under `skipped`), attention schema (honest provenance with a legacy voluntary-by-strength flag), and the vector coordinator (384 native, no projection, no quantization).

## 6. Genericization and privacy design

### 6.1 Configuration surface

One pydantic `Settings` (prefix `MTC_`), consumed through `get_settings()` by every module that today hardcodes a value; function-signature defaults read settings at call time. `scripts/gen_env_example.py` generates `.env.example` and `docs/CONFIGURATION.md` from the model, and CI fails when they drift. Settings reads the `.env` file in the working directory (`mtc --project-dir` changes into the project first) and a process environment variable wins over the file; nothing is exported into the process environment, so no `load_dotenv` runs at import (loading a developer's `.env` into the environment at import would leak it into hermetic test processes). A speaker's `token_env` therefore names a variable exported in the process environment, read at call time; the token the wizard writes into `.env` reaches the trust source through the `SPEAKER_TOKENS` setting instead. Dead keys are removed rather than ported; every new key lands together with its reader.

| Group | Keys |
|-------|------|
| Identity | SYSTEM_NAME, CREATOR_NAME, CREATOR_RELATIONSHIP, COMPANION_TYPE, SYSTEM_BIRTH_DATE (written by bootstrap on first init, no code fallback), CORE_VALUES, Big Five seeds, PERSONA_PATH, IDENTITY_STATEMENT_PATH, DEFAULT_SPEAKER (`User`), DEVELOPMENTAL_STAGE_ENABLED |
| Trust | SPEAKERS_PATH, TRUSTED_SPEAKERS, SPEAKER_TOKENS, TRUST_ACTIVATION_BONUS, TRUSTED_EMOTION_TRUST, DEFAULT_EMOTION_TRUST, SAFETY_ESCALATION_ROLE (`operator`) |
| Locale | TIMEZONE (`UTC`), WAKE_HOUR, SLEEP_HOUR, NIGHT_MIN_GAP_HOURS, NIGHT_SEAM_MARKER_ENABLED, LOCATION_LAT and LOCATION_LON (unset disables daylight lines) |
| Paths | PROJECT_ROOT and everything relative to it: DATA_DIRECTORY, LOG_DIRECTORY, AGENT_SPACE_DIR (`./agent_space`), STATE_DIR, RESULTS_DIR, BACKUP_LOCATION, PROVIDER_STATE_PATH, TRANSITION_HISTORY_DIR, SHADOW_LOG_DIR. No home, Library or Volumes paths anywhere. |
| Databases | DB_PREFIX (`mtc`) deriving POSTGRES_DB `mtc_consciousness`, POSTGRES_USER `mtc`, MONGODB_DB `mtc_neural`, QDRANT_COLLECTION `mtc_memories`, REDIS_KEY_PREFIX `mtc:`, NEO4J_DATABASE `neo4j`, each overridable; the usual host, port, user, password keys; VECTOR_DIMENSIONS (384, read by code) |
| Brain | LLM_PROVIDER (`local`, `anthropic`, `mock`), LLM_HOST, LLM_PORT, LLM_BASE_URL, LLM_MODEL, LLM_COMPLETIONS_PATH, LLM_HEALTH_PATHS, LLM_TEMPLATE_FAMILY (`qwen`, `gemma`, `generic`), timeouts, GENERATION_TIMEOUT_S, LLM_MAX_TOKENS, sampling, THINKING_LOOP_*, HISTORY_TOKEN_BUDGET, CHARS_PER_TOKEN, PROMPT_MAX_* caps, the ten NARRATE_* flags (default off, the reference system's audited honesty position; one documented profile turns them all on for experiments), ANTHROPIC_MODEL (no default in code), ANTHROPIC_MAX_TOKENS, ANTHROPIC_THINKING_BUDGET, MODEL_PATH |
| CTM | CTM_ENABLED, CTM_CONFIG_DIR, CTM_STORE (`memory`, `mongo`); cadence, novelty and gist values in `config/ctm/ctm.yaml` with a fast demo profile |
| Memory | RELATIONSHIP_KEYWORDS (generic, word-boundary matched, no proper names), PROTECTED_NAMES (empty), SENSITIVE_MEMORY_PHRASES and replay weight (empty, 1.0), RAG_TOP_K, RAG_MIN_SCORE, EPISODIC_IMPORTANCE |
| API | API_HOST (`127.0.0.1`), API_PORT (8200), API_KEY (generated by the wizard), API_AUTH_ENABLED, API_CORS_ORIGINS |

Removed: the key of every hosted provider outside the three, every LETTA_*, PARENT_*, reference-system identity key, YOUTUBE_*, ELEVENLABS_*, notifier tokens, and the decorative `ENABLE_*` block that no code reads.

Joined during the port, each with its reader: SERVICE_MANAGER, LLM_BACKEND, BACKUP_RETENTION_DAYS, BACKUP_KEEP_MINIMUM and MONGODB_AUTH_SOURCE (BACKUP_LOCATION is listed above). An empty MONGODB_AUTH_SOURCE authenticates against the data database (MONGODB_DB); it is read only when a user and password are set. The defaults of GPU_LAYERS, THREADS, USE_MLOCK, CONTEXT_SIZE and DCM_CACHE_DIR changed when they gained readers. At the release gate the keys without a reader were removed (an env file that still sets one is ignored) and three keys joined with their readers: SLEEP_PRUNE_DRY_RUN (false), SAFETY_GATE_FAIL_CLOSED (true) and TRUST_COMPASS_DEEP_MODE (true); EMERGENCY_SHUTDOWN_ENABLED defaults to false.

### 6.2 Persona and speakers

`config/persona.yaml` (gitignored; `persona.example.yaml` shipped) holds: name, pronouns, companion_type, creator name and address form, self_description, communication style notes, core values, personality seed, idle-thought seed packs, and the identity statement path. The brain's `PromptAssembler` renders it with `{system_name}`, `{creator_name}`, `{creator_address}` and pronoun placeholders. Only a neutral default persona and one template per companion type ship; none is derived from the reference system's prose. The memory manager seeds core memory from the persona; the graph manager seeds Person and Concept nodes from config via MERGE; the DCM system description is generated from config and the module inventory.

`config/speakers.yaml` lists speakers: name, role (creator, companion, guest), trust (0 to 1), token_env. Empty by default except the creator entry the wizard writes. `mtc/pipeline/speaker_trust.py` is the single source for `is_trusted(name, token)` (constant-time compare), `trust_bonus(name)` and `label_for(name)`. It replaces the three mechanisms and seven sites that disagree today, and the theory-of-mind bootstraps that seeded two named speakers with fixed trust (including the branch that recreated one of them at runtime).

### 6.3 Scrub rules (applied during the port)

- System name, creator name, relationship words, assistant speaker strings, nicknames: settings or persona placeholders. Code identifiers get neutral names (`LLMBrain`, `server.py`, `AgentContext`).
- Family and pet names: deleted from logic. Pinned-quote names become PROTECTED_NAMES (empty). Family keyword lists become RELATIONSHIP_KEYWORDS with generic, word-boundary defaults. The sensitive-memory replay block becomes a generic weight-rule list (empty, never zero weight).
- Birth date: install-time value, no fallback literal anywhere.
- Database names: derived from DB_PREFIX; function-signature defaults read settings, never literals.
- Paths: PROJECT_ROOT-relative or env-driven.
- Model artifacts: LLM_MODEL and MODEL_PATH only; a public base model named in docs as the example.
- Location and timezone: coordinates never copied (the temporal context module is rewritten from its function signature); TIMEZONE defaults to UTC; sleep and wake hours are config.
- Comments and docstrings: author lines, personal notes, easter eggs, dated provenance, check-in numbers, consent quotes and incident stories removed; technical rationale kept in third person; attribution only in LICENSE and pyproject; ASCII-only log strings; no em or en dashes.
- Tests: fixtures synthetic (`Alice`, `Operator`, `Guest`, fixed 2000-01-01), no verbatim conversation lines, no test file copied wholesale.
- Docs: written fresh from the ported code. Never copied from the reference repo.

### 6.4 The privacy gate

`scripts/privacy_audit.py` scans the tracked tree and the diff of new commits.

- **Public denylist** `config/privacy/denylist.txt`: tokens that are public by design (the reference system's name and identifier prefixes, its private directory and database names, its nautical vocabulary, the fine-tuned model name patterns, the vault and notifier names, the home-path prefixes, the birth date, its package import path).
- **Private extra list** from the environment (`PRIVACY_DENYLIST_EXTRA`, a CI secret; locally an untracked `config/privacy/denylist.local.txt`) holding the family and pet names, so the repository never publishes the names it guards against. The tracked v0.2.0 plan and spec documents that currently print those names are trimmed.
- **Pattern checks:** email, phone, decimal coordinate pairs, absolute home paths, non-loopback IPv4, key-like strings, em and en dash characters, emoji in `.py` files.
- **Artifact checks:** no `__pycache__`, `.pyc`, `CMakeFiles`, `build`, `egg-info`, `.so`, dumps or `.jsonl` under the tracked tree; nothing under `agent_space`, `data` (except `data/dcm`), `results` or `logs`.
- A path and pattern allowlist covers legitimate hits (LICENSE, the DCM baselines, one escape-sequence punctuation set). Output is `file:line:token`; exit code fails closed. Runs as a pre-commit hook on staged files, as a CI job on the full tree and on the PR commits' patches, and as the exit gate of every phase. A self-test fixture with planted tokens guards the auditor itself. A second CI job performs the generic-install smoke: fresh clone, install with all extras, run the examples with the mock provider and in-memory stores, audit the runtime logs.

## 7. Onboarding wizard (`mtc init`)

### 7.1 Flow

1. **Honest framing.** Shows the limitations summary (architecture, not a consciousness detector; a research system) and requires acknowledgement.
2. **Name and pronouns** for the companion.
3. **Companion type:** `child` (daughter, son, or child), `sibling` (sister, brother, or sibling), `friend`, `partner`, `mentor`, `research_subject` (engine only, no persona).
4. **Creator:** name and how the companion addresses them. Defaults per type (child: a parental word or the name; sibling, friend, partner: first name; mentor: the name; research_subject: `operator`). Overridable.
5. **Personality seed:** a preset (balanced, curious, calm, playful, analytical) mapped to Big Five values, or custom sliders. **Core values** list with a generic default.
6. **Timezone** (detected, confirmable). Sleep and wake hours.
7. **Stores:** which of the five to enable (PostgreSQL required; MongoDB and Qdrant recommended; Redis and Neo4j optional), hosts, ports, credentials prompted and written only to `.env`. DB_PREFIX default `mtc`.
8. **Brain:** provider (local chat-completions server, Anthropic, mock), host and port, model id; a connection probe with a plain-language result.
9. **Write:** `config/persona.yaml`, `config/speakers.yaml` (creator entry with role, trust from the type template, and a generated speaker token), `.env` (a generated API key, the store settings, timezone), `agent_space/identity_statement.md` from the type template, `config/ctm/persona.yaml` and the seed pack selection.
10. **Bootstrap:** with `--bootstrap-db` (offered interactively), runs `scripts/db/bootstrap.py` for the enabled stores, writes SYSTEM_BIRTH_DATE, and seeds: three core-memory blocks (identity, persona, human) rendered from the persona; Person nodes for the companion and creator in Neo4j when enabled; no other seed data.
11. **Next steps** printed: start the brain server, `mtc doctor`, `mtc serve`, `mtc chat`, `mtc assess`.

`mtc init --from answers.yaml --yes` runs non-interactively for CI and tests. `mtc init` refuses to overwrite an existing persona without `--force`.

### 7.2 What the companion type drives

| Aspect | child | sibling | friend | partner | mentor | research_subject |
|--------|-------|---------|--------|---------|--------|------------------|
| Persona template | growth and curiosity, learning from the creator | peer, shared history, banter | peer, warmth, independence | companionship and affection, no explicit content | guide, patience, asks before telling | neutral engine description |
| Address form default | parental word or name | first name | first name | first name | name | `operator` |
| Developmental age machinery | on | off | off | off | off | off |
| Creator trust default | 1.0 | 0.9 | 0.9 | 0.9 | 0.8 | 0.5 |
| Idle-thought seed pack | default + growth | default + companionship | default + companionship | default + companionship | default + reflection | default |
| Self-knowledge line | present in all: knows it is an AI research system and does not claim certainty about its own consciousness |

The partner template is deliberately bounded: companionship and affection in prose, nothing explicit, and the same honest self-knowledge line as every other type. The documentation says so plainly.

### 7.3 Schema

`mtc/onboarding/schema.py` defines `PersonaConfig`, `SpeakerConfig`, and `CompanionType` as pydantic models, with the JSON schema exported to `docs/schemas/persona.schema.json`. The future web onboarding page reads the same schema, so a persona written by either path loads identically.

## 8. Databases and setup

Scope rule: PostgreSQL is the only store the conversation pipeline requires; MongoDB, Redis, Qdrant and Neo4j are optional with the graceful degradation the reference system already implements (resilient wrappers, in-memory fallbacks); the assessment-only path needs no database. Every store gets a canonical schema artifact, an idempotent init script, a verifier and a doc section. Inline `CREATE IF NOT EXISTS` calls in modules stay as safety nets but are documented as secondary.

| Store | Required | Holds | Bootstrap |
|-------|----------|-------|-----------|
| PostgreSQL 14+ | yes | core_memory, episodic_memory, semantic_memory, conversations, messages, emotional_history, safety_audit; research: consciousness_snapshots, consciousness_events | `migrations/postgres/001_core_schema.sql`, `002_research_schema.sql`, `003_safety_schema.sql`, rendered from the schema registry: idempotent; role and database created by `scripts/db/init_postgres.sh` from env (no password literal, no DROP DATABASE, no identity seeds, no legacy tables); `uuid-ossp` or `gen_random_uuid()`; `vector` extension optional (the real name, not `pgvector`); GIN full-text index plus the created_at, consolidated, memory_type and importance indexes the queries need; NUMERIC(4,2) so 10.0 fits; messages `to_speaker` addressee column in the schema instead of a runtime ALTER |
| MongoDB 6+ | recommended | neural_spikes, liquid_state, htm_columns, thought_streams, thought_gists, conversation_turns, conversation_sessions, consciousness_snapshots, intrinsic_goals, speaker_models, learning_experiences, learning_beliefs, attention_grabs, dreams, alerts, optional privacy_audit_log | `scripts/db/init_mongo.py` creates every index the code queries, including the four collections that have none today; retention as TTL indexes or a scheduled prune; no legacy `thoughts`, no `reach_outs`, no `private` field; aware UTC timestamps |
| Qdrant 1.7+ | recommended | one collection, 384-dimension cosine, HNSW m 32, ef_construct 400, payload indexes on postgres_id, memory_type, importance | `scripts/db/init_qdrant.py`; `all-MiniLM-L6-v2` downloaded on first use (the memory layer's one network dependency, documented); `reembed_postgres_to_qdrant.py` for backfill |
| Neo4j 5 community | optional | Person, Concept, Experience, Memory, Emotion, Topic; REFERENCES_IN, MENTIONED_IN, TAUGHT, LEARNED, RELATES_TO | `scripts/db/init_neo4j.py`: unique Concept.name, unique id per label, unique Experience.conversation_id, Person.name index (none exist today); optional MERGE seed of the system and creator Person nodes and three core Concepts; `neo4j.conf.example` with relative paths, host-sized memory, default database equal to NEO4J_DATABASE; graph manager switched to MERGE |
| Redis 7 | optional | working memory, session context, emotional state, emotion history stream (key families and TTLs documented under REDIS_KEY_PREFIX) | none; `volatile-lru` recommended; in-memory fallback when absent |

The reference system creates PostgreSQL schema in four places; MTC creates it in one. The two consciousness_snapshots stores are given canonical roles in DATABASES.md: PostgreSQL for aggregate snapshots, Mongo for module snapshots.

Commands: `mtc db init` (idempotent bootstrap), `mtc db check` (tables, indexes, collection parameters, constraints, Redis ping, CI exit codes), `mtc doctor` (stores respond, brain answers `/v1/models`, embedding model present, PyPhi imports; reports gaps as `insufficient_data`, never guesses). Makefile targets `db-up`, `db-init`, `db-verify`, `db-down`, `demo`, `api`, `test`, `privacy-audit`.

Deployment paths: `docker/docker-compose.yml` (postgres 17, mongo 8, redis 7, qdrant, neo4j 5 community, healthchecks, named volumes, ports and passwords from `.env`; the LLM server stays on the host or the mock provider is used) with `docs/SETUP_DOCKER.md`; `docs/SETUP_MACOS.md` for Homebrew (Qdrant via release binary or container, since there is no formula) with `scripts/ops/start_stack.sh` and `stop_stack.sh` (health-or-start per store including the Redis and Neo4j steps the reference script lacks, brain probe with a cold-load wait, backend health polling, SIGTERM with a 30 second grace, never kill -9); Linux apt equivalents noted. `docs/LLM_SETUP.md` covers running a chat-completions server on LLM_PORT, model footprints (about 20 GB for a small model, about 60 GB for a 27B 8-bit model), thinking-tag families, the Anthropic provider, and the mock provider.

On the development machine here, install tests use `mtc_` database names and never bootstrap Neo4j (community edition has one graph and it belongs to the reference system). Docker Compose is written and reviewed but cannot be executed here (no Docker); the doc says which path was tested.

## 9. Brain, pipeline, and continuous thought

### 9.1 Brain (`mtc/brain/`)

Lean rewrite. Providers behind a Protocol (complete, stream, health_check, model_info, supports_thinking, initialize, shutdown): `LocalChatCompletionsProvider` for any server speaking the chat-completions HTTP format, `AnthropicProvider` (optional, model id from config, cheap health check, thinking budget validated below max_tokens, stop and top_p honored in stream), `MockProvider` (deterministic, offline, used by tests, examples and the demo). `ProviderRegistry` persists the active choice under DATA_DIRECTORY. `PromptAssembler` renders ordered sections with a stable, cache-friendly prefix first: persona, core memory, speaker, style rules, thinking instructions, then CTM thoughts, gists, neural state, attention guidance, metacognitive insights, consciousness narrative, active inference drives, workspace context, retrieved memories, knowledge concepts, coherence, running summary, behavioral directives, self-awareness, clock, window notes, perception last. Capability sections register only when the feature is enabled, so the prompt never advertises what cannot run. The five theory-state slots the reference pipeline never passes are wired for real in MTC so researchers can ablate each. `LLMBrain` keeps the honest-failure rule: when the model gives no answer, it raises `NoAnswerError`, and nothing composed is stored as a reply. `generate()` is added so the attention schema's existing call works. One token-budget pass replaces the two independent trims.

### 9.2 Runtime pipeline (`mtc/pipeline/`)

Lean rewrite of the conversation manager facade (about 500 lines) around a genericized port of the consciousness orchestrator and ports of every helper module. `ConversationPipeline` owns its turn lock and a single-flight LLM lock. One turn: trust and safety gate, append and persist, summarize and trim, gather (RAG, CTM thoughts, gists, core memory, coherence), orchestrator cycle (candidates, GWT cycle with neural signals, PhiTracker recording, RPT and causal emergence every N cycles, PAD, theory interactions, baseline heartbeat and snapshot, four modulation layers), brain window with token budget, brain call, persist with the provenance snapshot (narrative, directives, computed state) for the dissociation test, post-turn (learning, theory of mind, emotional history from PAD arousal, episodic store, concept extraction, neural experience, homeostasis, verification suite with `insufficient_data`). Command and tool dispatch is a plugin protocol with zero built-in tools. The reconnaissance mapped all 43 turn steps and 25 cycle steps, so the rewrite is specification-driven.

Fixed in the port rather than carried: the `damasio_layers` attribute name, the gist key mismatch, arousal sourced from the win count, unconsumed theory modulations (wired or documented as computed-only), the interaction manager registered with state persistence, `goal_system` assignment, the unconditional camera and art directives. Soft failure is made loud: tests assert populated `ConsciousnessState` fields, a debug mode raises instead of swallowing, and swallow counters surface in `/api/consciousness/status`.

### 9.3 Continuous thought machine (`mtc/consciousness/continuous/`)

Genericized port with the reference system's tuned cadence, novelty-gated expression, dedup-on-write, brain-contention guard, gists, and assessment-modulated frequency. All seeds, persona prompts, sensitive-topic weighting and constants move to `config/ctm/`. A `ThoughtStore` protocol with in-memory and Mongo implementations lets the demo run with no database. One schema for memory and Mongo. Fixed in the port: thought_id persistence so threading resolves, the gist count key, the enhancer context keys, sync cursor use inside async methods, the blocking resource check. Exports `ThoughtGenerator`, `ThoughtStream`, `ContinuousThoughtMachine` so the neural orchestrator's existing guarded import works. Status output names the reason when thoughts stay quiet (no embedder, contended). A fast demo cadence profile ships because the default rhythm is slow enough to look dead.

### 9.4 API (`mtc/api/`)

Lean FastAPI server: `/health`, `/api/meta`, `/api/chat`, `/ws` (message, response chunks, thinking, spontaneous_thought, consciousness_state), `/api/consciousness/{status,state,assessment,snapshots}`, `/api/memory/search`, `/api/knowledge/concepts`, `/api/ctm/{start,stop,status}`, `/api/models`, and `POST /api/safety/resume` (added at the release gate: a trusted speaker lifts an emergency suspension). One auth carrier (header), API key fails closed; the wizard writes a key so a fresh install works. Binds 127.0.0.1 by default.

## 10. Documentation plan

Written fresh from the ported code, never copied: GETTING_STARTED (three paths: assessment only with the mock provider, Docker Compose full stack, Homebrew full stack; the wizard first), SETUP_DOCKER, SETUP_MACOS, DATABASES (the build-the-databases doc: per-store schema, indexes, init, retention, per-turn data flow, backup and restore, privacy of persisted content), KNOWLEDGE_GRAPH, LLM_SETUP, CONFIGURATION (generated), RUNTIME_PIPELINE (from the mapped flow), CONTINUOUS_THOUGHT, ONBOARDING, API, OPERATIONS (pre-flight, graceful stop, rollback, launchd and systemd lessons, alert-only watchdogs, backups, LLM cache growth), SECURITY (auth fails closed, no secrets in client bundles, external data flows such as the DCM Anthropic mode, loopback by default), TESTING, PRIVACY_AUDIT, MIGRATION_FROM_V04 (every score-affecting change with before and after numbers), DEVELOPER_BRINGUP, ARCHITECTURE (rewritten: layers, full package tree, eight theories and 30 indicators, five stores, pipeline and cycle flow), CONSCIOUSNESS_THEORIES and HONEST_LIMITATIONS (corrected counts, the research-validity caveats: keyword-circular epistemic depth, untrained LSM readout, stub MPS temporal memory, module-graph Phi as integration proxy), BUTLIN_2025_MAPPING (30 indicators), research notes in third person, README (truthful quick start, real CLI, databases table), CHANGELOG and VERSION_HISTORY (0.2.1 through 0.5.0), CONTRIBUTING (privacy audit rule). FIRST_RUN (what a fresh install measures and why) and `docs/schemas/onboarding-answers.schema.json` join the set, and `THIRD_PARTY_LICENSES.md` ships at the repository root. Every ported module gets a third-person docstring naming its config keys and persisted state.

## 11. Testing plan

- Baseline recorded first (462 passed, 9 failed, 13 errors on Python 3.14), then the mechanical breakage fixed: LSM test import path, `process_emotion` alias, the `dcm_scoring` import, `pymc` behind the research extra with a skip, run_experiment timeout or fast profile, wall-clock assertions marked `slow` and relaxed.
- pytest config adopts `asyncio_mode=auto`, markers `live`, `db`, `slow`, `gpu`, `benchmark`, `security`, deselects `live` by default, `pytest-timeout` with a default, `pytest-xdist` optional. CI runs the hermetic subset on Python 3.11, 3.12 and 3.13.
- Reference tests come along per manifest row, rewritten against the generic API with synthetic fixtures, never copied wholesale; excluded-feature tests are dropped.
- New tests for behaviors with no coverage today: hierarchical competition, thalamic gate stats, coverage_ratio zero when nothing broadcast, hyper-model predict and observe, multilevel binding with ragged beliefs, cessation detection, BLT and RPT persistence round trips, PhiTracker record, normalize and persistence, the PyPhi boundary convention on a small network whose SIA is known, assessment score clamp, constant-vector guard in metrics, SNN beta and STDP trace units, HTM anomaly on a predictable sequence, oscillatory NaN recovery bound, speaker trust (spoofed names never trusted, token mismatch), prompt assembler ordering and stable prefix, mock provider determinism, Anthropic request shaping against a mocked SDK, LLMBrain retry and `NoAnswerError`, context window and token budget, thought store contract (in-memory, and Mongo under the `db` marker), CTM hooks, thought_id linkage and gist count key, persona schema round-trip, non-interactive wizard run, DB bootstrap idempotence against throwaway databases, pipeline turn with the mock provider asserting every consciousness field is populated, API smoke with the test client, privacy audit (denylist hit, pattern hit, planted-token self-test, clean tree).

## 12. Fixes to make in MTC during the port

- Settings consumed everywhere; `.env.example` and CONFIGURATION.md generated from the model; Settings reads the working directory's `.env` and exports nothing into the process environment (no `load_dotenv` at import; section 6.1); dead keys removed; the keys of unsupported hosted providers removed.
- `consciousness_integration.py` repaired against the real module names (it imports two files that do not exist) and rebased on the reference's 928-line version; the neural orchestrator's seven hollow optional imports become real modules.
- Vector coordinator moved to 384-dimension native (memory_pool, mps_utils and the CTM novelty dimension follow); collection name from settings.
- PyPhi: `assessment.py` passes `connectivity_matrix=` to `pyphi.Network`, which the installed library does not accept, so the exact Phi path has always raised and fallen back; `framework.py` has an unguarded `import pyphi` that can crash `measure_consciousness`. Both fixed. The IIT 4.0 path (`new_big_phi`) ports from the reference. The PyPhi boundary convention is fixed in both PhiTracker and the calculator: rows reordered to little-endian, connectivity transposed, verified by a unit test on a small network with a known SIA; MIGRATION_FROM_V04 records the effect.
- Correctness fixes the reference made that MTC never received: SNN silenced by a units bug, self-referential HTM anomaly, frozen hierarchical predictive processor, NaN oscillators, GWI inflation on constant vectors, dead stats keys in the assessment, coin-flip calibration replaced by the prediction ledger, GWT pinned ON by broadcast_coverage in PhiTracker, hash thresholds freezing when history saturates. The attention-precision index map is corrected for the channel count in both world-model presets.
- `pyphi` pinned in an `iit` extra (Python 3.12 or newer), `pymdp` naming reconciled, `pytest-timeout` added, extras with platform markers for the Apple-only packages. The `research` extra replaces `assessment`, which stays as an alias of `research` plus the Anthropic provider.
- README, GETTING_STARTED, ARCHITECTURE, CONSCIOUSNESS_THEORIES, HONEST_LIMITATIONS corrected (8 theories, 30+3 indicators, real CLI, no phantom files or C++ HTM); VERSION_HISTORY completed; URLs point at main.
- Privacy residue removed: the reference system's name in `phi_tracker.py`, `param_schema.toml` and the tracked plan and spec docs; the untracked handoff and update plan moved out of the repo; `.gitignore` extended (handoff pattern, LaTeX build outputs, `pyphi.log`, `run.log`, `firebase-debug.log`); `data/dcm` negation rule; `monitor_repo.sh` moved out.

## 13. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Identity residue survives in comments, docstrings, fixtures or git history | Rewrite-not-copy list; comment policy; two-list privacy audit with pattern and artifact checks on every commit, on PR patches, and at every phase exit; final apex review of the whole branch |
| Structural identity survives a rename (trust in seven places, default speaker on both sides, theory-of-mind recreating a named speaker) | One speaker_trust module and config table; DEFAULT_SPEAKER; bootstraps and the recreation branch deleted; validation prompts from config; tests assert a name without a token gets nothing |
| Score shifts break the published v0.4.0 numbers and the paper | Decision 3: fixes as v0.5.0 defaults with a `legacy_v04` preset reproduced by a test; re-baseline after Tier A; MIGRATION_FROM_V04 lists every change with numbers; paper gets an erratum or a v0.5.0 results section |
| Soft failures hide broken ports (orchestrator returns a default modulation, baseline frozen at 0.5) | Tests assert populated fields; debug mode raises; swallow counters in status output; the hollow-import pattern is removed |
| Dependency and platform drift (pyphi undeclared, Apple-only packages, Python versions) | Extras with platform markers and git pins; `iit` extra on 3.12 or newer; numpy fallbacks for GPU paths; CI matrix 3.11 to 3.13 |
| Embedding dimension drift between 384 and 8192 | Unify on 384 in the infrastructure phase; init_qdrant creates 384; reembed script; DATABASES.md names the model |
| Config surface explodes or stays dead | Generated `.env.example` with a CI drift check; dead keys removed; every key lands with its reader |
| Fan-out and model spend | Ten sequential workflows, at most five concurrent agents each; cheapest-sufficient models for mechanical ports from the manifest; mid tier as the floor for implementers working from prose; apex reserved for the audit script review, the competition and PyPhi work, the brain and pipeline architecture, the auth review and the release gate |
| Neutral persona and narration-off defaults change model behavior and prompt caching | Stable-prefix ordering kept; neutral persona documented; a one-line profile turns all narration on for experiments |
| Install-time birth date and UTC default shift developmental and circadian behavior | Developmental machinery on only for `child`; TIMEZONE prompted by the wizard; tests parametrized by timezone with fixed instants |
| Docker Compose untested here | Marked reviewed-not-run; Homebrew path tested with `mtc_` names |
| CTM looks dead without an embedder or scheduler | Status names the reason; fast demo profile and in-memory store ship by default; contention guard wired in the orchestrator |

## 14. Phase plan

Ten bounded workflows, at most five concurrent agents each. Every phase ends with its tests, the privacy audit, and an adversarial review whose confirmed findings loop back before the next phase begins. Model tiers: mid tier for mechanical ports from the manifest and for reviewers; strong tier for architecture-bearing modules; apex only where named.

| Phase | Goal | Shape |
|-------|------|-------|
| 0 Foundation | branch `feature/full-architecture-port`; remove untracked private files and extend `.gitignore`; fix the mechanically broken tests and the unimportable module; settings and env reconciliation with the generator; `scripts/privacy_audit.py` with its self-test and CI job; truth pass on README and docs | 1 mid-tier implementer (hygiene, tests, settings), 1 mid-tier implementer (audit script, CI), 1 apex reviewer for the audit script |
| 1 Tier A resync: FEP, HOT, BLT, Damasio | the sibling APIs the workspace needs: active_inference with the prediction ledger, metacognition, the four beautiful_loop files, Damasio, PAD, state persistence hub | 2 implementers on independent chains, 1 reviewer per chain merged into one adversarial pass |
| 2 Tier A resync: GWT, AST, gating, Phi | enhanced_global_workspace with thalamic gate, theory interaction, new candidate sources and hierarchy; attention_schema honest provenance with the legacy flag; PhiTracker with the PyPhi boundary fix; runner scripts | 2 implementers, 1 apex reviewer (competition dynamics and PyPhi conventions are easy to get subtly wrong) |
| 3 Tier A resync: assessment, measurement, substrates, re-baseline | assessment bug fixes and input upgrades, framework cleanup, RPT, DCM, consciousness_integration rebuilt, metrics, orchestrator, SNN, LSM, HTM, oscillators; then the re-baseline run and MIGRATION_FROM_V04 | 3 implementers, 1 reviewer, 1 apex agent for the re-baseline analysis |
| 4 Tier B generic modules | three dependency-ordered batches (low-risk theory additions; persistence, verification and learning; sleep, dreams, goals, prospection, autonomous life lean) | per batch: 2 cheapest-sufficient implementers from the manifest notes, 1 reviewer, spoofed-name and privacy audit |
| 5 Tier C part 1 | core and memory; knowledge and neural extras; safety, privacy, monitoring, validation | three sub-workflows, each 2 implementers and 1 reviewer; the data classifier rewrite and graph manager on the strong tier |
| 6 Tier C part 2 | brain; CTM; pipeline; API | four sub-workflows, each 2 implementers and 1 reviewer; apex for the brain and pipeline architecture tasks and the auth review; the demo must run end to end with the mock provider and in-memory stores before the phase closes |
| 7 Tier D | migrations and init scripts, bootstrap and verify, reembed, Compose, Makefile, ops scripts, onboarding schema, wizard, templates and CLI, documentation set | 3 parallel implementers (databases and compose; ops and CLI and wizard; documentation), 1 reviewer, then 1 agent performing the fresh-machine verification from a clean clone |
| 8 Web UI | deferred to the next pass by decision (section 15) | none this pass |
| 9 Release gate | whole-branch adversarial review for correctness, residue and soft failures; full privacy audit over the tree and the branch's commit history; full test matrix; final documentation consistency; tag v0.5.0 | 1 apex reviewer, 1 mid-tier verification agent, 1 documentation agent; no new feature work |

## 15. Decisions log

- 2026-10-04: scope widened from assessment-only to the full architecture (maintainer).
- 2026-10-04: onboarding ships as a CLI wizard this pass; web onboarding rides with the generic UI next pass as a separate optional package (maintainer, option 1).
- 2026-10-04: conversation manager, brain, server and init SQL are lean rewrites; the consciousness orchestrator and every helper module are genericized ports (reconnaissance recommendation, accepted).
- 2026-10-04: both Docker Compose and Homebrew paths documented; Compose untested on the development machine.
- 2026-10-04: the partner companion type is included and bounded to companionship and affection with no explicit content.
- 2026-10-04: canonical stores: PostgreSQL for aggregate consciousness snapshots, Mongo for module snapshots.
- 2026-10-04: world model default stays 8/5/4 with the 5-channel encoder (the parameter-sweep harness depends on it); the reference's 16/10/8 conversational model ships as a named preset. Ignition threshold default moves to 0.3 with 0.1 as an experiment preset.
- 2026-10-04: correctness fixes become v0.5.0 defaults with a `legacy_v04` preset that reproduces the paper's numbers (maintainer, option 1). The paper gets a v0.5.0 results section or erratum; MIGRATION_FROM_V04 records every change with before and after numbers.
- 2026-10-04: spec approved; Phase 0 begins on branch `feature/full-architecture-port`.
- 2026-10-05: the attention-agency lever (a stay, drift or turn-back choice delegated to the model) is excluded from the port; the attention schema's release API and the capture journal stay. A shared embedding seam in `mtc/core/embeddings.py` is added so memory, affect estimation and drift sampling use one encoder path.
- 2026-10-18: amendments recorded at the release gate: Settings reads the working directory's `.env` and exports nothing, and a speaker's `token_env` names an exported variable (sections 6.1 and 12); an empty MongoDB auth source means the data database (section 6.1); the PAD model has no regulation strength and the developmental modulator reports the value as skipped (section 5); migrations live under `migrations/postgres/` in three rendered files (sections 4 and 8); the tree gains `mtc/db/`, `mtc/resources/`, the CLI helper modules, the core helper modules and the backup status module (section 4); five keys joined and five defaults changed with their readers (section 6.1); FIRST_RUN, the answers schema and the third-party licence list join the documentation (section 10); the `research` extra replaces `assessment` (section 12); the safety resume route and the `mtc safety` command were added (sections 4 and 9.4); the key table of section 6.1 follows the release: the pronouns live in the persona file, the speakers file is `SPEAKERS_PATH`, and the planned CREATOR_ALIASES, FILE_ACCESS_ROOTS, MODEL_NAME and VLM_MODEL_PATH keys were not added.
