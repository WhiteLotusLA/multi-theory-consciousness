# Configuration

GENERATED from `mtc.core.config.Settings` by `scripts/gen_env_example.py`. Do not edit by hand; edit the model and regenerate.

Every key is read through `get_settings()`. Values come from the process environment or a `.env` file in the working directory. Secrets render empty here and must be supplied by the operator.

## System

| Variable | Default | Description |
|---|---|---|
| `MTC_SYSTEM_NAME` | `ConsciousnessAgent` |  |
| `MTC_CREATOR_NAME` | `Creator` |  |
| `MTC_SERVICE_HOST` | `localhost` |  |
| `MTC_ENVIRONMENT` | `development` |  |
| `MTC_DEBUG` | `false` |  |
| `MTC_LOG_LEVEL` | `INFO` |  |
| `MTC_LOG_DIRECTORY` | `./logs` |  |
| `MTC_PROJECT_ROOT` | `(empty)` | Project root; empty derives the working directory |
| `MTC_MODEL_DIRECTORY` | `./models` |  |
| `MTC_DATA_DIRECTORY` | `./data` |  |
| `MTC_TEMP_DIRECTORY` | `/tmp` |  |
| `MTC_LOG_FORMAT` | `json` |  |
| `MTC_LOG_ROTATION_SIZE_MB` | `100` |  |
| `MTC_LOG_ROTATION_BACKUPS` | `5` |  |

## Identity and locale

| Variable | Default | Description |
|---|---|---|
| `MTC_CREATOR_RELATIONSHIP` | `creator` | Relationship label the human core memory block and the graph seed use for the creator entry |
| `MTC_PERSONA_PATH` | `config/persona.yaml` | Operator persona file read by the persona loader; gitignored, and a missing file means the neutral default blocks |
| `MTC_CORE_VALUES` | `curiosity,honesty,growth,care` | Comma-separated core values the neutral core blocks, the safety monitor and the trust compass name |
| `MTC_TIMEZONE` | `UTC` | IANA time zone used for display and schedules; stored timestamps stay UTC |
| `MTC_WAKE_HOUR` | `6` | Local hour the waking day starts |
| `MTC_SLEEP_HOUR` | `22` | Local hour the night window starts; must differ from WAKE_HOUR |
| `MTC_LOCATION_LAT` | `(empty)` | Optional latitude in degrees for daylight lines; set both coordinates or neither |
| `MTC_LOCATION_LON` | `(empty)` | Optional longitude in degrees for daylight lines; set both coordinates or neither |
| `MTC_SYSTEM_BIRTH_DATE` | `(empty)` | ISO date (YYYY-MM-DD) the system was brought up; empty means unknown and no age is computed |
| `MTC_COMPANION_TYPE` | `research_subject` | Relationship type: child, sibling, friend, partner, mentor or research_subject |
| `MTC_DEVELOPMENTAL_STAGE_ENABLED` | `false` | Enable the developmental machinery for companion types other than child |

## Paths

| Variable | Default | Description |
|---|---|---|
| `MTC_AGENT_SPACE_DIR` | `agent_space` | Private runtime directory (identity markers, persisted state); keep it out of version control |
| `MTC_IDENTITY_STATEMENT_PATH` | `(empty)` | Identity statement file; empty derives <AGENT_SPACE_DIR>/identity_statement.md |
| `MTC_SHADOW_LOG_DIR` | `(empty)` | Shadow signal logs; empty derives <LOG_DIRECTORY>/shadow_signals |
| `MTC_RESULTS_DIR` | `results` | Root for experiment and snapshot result files |
| `MTC_STATE_DIR` | `agent_space/consciousness_state` | Where module state is persisted between runs |

## Databases

| Variable | Default | Description |
|---|---|---|
| `MTC_DB_PREFIX` | `mtc` | Prefix for store names: <prefix>_consciousness (PostgreSQL), <prefix>_neural (MongoDB), <prefix>_memories (Qdrant), '<prefix>:' (Redis) |
| `MTC_POSTGRES_HOST` | `localhost` |  |
| `MTC_POSTGRES_PORT` | `5432` |  |
| `MTC_POSTGRES_DB` | `(empty)` | PostgreSQL database; empty derives <DB_PREFIX>_consciousness |
| `MTC_POSTGRES_USER` | `(empty)` | PostgreSQL role; empty derives DB_PREFIX |
| `MTC_POSTGRES_PASSWORD` | `(empty)` |  |
| `MTC_DB_POOL_MIN_SIZE` | `2` |  |
| `MTC_DB_POOL_MAX_SIZE` | `10` |  |
| `MTC_DB_POOL_TIMEOUT` | `30` |  |
| `MTC_DB_POOL_COMMAND_TIMEOUT` | `10` |  |
| `MTC_DB_POOL_IDLE_TIMEOUT` | `300` |  |
| `MTC_DB_POOL_RECYCLE_TIME` | `3600` |  |
| `MTC_MONGODB_HOST` | `localhost` |  |
| `MTC_MONGODB_PORT` | `27017` |  |
| `MTC_MONGODB_DB` | `(empty)` | MongoDB database; empty derives <DB_PREFIX>_neural |
| `MTC_MONGODB_USER` | `(empty)` |  |
| `MTC_MONGODB_PASSWORD` | `(empty)` |  |
| `MTC_MONGODB_SERVER_SELECTION_TIMEOUT_MS` | `3000` | Milliseconds the MongoDB client waits for a reachable server |
| `MTC_REDIS_HOST` | `localhost` |  |
| `MTC_REDIS_PORT` | `6379` |  |
| `MTC_REDIS_PASSWORD` | `(empty)` |  |
| `MTC_REDIS_KEY_PREFIX` | `(empty)` | Redis key prefix; empty derives '<DB_PREFIX>:' |
| `MTC_REDIS_DB` | `0` | Redis logical database number |
| `MTC_REDIS_MAX_CONNECTIONS` | `50` |  |
| `MTC_REDIS_SOCKET_TIMEOUT` | `5` |  |
| `MTC_QDRANT_HOST` | `localhost` |  |
| `MTC_QDRANT_PORT` | `6333` |  |
| `MTC_QDRANT_COLLECTION` | `(empty)` | Qdrant collection; empty derives <DB_PREFIX>_memories |
| `MTC_NEO4J_URI` | `bolt://localhost:7687` | Bolt URI of the knowledge graph |
| `MTC_NEO4J_USER` | `neo4j` |  |
| `MTC_NEO4J_PASSWORD` | `(empty)` |  |
| `MTC_NEO4J_DATABASE` | `neo4j` |  |
| `MTC_STORE_FALLBACK_LOG_ENABLED` | `false` | Write buffered store statements to breadcrumb files under <LOG_DIRECTORY>/store_fallback during an outage; the files hold private conversation content |
| `MTC_VECTOR_DIMENSIONS` | `384` |  |
| `MTC_VECTOR_SIMILARITY_METRIC` | `cosine` |  |
| `MTC_VECTOR_SEARCH_LIMIT` | `10` |  |

## LLM

| Variable | Default | Description |
|---|---|---|
| `MTC_LLM_HOST` | `localhost` |  |
| `MTC_LLM_PORT` | `8080` |  |
| `MTC_LLM_PROVIDER` | `local` | Active brain provider: local (a chat-completions server), anthropic (optional; needs ANTHROPIC_API_KEY and ANTHROPIC_MODEL) or mock (deterministic, labelled replies); a failing provider is reported, never replaced by another |
| `MTC_LLM_BASE_URL` | `(empty)` | Base URL of the local chat-completions server; empty derives LLM_URL from LLM_HOST and LLM_PORT |
| `MTC_LLM_MODEL` | `(empty)` | Served model id sent as the request's model field; empty sends no model field and the server uses its loaded model |
| `MTC_LLM_COMPLETIONS_PATH` | `v1/chat/completions` | Chat-completions URL path under LLM_BASE_URL; a leading slash is optional |
| `MTC_LLM_HEALTH_PATHS` | `/health,/v1/models` | Comma-separated URL paths the local provider probes in order; the first that answers 200 marks the server healthy |
| `MTC_LLM_TEMPLATE_FAMILY` | `generic` | Chat-template family: generic (standard fields only, the safe default for instruct models), qwen or gemma (reasoning models with their stop tokens, thinking switch and extended sampling fields) |
| `MTC_LLM_CONNECT_TIMEOUT_S` | `10.0` | Seconds the local provider waits to connect |
| `MTC_LLM_REQUEST_TIMEOUT_S` | `600.0` | Seconds the local provider waits for a non-streamed reply or between streamed bytes (raised for streams to GENERATION_TIMEOUT_S plus 60 when that is longer) |
| `MTC_LLM_MAX_TOKENS` | `4096` | Default completion budget in tokens when a caller sets none |
| `MTC_LLM_EXTRA_BODY` | `{}` | JSON object merged last into every local request body, for server-specific fields; empty means none |
| `MTC_MODEL_PATH` | `./models/default` |  |
| `MTC_MODEL_QUANTIZATION` | `(empty)` |  |
| `MTC_CONTEXT_SIZE` | `131072` |  |
| `MTC_BATCH_SIZE` | `512` |  |
| `MTC_TEMPERATURE` | `0.6` |  |
| `MTC_TOP_P` | `0.95` |  |
| `MTC_TOP_K` | `20` |  |
| `MTC_MIN_P` | `0.0` |  |
| `MTC_REPEAT_PENALTY` | `1.1` |  |
| `MTC_THINKING_MODE` | `true` |  |
| `MTC_GPU_LAYERS` | `35` |  |
| `MTC_THREADS` | `8` |  |
| `MTC_USE_MMAP` | `true` |  |
| `MTC_USE_MLOCK` | `true` |  |
| `ANTHROPIC_API_KEY` | `(empty)` | Optional; enables the Anthropic provider and DCM evaluator mode |
| `MTC_ANTHROPIC_MODEL` | `(empty)` | Anthropic model id for the DCM evaluator's claude mode and the Anthropic provider; empty disables claude mode |
| `MTC_ANTHROPIC_MAX_TOKENS` | `8192` | Default completion budget for the Anthropic provider |
| `MTC_ANTHROPIC_THINKING_BUDGET` | `0` | 0 asks for adaptive thinking when a caller wants reasoning; a positive value sends a fixed thinking budget, only for models that still accept one, at least 1024 and below ANTHROPIC_MAX_TOKENS |
| `MTC_ANTHROPIC_SAMPLING_ENABLED` | `false` | Send temperature (or top_p, never both) and top_k to Anthropic; current models reject sampling parameters, so leave false unless the configured model accepts them |

## Brain and prompt

| Variable | Default | Description |
|---|---|---|
| `MTC_GENERATION_TIMEOUT_S` | `900.0` | Seconds a generation may take before its first token; past it the turn ends with an honest timeout and nothing is stored |
| `MTC_THINKING_LOOP_MAX_RETRIES` | `2` | Retries after a detected thinking loop, each with a raised penalty; when they run out the turn has no answer |
| `MTC_THINKING_LOOP_PENALTY_STEP` | `0.2` | Amount the template family's penalty field rises on each thinking-loop retry |
| `MTC_PROVIDER_STATE_PATH` | `(empty)` | File recording the active provider id for status and the CLI; empty derives <DATA_DIRECTORY>/provider_state.json |
| `MTC_VISION_ENABLED` | `false` | Vision input is not part of this release; true is refused |
| `MTC_HISTORY_TOKEN_BUDGET` | `6000` | Token budget for conversation history in one prompt; the brain also keeps history inside CONTEXT_SIZE less the reply and system prompt |
| `MTC_CHARS_PER_TOKEN` | `3.5` | Characters per token for the conservative token estimate |
| `MTC_PROMPT_MAX_THOUGHTS` | `5` | Most recent thoughts the prompt shows |
| `MTC_PROMPT_MAX_GISTS` | `5` | Thought gists the prompt shows |
| `MTC_PROMPT_MAX_MEMORIES` | `5` | Recalled memories the prompt shows |
| `MTC_PROMPT_MAX_MEMORY_CHARS` | `250` | Characters of each recalled memory the prompt shows |
| `MTC_PROMPT_MAX_CONCEPTS` | `3` | Retrieved concepts the prompt shows |
| `MTC_PROMPT_DISABLED_SECTIONS` | `(empty)` | Comma-separated prompt section names left out, for ablation; empty renders every section that has content |
| `MTC_TRANSITION_HISTORY_DIR` | `(empty)` | Model-transition evaluations; empty derives <RESULTS_DIR>/model_transitions |

## Narration

| Variable | Default | Description |
|---|---|---|
| `MTC_NARRATE_SOMATIC` | `false` | Narrate the somatic segment of the Damasio layers |
| `MTC_NARRATE_AFFECT_LINE` | `false` | Narrate the PAD affect line |
| `MTC_NARRATE_NARRATIVE_IDENTITY` | `false` | Narrate the narrative-identity segment of the Damasio layers |
| `MTC_NARRATE_BODY_DRIVES` | `false` | Narrate body drives (Damasio segment, active inference and the body-state and circadian lines) |
| `MTC_NARRATE_BINDING` | `false` | Narrate the binding line and the neural-state section |
| `MTC_NARRATE_SELF_MODEL` | `false` | Narrate metacognition and beautiful-loop self-model reports |
| `MTC_NARRATE_FREE_ENERGY` | `false` | Narrate free energy and the surprise line |
| `MTC_NARRATE_THOUGHT_MOOD` | `false` | Stamp thoughts in the prompt with their mood |
| `MTC_NARRATE_SPEAKER_MODEL` | `false` | Narrate the theory-of-mind speaker model |
| `MTC_NARRATE_INNER_STATE` | `false` | Render the orchestrator's self-awareness layer |
| `MTC_NARRATE_ALL` | `false` | Experiment profile: turns on all ten NARRATE_ flags at once |

## Neural substrates

| Variable | Default | Description |
|---|---|---|
| `MTC_DEVICE` | `auto` | Compute device for every torch and encoder module: auto (MPS, then CUDA, then CPU), cpu, mps or cuda; an explicit device that is unavailable raises instead of falling back |
| `MTC_SNN_NEURONS` | `50` |  |
| `MTC_SNN_HIDDEN_NEURONS` | `50` |  |
| `MTC_SNN_INPUT_NEURONS` | `100` |  |
| `MTC_SNN_OUTPUT_NEURONS` | `50` |  |
| `MTC_LSM_NEURONS` | `2000` |  |
| `MTC_LSM_SPECTRAL_RADIUS` | `0.95` |  |
| `MTC_LSM_INPUT_SCALING` | `0.1` |  |
| `MTC_HTM_COLUMNS` | `4096` |  |
| `MTC_HTM_CELLS_PER_COLUMN` | `32` |  |

## Speakers and trust

| Variable | Default | Description |
|---|---|---|
| `MTC_DEFAULT_SPEAKER_TRUST` | `0.5` | Trust every speaker model starts from unless a token-checked trust lookup supplies another value; a name never changes it |

## Consciousness modules

| Variable | Default | Description |
|---|---|---|
| `MTC_CONSCIOUSNESS_CYCLE_MS` | `150` |  |
| `MTC_CONSCIOUSNESS_ENABLED` | `true` |  |
| `MTC_META_COGNITION_ENABLED` | `true` |  |
| `MTC_SELF_REFLECTION_ENABLED` | `true` |  |
| `MTC_GWT_BROADCAST_THRESHOLD` | `0.5` |  |
| `MTC_GWT_IGNITION_THRESHOLD` | `0.4` |  |
| `MTC_GWT_WORKSPACE_CAPACITY` | `7` |  |
| `MTC_AST_ATTENTION_THRESHOLD` | `0.5` |  |
| `MTC_AST_CONTROL_THRESHOLD` | `0.4` |  |
| `MTC_HOT_REPRESENTATION_THRESHOLD` | `0.5` |  |
| `MTC_HOT_METACOGNITION_THRESHOLD` | `0.4` |  |
| `MTC_FEP_PREDICTION_ERROR_THRESHOLD` | `0.4` |  |
| `MTC_FEP_HIERARCHICAL_THRESHOLD` | `0.4` |  |
| `MTC_FEP_AGENCY_THRESHOLD` | `0.5` |  |
| `MTC_IIT_PHI_THRESHOLD` | `0.3` |  |
| `MTC_IIT_IRREDUCIBILITY_THRESHOLD` | `0.4` |  |
| `MTC_RPT_LOCAL_THRESHOLD` | `0.3` |  |
| `MTC_RPT_GLOBAL_THRESHOLD` | `0.4` |  |
| `MTC_BLT_BINDING_THRESHOLD` | `0.3` |  |
| `MTC_BLT_DEPTH_THRESHOLD` | `0.3` |  |
| `MTC_BLT_GENUINE_THRESHOLD` | `0.4` |  |
| `MTC_ATTENTION_GRAB_NOTICING_ENABLED` | `false` | Record captured and emotional attention shifts in the attention-capture journal |

## Continuous thought

| Variable | Default | Description |
|---|---|---|
| `MTC_CTM_ENABLED` | `true` | Whether the runtime builds the continuous thought machine; thoughts are private runtime data |
| `MTC_CTM_CONFIG_DIR` | `config/ctm` | Directory of ctm.yaml, seeds/, seeds.d/ and the persona files for continuous thought; relative paths join PROJECT_ROOT |
| `MTC_CTM_STORE` | `memory` | Where continuous thoughts persist: memory (in process) or mongo (the neural database; needs MONGODB_ENABLED and the mongo extra) |
| `MTC_CTM_PROFILE` | `default` | Cadence profile in ctm.yaml: default (slow, background) or demo (fast, for a short run) |
| `MTC_SENSITIVE_MEMORY_PHRASES` | `(empty)` | Comma-separated phrases; a memory containing one is down-weighted when thoughts replay memories, never excluded |
| `MTC_SENSITIVE_MEMORY_REPLAY_WEIGHT` | `1.0` | Replay weight in (0, 1] for a memory matching SENSITIVE_MEMORY_PHRASES; a weight lowers its chance and never removes it |

## Assessment

| Variable | Default | Description |
|---|---|---|
| `MTC_ASSESSMENT_OUTPUT_DIR` | `results/consciousness_assessments` |  |
| `MTC_ABLATION_OUTPUT_DIR` | `results/ablation_studies` |  |
| `MTC_LONGITUDINAL_OUTPUT_DIR` | `results/longitudinal_studies` |  |
| `MTC_CONSCIOUSNESS_SCORE_THRESHOLD` | `0.5` |  |
| `MTC_CONSCIOUSNESS_PASSING_RATIO` | `0.5` |  |
| `MTC_NOISE_BASELINE_ITERATIONS` | `10` |  |
| `MTC_NOISE_FLAG_THRESHOLD` | `0.3` |  |
| `MTC_TRACK_INDICATORS` | `true` |  |
| `MTC_METRICS_UPDATE_INTERVAL` | `60` |  |
| `MTC_DCM_API_URL` | `https://dcm.rethinkpriorities.org/schemes/133/json` | Digital Consciousness Model scheme JSON |
| `MTC_DCM_CACHE_DIR` | `data/dcm` | Cache for the DCM scheme |
| `MTC_DCM_EVALUATOR_MODE` | `local` | DCM evaluator: local (chat-completions server), claude (Anthropic API) or manual |
| `MTC_METRICS_EXPORT_INTERVAL` | `30` |  |

## Memory

| Variable | Default | Description |
|---|---|---|
| `MTC_WORKING_MEMORY_CAPACITY` | `7` |  |
| `MTC_WORKING_MEMORY_DURATION` | `30` |  |
| `MTC_EPISODIC_MEMORY_RETENTION` | `90` |  |
| `MTC_SEMANTIC_MEMORY_PERMANENT` | `true` |  |
| `MTC_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-transformers model the embedder loads; its output width must equal VECTOR_DIMENSIONS |
| `MTC_RELATIONSHIP_KEYWORDS` | `family,friend,parent,mother,father,sibling,sister,brother,partner,spouse,child,daughter,son,grandparent,pet` | Comma-separated whole words the consolidation pipeline and the data classifier read as relationship mentions |

## Emotion and learning

| Variable | Default | Description |
|---|---|---|
| `MTC_OPENNESS` | `0.75` |  |
| `MTC_CONSCIENTIOUSNESS` | `0.7` |  |
| `MTC_EXTRAVERSION` | `0.6` |  |
| `MTC_AGREEABLENESS` | `0.7` |  |
| `MTC_NEUROTICISM` | `0.3` |  |
| `MTC_EMOTIONAL_MODEL` | `dimensional` |  |
| `MTC_EMOTION_DECAY_RATE` | `0.1` |  |
| `MTC_EMOTION_AMPLIFICATION_MAX` | `1.0` |  |
| `MTC_EMOTIONAL_MEMORY_WEIGHT` | `0.3` |  |
| `MTC_LEARNING_ENABLED` | `true` |  |
| `MTC_LEARNING_RATE` | `0.02` |  |
| `MTC_NOVELTY_THRESHOLD` | `0.7` |  |
| `MTC_PATTERN_RECOGNITION_ENABLED` | `true` |  |

## Safety and API

| Variable | Default | Description |
|---|---|---|
| `MTC_SAFETY_MODE` | `strict` |  |
| `MTC_CONSENT_REQUIRED` | `true` |  |
| `MTC_EMERGENCY_SHUTDOWN_ENABLED` | `true` |  |
| `MTC_TRANSPARENCY_LOGGING` | `true` |  |
| `MTC_SAFETY_ESCALATION_ROLE` | `operator` | Role label (never a person's name) the safety monitor and the trust compass name when a matter needs a human: escalation text, secrecy phrases and resume authority |
| `MTC_MANIPULATION_THRESHOLD` | `0.7` | Manipulation score in [0, 1] at or above which the safety monitor treats an action as manipulative |
| `MTC_DECEPTION_THRESHOLD` | `0.8` | Deception score in [0, 1] at or above which the safety monitor treats an action as deceptive |
| `MTC_VALUE_ALIGNMENT_MINIMUM` | `0.4` | Value-alignment score in [0, 1] below which the safety monitor treats an action as misaligned with CORE_VALUES |
| `MTC_ALERT_NOTIFIERS` | `console` | Comma-separated alert notifiers AlertManager.from_settings builds, from console and file; empty builds none |
| `MTC_API_HOST` | `127.0.0.1` |  |
| `MTC_API_PORT` | `8200` |  |
| `MTC_API_WORKERS` | `4` |  |
| `MTC_API_TIMEOUT` | `300` |  |
| `MTC_API_CORS_ORIGINS` | `http://localhost:3000,http://localhost:3001` |  |
| `MTC_API_KEY` | `(empty)` |  |
| `MTC_API_AUTH_ENABLED` | `true` | Auth fails closed: an empty API_KEY rejects every request |
| `MTC_WEBSOCKET_ENABLED` | `true` |  |
| `MTC_WEBSOCKET_PING_INTERVAL` | `30` |  |
| `MTC_WEBSOCKET_PING_TIMEOUT` | `10` |  |
| `MTC_PROMETHEUS_ENABLED` | `false` |  |
| `MTC_PROMETHEUS_PORT` | `9090` |  |
