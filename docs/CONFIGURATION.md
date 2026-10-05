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
| `MTC_REDIS_HOST` | `localhost` |  |
| `MTC_REDIS_PORT` | `6379` |  |
| `MTC_REDIS_PASSWORD` | `(empty)` |  |
| `MTC_REDIS_KEY_PREFIX` | `(empty)` | Redis key prefix; empty derives '<DB_PREFIX>:' |
| `MTC_REDIS_MAX_CONNECTIONS` | `50` |  |
| `MTC_REDIS_SOCKET_TIMEOUT` | `5` |  |
| `MTC_QDRANT_HOST` | `localhost` |  |
| `MTC_QDRANT_PORT` | `6333` |  |
| `MTC_QDRANT_COLLECTION` | `(empty)` | Qdrant collection; empty derives <DB_PREFIX>_memories |
| `MTC_NEO4J_URI` | `bolt://localhost:7687` | Bolt URI of the knowledge graph |
| `MTC_NEO4J_USER` | `neo4j` |  |
| `MTC_NEO4J_PASSWORD` | `(empty)` |  |
| `MTC_NEO4J_DATABASE` | `neo4j` |  |
| `MTC_VECTOR_DIMENSIONS` | `384` |  |
| `MTC_VECTOR_SIMILARITY_METRIC` | `cosine` |  |
| `MTC_VECTOR_SEARCH_LIMIT` | `10` |  |

## LLM

| Variable | Default | Description |
|---|---|---|
| `MTC_LLM_HOST` | `localhost` |  |
| `MTC_LLM_PORT` | `8080` |  |
| `MTC_MODEL_PATH` | `./models/default` |  |
| `MTC_MODEL_TYPE` | `default` |  |
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

## Neural substrates

| Variable | Default | Description |
|---|---|---|
| `MTC_SNN_NEURONS` | `50` |  |
| `MTC_SNN_HIDDEN_NEURONS` | `50` |  |
| `MTC_SNN_INPUT_NEURONS` | `100` |  |
| `MTC_SNN_OUTPUT_NEURONS` | `50` |  |
| `MTC_LSM_NEURONS` | `2000` |  |
| `MTC_LSM_SPECTRAL_RADIUS` | `0.95` |  |
| `MTC_LSM_INPUT_SCALING` | `0.1` |  |
| `MTC_HTM_COLUMNS` | `4096` |  |
| `MTC_HTM_CELLS_PER_COLUMN` | `32` |  |

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
