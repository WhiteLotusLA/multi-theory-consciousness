"""
Multi-Theory Consciousness Framework - Configuration
=====================================================

Clean, generic configuration for the MTC framework.
Loads settings from environment variables and .env files.

Provides type-safe access to all framework settings including:
- Database connections
- Neural network parameters
- Consciousness module settings
- Assessment parameters
- LLM connection settings
"""

import os
from datetime import date
from typing import Any, List, Literal, Optional
from zoneinfo import ZoneInfo

from pydantic import (
    AliasChoices,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Companion relationship types the onboarding schema and the developmental
# gate recognise.
COMPANION_TYPES = (
    "child",
    "sibling",
    "friend",
    "partner",
    "mentor",
    "research_subject",
)


def csv_list(value: str) -> List[str]:
    """Split a comma-separated setting into trimmed, non-empty items."""
    return [item.strip() for item in str(value).split(",") if item.strip()]


class Settings(BaseSettings):
    """
    Multi-Theory Consciousness Framework configuration settings.

    All settings are loaded from .env file and environment variables.
    """

    # ========================================================================
    # SYSTEM IDENTITY
    # ========================================================================

    SYSTEM_NAME: str = "ConsciousnessAgent"
    CREATOR_NAME: str = "Creator"
    CREATOR_RELATIONSHIP: str = Field(
        "creator",
        description="Relationship label the human core memory block and the "
        "graph seed use for the creator entry",
    )
    PERSONA_PATH: str = Field(
        "config/persona.yaml",
        description="Operator persona file read by the persona loader; "
        "gitignored, and a missing file means the neutral default blocks",
    )
    CORE_VALUES: str = Field(
        "curiosity,honesty,growth,care",
        description="Comma-separated core values the neutral core blocks, "
        "the safety monitor and the trust compass name",
    )

    # ========================================================================
    # IDENTITY AND LOCALE
    # ========================================================================

    TIMEZONE: str = Field(
        "UTC",
        description="IANA time zone used for display and schedules; stored "
        "timestamps stay UTC",
    )
    WAKE_HOUR: int = Field(
        6, ge=0, le=23, description="Local hour the waking day starts"
    )
    SLEEP_HOUR: int = Field(
        22,
        ge=0,
        le=23,
        description="Local hour the night window starts; must differ from WAKE_HOUR",
    )
    LOCATION_LAT: Optional[float] = Field(
        None,
        description="Optional latitude in degrees for daylight lines; set both "
        "coordinates or neither",
    )
    LOCATION_LON: Optional[float] = Field(
        None,
        description="Optional longitude in degrees for daylight lines; set both "
        "coordinates or neither",
    )
    SYSTEM_BIRTH_DATE: str = Field(
        "",
        description="ISO date (YYYY-MM-DD) the system was brought up; empty means "
        "unknown and no age is computed",
    )
    COMPANION_TYPE: str = Field(
        "research_subject",
        description="Relationship type: child, sibling, friend, partner, mentor "
        "or research_subject",
    )
    DEVELOPMENTAL_STAGE_ENABLED: bool = Field(
        False,
        description="Enable the developmental machinery for companion types "
        "other than child",
    )

    # ========================================================================
    # SPEAKERS AND TRUST
    # ========================================================================

    DEFAULT_SPEAKER_TRUST: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Trust every speaker model starts from unless a token-checked "
        "trust lookup supplies another value; a name never changes it",
    )

    # ========================================================================
    # SYSTEM CONFIGURATION
    # ========================================================================

    SERVICE_HOST: str = "localhost"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_DIRECTORY: str = "./logs"

    PROJECT_ROOT: str = Field(
        "", description="Project root; empty derives the working directory"
    )
    MODEL_DIRECTORY: str = "./models"
    DATA_DIRECTORY: str = "./data"
    TEMP_DIRECTORY: str = "/tmp"

    # ========================================================================
    # PATHS
    # ========================================================================

    AGENT_SPACE_DIR: str = Field(
        "agent_space",
        description="Private runtime directory (identity markers, persisted "
        "state); keep it out of version control",
    )
    IDENTITY_STATEMENT_PATH: str = Field(
        "",
        description="Identity statement file; empty derives "
        "<AGENT_SPACE_DIR>/identity_statement.md",
    )
    SHADOW_LOG_DIR: str = Field(
        "",
        description="Shadow signal logs; empty derives <LOG_DIRECTORY>/shadow_signals",
    )
    RESULTS_DIR: str = Field(
        "results", description="Root for experiment and snapshot result files"
    )

    # ========================================================================
    # DATABASE CONFIGURATION
    # ========================================================================

    # Store-name prefix shared by every database below
    DB_PREFIX: str = Field(
        "mtc",
        min_length=1,
        description="Prefix for store names: <prefix>_consciousness (PostgreSQL), "
        "<prefix>_neural (MongoDB), <prefix>_memories (Qdrant), '<prefix>:' (Redis)",
    )

    # PostgreSQL (Primary Memory Database)
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = Field(
        "", description="PostgreSQL database; empty derives <DB_PREFIX>_consciousness"
    )
    POSTGRES_USER: str = Field(
        "", description="PostgreSQL role; empty derives DB_PREFIX"
    )
    POSTGRES_PASSWORD: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def POSTGRES_URL(self) -> str:
        """Construct PostgreSQL connection URL."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Connection Pool Settings
    DB_POOL_MIN_SIZE: int = 2
    DB_POOL_MAX_SIZE: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_COMMAND_TIMEOUT: int = 10
    DB_POOL_IDLE_TIMEOUT: int = 300
    DB_POOL_RECYCLE_TIME: int = 3600

    # MongoDB (Document Storage & Unstructured Data)
    MONGODB_HOST: str = "localhost"
    MONGODB_PORT: int = 27017
    MONGODB_DB: str = Field(
        "", description="MongoDB database; empty derives <DB_PREFIX>_neural"
    )
    MONGODB_USER: str = ""
    MONGODB_PASSWORD: str = ""
    MONGODB_SERVER_SELECTION_TIMEOUT_MS: int = Field(
        3000,
        gt=0,
        description="Milliseconds the MongoDB client waits for a reachable server",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def MONGODB_URL(self) -> str:
        """Construct MongoDB connection URL."""
        if self.MONGODB_USER and self.MONGODB_PASSWORD:
            return (
                f"mongodb://{self.MONGODB_USER}:{self.MONGODB_PASSWORD}"
                f"@{self.MONGODB_HOST}:{self.MONGODB_PORT}/{self.MONGODB_DB}"
            )
        return f"mongodb://{self.MONGODB_HOST}:{self.MONGODB_PORT}/{self.MONGODB_DB}"

    # Redis (Real-time State & Working Memory)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_KEY_PREFIX: str = Field(
        "", description="Redis key prefix; empty derives '<DB_PREFIX>:'"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def REDIS_URL(self) -> str:
        """Construct Redis connection URL."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}"

    REDIS_DB: int = Field(0, ge=0, description="Redis logical database number")
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: int = 5

    # Qdrant (Vector Database for Semantic Search)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = Field(
        "", description="Qdrant collection; empty derives <DB_PREFIX>_memories"
    )

    # Neo4j (Knowledge Graph)
    NEO4J_URI: str = Field(
        "bolt://localhost:7687", description="Bolt URI of the knowledge graph"
    )
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""
    NEO4J_DATABASE: str = "neo4j"

    STORE_FALLBACK_LOG_ENABLED: bool = Field(
        False,
        description="Write buffered store statements to breadcrumb files under "
        "<LOG_DIRECTORY>/store_fallback during an outage; the files hold "
        "private conversation content",
    )

    # ========================================================================
    # LLM CONFIGURATION
    # ========================================================================

    LLM_HOST: str = "localhost"
    LLM_PORT: int = 8080

    @computed_field  # type: ignore[prop-decorator]
    @property
    def LLM_URL(self) -> str:
        """Construct LLM server URL."""
        return f"http://{self.LLM_HOST}:{self.LLM_PORT}"

    MODEL_PATH: str = "./models/default"
    MODEL_TYPE: str = "default"
    MODEL_QUANTIZATION: str = ""

    # Model Parameters
    CONTEXT_SIZE: int = 131072
    BATCH_SIZE: int = 512
    TEMPERATURE: float = 0.6
    TOP_P: float = 0.95
    TOP_K: int = 20
    MIN_P: float = 0.0
    REPEAT_PENALTY: float = 1.1
    THINKING_MODE: bool = True
    GPU_LAYERS: int = 35
    THREADS: int = 8
    USE_MMAP: bool = True
    USE_MLOCK: bool = True

    # ========================================================================
    # NEURAL NETWORK PARAMETERS
    # ========================================================================

    DEVICE: Literal["auto", "cpu", "mps", "cuda"] = Field(
        "auto",
        description="Compute device for every torch and encoder module: auto "
        "(MPS, then CUDA, then CPU), cpu, mps or cuda; an explicit device that "
        "is unavailable raises instead of falling back",
    )

    # Spiking Neural Network (SNN)
    SNN_NEURONS: int = 50
    SNN_HIDDEN_NEURONS: int = 50
    SNN_INPUT_NEURONS: int = 100
    SNN_OUTPUT_NEURONS: int = 50

    # Liquid State Machine (LSM)
    LSM_NEURONS: int = 2000
    LSM_SPECTRAL_RADIUS: float = 0.95
    LSM_INPUT_SCALING: float = 0.1

    # Hierarchical Temporal Memory (HTM)
    HTM_COLUMNS: int = 4096
    HTM_CELLS_PER_COLUMN: int = 32

    # ========================================================================
    # CONSCIOUSNESS MODULE SETTINGS
    # ========================================================================

    # Consciousness Loop Settings
    CONSCIOUSNESS_CYCLE_MS: int = 150  # ~150ms per cycle (like human)
    CONSCIOUSNESS_ENABLED: bool = True
    META_COGNITION_ENABLED: bool = True
    SELF_REFLECTION_ENABLED: bool = True

    # Global Workspace Theory (GWT)
    GWT_BROADCAST_THRESHOLD: float = 0.5
    GWT_IGNITION_THRESHOLD: float = 0.4
    GWT_WORKSPACE_CAPACITY: int = 7

    # Attention Schema Theory (AST)
    AST_ATTENTION_THRESHOLD: float = 0.5
    AST_CONTROL_THRESHOLD: float = 0.4

    # Higher-Order Thought (HOT)
    HOT_REPRESENTATION_THRESHOLD: float = 0.5
    HOT_METACOGNITION_THRESHOLD: float = 0.4

    # Free Energy Principle (FEP)
    FEP_PREDICTION_ERROR_THRESHOLD: float = 0.4
    FEP_HIERARCHICAL_THRESHOLD: float = 0.4
    FEP_AGENCY_THRESHOLD: float = 0.5

    # Integrated Information Theory (IIT)
    IIT_PHI_THRESHOLD: float = 0.3
    IIT_IRREDUCIBILITY_THRESHOLD: float = 0.4

    # Recurrent Processing Theory (RPT)
    RPT_LOCAL_THRESHOLD: float = 0.3
    RPT_GLOBAL_THRESHOLD: float = 0.4

    # Beautiful Loop Theory (BLT)
    BLT_BINDING_THRESHOLD: float = 0.3
    BLT_DEPTH_THRESHOLD: float = 0.3
    BLT_GENUINE_THRESHOLD: float = 0.4

    # Attention-capture journal (mtc.consciousness.attention_observer)
    ATTENTION_GRAB_NOTICING_ENABLED: bool = Field(
        False,
        description="Record captured and emotional attention shifts in the "
        "attention-capture journal",
    )

    # Module state persistence (mtc.consciousness.state_persistence)
    STATE_DIR: str = Field(
        "agent_space/consciousness_state",
        description="Where module state is persisted between runs",
    )

    # ========================================================================
    # ASSESSMENT PARAMETERS
    # ========================================================================

    ASSESSMENT_OUTPUT_DIR: str = "results/consciousness_assessments"
    ABLATION_OUTPUT_DIR: str = "results/ablation_studies"
    LONGITUDINAL_OUTPUT_DIR: str = "results/longitudinal_studies"

    # Assessment thresholds
    CONSCIOUSNESS_SCORE_THRESHOLD: float = 0.5
    CONSCIOUSNESS_PASSING_RATIO: float = 0.5  # At least half indicators must pass

    # Noise baseline settings
    NOISE_BASELINE_ITERATIONS: int = 10
    NOISE_FLAG_THRESHOLD: float = 0.3

    # Track consciousness indicators
    TRACK_INDICATORS: bool = True
    METRICS_UPDATE_INTERVAL: int = 60

    # Digital Consciousness Model (DCM) benchmark
    DCM_API_URL: str = Field(
        "https://dcm.rethinkpriorities.org/schemes/133/json",
        description="Digital Consciousness Model scheme JSON",
    )
    DCM_CACHE_DIR: str = Field("data/dcm", description="Cache for the DCM scheme")
    DCM_EVALUATOR_MODE: str = Field(
        "local",
        description="DCM evaluator: local (chat-completions server), "
        "claude (Anthropic API) or manual",
    )

    # ========================================================================
    # MEMORY CONFIGURATION
    # ========================================================================

    # Memory Tiers
    WORKING_MEMORY_CAPACITY: int = 7  # 7+-2 items
    WORKING_MEMORY_DURATION: int = 30  # seconds
    EPISODIC_MEMORY_RETENTION: int = 90  # days
    SEMANTIC_MEMORY_PERMANENT: bool = True

    EMBEDDING_MODEL: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformers model the embedder loads; its "
        "output width must equal VECTOR_DIMENSIONS",
    )
    RELATIONSHIP_KEYWORDS: str = Field(
        "family,friend,parent,mother,father,sibling,sister,brother,partner,"
        "spouse,child,daughter,son,grandparent,pet",
        description="Comma-separated whole words the consolidation pipeline and "
        "the data classifier read as relationship mentions",
    )

    # Vector Search
    VECTOR_DIMENSIONS: int = 384
    VECTOR_SIMILARITY_METRIC: str = "cosine"
    VECTOR_SEARCH_LIMIT: int = 10

    # ========================================================================
    # EMOTIONAL PROCESSING CONFIGURATION
    # ========================================================================

    # Big Five Personality Traits (defaults)
    OPENNESS: float = 0.75
    CONSCIENTIOUSNESS: float = 0.70
    EXTRAVERSION: float = 0.60
    AGREEABLENESS: float = 0.70
    NEUROTICISM: float = 0.30

    # Emotional Processing
    EMOTIONAL_MODEL: str = "dimensional"  # dimensional | categorical
    EMOTION_DECAY_RATE: float = 0.1
    EMOTION_AMPLIFICATION_MAX: float = 1.0
    EMOTIONAL_MEMORY_WEIGHT: float = 0.3

    # ========================================================================
    # LEARNING CONFIGURATION
    # ========================================================================

    LEARNING_ENABLED: bool = True
    LEARNING_RATE: float = 0.02
    NOVELTY_THRESHOLD: float = 0.7
    PATTERN_RECOGNITION_ENABLED: bool = True

    # ========================================================================
    # SAFETY & ETHICS CONFIGURATION
    # ========================================================================

    SAFETY_MODE: str = "strict"  # strict | moderate | permissive
    CONSENT_REQUIRED: bool = True
    EMERGENCY_SHUTDOWN_ENABLED: bool = True
    TRANSPARENCY_LOGGING: bool = True

    # ========================================================================
    # API SERVER CONFIGURATION
    # ========================================================================

    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8200
    API_WORKERS: int = 4
    API_TIMEOUT: int = 300
    API_CORS_ORIGINS: str = "http://localhost:3000,http://localhost:3001"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def API_CORS_ORIGINS_LIST(self) -> List[str]:
        """Get CORS origins as list."""
        return [origin.strip() for origin in self.API_CORS_ORIGINS.split(",")]

    API_KEY: str = ""
    API_AUTH_ENABLED: bool = Field(
        True, description="Auth fails closed: an empty API_KEY rejects every request"
    )

    # WebSocket Settings
    WEBSOCKET_ENABLED: bool = True
    WEBSOCKET_PING_INTERVAL: int = 30
    WEBSOCKET_PING_TIMEOUT: int = 10

    # ========================================================================
    # MONITORING & OBSERVABILITY
    # ========================================================================

    PROMETHEUS_ENABLED: bool = False
    PROMETHEUS_PORT: int = 9090
    METRICS_EXPORT_INTERVAL: int = 30

    LOG_FORMAT: str = "json"
    LOG_ROTATION_SIZE_MB: int = 100
    LOG_ROTATION_BACKUPS: int = 5

    # ========================================================================
    # EXTERNAL API KEYS
    # ========================================================================

    ANTHROPIC_API_KEY: str = Field(
        "",
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "MTC_ANTHROPIC_API_KEY"),
        description="Optional; enables the Anthropic provider and DCM evaluator mode",
    )
    ANTHROPIC_MODEL: str = Field(
        "",
        description="Anthropic model id for the DCM evaluator's claude mode and "
        "the Anthropic provider; empty disables claude mode",
    )

    # ========================================================================
    # Pydantic Configuration
    # ========================================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore extra fields from .env
        env_prefix="MTC_",  # All env vars prefixed with MTC_
    )

    # ========================================================================
    # Derived values
    # ========================================================================

    @field_validator("LOCATION_LAT", "LOCATION_LON", mode="before")
    @classmethod
    def _empty_coordinate_is_unset(cls, value: Any) -> Any:
        """An empty env value (the generated example renders None so) is unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("TIMEZONE")
    @classmethod
    def _timezone_is_iana(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (KeyError, ValueError, OSError) as exc:
            raise ValueError(f"TIMEZONE {value!r} is not an IANA zone name") from exc
        return value

    @field_validator("SYSTEM_BIRTH_DATE")
    @classmethod
    def _birth_date_is_iso(cls, value: str) -> str:
        if value:
            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(
                    f"SYSTEM_BIRTH_DATE {value!r} is not an ISO date (YYYY-MM-DD)"
                ) from exc
        return value

    @field_validator("COMPANION_TYPE")
    @classmethod
    def _companion_type_is_known(cls, value: str) -> str:
        if value not in COMPANION_TYPES:
            raise ValueError(f"COMPANION_TYPE must be one of {COMPANION_TYPES}")
        return value

    @field_validator("LOCATION_LAT")
    @classmethod
    def _latitude_in_range(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not -90.0 <= value <= 90.0:
            raise ValueError("LOCATION_LAT must be within [-90, 90]")
        return value

    @field_validator("LOCATION_LON")
    @classmethod
    def _longitude_in_range(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not -180.0 <= value <= 180.0:
            raise ValueError("LOCATION_LON must be within [-180, 180]")
        return value

    @model_validator(mode="after")
    def _check_locale_pairs(self) -> "Settings":
        if (self.LOCATION_LAT is None) != (self.LOCATION_LON is None):
            raise ValueError(
                "LOCATION_LAT and LOCATION_LON must be set together or not at all"
            )
        if self.WAKE_HOUR == self.SLEEP_HOUR:
            raise ValueError("WAKE_HOUR and SLEEP_HOUR must differ")
        return self

    @model_validator(mode="after")
    def _derive_store_names(self) -> "Settings":
        prefix = self.DB_PREFIX
        if not self.POSTGRES_DB:
            self.POSTGRES_DB = f"{prefix}_consciousness"
        if not self.POSTGRES_USER:
            self.POSTGRES_USER = prefix
        if not self.MONGODB_DB:
            self.MONGODB_DB = f"{prefix}_neural"
        if not self.QDRANT_COLLECTION:
            self.QDRANT_COLLECTION = f"{prefix}_memories"
        if not self.REDIS_KEY_PREFIX:
            self.REDIS_KEY_PREFIX = f"{prefix}:"
        if not self.PROJECT_ROOT:
            self.PROJECT_ROOT = os.getcwd()
        if not self.IDENTITY_STATEMENT_PATH:
            self.IDENTITY_STATEMENT_PATH = os.path.join(
                self.AGENT_SPACE_DIR, "identity_statement.md"
            )
        if not self.SHADOW_LOG_DIR:
            self.SHADOW_LOG_DIR = os.path.join(self.LOG_DIRECTORY, "shadow_signals")
        return self


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the singleton Settings instance.

    Returns:
        Initialized Settings instance
    """
    global _settings

    if _settings is None:
        _settings = Settings()

    return _settings


if __name__ == "__main__":
    # Test configuration loading
    settings = get_settings()
    print("Multi-Theory Consciousness Framework Configuration:")
    print(f" System Name: {settings.SYSTEM_NAME}")
    print(f" Creator: {settings.CREATOR_NAME}")
    print(f" Environment: {settings.ENVIRONMENT}")
    print(f" DB Prefix: {settings.DB_PREFIX}")
    print(f" PostgreSQL DB: {settings.POSTGRES_DB}")
    print(f" MongoDB DB: {settings.MONGODB_DB}")
    print(f" Qdrant Collection: {settings.QDRANT_COLLECTION}")
    print(f" Neo4j URI: {settings.NEO4J_URI}")
    print(f" LLM URL: {settings.LLM_URL}")
    print(f" API Port: {settings.API_PORT}")
    print(f" SNN Neurons: {settings.SNN_NEURONS}")
    print(f" LSM Neurons: {settings.LSM_NEURONS}")
    print(f" HTM Columns: {settings.HTM_COLUMNS}")
    print("Configuration test complete!")
