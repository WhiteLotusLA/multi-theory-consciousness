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
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field


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

    # ========================================================================
    # SYSTEM CONFIGURATION
    # ========================================================================

    SERVICE_HOST: str = "localhost"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_DIRECTORY: str = "./logs"

    PROJECT_ROOT: str = Field(default_factory=lambda: os.getcwd())
    MODEL_DIRECTORY: str = Field(
        default_factory=lambda: os.environ.get("MODEL_DIRECTORY", "./models")
    )
    DATA_DIRECTORY: str = "./data"
    TEMP_DIRECTORY: str = "/tmp"

    # ========================================================================
    # DATABASE CONFIGURATION
    # ========================================================================

    # PostgreSQL (Primary Memory Database)
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "consciousness_db"
    POSTGRES_USER: str = "consciousness"
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
    MONGODB_DB: str = "consciousness_memories"
    MONGODB_USER: str = ""
    MONGODB_PASSWORD: str = ""

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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def REDIS_URL(self) -> str:
        """Construct Redis connection URL."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}"

    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: int = 5

    # Qdrant (Vector Database for Semantic Search)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "consciousness_memories"

    # Neo4j (Knowledge Graph)
    NEO4J_HOST: str = "localhost"
    NEO4J_PORT: int = 7474
    NEO4J_BOLT_PORT: int = 7687
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""
    NEO4J_DATABASE: str = "neo4j"

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

    MODEL_PATH: str = Field(
        default_factory=lambda: os.environ.get("MODEL_PATH", "./models/default")
    )
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

    # ========================================================================
    # MEMORY CONFIGURATION
    # ========================================================================

    # Memory Tiers
    WORKING_MEMORY_CAPACITY: int = 7  # 7+-2 items
    WORKING_MEMORY_DURATION: int = 30  # seconds
    EPISODIC_MEMORY_RETENTION: int = 90  # days
    SEMANTIC_MEMORY_PERMANENT: bool = True

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
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_TIMEOUT: int = 300
    API_CORS_ORIGINS: str = "http://localhost:3000,http://localhost:3001"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def API_CORS_ORIGINS_LIST(self) -> List[str]:
        """Get CORS origins as list."""
        return [origin.strip() for origin in self.API_CORS_ORIGINS.split(",")]

    API_KEY: str = ""
    API_AUTH_ENABLED: bool = False

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

    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # ========================================================================
    # FEATURE FLAGS
    # ========================================================================

    ENABLE_CONSCIOUSNESS_METRICS: bool = True
    ENABLE_SAFETY_MONITOR: bool = True
    ENABLE_VECTOR_SEARCH: bool = False
    ENABLE_GRAPH_DATABASE: bool = False
    ENABLE_MULTI_MODAL: bool = False
    ENABLE_DREAM_PROCESSING: bool = False

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
    print(f"  System Name: {settings.SYSTEM_NAME}")
    print(f"  Creator: {settings.CREATOR_NAME}")
    print(f"  Environment: {settings.ENVIRONMENT}")
    print(f"  PostgreSQL URL: {settings.POSTGRES_URL}")
    print(f"  LLM URL: {settings.LLM_URL}")
    print(f"  SNN Neurons: {settings.SNN_NEURONS}")
    print(f"  LSM Neurons: {settings.LSM_NEURONS}")
    print(f"  HTM Columns: {settings.HTM_COLUMNS}")
    print("Configuration test complete!")
