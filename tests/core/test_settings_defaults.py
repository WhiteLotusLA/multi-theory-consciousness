"""Settings must be real: consumed defaults, derived store names, aliases."""

import os

import pytest

from mtc.core.config import Settings


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env file here
    return tmp_path


def test_settings_construct_without_env_file(clean_env):
    s = Settings()
    assert s.DB_PREFIX == "mtc"
    assert s.POSTGRES_DB == "mtc_consciousness"
    assert s.POSTGRES_USER == "mtc"
    assert s.MONGODB_DB == "mtc_neural"
    assert s.QDRANT_COLLECTION == "mtc_memories"
    assert s.REDIS_KEY_PREFIX == "mtc:"
    assert s.NEO4J_URI == "bolt://localhost:7687"
    assert s.NEO4J_DATABASE == "neo4j"


def test_prefix_changes_every_derived_name(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_DB_PREFIX", "lab")
    s = Settings()
    assert s.POSTGRES_DB == "lab_consciousness"
    assert s.MONGODB_DB == "lab_neural"
    assert s.QDRANT_COLLECTION == "lab_memories"
    assert s.REDIS_KEY_PREFIX == "lab:"


def test_explicit_store_name_wins_over_prefix(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_POSTGRES_DB", "research")
    s = Settings()
    assert s.POSTGRES_DB == "research"
    assert s.MONGODB_DB == "mtc_neural"


def test_api_defaults_fail_closed(clean_env):
    s = Settings()
    assert s.API_PORT == 8200
    assert s.API_AUTH_ENABLED is True


def test_anthropic_key_readable_unprefixed(clean_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-value")
    assert Settings().ANTHROPIC_API_KEY == "test-value"


def test_anthropic_key_readable_prefixed(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_ANTHROPIC_API_KEY", "test-value-2")
    assert Settings().ANTHROPIC_API_KEY == "test-value-2"


def test_removed_keys_are_gone(clean_env):
    fields = Settings.model_fields
    # Only the supported providers carry a provider key: a key for any other
    # provider returning to Settings fails here.
    provider_keys = {name for name in fields if name.endswith("_API_KEY")}
    assert provider_keys == {"ANTHROPIC_API_KEY"}
    for gone in [
        "NEO4J_HOST",
        "NEO4J_PORT",
        "NEO4J_BOLT_PORT",
        "ENABLE_CONSCIOUSNESS_METRICS",
        "ENABLE_SAFETY_MONITOR",
        "ENABLE_VECTOR_SEARCH",
        "ENABLE_GRAPH_DATABASE",
        "ENABLE_MULTI_MODAL",
        "ENABLE_DREAM_PROCESSING",
    ]:
        assert gone not in fields, gone


def test_reset_settings_rebuilds_the_singleton(clean_env, monkeypatch):
    from mtc.core import config

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("MTC_SYSTEM_NAME", "Testagent")
    first = config.get_settings()
    assert first.SYSTEM_NAME == "Testagent"
    monkeypatch.setenv("MTC_SYSTEM_NAME", "Testagent2")
    # The instance is built once per process: a changed environment is not
    # seen until the singleton is reset.
    assert config.get_settings() is first
    assert config.get_settings().SYSTEM_NAME == "Testagent"
    assert config.reset_settings() is None
    rebuilt = config.get_settings()
    assert rebuilt is not first
    assert rebuilt.SYSTEM_NAME == "Testagent2"
    assert config.get_settings() is rebuilt


def test_cors_origin_list_drops_empty_entries(clean_env):
    assert Settings(API_CORS_ORIGINS="").API_CORS_ORIGINS_LIST == []
    assert Settings(API_CORS_ORIGINS="a, ,b,").API_CORS_ORIGINS_LIST == ["a", "b"]
    default = Settings()
    assert default.API_CORS_ORIGINS_LIST == [
        item.strip() for item in default.API_CORS_ORIGINS.split(",")
    ]


def test_dcm_settings_exist(clean_env):
    s = Settings()
    assert s.DCM_EVALUATOR_MODE == "local"
    assert s.DCM_CACHE_DIR == "data/dcm_cache"
    assert s.DCM_API_URL.startswith("https://")
