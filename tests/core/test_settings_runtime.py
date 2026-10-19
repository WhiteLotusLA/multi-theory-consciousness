"""The runtime Settings keys: every store off by default, bounded intervals,
a bare preset name, the removed API keys and a generated example that parses."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from mtc.core import config as config_module
from mtc.core.config import LOOPBACK_HOSTS, Settings
from mtc.core.env_example import render_env_example

ROOT = Path(__file__).resolve().parents[2]

STORE_ENABLE_KEYS = (
    "POSTGRES_ENABLED",
    "MONGODB_ENABLED",
    "REDIS_ENABLED",
    "QDRANT_ENABLED",
    "NEO4J_ENABLED",
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_store_enable_defaults_all_off(clean_env):
    settings = Settings()
    for key in STORE_ENABLE_KEYS:
        assert getattr(settings, key) is False, key
        assert Settings.model_fields[key].annotation is bool


def test_pipeline_preset_name_validated(clean_env):
    assert Settings().PIPELINE_PRESET == ""
    assert Settings(PIPELINE_PRESET="legacy_v04").PIPELINE_PRESET == "legacy_v04"
    assert Settings(PIPELINE_PRESET=" demo-1 ").PIPELINE_PRESET == "demo-1"
    for bad in ("presets/legacy_v04", "../legacy", "legacy.toml", "a b", "/abs"):
        with pytest.raises(ValidationError):
            Settings(PIPELINE_PRESET=bad)


def test_sampling_intervals_non_negative(clean_env):
    settings = Settings()
    assert settings.PHI_IDLE_SAMPLE_S == 0.0
    assert settings.SELF_ASSESSMENT_INTERVAL_S == 0.0
    assert settings.SLEEP_SCHEDULER_ENABLED is True
    assert settings.AUTONOMOUS_LIFE_ENABLED is False
    assert Settings(PHI_IDLE_SAMPLE_S=30).PHI_IDLE_SAMPLE_S == 30.0
    for key in ("PHI_IDLE_SAMPLE_S", "SELF_ASSESSMENT_INTERVAL_S"):
        with pytest.raises(ValidationError):
            Settings(**{key: -1.0})
        with pytest.raises(ValidationError):
            Settings(**{key: float("nan")})


def test_generation_bounds_positive(clean_env):
    settings = Settings()
    assert settings.GENERATION_STALL_TIMEOUT_S == 300.0
    assert settings.THINKING_HEARTBEAT_S == 15.0
    for key in ("GENERATION_STALL_TIMEOUT_S", "THINKING_HEARTBEAT_S"):
        for bad in (0.0, -5.0, float("inf")):
            with pytest.raises(ValidationError):
                Settings(**{key: bad})


def test_removed_api_keys(clean_env):
    fields = Settings.model_fields
    for gone in ("API_WORKERS", "API_TIMEOUT", "MODEL_TYPE"):
        assert gone not in fields, gone
    rendered = render_env_example(Settings)
    assert "MTC_API_WORKERS" not in rendered
    assert "MTC_API_TIMEOUT" not in rendered
    doc = (ROOT / "docs" / "CONFIGURATION.md").read_text(encoding="utf-8")
    assert "API_WORKERS" not in doc
    assert "API_TIMEOUT" not in doc


def test_loopback_hosts_constant(clean_env):
    assert LOOPBACK_HOSTS == ("127.0.0.1", "::1", "localhost")
    assert isinstance(config_module.LOOPBACK_HOSTS, tuple)
    assert Settings().API_HOST in LOOPBACK_HOSTS


def test_generated_env_example_parses(clean_env, tmp_path):
    rendered = render_env_example(Settings)
    for key in STORE_ENABLE_KEYS:
        assert f"MTC_{key}=false" in rendered, key
    assert "MTC_PIPELINE_PRESET=" in rendered
    assert "MTC_THINKING_HEARTBEAT_S=15.0" in rendered
    env_file = tmp_path / "generated.env"
    env_file.write_text(rendered, encoding="utf-8")
    loaded = Settings(_env_file=str(env_file))
    defaults = Settings(_env_file=None)
    for key in STORE_ENABLE_KEYS + (
        "PIPELINE_PRESET",
        "SLEEP_SCHEDULER_ENABLED",
        "AUTONOMOUS_LIFE_ENABLED",
        "PHI_IDLE_SAMPLE_S",
        "SELF_ASSESSMENT_INTERVAL_S",
        "GENERATION_STALL_TIMEOUT_S",
        "THINKING_HEARTBEAT_S",
    ):
        assert getattr(loaded, key) == getattr(defaults, key), key
