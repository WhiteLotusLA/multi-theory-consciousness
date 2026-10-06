"""Phase 6 Settings keys for the continuous thought machine."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from mtc.core.config import Settings, csv_list
from mtc.core.env_example import render_env_example

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env file here
    return tmp_path


def test_ctm_defaults(clean_env):
    s = Settings()
    assert s.CTM_ENABLED is True
    assert s.CTM_CONFIG_DIR == "config/ctm"
    assert s.CTM_STORE == "memory"
    assert s.CTM_PROFILE == "default"
    assert csv_list(s.SENSITIVE_MEMORY_PHRASES) == []
    assert s.SENSITIVE_MEMORY_REPLAY_WEIGHT == 1.0


def test_ctm_store_and_profile_choices(clean_env, monkeypatch):
    for choice in ("memory", "mongo"):
        monkeypatch.setenv("MTC_CTM_STORE", choice)
        assert Settings().CTM_STORE == choice
    monkeypatch.setenv("MTC_CTM_STORE", "postgres")
    with pytest.raises(ValidationError):
        Settings()
    monkeypatch.delenv("MTC_CTM_STORE")
    for choice in ("default", "demo"):
        monkeypatch.setenv("MTC_CTM_PROFILE", choice)
        assert Settings().CTM_PROFILE == choice
    monkeypatch.setenv("MTC_CTM_PROFILE", "fast")
    with pytest.raises(ValidationError):
        Settings()


def test_sensitive_weight_never_zero(clean_env, monkeypatch):
    # A weight down-weights a matching memory and never suppresses it.
    for bad in ("0", "-0.2", "1.5", "nan", "inf"):
        monkeypatch.setenv("MTC_SENSITIVE_MEMORY_REPLAY_WEIGHT", bad)
        with pytest.raises(ValidationError, match="SENSITIVE_MEMORY_REPLAY_WEIGHT"):
            Settings()
    for good, expected in (("1", 1.0), ("0.25", 0.25), ("0.01", 0.01)):
        monkeypatch.setenv("MTC_SENSITIVE_MEMORY_REPLAY_WEIGHT", good)
        assert Settings().SENSITIVE_MEMORY_REPLAY_WEIGHT == pytest.approx(expected)


def test_sensitive_phrases_parse_with_csv_list(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_SENSITIVE_MEMORY_PHRASES", " first topic, ,Second Topic ")
    assert csv_list(Settings().SENSITIVE_MEMORY_PHRASES) == [
        "first topic",
        "Second Topic",
    ]


def test_generated_env_example_parses(clean_env):
    text = render_env_example(Settings)
    assert "# === Continuous thought ===" in text
    section = text.split("# === Continuous thought ===", 1)[1].split("# === ", 1)[0]
    keys = [
        line.split("=", 1)[0]
        for line in section.splitlines()
        if line and not line.startswith("#")
    ]
    assert keys == [
        "MTC_CTM_ENABLED",
        "MTC_CTM_CONFIG_DIR",
        "MTC_CTM_STORE",
        "MTC_CTM_PROFILE",
        "MTC_SENSITIVE_MEMORY_PHRASES",
        "MTC_SENSITIVE_MEMORY_REPLAY_WEIGHT",
    ]
    # The rendered defaults load back into a valid Settings object.
    values = {}
    for line in section.splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            values[key[len("MTC_") :]] = value
    loaded = Settings(**values)
    assert loaded.CTM_STORE == "memory"
    assert loaded.SENSITIVE_MEMORY_REPLAY_WEIGHT == 1.0
