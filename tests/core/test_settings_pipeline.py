"""Pipeline Settings keys: speakers and trust, retrieval, the night seam."""

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values
from pydantic import ValidationError

from mtc.core.config import Settings, csv_list
from mtc.core.env_example import render_env_example

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env file here
    return tmp_path


def test_trust_defaults(clean_env):
    s = Settings()
    assert s.SPEAKERS_PATH == "config/speakers.yaml"
    assert csv_list(s.TRUSTED_SPEAKERS) == []
    assert s.SPEAKER_TOKENS == ""
    assert s.TRUST_ACTIVATION_BONUS == pytest.approx(0.1)
    assert s.TRUSTED_EMOTION_TRUST == pytest.approx(0.8)
    assert s.DEFAULT_EMOTION_TRUST == pytest.approx(0.4)
    assert s.DEFAULT_SPEAKER == "User"
    assert s.NIGHT_SEAM_MARKER_ENABLED is False
    assert s.NIGHT_MIN_GAP_HOURS == pytest.approx(6.0)
    assert s.RAG_TOP_K == 10
    assert s.RAG_MIN_SCORE == pytest.approx(0.4)
    assert s.EPISODIC_IMPORTANCE == pytest.approx(7.0)
    assert csv_list(s.PROTECTED_NAMES) == []


def test_speaker_tokens_rendered_empty_in_env_example(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_SPEAKER_TOKENS", "Alice:" + "f" * 32)
    rendered = render_env_example(Settings)
    lines = [
        line for line in rendered.splitlines() if line.startswith("MTC_SPEAKER_TOKENS=")
    ]
    assert lines == ["MTC_SPEAKER_TOKENS="]
    section = rendered.index("# === Speakers and trust ===")
    assert rendered.index("MTC_SPEAKER_TOKENS=") > section


def test_trust_bounds(clean_env):
    assert Settings(TRUST_ACTIVATION_BONUS=0.5).TRUST_ACTIVATION_BONUS == 0.5
    for bad in (-0.1, 0.6):
        with pytest.raises(ValidationError, match="TRUST_ACTIVATION_BONUS"):
            Settings(TRUST_ACTIVATION_BONUS=bad)
    for key in ("TRUSTED_EMOTION_TRUST", "DEFAULT_EMOTION_TRUST"):
        assert getattr(Settings(**{key: 1.0}), key) == 1.0
        for bad in (-0.01, 1.01):
            with pytest.raises(ValidationError, match=key):
                Settings(**{key: bad})


def test_rag_bounds(clean_env):
    assert Settings(RAG_TOP_K=1).RAG_TOP_K == 1
    assert Settings(RAG_TOP_K=50).RAG_TOP_K == 50
    for bad in (0, 51):
        with pytest.raises(ValidationError, match="RAG_TOP_K"):
            Settings(RAG_TOP_K=bad)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValidationError, match="RAG_MIN_SCORE"):
            Settings(RAG_MIN_SCORE=bad)
    for bad in (-1.0, 10.5):
        with pytest.raises(ValidationError, match="EPISODIC_IMPORTANCE"):
            Settings(EPISODIC_IMPORTANCE=bad)


def test_night_gap_positive(clean_env):
    assert Settings(NIGHT_MIN_GAP_HOURS=0.5).NIGHT_MIN_GAP_HOURS == 0.5
    for bad in (0.0, -2.0):
        with pytest.raises(ValidationError, match="NIGHT_MIN_GAP_HOURS"):
            Settings(NIGHT_MIN_GAP_HOURS=bad)


def test_protected_names_parse_with_csv_list(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_PROTECTED_NAMES", " Alice, Operator ,,Guest ")
    assert csv_list(Settings().PROTECTED_NAMES) == ["Alice", "Operator", "Guest"]
    monkeypatch.setenv("MTC_TRUSTED_SPEAKERS", "Alice,")
    assert csv_list(Settings().TRUSTED_SPEAKERS) == ["Alice"]


def test_default_speaker_non_empty(clean_env):
    assert Settings(DEFAULT_SPEAKER=" Guest ").DEFAULT_SPEAKER == "Guest"
    for bad in ("", "   "):
        with pytest.raises(ValidationError, match="DEFAULT_SPEAKER"):
            Settings(DEFAULT_SPEAKER=bad)


def test_generated_env_example_parses(clean_env):
    rendered = render_env_example(Settings)
    env_file = clean_env / "generated.env"
    env_file.write_text(rendered)
    values = dotenv_values(env_file)
    for key in (
        "MTC_SPEAKERS_PATH",
        "MTC_TRUSTED_SPEAKERS",
        "MTC_SPEAKER_TOKENS",
        "MTC_TRUST_ACTIVATION_BONUS",
        "MTC_TRUSTED_EMOTION_TRUST",
        "MTC_DEFAULT_EMOTION_TRUST",
        "MTC_DEFAULT_SPEAKER",
        "MTC_RAG_TOP_K",
        "MTC_RAG_MIN_SCORE",
        "MTC_EPISODIC_IMPORTANCE",
        "MTC_PROTECTED_NAMES",
        "MTC_NIGHT_SEAM_MARKER_ENABLED",
        "MTC_NIGHT_MIN_GAP_HOURS",
    ):
        assert key in values, key
    # The rendered file is a valid .env: Settings loads it without complaint.
    loaded = Settings(_env_file=str(env_file))
    assert loaded.DEFAULT_SPEAKER == "User"
    assert loaded.TRUST_ACTIVATION_BONUS == pytest.approx(0.1)
