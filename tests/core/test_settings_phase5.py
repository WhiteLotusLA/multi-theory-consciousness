"""Phase 5 Settings keys: device, embedding model, store, identity and safety keys."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from mtc.core.config import Settings, csv_list
from mtc.core.env_example import render_env_example

ROOT = Path(__file__).resolve().parents[2]

DOCUMENTED_KEYWORDS = [
    "family",
    "friend",
    "parent",
    "mother",
    "father",
    "sibling",
    "sister",
    "brother",
    "partner",
    "spouse",
    "child",
    "daughter",
    "son",
    "grandparent",
    "pet",
]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env file here
    return tmp_path


def test_device_defaults_to_auto(clean_env, monkeypatch):
    assert Settings().DEVICE == "auto"
    for choice in ("auto", "cpu", "mps", "cuda"):
        monkeypatch.setenv("MTC_DEVICE", choice)
        assert Settings().DEVICE == choice


def test_invalid_device_rejected(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_DEVICE", "tpu")
    with pytest.raises(ValidationError):
        Settings()


def test_embedding_model_default(clean_env):
    assert Settings().EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_redis_db_non_negative(clean_env, monkeypatch):
    assert Settings().REDIS_DB == 0
    monkeypatch.setenv("MTC_REDIS_DB", "3")
    assert Settings().REDIS_DB == 3
    monkeypatch.setenv("MTC_REDIS_DB", "-1")
    with pytest.raises(ValidationError):
        Settings()


def test_mongo_timeout_positive(clean_env, monkeypatch):
    assert Settings().MONGODB_SERVER_SELECTION_TIMEOUT_MS == 3000
    for bad in ("0", "-5"):
        monkeypatch.setenv("MTC_MONGODB_SERVER_SELECTION_TIMEOUT_MS", bad)
        with pytest.raises(ValidationError):
            Settings()


def test_fallback_log_off_by_default(clean_env, monkeypatch):
    assert Settings().STORE_FALLBACK_LOG_ENABLED is False
    monkeypatch.setenv("MTC_STORE_FALLBACK_LOG_ENABLED", "true")
    assert Settings().STORE_FALLBACK_LOG_ENABLED is True


def test_persona_path_default_is_gitignored(clean_env):
    default = Settings().PERSONA_PATH
    assert default == "config/persona.yaml"
    lines = (ROOT / ".gitignore").read_text().splitlines()
    assert default in [line.strip() for line in lines]


def test_core_values_parse_to_list(clean_env, monkeypatch):
    assert csv_list(Settings().CORE_VALUES) == [
        "curiosity",
        "honesty",
        "growth",
        "care",
    ]
    monkeypatch.setenv("MTC_CORE_VALUES", " a , b,,c ")
    assert csv_list(Settings().CORE_VALUES) == ["a", "b", "c"]


def test_creator_relationship_default(clean_env):
    assert Settings().CREATOR_RELATIONSHIP == "creator"


def test_relationship_keywords_are_generic(clean_env):
    keywords = csv_list(Settings().RELATIONSHIP_KEYWORDS)
    assert keywords == DOCUMENTED_KEYWORDS
    for word in keywords:
        assert word.isalpha() and word == word.lower()


def test_csv_list_strips_and_drops_empties():
    assert csv_list("a, b ,, c,") == ["a", "b", "c"]
    assert csv_list("") == []
    assert csv_list(" , ,") == []
    assert csv_list("single") == ["single"]


def test_generated_env_example_parses(clean_env):
    (clean_env / ".env").write_text(render_env_example(Settings), encoding="utf-8")
    s = Settings()
    assert s.DEVICE == "auto"
    assert s.REDIS_DB == 0
    assert s.PERSONA_PATH == "config/persona.yaml"
    assert csv_list(s.RELATIONSHIP_KEYWORDS) == DOCUMENTED_KEYWORDS
    assert csv_list(s.CORE_VALUES) == ["curiosity", "honesty", "growth", "care"]


def test_new_keys_land_in_their_sections():
    rendered = render_env_example(Settings)
    assert "# === Other ===" not in rendered
    sections = {}
    current = ""
    for line in rendered.splitlines():
        if line.startswith("# === "):
            current = line[6:-4]
        elif line and not line.startswith("#"):
            sections[line.split("=", 1)[0]] = current
    assert sections["MTC_DEVICE"] == "Neural substrates"
    assert sections["MTC_EMBEDDING_MODEL"] == "Memory"
    assert sections["MTC_RELATIONSHIP_KEYWORDS"] == "Memory"
    assert sections["MTC_STORE_FALLBACK_LOG_ENABLED"] == "Databases"
    assert sections["MTC_PERSONA_PATH"] == "Identity and locale"
    assert sections["MTC_CORE_VALUES"] == "Identity and locale"
    assert sections["MTC_CREATOR_RELATIONSHIP"] == "Identity and locale"


# Sub-workflow 3: the safety section (escalation role, thresholds, notifiers).


def test_escalation_role_default_operator(clean_env, monkeypatch):
    assert Settings().SAFETY_ESCALATION_ROLE == "operator"
    monkeypatch.setenv("MTC_SAFETY_ESCALATION_ROLE", "guardian")
    assert Settings().SAFETY_ESCALATION_ROLE == "guardian"
    monkeypatch.setenv("MTC_SAFETY_ESCALATION_ROLE", "  ")
    with pytest.raises(ValidationError):
        Settings()


def test_safety_thresholds_bounded(clean_env, monkeypatch):
    s = Settings()
    assert s.MANIPULATION_THRESHOLD == 0.7
    assert s.DECEPTION_THRESHOLD == 0.8
    assert s.VALUE_ALIGNMENT_MINIMUM == 0.4
    for key in (
        "MTC_MANIPULATION_THRESHOLD",
        "MTC_DECEPTION_THRESHOLD",
        "MTC_VALUE_ALIGNMENT_MINIMUM",
    ):
        for edge in ("0", "1", "0.5"):
            monkeypatch.setenv(key, edge)
            Settings()
        for bad in ("-0.01", "1.01", "nan", "inf"):
            monkeypatch.setenv(key, bad)
            with pytest.raises(ValidationError):
                Settings()
        monkeypatch.delenv(key)


def test_alert_notifiers_parse_and_reject_unknown(clean_env, monkeypatch):
    assert csv_list(Settings().ALERT_NOTIFIERS) == ["console"]
    monkeypatch.setenv("MTC_ALERT_NOTIFIERS", " console , file ")
    assert csv_list(Settings().ALERT_NOTIFIERS) == ["console", "file"]
    monkeypatch.setenv("MTC_ALERT_NOTIFIERS", "")
    assert csv_list(Settings().ALERT_NOTIFIERS) == []
    for bad in ("console,push", "sms", "Console"):
        monkeypatch.setenv("MTC_ALERT_NOTIFIERS", bad)
        with pytest.raises(ValidationError):
            Settings()


def test_safety_keys_land_in_the_safety_section():
    rendered = render_env_example(Settings)
    sections = {}
    current = ""
    for line in rendered.splitlines():
        if line.startswith("# === "):
            current = line[6:-4]
        elif line and not line.startswith("#"):
            sections[line.split("=", 1)[0]] = current
    for key in (
        "MTC_SAFETY_ESCALATION_ROLE",
        "MTC_MANIPULATION_THRESHOLD",
        "MTC_DECEPTION_THRESHOLD",
        "MTC_VALUE_ALIGNMENT_MINIMUM",
        "MTC_ALERT_NOTIFIERS",
    ):
        assert sections[key] == "Safety and API", key
