"""Phase 4 Settings keys: locale, identity paths, development, results."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from mtc.core.config import COMPANION_TYPES, Settings
from mtc.core.env_example import render_env_example


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env file here
    return tmp_path


def test_locale_defaults(clean_env):
    s = Settings()
    assert s.TIMEZONE == "UTC"
    assert s.WAKE_HOUR == 6
    assert s.SLEEP_HOUR == 22
    assert s.LOCATION_LAT is None
    assert s.LOCATION_LON is None


def test_empty_env_values_mean_unset(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_LOCATION_LAT", "")
    monkeypatch.setenv("MTC_LOCATION_LON", "")
    s = Settings()
    assert s.LOCATION_LAT is None
    assert s.LOCATION_LON is None


def test_both_coordinates_accepted(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_LOCATION_LAT", "10.5")
    monkeypatch.setenv("MTC_LOCATION_LON", "-20.25")
    s = Settings()
    assert s.LOCATION_LAT == 10.5
    assert s.LOCATION_LON == -20.25


@pytest.mark.parametrize("key", ["MTC_LOCATION_LAT", "MTC_LOCATION_LON"])
def test_one_coordinate_alone_is_rejected(clean_env, monkeypatch, key):
    monkeypatch.setenv(key, "10.5")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    "lat,lon", [("90.5", "0"), ("-91", "0"), ("0", "180.5"), ("0", "-181")]
)
def test_coordinates_bounded(clean_env, monkeypatch, lat, lon):
    monkeypatch.setenv("MTC_LOCATION_LAT", lat)
    monkeypatch.setenv("MTC_LOCATION_LON", lon)
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("value", ["Not/AZone", "", "../etc", "UTC+99"])
def test_invalid_timezone_rejected(clean_env, monkeypatch, value):
    monkeypatch.setenv("MTC_TIMEZONE", value)
    with pytest.raises(ValidationError):
        Settings()


def test_valid_timezone_accepted(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_TIMEZONE", "Asia/Tokyo")
    assert Settings().TIMEZONE == "Asia/Tokyo"


def test_wake_and_sleep_must_differ(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_WAKE_HOUR", "7")
    monkeypatch.setenv("MTC_SLEEP_HOUR", "7")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("key,value", [("WAKE_HOUR", "24"), ("SLEEP_HOUR", "-1")])
def test_hours_bounded(clean_env, monkeypatch, key, value):
    monkeypatch.setenv(f"MTC_{key}", value)
    with pytest.raises(ValidationError):
        Settings()


def test_birth_date_has_no_fallback(clean_env):
    assert Settings().SYSTEM_BIRTH_DATE == ""


def test_birth_date_must_be_iso(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_SYSTEM_BIRTH_DATE", "01/02/2000")
    with pytest.raises(ValidationError):
        Settings()
    monkeypatch.setenv("MTC_SYSTEM_BIRTH_DATE", "2000-01-01")
    assert Settings().SYSTEM_BIRTH_DATE == "2000-01-01"


def test_companion_type_defaults_and_rejects_unknown(clean_env, monkeypatch):
    assert Settings().COMPANION_TYPE == "research_subject"
    assert Settings().DEVELOPMENTAL_STAGE_ENABLED is False
    assert COMPANION_TYPES == (
        "child",
        "sibling",
        "friend",
        "partner",
        "mentor",
        "research_subject",
    )
    for kind in COMPANION_TYPES:
        monkeypatch.setenv("MTC_COMPANION_TYPE", kind)
        assert Settings().COMPANION_TYPE == kind
    monkeypatch.setenv("MTC_COMPANION_TYPE", "stranger")
    with pytest.raises(ValidationError):
        Settings()


def test_derived_paths_follow_agent_space_and_log_dir(clean_env, monkeypatch):
    s = Settings()
    assert s.AGENT_SPACE_DIR == "agent_space"
    assert s.IDENTITY_STATEMENT_PATH == "agent_space/identity_statement.md"
    assert s.SHADOW_LOG_DIR == "./logs/shadow_signals"
    assert s.RESULTS_DIR == "results"
    monkeypatch.setenv("MTC_AGENT_SPACE_DIR", "space")
    monkeypatch.setenv("MTC_LOG_DIRECTORY", "var/log")
    s = Settings()
    assert s.IDENTITY_STATEMENT_PATH == "space/identity_statement.md"
    assert s.SHADOW_LOG_DIR == "var/log/shadow_signals"
    monkeypatch.setenv("MTC_IDENTITY_STATEMENT_PATH", "elsewhere/statement.md")
    monkeypatch.setenv("MTC_SHADOW_LOG_DIR", "elsewhere/shadow")
    s = Settings()
    assert s.IDENTITY_STATEMENT_PATH == "elsewhere/statement.md"
    assert s.SHADOW_LOG_DIR == "elsewhere/shadow"


def test_generated_env_example_parses(clean_env):
    (clean_env / ".env").write_text(render_env_example(Settings), encoding="utf-8")
    s = Settings()
    assert s.LOCATION_LAT is None
    assert s.LOCATION_LON is None
    assert s.TIMEZONE == "UTC"
    assert s.SYSTEM_BIRTH_DATE == ""
    assert s.IDENTITY_STATEMENT_PATH.endswith("identity_statement.md")
    assert Path(s.SHADOW_LOG_DIR).name == "shadow_signals"


def test_new_keys_do_not_land_under_other():
    rendered = render_env_example(Settings)
    assert "# === Other ===" not in rendered


def test_attention_noticing_off_by_default(clean_env, monkeypatch):
    assert Settings().ATTENTION_GRAB_NOTICING_ENABLED is False
    monkeypatch.setenv("MTC_ATTENTION_GRAB_NOTICING_ENABLED", "true")
    assert Settings().ATTENTION_GRAB_NOTICING_ENABLED is True


def test_default_speaker_trust_neutral_and_bounded(clean_env, monkeypatch):
    assert Settings().DEFAULT_SPEAKER_TRUST == 0.5
    for value in ("0", "1", "0.25"):
        monkeypatch.setenv("MTC_DEFAULT_SPEAKER_TRUST", value)
        assert Settings().DEFAULT_SPEAKER_TRUST == float(value)
    for value in ("-0.01", "1.01", "nan"):
        monkeypatch.setenv("MTC_DEFAULT_SPEAKER_TRUST", value)
        with pytest.raises(ValidationError):
            Settings()
