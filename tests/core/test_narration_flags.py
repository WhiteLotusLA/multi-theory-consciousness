"""Narration flags: one reader for the NARRATE_* family, every flag off by default."""

import os
from dataclasses import fields

import pytest

from mtc.core import narration
from mtc.core.config import Settings
from mtc.core.narration import NARRATION_FLAG_KEYS, NarrationFlags

FLAG_FIELDS = (
    "somatic",
    "affect_line",
    "narrative_identity",
    "body_drives",
    "binding",
    "self_model",
    "free_energy",
    "thought_mood",
    "speaker_model",
    "inner_state",
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env file here
    return tmp_path


def test_flag_keys_match_fields_and_settings():
    assert len(NARRATION_FLAG_KEYS) == 10
    assert tuple(f.name for f in fields(NarrationFlags)) == FLAG_FIELDS
    assert NARRATION_FLAG_KEYS == tuple(f"NARRATE_{n.upper()}" for n in FLAG_FIELDS)
    for key in NARRATION_FLAG_KEYS + ("NARRATE_ALL",):
        assert key in Settings.model_fields, key


def test_all_flags_default_off(clean_env):
    assert NarrationFlags() == NarrationFlags.from_settings(Settings())
    assert not any(NarrationFlags.from_settings(Settings()).as_dict().values())
    assert not any(NarrationFlags().as_dict().values())


def test_each_key_turns_on_its_own_flag(clean_env, monkeypatch):
    for name, key in zip(FLAG_FIELDS, NARRATION_FLAG_KEYS):
        monkeypatch.setenv(f"MTC_{key}", "true")
        flags = NarrationFlags.from_settings(Settings()).as_dict()
        assert [n for n, on in flags.items() if on] == [name], key
        monkeypatch.delenv(f"MTC_{key}")


def test_narrate_all_turns_every_flag_on(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_NARRATE_ALL", "true")
    flags = NarrationFlags.from_settings(Settings())
    assert all(flags.as_dict().values())
    assert flags == NarrationFlags.all_on()


def test_flags_read_at_call_time(clean_env, monkeypatch):
    current = {"settings": Settings()}
    monkeypatch.setattr(narration, "get_settings", lambda: current["settings"])
    assert not any(NarrationFlags.from_settings().as_dict().values())
    current["settings"] = Settings(NARRATE_THOUGHT_MOOD=True)
    assert NarrationFlags.from_settings().thought_mood is True
    current["settings"] = Settings(NARRATE_ALL=True)
    assert NarrationFlags.from_settings() == NarrationFlags.all_on()


def test_all_on_profile():
    on = NarrationFlags.all_on()
    assert on.as_dict() == {name: True for name in FLAG_FIELDS}
    assert list(on.as_dict()) == list(FLAG_FIELDS)


def test_effect_map_names_every_flag():
    doc = NarrationFlags.__doc__ or ""
    for name, key in zip(FLAG_FIELDS, NARRATION_FLAG_KEYS):
        assert name in doc, name
        assert key in (narration.__doc__ or "") + doc, key
    assert "NARRATE_ALL" in (narration.__doc__ or "") + doc
