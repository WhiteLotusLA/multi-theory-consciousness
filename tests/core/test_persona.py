"""Persona loading: a missing file is neutral, a broken one names its key."""

import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from mtc.core import persona as persona_module
from mtc.core.config import Settings
from mtc.core.persona import PERSONA_KEYS, load_persona, unknown_persona_keys

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fresh_warning_memory(monkeypatch):
    monkeypatch.setattr(persona_module, "_WARNED_MISSING", set())


def _settings(tmp_path: Path, persona_path: str = "config/persona.yaml") -> Settings:
    return Settings(PROJECT_ROOT=str(tmp_path), PERSONA_PATH=persona_path)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_missing_file_returns_none_and_logs_once(tmp_path, caplog):
    settings = _settings(tmp_path)
    with caplog.at_level(logging.WARNING, logger="mtc.core.persona"):
        assert load_persona(settings=settings) is None
        assert load_persona(settings=settings) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "PERSONA_PATH" in warnings[0].getMessage()


def test_reads_configured_path_at_call_time(tmp_path):
    _write(tmp_path / "one.yaml", "name: Unit One\n")
    _write(tmp_path / "two.yaml", "name: Unit Two\n")
    settings = _settings(tmp_path, "one.yaml")
    assert load_persona(settings=settings)["name"] == "Unit One"
    settings.PERSONA_PATH = "two.yaml"
    assert load_persona(settings=settings)["name"] == "Unit Two"
    # An explicit path wins over the setting.
    explicit = load_persona(tmp_path / "one.yaml", settings=settings)
    assert explicit["name"] == "Unit One"


def test_relative_path_resolves_under_project_root(tmp_path):
    _write(tmp_path / "config" / "persona.yaml", "name: Rooted\n")
    assert load_persona(settings=_settings(tmp_path))["name"] == "Rooted"


def test_unparseable_file_raises_naming_the_key(tmp_path):
    _write(tmp_path / "config" / "persona.yaml", "name: [unclosed\n")
    with pytest.raises(ValueError, match="PERSONA_PATH"):
        load_persona(settings=_settings(tmp_path))


def test_non_mapping_file_raises_naming_the_key(tmp_path):
    _write(tmp_path / "config" / "persona.yaml", "- just\n- a list\n")
    with pytest.raises(ValueError, match="PERSONA_PATH"):
        load_persona(settings=_settings(tmp_path))


def test_bad_personality_value_raises_naming_the_field(tmp_path):
    _write(
        tmp_path / "config" / "persona.yaml",
        """
        personality:
          openness: 3.5
        """,
    )
    with pytest.raises(ValueError, match="personality.openness"):
        load_persona(settings=_settings(tmp_path))


def test_empty_file_is_an_empty_persona(tmp_path):
    _write(tmp_path / "config" / "persona.yaml", "")
    assert load_persona(settings=_settings(tmp_path)) == {}


def test_unknown_keys_preserved_and_reported(tmp_path, caplog):
    _write(
        tmp_path / "config" / "persona.yaml",
        """
        name: Unit
        favourite_colour: green
        creator:
          name: Alice
          nickname: Al
        personality:
          openness: 0.4
          whimsy: 0.9
        """,
    )
    with caplog.at_level(logging.WARNING, logger="mtc.core.persona"):
        loaded = load_persona(settings=_settings(tmp_path))
    assert loaded["favourite_colour"] == "green"
    assert loaded["creator"]["nickname"] == "Al"
    assert loaded["personality"]["whimsy"] == 0.9
    unknown = unknown_persona_keys(loaded)
    assert unknown == ["creator.nickname", "favourite_colour", "personality.whimsy"]
    message = " ".join(r.getMessage() for r in caplog.records)
    for key in unknown:
        assert key in message


def test_persona_keys_cover_the_documented_schema():
    assert set(PERSONA_KEYS) == {
        "name",
        "pronouns",
        "companion_type",
        "creator",
        "self_description",
        "communication_style",
        "core_values",
        "personality",
        "core_memory",
        "style_rules",
        "speaker_templates",
    }
    assert PERSONA_KEYS["creator"] == ("name", "address_form")
    assert PERSONA_KEYS["personality"] == (
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    )
    assert PERSONA_KEYS["core_memory"] == ("identity", "persona", "human")


def test_prompt_assembler_persona_keys_known(tmp_path, caplog):
    """The prompt assembler's keys load without an unknown-key warning."""
    assert PERSONA_KEYS["style_rules"] is None
    assert PERSONA_KEYS["speaker_templates"] == ("creator", "known", "unverified")
    _write(
        tmp_path / "config" / "persona.yaml",
        """
        name: Testagent
        style_rules:
          - Answer directly.
          - Say so when something is not remembered.
        speaker_templates:
          creator: Talking with {creator_name}.
          known: Talking with {speaker}, verified.
          unverified: Talking with someone called {speaker}, unverified.
          stranger: not a documented template
        """,
    )
    with caplog.at_level(logging.WARNING, logger="mtc.core.persona"):
        loaded = load_persona(settings=_settings(tmp_path))
    assert loaded["style_rules"] == [
        "Answer directly.",
        "Say so when something is not remembered.",
    ]
    assert unknown_persona_keys(loaded) == ["speaker_templates.stranger"]
    assert "speaker_templates.stranger" in " ".join(
        r.getMessage() for r in caplog.records
    )


def test_yaml_imported_lazily():
    code = (
        "import sys\n"
        "import mtc.core.persona\n"
        "assert 'yaml' not in sys.modules, 'yaml imported at module import'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_messages_name_the_path_relative_to_the_project_root(tmp_path, caplog):
    """A log line or error never carries the absolute path (it holds the
    machine's home directory): a file under PROJECT_ROOT is named relative to
    it, a file elsewhere by its name alone, always with the key."""
    project = tmp_path / "project"
    project.mkdir()
    settings = _settings(project, "config/absent_persona.yaml")
    with caplog.at_level(logging.WARNING, logger="mtc.core.persona"):
        assert load_persona(settings=settings) is None
    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    message = warning.getMessage()
    assert "config/absent_persona.yaml" in message
    assert "PERSONA_PATH" in message
    assert str(tmp_path) not in message

    caplog.clear()
    outside = tmp_path / "elsewhere" / "operator_persona.yaml"
    with caplog.at_level(logging.WARNING, logger="mtc.core.persona"):
        assert load_persona(outside, settings=settings) is None
    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    message = warning.getMessage()
    assert "operator_persona.yaml" in message
    assert str(tmp_path) not in message and "elsewhere" not in message

    _write(project / "config" / "persona.yaml", "name: [unclosed\n")
    with pytest.raises(ValueError) as info:
        load_persona(settings=_settings(project))
    assert "config/persona.yaml" in str(info.value)
    assert str(tmp_path) not in str(info.value)

    caplog.clear()
    _write(project / "config" / "extra.yaml", "name: Unit\nfavourite_colour: green\n")
    with caplog.at_level(logging.WARNING, logger="mtc.core.persona"):
        load_persona(settings=_settings(project, "config/extra.yaml"))
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "config/extra.yaml" in message and str(tmp_path) not in message
