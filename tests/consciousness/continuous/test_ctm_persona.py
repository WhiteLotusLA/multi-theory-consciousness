"""CTMPersona: resolution order, name, prompts and honest errors."""

import subprocess
import sys
from pathlib import Path

import pytest

from mtc.consciousness.continuous.persona import (
    BUILTIN_CATEGORY_PROMPTS,
    BUILTIN_PROMPT_HEADERS,
    BUILTIN_SYSTEM_PROMPT,
    CATEGORIES,
    CTMPersona,
    PersonaError,
)
from mtc.core.config import Settings
from mtc.devtools.privacy_audit import load_allowlist, load_denylist, run_audit

ROOT = Path(__file__).resolve().parents[3]
SHIPPED = ROOT / "config" / "ctm"


def _settings(tmp_path, **extra) -> Settings:
    return Settings(PROJECT_ROOT=str(tmp_path), SYSTEM_NAME="Testagent", **extra)


def _persona_file(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_resolution_order(tmp_path):
    cfg = _settings(tmp_path)
    # Nothing: the built-in neutral text.
    persona = CTMPersona.load(tmp_path / "empty", settings=cfg)
    assert persona.source == "builtin" and persona.path is None
    assert persona.system_prompt == BUILTIN_SYSTEM_PROMPT
    # The example file only.
    folder = tmp_path / "cfg"
    example = _persona_file(
        folder, "persona.example.yaml", "system_prompt: 'Example for {name}.'\n"
    )
    persona = CTMPersona.load(folder, settings=cfg)
    assert persona.source == "example" and persona.path == example
    assert persona.render_system() == "Example for Testagent."
    # persona.yaml wins over the example.
    mine = _persona_file(folder, "persona.yaml", "system_prompt: 'Mine for {name}.'\n")
    persona = CTMPersona.load(folder, settings=cfg)
    assert persona.source == "configured" and persona.path == mine
    assert persona.render_system() == "Mine for Testagent."
    # A partial file keeps the built-in text for every key it leaves out.
    assert persona.category_prompts == BUILTIN_CATEGORY_PROMPTS
    assert persona.prompt_headers == BUILTIN_PROMPT_HEADERS


def test_shipped_example_matches_the_builtin_text(tmp_path):
    persona = CTMPersona.load(SHIPPED, settings=_settings(tmp_path))
    assert persona.source == "example"
    assert persona.system_prompt == BUILTIN_SYSTEM_PROMPT
    assert persona.category_prompts == BUILTIN_CATEGORY_PROMPTS
    assert persona.prompt_headers == BUILTIN_PROMPT_HEADERS
    hits = run_audit(
        [SHIPPED / "persona.example.yaml"],
        load_denylist(ROOT / "config" / "privacy" / "denylist.txt"),
        load_allowlist(ROOT / "config" / "privacy" / "allowlist.txt"),
        root=ROOT,
    )
    assert hits == []


def test_name_from_system_name(tmp_path):
    cfg = _settings(tmp_path)
    persona = CTMPersona.load(SHIPPED, settings=cfg)
    assert persona.name == "Testagent"
    assert "Testagent" in persona.render_system()
    assert "{name}" not in persona.render_system()
    # A name in the file wins; an empty name falls back to SYSTEM_NAME.
    _persona_file(tmp_path / "a", "persona.yaml", "name: Operatorbot\n")
    assert CTMPersona.load(tmp_path / "a", settings=cfg).name == "Operatorbot"
    cfg.SYSTEM_NAME = "Another"
    assert CTMPersona.load(SHIPPED, settings=cfg).name == "Another"


def test_no_creator_placeholder(tmp_path):
    cfg = _settings(tmp_path, CREATOR_NAME="Operator")
    persona = CTMPersona.load(SHIPPED, settings=cfg)
    texts = [persona.render_system()] + [
        persona.render_category(
            category,
            {"recent_memory": "a walk", "emotions": {"calm": 0.5}, "recent_topic": "x"},
        )
        for category in CATEGORIES
    ]
    for text in texts:
        assert "Operator" not in text
        for word in ("creator", "{", "}"):
            assert word not in text.lower().replace("{name}", "")
    # A file that tries to name the creator is refused.
    for text in (
        "system_prompt: 'You belong to {creator_name}.'\n",
        "system_prompt: 'Hello {name} and {speaker}.'\n",
        "category_prompts:\n  memory: 'Tell {creator} about it.'\n",
    ):
        folder = tmp_path / str(abs(hash(text)))
        _persona_file(folder, "persona.yaml", text)
        with pytest.raises(PersonaError, match="persona.yaml"):
            CTMPersona.load(folder, settings=cfg)


def test_category_prompts_cover_five(tmp_path):
    persona = CTMPersona.load(SHIPPED, settings=_settings(tmp_path))
    assert set(persona.category_prompts) == set(CATEGORIES)
    assert len(CATEGORIES) == 5
    assert all(text.strip() for text in persona.category_prompts.values())
    # Each category renders from the context it needs.
    context = {"recent_memory": "a quiet walk", "emotions": {"calm": 0.7, "joy": 0.2}}
    memory = persona.render_category("memory", context)
    assert "A recent memory: a quiet walk" in memory
    assert memory.endswith(persona.category_prompts["memory"])
    emotion = persona.render_category("emotion", context)
    assert "Current emotional state: calm (0.70), joy (0.20)" in emotion
    assert persona.render_category("reflection", {}) == (
        persona.category_prompts["reflection"]
    )
    question = persona.render_category("question", {"recent_topic": "gardens"})
    assert "A recent topic: gardens" in question
    # An unfilled optional value is left out, not invented.
    assert "recent topic" not in persona.render_category("question", {}).lower()


def test_missing_required_context_is_named(tmp_path):
    persona = CTMPersona.load(SHIPPED, settings=_settings(tmp_path))
    with pytest.raises(PersonaError, match="recent_memory"):
        persona.render_category("memory", {})
    with pytest.raises(PersonaError, match="recent_memory"):
        persona.render_category("memory", {"recent_memory": "   "})
    with pytest.raises(PersonaError, match="emotions"):
        persona.render_category("emotion", {"emotions": {}})
    with pytest.raises(PersonaError, match="unknown thought category"):
        persona.render_category("dream", {})


def test_emotions_only_shown_when_passed(tmp_path):
    persona = CTMPersona.load(SHIPPED, settings=_settings(tmp_path))
    assert "emotional" not in persona.render_category("reflection", {}).lower()
    shown = persona.render_category("reflection", {"emotions": {"calm": 0.4}})
    assert "calm (0.40)" in shown


def test_non_numeric_emotion_scores_are_skipped_never_zeroed(tmp_path):
    persona = CTMPersona.load(SHIPPED, settings=_settings(tmp_path))
    mixed = {
        "calm": 0.7,
        "unease": "high",
        "wonder": None,
        "joy": 0.2,
        "flag": True,
        "odd": float("nan"),
    }
    text = persona.render_category("emotion", {"emotions": mixed})
    assert "Current emotional state: calm (0.70), joy (0.20)" in text
    for word in ("unease", "wonder", "flag", "odd", "0.00", "1.00", "nan"):
        assert word not in text, word
    # With no numeric score at all, the emotion prompt has nothing to show.
    with pytest.raises(PersonaError, match="emotions"):
        persona.render_category("emotion", {"emotions": {"calm": "high"}})
    # An optional emotional state with no numeric score is left out entirely.
    shown = persona.render_category("reflection", {"emotions": {"calm": "x"}})
    assert shown == persona.category_prompts["reflection"]


def test_clock_line_optional(tmp_path):
    persona = CTMPersona.load(SHIPPED, settings=_settings(tmp_path))
    plain = persona.render_system()
    assert persona.render_system(None) == plain
    assert persona.render_system("   ") == plain
    clocked = persona.render_system("It is a fixed test moment.")
    assert clocked == plain + " It is a fixed test moment."


def test_invalid_files_raise_naming_the_file(tmp_path):
    cfg = _settings(tmp_path)
    for text in (
        "- a list\n",
        "unknown_key: 1\n",
        "system_prompt: ''\n",
        "category_prompts:\n  nonesuch: text\n",
        "category_prompts:\n  memory: ''\n",
        "prompt_headers:\n  unknown: text\n",
        "name: [1]\n",
        "key: [unclosed\n",
    ):
        folder = tmp_path / str(abs(hash(text)))
        path = _persona_file(folder, "persona.yaml", text)
        with pytest.raises(PersonaError) as err:
            CTMPersona.load(folder, settings=cfg)
        assert str(path) in str(err.value), text


def test_config_dir_read_at_call_time(tmp_path):
    cfg = _settings(tmp_path, CTM_CONFIG_DIR="one")
    _persona_file(tmp_path / "one", "persona.yaml", "name: First\n")
    _persona_file(tmp_path / "two", "persona.yaml", "name: Second\n")
    assert CTMPersona.load(settings=cfg).name == "First"
    cfg.CTM_CONFIG_DIR = "two"
    assert CTMPersona.load(settings=cfg).name == "Second"


def test_yaml_imported_lazily():
    code = (
        "import sys\n"
        "import mtc.consciousness.continuous.persona\n"
        "assert 'yaml' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stderr
