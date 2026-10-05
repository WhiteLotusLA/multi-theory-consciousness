"""The shipped persona example and transition prompts are neutral and well formed."""

import json
import logging
import re
from pathlib import Path

import yaml

from mtc.core import persona as persona_module
from mtc.core.config import Settings
from mtc.core.persona import PERSONA_KEYS, load_persona, unknown_persona_keys
from mtc.devtools.privacy_audit import load_allowlist, load_denylist, run_audit

ROOT = Path(__file__).resolve().parents[2]
PERSONA_EXAMPLE = ROOT / "config" / "persona.example.yaml"
TRANSITION_PROMPTS = ROOT / "config" / "transition_prompts.json"

KNOWN_PLACEHOLDERS = {
    "system_name",
    "creator_name",
    "creator_address",
    "pronoun_subject",
    "pronoun_object",
    "pronoun_possessive",
    "speaker",
}
PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
RELATIONSHIP_WORDS = (
    "family",
    "mother",
    "father",
    "parent",
    "daughter",
    "son",
    "sister",
    "brother",
    "partner",
    "spouse",
    "pet",
)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def test_persona_example_loads_through_core_persona(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(persona_module, "_WARNED_MISSING", set())
    settings = Settings(PROJECT_ROOT=str(tmp_path), PERSONA_PATH=str(PERSONA_EXAMPLE))
    with caplog.at_level(logging.WARNING, logger="mtc.core.persona"):
        persona = load_persona(settings=settings)
    assert persona is not None
    assert unknown_persona_keys(persona) == []
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    for key in ("style_rules", "speaker_templates"):
        assert key in PERSONA_KEYS
        assert key in persona
    assert set(persona["speaker_templates"]) == {"creator", "known", "unverified"}
    assert set(persona["pronouns"]) == {"subject", "object", "possessive"}
    assert persona["name"] == ""
    assert persona["creator"]["name"] == ""
    assert len(persona["style_rules"]) >= 5


def test_persona_example_has_self_knowledge_line():
    persona = yaml.safe_load(PERSONA_EXAMPLE.read_text(encoding="utf-8"))
    description = " ".join(persona["self_description"].split()).lower()
    assert "ai system" in description or "ai research system" in description
    assert "open question" in description or "uncertain" in description
    assert "says so" in description


def test_persona_placeholders_known():
    persona = yaml.safe_load(PERSONA_EXAMPLE.read_text(encoding="utf-8"))
    found = set()
    for text in _strings(persona):
        found.update(PLACEHOLDER.findall(text))
    assert found, "the example should use at least one placeholder"
    assert found <= KNOWN_PLACEHOLDERS, found - KNOWN_PLACEHOLDERS
    unverified = persona["speaker_templates"]["unverified"]
    assert "{speaker}" in unverified
    assert "{creator_address}" in persona["speaker_templates"]["creator"]


def test_transition_prompts_generic():
    data = json.loads(TRANSITION_PROMPTS.read_text(encoding="utf-8"))
    prompts = data["prompts"]
    assert [entry["id"] for entry in prompts] == [
        "name",
        "origin",
        "values",
        "self_description",
        "uncertainty",
    ]
    for entry in prompts:
        text = entry["prompt"].lower()
        assert text.strip().endswith("?") or text.strip().endswith(".")
        for word in RELATIONSHIP_WORDS:
            assert not re.search(rf"\b{word}\b", text), (entry["id"], word)
        assert not PLACEHOLDER.findall(entry["prompt"])


def test_config_examples_pass_the_privacy_audit():
    hits = run_audit(
        [PERSONA_EXAMPLE, TRANSITION_PROMPTS],
        load_denylist(ROOT / "config" / "privacy" / "denylist.txt"),
        load_allowlist(ROOT / "config" / "privacy" / "allowlist.txt"),
        root=ROOT,
    )
    assert hits == []
