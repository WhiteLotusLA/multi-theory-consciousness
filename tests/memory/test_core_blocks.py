"""Core memory blocks: neutral by default, persona-driven when configured."""

import re

import pytest

from mtc.core.config import Settings
from mtc.memory.core_blocks import (
    CORE_BLOCK_NAMES,
    NEUTRAL_TEMPLATES,
    render_core_blocks,
)

BIG_FIVE = (
    "OPENNESS",
    "CONSCIENTIOUSNESS",
    "EXTRAVERSION",
    "AGREEABLENESS",
    "NEUROTICISM",
)
FIRST_OR_SECOND_PERSON = re.compile(
    r"\b(i|me|my|mine|myself|we|us|our|you|your|yours)\b", re.IGNORECASE
)
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        PROJECT_ROOT=str(tmp_path), SYSTEM_NAME="Unit", CREATOR_NAME="Alice"
    )


def test_neutral_blocks_without_persona(settings):
    blocks = render_core_blocks(settings=settings)
    assert tuple(blocks) == CORE_BLOCK_NAMES == ("identity", "persona", "human")
    assert set(NEUTRAL_TEMPLATES) == set(CORE_BLOCK_NAMES)
    assert all(text.strip() for text in blocks.values())
    assert "Unit" in blocks["identity"]
    assert "Alice" in blocks["human"]
    for value in ("curiosity", "honesty", "growth", "care"):
        assert value in blocks["identity"]


def test_neutral_blocks_have_no_trust_bond_or_date(settings):
    settings.SYSTEM_BIRTH_DATE = "2000-01-01"
    blocks = render_core_blocks(settings=settings)
    seeds = {f"{getattr(settings, key):.2f}" for key in BIG_FIVE}
    for name, text in blocks.items():
        lowered = text.lower()
        assert "trust" not in lowered, name
        assert "bond" not in lowered, name
        assert not ISO_DATE.search(text), name
        assert "2000" not in text, name
        numbers = set(NUMBER.findall(text))
        assert numbers <= seeds, (name, numbers - seeds)
    assert set(NUMBER.findall(blocks["persona"])) == seeds


def test_identity_block_has_self_knowledge_line(settings):
    identity = render_core_blocks(settings=settings)["identity"]
    assert "AI research system" in identity
    assert "does not claim certainty about its own consciousness" in identity


def test_persona_fields_render(settings):
    persona = {
        "name": "Probe",
        "pronouns": "they/them",
        "creator": {"name": "Operator", "address_form": "the operator"},
        "self_description": "Probe studies how attention moves.",
        "communication_style": "plain and brief",
        "core_values": ["patience", "clarity"],
        "personality": {"openness": 0.9, "neuroticism": 0.1},
    }
    blocks = render_core_blocks(persona, settings=settings)
    assert "Probe" in blocks["identity"]
    assert "Unit" not in blocks["identity"]
    assert "they/them" in blocks["identity"]
    assert "Probe studies how attention moves." in blocks["identity"]
    assert "patience" in blocks["identity"] and "clarity" in blocks["identity"]
    assert "curiosity" not in blocks["identity"]
    assert "0.90" in blocks["persona"] and "0.10" in blocks["persona"]
    # Missing traits fall back to the Settings seeds.
    assert f"{settings.CONSCIENTIOUSNESS:.2f}" in blocks["persona"]
    assert "plain and brief" in blocks["persona"]
    assert "Operator" in blocks["human"]
    assert "the operator" in blocks["human"]
    assert "Alice" not in blocks["human"]


def test_core_values_accept_a_comma_string(settings):
    blocks = render_core_blocks({"core_values": "rigor, warmth"}, settings=settings)
    assert "rigor, warmth" in blocks["identity"]


def test_persona_core_memory_section_overrides(settings):
    persona = {
        "name": "Probe",
        "core_memory": {
            "human": "{creator_name} maintains {system_name}.",
            "identity": "   ",
        },
    }
    blocks = render_core_blocks(persona, settings=settings)
    assert blocks["human"] == "Alice maintains Probe."
    # A blank override keeps the rendered block.
    assert "AI research system" in blocks["identity"]


def test_override_with_unknown_placeholder_is_left_verbatim(settings):
    persona = {"core_memory": {"persona": "Seed {unknown} for {system_name}."}}
    blocks = render_core_blocks(persona, settings=settings)
    assert blocks["persona"] == "Seed {unknown} for Unit."


def test_blocks_are_ascii_third_person(settings):
    persona = {"communication_style": "measured", "self_description": "Unit reads."}
    for blocks in (
        render_core_blocks(settings=settings),
        render_core_blocks(persona, settings=settings),
    ):
        for name, text in blocks.items():
            assert text.isascii(), name
            assert not FIRST_OR_SECOND_PERSON.search(text), (name, text)
            assert chr(0x2014) not in text and chr(0x2013) not in text


def test_render_reads_settings_at_call_time(settings):
    first = render_core_blocks(settings=settings)
    settings.SYSTEM_NAME = "Unit Two"
    settings.CREATOR_RELATIONSHIP = "maintainer"
    settings.CORE_VALUES = "rigor"
    settings.OPENNESS = 0.11
    second = render_core_blocks(settings=settings)
    assert "Unit Two" in second["identity"] and "Unit Two" not in first["identity"]
    assert "maintainer" in second["human"]
    assert "rigor" in second["identity"]
    assert "0.11" in second["persona"]
