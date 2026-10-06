"""docs/ONBOARDING.md's type table equals the shipped type defaults, and the
document states the bounds every template carries, word for word."""

from pathlib import Path

from mtc.consciousness.developmental_modulator import developmental_enabled
from mtc.core.config import COMPANION_TYPES, Settings
from mtc.onboarding.schema import (
    CONTENT_BOUNDARY_RULE,
    NON_ROMANTIC_RULE,
    PARTNER_BOUNDS_SENTENCE,
    SELF_KNOWLEDGE_LINE,
    load_type_defaults,
)

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "ONBOARDING.md"


def _type_rows(text: str):
    section = text.split("## Companion types", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0].startswith("`"):
            rows[cells[0].strip("`")] = cells[1:]
    return rows


def _flatten(text: str) -> str:
    """Join the lines of block quotes so a wrapped sentence reads whole."""
    lines = [line.lstrip("> ").strip() for line in text.splitlines()]
    return " ".join(line for line in lines if line)


def test_type_table_matches_types_yaml():
    rows = _type_rows(DOC.read_text(encoding="utf-8"))
    defaults = load_type_defaults()
    assert set(rows) == set(COMPANION_TYPES)
    for companion_type, row in defaults.items():
        name = companion_type.value
        address, trust, pack, relationship, stages = rows[name]
        assert address == row.address_form_rule, name
        assert float(trust) == row.creator_trust, name
        assert pack == (row.seed_pack or "none"), name
        assert relationship == row.creator_relationship, name
        enabled = developmental_enabled(
            Settings(
                _env_file=None, COMPANION_TYPE=name, DEVELOPMENTAL_STAGE_ENABLED=False
            )
        )
        assert stages == ("on" if enabled else "off"), name


def test_partner_bounds_stated():
    text = _flatten(DOC.read_text(encoding="utf-8"))
    for sentence in (
        SELF_KNOWLEDGE_LINE,
        CONTENT_BOUNDARY_RULE,
        NON_ROMANTIC_RULE,
        PARTNER_BOUNDS_SENTENCE,
    ):
        assert sentence in text, sentence


def test_doc_does_not_promise_answers_files_hold_no_secret():
    """The answers model accepts a typed store password and the Anthropic key
    (the prompt fills them, and the web page will), so the document can only
    say an answers file should not hold one, and it says where the field is
    for; and it says that nothing creates a configured MongoDB user."""
    import re

    from mtc.onboarding.schema import OnboardingAnswers

    accepted = OnboardingAnswers.model_validate(
        {
            "schema_version": 1,
            "acknowledged": True,
            "persona": {
                "name": "Tessera",
                "pronouns": {
                    "subject": "they",
                    "object": "them",
                    "possessive": "their",
                },
                "companion_type": "friend",
                "creator_name": "Alice",
            },
            "locale": {"timezone": "UTC", "wake_hour": 6, "sleep_hour": 22},
            "stores": {"mongodb": {"enabled": True, "password": "typed0value1"}},
            "brain": {
                "provider": "anthropic",
                "anthropic_model": "example-model",
                "anthropic_api_key": "typed0value2",
            },
        }
    )
    assert accepted.stores.mongodb.password is not None
    assert accepted.brain.anthropic_api_key is not None
    text = re.sub(r"\s+", " ", DOC.read_text(encoding="utf-8"))
    assert "an answers file never holds" not in text
    assert "an answers file should not hold one" in text
    assert "nothing creates the MongoDB user" in text
