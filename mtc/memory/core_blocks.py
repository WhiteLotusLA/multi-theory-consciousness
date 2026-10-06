"""
Core Memory Blocks
==================

Renders the three core memory blocks (``identity``, ``persona``, ``human``)
the memory manager seeds into core memory and the prompt assembler places at
the front of every prompt. The text comes from the operator's persona
(``mtc.core.persona.load_persona``, read from ``PERSONA_PATH``) or, without
one, from the neutral templates below filled from Settings at call time:
``SYSTEM_NAME``, ``CREATOR_NAME``, ``CREATOR_RELATIONSHIP``, ``CORE_VALUES``
and the Big Five seeds ``OPENNESS`` to ``NEUROTICISM``.

- ``identity``: the system name, that it is an AI research system built on
  this framework, the self-knowledge line every companion type shares (it
  knows it is an AI research system and does not claim certainty about its
  own consciousness), the core values, and the persona's pronouns and
  self-description when configured. A pronoun mapping renders as
  ``subject/object/possessive``; a self-description that already carries
  the self-knowledge line adds the rest of its text, so the line is
  stated once.
- ``persona``: the Big Five seed and the communication-style notes when
  configured, closed with one sentence mark whether or not the notes end
  in one.
- ``human``: the creator's name, address form and relationship label. It
  holds no trust value, no bond strength, no birth date and no model name:
  trust comes only from token-checked speaker entries, never from a block.

A persona's optional ``core_memory`` section replaces a rendered block with
its own text; the placeholders ``{system_name}``, ``{creator_name}``,
``{creator_address}``, ``{creator_relationship}``, ``{core_values}`` and
``{pronouns}`` are filled and any other brace text is left as written. The
blocks are written in the third person and in ASCII. This module reads
``CREATOR_NAME`` only to render the human block for display.
"""

import re
from typing import Any, Dict, Mapping, Optional

from mtc.core.config import Settings, csv_list, get_settings

CORE_BLOCK_NAMES = ("identity", "persona", "human")

BIG_FIVE_TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)

SELF_KNOWLEDGE_LINE = (
    "{system_name} knows it is an AI research system and does not claim "
    "certainty about its own consciousness."
)

NEUTRAL_TEMPLATES: Dict[str, str] = {
    "identity": (
        "{system_name} is an AI research system built on the Multi-Theory "
        "Consciousness framework. " + SELF_KNOWLEDGE_LINE + " Core values: "
        "{core_values}."
    ),
    "persona": (
        "Personality seed (Big Five): openness {openness}, conscientiousness "
        "{conscientiousness}, extraversion {extraversion}, agreeableness "
        "{agreeableness}, neuroticism {neuroticism}."
    ),
    "human": "{creator_name} is the {creator_relationship} of {system_name}.",
}

PRONOUN_KEYS = ("subject", "object", "possessive")

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
_SENTENCE_MARKS = (".", "!", "?")


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _pronouns(value: Any) -> str:
    """A pronoun mapping as ``subject/object/possessive``; other text as written."""
    if isinstance(value, Mapping):
        parts = [_text(value.get(key)) for key in PRONOUN_KEYS]
        return "/".join(part for part in parts if part)
    return _text(value)


def _closed(text: str) -> str:
    """``text`` ending in one sentence mark."""
    return text if text.endswith(_SENTENCE_MARKS) else f"{text}."


def _without(text: str, sentence: str) -> str:
    """``text`` without ``sentence``; left as written when it holds no copy."""
    flat = " ".join(text.split())
    if sentence not in flat:
        return text
    return " ".join(flat.replace(sentence, " ").split())


def _section(persona: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = persona.get(key)
    return value if isinstance(value, Mapping) else {}


def _core_values(persona: Mapping[str, Any], settings: Settings) -> str:
    raw = persona.get("core_values")
    if isinstance(raw, str):
        values = csv_list(raw)
    elif isinstance(raw, (list, tuple)):
        values = [_text(item) for item in raw if _text(item)]
    else:
        values = []
    if not values:
        values = csv_list(settings.CORE_VALUES)
    return ", ".join(values)


def _fill(template: str, values: Mapping[str, str]) -> str:
    """Fill known placeholders; leave any other brace text as written."""

    def replace(match: "re.Match[str]") -> str:
        key = match.group(1)
        return values[key] if key in values else match.group(0)

    return _PLACEHOLDER.sub(replace, template)


def render_core_blocks(
    persona: Optional[Mapping[str, Any]] = None,
    *,
    settings: Optional[Settings] = None,
) -> Dict[str, str]:
    """Return the ``identity``, ``persona`` and ``human`` blocks, in that order.

    ``persona`` is a loaded persona mapping or None for the neutral blocks;
    every value it lacks falls back to Settings, read at call time.
    """
    cfg = settings if settings is not None else get_settings()
    persona = persona or {}
    creator = _section(persona, "creator")
    personality = _section(persona, "personality")
    overrides = _section(persona, "core_memory")

    system_name = _text(persona.get("name")) or cfg.SYSTEM_NAME
    creator_name = _text(creator.get("name")) or cfg.CREATOR_NAME
    creator_address = _text(creator.get("address_form")) or creator_name
    values = {
        "system_name": system_name,
        "creator_name": creator_name,
        "creator_address": creator_address,
        "creator_relationship": cfg.CREATOR_RELATIONSHIP,
        "core_values": _core_values(persona, cfg),
        "pronouns": _pronouns(persona.get("pronouns")),
    }
    traits: Dict[str, str] = {}
    for trait in BIG_FIVE_TRAITS:
        raw = personality.get(trait)
        seed = float(raw) if raw is not None else float(getattr(cfg, trait.upper()))
        traits[trait] = f"{seed:.2f}"

    identity = _fill(NEUTRAL_TEMPLATES["identity"], values)
    if values["pronouns"]:
        identity += f" Pronouns: {values['pronouns']}."
    # The template above already states the self-knowledge line, with the
    # name filled or still as a placeholder in the persona's own text.
    description = _text(persona.get("self_description"))
    for line in (_fill(SELF_KNOWLEDGE_LINE, values), SELF_KNOWLEDGE_LINE):
        description = _without(description, line)
    if description:
        identity += f" {description}"

    persona_block = _fill(NEUTRAL_TEMPLATES["persona"], {**values, **traits})
    style = _text(persona.get("communication_style"))
    if style:
        persona_block += f" Communication style: {_closed(style)}"

    human = _fill(NEUTRAL_TEMPLATES["human"], values)
    if creator_address != creator_name:
        human += f" {system_name} addresses {creator_name} as {creator_address}."

    blocks = {"identity": identity, "persona": persona_block, "human": human}
    for name in CORE_BLOCK_NAMES:
        override = _text(overrides.get(name))
        if override:
            blocks[name] = _fill(override, values)
    return blocks
