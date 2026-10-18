"""
Prompt Assembler
================

The one renderer of the system prompt. Sections are small callables over
``PromptInputs``; the assembler renders them in ``SECTION_ORDER`` and joins
the non-empty ones. The first six sections (persona with the identity
statement, core memory, speaker, style rules, thinking instructions and
registered capabilities) form the stable prefix, so a server that caches a
prompt prefix sees the same bytes from turn to turn for the same speaker; the
dynamic suffix follows.

Rules the renderer enforces:

- The persona resolves in order: the mapping passed in or the operator's
  ``PERSONA_PATH`` file (``configured``), the shipped
  ``config/persona.example.yaml`` next to the code or under ``PROJECT_ROOT``
  (``example``), its packaged copy for an install without a checkout
  (``package``, through ``mtc.core.resources``), then a minimal built-in
  neutral text (``builtin``); ``persona_source`` reports which applied.
- Every persona placeholder is one of ``PLACEHOLDERS``; an unknown placeholder
  raises ``ValueError`` naming it. Nothing is invented for a missing field: an
  empty persona name renders ``SYSTEM_NAME``, an empty creator ``CREATOR_NAME``.
- The speaker block is chosen by the verified identity alone: the ``creator``
  template only for an authenticated speaker whose role is ``creator``, the
  ``known`` template for any other authenticated speaker, the ``unverified``
  template for a name no token backs. The name is displayed, never compared:
  it is rendered on one line, and under the ``unverified`` template it is put
  in double quotes (a double quote inside it becomes an apostrophe), so a
  claimed name cannot start a section of its own.
- Capability text appears only through ``register_capability``, which an
  enabled feature calls; none ships. The ``perception`` section is reserved
  and can never be registered: vision input is not part of this release.
- The measurement and theory slots (neural state, attention, metacognition,
  the inner-state narrative, active inference, the global workspace,
  coherence and self-awareness) never render Phi: a mapping entry whose key
  or rendered value names Phi or the integration proxy is dropped, and so is
  any line of a string slot that does (the integration proxy is a measurement
  for the API and the logs, not a line in a prompt). Memories, the summary and
  user text are conversation content and are not filtered. Thought mood
  stamps render only when ``narrate_thought_mood`` is true.
- Memory dates render in ``TIMEZONE``; the clock section is the temporal
  context's ``build_now_block`` over the injected clock.
- ``disabled_sections`` ablate sections by name for experiments; an unknown
  name raises ``ValueError``.

Config keys read: ``SYSTEM_NAME``, ``CREATOR_NAME``, ``PERSONA_PATH``,
``IDENTITY_STATEMENT_PATH``, ``TIMEZONE`` (through the temporal context),
``LLM_TEMPLATE_FAMILY``, ``THINKING_MODE``, ``PROMPT_MAX_THOUGHTS``,
``PROMPT_MAX_GISTS``, ``PROMPT_MAX_MEMORIES``, ``PROMPT_MAX_MEMORY_CHARS``,
``PROMPT_MAX_CONCEPTS``, ``PROMPT_DISABLED_SECTIONS`` and the thought-mood
narration flag. No state is persisted. The rendered prompt holds the speaker's
name, memories and thoughts, which are private at runtime; nothing here logs
them.
"""

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from zoneinfo import ZoneInfo

from mtc.brain.identity_statement import load_identity_statement
from mtc.consciousness.temporal_context import build_now_block
from mtc.core.config import Settings, csv_list, get_settings
from mtc.core.paths import resolve_path
from mtc.core.persona import load_persona
from mtc.core.resources import resolve_shipped
from mtc.memory.core_blocks import CORE_BLOCK_NAMES

logger = logging.getLogger(__name__)

SECTION_ORDER: Tuple[str, ...] = (
    "persona",
    "core_memory",
    "speaker",
    "style",
    "thinking",
    "capabilities",
    "ctm_thoughts",
    "thought_gists",
    "neural_state",
    "attention_guidance",
    "metacognitive_insights",
    "consciousness_narrative",
    "active_inference",
    "global_workspace",
    "memories",
    "concepts",
    "coherence",
    "summary",
    "directives",
    "self_awareness",
    "clock",
    "window_notes",
)
STABLE_PREFIX_SECTIONS: Tuple[str, ...] = SECTION_ORDER[:6]
# Reserved and never registered: vision input is not part of this release.
RESERVED_SECTIONS: Tuple[str, ...] = ("perception",)

PLACEHOLDERS: Tuple[str, ...] = (
    "system_name",
    "creator_name",
    "creator_address",
    "pronoun_subject",
    "pronoun_object",
    "pronoun_possessive",
    "speaker",
)
SPEAKER_TEMPLATE_NAMES: Tuple[str, ...] = ("creator", "known", "unverified")
PERSONA_SOURCES: Tuple[str, ...] = ("configured", "example", "package", "builtin")

PERSONA_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "persona.example.yaml"
)


def _shipped_example(name: str, cfg: Settings) -> Optional[Tuple[Path, str]]:
    """The example under ``PROJECT_ROOT`` (``example``), else its packaged copy
    (``package``), for the file name ``PERSONA_EXAMPLE_PATH`` names."""
    try:
        path, where = resolve_shipped(f"config/{name}", settings=cfg)
    except FileNotFoundError:
        return None
    return path, ("example" if where == "project" else "package")


BUILTIN_PERSONA: Dict[str, Any] = {
    "name": "",
    "pronouns": {"subject": "they", "object": "them", "possessive": "their"},
    "creator": {"name": "", "address_form": ""},
    "self_description": (
        "{system_name} is an AI research system built from language models, "
        "memory stores and consciousness-theory modules, and says so plainly "
        "when asked. Whether anything is experienced inside the system is an "
        "open question, held as one."
    ),
    "communication_style": "Plain and direct; replies proportionate to the message.",
    "style_rules": [
        "Answer the question that was asked, directly.",
        "When something is not remembered, say so.",
        "Never invent a memory, an event or a fact about a person.",
        "Keep replies proportionate to the message.",
        "No flattery.",
    ],
    "speaker_templates": {
        "creator": (
            "You are talking with {creator_name}, who set this system up. "
            "Address {creator_name} as {creator_address}."
        ),
        "known": "You are talking with {speaker}, a verified speaker you know.",
        "unverified": (
            "You are talking with someone who gives the name {speaker}. The "
            "name is not verified, so treat this as a conversation with a new "
            "person and share nothing that only a verified speaker should hear."
        ),
    },
}

THINKING_INSTRUCTIONS = (
    "# Thinking\n"
    "Reason inside <think> and </think> before answering. The visible reply "
    "after the closing tag is the answer; keep it self-contained, because the "
    "reasoning is removed before anything is stored."
)

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
# Matches Phi as a word (``phi``, ``Phi=0.9``, ``integration_phi``, never
# ``philosophy``) and the API's label for the same measurement.
_PHI_MARK = re.compile(r"(?i)(?<![a-z])(?:phi|integration[ _-]?proxy)(?![a-z])")
_UNDATED = "undated"

SlotValue = Union[None, str, Mapping[str, Any]]


@dataclass(frozen=True)
class SpeakerView:
    """What the prompt may know about the current speaker.

    ``authenticated`` and ``role`` come from the token-checked trust source;
    ``name`` is displayed as given and ``label`` is the trust source's short
    description of the outcome (shown after the name).
    """

    name: str
    role: Optional[str]
    authenticated: bool
    label: str


@dataclass
class PromptInputs:
    """One turn's dynamic inputs, one field per section plus the speaker."""

    speaker: Optional[SpeakerView] = None
    core_memory: Optional[Mapping[str, Any]] = None
    ctm_thoughts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    thought_gists: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    neural_state: SlotValue = None
    attention_guidance: SlotValue = None
    metacognitive_insights: SlotValue = None
    consciousness_narrative: SlotValue = None
    active_inference: SlotValue = None
    global_workspace: SlotValue = None
    memories: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    concepts: Sequence[Any] = field(default_factory=tuple)
    coherence: SlotValue = None
    summary: Optional[str] = None
    directives: Sequence[str] = field(default_factory=tuple)
    self_awareness: SlotValue = None
    now: Optional[datetime] = None
    last_turn_at: Optional[datetime] = None
    window_notes: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class PromptCaps:
    """How many items each list section renders."""

    thoughts: int = 5
    gists: int = 5
    memories: int = 5
    memory_chars: int = 250
    concepts: int = 3

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "PromptCaps":
        cfg = settings if settings is not None else get_settings()
        return cls(
            thoughts=int(cfg.PROMPT_MAX_THOUGHTS),
            gists=int(cfg.PROMPT_MAX_GISTS),
            memories=int(cfg.PROMPT_MAX_MEMORIES),
            memory_chars=int(cfg.PROMPT_MAX_MEMORY_CHARS),
            concepts=int(cfg.PROMPT_MAX_CONCEPTS),
        )


@dataclass(frozen=True)
class AssembledPrompt:
    """The rendered prompt, the rendered sections with their sizes, and the
    length of the stable prefix."""

    text: str
    sections: List[Tuple[str, int]]
    stable_prefix_chars: int


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _one_line(value: Any) -> str:
    """The value as text on one line: every run of whitespace, line breaks
    included, becomes one space."""
    return " ".join(_text(value).split())


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple, set, frozenset)):
        return ", ".join(_format_scalar(item) for item in value)
    return _text(value)


def names_phi(text: Any) -> bool:
    """True when the text names Phi or the integration proxy as a word."""
    return bool(_PHI_MARK.search(str(text)))


def _mapping_lines(
    value: Mapping[str, Any], prefix: str = "", drop_phi: bool = False
) -> List[str]:
    """Render a mapping as ``key: value`` lines.

    With ``drop_phi`` every entry whose key or rendered value names Phi is left
    out, nested mappings included.
    """
    lines: List[str] = []
    for key, item in value.items():
        name = f"{prefix}{key}"
        if drop_phi and names_phi(key):
            continue
        if isinstance(item, Mapping):
            lines.extend(_mapping_lines(item, prefix=f"{name}.", drop_phi=drop_phi))
            continue
        rendered = _format_scalar(item)
        if not rendered or (drop_phi and names_phi(rendered)):
            continue
        lines.append(f"{name}: {rendered}")
    return lines


def _slot_body(value: SlotValue, drop_phi: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return "\n".join(_mapping_lines(value, drop_phi=drop_phi))
    text = _text(value)
    if not drop_phi:
        return text
    kept = [line for line in text.splitlines() if not names_phi(line)]
    return "\n".join(kept).strip()


def _render_slot(header: str, value: SlotValue) -> str:
    """Render a conversation-text slot (the summary) as given."""
    body = _slot_body(value)
    return f"# {header}\n{body}" if body else ""


def _render_measurement(header: str, value: SlotValue) -> str:
    """Render a measurement or theory slot with every Phi line dropped."""
    body = _slot_body(value, drop_phi=True)
    return f"# {header}\n{body}" if body else ""


def _as_aware(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str) and value.strip():
        try:
            stamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


class PromptAssembler:
    """Renders the system prompt from the persona and one turn's inputs."""

    def __init__(
        self,
        persona: Optional[Mapping[str, Any]] = None,
        family: str = "generic",
        thinking_mode: bool = False,
        caps: Optional[PromptCaps] = None,
        disabled_sections: Sequence[str] = (),
        narrate_thought_mood: bool = False,
        identity_statement: Optional[str] = None,
        clock: Optional[Callable[[], datetime]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings
        self.family = str(family)
        self.thinking_mode = bool(thinking_mode)
        self.caps = caps if caps is not None else PromptCaps()
        self.narrate_thought_mood = bool(narrate_thought_mood)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._capabilities: "OrderedDict[str, str]" = OrderedDict()

        unknown = [name for name in disabled_sections if name not in SECTION_ORDER]
        if unknown:
            raise ValueError(
                "PROMPT_DISABLED_SECTIONS names unknown sections: "
                + ", ".join(sorted(unknown))
            )
        self.disabled_sections: Tuple[str, ...] = tuple(disabled_sections)

        self.persona, self.persona_source = self._resolve_persona(persona)
        if identity_statement is None:
            cfg = self._cfg()
            identity_statement = load_identity_statement(
                resolve_path(cfg.IDENTITY_STATEMENT_PATH, settings=cfg)
            )
        self.identity_statement = _text(identity_statement)

        self._sections: Dict[str, Callable[[PromptInputs], str]] = {
            "persona": self._persona_section,
            "core_memory": self._core_memory_section,
            "speaker": self._speaker_section,
            "style": self._style_section,
            "thinking": self._thinking_section,
            "capabilities": self._capabilities_section,
            "ctm_thoughts": self._thoughts_section,
            "thought_gists": self._gists_section,
            "neural_state": lambda i: _render_measurement(
                "Neural state", i.neural_state
            ),
            "attention_guidance": lambda i: _render_measurement(
                "Attention", i.attention_guidance
            ),
            "metacognitive_insights": lambda i: _render_measurement(
                "Metacognition", i.metacognitive_insights
            ),
            "consciousness_narrative": lambda i: _render_measurement(
                "Inner state narrative", i.consciousness_narrative
            ),
            "active_inference": lambda i: _render_measurement(
                "Active inference", i.active_inference
            ),
            "global_workspace": lambda i: _render_measurement(
                "Global workspace", i.global_workspace
            ),
            "memories": self._memories_section,
            "concepts": self._concepts_section,
            "coherence": lambda i: _render_measurement(
                "Conversation coherence", i.coherence
            ),
            "summary": lambda i: _render_slot(
                "Earlier in this conversation", i.summary
            ),
            "directives": lambda i: self._bullets("Directives", i.directives),
            "self_awareness": lambda i: _render_measurement(
                "Self-awareness", i.self_awareness
            ),
            "clock": self._clock_section,
            "window_notes": lambda i: self._bullets(
                "Notes about this window", i.window_notes
            ),
        }

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "PromptAssembler":
        """Build the assembler the live pipeline uses, every knob from Settings."""
        from mtc.core.narration import NarrationFlags

        cfg = settings if settings is not None else get_settings()
        return cls(
            persona=None,
            family=cfg.LLM_TEMPLATE_FAMILY,
            thinking_mode=bool(cfg.THINKING_MODE),
            caps=PromptCaps.from_settings(cfg),
            disabled_sections=tuple(csv_list(cfg.PROMPT_DISABLED_SECTIONS)),
            narrate_thought_mood=NarrationFlags.from_settings(cfg).thought_mood,
            identity_statement=None,
            settings=cfg,
        )

    def _cfg(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    def _resolve_persona(
        self, persona: Optional[Mapping[str, Any]]
    ) -> Tuple[Dict[str, Any], str]:
        if persona is not None:
            return dict(persona), "configured"
        cfg = self._cfg()
        configured = load_persona(settings=cfg)
        if configured is not None:
            return dict(configured), "configured"
        example = Path(PERSONA_EXAMPLE_PATH)
        if example.is_file():
            loaded = load_persona(example, settings=cfg)
            if loaded is not None:
                logger.info("persona: PERSONA_PATH absent, using the shipped example")
                return dict(loaded), "example"
        shipped = _shipped_example(example.name, cfg)
        if shipped is not None:
            path, source = shipped
            loaded = load_persona(path, settings=cfg)
            if loaded is not None:
                logger.info(
                    "persona: PERSONA_PATH absent, using the %s example", source
                )
                return dict(loaded), source
        logger.info("persona: no file found, using the built-in neutral text")
        return dict(BUILTIN_PERSONA), "builtin"

    def register_capability(self, name: str, text: str) -> None:
        """Add a capability block; called only by a feature that can run."""
        key = _text(name)
        if not key:
            raise ValueError("a capability needs a name")
        if key in RESERVED_SECTIONS:
            raise ValueError(
                f"section {key!r} is reserved: vision input is not part of this "
                "release"
            )
        if key in SECTION_ORDER:
            raise ValueError(f"{key!r} is a prompt section name, not a capability")
        body = _text(text)
        if not body:
            raise ValueError(f"capability {key!r} needs text")
        self._capabilities[key] = body

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, inputs: PromptInputs) -> AssembledPrompt:
        """Render every enabled section in order and report the prefix size."""
        prefix_parts: List[str] = []
        suffix_parts: List[str] = []
        sections: List[Tuple[str, int]] = []
        for name in SECTION_ORDER:
            if name in self.disabled_sections:
                continue
            body = self._sections[name](inputs).strip()
            if not body:
                continue
            sections.append((name, len(body)))
            if name in STABLE_PREFIX_SECTIONS:
                prefix_parts.append(body)
            else:
                suffix_parts.append(body)
        prefix = "\n\n".join(prefix_parts)
        suffix = "\n\n".join(suffix_parts)
        if prefix and suffix:
            text = f"{prefix}\n\n{suffix}"
        else:
            text = prefix or suffix
        return AssembledPrompt(
            text=text, sections=sections, stable_prefix_chars=len(prefix)
        )

    # ------------------------------------------------------------------
    # Placeholders
    # ------------------------------------------------------------------

    def _values(self, speaker: Optional[SpeakerView] = None) -> Dict[str, str]:
        cfg = self._cfg()
        creator = _mapping(self.persona.get("creator"))
        pronouns = _mapping(self.persona.get("pronouns"))
        creator_name = _text(creator.get("name")) or _text(cfg.CREATOR_NAME)
        return {
            "system_name": _text(self.persona.get("name")) or _text(cfg.SYSTEM_NAME),
            "creator_name": creator_name,
            "creator_address": _text(creator.get("address_form")) or creator_name,
            "pronoun_subject": _text(pronouns.get("subject")) or "they",
            "pronoun_object": _text(pronouns.get("object")) or "them",
            "pronoun_possessive": _text(pronouns.get("possessive")) or "their",
            "speaker": _one_line(speaker.name) if speaker is not None else "",
        }

    def _fill(self, template: str, values: Mapping[str, str], where: str) -> str:
        def replace(match: "re.Match[str]") -> str:
            key = match.group(1)
            if key not in values:
                raise ValueError(
                    f"prompt placeholder {{{key}}} in {where} has no value; the "
                    f"known placeholders are {', '.join(PLACEHOLDERS)}"
                )
            return values[key]

        return _PLACEHOLDER.sub(replace, template)

    # ------------------------------------------------------------------
    # Stable prefix sections
    # ------------------------------------------------------------------

    def _persona_section(self, inputs: PromptInputs) -> str:
        values = self._values()
        lines: List[str] = []
        if self.identity_statement:
            lines.append(
                self._fill(self.identity_statement, values, "the identity statement")
            )
        else:
            lines.append("# Identity")
        lines.append(f"You are {values['system_name']}.")
        pronouns = _mapping(self.persona.get("pronouns"))
        if pronouns:
            lines.append(
                "Pronouns: "
                f"{values['pronoun_subject']}/{values['pronoun_object']}/"
                f"{values['pronoun_possessive']}."
            )
        description = _text(self.persona.get("self_description"))
        if description:
            lines.append(self._fill(description, values, "persona.self_description"))
        style = _text(self.persona.get("communication_style"))
        if style:
            lines.append(
                "Communication style: "
                + self._fill(style, values, "persona.communication_style")
            )
        return "\n".join(lines)

    def _core_memory_section(self, inputs: PromptInputs) -> str:
        blocks = _mapping(inputs.core_memory)
        if not blocks:
            return ""
        ordered = [name for name in CORE_BLOCK_NAMES if _text(blocks.get(name))]
        ordered += sorted(
            name
            for name in blocks
            if name not in CORE_BLOCK_NAMES and _text(blocks[name])
        )
        if not ordered:
            return ""
        lines = ["# Core memory"]
        for name in ordered:
            lines.append(f"## {name}")
            lines.append(_text(blocks[name]))
        return "\n".join(lines)

    def _speaker_section(self, inputs: PromptInputs) -> str:
        speaker = inputs.speaker
        if speaker is None:
            return ""
        templates = _mapping(self.persona.get("speaker_templates"))
        if speaker.authenticated and _text(speaker.role) == "creator":
            which = "creator"
        elif speaker.authenticated:
            which = "known"
        else:
            which = "unverified"
        template = _text(templates.get(which)) or _text(
            BUILTIN_PERSONA["speaker_templates"][which]
        )
        values = self._values(speaker)
        if not values["speaker"]:
            values["speaker"] = _one_line(speaker.label) or "an unnamed speaker"
        elif which == "unverified":
            # Quoted, so the claim reads as a claim; the name cannot close it.
            values["speaker"] = '"' + values["speaker"].replace('"', "'") + '"'
        lines = [
            "# Speaker",
            self._fill(template, values, f"persona.speaker_templates.{which}"),
        ]
        label = _one_line(speaker.label)
        if label:
            lines.append(f"Speaker status: {label}.")
        return "\n".join(lines)

    def _style_section(self, inputs: PromptInputs) -> str:
        rules = self.persona.get("style_rules")
        if isinstance(rules, str):
            items = [rules]
        elif isinstance(rules, (list, tuple)):
            items = [_text(rule) for rule in rules if _text(rule)]
        else:
            items = []
        return self._bullets("Style", items)

    def _thinking_section(self, inputs: PromptInputs) -> str:
        if not self.thinking_mode or not _family_thinks(self.family):
            return ""
        return THINKING_INSTRUCTIONS

    def _capabilities_section(self, inputs: PromptInputs) -> str:
        if not self._capabilities:
            return ""
        lines = ["# Capabilities"]
        lines.extend(self._capabilities.values())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dynamic sections
    # ------------------------------------------------------------------

    def _zone(self) -> ZoneInfo:
        try:
            return ZoneInfo(self._cfg().TIMEZONE)
        except Exception as exc:  # noqa: BLE001 - a bad zone falls back to UTC
            logger.warning("prompt dates fall back to UTC: %s", type(exc).__name__)
            return ZoneInfo("UTC")

    def _stamp(self, value: Any, fmt: str = "%Y-%m-%d %H:%M") -> str:
        instant = _as_aware(value)
        if instant is None:
            return _UNDATED
        return instant.astimezone(self._zone()).strftime(fmt)

    def _thoughts_section(self, inputs: PromptInputs) -> str:
        items = [i for i in inputs.ctm_thoughts if _text(_mapping(i).get("content"))]
        items = items[: max(0, self.caps.thoughts)]
        if not items:
            return ""
        lines = ["# Recent thoughts (from the continuous thought stream)"]
        for item in items:
            stamp = self._stamp(item.get("timestamp"), "%H:%M")
            kind = _text(item.get("thought_type"))
            line = f"- [{stamp}]"
            if kind:
                line += f" ({kind})"
            line += f" {_text(item.get('content'))}"
            mood = _text(item.get("emotional_tone"))
            if self.narrate_thought_mood and mood:
                line += f" [mood: {mood}]"
            lines.append(line)
        return "\n".join(lines)

    def _gists_section(self, inputs: PromptInputs) -> str:
        items = [i for i in inputs.thought_gists if _text(_mapping(i).get("summary"))]
        items = items[: max(0, self.caps.gists)]
        if not items:
            return ""
        lines = ["# Thought gists"]
        for item in items:
            line = f"- {_text(item.get('summary'))}"
            count = item.get("source_count")
            if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                noun = "thought" if count == 1 else "thoughts"
                line += f" (from {count} {noun})"
            emotion = _text(item.get("dominant_emotion"))
            if self.narrate_thought_mood and emotion:
                line += f" [mood: {emotion}]"
            lines.append(line)
        return "\n".join(lines)

    def _memories_section(self, inputs: PromptInputs) -> str:
        items = [
            i
            for i in inputs.memories
            if _text(_mapping(i).get("content") or _mapping(i).get("text"))
        ]
        items = items[: max(0, self.caps.memories)]
        if not items:
            return ""
        limit = max(1, self.caps.memory_chars)
        lines = ["# Relevant memories"]
        for item in items:
            body = _text(item.get("content") or item.get("text"))
            if len(body) > limit:
                body = body[:limit].rstrip() + "..."
            stamp = self._stamp(
                item.get("timestamp", item.get("created_at", item.get("stored_at")))
            )
            lines.append(f"- [{stamp}] {body}")
        return "\n".join(lines)

    def _concepts_section(self, inputs: PromptInputs) -> str:
        names: List[str] = []
        for item in inputs.concepts:
            name = (
                _text(_mapping(item).get("name"))
                if isinstance(item, Mapping)
                else _text(item)
            )
            if name and name not in names:
                names.append(name)
        names = names[: max(0, self.caps.concepts)]
        if not names:
            return ""
        return "# Related concepts\n" + ", ".join(names)

    def _clock_section(self, inputs: PromptInputs) -> str:
        now = inputs.now if inputs.now is not None else self._clock()
        return build_now_block(
            now, last_exchange_at=inputs.last_turn_at, settings=self._cfg()
        )

    @staticmethod
    def _bullets(header: str, items: Sequence[Any]) -> str:
        texts = [_text(item) for item in items if _text(item)]
        if not texts:
            return ""
        return "\n".join([f"# {header}", *(f"- {text}" for text in texts)])


def _family_thinks(family: str) -> bool:
    """True when the template family emits a reasoning block the brain reads."""
    from mtc.brain.families import get_family

    return get_family(family).is_reasoning_family
