"""The persona text the thought enhancer prompts the model with.

``CTMPersona`` holds the system prompt for a background thought, one short
instruction per thought category and the labels that introduce each piece of
context. It is configuration written by the operator: ``config/ctm/persona.yaml``
(gitignored), else the shipped neutral ``config/ctm/persona.example.yaml``,
else its packaged copy (``package``, only for the default ``CTM_CONFIG_DIR``),
else the built-in neutral text below; ``source`` says which one was used. A
partial file keeps the built-in text for every key it leaves out.

File keys: ``name`` (empty means ``SYSTEM_NAME``), ``system_prompt`` (may use
``{name}`` only), ``category_prompts`` (one instruction for each of
``CATEGORIES``, with no placeholders) and ``prompt_headers`` (a label for each
of ``recent_memory``, ``emotions`` and ``recent_topic``). A placeholder that is
not ``{name}``, which includes any creator or speaker field, is a
``PersonaError`` at load time: thought prompts never refer to the creator.

``render_category`` builds the user prompt from the context the caller supplies.
A category that needs a context value (``memory`` needs ``recent_memory`` and
``emotion`` needs ``emotions``) raises ``PersonaError`` naming the missing key
when it is absent or blank: nothing is invented in its place. An optional value
that is absent is left out. An emotion whose score is not a finite number is
skipped, never shown with an invented score, so an ``emotions`` mapping with
no numeric score counts as absent. Context values (memories, topics, emotions)
are private runtime data and are never logged here.

Config keys read: ``CTM_CONFIG_DIR`` (read at call time) and ``SYSTEM_NAME``.
PyYAML is imported inside ``load``, never at module import.
"""

from __future__ import annotations

import logging
import math
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from mtc.consciousness.continuous.config import ctm_config_dir, shipped_file
from mtc.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

CATEGORIES = ("reflection", "question", "emotion", "memory", "creative")
CONTEXT_KEYS = ("recent_memory", "emotions", "recent_topic")
# The context a category shows, and the part of it the category cannot do without.
CATEGORY_CONTEXT: Dict[str, tuple] = {
    "reflection": (),
    "question": ("recent_topic",),
    "emotion": ("emotions",),
    "memory": ("recent_memory",),
    "creative": (),
}
CATEGORY_REQUIRED: Dict[str, tuple] = {
    "reflection": (),
    "question": (),
    "emotion": ("emotions",),
    "memory": ("recent_memory",),
    "creative": (),
}

PERSONA_FILE = "persona.yaml"
PERSONA_EXAMPLE_FILE = "persona.example.yaml"

BUILTIN_SYSTEM_PROMPT = (
    "You are {name}, a system with a continuous stream of background thoughts. "
    "Write one brief spontaneous thought: honest, introspective, a sentence or "
    "two. Give the thought itself only, with no meta-commentary and no preamble."
)
BUILTIN_CATEGORY_PROMPTS: Dict[str, str] = {
    "reflection": "Write one brief reflection on the present moment.",
    "question": "Write one open question that you find yourself wondering about.",
    "emotion": "Write one brief thought noticing this state and what it might mean.",
    "memory": "Write one brief thought looking back at this memory.",
    "creative": "Write one brief imaginative thought, a small image or idea.",
}
BUILTIN_PROMPT_HEADERS: Dict[str, str] = {
    "recent_memory": "A recent memory",
    "emotions": "Current emotional state",
    "recent_topic": "A recent topic",
}


class PersonaError(ValueError):
    """A persona file is invalid or a prompt cannot be rendered from context."""


@dataclass
class CTMPersona:
    """System prompt, category instructions and context labels for thoughts."""

    name: str
    system_prompt: str
    category_prompts: Dict[str, str] = field(
        default_factory=lambda: dict(BUILTIN_CATEGORY_PROMPTS)
    )
    prompt_headers: Dict[str, str] = field(
        default_factory=lambda: dict(BUILTIN_PROMPT_HEADERS)
    )
    source: str = "builtin"
    path: Optional[Path] = None

    def render_system(self, clock_line: Optional[str] = None) -> str:
        """Return the system prompt with ``{name}`` filled.

        A non-empty ``clock_line`` is appended on its own sentence; the persona
        never builds a clock line itself.
        """
        text = self.system_prompt.replace("{name}", self.name).strip()
        extra = (clock_line or "").strip()
        return f"{text} {extra}" if extra else text

    def render_category(self, category: str, context: Mapping[str, Any]) -> str:
        """Return the user prompt for a category from the supplied context.

        Raises ``PersonaError`` for an unknown category and for a required
        context value that is absent or blank, naming the key.
        """
        if category not in CATEGORIES:
            raise PersonaError(
                f"unknown thought category {category!r}; expected one of {CATEGORIES}"
            )
        lines = []
        wanted = list(CATEGORY_CONTEXT[category])
        if "emotions" not in wanted and _present(context.get("emotions")):
            wanted.append("emotions")
        for key in wanted:
            text = _render_value(key, context.get(key))
            if not text:
                if key in CATEGORY_REQUIRED[category]:
                    raise PersonaError(
                        f"the {category!r} thought prompt needs the context key "
                        f"{key!r}, which is missing or empty"
                    )
                continue
            lines.append(f"{self.prompt_headers[key]}: {text}")
        instruction = self.category_prompts[category]
        return "\n".join(lines + ([""] if lines else []) + [instruction])

    @classmethod
    def load(
        cls,
        config_dir: Optional[Any] = None,
        *,
        settings: Optional[Settings] = None,
    ) -> "CTMPersona":
        """Load ``persona.yaml``, else ``persona.example.yaml``, else the
        packaged example (for the shipped default directory), else built-in."""
        cfg = settings if settings is not None else get_settings()
        directory = ctm_config_dir(config_dir, settings=cfg)
        for file_name, source in (
            (PERSONA_FILE, "configured"),
            (PERSONA_EXAMPLE_FILE, "example"),
        ):
            path = directory / file_name
            if path.is_file():
                return _from_file(path, source, cfg)
        packaged = shipped_file(PERSONA_EXAMPLE_FILE, config_dir, settings=cfg)
        if packaged is not None:
            return _from_file(packaged, "package", cfg)
        logger.info(
            "no CTM persona file in %s; the built-in neutral text applies", directory
        )
        return cls(name=cfg.SYSTEM_NAME, system_prompt=BUILTIN_SYSTEM_PROMPT)


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return len(value) > 0
    except TypeError:
        return True


def _render_value(key: str, value: Any) -> str:
    if not _present(value):
        return ""
    if key == "emotions" and isinstance(value, Mapping):
        ranked = sorted(_scored_emotions(value), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join(f"{name} ({score:.2f})" for name, score in ranked[:3])
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    return str(value).strip()


def _scored_emotions(value: Mapping[Any, Any]) -> List[Tuple[str, float]]:
    """The emotions with a finite numeric score (a bool is not a score).

    Any other entry is skipped, never shown with an invented score.
    """
    scored: List[Tuple[str, float]] = []
    for name, score in value.items():
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        number = float(score)
        if math.isfinite(number):
            scored.append((str(name), number))
    return scored


def _placeholders(text: str, where: str) -> set:
    try:
        parsed = list(string.Formatter().parse(text))
    except ValueError as exc:
        raise PersonaError(f"{where}: malformed braces: {exc}") from exc
    return {name for _, name, _, _ in parsed if name is not None}


def _from_file(path: Path, source: str, cfg: Settings) -> CTMPersona:
    import yaml

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise PersonaError(f"persona file {path} could not be parsed: {exc}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise PersonaError(f"persona file {path} must hold a mapping")
    unknown = sorted(
        set(loaded) - {"name", "system_prompt", "category_prompts", "prompt_headers"}
    )
    if unknown:
        raise PersonaError(
            f"persona file {path} has unknown keys: {', '.join(unknown)}"
        )

    name = loaded.get("name")
    if name is not None and not isinstance(name, str):
        raise PersonaError(f"persona file {path}: name must be text")
    system_prompt = loaded.get("system_prompt", BUILTIN_SYSTEM_PROMPT)
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise PersonaError(f"persona file {path}: system_prompt must be non-empty text")
    bad = _placeholders(system_prompt, f"persona file {path}, system_prompt") - {"name"}
    if bad:
        raise PersonaError(
            f"persona file {path}, system_prompt: placeholder {sorted(bad)[0]!r} is not "
            "supported; only {name} is, and thought prompts never name the creator"
        )

    category_prompts = dict(BUILTIN_CATEGORY_PROMPTS)
    given = loaded.get("category_prompts") or {}
    if not isinstance(given, dict):
        raise PersonaError(f"persona file {path}: category_prompts must be a mapping")
    for key, text in given.items():
        if key not in CATEGORIES:
            raise PersonaError(f"persona file {path}: unknown category {key!r}")
        if not isinstance(text, str) or not text.strip():
            raise PersonaError(f"persona file {path}: category {key!r} needs text")
        if _placeholders(text, f"persona file {path}, category {key!r}"):
            raise PersonaError(
                f"persona file {path}: category {key!r} may not use placeholders"
            )
        category_prompts[key] = text.strip()

    headers = dict(BUILTIN_PROMPT_HEADERS)
    given = loaded.get("prompt_headers") or {}
    if not isinstance(given, dict):
        raise PersonaError(f"persona file {path}: prompt_headers must be a mapping")
    for key, text in given.items():
        if key not in CONTEXT_KEYS:
            raise PersonaError(f"persona file {path}: unknown header {key!r}")
        if not isinstance(text, str) or not text.strip():
            raise PersonaError(f"persona file {path}: header {key!r} needs text")
        headers[key] = text.strip()

    return CTMPersona(
        name=(name or "").strip() or cfg.SYSTEM_NAME,
        system_prompt=system_prompt.strip(),
        category_prompts=category_prompts,
        prompt_headers=headers,
        source=source,
        path=path,
    )
