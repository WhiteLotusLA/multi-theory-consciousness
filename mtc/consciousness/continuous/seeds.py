"""Seed packs: the template text the thought generator draws from.

A pack is a YAML mapping of lists of short templates and plain words. The
shipped ``config/ctm/seeds/default.yaml`` is neutral; an operator adds packs as
``config/ctm/seeds.d/*.yaml`` (gitignored, private), read in name order after
the default. Per key, a list extends the list built so far (an entry already
present is not repeated) and a scalar overrides; a pack may only use the
documented keys.

Pack keys:

- templates: ``memory_templates``, ``emotion_templates``,
  ``association_templates``, ``questions``, ``spontaneous_lines`` and the four
  dream families ``dream_consolidation_templates``,
  ``dream_creative_templates``, ``dream_emotion_templates`` and
  ``dream_spontaneous_templates``;
- plain entries: ``concept_pairs`` (two words each), ``creative_concepts`` and
  ``memory_query_terms``;
- slot words (``SLOT_KEYS``): ``reflection``, ``feeling``, ``emotion``,
  ``descriptor``, ``assessment`` and ``observation``;
- the scalar ``description``.

A template names a slot as ``{name}``. A slot word list fills it with one random
entry. A context slot (``CONTEXT_SLOTS``: ``memory``, ``current_emotion``,
``first``, ``second``, ``concept``) is filled by the caller with real material;
the thought generator supplies each of them for some strategy, so a template
using one can be chosen. A template naming any other slot is a ``SeedError`` at
load time (naming the file, the key and the slot), so a template no caller
could ever fill is refused rather than shipped dead; a context slot the caller
did not supply raises ``SeedError`` when the template is filled: nothing is
invented in its place. Slot word entries and plain entries carry no braces.

Config keys read: ``CTM_CONFIG_DIR`` (the pack directory, read at call time).
Pack text is configuration, not thought data; PyYAML is imported inside the
loader, never at module import.
"""

from __future__ import annotations

import logging
import random
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mtc.consciousness.continuous.config import ctm_config_dir, shipped_file
from mtc.core.config import Settings

logger = logging.getLogger(__name__)

SLOT_KEYS = (
    "reflection",
    "feeling",
    "emotion",
    "descriptor",
    "assessment",
    "observation",
)
CONTEXT_SLOTS = ("memory", "current_emotion", "first", "second", "concept")

TEMPLATE_KEYS = (
    "memory_templates",
    "emotion_templates",
    "association_templates",
    "questions",
    "spontaneous_lines",
    "dream_consolidation_templates",
    "dream_creative_templates",
    "dream_emotion_templates",
    "dream_spontaneous_templates",
)
WORD_KEYS = ("creative_concepts", "memory_query_terms") + SLOT_KEYS
PAIR_KEYS = ("concept_pairs",)
LIST_KEYS = TEMPLATE_KEYS + WORD_KEYS + PAIR_KEYS
SCALAR_KEYS = ("description",)
ALL_KEYS = LIST_KEYS + SCALAR_KEYS

DEFAULT_PACK = Path("seeds") / "default.yaml"
USER_PACK_DIR = "seeds.d"


class SeedError(ValueError):
    """A seed pack is unreadable or invalid, or a template cannot be filled."""


def template_fields(template: str) -> List[str]:
    """Return the slot names a template uses, in order of first use.

    Raises ``SeedError`` for a malformed template or a field with a format
    specifier, a conversion or no name.
    """
    names: List[str] = []
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise SeedError(f"malformed template {template!r}: {exc}") from exc
    for _, name, spec, conversion in parsed:
        if name is None:
            continue
        if not name or spec or conversion or not name.replace("_", "").isalpha():
            raise SeedError(f"unsupported slot syntax in template {template!r}")
        if name not in names:
            names.append(name)
    return names


class SeedLibrary:
    """The merged seed packs, read-only after construction.

    ``packs`` lists the files read, default first; ``source`` is ``file`` when
    the default pack was found and ``missing`` otherwise (every ``choose`` then
    raises ``SeedError``: there is no built-in text to fall back on).
    """

    def __init__(
        self,
        data: Mapping[str, Any],
        packs: Optional[Sequence[Path]] = None,
        source: Optional[str] = None,
    ) -> None:
        self._data: Dict[str, Any] = {k: v for k, v in data.items()}
        self.packs: List[Path] = list(packs or [])
        self._source = source

    @property
    def source(self) -> str:
        """``file``, ``package`` (the packaged default pack) or ``missing``."""
        if self._source:
            return self._source
        return "file" if self.packs else "missing"

    @property
    def description(self) -> str:
        return str(self._data.get("description", ""))

    def keys(self) -> List[str]:
        """Return the list keys that hold at least one entry."""
        return [k for k in LIST_KEYS if self._data.get(k)]

    def entries(self, key: str) -> List[Any]:
        """Return a copy of the entries under a key."""
        if key not in LIST_KEYS:
            raise SeedError(f"{key!r} is not a seed pack key")
        return list(self._data.get(key, []))

    def choose(
        self,
        key: str,
        rng: random.Random,
        *,
        available: Optional[Iterable[str]] = None,
    ) -> Any:
        """Return one entry of ``key`` chosen with ``rng``.

        ``available`` names the context slots the caller can fill; only a
        template whose context slots are all available is a candidate. A
        ``concept_pairs`` entry is returned as a tuple. No candidate raises
        ``SeedError`` naming the key.
        """
        entries = self.entries(key)
        if available is not None and key in TEMPLATE_KEYS:
            have = set(available)
            entries = [
                e
                for e in entries
                if all(name in SLOT_KEYS or name in have for name in template_fields(e))
            ]
        if not entries:
            raise SeedError(f"no usable entry under seed key {key!r}")
        choice = entries[rng.randrange(len(entries))]
        return tuple(choice) if key in PAIR_KEYS else choice

    def fill(self, template: str, rng: random.Random, **context: str) -> str:
        """Fill a template: slot words by ``rng``, context slots from ``context``.

        A context slot with no value raises ``SeedError`` naming it. The first
        letter of the result is capitalized.
        """
        out: List[str] = []
        for literal, name, _spec, _conv in string.Formatter().parse(template):
            out.append(literal)
            if name is None:
                continue
            if name in SLOT_KEYS:
                words = self._data.get(name) or []
                if not words:
                    raise SeedError(f"slot {name!r} has no words in the seed packs")
                out.append(words[rng.randrange(len(words))])
            elif name in context and context[name] is not None:
                out.append(str(context[name]))
            else:
                raise SeedError(
                    f"template {template!r} needs the context slot {name!r}, "
                    "which was not supplied"
                )
        text = "".join(out)
        return text[:1].upper() + text[1:]

    @classmethod
    def from_config_dir(
        cls,
        config_dir: Optional[Any] = None,
        *,
        settings: Optional[Settings] = None,
    ) -> "SeedLibrary":
        """Read ``seeds/default.yaml`` then ``seeds.d/*.yaml`` in name order.

        For the shipped default directory whose ``seeds/default.yaml`` is
        missing, the packaged default pack is read (``source`` is ``package``);
        user packs still come from the project's ``seeds.d``.
        A missing default pack is logged and leaves the library empty; a user
        pack directory that does not exist is ignored. An invalid pack raises
        ``SeedError`` naming the file.
        """
        directory = ctm_config_dir(config_dir, settings=settings)
        merged: Dict[str, Any] = {}
        used: List[Path] = []
        source: Optional[str] = None
        default = directory / DEFAULT_PACK
        if not default.is_file():
            packaged = shipped_file(DEFAULT_PACK, config_dir, settings=settings)
            if packaged is not None:
                default = packaged
                source = "package"
        if default.is_file():
            _merge_pack(merged, _read_pack(default), default)
            used.append(default)
        else:
            logger.warning("default seed pack %s not found; no seed text", default)
        user_dir = directory / USER_PACK_DIR
        if user_dir.is_dir():
            for path in sorted(user_dir.glob("*.yaml")):
                _merge_pack(merged, _read_pack(path), path)
                used.append(path)
        return cls(merged, used, source)


def _read_pack(path: Path) -> Dict[str, Any]:
    import yaml

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise SeedError(f"seed pack {path} could not be parsed: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SeedError(
            f"seed pack {path} must hold a mapping, got {type(loaded).__name__}"
        )
    return loaded


def _check_entries(key: str, value: Any, path: Path) -> List[Any]:
    where = f"seed pack {path}, key {key!r}"
    if not isinstance(value, list):
        raise SeedError(f"{where}: expected a list, got {type(value).__name__}")
    checked: List[Any] = []
    for item in value:
        if key in PAIR_KEYS:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(w, str) and w.strip() for w in item)
            ):
                raise SeedError(f"{where}: each entry must be two non-empty words")
            if any("{" in w or "}" in w for w in item):
                raise SeedError(f"{where}: plain entries carry no braces")
            checked.append(list(item))
            continue
        if not isinstance(item, str) or not item.strip():
            raise SeedError(f"{where}: each entry must be a non-empty string")
        if key in TEMPLATE_KEYS:
            try:
                names = template_fields(item)
            except SeedError as exc:
                raise SeedError(f"{where}: {exc}") from exc
            unknown = [n for n in names if n not in SLOT_KEYS + CONTEXT_SLOTS]
            if unknown:
                raise SeedError(
                    f"{where}: unknown slot {unknown[0]!r} in template {item!r}"
                )
        elif "{" in item or "}" in item:
            raise SeedError(f"{where}: plain entries carry no braces")
        checked.append(item)
    return checked


def _merge_pack(merged: Dict[str, Any], pack: Mapping[str, Any], path: Path) -> None:
    for key, value in pack.items():
        if key not in ALL_KEYS:
            raise SeedError(f"seed pack {path} has an unknown key {key!r}")
        if key in SCALAR_KEYS:
            if not isinstance(value, str):
                raise SeedError(f"seed pack {path}, key {key!r}: expected text")
            merged[key] = value
            continue
        entries = _check_entries(key, value, path)
        current: List[Any] = merged.setdefault(key, [])
        for entry in entries:
            if entry not in current:
                current.append(entry)


__all__: Tuple[str, ...] = (
    "ALL_KEYS",
    "CONTEXT_SLOTS",
    "LIST_KEYS",
    "SLOT_KEYS",
    "SeedError",
    "SeedLibrary",
    "TEMPLATE_KEYS",
    "template_fields",
)
