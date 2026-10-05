"""
Persona Loading
===============

Reads the operator's persona file, the YAML document at ``PERSONA_PATH``
(default ``config/persona.yaml``, gitignored; a relative path joins
``PROJECT_ROOT``). The persona is identity configuration: the system's name,
pronouns, companion type, the creator's name and address form, a
self-description, communication-style notes, core values, the Big Five
personality seed and optional core-memory block text. ``PERSONA_KEYS`` lists
the documented keys and their sub-keys; Phase 7's ``PersonaConfig`` schema
keeps the same keys, so a persona written by the wizard loads identically.

- A missing file is not an error: ``load_persona`` returns None (the neutral
  core blocks apply) and logs one WARNING per path.
- An unparseable file, a top level that is not a mapping, a documented
  section that is not a mapping, or a personality value outside [0, 1]
  raises ValueError naming ``PERSONA_PATH`` or the offending field.
- Unknown keys are kept in the returned mapping and reported in one WARNING;
  ``unknown_persona_keys`` lists them.

PyYAML is imported inside ``load_persona``, never at module import. The
persona holds the operator's identity choices and the creator's name; it is
private configuration and stays out of version control.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from mtc.core.config import Settings, get_settings
from mtc.core.paths import resolve_path

logger = logging.getLogger(__name__)

BIG_FIVE_KEYS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)

# Documented persona keys; a tuple value lists the keys of a nested section.
PERSONA_KEYS: Dict[str, Optional[Tuple[str, ...]]] = {
    "name": None,
    "pronouns": None,
    "companion_type": None,
    "creator": ("name", "address_form"),
    "self_description": None,
    "communication_style": None,
    "core_values": None,
    "personality": BIG_FIVE_KEYS,
    "core_memory": ("identity", "persona", "human"),
}

_WARNED_MISSING: Set[str] = set()


def unknown_persona_keys(persona: Mapping[str, Any]) -> List[str]:
    """Return the keys of ``persona`` outside ``PERSONA_KEYS``, sorted.

    Unknown sub-keys of a documented section are reported as ``section.key``.
    """
    unknown: List[str] = []
    for key, value in persona.items():
        if key not in PERSONA_KEYS:
            unknown.append(str(key))
            continue
        sub_keys = PERSONA_KEYS[key]
        if sub_keys is not None and isinstance(value, Mapping):
            unknown.extend(f"{key}.{sub}" for sub in value if sub not in sub_keys)
    return sorted(unknown)


def _validate(persona: Mapping[str, Any], source: Path) -> None:
    for key, sub_keys in PERSONA_KEYS.items():
        value = persona.get(key)
        if sub_keys is not None and value is not None:
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"persona field {key!r} in {source} (PERSONA_PATH) must be "
                    "a mapping"
                )
    personality = persona.get("personality") or {}
    for trait in BIG_FIVE_KEYS:
        if trait not in personality:
            continue
        raw = personality[trait]
        try:
            seed = float(raw)
        except (TypeError, ValueError):
            seed = float("nan")
        if isinstance(raw, bool) or not 0.0 <= seed <= 1.0:
            raise ValueError(
                f"persona field personality.{trait} in {source} (PERSONA_PATH) "
                f"must be a number in [0, 1], got {raw!r}"
            )
    values = persona.get("core_values")
    if values is not None and not isinstance(values, (str, list, tuple)):
        raise ValueError(
            f"persona field core_values in {source} (PERSONA_PATH) must be a "
            "list or a comma-separated string"
        )


def load_persona(
    path: Optional[Union[str, Path]] = None,
    *,
    settings: Optional[Settings] = None,
) -> Optional[Dict[str, Any]]:
    """Return the persona mapping, or None when the file does not exist.

    ``path`` overrides ``PERSONA_PATH``, which is read at call time.
    """
    cfg = settings if settings is not None else get_settings()
    source = resolve_path(path if path is not None else cfg.PERSONA_PATH, settings=cfg)
    if not source.is_file():
        key = str(source)
        if key not in _WARNED_MISSING:
            _WARNED_MISSING.add(key)
            logger.warning(
                "persona file %s (PERSONA_PATH) not found; the neutral core "
                "blocks apply",
                source,
            )
        return None

    import yaml

    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"persona file {source} (PERSONA_PATH) could not be parsed: {exc}"
        ) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"persona file {source} (PERSONA_PATH) must hold a mapping at the "
            f"top level, got {type(loaded).__name__}"
        )
    _validate(loaded, source)
    unknown = unknown_persona_keys(loaded)
    if unknown:
        logger.warning(
            "persona file %s (PERSONA_PATH) has keys outside the documented "
            "schema, kept as given: %s",
            source,
            ", ".join(unknown),
        )
    return loaded
