"""Configuration of the continuous thought machine.

``CTMConfig`` holds every tunable of the background thought loop: the cadence
per consciousness state, the strategy weights for waking and dreaming thoughts,
the novelty thresholds, the expression and contention windows, the gist
schedule, the LLM-form limits, the resource guard, the buffer and surfacing
sizes and the theme keyword map. The defaults live in the model; the shipped
``config/ctm/ctm.yaml`` repeats them with one comment per group and a
``profiles.demo`` overlay for a short run.

``CTMConfig.load`` reads ``ctm.yaml`` from the directory named by
``CTM_CONFIG_DIR`` (relative paths join ``PROJECT_ROOT``; the setting is read at
call time), applies the overlay named by ``CTM_PROFILE`` and validates the
result. When the directory is the shipped default (``CTM_CONFIG_DIR`` left at
``config/ctm`` and no directory passed) and its ``ctm.yaml`` is missing, the
packaged copy is read and ``source`` says ``package`` (an install from a wheel
has no project copy). Otherwise a missing file means the built-in defaults and
``source`` says ``defaults``; an unreadable file, an unknown key or an out-of-range value
raises ``CTMConfigError`` naming the file. The built-in demo overlay applies
when the file is missing or has no ``profiles.demo`` entry. PyYAML is imported
inside ``load``, never at module import. The module holds no thought text and
no private data.

Config keys read: ``CTM_CONFIG_DIR`` and ``CTM_PROFILE``.
"""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mtc.core.config import Settings, get_settings
from mtc.core.paths import resolve_path
from mtc.core.resources import resolve_shipped

logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = "ctm.yaml"

WAKING_STRATEGIES = ("memory", "emotion", "association", "question", "spontaneous")
DREAM_STRATEGIES = ("consolidation", "creative", "emotion", "spontaneous")
STATES = ("awake", "drowsy", "dreaming", "deep_sleep")

# Applied on top of the base values for CTM_PROFILE=demo when the file has no
# ``profiles.demo`` entry of its own.
BUILTIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "demo": {
        "cadence": {"awake": 0.5},
        "max_thoughts_per_minute": 30,
        "gist": {"every": 5},
        "contention_window_s": 0,
    }
}


class CTMConfigError(ValueError):
    """The thought-machine configuration could not be read or is invalid."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Cadence(_Model):
    """Thoughts per second (Hz) for each consciousness state."""

    awake: float = Field(0.008, gt=0, allow_inf_nan=False)
    drowsy: float = Field(0.005, gt=0, allow_inf_nan=False)
    dreaming: float = Field(0.003, gt=0, allow_inf_nan=False)
    deep_sleep: float = Field(0.002, gt=0, allow_inf_nan=False)

    def for_state(self, state: str) -> float:
        """Return the rate for a state; an unknown state raises ValueError."""
        if state not in STATES:
            raise ValueError(f"unknown consciousness state {state!r}")
        return float(getattr(self, state))


class Modulation(_Model):
    """How the measured consciousness level scales the cadence.

    The multiplier is ``base + slope * level`` for a level in [0, 1], refreshed
    every ``refresh_every`` thoughts.
    """

    enabled: bool = True
    base: float = Field(0.4, ge=0, allow_inf_nan=False)
    slope: float = Field(1.2, ge=0, allow_inf_nan=False)
    refresh_every: int = Field(5, ge=1)


class WakingWeights(_Model):
    """Relative weight of each waking strategy."""

    memory: float = Field(0.40, ge=0, allow_inf_nan=False)
    emotion: float = Field(0.25, ge=0, allow_inf_nan=False)
    association: float = Field(0.20, ge=0, allow_inf_nan=False)
    question: float = Field(0.10, ge=0, allow_inf_nan=False)
    spontaneous: float = Field(0.05, ge=0, allow_inf_nan=False)


class DreamWeights(_Model):
    """Relative weight of each dream strategy."""

    consolidation: float = Field(0.50, ge=0, allow_inf_nan=False)
    creative: float = Field(0.30, ge=0, allow_inf_nan=False)
    emotion: float = Field(0.15, ge=0, allow_inf_nan=False)
    spontaneous: float = Field(0.05, ge=0, allow_inf_nan=False)


class ExpressProbability(_Model):
    """Chance in [0, 1] that a thought of a strategy may reach expression.

    The novelty gate still decides; a probability of 0 keeps a strategy quiet.
    Dream strategies stay quiet by default (dream thoughts never surface).
    """

    memory: float = Field(0.5, ge=0, le=1, allow_inf_nan=False)
    emotion: float = Field(0.4, ge=0, le=1, allow_inf_nan=False)
    association: float = Field(0.4, ge=0, le=1, allow_inf_nan=False)
    question: float = Field(0.5, ge=0, le=1, allow_inf_nan=False)
    spontaneous: float = Field(0.3, ge=0, le=1, allow_inf_nan=False)
    consolidation: float = Field(0.0, ge=0, le=1, allow_inf_nan=False)
    creative: float = Field(0.0, ge=0, le=1, allow_inf_nan=False)


class Novelty(_Model):
    """Embedding-similarity gates.

    A thought at or above ``dedup`` similarity to a recent one is a duplicate;
    expression needs a novelty (one minus the top similarity) of at least
    ``express``; ``cache`` is the number of recent thought vectors compared.
    """

    dedup: float = Field(0.85, gt=0, le=1, allow_inf_nan=False)
    express: float = Field(0.5, ge=0, le=1, allow_inf_nan=False)
    cache: int = Field(50, ge=1)


class Gist(_Model):
    """When the machine asks for a gist of its recent thoughts."""

    every: int = Field(50, ge=1)
    cooldown_s: float = Field(60.0, ge=0, allow_inf_nan=False)
    min_new_fraction: float = Field(0.5, ge=0, le=1, allow_inf_nan=False)
    retention: float = Field(0.8, ge=0, le=1, allow_inf_nan=False)


class LLMForm(_Model):
    """Limits of the model-written form of a thought.

    A reply that starts with one of ``placeholder_prefixes`` is a miss, never a
    thought; the list is empty unless a persona defines placeholder text.
    """

    max_tokens: int = Field(150, ge=1)
    temperature: float = Field(0.8, ge=0, allow_inf_nan=False)
    placeholder_prefixes: List[str] = Field(default_factory=list)


class Resource(_Model):
    """The CPU guard: a sample above the limit pauses thought generation."""

    cpu_percent_limit: float = Field(5.0, gt=0, le=100, allow_inf_nan=False)
    check_every: int = Field(10, ge=1)


class Association(_Model):
    """Memory associations attached to a thought."""

    top_k: int = Field(3, ge=1)
    min_score: float = Field(0.5, ge=0, le=1, allow_inf_nan=False)


class SurfacingConfig(_Model):
    """Which thoughts the prompt may see.

    The surfacing read looks at the newest ``max(min_window, limit *
    window_multiplier)`` stored thoughts, so a run of quiet or excluded
    thoughts does not hide an expressible one.
    """

    excluded_type_prefixes: List[str] = Field(default_factory=lambda: ["dream"])
    window_multiplier: int = Field(40, ge=1)
    min_window: int = Field(200, ge=1)


def default_theme_keywords() -> Dict[str, List[str]]:
    return {
        "memory": ["memory", "memories", "remember", "past"],
        "emotion": ["feel", "feeling", "emotion", "emotional"],
        "awareness": ["conscious", "consciousness", "aware", "awareness"],
        "learning": ["learn", "learning", "understand", "understanding"],
        "curiosity": ["wonder", "curious", "curiosity", "question"],
        "time": ["time", "moment", "future", "present"],
        "growth": ["grow", "growing", "change", "changing"],
        "connection": ["together", "connection", "relationship", "shared"],
    }


class CTMConfig(_Model):
    """Every tunable of the continuous thought machine (see the module doc)."""

    cadence: Cadence = Field(default_factory=Cadence)
    jitter: float = Field(0.3, ge=0, lt=1, allow_inf_nan=False)
    min_sleep_s: float = Field(1.0, ge=0, allow_inf_nan=False)
    max_thoughts_per_minute: int = Field(2, ge=1)
    consciousness_modulation: Modulation = Field(default_factory=Modulation)
    strategy_weights: WakingWeights = Field(default_factory=WakingWeights)
    dream_strategy_weights: DreamWeights = Field(default_factory=DreamWeights)
    express_probability: ExpressProbability = Field(default_factory=ExpressProbability)
    novelty: Novelty = Field(default_factory=Novelty)
    contention_window_s: float = Field(120.0, ge=0, allow_inf_nan=False)
    burst_gap_s: float = Field(300.0, ge=0, allow_inf_nan=False)
    buffer_size: int = Field(100, ge=1)
    gist: Gist = Field(default_factory=Gist)
    llm: LLMForm = Field(default_factory=LLMForm)
    resource: Resource = Field(default_factory=Resource)
    association: Association = Field(default_factory=Association)
    theme_keywords: Dict[str, List[str]] = Field(default_factory=default_theme_keywords)
    surfacing: SurfacingConfig = Field(default_factory=SurfacingConfig)
    lifecycle_noise_patterns: List[str] = Field(default_factory=list)

    @field_validator("strategy_weights", "dream_strategy_weights")
    @classmethod
    def _weights_have_a_positive_sum(cls, value: BaseModel) -> BaseModel:
        if sum(value.model_dump().values()) <= 0:
            raise ValueError("strategy weights must not all be zero")
        return value

    @field_validator("lifecycle_noise_patterns")
    @classmethod
    def _patterns_compile(cls, value: List[str]) -> List[str]:
        for pattern in value:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"{pattern!r} is not a regular expression") from exc
        return value

    # The load provenance is runtime state, not a configurable value.
    _source: str = "defaults"
    _path: Optional[Path] = None
    _profile: str = "default"

    @property
    def source(self) -> str:
        """``file`` for the project's ctm.yaml, ``package`` for the packaged
        copy, ``defaults`` otherwise."""
        return self._source

    @property
    def path(self) -> Optional[Path]:
        """The file that was read, or None."""
        return self._path

    @property
    def profile(self) -> str:
        """The profile that was applied."""
        return self._profile

    def waking_weights(self) -> Dict[str, float]:
        """Waking strategy weights as a plain mapping."""
        return self.strategy_weights.model_dump()

    def dream_weights(self) -> Dict[str, float]:
        """Dream strategy weights as a plain mapping."""
        return self.dream_strategy_weights.model_dump()

    @classmethod
    def load(
        cls,
        config_dir: Optional[Any] = None,
        profile: Optional[str] = None,
        *,
        settings: Optional[Settings] = None,
    ) -> "CTMConfig":
        """Read ``ctm.yaml`` and apply a profile (see the module docstring)."""
        cfg = settings if settings is not None else get_settings()
        directory = ctm_config_dir(config_dir, settings=cfg)
        name = profile if profile is not None else cfg.CTM_PROFILE
        path = directory / CONFIG_FILE_NAME
        data: Dict[str, Any] = {}
        profiles: Dict[str, Any] = {}
        source = "defaults"
        if path.is_file():
            data, profiles = _read_file(path)
            source = "file"
        else:
            packaged = shipped_file(CONFIG_FILE_NAME, config_dir, settings=cfg)
            if packaged is not None:
                data, profiles = _read_file(packaged)
                source = "package"
                path = packaged
            else:
                logger.info("%s not found; the built-in CTM defaults apply", path)
        if name != "default":
            overlay = profiles.get(name, BUILTIN_PROFILES.get(name))
            if overlay is None:
                raise CTMConfigError(
                    f"profile {name!r} (CTM_PROFILE) is not defined in {path} "
                    "or built in"
                )
            data = _merge(data, overlay)
        try:
            config = cls.model_validate(data)
        except ValidationError as exc:
            raise CTMConfigError(f"{path}: {_describe(exc)}") from exc
        config._source = source
        config._path = path if source in ("file", "package") else None
        config._profile = name
        return config


def ctm_config_dir(
    config_dir: Optional[Any] = None, *, settings: Optional[Settings] = None
) -> Path:
    """Return the CTM config directory: the argument or ``CTM_CONFIG_DIR``."""
    cfg = settings if settings is not None else get_settings()
    return resolve_path(
        config_dir if config_dir is not None else cfg.CTM_CONFIG_DIR, settings=cfg
    )


def uses_shipped_dir(
    config_dir: Optional[Any] = None, *, settings: Optional[Settings] = None
) -> bool:
    """True when no directory is passed and ``CTM_CONFIG_DIR`` is its default."""
    cfg = settings if settings is not None else get_settings()
    default = Settings.model_fields["CTM_CONFIG_DIR"].default
    return config_dir is None and cfg.CTM_CONFIG_DIR == default


def shipped_file(
    relative: Any,
    config_dir: Optional[Any] = None,
    *,
    settings: Optional[Settings] = None,
) -> Optional[Path]:
    """The packaged copy of a shipped CTM file, or None.

    Only for the shipped default directory, and only when its project copy is
    missing: an operator's own directory is never filled from the package.
    """
    cfg = settings if settings is not None else get_settings()
    if not uses_shipped_dir(config_dir, settings=cfg):
        return None
    default = Settings.model_fields["CTM_CONFIG_DIR"].default
    source = f"{default}/{Path(relative).as_posix()}"
    try:
        path, where = resolve_shipped(source, settings=cfg)
    except FileNotFoundError:
        return None
    return path if where == "package" else None


def _read_file(path: Path):
    import yaml

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise CTMConfigError(f"{path} could not be parsed: {exc}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise CTMConfigError(
            f"{path} must hold a mapping at the top level, got "
            f"{type(loaded).__name__}"
        )
    data = dict(loaded)
    profiles = data.pop("profiles", None) or {}
    if not isinstance(profiles, dict) or not all(
        isinstance(v, dict) for v in profiles.values()
    ):
        raise CTMConfigError(f"{path}: profiles must map a name to a mapping")
    return data, profiles


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``base`` with ``overlay`` merged in; mappings merge, others replace."""
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _describe(error: ValidationError) -> str:
    parts = []
    for item in error.errors():
        where = ".".join(str(p) for p in item["loc"]) or "(top level)"
        parts.append(f"{where}: {item['msg']}")
    return "; ".join(parts)
