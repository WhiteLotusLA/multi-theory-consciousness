"""
Dream Engine
============

Builds the prompt for one dream: it picks a dream type from a weighted
profile, fills that type's template from the dream context, adds grounding
lines (recent topics, an unresolved question, an emotional undertone) and a
nap or night texture line, and returns the generation parameters. It never
calls a model; the sleep scheduler sends the prompt through the brain seam.

Configuration comes from ``config_path`` when one is passed (``source``
``configured``), else from the shipped ``config/dreams.yaml`` next to the code
or under ``PROJECT_ROOT`` (``file``), else from its packaged copy
(``package``, through ``mtc.core.resources``, for an install without a
checkout). The file holds:

- ``profiles.default`` is used when no developmental stage is given.
- ``profiles.bands`` map an inclusive stage range to type weights and to a
  temperature and token budget. Developmental accounts of dreaming describe
  dream complexity tracking cognitive capacity; the bands encode that as a
  configurable lean, not as a measured fact about the system.
- ``templates`` hold one neutral prompt per dream type; ``fillers`` stand in
  for ``{hypothetical}`` only when the context has no question or topic for
  it. ``{dominant_emotion}`` has no filler: a type whose template uses it
  asserts a feeling, so it is drawn only for a context with a measured emotion
  (its weight is zero for a draw without one) and an explicit request for it
  without an emotion raises ValueError. A grounding line is left out when the
  context has no value for it.
- ``textures`` scale the token budget for a nap or a night dream. The texture
  is a qualitative lean (naps brief and fragmentary, nights fuller), not a
  sleep-stage claim.

Selection halves the weight of any type used in the last ``repeat_window``
dreams. The comfort bias is an optional welfare setting, off by default: when
on, the comforting types join the candidates (at the profile's mean weight)
and their weights are multiplied, so they dominate without locking the other
types out.

Persisted state (``to_state_dict``): the bounded history of recent dream
types. The module also holds ``trim_to_sentence_end`` and
``dream_sentence_allowance``, used when a dream reaches its token cap.

PyYAML is imported lazily when a configuration file is loaded.
"""

import logging
import random
import re
import string
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Deque, Dict, FrozenSet, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "dreams.yaml"

DREAM_MODES = ("night", "nap")
TEMPLATE_PLACEHOLDERS = frozenset({"dominant_emotion", "hypothetical"})
# A template using this placeholder asserts a feeling: its dream type needs a
# measured emotion and has no filler.
EMOTION_PLACEHOLDER = "dominant_emotion"
_FILLER_KEYS = {
    "hypothetical": frozenset(),
    "hypothetical_from_topic": frozenset({"topic"}),
}
_GROUNDING_KEYS = {
    "topics": frozenset({"topics"}),
    "question": frozenset({"question"}),
    "emotion": frozenset({"emotion"}),
}
MAX_GROUNDING_TOPICS = 3


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class DreamProfile:
    """Type weights and generation parameters for one profile."""

    weights: Dict[str, float]
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class StageBand:
    """A profile that applies to developmental stages ``low`` to ``high``."""

    low: int
    high: int
    weights: Dict[str, float]
    temperature: float
    max_tokens: int

    def contains(self, stage: int) -> bool:
        return self.low <= stage <= self.high


@dataclass(frozen=True)
class DreamTexture:
    """Token scaling and the prompt line for a nap or a night dream."""

    token_scale: float
    min_tokens: int
    suffix: str


@dataclass(frozen=True)
class DreamEngineConfig:
    """The parsed contents of a dreams configuration file."""

    default: DreamProfile
    bands: Tuple[StageBand, ...]
    templates: Dict[str, str]
    fillers: Dict[str, str]
    grounding: Dict[str, str]
    textures: Dict[str, DreamTexture]
    system_prompt: str
    closing_instruction: str
    history_size: int = 20
    repeat_window: int = 3
    repeat_penalty: float = 0.5
    comforting_types: Tuple[str, ...] = ()
    comfort_multiplier: float = 12.0
    # The dream types whose template uses {dominant_emotion}: drawn and built
    # only for a context with a measured emotion.
    emotion_types: FrozenSet[str] = frozenset()
    # Where the file came from: configured, file or package; empty when the
    # config was parsed from a mapping.
    source: str = ""


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if not isinstance(mapping, Mapping) or key not in mapping:
        raise ValueError(f"dreams config: missing '{key}' in {where}")
    return mapping[key]


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"dreams config: {where} must be a non-empty string")
    return value.strip()


def _fields_of(text: str) -> set:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def _check_fields(text: str, allowed: frozenset, where: str) -> str:
    unknown = _fields_of(text) - allowed
    if unknown:
        raise ValueError(
            f"dreams config: {where} uses unknown placeholders {sorted(unknown)}"
        )
    return text


def _weights(raw: Any, where: str) -> Dict[str, float]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"dreams config: {where} weights must be a non-empty map")
    weights: Dict[str, float] = {}
    for name, value in raw.items():
        weight = float(value)
        if not weight > 0.0:
            raise ValueError(f"dreams config: weight of {name} in {where} must be > 0")
        weights[str(name)] = weight
    return weights


def _parameters(raw: Any, where: str) -> Tuple[float, int]:
    temperature = float(_require(raw, "temperature", where))
    max_tokens = int(_require(raw, "max_tokens", where))
    if temperature < 0.0 or max_tokens < 1:
        raise ValueError(f"dreams config: invalid parameters in {where}")
    return temperature, max_tokens


def parse_dream_config(raw: Mapping[str, Any]) -> DreamEngineConfig:
    """Validate a loaded mapping and build a ``DreamEngineConfig``.

    Raises ValueError naming the first problem: a missing key, a dream type
    without a template, an unknown template placeholder, overlapping stage
    bands, a missing nap or night texture, or a ``fillers.dominant_emotion``
    entry (a dream names an emotion only when one was measured, so nothing
    stands in for it).
    """
    profiles = _require(raw, "profiles", "the file")
    default_raw = _require(profiles, "default", "profiles")
    temperature, max_tokens = _parameters(
        _require(default_raw, "parameters", "profiles.default"), "profiles.default"
    )
    default = DreamProfile(
        weights=_weights(
            _require(default_raw, "weights", "profiles.default"), "profiles.default"
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    )

    bands: List[StageBand] = []
    for index, band_raw in enumerate(profiles.get("bands") or []):
        where = f"profiles.bands[{index}]"
        stages = _require(band_raw, "stages", where)
        if not isinstance(stages, (list, tuple)) or len(stages) != 2:
            raise ValueError(f"dreams config: {where}.stages must be [low, high]")
        low, high = int(stages[0]), int(stages[1])
        if low > high:
            raise ValueError(f"dreams config: {where}.stages low exceeds high")
        temperature, max_tokens = _parameters(
            _require(band_raw, "parameters", where), where
        )
        bands.append(
            StageBand(
                low=low,
                high=high,
                weights=_weights(_require(band_raw, "weights", where), where),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
    bands.sort(key=lambda band: band.low)
    for earlier, later in zip(bands, bands[1:]):
        if later.low <= earlier.high:
            raise ValueError("dreams config: stage bands overlap")

    templates_raw = _require(raw, "templates", "the file")
    if not isinstance(templates_raw, Mapping) or not templates_raw:
        raise ValueError("dreams config: templates must be a non-empty map")
    templates = {
        str(name): _check_fields(
            _text(text, f"templates.{name}"),
            TEMPLATE_PLACEHOLDERS,
            f"templates.{name}",
        )
        for name, text in templates_raw.items()
    }

    selection = raw.get("selection") or {}
    comforting = tuple(str(name) for name in selection.get("comforting_types", []))
    used = set(default.weights) | set(comforting)
    for band in bands:
        used |= set(band.weights)
    missing = sorted(used - set(templates))
    if missing:
        raise ValueError(f"dreams config: no template for dream types {missing}")

    emotion_types = frozenset(
        name
        for name, text in templates.items()
        if EMOTION_PLACEHOLDER in _fields_of(text)
    )

    fillers_raw = _require(raw, "fillers", "the file")
    if isinstance(fillers_raw, Mapping) and EMOTION_PLACEHOLDER in fillers_raw:
        raise ValueError(
            f"dreams config: fillers.{EMOTION_PLACEHOLDER} is not accepted: a "
            "dream names an emotion only when one was measured, and a type whose "
            "template uses it is not drawn without one"
        )
    fillers = {
        key: _check_fields(
            _text(_require(fillers_raw, key, "fillers"), f"fillers.{key}"),
            allowed,
            f"fillers.{key}",
        )
        for key, allowed in _FILLER_KEYS.items()
    }
    grounding_raw = _require(raw, "grounding", "the file")
    grounding = {
        key: _check_fields(
            _text(_require(grounding_raw, key, "grounding"), f"grounding.{key}"),
            allowed,
            f"grounding.{key}",
        )
        for key, allowed in _GROUNDING_KEYS.items()
    }

    textures_raw = _require(raw, "textures", "the file")
    textures: Dict[str, DreamTexture] = {}
    for mode in DREAM_MODES:
        texture_raw = _require(textures_raw, mode, "textures")
        scale = float(_require(texture_raw, "token_scale", f"textures.{mode}"))
        floor = int(texture_raw.get("min_tokens", 1))
        if not scale > 0.0 or floor < 1:
            raise ValueError(f"dreams config: invalid scaling in textures.{mode}")
        textures[mode] = DreamTexture(
            token_scale=scale,
            min_tokens=floor,
            suffix=_text(
                _require(texture_raw, "suffix", f"textures.{mode}"),
                f"textures.{mode}.suffix",
            ),
        )

    history_size = int(selection.get("history_size", 20))
    repeat_window = int(selection.get("repeat_window", 3))
    repeat_penalty = float(selection.get("repeat_penalty", 0.5))
    multiplier = float(selection.get("comfort_multiplier", 12.0))
    if history_size < 1 or repeat_window < 0 or not 0.0 < repeat_penalty <= 1.0:
        raise ValueError("dreams config: invalid selection settings")
    if not multiplier > 0.0:
        raise ValueError("dreams config: comfort_multiplier must be > 0")

    return DreamEngineConfig(
        default=default,
        bands=tuple(bands),
        templates=templates,
        fillers=fillers,
        grounding=grounding,
        textures=textures,
        system_prompt=_text(_require(raw, "system_prompt", "the file"), "system"),
        closing_instruction=_text(
            _require(raw, "closing_instruction", "the file"), "closing_instruction"
        ),
        history_size=history_size,
        repeat_window=repeat_window,
        repeat_penalty=repeat_penalty,
        comforting_types=comforting,
        comfort_multiplier=multiplier,
        emotion_types=emotion_types,
    )


def _dream_config_file(path: Optional[Path]) -> Tuple[Path, str]:
    if path is not None:
        return Path(path), "configured"
    default = Path(DEFAULT_CONFIG_PATH)
    if default.is_file():
        return default, "file"
    from mtc.core.resources import resolve_shipped

    try:
        found, where = resolve_shipped(f"config/{default.name}")
    except FileNotFoundError:
        return default, "file"
    return found, ("file" if where == "project" else "package")


def load_dream_config(path: Optional[Path] = None) -> DreamEngineConfig:
    """Read and validate a dreams YAML file (the shipped file by default)."""
    import yaml

    source, origin = _dream_config_file(path)
    if not source.is_file():
        raise FileNotFoundError(f"dreams config not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"dreams config {source} is not a mapping")
    return replace(parse_dream_config(raw), source=origin)


# ----------------------------------------------------------------------------
# Context and prompt
# ----------------------------------------------------------------------------


@dataclass
class DreamContext:
    """What a dream is grounded in.

    ``recent_topics`` are short themes (gists rather than verbatim text),
    ``dominant_emotion`` a single affect word or None when nothing is known,
    ``unresolved_question`` an open question or goal or None, and ``mode``
    either ``night`` or ``nap``.
    """

    recent_topics: List[str] = field(default_factory=list)
    dominant_emotion: Optional[str] = None
    unresolved_question: Optional[str] = None
    mode: str = "night"

    def __post_init__(self) -> None:
        if self.mode not in DREAM_MODES:
            raise ValueError(f"mode must be one of {DREAM_MODES}, got {self.mode!r}")


@dataclass(frozen=True)
class DreamPrompt:
    """A ready request: prompts, sampling temperature, token cap and type."""

    system_prompt: str
    prompt: str
    temperature: float
    max_tokens: int
    dream_type: str


def _clean(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


class DreamEngine:
    """Weighted dream-type selection and prompt building from configuration."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        stage: Optional[int] = None,
        comfort_bias: bool = False,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.config = load_dream_config(config_path)
        self.comfort_bias = bool(comfort_bias)
        self._rng = rng if rng is not None else random.Random()
        self._history: Deque[str] = deque(maxlen=self.config.history_size)
        self._stage: Optional[int] = None
        self.set_stage(stage)

    # -- stage --------------------------------------------------------------

    @property
    def stage(self) -> Optional[int]:
        return self._stage

    def set_stage(self, stage: Optional[int]) -> None:
        """Set the developmental stage, clamped to the configured bands.

        None selects the default profile. A stage that falls in a gap between
        bands raises ValueError.
        """
        if stage is None:
            self._stage = None
            return
        bands = self.config.bands
        if not bands:
            raise ValueError("dreams config defines no stage bands")
        clamped = max(bands[0].low, min(bands[-1].high, int(stage)))
        if not any(band.contains(clamped) for band in bands):
            raise ValueError(f"no dream profile covers stage {clamped}")
        self._stage = clamped

    def _profile_name(self) -> str:
        if self._stage is None:
            return "profiles.default"
        for index, band in enumerate(self.config.bands):
            if band.contains(self._stage):
                return f"profiles.bands[{index}] (stages {band.low}-{band.high})"
        return f"stage {self._stage}"

    def needs_emotion(self, dream_type: str) -> bool:
        """Whether the dream type's template asserts a feeling."""
        return dream_type in self.config.emotion_types

    def _profile(self) -> Tuple[Dict[str, float], float, int]:
        if self._stage is None:
            default = self.config.default
            return default.weights, default.temperature, default.max_tokens
        for band in self.config.bands:
            if band.contains(self._stage):
                return band.weights, band.temperature, band.max_tokens
        raise ValueError(f"no dream profile covers stage {self._stage}")

    def parameters(self) -> Dict[str, Any]:
        """Temperature and token budget of the active profile, before texture."""
        _, temperature, max_tokens = self._profile()
        return {"temperature": temperature, "max_tokens": max_tokens}

    # -- selection ----------------------------------------------------------

    @property
    def recent_types(self) -> List[str]:
        return list(self._history)

    def type_weights(
        self,
        comfort_bias: Optional[bool] = None,
        *,
        emotion_available: bool = True,
    ) -> Dict[str, float]:
        """Current selection weights after repeat avoidance and comfort bias.

        ``comfort_bias`` None uses the engine's setting. ``emotion_available``
        False (a context with no measured emotion; ``build_prompt`` passes it)
        gives every type whose template uses ``{dominant_emotion}`` the weight
        zero for this draw.
        """
        bias = self.comfort_bias if comfort_bias is None else bool(comfort_bias)
        profile, _, _ = self._profile()
        weights = dict(profile)
        comforting = self.config.comforting_types
        if bias and weights:
            mean = sum(weights.values()) / len(weights)
            for name in comforting:
                weights.setdefault(name, mean)
        window = self.config.repeat_window
        recent = set(list(self._history)[-window:]) if window else set()
        for name in weights:
            if name in recent:
                weights[name] *= self.config.repeat_penalty
        if bias:
            for name in comforting:
                if name in weights:
                    weights[name] *= self.config.comfort_multiplier
        if not emotion_available:
            for name in weights:
                if self.needs_emotion(name):
                    weights[name] = 0.0
        return weights

    def select_dream_type(
        self,
        comfort_bias: Optional[bool] = None,
        *,
        emotion_available: bool = True,
    ) -> str:
        """Draw one dream type and record it in the repeat-avoidance history.

        ``emotion_available`` is passed to ``type_weights``. When no type keeps
        any weight for the draw, ValueError names the profile and nothing is
        recorded.
        """
        weights = self.type_weights(comfort_bias, emotion_available=emotion_available)
        names = [name for name, weight in weights.items() if weight > 0.0]
        if not names:
            raise ValueError(
                f"no dream type of {self._profile_name()} can be drawn: every "
                "type left needs a measured emotion and the context has none"
            )
        chosen = self._rng.choices(names, weights=[weights[n] for n in names], k=1)[0]
        self._history.append(chosen)
        return chosen

    # -- prompt -------------------------------------------------------------

    def build_prompt(
        self,
        context: DreamContext,
        *,
        dream_type: Optional[str] = None,
        comfort_bias: Optional[bool] = None,
    ) -> DreamPrompt:
        """Build the prompt and parameters for one dream.

        A ``dream_type`` passed in is used as given and does not touch the
        selection history; otherwise one type is drawn, and a context with no
        measured emotion never draws a type whose template uses
        ``{dominant_emotion}``. Such a type passed explicitly without an
        emotion raises ValueError. The token cap is the profile's budget
        scaled by the context's texture.
        """
        emotion = _clean(context.dominant_emotion)
        if dream_type is None:
            dream_type = self.select_dream_type(
                comfort_bias, emotion_available=emotion is not None
            )
        template = self.config.templates.get(dream_type)
        if template is None:
            raise ValueError(f"unknown dream type {dream_type!r}")
        if emotion is None and self.needs_emotion(dream_type):
            raise ValueError(f"dream type {dream_type!r} needs a measured emotion")
        fillers = self.config.fillers
        grounding = self.config.grounding

        topics = [
            topic
            for topic in (_clean(item) for item in context.recent_topics or [])
            if topic
        ][:MAX_GROUNDING_TOPICS]
        question = _clean(context.unresolved_question)
        if question:
            hypothetical = question
        elif topics:
            hypothetical = fillers["hypothetical_from_topic"].format(topic=topics[0])
        else:
            hypothetical = fillers["hypothetical"]

        parts = [template.format(dominant_emotion=emotion, hypothetical=hypothetical)]
        if topics:
            parts.append(grounding["topics"].format(topics=", ".join(topics)))
        if question:
            parts.append(grounding["question"].format(question=question))
        if emotion:
            parts.append(grounding["emotion"].format(emotion=emotion))
        texture = self.config.textures[context.mode]
        parts.append(texture.suffix)
        parts.append(self.config.closing_instruction)

        _, temperature, base_tokens = self._profile()
        max_tokens = max(texture.min_tokens, int(base_tokens * texture.token_scale))
        logger.debug(
            "dream prompt built: type=%s stage=%s mode=%s",
            dream_type,
            self._stage,
            context.mode,
        )
        return DreamPrompt(
            system_prompt=self.config.system_prompt,
            prompt="\n".join(parts),
            temperature=temperature,
            max_tokens=max_tokens,
            dream_type=dream_type,
        )

    # -- persistence --------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Export the recent dream types for the persistence hub."""
        return {
            "scalars": {},
            "arrays": {},
            "histories": {"dream_types": list(self._history)},
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the recent dream types into a deque of the configured size."""
        history = (state or {}).get("histories", {}).get("dream_types", [])
        self._history = deque(
            (str(name) for name in history), maxlen=self.config.history_size
        )


# ----------------------------------------------------------------------------
# Sentence trimming for capped dreams
# ----------------------------------------------------------------------------

SENTENCE_ALLOWANCE_MIN = 16
SENTENCE_ALLOWANCE_MAX = 64
SENTENCE_ALLOWANCE_DIVISOR = 8
TRIM_MAX_CHARS = 320
TRIM_MIN_KEPT_FRACTION = 0.5

TRIM_COMPLETE = "complete"
TRIM_TRIMMED = "trimmed"
TRIM_NO_BOUNDARY = "no_boundary"
TRIM_REFUSED = "refused"

# Terminal punctuation is not a sentence end right after one of these
# abbreviations or after a single capital initial.
_ABBREVIATIONS = (
    "Mr",
    "Mrs",
    "Ms",
    "Dr",
    "St",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "a.m",
    "p.m",
)
_NOT_AFTER = "".join(rf"(?<!\b{re.escape(word)})" for word in _ABBREVIATIONS)
_NOT_AFTER += r"(?<!\b[A-Z])"
# Terminal punctuation, then any closing quote, bracket or emphasis mark, then
# the end of the text or whitespace and the opener of a new sentence (a
# capital, an opening quote, a bracket or an emphasis mark). A lowercase
# continuation is not an end.
_SENTENCE_END = re.compile(
    _NOT_AFTER
    + r"[.!?\u2026]+"
    + r"[\"'\u201d\u2019)\]*_]*"
    + r"(?=\s+[A-Z\"'\u201c\u2018(\[*_]|\s*$)"
)
# A last line holding only a list marker, left behind by a cut.
_DANGLING_MARKER = re.compile(r"\n\s*(?:\d+[.)]|[-*])\s*$")


@dataclass(frozen=True)
class TrimResult:
    """The kept text, how many characters were removed, and why.

    ``result`` is ``complete`` (already ends at a sentence end), ``trimmed``
    (a tail was removed), ``no_boundary`` (no sentence end at all) or
    ``refused`` (the cut would remove more than ``TRIM_MAX_CHARS`` or keep
    less than ``TRIM_MIN_KEPT_FRACTION`` of the text). Every result except
    ``trimmed`` returns the text unchanged with 0 removed.
    """

    text: str
    trimmed_chars: int
    result: str


def trim_to_sentence_end(text: str) -> TrimResult:
    """Cut a text that stopped mid-sentence back to its last sentence end.

    The refusal guard keeps a false sentence end from turning a dream into a
    fragment: at worst a dream loses its final partial sentence. Nothing is
    ever emptied.
    """
    if not text:
        return TrimResult(text, 0, TRIM_NO_BOUNDARY)
    last = None
    for match in _SENTENCE_END.finditer(text):
        last = match
    if last is None:
        return TrimResult(text, 0, TRIM_NO_BOUNDARY)
    kept = _DANGLING_MARKER.sub("", text[: last.end()])
    removed = len(text) - len(kept)
    if removed == 0:
        return TrimResult(text, 0, TRIM_COMPLETE)
    if removed > TRIM_MAX_CHARS or len(kept) < TRIM_MIN_KEPT_FRACTION * len(text):
        return TrimResult(text, 0, TRIM_REFUSED)
    return TrimResult(kept, removed, TRIM_TRIMMED)


def dream_sentence_allowance(max_tokens: int) -> int:
    """Tokens requested past a dream's cap so the sentence in flight can close.

    An eighth of the cap, at least ``SENTENCE_ALLOWANCE_MIN`` and at most
    ``SENTENCE_ALLOWANCE_MAX``. The cap itself is what the record stores.
    """
    if int(max_tokens) < 1:
        raise ValueError("max_tokens must be at least 1")
    share = int(max_tokens) // SENTENCE_ALLOWANCE_DIVISOR
    return min(SENTENCE_ALLOWANCE_MAX, max(SENTENCE_ALLOWANCE_MIN, share))
