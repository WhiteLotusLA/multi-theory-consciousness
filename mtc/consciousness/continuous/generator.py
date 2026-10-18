"""
Thought Generator
=================

Composes one background thought at a time from the seed packs
(``mtc.consciousness.continuous.seeds``), grounded in what the injected sources
can actually supply. Every template, slot word, concept and memory query term
comes from the seed library; this module holds no thought text of its own and
never calls a language model (the machine's enhancer does that, behind the
contention guard).

Strategies. A waking thought (``awake`` or ``drowsy``) is chosen by the
``strategy_weights`` of the CTM config (memory 0.40, emotion 0.25,
association 0.20, question 0.10, spontaneous 0.05); a dream thought
(``dreaming`` or ``deep_sleep``) by ``dream_strategy_weights`` (consolidation
0.50, creative 0.30, emotion 0.15, spontaneous 0.05). After three thoughts of
one waking type in a row the memory and emotion weights drop to zero for the
next waking choice, so the stream does not circle one kind of thought (dream
thoughts share one type and are not guarded). A strategy whose
material is missing is skipped for that choice and counted in
``status()["skipped"]``:

- memory needs a ``MemorySource``; a query term from the pack pulls candidate
  memories, those matching ``lifecycle_noise_patterns`` (system chatter such as
  end-of-conversation boilerplate) are dropped, and one is drawn with the
  weights of the ``MemoryWeightRule`` list (a matching rule multiplies a
  memory's weight; a weight lowers its chance and never removes it);
- emotion needs an ``EmotionSource`` with a reading outside the neutral band:
  a reading whose absolute valence is below ``NEUTRAL_PLEASURE_BAND`` (the
  PAD model's band, inside which its label reads Neutral) names no emotion,
  and the strategy is skipped and counted. ``PadEmotionSource`` gives a
  reading only when the PAD state's pleasure and arousal were both computed
  from measured inputs. Valence in [-1, 1] and
  arousal in [0, 1] are named by quadrant of the circumplex model of affect
  (pleasant and active, pleasant and calm, unpleasant and active, unpleasant
  and calm); the thought's ``emotional_context`` holds that name and the
  reading's distance from the neutral centre, and the raw reading goes to
  ``processing_context["affect"]``. An emotion thought, waking or dreaming,
  uses only a template that names the measured emotion (``{current_emotion}``)
  and uses no affect slot (``AFFECT_SLOTS``: ``{emotion}``, ``{feeling}``,
  which the seed loader refuses in every template family), so its text never
  names a feeling the reading did not; with no such template the strategy has
  no material for that choice. An untrained
  emotion bridge has no reading, so no emotion is ever attributed in its place;
- association pairs two plain ideas from the pack; questions and spontaneous
  lines need nothing; the dream families use the pack's dream templates, with a
  memory when one is available and the creative concepts. Every dream-state
  thought has the type ``dream``, which surfacing excludes.

When no strategy can produce text, ``generate_thought`` returns None and
``last_miss_reason`` reads ``no_seeds``; a text matching a noise pattern is
dropped with ``lifecycle_noise``. Intensity follows the state (``awake``
moderate, ``drowsy`` and ``dreaming`` quiet, ``deep_sleep`` a whisper; a strong
affect raises an emotion thought one step). Salience is the memory's
importance (scaled from 0 to 10), the affect's strength, or a per-strategy
design constant, and ``processing_context["salience_basis"]`` says which.
Every generated thought carries ``processing_context["form"]`` ``template``
(its words come from a seed template). ``should_express`` is provisional: it is drawn from ``express_probability``
(``processing_context["express_basis"]`` is ``provisional``), and the machine
replaces it with its novelty decision before anything is broadcast or stored.

Sources and rules. ``MemoryManagerSource`` adapts the memory manager's episodic
text search, read without counting an access (a replay is not a recall; see
its docstring), ``PadEmotionSource`` the PAD model's latest state and
``BridgeEmotionSource`` the emotion bridge (or a callable returning the bridge
of the moment, so a bridge built later is seen). ``MemoryWeightRule.from_settings``
builds the one generic rule from ``SENSITIVE_MEMORY_PHRASES`` and
``SENSITIVE_MEMORY_REPLAY_WEIGHT``.

Clock and randomness are injectable (``clock`` returns an aware UTC datetime,
``rng`` is a ``random.Random``), so a seeded generator is reproducible.
Persisted state (``to_state_dict``): the recent thought types, the per-strategy
counts and the last miss reason; no thought text. Generated thoughts, memory
snippets and affect readings are private runtime data.

Soft failure is loud. A memory or emotion source that raises is counted by
source name in ``status()["source_errors"]``, logged at WARNING the first time
and every hundredth time, and treated as no material for that thought; when
``Settings.DEBUG`` is true the exception is re-raised instead, so the machine's
``generate`` step counts it and raises in turn.

Config keys read: ``SENSITIVE_MEMORY_PHRASES`` and
``SENSITIVE_MEMORY_REPLAY_WEIGHT`` (through ``MemoryWeightRule.from_settings``)
and ``DEBUG`` (from the passed Settings, else the process settings).
"""

import logging
import math
import random
import re
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from mtc.consciousness.continuous.config import CTMConfig
from mtc.consciousness.continuous.schemas import (
    CONSCIOUSNESS_STATES,
    ThoughtIntensity,
    ThoughtRecord,
    ThoughtType,
    new_thought_id,
)
from mtc.consciousness.continuous.seeds import (
    AFFECT_SLOTS,
    SLOT_KEYS,
    SeedError,
    SeedLibrary,
    template_fields,
)
from mtc.consciousness.pad_affect import NEUTRAL_PLEASURE_BAND, measured_dimensions
from mtc.core.config import Settings, csv_list, get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "AFFECT_QUADRANTS",
    "AFFECT_SLOTS",
    "MEASURED_SALIENCE_BASES",
    "NEUTRAL_PLEASURE_BAND",
    "BridgeEmotionSource",
    "EmotionSource",
    "MemoryManagerSource",
    "MemorySource",
    "MemoryWeightRule",
    "PadEmotionSource",
    "ThoughtGenerator",
    "name_affect",
]

DREAM_STATES = ("dreaming", "deep_sleep")

WAKING_TEMPLATE_KEYS: Dict[str, str] = {
    "memory": "memory_templates",
    "emotion": "emotion_templates",
    "association": "association_templates",
    "question": "questions",
    "spontaneous": "spontaneous_lines",
}
DREAM_TEMPLATE_KEYS: Dict[str, str] = {
    "consolidation": "dream_consolidation_templates",
    "creative": "dream_creative_templates",
    "emotion": "dream_emotion_templates",
    "spontaneous": "dream_spontaneous_templates",
}
WAKING_TYPES: Dict[str, ThoughtType] = {
    "memory": ThoughtType.MEMORY_REPLAY,
    "emotion": ThoughtType.EMOTION,
    "association": ThoughtType.ASSOCIATION,
    "question": ThoughtType.QUESTION,
    "spontaneous": ThoughtType.SPONTANEOUS,
}

INTENSITY_BY_STATE: Dict[str, ThoughtIntensity] = {
    "awake": ThoughtIntensity.MODERATE,
    "drowsy": ThoughtIntensity.QUIET,
    "dreaming": ThoughtIntensity.QUIET,
    "deep_sleep": ThoughtIntensity.WHISPER,
}
_INTENSITY_ORDER = (
    ThoughtIntensity.WHISPER,
    ThoughtIntensity.QUIET,
    ThoughtIntensity.MODERATE,
    ThoughtIntensity.STRONG,
    ThoughtIntensity.URGENT,
)

# The salience bases that are measurements (a memory's stored importance, the
# affect reading's strength); every other basis is a design constant.
MEASURED_SALIENCE_BASES: Tuple[str, ...] = ("affect", "memory_importance")

# Design constants (not measurements) for strategies with no measured basis.
STRATEGY_SALIENCE: Dict[str, float] = {
    "association": 0.4,
    "question": 0.5,
    "spontaneous": 0.3,
    "consolidation": 0.4,
    "creative": 0.4,
    "memory": 0.5,
    "emotion": 0.4,
}

# Circumplex quadrants: (valence >= 0, arousal >= 0.5) -> emotion name.
AFFECT_QUADRANTS: Dict[Tuple[bool, bool], str] = {
    (True, True): "excitement",
    (True, False): "contentment",
    (False, True): "tension",
    (False, False): "sadness",
}
STRONG_AFFECT = 0.6
# AFFECT_SLOTS (imported from the seed loader) are the slot names that would
# name a feeling at random: the loader refuses them, and an emotion thought
# never takes a template that uses one.

ANTI_REPEAT_RUN = 3
ANTI_REPEAT_ZEROED = ("memory", "emotion")
MEMORY_SEARCH_LIMIT = 5
MEMORY_MIN_IMPORTANCE = 0.0
MEMORY_SNIPPET_CHARS = 120
RECENT_TYPES_KEPT = 10
# A failing source is logged at WARNING the first time and every this many.
FAILURE_LOG_EVERY = 100


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _nameable(affect: Optional[Mapping[str, float]]) -> bool:
    """Whether a reading lies outside the neutral band, so an emotion is named.

    The band is the PAD model's ``NEUTRAL_PLEASURE_BAND`` (inside it the PAD
    label reads Neutral): a reading whose absolute valence is below it names
    no emotion.
    """
    return affect is not None and abs(affect["valence"]) >= NEUTRAL_PLEASURE_BAND


def name_affect(valence: float, arousal: float) -> Tuple[str, float]:
    """Name an affect reading by circumplex quadrant; return (name, strength).

    Strength is the reading's distance from the neutral centre (valence 0,
    arousal 0.5), scaled to [0, 1].
    """
    v = max(-1.0, min(1.0, float(valence)))
    a = max(0.0, min(1.0, float(arousal)))
    name = AFFECT_QUADRANTS[(v >= 0.0, a >= 0.5)]
    strength = math.sqrt(v * v + (2.0 * a - 1.0) ** 2) / math.sqrt(2.0)
    return name, round(min(1.0, strength), 3)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@runtime_checkable
class MemorySource(Protocol):
    """Searchable memories (private content) the memory strategy replays."""

    async def search(
        self, query: str, limit: int, min_importance: float
    ) -> List[Dict[str, Any]]:
        """Return memories as dicts with ``content`` and optionally ``id`` and
        ``importance`` (0 to 10)."""
        ...


@runtime_checkable
class EmotionSource(Protocol):
    """The current affect, or None when nothing measures it."""

    def get_affect(self) -> Optional[Dict[str, float]]:
        """Return ``{"valence": [-1, 1], "arousal": [0, 1]}`` or None."""
        ...


class MemoryManagerSource:
    """``MemorySource`` over the memory manager's episodic full-text search,
    read without counting an access.

    A background replay is not a recall. The manager's ``search_memories``
    touches every hit (one more ``accessed_count`` and a new
    ``last_accessed``), and the pruner and the consolidation pass weigh those,
    so replaying through it would make every replayed memory look used. The
    source therefore calls ``search_text`` on the manager's episodic store
    (``manager.episodic``, the ``EpisodicStore`` protocol's search, which
    touches nothing) and returns ``id``, ``content``, ``importance`` and
    ``memory_type`` of each row. A manager without such a store is searched
    through ``search_memories``, which does count the access;
    ``touches_access`` says which read is in use and the first fallback read
    is logged. The manager's initialization (schema and core blocks) is the
    runtime's job and this read does not trigger it; a store that cannot
    answer raises, and the generator counts that as a memory source failure.
    """

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self._fallback_logged = False

    def _store_search(self) -> Optional[Callable[..., Any]]:
        store = getattr(self.manager, "episodic", None)
        search = getattr(store, "search_text", None)
        return search if callable(search) else None

    @property
    def touches_access(self) -> bool:
        """Whether a replay read counts an access (the fallback read)."""
        return self._store_search() is None

    async def search(
        self, query: str, limit: int, min_importance: float
    ) -> List[Dict[str, Any]]:
        search_text = self._store_search()
        if search_text is not None:
            rows = await search_text(query, limit=limit, min_importance=min_importance)
            return [
                {
                    "id": row.get("id"),
                    "content": row.get("content"),
                    "importance": row.get("importance"),
                    "memory_type": row.get("memory_type"),
                }
                for row in rows or []
            ]
        if not self._fallback_logged:
            self._fallback_logged = True
            logger.info(
                "memory replay reads through search_memories, which counts each "
                "hit as an access"
            )
        rows = await self.manager.search_memories(
            query, limit=limit, min_importance=min_importance
        )
        return [dict(row) for row in rows or []]


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _affect(valence: Any, arousal: Any) -> Optional[Dict[str, float]]:
    v, a = _finite(valence), _finite(arousal)
    if v is None or a is None:
        return None
    return {"valence": max(-1.0, min(1.0, v)), "arousal": max(0.0, min(1.0, a))}


_PAD_READING_DIMENSIONS = frozenset({"pleasure", "arousal"})


class PadEmotionSource:
    """``EmotionSource`` over the PAD model's latest state (pleasure, arousal).

    The latest state is a reading only when both its pleasure and its arousal
    were computed from at least one measured input
    (``mtc.consciousness.pad_affect.measured_dimensions``). A state of unknown
    provenance (no ``measured_inputs``) or with either dimension unmeasured
    gives None: no emotion is named from default inputs.
    """

    def __init__(self, pad_model: Any) -> None:
        self.pad_model = pad_model

    def get_affect(self) -> Optional[Dict[str, float]]:
        latest = self.pad_model.get_latest()
        if latest is None:
            return None
        if not _PAD_READING_DIMENSIONS <= measured_dimensions(latest):
            return None
        return _affect(getattr(latest, "pleasure", None), latest.arousal)


class BridgeEmotionSource:
    """``EmotionSource`` over the emotion bridge's latest subconscious state.

    ``bridge`` is the bridge itself or a zero-argument callable returning the
    bridge of the moment (None while it is not built). The reading is None
    while the bridge's readout is untrained or it has no state yet.
    """

    def __init__(self, bridge: Any) -> None:
        self._bridge = bridge

    def _current(self) -> Any:
        if callable(self._bridge) and not hasattr(self._bridge, "state_history"):
            return self._bridge()
        return self._bridge

    def get_affect(self) -> Optional[Dict[str, float]]:
        bridge = self._current()
        if bridge is None:
            return None
        trained = getattr(bridge, "readout_trained", None)
        if callable(trained):
            trained = trained()
        if not trained:
            return None
        history = getattr(bridge, "state_history", None) or []
        if not history:
            return None
        latest = history[-1]
        if not getattr(latest, "readout_trained", True):
            return None
        return _affect(latest.intuitive_valence, latest.intuitive_arousal)


# ---------------------------------------------------------------------------
# Memory weighting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryWeightRule:
    """Multiply the replay weight of memories containing any of ``phrases``.

    ``weight`` is in (0, 1]: a rule lowers a memory's chance of replay and never
    removes it. Phrases match case-insensitively on word boundaries.
    """

    name: str
    phrases: Tuple[str, ...]
    weight: float

    def __post_init__(self) -> None:
        phrases = tuple(str(p).strip() for p in self.phrases if str(p).strip())
        if not phrases:
            raise ValueError(f"memory weight rule {self.name!r} needs a phrase")
        weight = float(self.weight)
        if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
            raise ValueError(
                f"memory weight rule {self.name!r} weight must be in (0, 1]"
            )
        object.__setattr__(self, "phrases", phrases)
        object.__setattr__(self, "weight", weight)
        pattern = "|".join(rf"\b{re.escape(p)}\b" for p in phrases)
        object.__setattr__(self, "_pattern", re.compile(pattern, re.IGNORECASE))

    def matches(self, text: str) -> bool:
        """Whether the text contains one of the phrases."""
        return bool(self._pattern.search(text or ""))

    @classmethod
    def from_settings(
        cls, settings: Optional[Settings] = None
    ) -> List["MemoryWeightRule"]:
        """The rule from ``SENSITIVE_MEMORY_PHRASES`` and its weight; [] when
        no phrase is set."""
        cfg = settings if settings is not None else get_settings()
        phrases = csv_list(cfg.SENSITIVE_MEMORY_PHRASES)
        if not phrases:
            return []
        return [cls("sensitive", tuple(phrases), cfg.SENSITIVE_MEMORY_REPLAY_WEIGHT)]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclass
class _Draft:
    """The material of one strategy's thought before it becomes a record."""

    strategy: str
    template_key: str
    text: str
    thought_type: ThoughtType
    salience: float
    salience_basis: str
    emotional_context: Dict[str, float]
    processing: Dict[str, Any]
    associated_memories: List[str]
    material: Dict[str, Any]
    strong_affect: bool = False


class _Unavailable(Exception):
    """A strategy's material is missing for this choice."""


class ThoughtGenerator:
    """Builds background thoughts from seed packs and the injected sources."""

    def __init__(
        self,
        seeds: SeedLibrary,
        config: CTMConfig,
        memory_source: Optional[MemorySource] = None,
        emotion_source: Optional[EmotionSource] = None,
        weight_rules: Optional[Sequence[MemoryWeightRule]] = None,
        rng: Optional[random.Random] = None,
        clock: Optional[Callable[[], datetime]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.seeds = seeds
        self.config = config
        self._memory_source = memory_source
        self.emotion_source = emotion_source
        if weight_rules is None:
            weight_rules = MemoryWeightRule.from_settings(settings)
        self.weight_rules: List[MemoryWeightRule] = list(weight_rules)
        self.rng = rng if rng is not None else random.Random()
        self.clock = clock or _utcnow
        self._settings = settings
        self._noise = [re.compile(p) for p in config.lifecycle_noise_patterns]
        self.recent_thought_types: Deque[str] = deque(maxlen=RECENT_TYPES_KEPT)
        self.strategy_counts: Counter = Counter()
        self.skipped: Counter = Counter()
        self.noise_filtered = 0
        self.source_errors: Counter = Counter()
        self.generated = 0
        self.misses = 0
        self.last_miss_reason: Optional[str] = None
        self._last_material: Dict[str, Any] = {}
        self._warned: set = set()

    # -- sources ------------------------------------------------------------

    @property
    def memory_source(self) -> Optional[MemorySource]:
        """The memory source (settable, for wiring after construction)."""
        return self._memory_source

    @memory_source.setter
    def memory_source(self, source: Optional[MemorySource]) -> None:
        self._memory_source = source

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.info(message)

    def _debug(self) -> bool:
        settings = self._settings if self._settings is not None else get_settings()
        return bool(getattr(settings, "DEBUG", False))

    def _source_failure(self, source: str, exc: BaseException) -> None:
        """Count a failed source, log the first and every hundredth, raise in DEBUG."""
        self.source_errors[source] += 1
        count = self.source_errors[source]
        if count == 1 or count % FAILURE_LOG_EVERY == 0:
            logger.warning(
                "%s source failed (%d so far): %s",
                source,
                count,
                type(exc).__name__,
            )
        if self._debug():
            raise exc

    def _read_affect(self) -> Optional[Dict[str, float]]:
        if self.emotion_source is None:
            return None
        try:
            reading = self.emotion_source.get_affect()
        except Exception as exc:  # a failing source measures nothing
            self._source_failure("emotion", exc)
            return None
        if not reading:
            return None
        return _affect(reading.get("valence"), reading.get("arousal"))

    def _is_noise(self, text: str) -> bool:
        return any(pattern.search(text or "") for pattern in self._noise)

    async def _pick_memory(self) -> Optional[Dict[str, Any]]:
        source = self._memory_source
        if source is None:
            return None
        try:
            query = self.seeds.choose("memory_query_terms", self.rng)
        except SeedError:
            self._warn_once("query_terms", "the seed packs have no memory query terms")
            return None
        try:
            rows = await source.search(
                query, MEMORY_SEARCH_LIMIT, MEMORY_MIN_IMPORTANCE
            )
        except Exception as exc:  # an unavailable store replays nothing
            self._source_failure("memory", exc)
            return None
        candidates = []
        for row in rows or []:
            content = row.get("content") if isinstance(row, Mapping) else None
            if not isinstance(content, str) or not content.strip():
                continue
            if self._is_noise(content):
                self.noise_filtered += 1
                continue
            candidates.append(row)
        if not candidates:
            return None
        weights = [self._memory_weight(row["content"]) for row in candidates]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _memory_weight(self, content: str) -> float:
        weight = 1.0
        for rule in self.weight_rules:
            if rule.matches(content):
                weight *= rule.weight
        return weight

    @staticmethod
    def _snippet(content: str) -> str:
        text = " ".join(content.split())
        if len(text) > MEMORY_SNIPPET_CHARS:
            cut = text[:MEMORY_SNIPPET_CHARS].rsplit(" ", 1)[0]
            text = cut or text[:MEMORY_SNIPPET_CHARS]
        return text.rstrip(" .,;:!?")

    @staticmethod
    def _memory_salience(row: Mapping[str, Any]) -> Optional[float]:
        importance = _finite(row.get("importance"))
        if importance is None:
            return None
        return max(0.0, min(1.0, importance / 10.0))

    # -- strategies ---------------------------------------------------------

    def _weights_for(self, dreaming: bool, affect_available: bool) -> Dict[str, float]:
        weights = dict(
            self.config.dream_weights() if dreaming else self.config.waking_weights()
        )
        if not dreaming and self._memory_source is None and weights.get("memory"):
            weights["memory"] = 0.0
            self.skipped["memory"] += 1
            self._warn_once(
                "memory",
                "thought generator has no memory source; memory thoughts are "
                "skipped",
            )
        if weights.get("emotion") and not affect_available:
            weights["emotion"] = 0.0
            self.skipped["emotion"] += 1
            self._warn_once(
                "emotion",
                "thought generator has no affect reading outside the neutral "
                "band; emotion thoughts are skipped",
            )
        recent = list(self.recent_thought_types)[-ANTI_REPEAT_RUN:]
        if (
            not dreaming
            and len(recent) == ANTI_REPEAT_RUN
            and len(set(recent)) == 1
            and recent[0] != ThoughtType.DREAM.value
        ):
            guarded = {
                k: (0.0 if k in ANTI_REPEAT_ZEROED else v) for k, v in weights.items()
            }
            if sum(guarded.values()) > 0:
                weights = guarded
        return weights

    async def _draft(
        self, strategy: str, dreaming: bool, affect: Optional[Dict[str, float]]
    ) -> _Draft:
        rng = self.rng
        seeds = self.seeds
        processing: Dict[str, Any] = {}
        emotional: Dict[str, float] = {}
        associated: List[str] = []
        material: Dict[str, Any] = {"recent_memory": "", "emotions": {}}
        salience = STRATEGY_SALIENCE.get(strategy, 0.4)
        basis = "strategy_default"
        context: Dict[str, str] = {}
        strong = False
        key = (DREAM_TEMPLATE_KEYS if dreaming else WAKING_TEMPLATE_KEYS)[strategy]

        if strategy in ("memory", "consolidation"):
            row = await self._pick_memory()
            if row is None and strategy == "memory":
                raise _Unavailable("no memory")
            if row is not None:
                snippet = self._snippet(row["content"])
                context["memory"] = f'"{snippet}"'
                material["recent_memory"] = snippet
                if row.get("id") is not None:
                    associated.append(str(row["id"]))
                    processing["memory_id"] = str(row["id"])
                measured = self._memory_salience(row)
                if measured is not None:
                    salience, basis = measured, "memory_importance"
        if strategy == "emotion":
            if not _nameable(affect):
                raise _Unavailable("no affect reading outside the neutral band")
            name, strength = name_affect(affect["valence"], affect["arousal"])
            context["current_emotion"] = name
            emotional = {name: strength}
            material["emotions"] = dict(emotional)
            processing["affect"] = dict(affect)
            salience, basis = strength, "affect"
            strong = strength >= STRONG_AFFECT
        if strategy in ("association", "creative"):
            first, second = seeds.choose("concept_pairs", rng)
            context["first"], context["second"] = first, second
            material["recent_topic"] = f"{first} and {second}"
        if strategy == "creative":
            context["concept"] = seeds.choose("creative_concepts", rng)

        template = self._choose_template(key, context, strategy)
        text = seeds.fill(template, rng, **context)
        if dreaming:
            thought_type = ThoughtType.DREAM
        else:
            thought_type = WAKING_TYPES[strategy]
        if affect is not None and strategy != "emotion":
            processing["affect"] = dict(affect)
        return _Draft(
            strategy=strategy,
            template_key=key,
            text=text,
            thought_type=thought_type,
            salience=salience,
            salience_basis=basis,
            emotional_context=emotional,
            processing=processing,
            associated_memories=associated,
            material=material,
            strong_affect=strong,
        )

    def _choose_template(
        self, key: str, context: Mapping[str, str], strategy: str
    ) -> str:
        """Pick a template the context can fill.

        An emotion thought (waking or dreaming) takes only a template that
        names the measured emotion and uses no affect slot word, so no slot
        word names a feeling the reading did not; without one the strategy is
        unavailable for this choice.
        """
        if strategy == "emotion":
            naming = []
            for template in self.seeds.entries(key):
                names = template_fields(template)
                if (
                    "current_emotion" in names
                    and not set(names) & set(AFFECT_SLOTS)
                    and all(n in SLOT_KEYS or n in context for n in names)
                ):
                    naming.append(template)
            if not naming:
                raise _Unavailable(f"no {key} entry names the measured emotion")
            return naming[self.rng.randrange(len(naming))]
        return self.seeds.choose(key, self.rng, available=context.keys())

    def _intensity(self, state: str, strong: bool) -> ThoughtIntensity:
        base = INTENSITY_BY_STATE[state]
        if not strong:
            return base
        index = _INTENSITY_ORDER.index(base)
        return _INTENSITY_ORDER[min(index + 1, len(_INTENSITY_ORDER) - 1)]

    def _provisional_express(self, strategy: str, dreaming: bool) -> bool:
        probabilities = self.config.express_probability.model_dump()
        if dreaming and strategy not in ("consolidation", "creative"):
            return False
        chance = probabilities.get(strategy, 0.0)
        return self.rng.random() < chance

    async def generate_thought(self, state: str = "awake") -> Optional[ThoughtRecord]:
        """Return one thought for the state, or None (see ``last_miss_reason``)."""
        if state not in CONSCIOUSNESS_STATES:
            raise ValueError(
                f"unknown consciousness state {state!r}; expected one of "
                f"{CONSCIOUSNESS_STATES}"
            )
        dreaming = state in DREAM_STATES
        affect = self._read_affect()
        weights = self._weights_for(dreaming, _nameable(affect))
        draft: Optional[_Draft] = None
        while True:
            names = [k for k, v in weights.items() if v > 0]
            if not names:
                break
            strategy = self.rng.choices(
                names, weights=[weights[k] for k in names], k=1
            )[0]
            try:
                draft = await self._draft(strategy, dreaming, affect)
                break
            except (_Unavailable, SeedError) as exc:
                logger.debug("strategy %s produced nothing: %s", strategy, exc)
                weights[strategy] = 0.0
        if draft is None:
            return self._miss("no_seeds")
        if self._is_noise(draft.text):
            self.noise_filtered += 1
            return self._miss("lifecycle_noise")

        processing = dict(draft.processing)
        processing.update(
            {
                "form": "template",
                "strategy": draft.strategy,
                "template_key": draft.template_key,
                "state": state,
                "salience_basis": draft.salience_basis,
                "express_basis": "provisional",
            }
        )
        record = ThoughtRecord(
            thought_id=new_thought_id(),
            timestamp=self.clock(),
            thought_type=draft.thought_type,
            content=draft.text,
            intensity=self._intensity(state, draft.strong_affect),
            salience_score=draft.salience,
            triggered_by=(f"dream_{draft.strategy}" if dreaming else draft.strategy),
            associated_memories=draft.associated_memories,
            emotional_context=draft.emotional_context,
            consciousness_state=state,
            processing_context=processing,
            should_express=self._provisional_express(draft.strategy, dreaming),
        )
        self.recent_thought_types.append(record.thought_type.value)
        self.strategy_counts[draft.strategy] += 1
        self.generated += 1
        self.last_miss_reason = None
        material = dict(draft.material)
        material.setdefault("recent_topic", "")
        self._last_material = material
        return record

    def _miss(self, reason: str) -> Optional[ThoughtRecord]:
        self.misses += 1
        self.last_miss_reason = reason
        self._last_material = {}
        logger.debug("no thought generated: %s", reason)
        return None

    def llm_context(self) -> Dict[str, Any]:
        """The grounded material of the last thought, in the enhancer's keys
        (``recent_memory``, ``emotions``, ``recent_topic``); empty strings and
        mappings where the thought had none."""
        material = self._last_material or {}
        return {
            "recent_memory": material.get("recent_memory", ""),
            "emotions": dict(material.get("emotions") or {}),
            "recent_topic": material.get("recent_topic", ""),
        }

    def status(self) -> Dict[str, Any]:
        """Sources present, counts, skips and the last miss reason."""
        return {
            "memory_source": self._memory_source is not None,
            "emotion_source": self.emotion_source is not None,
            "weight_rules": [rule.name for rule in self.weight_rules],
            "seed_source": self.seeds.source,
            "generated": self.generated,
            "misses": self.misses,
            "last_miss_reason": self.last_miss_reason,
            "by_strategy": dict(self.strategy_counts),
            "skipped": dict(self.skipped),
            "noise_filtered": self.noise_filtered,
            "source_errors": dict(self.source_errors),
        }

    # -- persistence --------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Counters, recent thought types and the last miss reason."""
        return {
            "scalars": {
                "generated": self.generated,
                "misses": self.misses,
                "noise_filtered": self.noise_filtered,
                "last_miss_reason": self.last_miss_reason,
            },
            "arrays": {},
            "histories": {
                "recent_thought_types": list(self.recent_thought_types),
                "strategy_counts": [
                    [name, int(count)] for name, count in self.strategy_counts.items()
                ],
            },
        }

    def from_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore what ``to_state_dict`` wrote; malformed entries are skipped."""
        scalars = state.get("scalars") or {}
        histories = state.get("histories") or {}
        for name in ("generated", "misses", "noise_filtered"):
            value = scalars.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                setattr(self, name, value)
        reason = scalars.get("last_miss_reason")
        if reason is None or isinstance(reason, str):
            self.last_miss_reason = reason
        known = {t.value for t in ThoughtType}
        types = histories.get("recent_thought_types")
        if isinstance(types, list):
            self.recent_thought_types.clear()
            for item in types:
                if isinstance(item, str) and item in known:
                    self.recent_thought_types.append(item)
        counts = histories.get("strategy_counts")
        if isinstance(counts, list):
            restored: Counter = Counter()
            for entry in counts:
                if (
                    isinstance(entry, (list, tuple))
                    and len(entry) == 2
                    and isinstance(entry[0], str)
                    and isinstance(entry[1], int)
                    and entry[1] >= 0
                ):
                    restored[entry[0]] = entry[1]
            self.strategy_counts = restored
