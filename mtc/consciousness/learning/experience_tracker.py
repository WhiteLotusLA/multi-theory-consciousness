"""
Experience Tracker
==================

Records what happened in each conversation turn so pattern detection has raw
material: the topic, the emotional trajectory, engagement signals and the
consciousness metrics of the turn. A rolling in-memory window serves fast
queries; every record is also written through a ``LearningStore``.

Config keys: ``TIMEZONE`` (the zone used for the time-of-day bucket; stored
timestamps stay UTC). Persisted state: one experience document per turn in the
store (the window is rebuilt from the store by ``initialize``). Experiences
hold real people's names (the ``speaker`` field) and conversation text lengths
at runtime and are private data.

Missing evidence stays missing: a metric the state does not carry is stored as
None, never as zero, and an interaction whose emotional valence is unknown has
no outcome class.

The outcome classes (positive, neutral, negative, deepening, playful, repair)
and their thresholds are heuristics, not measurements; the thresholds are
module constants so a reader can see and tune them. The dominant-theory label
is likewise a heuristic summary of three metrics.
"""

import logging
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional
from zoneinfo import ZoneInfo

from mtc.core.config import get_settings
from mtc.memory.protocols import LearningStore

logger = logging.getLogger(__name__)

# Outcome heuristics (thresholds on valence shift and response length).
DEEPENING_MIN_RESPONSE_CHARS = 500
DEEPENING_MIN_VALENCE_SHIFT = 0.1
DEEPENING_MIN_DEPTH = 3
PLAYFUL_MAX_RESPONSE_CHARS = 300
PLAYFUL_LABELS = ("joy", "surprise", "amusement")
NEGATIVE_MAX_VALENCE_SHIFT = -0.2
REPAIR_MIN_VALENCE_SHIFT = 0.3
POSITIVE_MIN_VALENCE_SHIFT = 0.05

# Experience field -> ConsciousnessState attribute (see
# experience_metrics_from_state for the two derived entries).
STATE_FIELD_MAP: Dict[str, str] = {
    "gwt_ignition_count": "ignition_events",
    "beautiful_loop_quality": "beautiful_loop_quality",
    "architectural_phi": "architectural_phi",
    "metacognitive_events": "higher_order_thoughts",
    "prediction_errors": "prediction_error",
    "pad_pleasure": "pad_pleasure",
    "pad_arousal": "pad_arousal",
    "pad_dominance": "pad_dominance",
    "gwt_broadcast_coverage": "broadcast_coverage",
    "gwt_integration_level": "integration_level",
    "epistemic_depth": "epistemic_depth",
    "binding_quality": "binding_quality",
    "protoself_stability": "protoself_stability",
    "feeling_of_knowing": "feeling_of_knowing",
    "homeostatic_urgency": "homeostatic_urgency",
}


class InteractionOutcome(Enum):
    """How an interaction went, a heuristic classification."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    DEEPENING = "deepening"
    PLAYFUL = "playful"
    REPAIR = "repair"


@dataclass
class Experience:
    """One logged interaction. Unknown measurements are None."""

    timestamp: str
    speaker: str
    topic: Optional[str]
    message_length: int
    response_length: int

    emotion_before: Optional[str] = None
    emotion_after: Optional[str] = None
    valence_shift: Optional[float] = None
    arousal_shift: Optional[float] = None

    outcome: Optional[str] = None
    conversation_depth: int = 0
    response_time_ms: Optional[int] = None

    consciousness_score: Optional[float] = None
    dominant_theory: Optional[str] = None
    gwt_ignition_count: Optional[float] = None

    time_of_day: str = ""
    is_first_message: bool = False
    conversation_id: Optional[str] = None

    topics_discussed: List[str] = field(default_factory=list)
    metacognitive_events: Optional[float] = None
    prediction_errors: Optional[float] = None

    pad_pleasure: Optional[float] = None
    pad_arousal: Optional[float] = None
    pad_dominance: Optional[float] = None
    gwt_broadcast_coverage: Optional[float] = None
    gwt_integration_level: Optional[float] = None
    beautiful_loop_quality: Optional[float] = None
    epistemic_depth: Optional[float] = None
    binding_quality: Optional[float] = None
    protoself_stability: Optional[float] = None
    feeling_of_knowing: Optional[float] = None
    architectural_phi: Optional[float] = None
    homeostatic_drive: Optional[str] = None
    homeostatic_urgency: Optional[float] = None
    attention_focus: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def experience_metrics_from_state(state: Any) -> Dict[str, Optional[float]]:
    """Map a ``ConsciousnessState`` onto the tracker's metric keys.

    Every attribute read is listed in ``STATE_FIELD_MAP``. A field that is None
    stays None. ``metacognitive_events`` is the number of higher-order thoughts
    in the state; ``homeostatic_urgency`` is the urgency half of the state's
    ``(drive, urgency)`` pair.
    """
    metrics: Dict[str, Optional[float]] = {}
    for key, attribute in STATE_FIELD_MAP.items():
        value = getattr(state, attribute, None)
        if value is None:
            metrics[key] = None
        elif key == "metacognitive_events":
            metrics[key] = float(len(value))
        elif key == "homeostatic_urgency":
            metrics[key] = float(value[1])
        else:
            metrics[key] = float(value)
    return metrics


def classify_outcome(
    valence_shift: Optional[float],
    response_length: int,
    conversation_depth: int,
    emotion_label: Optional[str],
) -> Optional[InteractionOutcome]:
    """Classify an interaction from the available signals (a heuristic).

    Returns None when the valence shift is unknown, since every class depends
    on it.
    """
    if valence_shift is None:
        return None
    if (
        response_length > DEEPENING_MIN_RESPONSE_CHARS
        and valence_shift > DEEPENING_MIN_VALENCE_SHIFT
        and conversation_depth > DEEPENING_MIN_DEPTH
    ):
        return InteractionOutcome.DEEPENING
    if (
        valence_shift > 0
        and emotion_label in PLAYFUL_LABELS
        and response_length < PLAYFUL_MAX_RESPONSE_CHARS
    ):
        return InteractionOutcome.PLAYFUL
    if valence_shift < NEGATIVE_MAX_VALENCE_SHIFT:
        return InteractionOutcome.NEGATIVE
    if valence_shift > REPAIR_MIN_VALENCE_SHIFT:
        return InteractionOutcome.REPAIR
    if valence_shift > POSITIVE_MIN_VALENCE_SHIFT:
        return InteractionOutcome.POSITIVE
    return InteractionOutcome.NEUTRAL


def dominant_theory_label(
    ignitions: Optional[float],
    loop_quality: Optional[float],
    phi: Optional[float],
) -> Optional[str]:
    """Label the strongest of three metrics (a heuristic); None without data."""
    if ignitions is None and loop_quality is None and phi is None:
        return None
    ignited = ignitions is not None and ignitions > 0
    if ignited and loop_quality is not None and loop_quality > 0.5:
        return "gwt+blt"
    if ignited:
        return "gwt"
    if loop_quality is not None and loop_quality > 0.3:
        return "blt"
    if phi is not None and phi > 0:
        return "iit"
    return "baseline"


def _time_of_day(local_hour: int) -> str:
    if local_hour < 6:
        return "night"
    if local_hour < 12:
        return "morning"
    if local_hour < 18:
        return "afternoon"
    return "evening"


def _as_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


class ExperienceTracker:
    """Captures interaction outcomes and writes them through a store."""

    def __init__(
        self,
        store: LearningStore,
        window: int = 200,
        settings: Any = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._store = store
        self._window: Deque[Experience] = deque(maxlen=window)
        self._settings = settings
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._total_logged = 0

    def _now(self) -> datetime:
        moment = self._clock()
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("the clock must return timezone-aware datetimes")
        return moment.astimezone(timezone.utc)

    def _timezone_name(self) -> str:
        settings = self._settings if self._settings is not None else get_settings()
        return str(settings.TIMEZONE)

    async def initialize(self) -> None:
        """Load the most recent experiences from the store into the window."""
        try:
            docs = await self._store.recent_experiences(self._window.maxlen or 200)
        except Exception as exc:  # store failures never stop the tracker
            logger.warning("Could not load experiences: %s", exc)
            return
        self._window.clear()
        for doc in reversed(docs):
            experience = self._from_doc(doc)
            if experience is not None:
                self._window.append(experience)
        logger.info("ExperienceTracker loaded %d experiences", len(self._window))

    async def record(
        self,
        speaker: str,
        message: str,
        response: str,
        emotion_before: Optional[Mapping[str, Any]] = None,
        emotion_after: Optional[Mapping[str, Any]] = None,
        metrics: Optional[Mapping[str, Optional[float]]] = None,
        conversation_depth: int = 1,
        response_time_ms: Optional[int] = None,
        conversation_id: Optional[str] = None,
        topic: Optional[str] = None,
        topics_discussed: Optional[List[str]] = None,
        consciousness_score: Optional[float] = None,
        homeostatic_drive: Optional[str] = None,
        attention_focus: Optional[str] = None,
    ) -> Experience:
        """Log one turn and write it through the store.

        ``metrics`` is the output of ``experience_metrics_from_state`` (or any
        mapping with the same keys); absent keys stay None. ``emotion_*`` carry
        optional ``valence``, ``arousal`` and ``label`` entries. A store
        failure is logged and the in-memory window still receives the record.
        """
        before = emotion_before or {}
        after = emotion_after or {}
        valence_shift = self._shift(before.get("valence"), after.get("valence"))
        arousal_shift = self._shift(before.get("arousal"), after.get("arousal"))
        outcome = classify_outcome(
            valence_shift,
            len(response),
            conversation_depth,
            after.get("label"),
        )
        values = dict(metrics or {})
        known = {key: _as_float(values.get(key)) for key in STATE_FIELD_MAP}
        moment = self._now()
        local = moment.astimezone(ZoneInfo(self._timezone_name()))
        experience = Experience(
            timestamp=moment.isoformat(),
            speaker=speaker,
            topic=topic,
            message_length=len(message),
            response_length=len(response),
            emotion_before=before.get("label"),
            emotion_after=after.get("label"),
            valence_shift=valence_shift,
            arousal_shift=arousal_shift,
            outcome=outcome.value if outcome is not None else None,
            conversation_depth=conversation_depth,
            response_time_ms=response_time_ms,
            consciousness_score=_as_float(consciousness_score),
            dominant_theory=dominant_theory_label(
                known["gwt_ignition_count"],
                known["beautiful_loop_quality"],
                known["architectural_phi"],
            ),
            time_of_day=_time_of_day(local.hour),
            is_first_message=conversation_depth <= 1,
            conversation_id=conversation_id,
            topics_discussed=list((topics_discussed or [])[:5]),
            homeostatic_drive=homeostatic_drive,
            attention_focus=attention_focus,
            **known,
        )
        self._window.append(experience)
        self._total_logged += 1
        try:
            await self._store.add_experience(experience.to_dict())
        except Exception as exc:
            logger.warning("Failed to persist experience: %s", exc)
        return experience

    @staticmethod
    def _shift(before: Any, after: Any) -> Optional[float]:
        if before is None or after is None:
            return None
        return float(after) - float(before)

    @staticmethod
    def _from_doc(doc: Mapping[str, Any]) -> Optional[Experience]:
        names = {f.name for f in fields(Experience)}
        data = {k: v for k, v in doc.items() if k in names}
        try:
            return Experience(**data)
        except TypeError:
            logger.warning("Skipped a malformed stored experience")
            return None

    def get_recent(self, n: int = 50) -> List[Experience]:
        """The ``n`` most recent experiences in the window, oldest first."""
        if n <= 0:
            return []
        return list(self._window)[-n:]

    def get_by_speaker(self, speaker: str, n: int = 50) -> List[Experience]:
        """Recent experiences with one speaker."""
        matching = [e for e in self._window if e.speaker == speaker]
        return matching[-n:] if n > 0 else []

    def get_by_outcome(
        self, outcome: InteractionOutcome, n: int = 50
    ) -> List[Experience]:
        """Recent experiences with one outcome class."""
        matching = [e for e in self._window if e.outcome == outcome.value]
        return matching[-n:] if n > 0 else []

    def get_stats(self) -> Dict[str, Any]:
        """Summary of the window; the average is None without valence data."""
        outcomes: Dict[str, int] = {}
        shifts: List[float] = []
        for exp in self._window:
            if exp.outcome is not None:
                outcomes[exp.outcome] = outcomes.get(exp.outcome, 0) + 1
            if exp.valence_shift is not None:
                shifts.append(exp.valence_shift)
        return {
            "total": len(self._window),
            "total_ever": self._total_logged,
            "outcomes": outcomes,
            "avg_valence_shift": sum(shifts) / len(shifts) if shifts else None,
            "speakers": sorted({e.speaker for e in self._window}),
        }
