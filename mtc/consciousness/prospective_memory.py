"""
Prospective Simulation
======================

``ProspectiveSimulator`` implements prospective simulation, also called
episodic future thinking: imagining what might happen next. It produces
workspace candidates that compete alongside memories and thoughts for access;
it never calls a language model and chooses nothing on its own. The Phase 6
orchestrator calls ``maybe_generate`` once per cycle and runs the competition.

Five simulation kinds (anticipatory, planning, empathic, temporal,
counterfactual) are drawn with time-of-day weights, boosted when goals are
present (planning), when recent messages hold an emotion word (empathic) and
when arousal is high (counterfactual). The time of day is the local hour in
``TIMEZONE``, classified with ``WAKE_HOUR`` and ``SLEEP_HOUR`` through the
temporal context module.

Candidates carry source ``INTERNAL`` (an internal process; the thought-stream
source is reserved for the continuous thought machine), hierarchy level 2,
the confidence heuristic as activation, and the simulation kind and prompt in
metadata. Confidence is deterministic: a base of 0.4, plus recent-context
richness, plus a kind-specific boost, capped at 0.85. The type draw uses an
injectable ``random.Random``.

``recent_messages`` is a list of dicts with ``content`` (or ``message``), and
optionally ``speaker`` and ``role``. Speaker labels pass through into text
unchanged; no label is treated differently from another. Topics, history and
message text can hold real people's names at runtime and are private data.

Persisted state (``to_state_dict``): the last simulation time (None before the
first), the interval, recent topics, simulation history and active goals.
"""

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from mtc.consciousness.enhanced_global_workspace import (
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)
from mtc.consciousness.temporal_context import part_of_day
from mtc.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

SIMULATION_TYPES: Dict[str, str] = {
    "anticipatory": "What will happen next in this conversation?",
    "planning": "How should I approach my goals?",
    "empathic": "How is the speaker feeling? What do they need?",
    "temporal": "What time-related events are coming up?",
    "counterfactual": "What if I had responded differently?",
}

# Weights favour planning in the morning, anticipation at midday, empathy and
# time awareness in the evening and at night. Tunable design values.
TIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "morning": {
        "anticipatory": 0.20,
        "planning": 0.35,
        "empathic": 0.20,
        "temporal": 0.15,
        "counterfactual": 0.10,
    },
    "afternoon": {
        "anticipatory": 0.30,
        "planning": 0.20,
        "empathic": 0.25,
        "temporal": 0.10,
        "counterfactual": 0.15,
    },
    "evening": {
        "anticipatory": 0.15,
        "planning": 0.10,
        "empathic": 0.35,
        "temporal": 0.25,
        "counterfactual": 0.15,
    },
    "night": {
        "anticipatory": 0.15,
        "planning": 0.15,
        "empathic": 0.20,
        "temporal": 0.30,
        "counterfactual": 0.20,
    },
}

GOAL_PLANNING_BOOST = 0.15
EMOTION_EMPATHIC_BOOST = 0.15
AROUSAL_COUNTERFACTUAL_BOOST = 0.10
HIGH_AROUSAL = 0.6
MODERATE_AROUSAL = 0.3
CONFIDENCE_BASE = 0.4
CONFIDENCE_CAP = 0.85

# Generic English emotion words used when the context supplies none.
DEFAULT_EMOTION_WORDS = (
    "feel",
    "feeling",
    "happy",
    "sad",
    "worried",
    "excited",
    "frustrated",
    "love",
    "miss",
    "afraid",
    "proud",
    "tired",
)


@dataclass
class ProspectiveContext:
    """What one call sees.

    ``arousal`` is None when unknown (no boost applies). ``emotion_words`` are
    the words searched for in the last three messages. ``now`` defaults to the
    simulator's clock.
    """

    recent_messages: List[Dict[str, str]]
    goals: List[str] = field(default_factory=list)
    emotion_words: List[str] = field(
        default_factory=lambda: list(DEFAULT_EMOTION_WORDS)
    )
    arousal: Optional[float] = None
    now: Optional[datetime] = None


def _message_text(message: Dict[str, str]) -> str:
    return str(message.get("content", message.get("message", "")) or "")


class ProspectiveSimulator:
    """Generates prospective-simulation workspace candidates.

    Reads ``TIMEZONE``, ``WAKE_HOUR`` and ``SLEEP_HOUR``.
    """

    MAX_RECENT_TOPICS = 20
    MAX_SIMULATION_HISTORY = 50

    def __init__(
        self,
        min_interval_s: float = 120.0,
        clock: Optional[Callable[[], datetime]] = None,
        settings: Optional[Settings] = None,
        rng: Optional[random.Random] = None,
    ):
        self.min_interval_s = float(min_interval_s)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._settings = settings
        self._rng = rng or random.Random()
        self._recent_topics: Deque[str] = deque(maxlen=self.MAX_RECENT_TOPICS)
        self._simulation_history: Deque[str] = deque(maxlen=self.MAX_SIMULATION_HISTORY)
        self._active_goals: List[str] = []
        self._last_simulation_time: Optional[float] = None

    # -- public API ---------------------------------------------------------

    def update_topics(self, message: str) -> None:
        """Track rough topic words (longer than four characters) from a message."""
        words = [w.strip(".,!?;:'\"") for w in message.lower().split() if len(w) > 4]
        for word in words[:5]:
            if word and word not in self._recent_topics:
                self._recent_topics.append(word)

    def set_goals(self, goals: List[str]) -> None:
        """Replace the stored active goals."""
        self._active_goals = list(goals)

    def maybe_generate(self, context: ProspectiveContext) -> List[WorkspaceCandidate]:
        """Return zero or one candidate.

        Nothing is produced inside ``min_interval_s`` of the last simulation,
        without any conversational context, or when the content would exactly
        repeat a recent one.
        """
        now = self._now(context)
        stamp = now.timestamp()
        if (
            self._last_simulation_time is not None
            and stamp - self._last_simulation_time < self.min_interval_s
        ):
            return []
        if context.goals:
            self._active_goals = list(context.goals)
        if (
            not context.recent_messages
            and not self._recent_topics
            and not self._active_goals
        ):
            return []
        weights = self.type_weights(context)
        kinds = list(weights)
        sim_type = self._rng.choices(kinds, weights=[weights[k] for k in kinds], k=1)[0]
        content, prompt = self.build(sim_type, context)
        if content in self._simulation_history:
            return []
        confidence = self.confidence(sim_type, context)
        self._last_simulation_time = stamp
        self._simulation_history.append(content)
        candidate = WorkspaceCandidate(
            content=content,
            content_type="prospective_simulation",
            summary=content,
            source=WorkspaceCandidateSource.INTERNAL,
            source_module="prospective_simulator",
            activation_level=confidence,
            emotional_salience=(
                max(0.0, min(1.0, context.arousal))
                if context.arousal is not None
                else 0.0
            ),
            timestamp=stamp,
            metadata={
                "sim_type": sim_type,
                "prompt": prompt,
                "confidence": confidence,
                "period": self.period_for(now),
            },
        )
        logger.debug(
            "prospective simulation [%s] confidence %.2f", sim_type, confidence
        )
        return [candidate]

    # -- time ---------------------------------------------------------------

    def _now(self, context: ProspectiveContext) -> datetime:
        moment = context.now if context.now is not None else self._clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment

    def _cfg(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    def local_time(self, moment: datetime) -> datetime:
        """Convert an instant to local time in ``TIMEZONE`` (naive counts as UTC)."""
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(ZoneInfo(self._cfg().TIMEZONE))

    def period_for(self, moment: datetime) -> str:
        """Return morning, afternoon, evening or night for an instant."""
        cfg = self._cfg()
        return part_of_day(
            self.local_time(moment), int(cfg.WAKE_HOUR), int(cfg.SLEEP_HOUR)
        )

    # -- selection ----------------------------------------------------------

    def type_weights(self, context: ProspectiveContext) -> Dict[str, float]:
        """Normalised weights over the five kinds for this context."""
        period = self.period_for(self._now(context))
        weights = dict(TIME_WEIGHTS.get(period, TIME_WEIGHTS["afternoon"]))
        if context.goals or self._active_goals:
            weights["planning"] += GOAL_PLANNING_BOOST
        recent_text = " ".join(
            _message_text(m) for m in context.recent_messages[-3:]
        ).lower()
        if any(word.lower() in recent_text for word in context.emotion_words if word):
            weights["empathic"] += EMOTION_EMPATHIC_BOOST
        if context.arousal is not None and context.arousal > HIGH_AROUSAL:
            weights["counterfactual"] += AROUSAL_COUNTERFACTUAL_BOOST
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def confidence(self, sim_type: str, context: ProspectiveContext) -> float:
        """Confidence from context richness; deterministic, capped at 0.85."""
        value = CONFIDENCE_BASE
        if context.recent_messages:
            value += min(0.2, len(context.recent_messages) * 0.04)
        if sim_type == "anticipatory" and self._recent_topics:
            value += 0.1
        if sim_type == "planning" and (context.goals or self._active_goals):
            value += GOAL_PLANNING_BOOST
        if (
            sim_type == "empathic"
            and context.arousal is not None
            and context.arousal > MODERATE_AROUSAL
        ):
            value += 0.1
        return min(CONFIDENCE_CAP, value)

    # -- content ------------------------------------------------------------

    def build(self, sim_type: str, context: ProspectiveContext) -> Tuple[str, str]:
        """Return ``(content, prompt)`` for a simulation kind."""
        now = self.local_time(self._now(context))
        last_snippet = ""
        if context.recent_messages:
            last_snippet = _message_text(context.recent_messages[-1])[:120]
        topics = ", ".join(list(self._recent_topics)[-5:]) or "general conversation"
        goals_source = context.goals or self._active_goals
        goals = "; ".join(goals_source[:3]) or "no specific goals"
        if sim_type == "anticipatory":
            prompt = (
                f"Recent topics: {topics}. Last message: '{last_snippet}'. "
                "What direction might this conversation go next?"
            )
            return self._anticipatory(topics, last_snippet), prompt
        if sim_type == "planning":
            prompt = (
                f"Active goals: {goals}. Current hour: {now.hour}. "
                "What steps could I take toward these goals?"
            )
            return self._planning(goals), prompt
        if sim_type == "empathic":
            prompt = (
                f"Last message: '{last_snippet}'. Topics: {topics}. "
                "How might the speaker be feeling? What do they need from me?"
            )
            return self._empathic_content(last_snippet, context), prompt
        if sim_type == "temporal":
            prompt = (
                f"Current time: {now:%H:%M %A}. "
                "What time-related events or transitions are coming?"
            )
            return self._temporal(now), prompt
        prompt = (
            f"Last exchange snippet: '{last_snippet}'. "
            "What if I had responded differently?"
        )
        return self._counterfactual(last_snippet), prompt

    def _anticipatory(self, topics: str, last_snippet: str) -> str:
        first = topics.split(",")[0].strip()
        options = [
            f"The conversation might turn toward {topics} next.",
            f"Given the discussion so far ({topics}), a deeper look at it may follow.",
            f"After '{last_snippet[:60]}', a natural next step might be to go deeper.",
            f"The conversation may circle back to {first}.",
        ]
        return self._rng.choice(options)

    def _planning(self, goals: str) -> str:
        options = [
            f"To work toward {goals}, the first step is a good place to start.",
            f"Approaching {goals} might go better broken into smaller pieces.",
            f"With these goals ({goals}), what is the most useful next action?",
            f"Planning ahead for {goals} means being deliberate about order.",
        ]
        return self._rng.choice(options)

    def _empathic_content(self, last_snippet: str, context: ProspectiveContext) -> str:
        speaker = "they"
        if context.recent_messages:
            last = context.recent_messages[-1]
            speaker = last.get("speaker", last.get("role", "they"))
        options = [
            f"{speaker} might be feeling something beneath what was said.",
            f"{speaker} brought up '{last_snippet[:40]}'; what is on their mind?",
            f"It matters that {speaker} feels heard right now.",
            f"Reading between the lines, {speaker} might need support or space.",
        ]
        return self._rng.choice(options)

    def _temporal(self, local: datetime) -> str:
        day = f"{local:%A}"
        return {
            "morning": f"Morning on {day}. The day is still taking shape.",
            "afternoon": f"Afternoon on {day}. The day's momentum is shifting.",
            "evening": f"Evening on {day}. The waking day is moving toward its close.",
            "night": f"Night on {day}. Things are quiet, a time for reflection.",
        }[self.period_for(local)]

    def _counterfactual(self, last_snippet: str) -> str:
        options = [
            f"What if a follow-up question had been asked about '{last_snippet[:40]}'?",
            "The conversation might differ with a more curious reply.",
            "Looking back at the last reply, was there a better way to put it?",
            "A more direct reply might have opened a deeper conversation.",
        ]
        return self._rng.choice(options)

    # -- persistence --------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Return ``{"scalars", "arrays", "histories"}`` for the persistence hub."""
        return {
            "scalars": {
                "last_simulation_time": self._last_simulation_time,
                "min_interval_s": self.min_interval_s,
            },
            "arrays": {},
            "histories": {
                "recent_topics": list(self._recent_topics),
                "simulation_history": list(self._simulation_history),
                "active_goals": list(self._active_goals),
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore from a state dict, rebuilding the bounded deques."""
        scalars = state.get("scalars", {})
        last = scalars.get("last_simulation_time")
        self._last_simulation_time = None if last is None else float(last)
        self.min_interval_s = float(scalars.get("min_interval_s", self.min_interval_s))
        histories = state.get("histories", {})
        self._recent_topics = deque(
            histories.get("recent_topics", []), maxlen=self.MAX_RECENT_TOPICS
        )
        self._simulation_history = deque(
            histories.get("simulation_history", []),
            maxlen=self.MAX_SIMULATION_HISTORY,
        )
        self._active_goals = list(histories.get("active_goals", []))
