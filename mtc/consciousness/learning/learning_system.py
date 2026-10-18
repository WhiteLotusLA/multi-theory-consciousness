"""
Learning System
===============

Top-level orchestrator of experiential learning. It wires the experience
tracker, the pattern learner, the behavioral adapter and the affect-modulated
rate behind one interface:

- after each turn, ``await record_turn(...)`` logs the experience and, every
  ``LEARNING_INTERVAL`` turns, runs an incremental learning pass;
- before a reply, ``get_modulation(...)`` returns the bounded modulation from
  relevant beliefs;
- during consolidation, ``await consolidate()`` runs a full pass over the
  window;
- with a goal system attached (``goals``, an ``IntrinsicGoalSystem``), every
  pass that changed beliefs (incremental or consolidation) calls its
  ``generate_goals`` with the active beliefs of at least
  ``MIN_PROPOSAL_STRENGTH``.

Outcomes classify the system's own PAD shift between consecutive turns of a
conversation (see ``record_turn``). A failing step (``goal_generation``) is
counted by name in ``get_stats()["failures"]``, logged at WARNING the first
time and every hundredth time with the exception type only, and re-raised
when ``DEBUG`` is true.

Config keys: ``LEARNING_ENABLED`` (gates ``record_turn``, ``consolidate`` and
``get_modulation``), ``LEARNING_RATE`` (the base of the affect-modulated
reinforcement rate) and ``NOVELTY_THRESHOLD`` (read and reported; no pattern
family consumes it yet, as in the original design) and ``DEBUG``. ``TIMEZONE``
is read by the tracker. Settings are read at call time.

Persisted state: experiences and beliefs, through a ``LearningStore``. With no
store the system uses an ``InMemoryLearningStore`` and never opens a network
client. ``attach_store`` swaps the store; the next ``initialize`` reloads the
window and beliefs from it. Experiences and beliefs hold real people's names at
runtime and are private data. Store failures are logged and never raised.
"""

import logging
import math
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional

from mtc.consciousness.learning.affect_modulation import AffectModulatedRate
from mtc.consciousness.learning.behavioral_adapter import (
    BehavioralAdapter,
    BehavioralModulation,
)
from mtc.consciousness.learning.experience_tracker import (
    Experience,
    ExperienceTracker,
    experience_metrics_from_state,
)
from mtc.consciousness.learning.pattern_learner import PatternLearner
from mtc.consciousness.pad_affect import PADState
from mtc.core.config import get_settings
from mtc.memory.inmemory import InMemoryLearningStore
from mtc.memory.protocols import LearningStore

logger = logging.getLogger(__name__)


class LearningSystem:
    """Experiential learning behind a store protocol."""

    # Run an incremental learning pass every N recorded turns.
    LEARNING_INTERVAL = 10
    INCREMENTAL_WINDOW = 50
    CONSOLIDATION_WINDOW = 200
    MIN_INCREMENTAL_EXPERIENCES = 3
    # The previous turn's recorded affect is kept, in memory only, for this
    # many conversations (the least recently used is forgotten first).
    PREVIOUS_AFFECT_CONVERSATIONS = 100
    DEFAULT_CONVERSATION = "default"
    # A failing step is logged at WARNING the first time and every this many.
    FAILURE_LOG_EVERY = 100

    def __init__(
        self,
        store: Optional[LearningStore] = None,
        settings: Any = None,
        clock: Optional[Callable[[], datetime]] = None,
        goals: Any = None,
    ) -> None:
        self._settings = settings
        # an IntrinsicGoalSystem (or anything with async generate_goals)
        self.goals = goals
        self.failures: Counter = Counter()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.store: LearningStore = (
            store if store is not None else InMemoryLearningStore()
        )
        self._build(self.store)
        self.adapter = BehavioralAdapter(clock=self._clock)
        self._pad: Optional[PADState] = None
        # conversation id -> the affect this system recorded for that
        # conversation's previous turn (valence, arousal, optional label)
        self._previous_affect: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._turns_since_learning = 0
        self._loaded = False

    def _build(self, store: LearningStore) -> None:
        self.tracker = ExperienceTracker(
            store,
            window=self.CONSOLIDATION_WINDOW,
            settings=self._settings,
            clock=self._clock,
        )
        self.learner = PatternLearner(
            store, min_evidence=3, max_beliefs=500, clock=self._clock
        )

    def _cfg(self) -> Any:
        return self._settings if self._settings is not None else get_settings()

    def _enabled(self) -> bool:
        return bool(self._cfg().LEARNING_ENABLED)

    def _failure(self, step: str, exc: BaseException) -> None:
        """Count a failed step, log the first and every hundredth, raise in DEBUG."""
        self.failures[step] += 1
        count = self.failures[step]
        if count == 1 or count % self.FAILURE_LOG_EVERY == 0:
            logger.warning(
                "learning step %s failed (%d so far): %s",
                step,
                count,
                type(exc).__name__,
            )
        if bool(getattr(self._cfg(), "DEBUG", False)):
            raise exc

    async def _propose_goals(self, changed: int) -> None:
        """After a pass that changed beliefs, let the goal system propose."""
        goals = self.goals
        if goals is None or not changed:
            return
        # imported here: the goals module imports this package
        from mtc.consciousness.intrinsic_goals import MIN_PROPOSAL_STRENGTH

        try:
            beliefs = self.learner.get_active_beliefs(
                min_strength=MIN_PROPOSAL_STRENGTH
            )
            adopted = await goals.generate_goals(beliefs)
        except Exception as exc:
            self._failure("goal_generation", exc)
            return
        if adopted:
            logger.info("Learning pass adopted %d goals", adopted)

    def _rate(self) -> float:
        base = float(self._cfg().LEARNING_RATE)
        return AffectModulatedRate(base_rate=base).rate(self._pad)

    async def initialize(self) -> None:
        """Load the experience window and beliefs from the store."""
        await self.tracker.initialize()
        await self.learner.initialize()
        self._loaded = True
        cfg = self._cfg()
        logger.info(
            "Learning system initialized (rate=%s, novelty=%s)",
            cfg.LEARNING_RATE,
            cfg.NOVELTY_THRESHOLD,
        )

    def attach_store(self, store: LearningStore) -> None:
        """Swap the store; in-memory state is dropped until the next load.

        The window and beliefs reload from the new store on the next
        ``initialize`` (or implicitly at the next ``record_turn`` or
        ``consolidate``). ``get_modulation`` returns a neutral modulation until
        then.
        """
        self.store = store
        self._build(store)
        self._turns_since_learning = 0
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            await self.initialize()

    async def record_turn(
        self,
        speaker: str,
        message: str,
        response: str,
        emotion_before: Optional[Mapping[str, Any]] = None,
        emotion_after: Optional[Mapping[str, Any]] = None,
        state: Any = None,
        metrics: Optional[Mapping[str, Optional[float]]] = None,
        pad: Optional[PADState] = None,
        conversation_depth: int = 1,
        response_time_ms: Optional[int] = None,
        conversation_id: Optional[str] = None,
        topic: Optional[str] = None,
        topics_discussed: Optional[List[str]] = None,
        consciousness_score: Optional[float] = None,
        homeostatic_drive: Optional[str] = None,
        attention_focus: Optional[str] = None,
    ) -> Optional[Experience]:
        """Record one turn; returns None when learning is disabled.

        ``state`` (a ``ConsciousnessState``) is mapped to metrics with
        ``experience_metrics_from_state``; an explicit ``metrics`` mapping wins
        for the keys it carries. The turn's PAD is ``pad`` or, when absent, the
        state's PAD fields if all three are set; it sets the learning rate.

        Outcomes classify the system's own PAD shift between consecutive turns
        of a conversation. Without an ``emotion_after`` the turn's PAD gives it
        (``valence`` the pleasure, ``arousal`` the arousal). Without an
        ``emotion_before`` it is the affect this system recorded for the
        previous turn of the same ``conversation_id`` (``default`` when None):
        absent on a conversation's first turn, and after a turn that had none,
        so that turn's valence shift and outcome stay None. A turn without a
        complete PAD and without an ``emotion_after`` records no shift. The
        previous turn's affect is held in memory only, for the most recent
        ``PREVIOUS_AFFECT_CONVERSATIONS`` conversations.
        """
        if not self._enabled():
            return None
        await self._ensure_loaded()
        merged: Dict[str, Optional[float]] = {}
        if state is not None:
            merged.update(experience_metrics_from_state(state))
            if pad is None:
                pad = self._pad_from_state(state)
        if metrics:
            merged.update(metrics)
        if pad is not None:
            self._pad = pad
        if emotion_after is None and pad is not None:
            emotion_after = {"valence": pad.pleasure, "arousal": pad.arousal}
        recorded = _affect_reading(emotion_after)
        key = conversation_id or self.DEFAULT_CONVERSATION
        if emotion_before is None and recorded is not None:
            emotion_before = self._previous_affect.get(key)
        self._remember_affect(key, recorded)
        experience = await self.tracker.record(
            speaker,
            message,
            response,
            emotion_before=emotion_before,
            emotion_after=emotion_after,
            metrics=merged,
            conversation_depth=conversation_depth,
            response_time_ms=response_time_ms,
            conversation_id=conversation_id,
            topic=topic,
            topics_discussed=topics_discussed,
            consciousness_score=consciousness_score,
            homeostatic_drive=homeostatic_drive,
            attention_focus=attention_focus,
        )
        self._turns_since_learning += 1
        if self._turns_since_learning >= self.LEARNING_INTERVAL:
            await self._run_learning()
            self._turns_since_learning = 0
        return experience

    def _remember_affect(self, key: str, reading: Optional[Dict[str, Any]]) -> None:
        """Keep this turn's affect as the next turn's ``emotion_before``."""
        self._previous_affect.pop(key, None)
        if reading is None:
            return
        self._previous_affect[key] = reading
        while len(self._previous_affect) > self.PREVIOUS_AFFECT_CONVERSATIONS:
            self._previous_affect.popitem(last=False)

    @staticmethod
    def _pad_from_state(state: Any) -> Optional[PADState]:
        values = [
            getattr(state, name, None)
            for name in ("pad_pleasure", "pad_arousal", "pad_dominance")
        ]
        if any(v is None for v in values):
            return None
        return PADState(pleasure=values[0], arousal=values[1], dominance=values[2])

    def get_modulation(
        self,
        speaker: Optional[str] = None,
        topic: Optional[str] = None,
        time_of_day: Optional[str] = None,
        drives: Any = None,
    ) -> BehavioralModulation:
        """Return the modulation from beliefs relevant to the context."""
        if not self._enabled():
            return BehavioralModulation()
        beliefs = self.learner.get_beliefs_for_context(
            speaker=speaker, topic=topic, time_of_day=time_of_day
        )
        return self.adapter.adapt(beliefs, drives=drives)

    async def consolidate(self) -> Dict[str, int]:
        """Run a full learning pass over the window.

        Returns the number of experiences analysed and beliefs changed.
        """
        result = {"experiences": 0, "beliefs_changed": 0}
        if not self._enabled():
            return result
        await self._ensure_loaded()
        experiences = self.tracker.get_recent(self.CONSOLIDATION_WINDOW)
        if not experiences:
            return result
        changed = await self.learner.learn_from_experiences(
            experiences, rate=self._rate()
        )
        result["experiences"] = len(experiences)
        result["beliefs_changed"] = len(changed)
        logger.info(
            "Consolidation analysed %d experiences, changed %d beliefs",
            len(experiences),
            len(changed),
        )
        await self._propose_goals(len(changed))
        return result

    async def _run_learning(self) -> None:
        recent = self.tracker.get_recent(self.INCREMENTAL_WINDOW)
        if len(recent) < self.MIN_INCREMENTAL_EXPERIENCES:
            return
        changed = await self.learner.learn_from_experiences(recent, rate=self._rate())
        if changed:
            logger.info(
                "Learned %d patterns from the last %d experiences",
                len(changed),
                len(recent),
            )
        await self._propose_goals(len(changed))

    def get_stats(self) -> Dict[str, Any]:
        """Combined statistics of the learning components."""
        cfg = self._cfg()
        return {
            "experiences": self.tracker.get_stats(),
            "beliefs": self.learner.get_stats(),
            "enabled": bool(cfg.LEARNING_ENABLED),
            "learning_rate": cfg.LEARNING_RATE,
            "novelty_threshold": cfg.NOVELTY_THRESHOLD,
            "failures": dict(self.failures),
        }

    def get_belief_summary(self) -> str:
        """Neutral third-person summary of the strongest beliefs, or ""."""
        if not self._enabled():
            return ""
        active = [b for b in self.learner.get_active_beliefs(min_strength=0.3)]
        if not active:
            return ""
        active.sort(key=self.learner.strength, reverse=True)
        lines = ["Learned patterns:"]
        for belief in active[:5]:
            lines.append(
                f"- {belief.description} (confidence: {belief.confidence:.0%})"
            )
        return "\n".join(lines)


def _affect_reading(emotion: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The valence, arousal and label of a turn's affect, or None.

    A reading needs a finite numeric valence; an arousal that is not a finite
    number is left out.
    """
    if not emotion:
        return None
    valence = _finite(emotion.get("valence"))
    if valence is None:
        return None
    reading: Dict[str, Any] = {"valence": valence}
    arousal = _finite(emotion.get("arousal"))
    if arousal is not None:
        reading["arousal"] = arousal
    label = emotion.get("label")
    if isinstance(label, str) and label:
        reading["label"] = label
    return reading


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
