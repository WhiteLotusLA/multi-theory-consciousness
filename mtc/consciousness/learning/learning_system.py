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
  window.

Config keys: ``LEARNING_ENABLED`` (gates ``record_turn``, ``consolidate`` and
``get_modulation``), ``LEARNING_RATE`` (the base of the affect-modulated
reinforcement rate) and ``NOVELTY_THRESHOLD`` (read and reported; no pattern
family consumes it yet, as in the original design). ``TIMEZONE`` is read by the
tracker. Settings are read at call time.

Persisted state: experiences and beliefs, through a ``LearningStore``. With no
store the system uses an ``InMemoryLearningStore`` and never opens a network
client. ``attach_store`` swaps the store; the next ``initialize`` reloads the
window and beliefs from it. Experiences and beliefs hold real people's names at
runtime and are private data. Store failures are logged and never raised.
"""

import logging
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

    def __init__(
        self,
        store: Optional[LearningStore] = None,
        settings: Any = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._settings = settings
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.store: LearningStore = (
            store if store is not None else InMemoryLearningStore()
        )
        self._build(self.store)
        self.adapter = BehavioralAdapter(clock=self._clock)
        self._pad: Optional[PADState] = None
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
        for the keys it carries. The affect coordinate for the learning rate is
        ``pad`` or, when absent, the state's PAD fields if all three are set.
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

    def get_stats(self) -> Dict[str, Any]:
        """Combined statistics of the learning components."""
        cfg = self._cfg()
        return {
            "experiences": self.tracker.get_stats(),
            "beliefs": self.learner.get_stats(),
            "enabled": bool(cfg.LEARNING_ENABLED),
            "learning_rate": cfg.LEARNING_RATE,
            "novelty_threshold": cfg.NOVELTY_THRESHOLD,
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
