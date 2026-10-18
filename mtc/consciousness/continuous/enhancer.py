"""The model-written form of a background thought, with honest misses.

``LLMThoughtEnhancer`` asks the brain for one short thought in a category
(``reflection``, ``question``, ``emotion``, ``memory`` or ``creative``), using
the prompts of the CTM persona. It returns a ``ThoughtRecord`` or None. A miss
returns None, stores and credits nothing, and no replacement text is composed
here: a missing brain, a missing required context value, a ``NoAnswerError``
(empty, reasoning-only or looping reply), any other brain failure, an empty
reply, a reasoning-only reply and a reply that starts with one of
``CTMConfig.llm.placeholder_prefixes`` are all misses. ``misses``,
``successes``, ``miss_reasons`` and ``last_miss_reason`` count and name them;
the first miss and every hundredth are logged at WARNING (reason, category and
state only, never the thought text). The method never raises for a brain or
prompt failure while ``Settings.DEBUG`` is false; with DEBUG true an
unexpected failure (any exception other than ``NoAnswerError``) is counted as
a miss first and then re-raised, while a ``NoAnswerError`` stays a plain miss
in every mode. A cancellation always propagates.

Context keys: ``recent_memory`` (required by ``memory``), ``emotions`` (a
mapping of emotion name to score, required by ``emotion``; only finite numeric
scores are shown or stored) and ``recent_topic`` (optional for ``question``). The emotional state reaches the prompt of the
other categories only when ``NARRATE_THOUGHT_MOOD`` is on (read at call time
through ``NarrationFlags``, or the ``narrate_mood`` argument); it is stored on
the record as data either way. Reasoning blocks are removed with
``mtc.brain.thinking.split_thinking`` before the text is used.

The record's salience is the design constant ``LLM_SALIENCE``; its
``processing_context`` says so (``salience_basis`` ``llm_design_constant``)
and names the thought's form (``form`` ``model``).

The clock line is whatever the injected ``clock`` callable returns, added to
the system prompt of waking states only (``awake`` and ``drowsy``); with no
clock the prompt carries no time, and a clock that raises costs no thought.
The record's instant comes from ``now`` (aware UTC). ``should_express`` is
False: expression is decided later by the novelty gate. Config read:
``CTMConfig.llm`` (max tokens, temperature, placeholder prefixes),
``NARRATE_THOUGHT_MOOD`` and ``DEBUG`` (from the passed Settings, else the
process settings). Thought text, memories and emotions are private runtime
data.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional

from mtc.brain.protocol import BrainProtocol, NoAnswerError
from mtc.brain.thinking import split_thinking
from mtc.consciousness.continuous.config import CTMConfig
from mtc.consciousness.continuous.persona import (
    CATEGORIES,
    CTMPersona,
    PersonaError,
)
from mtc.consciousness.continuous.schemas import (
    CONSCIOUSNESS_STATES,
    ThoughtIntensity,
    ThoughtRecord,
    ThoughtType,
    new_thought_id,
)
from mtc.core.config import Settings, get_settings
from mtc.core.narration import NarrationFlags

logger = logging.getLogger(__name__)

CATEGORY_TO_TYPE: Dict[str, ThoughtType] = {
    "reflection": ThoughtType.REFLECTION,
    "question": ThoughtType.QUESTION,
    "emotion": ThoughtType.EMOTION,
    "memory": ThoughtType.MEMORY_REPLAY,
    "creative": ThoughtType.CREATIVE,
}

# A model-written thought is more deliberate than a template one. The value is
# a design constant, not a measurement, and the record's salience_basis says so.
LLM_SALIENCE = 0.8
LLM_SALIENCE_BASIS = "llm_design_constant"
WAKING_STATES = ("awake", "drowsy")
_STATE_BASE_LEVEL = {"deep_sleep": 0.3, "drowsy": 0.5, "dreaming": 0.7, "awake": 0.6}
_REFLECTIVE_WORDS = ("feel", "emotion", "wonder", "realize", "understand", "discover")
_LOG_EVERY = 100


def _intensity_for(content: str, state: str) -> ThoughtIntensity:
    """Intensity from the state's base level plus a small boost per reflective word."""
    lowered = content.lower()
    level = _STATE_BASE_LEVEL[state] + 0.05 * sum(
        w in lowered for w in _REFLECTIVE_WORDS
    )
    level = max(0.0, min(1.0, level))
    if level < 0.2:
        return ThoughtIntensity.WHISPER
    if level < 0.4:
        return ThoughtIntensity.QUIET
    if level < 0.6:
        return ThoughtIntensity.MODERATE
    if level < 0.8:
        return ThoughtIntensity.STRONG
    return ThoughtIntensity.URGENT


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _numeric_emotions(value: Any) -> Dict[str, float]:
    """The emotions with a finite numeric score (a bool is not a score).

    Any other entry, NaN and infinity included, is dropped and never stored.
    """
    if not isinstance(value, Mapping):
        return {}
    scores: Dict[str, float] = {}
    for name, score in value.items():
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        number = float(score)
        if math.isfinite(number):
            scores[str(name)] = number
    return scores


class LLMThoughtEnhancer:
    """Writes a thought through the brain, or reports a miss."""

    def __init__(
        self,
        brain: BrainProtocol,
        persona: CTMPersona,
        config: CTMConfig,
        clock: Optional[Callable[[], Optional[str]]] = None,
        *,
        now: Optional[Callable[[], datetime]] = None,
        narrate_mood: Optional[bool] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.brain = brain
        self.persona = persona
        self.config = config
        self._clock = clock
        self._now: Callable[[], datetime] = now or _utc_now
        self._narrate_mood = narrate_mood
        self._settings = settings
        self.misses = 0
        self.successes = 0
        self.miss_reasons: Dict[str, int] = {}
        self.last_miss_reason: Optional[str] = None
        self._clock_warned = False

    def status(self) -> Dict[str, Any]:
        """Return the counters and the last miss reason."""
        return {
            "successes": self.successes,
            "misses": self.misses,
            "miss_reasons": dict(self.miss_reasons),
            "last_miss_reason": self.last_miss_reason,
            "has_clock": self._clock is not None,
        }

    def _miss(self, reason: str, category: str, state: str) -> None:
        self.misses += 1
        self.miss_reasons[reason] = self.miss_reasons.get(reason, 0) + 1
        self.last_miss_reason = reason
        level = (
            logging.WARNING
            if self.misses == 1 or self.misses % _LOG_EVERY == 0
            else logging.DEBUG
        )
        logger.log(
            level,
            "thought miss #%d: %s (category=%s, state=%s); nothing stored",
            self.misses,
            reason,
            category,
            state,
        )

    def _debug(self) -> bool:
        settings = self._settings if self._settings is not None else get_settings()
        return bool(getattr(settings, "DEBUG", False))

    def _mood_allowed(self) -> bool:
        if self._narrate_mood is not None:
            return bool(self._narrate_mood)
        return NarrationFlags.from_settings().thought_mood

    def _system_prompt(self, state: str) -> str:
        line: Optional[str] = None
        if self._clock is not None and state in WAKING_STATES:
            try:
                line = self._clock()
            except Exception as exc:  # a broken clock never costs a thought
                if not self._clock_warned:
                    self._clock_warned = True
                    logger.warning("thought clock unavailable: %s", type(exc).__name__)
        return self.persona.render_system(line)

    async def generate(
        self,
        category: str,
        context: Optional[Mapping[str, Any]] = None,
        state: str = "awake",
    ) -> Optional[ThoughtRecord]:
        """Return a model-written thought, or None for a miss.

        An unexpected failure raises only under ``Settings.DEBUG``, after it
        was counted (see the module docstring).
        """
        if self.brain is None:
            self._miss("no_brain", category, state)
            return None
        if category not in CATEGORIES:
            self._miss("unknown_category", str(category), state)
            return None
        if state not in CONSCIOUSNESS_STATES:
            self._miss("unknown_state", category, str(state))
            return None
        context = context or {}
        try:
            prompt_context = dict(context)
            if category != "emotion" and not self._mood_allowed():
                prompt_context.pop("emotions", None)
            try:
                prompt = self.persona.render_category(category, prompt_context)
            except PersonaError:
                self._miss("missing_context", category, state)
                return None
            system_prompt = self._system_prompt(state)
            reply = await self.brain.think(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                thinking_mode=False,
            )
            raw = getattr(reply, "text", None) or ""
            reasoning, spoken = split_thinking(raw)
            content = spoken.strip()
            if not content:
                self._miss(
                    "reasoning_only" if reasoning else "empty_reply", category, state
                )
                return None
            prefixes = tuple(self.config.llm.placeholder_prefixes)
            if prefixes and content.startswith(prefixes):
                self._miss("placeholder_reply", category, state)
                return None
            record = ThoughtRecord(
                thought_id=new_thought_id(),
                timestamp=self._now(),
                thought_type=CATEGORY_TO_TYPE[category],
                content=content,
                intensity=_intensity_for(content, state),
                salience_score=LLM_SALIENCE,
                triggered_by=f"llm_{category}",
                emotional_context=_numeric_emotions(context.get("emotions")),
                consciousness_state=state,
                processing_context={
                    "salience_basis": LLM_SALIENCE_BASIS,
                    "form": "model",
                },
                should_express=False,
            )
        except NoAnswerError as exc:
            self._miss(f"no_answer_{exc.reason}", category, state)
            return None
        except Exception as exc:
            self._miss(f"error_{type(exc).__name__}", category, state)
            if self._debug():
                raise
            logger.debug("thought generation failed", exc_info=True)
            return None
        self.successes += 1
        return record


__all__ = (
    "CATEGORY_TO_TYPE",
    "LLM_SALIENCE",
    "LLM_SALIENCE_BASIS",
    "LLMThoughtEnhancer",
)
