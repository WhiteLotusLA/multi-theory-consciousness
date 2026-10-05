"""
Behavioral Adapter
==================

Translates active learned beliefs into bounded modulations of the pipeline:
a temperature delta, a verbosity delta, short directives for the system
prompt, topic attention bias and drive adjustments. It is the bridge from
"a pattern was learned" to "the next reply is adjusted by it".

Each belief is weighted by its current strength (confidence times recency,
using the injected clock and half-life). Beliefs below strength 0.1 are
ignored and directives need strength 0.3. Modulation is capped
(``MAX_TEMPERATURE_DELTA``, ``MAX_VERBOSITY_DELTA``, ``MAX_DRIVE_ADJUSTMENT``
and at most ``MAX_DIRECTIVES`` directives) so a pile of beliefs cannot run
away with the pipeline.

Directives come from a template table (``DEFAULT_TEMPLATES``, overridable per
adapter) written in a neutral third person; placeholders are ``{speaker}``,
``{topic}`` and ``{time_of_day}``, filled from the belief's own pattern.
Drive adjustments touch only drives the supplied ``HomeostaticDrives`` has,
restricted to ``curiosity`` and ``social_connection``. Behavioral effects with
no counterpart (patience, playfulness, openness, or a drive the supplied set
lacks) are reported in ``unapplied_effects`` and never applied.

The module holds no state and persists nothing. Beliefs name speakers at
runtime, so directives built from them are private data.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from mtc.consciousness.learning.pattern_learner import LearnedBelief

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATES: Dict[str, str] = {
    "speaker_deepening": "Conversations with {speaker} have tended to go deep.",
    "speaker_playful": "Conversations with {speaker} have tended to be playful.",
    "speaker_negative": (
        "Recent conversations with {speaker} have gone less well, so extra "
        "care in replies is indicated."
    ),
    "topic_deepens": (
        "The topic '{topic}' has tended to lead to meaningful exchanges."
    ),
    "temporal_playful": "Interactions in the {time_of_day} have tended to be playful.",
}

# Drives a belief effect may adjust, and the vocabulary assumed when the caller
# supplies no drive set (the five drives of HomeostaticDrives).
ADJUSTABLE_DRIVES = ("curiosity", "social_connection")
DEFAULT_DRIVE_NAMES = (
    "attention_budget",
    "curiosity",
    "social_connection",
    "coherence",
    "safety",
)
# Effects with a direct target in the modulation itself.
DIRECT_EFFECTS = ("warmth", "verbosity")


@dataclass
class BehavioralModulation:
    """Bounded adjustments to apply to the pipeline."""

    temperature_delta: float = 0.0
    verbosity_delta: float = 0.0
    directives: List[str] = field(default_factory=list)
    topic_bias: Dict[str, float] = field(default_factory=dict)
    drive_adjustments: Dict[str, float] = field(default_factory=dict)
    unapplied_effects: Dict[str, float] = field(default_factory=dict)
    source_beliefs: List[str] = field(default_factory=list)


def _clamp(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class BehavioralAdapter:
    """Converts learned beliefs into pipeline modulations."""

    MAX_TEMPERATURE_DELTA = 0.15
    MAX_VERBOSITY_DELTA = 0.2
    MAX_DRIVE_ADJUSTMENT = 0.2
    MAX_DIRECTIVES = 3
    MIN_BELIEF_STRENGTH = 0.1
    MIN_DIRECTIVE_STRENGTH = 0.3

    def __init__(
        self,
        templates: Optional[Dict[str, str]] = None,
        clock: Optional[Callable[[], datetime]] = None,
        half_life_days: float = 7.0,
    ) -> None:
        self.templates: Dict[str, str] = dict(DEFAULT_TEMPLATES)
        if templates:
            self.templates.update(templates)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._half_life_days = half_life_days

    def _strength(self, belief: LearnedBelief) -> float:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("the clock must return timezone-aware datetimes")
        return belief.strength(now, self._half_life_days)

    @staticmethod
    def _known_drives(drives: Any) -> Sequence[str]:
        if drives is None:
            return DEFAULT_DRIVE_NAMES
        return tuple(drives.get_drive_state().keys())

    def adapt(
        self, beliefs: Sequence[LearnedBelief], drives: Any = None
    ) -> BehavioralModulation:
        """Build a modulation from beliefs already filtered to the context.

        ``drives`` is an optional ``HomeostaticDrives`` (anything with
        ``get_drive_state()``); only drives it has are adjusted.
        """
        modulation = BehavioralModulation()
        if not beliefs:
            return modulation
        known = set(self._known_drives(drives))
        temp_deltas: List[float] = []
        ranked: List[tuple] = []
        drive_acc: Dict[str, float] = {}
        unapplied: Dict[str, float] = {}
        topic_acc: Dict[str, float] = {}
        verbosity = 0.0

        for belief in beliefs:
            strength = self._strength(belief)
            if strength < self.MIN_BELIEF_STRENGTH:
                continue
            modulation.source_beliefs.append(belief.belief_id)
            effects = belief.behavioral_effect
            if "warmth" in effects:
                temp_deltas.append(effects["warmth"] * strength * 0.5)
            if "verbosity" in effects:
                verbosity += effects["verbosity"] * strength
            for name, amount in effects.items():
                if name in DIRECT_EFFECTS:
                    continue
                if name in ADJUSTABLE_DRIVES and name in known:
                    drive_acc[name] = drive_acc.get(name, 0.0) + amount * strength
                else:
                    unapplied[name] = unapplied.get(name, 0.0) + amount * strength
            directive = self._directive(belief, strength)
            if directive:
                ranked.append((strength, directive))
            topic = belief.pattern.get("topic")
            if belief.category == "topic" and topic:
                topic_acc[topic] = topic_acc.get(topic, 0.0) + strength * 0.2

        if temp_deltas:
            modulation.temperature_delta = _clamp(
                sum(temp_deltas) / len(temp_deltas), self.MAX_TEMPERATURE_DELTA
            )
        modulation.verbosity_delta = _clamp(verbosity, self.MAX_VERBOSITY_DELTA)
        ranked.sort(key=lambda item: item[0], reverse=True)
        modulation.directives = [text for _, text in ranked[: self.MAX_DIRECTIVES]]
        modulation.drive_adjustments = {
            name: _clamp(delta, self.MAX_DRIVE_ADJUSTMENT)
            for name, delta in drive_acc.items()
        }
        modulation.unapplied_effects = unapplied
        modulation.topic_bias = topic_acc
        if modulation.directives:
            logger.debug(
                "Learning modulation: %d directives, temperature delta %+.3f",
                len(modulation.directives),
                modulation.temperature_delta,
            )
        return modulation

    def _directive(self, belief: LearnedBelief, strength: float) -> Optional[str]:
        """Pick a template for a strong belief and fill it from the pattern."""
        if strength < self.MIN_DIRECTIVE_STRENGTH:
            return None
        pattern = belief.pattern
        key: Optional[str] = None
        if belief.category == "speaker" and pattern.get("speaker"):
            outcome = pattern.get("dominant_outcome", "positive")
            if outcome in ("deepening", "playful", "negative"):
                key = f"speaker_{outcome}"
        elif belief.category == "topic" and "deepens" in belief.belief_id:
            key = "topic_deepens"
        elif belief.category == "temporal":
            if (
                pattern.get("time_of_day")
                and pattern.get("dominant_outcome") == "playful"
            ):
                key = "temporal_playful"
        template = self.templates.get(key) if key else None
        if not template:
            return None
        try:
            return template.format(
                speaker=pattern.get("speaker", ""),
                topic=pattern.get("topic", ""),
                time_of_day=pattern.get("time_of_day", ""),
            )
        except (KeyError, IndexError, ValueError):
            logger.warning("Directive template %s could not be filled", key)
            return None

    def summary(self, modulation: BehavioralModulation) -> str:
        """One-line description of a modulation."""
        parts = []
        if modulation.temperature_delta != 0:
            tone = "warmer" if modulation.temperature_delta > 0 else "cooler"
            parts.append(f"tone: {tone} ({modulation.temperature_delta:+.3f})")
        if modulation.directives:
            parts.append(f"{len(modulation.directives)} learned directives")
        if modulation.drive_adjustments:
            drives = ", ".join(
                f"{k}:{v:+.2f}" for k, v in modulation.drive_adjustments.items()
            )
            parts.append(f"drives: [{drives}]")
        return "; ".join(parts) if parts else "no learning modulation"
