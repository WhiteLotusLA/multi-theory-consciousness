"""
Emotion dynamics
================

Temporal physics for a classified emotion. A raw label and a PAD coordinate
(pleasure, arousal, dominance; ``PADState`` from ``mtc.consciousness.
pad_affect``) are run through four layers once per consciousness cycle:

1. Inertia. The PAD coordinate is exponentially smoothed; ``inertia`` is the
   weight of the previous value (0 none, 1 frozen).
2. Mood and acute. Mood is a very slow average of the PAD coordinate whose
   label comes from the ``mood_words`` registry; an acute spike registers when
   the raw coordinate jumps by more than ``acute_onset_threshold`` while the
   label changes, and loses ``acute_decay`` of its intensity every later
   cycle. A strong acute spike (above ``acute_override``) replaces the
   dominant label.
3. Blending. When a secondary label is present and the pair is in
   ``blend_pairs``, the state carries the blend name; a confidence at or
   above ``blend_confidence_max`` suppresses it.
4. Trajectory. Valence and arousal momentum (the change in the smoothed
   value) and volatility (the mean absolute raw jump over the last
   ``volatility_window`` cycles).

PAD ranges: MTC's ``PADState`` has pleasure in [-1, 1] and arousal and
dominance in [0, 1], which is also the range this module works in, so no
conversion is needed; inputs outside the range are clamped at the boundary
and a non-finite value raises ``ValueError``.

Honest start: with no history there is no prior, so the first update seeds
the smoothed and mood values with the observation itself; momentum and
volatility are None until a previous cycle exists, and label logic only runs
when a label is supplied. The default blend pairs and mood words are
documented defaults keyed to the extractor's default vocabulary; subtitles
exist only when configured.

Config: ``EmotionDynamicsConfig``. Persisted state (``to_state_dict``): the
smoothed and mood values, the acute layer, the previous cycle and recent
deltas; labels are conversation-derived and private. The pipeline runtime
(``mtc.pipeline.runtime``) registers the module with the persistence hub as
``emotion_dynamics``.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, FrozenSet, List, Optional
from collections import deque

from mtc.consciousness.pad_affect import PADState

logger = logging.getLogger(__name__)

DEFAULT_BLEND_PAIRS: Dict[FrozenSet[str], str] = {
    frozenset({"joyful", "sad"}): "bittersweet",
    frozenset({"curious", "uneasy"}): "anxious curiosity",
    frozenset({"excited", "scared"}): "thrilled",
    frozenset({"loving", "sad"}): "tender longing",
    frozenset({"thoughtful", "confused"}): "wrestling",
    frozenset({"calm", "curious"}): "contemplative",
    frozenset({"playful", "curious"}): "exploring",
    frozenset({"joyful", "loving"}): "radiant",
    frozenset({"peaceful", "loving"}): "serene warmth",
    frozenset({"fascinated", "overwhelmed"}): "awestruck",
}

DEFAULT_MOOD_WORDS: Dict[str, str] = {
    "positive_high": "energized",
    "positive_low": "content",
    "negative_high": "agitated",
    "negative_low": "subdued",
    "neutral_high": "alert",
    "neutral_low": "neutral",
}


@dataclass
class EmotionDynamicsConfig:
    """Constants of the four layers; every one is tunable."""

    inertia: float = 0.6
    mood_smoothing: float = 0.95
    acute_decay: float = 0.30
    acute_onset_threshold: float = 0.15
    acute_floor: float = 0.05
    acute_override: float = 0.3
    blend_confidence_max: float = 0.8
    volatility_window: int = 10
    mood_pleasure_threshold: float = 0.2
    mood_arousal_split: float = 0.5
    mood_stability_tau_s: float = 3600.0
    blend_pairs: Dict[FrozenSet[str], str] = field(
        default_factory=lambda: dict(DEFAULT_BLEND_PAIRS)
    )
    mood_words: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MOOD_WORDS))
    # Optional label to display-text map; the payload carries a subtitle only
    # when the current label has an entry here.
    subtitles: Dict[str, str] = field(default_factory=dict)


@dataclass
class DynamicEmotionState:
    """The emotional state after temporal processing; unknowns are None."""

    dominant: Optional[str]
    secondary: Optional[str]
    confidence: Optional[float]
    mood: str
    mood_valence: float
    mood_arousal: float
    mood_stability: float
    acute_emotion: Optional[str]
    acute_intensity: float
    acute_decay_remaining: float
    blend_name: Optional[str]
    blend_ratio: Optional[float]
    valence_momentum: Optional[float]
    arousal_momentum: Optional[float]
    emotional_volatility: Optional[float]
    pleasure: float
    arousal: float
    dominance: float


def _ema(old: float, new: float, weight_old: float) -> float:
    return weight_old * old + (1.0 - weight_old) * new


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class EmotionDynamics:
    """Applies inertia, mood, acute spikes, blends and trajectory per cycle."""

    def __init__(self, config: Optional[EmotionDynamicsConfig] = None) -> None:
        self.config = config or EmotionDynamicsConfig()
        self._initialized = False
        self._smoothed = (0.0, 0.5, 0.5)
        self._mood_p = 0.0
        self._mood_a = 0.5
        self._mood_label: Optional[str] = None
        self._mood_onset = 0.0
        self._acute: Optional[Dict[str, Any]] = None
        self._prev_raw_p = 0.0
        self._prev_raw_a = 0.5
        self._prev_smoothed_p = 0.0
        self._prev_smoothed_a = 0.5
        self._prev_dominant: Optional[str] = None
        self._deltas: Deque[float] = deque(maxlen=self.config.volatility_window)
        self._cycle_count = 0
        self._last: Optional[DynamicEmotionState] = None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        pad: PADState,
        dominant: Optional[str] = None,
        secondary: Optional[str] = None,
        now: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> DynamicEmotionState:
        """Process one cycle and return the resulting state."""
        values = (pad.pleasure, pad.arousal, pad.dominance)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("PAD values must be finite")
        p = _clamp(float(pad.pleasure), -1.0, 1.0)
        a = _clamp(float(pad.arousal), 0.0, 1.0)
        d = _clamp(float(pad.dominance), 0.0, 1.0)
        t = time.time() if now is None else float(now)
        cfg = self.config
        first = not self._initialized
        self._cycle_count += 1

        # 1. Inertia (the first observation seeds the state: there is no prior)
        if first:
            self._smoothed = (p, a, d)
            self._mood_p, self._mood_a = p, a
        else:
            sp, sa, sd = self._smoothed
            self._smoothed = (
                _ema(sp, p, cfg.inertia),
                _ema(sa, a, cfg.inertia),
                _ema(sd, d, cfg.inertia),
            )
            self._mood_p = _ema(self._mood_p, p, cfg.mood_smoothing)
            self._mood_a = _ema(self._mood_a, a, cfg.mood_smoothing)

        # 2. Mood
        new_mood = self._classify_mood(self._mood_p, self._mood_a)
        if first or new_mood != self._mood_label:
            self._mood_label = new_mood
            self._mood_onset = t
        mood_stability = 1.0 - math.exp(
            -max(0.0, t - self._mood_onset) / cfg.mood_stability_tau_s
        )

        # 3. Acute layer (needs a previous cycle and a label)
        pad_delta: Optional[float] = None
        if not first:
            pad_delta = max(abs(p - self._prev_raw_p), abs(a - self._prev_raw_a))
        spiked = (
            dominant is not None
            and pad_delta is not None
            and pad_delta > cfg.acute_onset_threshold
            and dominant != self._prev_dominant
        )
        if spiked:
            intensity = min(1.0, pad_delta * 2.0)
            self._acute = {
                "label": dominant,
                "intensity": intensity,
                "onset": t,
                "peak": intensity,
            }
        elif self._acute is not None:
            self._acute["intensity"] *= 1.0 - cfg.acute_decay
            if self._acute["intensity"] < cfg.acute_floor:
                self._acute = None

        # 4. Blending
        blend_name: Optional[str] = None
        blend_ratio: Optional[float] = None
        if dominant and secondary:
            low_confidence = confidence is None or confidence < cfg.blend_confidence_max
            name = cfg.blend_pairs.get(frozenset({dominant, secondary}))
            if name is not None and low_confidence:
                blend_name = name
                if confidence is not None:
                    blend_ratio = max(0.5, confidence)

        # 5. Momentum and volatility (need a previous cycle)
        valence_momentum: Optional[float] = None
        arousal_momentum: Optional[float] = None
        if not first:
            valence_momentum = self._smoothed[0] - self._prev_smoothed_p
            arousal_momentum = self._smoothed[1] - self._prev_smoothed_a
            self._deltas.append(pad_delta or 0.0)
        volatility = sum(self._deltas) / len(self._deltas) if self._deltas else None

        final_dominant = dominant
        if self._acute is not None and self._acute["intensity"] > cfg.acute_override:
            final_dominant = self._acute["label"]

        self._prev_raw_p, self._prev_raw_a = p, a
        self._prev_smoothed_p, self._prev_smoothed_a = (
            self._smoothed[0],
            self._smoothed[1],
        )
        self._prev_dominant = dominant
        self._initialized = True

        acute = self._acute
        state = DynamicEmotionState(
            dominant=final_dominant,
            secondary=secondary,
            confidence=confidence,
            mood=self._mood_label or "",
            mood_valence=self._mood_p,
            mood_arousal=self._mood_a,
            mood_stability=mood_stability,
            acute_emotion=acute["label"] if acute else None,
            acute_intensity=acute["intensity"] if acute else 0.0,
            acute_decay_remaining=(
                acute["intensity"] / acute["peak"]
                if acute and acute["peak"] > 0
                else 0.0
            ),
            blend_name=blend_name,
            blend_ratio=blend_ratio,
            valence_momentum=valence_momentum,
            arousal_momentum=arousal_momentum,
            emotional_volatility=volatility,
            pleasure=self._smoothed[0],
            arousal=self._smoothed[1],
            dominance=self._smoothed[2],
        )
        self._last = state
        return state

    def _classify_mood(self, pleasure: float, arousal: float) -> str:
        cfg = self.config
        if pleasure > cfg.mood_pleasure_threshold:
            sign = "positive"
        elif pleasure < -cfg.mood_pleasure_threshold:
            sign = "negative"
        else:
            sign = "neutral"
        level = "high" if arousal > cfg.mood_arousal_split else "low"
        return cfg.mood_words[f"{sign}_{level}"]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_payload(self) -> Dict[str, Any]:
        """Neutral-key dictionary of the latest state; empty before any update.

        A ``subtitle`` key appears only when the current label has an entry in
        ``config.subtitles``.
        """
        state = self._last
        if state is None:
            return {}

        def rounded(value: Optional[float], digits: int = 3) -> Optional[float]:
            return None if value is None else round(value, digits)

        payload: Dict[str, Any] = {
            "dominant": state.dominant,
            "secondary": state.secondary,
            "valence": rounded(state.pleasure),
            "arousal": rounded(state.arousal),
            "dominance": rounded(state.dominance),
            "confidence": rounded(state.confidence),
            "blend": state.blend_name,
            "blend_ratio": rounded(state.blend_ratio),
            "mood": state.mood,
            "mood_valence": rounded(state.mood_valence),
            "mood_arousal": rounded(state.mood_arousal),
            "mood_stability": rounded(state.mood_stability),
            "acute_emotion": state.acute_emotion,
            "acute_intensity": rounded(state.acute_intensity),
            "valence_momentum": rounded(state.valence_momentum, 4),
            "arousal_momentum": rounded(state.arousal_momentum, 4),
            "volatility": rounded(state.emotional_volatility),
        }
        subtitle = self.config.subtitles.get(state.blend_name or "") or (
            self.config.subtitles.get(state.dominant or "")
        )
        if subtitle:
            payload["subtitle"] = subtitle
        return payload

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Export state for the persistence hub."""
        return {
            "scalars": {
                "initialized": self._initialized,
                "smoothed_pleasure": self._smoothed[0],
                "smoothed_arousal": self._smoothed[1],
                "smoothed_dominance": self._smoothed[2],
                "mood_pleasure": self._mood_p,
                "mood_arousal": self._mood_a,
                "mood_label": self._mood_label,
                "mood_onset": self._mood_onset,
                "prev_raw_pleasure": self._prev_raw_p,
                "prev_raw_arousal": self._prev_raw_a,
                "prev_smoothed_pleasure": self._prev_smoothed_p,
                "prev_smoothed_arousal": self._prev_smoothed_a,
                "prev_dominant": self._prev_dominant,
                "cycle_count": self._cycle_count,
            },
            "arrays": {},
            "histories": {
                "recent_deltas": list(self._deltas),
                "acute": [dict(self._acute)] if self._acute else [],
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore every layer, rebuilding the bounded delta window."""
        s = state.get("scalars", {})
        h = state.get("histories", {})
        self._initialized = bool(s.get("initialized", False))
        self._smoothed = (
            float(s.get("smoothed_pleasure", 0.0)),
            float(s.get("smoothed_arousal", 0.5)),
            float(s.get("smoothed_dominance", 0.5)),
        )
        self._mood_p = float(s.get("mood_pleasure", 0.0))
        self._mood_a = float(s.get("mood_arousal", 0.5))
        self._mood_label = s.get("mood_label")
        self._mood_onset = float(s.get("mood_onset", 0.0))
        self._prev_raw_p = float(s.get("prev_raw_pleasure", 0.0))
        self._prev_raw_a = float(s.get("prev_raw_arousal", 0.5))
        self._prev_smoothed_p = float(s.get("prev_smoothed_pleasure", 0.0))
        self._prev_smoothed_a = float(s.get("prev_smoothed_arousal", 0.5))
        self._prev_dominant = s.get("prev_dominant")
        self._cycle_count = int(s.get("cycle_count", 0))
        self._deltas = deque(
            (float(x) for x in h.get("recent_deltas", [])),
            maxlen=self.config.volatility_window,
        )
        acute: List[Dict[str, Any]] = h.get("acute", [])
        self._acute = dict(acute[0]) if acute else None
