"""
Attention Appraisal
===================

A pure, deterministic appraisal of whether a thought is worth dwelling on
(sustain attention) or better released (let attention move on). It runs in
shadow mode: it computes and reports what it would do and forces no behavior;
the caller supplies the emotions and decides what to do with the result.

Design note: the weights are transparent, provisional and tunable, not a claim
about any mind. Emotional charge counts as salience (its magnitude), which
pulls toward dwelling; its sign is never used, so strong negative feeling is
never released automatically. Only a recurring, low-novelty theme pulls toward
release. A missing signal (no valence, no values match) is absent from the
score, never replaced by a number.

Each signal's signed contribution is a share of the total pull, so the
contributions sum exactly to ``score``, and ``score`` is the net dwell
balance in [-1, 1], equal to twice the dwell share minus one. The decision is
``dwell`` when ``score`` reaches ``AppraisalWeights.dwell_threshold`` (0.0 by
default, which is a dwell share of one half). Records describe mechanism:
whether anything is felt behind the appraisal is unknown.

``valence_from_emotions`` inverts a named-emotion mapping to a valence scalar
with an ``EmotionLexicon`` (generic provisional word lists; configurable) and
returns None when no known emotion is present.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Mapping, Optional


@dataclass(frozen=True)
class AppraisalInput:
    """The signals of one appraisal.

    ``novelty`` and ``recurrence`` are in [0, 1]; ``valence`` in [-1, 1] or
    None when unknown; ``values_match`` in [0, 1] or None when no real signal
    exists. Out-of-range values are clamped; non-finite required values raise.
    """

    novelty: float
    valence: Optional[float]
    recurrence: float
    values_match: Optional[float] = None


@dataclass(frozen=True)
class AppraisalWeights:
    """Provisional, tunable weights.

    ``baseline_release`` is the neutral prior that a passing thought simply
    passes unless novelty, salience or values make it worth staying with.
    """

    novelty: float = 1.0
    salience: float = 1.0
    values_match: float = 1.0
    recurrence: float = 1.0
    baseline_release: float = 1.0
    dwell_threshold: float = 0.0


@dataclass(frozen=True)
class AppraisalResult:
    """Outcome of one appraisal (shadow mode: nothing acts on it)."""

    decision: str  # "dwell" or "release"
    score: float  # net dwell balance in [-1, 1]; the sum of the contributions
    dwell_share: float  # dwell pull over total pull, in [0, 1]
    threshold: float
    contributions: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def appraise(
    item: AppraisalInput, weights: Optional[AppraisalWeights] = None
) -> AppraisalResult:
    """Score dwell versus release for one thought."""
    w = weights or AppraisalWeights()
    novelty = _clamp(_finite(item.novelty, "novelty"), 0.0, 1.0)
    recurrence = _clamp(_finite(item.recurrence, "recurrence"), 0.0, 1.0)

    pulls: Dict[str, float] = {
        "novelty": w.novelty * novelty,
        "recurrence": -(w.recurrence * recurrence),
        "baseline_release": -w.baseline_release,
    }
    if item.valence is not None:
        magnitude = abs(_clamp(_finite(item.valence, "valence"), -1.0, 1.0))
        pulls["salience"] = w.salience * magnitude
    if item.values_match is not None:
        match = _clamp(_finite(item.values_match, "values_match"), 0.0, 1.0)
        pulls["values_dwell"] = w.values_match * match
        pulls["values_release"] = -(w.values_match * (1.0 - match))

    dwell_pull = sum(v for v in pulls.values() if v > 0)
    release_pull = -sum(v for v in pulls.values() if v < 0)
    total = dwell_pull + release_pull
    if total > 0:
        contributions = {k: v / total for k, v in pulls.items()}
        dwell_share = dwell_pull / total
    else:
        contributions = {k: 0.0 for k in pulls}
        dwell_share = 0.5
    score = sum(contributions.values())
    decision = "dwell" if score >= w.dwell_threshold else "release"
    rationale = (
        f"appraisal gives {decision} (net balance {score:.2f}, threshold "
        f"{w.dwell_threshold:.2f}); mechanism only, whether this is felt is unknown"
    )
    return AppraisalResult(
        decision=decision,
        score=score,
        dwell_share=dwell_share,
        threshold=w.dwell_threshold,
        contributions=contributions,
        rationale=rationale,
    )


@dataclass(frozen=True)
class EmotionLexicon:
    """Named emotions treated as positive or negative; provisional defaults."""

    positive: FrozenSet[str] = frozenset(
        {
            "joy",
            "excitement",
            "contentment",
            "peace",
            "serenity",
            "love",
            "affection",
            "curiosity",
            "interest",
            "wonder",
            "awe",
            "gratitude",
            "hope",
            "pride",
            "delight",
        }
    )
    negative: FrozenSet[str] = frozenset(
        {
            "sadness",
            "anxiety",
            "concern",
            "fear",
            "anger",
            "grief",
            "loneliness",
            "frustration",
            "worry",
            "shame",
            "guilt",
            "despair",
            "dread",
            "boredom",
            "sorrow",
        }
    )


def valence_from_emotions(
    emotions: Optional[Mapping[str, float]],
    lexicon: Optional[EmotionLexicon] = None,
) -> Optional[float]:
    """Return positive minus negative intensity, clamped to [-1, 1].

    Unknown names and non-numeric intensities are ignored. Returns None when
    no known emotion is present, so a missing signal is never a number.
    """
    if not emotions:
        return None
    lex = lexicon or EmotionLexicon()
    positive = negative = 0.0
    seen = False
    for name, intensity in emotions.items():
        try:
            value = float(intensity)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        key = str(name).lower()
        if key in lex.positive:
            positive += value
            seen = True
        elif key in lex.negative:
            negative += value
            seen = True
    if not seen:
        return None
    return _clamp(positive - negative, -1.0, 1.0)
