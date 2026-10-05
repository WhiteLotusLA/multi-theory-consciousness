"""
Affect-Modulated Learning Rate
==============================

Modulates the learning rate from the affect coordinate, following the
Yerkes-Dodson pattern: moderate arousal learns best, while very high arousal
(overwhelm) and very low arousal (disengagement) learn less. Emotional
significance, the product of valence magnitude and arousal, boosts both the
rate and the retention weight of an experience.

Inputs are the ``PADState`` ranges of ``mtc.consciousness.pad_affect``:
pleasure in [-1, 1], arousal and dominance in [0, 1]; out-of-range values are
clamped. The base rate defaults to ``Settings.LEARNING_RATE``, read at call
time. The module is stateless and persists nothing. When no affect state is
available (``pad`` is None) the rate is the unmodulated base rate and the
retention multiplier is 1.0.
"""

import logging
import math
from typing import Optional

from mtc.consciousness.pad_affect import PADState
from mtc.core.config import get_settings

logger = logging.getLogger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


class AffectModulatedRate:
    """Computes an affect-modulated learning rate and retention multiplier."""

    # The effective rate stays within [base * MIN_MULT, base * MAX_MULT].
    MIN_MULT = 0.3
    MAX_MULT = 3.0
    # Retention multiplier bounds.
    MIN_RETENTION = 0.5
    MAX_RETENTION = 2.0
    # Arousal at which the inverted-U curve peaks.
    PEAK_AROUSAL = 0.6

    def __init__(self, base_rate: Optional[float] = None) -> None:
        self._base_rate = base_rate

    @property
    def base_rate(self) -> float:
        """The configured base rate (``LEARNING_RATE`` unless one was passed)."""
        if self._base_rate is not None:
            return float(self._base_rate)
        return float(get_settings().LEARNING_RATE)

    def rate(self, pad: Optional[PADState]) -> float:
        """Return the effective learning rate for an affect state.

        The multiplier is the product of an inverted-U arousal curve peaking at
        0.6, a significance boost of ``1 + |pleasure| * arousal`` and a
        dominance factor of ``0.8 + 0.2 * dominance`` (low dominance dampens
        learning slightly), clamped to [0.3, 3.0] times the base.
        """
        base = self.base_rate
        if pad is None:
            return base
        pleasure = _clamp(pad.pleasure, -1.0, 1.0)
        arousal = _clamp(pad.arousal, 0.0, 1.0)
        dominance = _clamp(pad.dominance, 0.0, 1.0)
        arousal_factor = math.exp(-((arousal - self.PEAK_AROUSAL) ** 2) / 0.18)
        significance_boost = 1.0 + abs(pleasure) * arousal
        dominance_factor = 0.8 + 0.2 * dominance
        multiplier = _clamp(
            arousal_factor * significance_boost * dominance_factor,
            self.MIN_MULT,
            self.MAX_MULT,
        )
        effective = base * multiplier
        if abs(multiplier - 1.0) > 0.5:
            logger.debug(
                "Affect-modulated learning rate %.4f (%.2fx)", effective, multiplier
            )
        return effective

    def retention(self, pad: Optional[PADState]) -> float:
        """Return the retention multiplier in [0.5, 2.0] for consolidation."""
        if pad is None:
            return 1.0
        pleasure = _clamp(pad.pleasure, -1.0, 1.0)
        arousal = _clamp(pad.arousal, 0.0, 1.0)
        significance = abs(pleasure) * arousal
        return _clamp(
            self.MIN_RETENTION + 1.5 * significance,
            self.MIN_RETENTION,
            self.MAX_RETENTION,
        )
