"""
Theory Interaction Manager - cross-theory modulation
====================================================

The adversarial collaboration that tested GWT against IIT (Nature 2025)
found that neither theory alone explains the data, which suggests the
theories should constrain and modulate each other rather than run side by
side.

This module computes cross-theory modulation signals from each cycle's
consciousness state and holds them for the NEXT cycle (store and modulate
next cycle, the same pattern as the FEP to GWT prediction-error feedback).

Six interactions:

  1. Damasio to GWT: protoself instability heightens workspace salience
  2. IIT to GWT: higher Phi enriches the competition
  3. HOT to FEP: metacognitive depth raises prediction confidence
  4. FEP to Damasio: free energy becomes somatic arousal
  5. RPT to BLT: deep recurrence strengthens binding
  6. BLT to AST: epistemic depth widens the attention field

Every signal is clamped to [-1, 1] and smoothed with an exponential moving
average. A state field that is missing, None or non-finite contributes no
modulation.
"""

import logging
import math
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TheoryInteractionConfig:
    """Scale constants and smoothing for the six interactions.

    ``smoothing`` is the weight the previous smoothed value keeps:
    ``new_smoothed = smoothing * old + (1 - smoothing) * raw``. Zero
    disables smoothing.
    """

    smoothing: float = 0.3

    # Signal strength per interaction
    somatic_scale: float = 0.15  # Damasio to GWT salience
    phi_scale: float = 0.10  # IIT Phi to GWT competition
    hot_precision_scale: float = 0.12  # HOT to FEP confidence
    vfe_arousal_scale: float = 0.20  # free energy to Damasio arousal
    recurrence_scale: float = 0.10  # RPT to BLT binding
    epistemic_scale: float = 0.08  # BLT depth to AST capacity

    # Protoself stability treated as neutral (below heightens, above calms)
    somatic_set_point: float = 0.5
    # Surprise (max of VFE and prediction error) at which arousal saturates
    surprise_reference: float = 0.5
    # HOT count that gives the full precision bias
    hot_saturation_count: int = 3
    # Epistemic depth that gives the full attention expansion
    epistemic_reference_depth: int = 5

    def __post_init__(self):
        if not 0.0 <= float(self.smoothing) < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {self.smoothing}")
        for name in ("surprise_reference",):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("hot_saturation_count", "epistemic_reference_depth"):
            if int(getattr(self, name)) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)}"
                )


@dataclass
class TheoryModulations:
    """Cross-theory modulation signals for the next consciousness cycle.

    Each field is a signal in [-1, 1] that its target theory consumes:
    positive is excitatory, negative inhibitory, 0 no modulation.
    """

    # Damasio to GWT: an unstable protoself heightens competition, a stable
    # one lowers it slightly
    somatic_salience_bias: float = 0.0
    # IIT to GWT: high Phi gives richer, more integrated competition
    phi_competition_bias: float = 0.0
    # HOT to FEP: more HOTs give more confidence in the generative model;
    # none at all lowers it (acting without awareness)
    metacognitive_precision_bias: float = 0.0
    # FEP to Damasio: high free energy registers as somatic stress
    free_energy_arousal: float = 0.0
    # RPT to BLT: deeper recurrence gives stronger binding
    recurrence_binding_boost: float = 0.0
    # BLT to AST: deeper self-modelling gives a broader attention field
    epistemic_attention_expansion: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    def active_count(self) -> int:
        """How many modulations are non-trivial (absolute value above 0.01)."""
        return sum(1 for v in self.to_dict().values() if abs(v) > 0.01)


_SIGNAL_NAMES = tuple(f.name for f in fields(TheoryModulations))


class TheoryInteractionManager:
    """Computes cross-theory modulations from a consciousness state.

    Call ``update_from_state`` after each cycle; the returned modulations
    are consumed by the target theories in the next cycle.

    Args:
        config: Scale constants and smoothing.
        phi_normalizer: Maps a raw Phi value to [0, 1]. The default clamps
            the raw value to [0, 1]; a caller with an empirical reference
            (for example a PhiTracker's rolling percentile) passes it here.
    """

    def __init__(
        self,
        config: Optional[TheoryInteractionConfig] = None,
        phi_normalizer: Optional[Callable[[float], float]] = None,
    ):
        self.config = config or TheoryInteractionConfig()
        self._current = TheoryModulations()
        self._cycle_count = 0
        self._normalize_phi: Callable[[float], float] = (
            phi_normalizer if phi_normalizer is not None else _unit_clamp
        )

    @property
    def modulations(self) -> TheoryModulations:
        """Current smoothed modulation signals for the next cycle."""
        return self._current

    def update_from_state(self, state: Any) -> TheoryModulations:
        """Compute this cycle's signals, smooth them and return the result."""
        cfg = self.config
        raw = TheoryModulations()

        # 1. Damasio to GWT
        stability = _finite(getattr(state, "protoself_stability", None))
        if stability is not None:
            raw.somatic_salience_bias = _clamp(
                (cfg.somatic_set_point - stability) * 2.0 * cfg.somatic_scale
            )

        # 2. IIT to GWT
        phi = _finite(getattr(state, "architectural_phi", None))
        if phi is not None and phi > 0.0:
            normalized = _finite(self._normalize_phi(phi))
            if normalized is not None:
                raw.phi_competition_bias = _clamp(normalized * cfg.phi_scale)

        # 3. HOT to FEP: zero HOTs is a measurement (slightly negative);
        # a state without the field says nothing
        hots = getattr(state, "higher_order_thoughts", None)
        if hots is not None:
            count = len(hots)
            if count == 0:
                raw.metacognitive_precision_bias = -0.5 * cfg.hot_precision_scale
            else:
                raw.metacognitive_precision_bias = _clamp(
                    min(count / cfg.hot_saturation_count, 1.0) * cfg.hot_precision_scale
                )

        # 4. FEP to Damasio: VFE and prediction error both count as surprise
        surprises = [
            value
            for value in (
                _finite(getattr(state, "variational_free_energy", None)),
                _finite(getattr(state, "prediction_error", None)),
            )
            if value is not None
        ]
        if surprises:
            surprise = max(surprises)
            if surprise > 0.0:
                normalized = min(surprise / cfg.surprise_reference, 1.0)
                raw.free_energy_arousal = _clamp(normalized * cfg.vfe_arousal_scale)

        # 5. RPT to BLT
        recurrence = _finite(getattr(state, "rpt_deep_recurrence", None))
        if recurrence is not None and recurrence > 0.0:
            raw.recurrence_binding_boost = _clamp(recurrence * cfg.recurrence_scale)

        # 6. BLT to AST: diminishing returns in depth
        depth = _finite(getattr(state, "epistemic_depth", None))
        if depth is not None and depth > 0.0:
            expansion = math.log1p(depth) / math.log1p(cfg.epistemic_reference_depth)
            raw.epistemic_attention_expansion = _clamp(expansion * cfg.epistemic_scale)

        self._current = _smooth(self._current, raw, cfg.smoothing)
        self._cycle_count += 1

        if self._cycle_count % 50 == 1 and self._current.active_count() > 0:
            logger.debug(
                "Theory interactions: %d/6 active (somatic=%+.3f, phi=%+.3f, "
                "hot_to_fep=%+.3f)",
                self._current.active_count(),
                self._current.somatic_salience_bias,
                self._current.phi_competition_bias,
                self._current.metacognitive_precision_bias,
            )
        return self._current

    def get_status(self) -> Dict[str, Any]:
        """Current interaction state for statistics and reports."""
        return {
            "cycle_count": self._cycle_count,
            "modulations": self._current.to_dict(),
            "active_interactions": self._current.active_count(),
        }

    def to_state_dict(self) -> Dict[str, Any]:
        """Persist the smoothed signals and the cycle count as scalars."""
        scalars: Dict[str, Any] = {"cycle_count": self._cycle_count}
        scalars.update(self._current.to_dict())
        return {"scalars": scalars, "arrays": {}, "histories": {}}

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore typed modulations; unknown or non-finite values are skipped."""
        s = (state or {}).get("scalars", {}) or {}
        restored = TheoryModulations()
        for name in _SIGNAL_NAMES:
            value = _finite(s.get(name))
            if value is not None:
                setattr(restored, name, _clamp(value))
        self._current = restored
        try:
            self._cycle_count = max(0, int(s.get("cycle_count", 0)))
        except (TypeError, ValueError):
            self._cycle_count = 0


def _finite(value: Any) -> Optional[float]:
    """The value as a finite float, or None (bools and non-numbers too)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unit_clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _smooth(
    old: TheoryModulations, new: TheoryModulations, alpha: float
) -> TheoryModulations:
    """Exponential moving average of every signal."""
    return TheoryModulations(
        **{
            name: alpha * getattr(old, name) + (1.0 - alpha) * getattr(new, name)
            for name in _SIGNAL_NAMES
        }
    )
