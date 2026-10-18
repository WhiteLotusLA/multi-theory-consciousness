"""
PAD Affect Model -- Pleasure-Arousal-Dominance
=============================================

Unified dimensional affect model aggregating fragmented emotional
signals into a single (P, A, D) coordinate. Read-only consumer of existing
modules (Damasio, FEP, SNN, GWT, AST, BLT).

Based on Mehrabian & Russell (1974). Used by the Consciousness AI (ACM)
architecture for sensory bid modulation.

Affect coherence (the assessment's affect_coherence input) has two methods:

- ``temporal_structure`` (default): how persistent each PAD dimension is over
  the recent window, measured as lag-1 autocorrelation. Noise scores near
  zero, alternation scores zero, smooth trajectories score near one.
- ``v04_variance``: the v0.4.0 formula, which rewarded variance, so pure noise
  scored at its maximum. Kept for the legacy_v04 preset.

Neither method says anything about a felt state; both describe the time
series of the computed coordinate.

Measured inputs: ``compute(sources, measured=...)`` takes the names of the
inputs that were measured in the cycle. Each dimension is then the weighted
mean over its measured inputs only (an absent input contributes nothing and
the remaining weights are renormalised), and no coordinate is produced when
a dimension has no measured input. The state records the list it was
computed from (``PADState.measured_inputs``); ``measured_dimensions`` reads
it back. A call without a list keeps the fixed-weight arithmetic over every
field of ``PADSources``, defaults included, and records no list.

The label reads ``Neutral`` while pleasure lies inside
``NEUTRAL_PLEASURE_BAND`` around zero; outside the band the octant table
applies.
"""

import dataclasses
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

COHERENCE_METHODS = ("temporal_structure", "v04_variance")

# A dimension whose standard deviation over the window falls below this is
# treated as flat: there are no affect dynamics to be persistent about.
FLAT_STD = 1e-3

# Affect label octants: (pleasure_positive, high_arousal, high_dominance) -> (intense, mild)
AFFECT_LABELS = {
    (True, True, True): ("Exhilarated", "Confident"),
    (True, True, False): ("Delighted", "Excited"),
    (True, False, True): ("Relaxed", "Content"),
    (True, False, False): ("Calm", "Peaceful"),
    (False, True, True): ("Angry", "Determined"),
    (False, True, False): ("Anxious", "Overwhelmed"),
    (False, False, True): ("Bored", "Indifferent"),
    (False, False, False): ("Sad", "Helpless"),
}

# A label threshold, not a measurement: pleasure whose magnitude is below this
# band reads Neutral. The thought generator uses the same constant for the
# valence below which it names no emotion.
NEUTRAL_PLEASURE_BAND = 0.1

# The fixed weights of each dimension, input name to weight. Each table sums
# to 1.0; with a measured list a dimension divides by the weights of its
# measured inputs.
PLEASURE_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "somatic_valence": 0.4,
        "homeostatic_satisfaction": 0.3,
        "snn_valence": 0.3,
    }
)
AROUSAL_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "protoself_arousal": 0.3,
        "snn_spike_rate": 0.2,
        "ignition_rate": 0.25,
        "prediction_error": 0.25,
    }
)
DOMINANCE_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "agency_score": 0.4,
        "voluntary_attention_ratio": 0.3,
        "epistemic_depth_normalized": 0.3,
    }
)

# (dimension, weights, lower bound, upper bound), in the order of PADState.
_DIMENSIONS: Tuple[Tuple[str, Mapping[str, float], float, float], ...] = (
    ("pleasure", PLEASURE_WEIGHTS, -1.0, 1.0),
    ("arousal", AROUSAL_WEIGHTS, 0.0, 1.0),
    ("dominance", DOMINANCE_WEIGHTS, 0.0, 1.0),
)


@dataclass
class PADSources:
    """Input sources for PAD computation. All pre-normalized.

    The defaults are used only by callers that pass no measured list to
    ``PADAffectModel.compute``; with a measured list an input that is not
    named contributes nothing, whatever value it holds here.
    """

    somatic_valence: float = 0.0
    homeostatic_satisfaction: float = 0.0  # [-1, 1] rescaled
    snn_valence: float = 0.0  # joy - sadness [-1, 1]
    protoself_arousal: float = 0.3
    snn_spike_rate: float = 0.0  # [0, 1] normalized
    ignition_rate: float = 0.0  # [0, 1]
    prediction_error: float = 0.0  # [0, 1] clamped
    agency_score: float = 0.5  # meta_confidence [0, 1]
    voluntary_attention_ratio: float = 0.5
    epistemic_depth_normalized: float = 0.0  # depth / 5.0 [0, 1]


_INPUT_NAMES: FrozenSet[str] = frozenset(
    item.name for item in dataclasses.fields(PADSources)
)


@dataclass
class PADState:
    """Current PAD coordinate.

    ``measured_inputs`` names the inputs the coordinate was computed from;
    None means the caller did not say (a call without a measured list, or a
    state built elsewhere), so the coordinate's provenance is unknown.
    """

    pleasure: float  # [-1, 1]
    arousal: float  # [0, 1]
    dominance: float  # [0, 1]
    label: str = ""
    timestamp: float = field(default_factory=time.time)
    measured_inputs: Optional[Tuple[str, ...]] = None


def measured_dimensions(state: Optional[PADState]) -> FrozenSet[str]:
    """The dimensions of ``state`` that have at least one measured input.

    Empty for None and for a state whose ``measured_inputs`` is None.
    """
    if state is None:
        return frozenset()
    inputs = getattr(state, "measured_inputs", None)
    if inputs is None:
        return frozenset()
    names = set(inputs)
    return frozenset(
        dimension
        for dimension, weights, _, _ in _DIMENSIONS
        if any(name in names for name in weights)
    )


def get_affect_label(pleasure: float, arousal: float, dominance: float) -> str:
    """Map (P, A, D) to a human-readable affect label.

    ``Neutral`` when the magnitude of pleasure is below
    ``NEUTRAL_PLEASURE_BAND``; otherwise the octant table, with pleasure
    above zero as the positive half.
    """
    if abs(pleasure) < NEUTRAL_PLEASURE_BAND:
        return "Neutral"
    key = (pleasure > 0, arousal > 0.5, dominance > 0.5)
    intense, mild = AFFECT_LABELS.get(key, ("Neutral", "Neutral"))
    # Use magnitude to pick intense vs mild
    magnitude = abs(pleasure) + abs(arousal - 0.5) + abs(dominance - 0.5)
    return intense if magnitude > 1.0 else mild


def _weighted_sum(
    sources: PADSources,
    weights: Mapping[str, float],
    measured: Optional[FrozenSet[str]] = None,
) -> Optional[float]:
    """Sum of weight times input, left to right in table order.

    With ``measured`` only the named inputs enter; None when none of the
    table's inputs is named.
    """
    total: Optional[float] = None
    for name, weight in weights.items():
        if measured is not None and name not in measured:
            continue
        term = weight * getattr(sources, name)
        total = term if total is None else total + term
    return total


def _measured_names(measured: Iterable[str]) -> Tuple[str, ...]:
    """The measured list as a tuple without repeats; unknown names raise."""
    if isinstance(measured, str):
        raise TypeError("measured must be a collection of input names, not a str")
    names = tuple(dict.fromkeys(str(name) for name in measured))
    unknown = [name for name in names if name not in _INPUT_NAMES]
    if unknown:
        raise ValueError(
            f"unknown PAD input name(s) {unknown}; "
            f"expected PADSources fields {sorted(_INPUT_NAMES)}"
        )
    return names


def _lag1_persistence(values: np.ndarray) -> float:
    """Positive part of the lag-1 autocorrelation; 0 for a flat or bad series."""
    if values.size < 3 or not np.all(np.isfinite(values)):
        return 0.0
    if float(np.std(values)) < FLAT_STD:
        return 0.0
    centred = values - values.mean()
    denominator = float(np.dot(centred, centred))
    if denominator <= 0.0:
        return 0.0
    autocorrelation = float(np.dot(centred[:-1], centred[1:])) / denominator
    return max(0.0, min(1.0, autocorrelation))


class PADAffectModel:
    """Unified Pleasure-Arousal-Dominance affect model.

    Aggregates emotional signals from Damasio, FEP, SNN, GWT, AST, BLT
    into a single (P, A, D) coordinate per consciousness cycle.
    """

    def __init__(
        self,
        history_window: int = 100,
        narrate_affect_line: bool = True,
        coherence_method: str = "temporal_structure",
        coherence_window: int = 20,
        coherence_min_history: int = 10,
    ):
        """
        Args:
            history_window: Number of PAD states kept for coherence scoring
            narrate_affect_line: When False, generate_context() returns an
                empty string; computation and history continue unchanged.
            coherence_method: ``temporal_structure`` (default) or
                ``v04_variance`` (the v0.4.0 formula).
            coherence_window: Number of recent states the temporal method
                scores.
            coherence_min_history: States required before the temporal
                method returns a score; fewer returns None.
        """
        if coherence_method not in COHERENCE_METHODS:
            raise ValueError(
                f"coherence_method must be one of {COHERENCE_METHODS}, "
                f"got {coherence_method!r}"
            )
        if int(coherence_window) < 3:
            raise ValueError(
                f"coherence_window must be at least 3, got {coherence_window}"
            )
        if int(coherence_min_history) < 2:
            raise ValueError(
                "coherence_min_history must be at least 2, "
                f"got {coherence_min_history}"
            )
        self._history: deque = deque(maxlen=history_window)
        self._computation_count = 0
        self.narrate_affect_line = narrate_affect_line
        self.coherence_method = coherence_method
        self.coherence_window = int(coherence_window)
        self.coherence_min_history = int(coherence_min_history)

        logger.info("PADAffectModel initialized")

    @property
    def history_length(self) -> int:
        """Number of PAD states currently held."""
        return len(self._history)

    def compute(
        self,
        sources: PADSources,
        measured: Optional[Iterable[str]] = None,
    ) -> Optional[PADState]:
        """Compute PAD coordinate from source signals.

        Args:
            sources: Pre-normalized input signals.
            measured: Names of the ``PADSources`` fields that were measured.
                None (the default) takes every field as supplied and keeps
                the fixed-weight arithmetic; the state records no list. With
                a collection each dimension is the weighted mean over its
                measured inputs (the sum of weight times input divided by the
                sum of their weights), then clipped, and the state records
                the list. An unknown name raises ValueError.

        Returns:
            PADState with pleasure [-1,1], arousal [0,1], dominance [0,1], and
            label; None when a measured list leaves a dimension without a
            measured input, in which case nothing is recorded and no
            computation is counted.
        """
        names: Optional[Tuple[str, ...]] = None
        values = []
        if measured is None:
            for _, weights, lower, upper in _DIMENSIONS:
                total = _weighted_sum(sources, weights)
                values.append(float(np.clip(total, lower, upper)))
        else:
            names = _measured_names(measured)
            present = frozenset(names)
            for dimension, weights, lower, upper in _DIMENSIONS:
                total = _weighted_sum(sources, weights, present)
                if total is None:
                    logger.debug(
                        "PAD not computed: no measured input for %s", dimension
                    )
                    return None
                mass = math.fsum(w for name, w in weights.items() if name in present)
                values.append(float(np.clip(total / mass, lower, upper)))

        self._computation_count += 1
        pleasure, arousal, dominance = values
        label = get_affect_label(pleasure, arousal, dominance)

        state = PADState(
            pleasure=pleasure,
            arousal=arousal,
            dominance=dominance,
            label=label,
            measured_inputs=names,
        )
        self._history.append(state)

        return state

    def compute_coherence(self) -> Optional[float]:
        """Affect coherence score for the assessment's affect_coherence input.

        Under ``temporal_structure`` the score measures persistence over time:
        for each PAD dimension over the last ``coherence_window`` states, a
        standard deviation below FLAT_STD scores 0 (no dynamics), otherwise
        the score is the positive part of the lag-1 autocorrelation. The
        result is the mean over the three dimensions. White noise has an
        autocorrelation near zero and alternation a negative one, so both
        score low; smooth trajectories score near 1. It does not measure
        agreement with any felt state. Returns None until
        ``coherence_min_history`` states exist (insufficient data).

        Under ``v04_variance`` the v0.4.0 formula runs unchanged, including
        the 0.3 warm-up value below five states.
        """
        if self.coherence_method == "v04_variance":
            return self._v04_variance_coherence()
        return self._temporal_structure_coherence()

    def _temporal_structure_coherence(self) -> Optional[float]:
        if len(self._history) < self.coherence_min_history:
            return None
        recent = list(self._history)[-self.coherence_window :]
        if len(recent) < 3:
            return None
        series = (
            [s.pleasure for s in recent],
            [s.arousal for s in recent],
            [s.dominance for s in recent],
        )
        scores = [_lag1_persistence(np.asarray(x, dtype=np.float64)) for x in series]
        return float(np.mean(scores))

    def _v04_variance_coherence(self) -> float:
        """The v0.4.0 formula. legacy_v04 compatibility, remove in v0.6.0."""
        if len(self._history) < 5:
            return 0.3

        recent = list(self._history)[-20:]
        p_vals = [s.pleasure for s in recent]
        a_vals = [s.arousal for s in recent]
        d_vals = [s.dominance for s in recent]

        stds = [np.std(p_vals), np.std(a_vals), np.std(d_vals)]

        # Temporal variance: emotions should change over time (not flat)
        temporal_var = np.mean(stds)
        variance_score = min(1.0, temporal_var * 5.0)

        # Dimensionality: all three PAD dimensions should show SOME variance
        # (not collapsed to a single axis)
        dim_active = sum(1 for s in stds if s > 0.01) / 3.0

        return float(0.5 * variance_score + 0.5 * dim_active)

    def get_latest(self) -> Optional[PADState]:
        """Return most recent PAD state."""
        return self._history[-1] if self._history else None

    def generate_context(self) -> str:
        """Generate context string for the reasoning module.

        Returns an empty string when narrate_affect_line is False, so no
        affect label is handed to the reasoning module as a felt state.
        """
        if not self.narrate_affect_line:
            return ""
        latest = self.get_latest()
        if not latest:
            return ""
        return (
            f"Current affect: {latest.label.lower()} "
            f"(P={latest.pleasure:.2f}, A={latest.arousal:.2f}, D={latest.dominance:.2f})"
        )

    def to_state_dict(self) -> Dict[str, Any]:
        """Export state for ConsciousnessStateManager."""
        return {
            "scalars": {
                "computation_count": self._computation_count,
            },
            "arrays": {},
            "histories": {
                "pad_history": [
                    {
                        "p": s.pleasure,
                        "a": s.arousal,
                        "d": s.dominance,
                        "label": s.label,
                        "t": s.timestamp,
                        "measured": (
                            list(s.measured_inputs)
                            if s.measured_inputs is not None
                            else None
                        ),
                    }
                    for s in self._history
                ],
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from persistence."""
        scalars = state.get("scalars", {})
        histories = state.get("histories", {})
        self._computation_count = scalars.get("computation_count", 0)
        if "pad_history" in histories:
            self._history.clear()
            for entry in histories["pad_history"]:
                inputs = entry.get("measured")
                self._history.append(
                    PADState(
                        pleasure=entry["p"],
                        arousal=entry["a"],
                        dominance=entry["d"],
                        label=entry.get("label", ""),
                        timestamp=entry.get("t", 0.0),
                        measured_inputs=(
                            tuple(str(name) for name in inputs)
                            if inputs is not None
                            else None
                        ),
                    )
                )
        logger.info(
            f"PADAffectModel restored: {self._computation_count} prior computations, "
            f"{len(self._history)} history entries"
        )
