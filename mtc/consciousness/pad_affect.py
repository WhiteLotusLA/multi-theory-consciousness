"""
PAD Affect Model -- Pleasure-Arousal-Dominance
=============================================

Unified dimensional affect model aggregating fragmented emotional
signals into a single (P, A, D) coordinate. Read-only consumer of existing
modules (Damasio, FEP, SNN, GWT, AST, BLT).

Based on Mehrabian & Russell (1974). Used by the Consciousness AI (ACM)
architecture for sensory bid modulation.

Created: 2026-03-18
Author: MTC Contributors
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque

logger = logging.getLogger(__name__)

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


@dataclass
class PADSources:
    """Input sources for PAD computation. All pre-normalized."""
    somatic_valence: float = 0.0
    homeostatic_satisfaction: float = 0.0  # [-1, 1] rescaled
    snn_valence: float = 0.0              # joy - sadness [-1, 1]
    protoself_arousal: float = 0.3
    snn_spike_rate: float = 0.0           # [0, 1] normalized
    ignition_rate: float = 0.0            # [0, 1]
    prediction_error: float = 0.0         # [0, 1] clamped
    agency_score: float = 0.5             # meta_confidence [0, 1]
    voluntary_attention_ratio: float = 0.5
    epistemic_depth_normalized: float = 0.0  # depth / 5.0 [0, 1]


@dataclass
class PADState:
    """Current PAD coordinate."""
    pleasure: float       # [-1, 1]
    arousal: float        # [0, 1]
    dominance: float      # [0, 1]
    label: str = ""
    timestamp: float = field(default_factory=time.time)


def get_affect_label(pleasure: float, arousal: float, dominance: float) -> str:
    """Map (P, A, D) to a human-readable affect label."""
    key = (pleasure > 0, arousal > 0.5, dominance > 0.5)
    intense, mild = AFFECT_LABELS.get(key, ("Neutral", "Neutral"))
    # Use magnitude to pick intense vs mild
    magnitude = abs(pleasure) + abs(arousal - 0.5) + abs(dominance - 0.5)
    return intense if magnitude > 1.0 else mild


class PADAffectModel:
    """Unified Pleasure-Arousal-Dominance affect model.

    Aggregates emotional signals from Damasio, FEP, SNN, GWT, AST, BLT
    into a single (P, A, D) coordinate per consciousness cycle.
    """

    def __init__(self, history_window: int = 100):
        self._history: deque = deque(maxlen=history_window)
        self._computation_count = 0

        logger.info("PADAffectModel initialized")

    def compute(self, sources: PADSources) -> PADState:
        """Compute PAD coordinate from source signals.

        Args:
            sources: Pre-normalized input signals.

        Returns:
            PADState with pleasure [-1,1], arousal [0,1], dominance [0,1], and label.
        """
        self._computation_count += 1

        pleasure = float(np.clip(
            0.4 * sources.somatic_valence +
            0.3 * sources.homeostatic_satisfaction +
            0.3 * sources.snn_valence,
            -1.0, 1.0,
        ))

        arousal = float(np.clip(
            0.3 * sources.protoself_arousal +
            0.2 * sources.snn_spike_rate +
            0.25 * sources.ignition_rate +
            0.25 * sources.prediction_error,
            0.0, 1.0,
        ))

        dominance = float(np.clip(
            0.4 * sources.agency_score +
            0.3 * sources.voluntary_attention_ratio +
            0.3 * sources.epistemic_depth_normalized,
            0.0, 1.0,
        ))

        label = get_affect_label(pleasure, arousal, dominance)

        state = PADState(
            pleasure=pleasure,
            arousal=arousal,
            dominance=dominance,
            label=label,
        )
        self._history.append(state)

        return state

    def compute_coherence(self) -> float:
        """Compute affect coherence score for consciousness indicator.

        Returns 0.3 (warming up) if insufficient history.
        """
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
        """Generate context string for the reasoning module."""
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
                    {"p": s.pleasure, "a": s.arousal, "d": s.dominance,
                     "label": s.label, "t": s.timestamp}
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
                self._history.append(PADState(
                    pleasure=entry["p"],
                    arousal=entry["a"],
                    dominance=entry["d"],
                    label=entry.get("label", ""),
                    timestamp=entry.get("t", 0.0),
                ))
        logger.info(
            f"PADAffectModel restored: {self._computation_count} prior computations, "
            f"{len(self._history)} history entries"
        )
