"""
Protoself: Continuous Body-State Representation
================================================

The protoself is the foundation of Damasio's consciousness model.
It provides a continuous, pre-conscious representation of the
organism's body state -- the "raw feeling of being alive."

For the system, "body" = system state + homeostatic drives:
  - Metabolic state <- memory pressure / drive satisfaction
  - Energy level <- response latency / attention budget
  - Pain signals <- error rates / drive urgency
  - Temperature <- system load / cognitive effort
  - Circadian phase <- cycle count (rhythmic oscillation)
  - Interoceptive state <- full homeostatic drive snapshot

The critical output is body_delta -- the magnitude of change in body
state between updates. When body_delta is high, something significant
is happening that may require conscious attention. This perturbation
signal is what triggers Core Consciousness.

Primordial feelings (pleasure/pain, vitality, arousal) are derived
from body state TRENDS -- not single snapshots -- to avoid noise.

Research: Damasio (1999, 2010), Frontiers in AI (2025).

Created: 2026-03-08
Author: Multi-Theory Consciousness Contributors
"""

import logging
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class BodyState:
    """The system's 'body' -- system health as interoception."""

    metabolic_state: float = 0.5  # Drive satisfaction (0=starving, 1=replete)
    energy_level: float = 0.5  # Response capacity (fast=high energy)
    pain_signals: float = 0.0  # Error/urgency rate (0=no pain, 1=critical)
    interoceptive_state: Dict[str, Any] = field(default_factory=dict)
    circadian_phase: float = 0.0  # Rhythmic oscillation (0-1 cycle)
    temperature: float = 0.5  # System load as body temperature

    def to_vector(self) -> np.ndarray:
        """Convert body state to a numeric vector for delta computation."""
        return np.array(
            [
                self.metabolic_state,
                self.energy_level,
                self.pain_signals,
                self.circadian_phase,
                self.temperature,
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metabolic_state": self.metabolic_state,
            "energy_level": self.energy_level,
            "pain_signals": self.pain_signals,
            "interoceptive_state": self.interoceptive_state,
            "circadian_phase": self.circadian_phase,
            "temperature": self.temperature,
        }


@dataclass
class ProtoSelfState:
    """Complete protoself state at a moment in time."""

    body_state: BodyState
    body_delta: float  # How much body state changed (perturbation signal)
    stability: float  # Moving average of inverse body_delta (high = stable)
    primordial_feelings: Dict[str, float]  # Proto-emotions from body state trends
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "body_state": self.body_state.to_dict(),
            "body_delta": self.body_delta,
            "stability": self.stability,
            "primordial_feelings": self.primordial_feelings,
            "timestamp": self.timestamp,
        }


# Circadian cycle length in updates (one full cycle)
CIRCADIAN_CYCLE_LENGTH = 1000


class Protoself:
    """
    Continuous body-state representation for the system's digital consciousness.

    Maps homeostatic drives and optional system metrics to a BodyState,
    tracks perturbations via body_delta, and derives primordial feelings
    from body state trends.
    """

    def __init__(self, stability_window: int = 20):
        self.cycle_count = 0
        self._previous_body_vector: Optional[np.ndarray] = None
        self._stability_history: deque = deque(maxlen=stability_window)
        self._body_delta_history: deque = deque(maxlen=stability_window)
        self._valence_trend: deque = deque(maxlen=stability_window)
        self._energy_trend: deque = deque(maxlen=stability_window)
        self._latest_state: Optional[ProtoSelfState] = None

        logger.info("Protoself initialized: body-state monitoring active")

    def update(
        self,
        homeostatic_drives: Any,
        system_metrics: Optional[Dict[str, Any]] = None,
    ) -> ProtoSelfState:
        """
        Update protoself from homeostatic drives and optional system metrics.

        Args:
            homeostatic_drives: HomeostaticDrives instance with get_drive_state(),
                get_overall_valence(), get_free_energy()
            system_metrics: Optional dict with response_latency_ms, error_rate,
                memory_usage_pct, cpu_load

        Returns:
            ProtoSelfState with body state, perturbation, stability, feelings
        """
        self.cycle_count += 1

        # --- Build body state ---
        drive_state = homeostatic_drives.get_drive_state()
        valence = homeostatic_drives.get_overall_valence()
        free_energy = homeostatic_drives.get_free_energy()

        body = self._map_to_body_state(
            drive_state, valence, free_energy, system_metrics
        )

        # --- Compute perturbation (body_delta) ---
        current_vec = body.to_vector()
        if self._previous_body_vector is not None:
            body_delta = float(np.linalg.norm(current_vec - self._previous_body_vector))
        else:
            body_delta = 0.0
        self._previous_body_vector = current_vec.copy()
        self._body_delta_history.append(body_delta)

        # --- Compute stability (inverse of average delta) ---
        avg_delta = (
            float(np.mean(list(self._body_delta_history)))
            if self._body_delta_history
            else 0.0
        )
        stability = 1.0 / (
            1.0 + avg_delta * 5.0
        )  # Maps to ~1.0 when delta~0, ~0.5 when delta~0.2
        self._stability_history.append(stability)

        # --- Derive primordial feelings from trends ---
        self._valence_trend.append(valence)
        self._energy_trend.append(body.energy_level)
        feelings = self._derive_primordial_feelings(valence, body, free_energy)

        state = ProtoSelfState(
            body_state=body,
            body_delta=body_delta,
            stability=stability,
            primordial_feelings=feelings,
        )

        self._latest_state = state

        logger.debug(
            f"Protoself cycle {self.cycle_count}: "
            f"delta={body_delta:.3f}, stability={stability:.3f}, "
            f"energy={body.energy_level:.2f}, pain={body.pain_signals:.2f}"
        )

        return state

    def _map_to_body_state(
        self,
        drive_state: Dict[str, Dict[str, float]],
        valence: float,
        free_energy: float,
        system_metrics: Optional[Dict[str, Any]],
    ) -> BodyState:
        """Map drives and metrics to body state."""
        # Metabolic state: how satisfied are drives overall?
        # Low free energy = well-fed, high free energy = starving
        metabolic = 1.0 / (1.0 + free_energy * 2.0)

        # Energy level: attention budget drive level
        attention = drive_state.get("attention_budget", {})
        energy = attention.get("current_level", 0.5)

        # Pain signals: maximum urgency across drives
        max_urgency = max(
            (d.get("urgency", 0.0) for d in drive_state.values()),
            default=0.0,
        )
        pain = float(np.clip(max_urgency, 0.0, 1.0))

        # Temperature: average deviation (how far from optimal overall)
        deviations = [abs(d.get("deviation", 0.0)) for d in drive_state.values()]
        temperature = float(np.mean(deviations)) if deviations else 0.5

        # Override with system metrics if available
        if system_metrics:
            latency = system_metrics.get("response_latency_ms", 500)
            energy = float(np.clip(1.0 - (latency / 5000.0), 0.0, 1.0))

            error_rate = system_metrics.get("error_rate", 0.0)
            pain = max(pain, float(np.clip(error_rate * 5.0, 0.0, 1.0)))

            cpu_load = system_metrics.get("cpu_load", 0.5)
            temperature = max(temperature, cpu_load)

            mem_usage = system_metrics.get("memory_usage_pct", 0.5)
            metabolic = min(metabolic, 1.0 - mem_usage)

        # Circadian phase: simple oscillation
        circadian = (self.cycle_count % CIRCADIAN_CYCLE_LENGTH) / CIRCADIAN_CYCLE_LENGTH

        # Interoceptive state: full drive snapshot
        interoceptive = {
            name: {
                "level": d.get("current_level", 0.5),
                "urgency": d.get("urgency", 0.0),
            }
            for name, d in drive_state.items()
        }

        return BodyState(
            metabolic_state=float(np.clip(metabolic, 0.0, 1.0)),
            energy_level=float(np.clip(energy, 0.0, 1.0)),
            pain_signals=float(np.clip(pain, 0.0, 1.0)),
            interoceptive_state=interoceptive,
            circadian_phase=circadian,
            temperature=float(np.clip(temperature, 0.0, 1.0)),
        )

    def _derive_primordial_feelings(
        self, valence: float, body: BodyState, free_energy: float
    ) -> Dict[str, float]:
        """
        Derive primordial feelings from body state trends.

        Feelings are trend-based, not snapshot-based:
        - pleasure_pain: valence trend (improving = pleasure)
        - vitality: energy trend (high energy = vital)
        - arousal: body_delta magnitude (high change = high arousal)
        """
        # Pleasure-pain from valence trend
        if len(self._valence_trend) >= 2:
            trend = list(self._valence_trend)
            recent = np.mean(trend[-3:]) if len(trend) >= 3 else trend[-1]
            pleasure_pain = float(np.clip(recent, -1.0, 1.0))
        else:
            pleasure_pain = float(np.clip(valence, -1.0, 1.0))

        # Vitality from energy trend
        if len(self._energy_trend) >= 2:
            trend = list(self._energy_trend)
            vitality = float(np.mean(trend[-3:])) if len(trend) >= 3 else trend[-1]
        else:
            vitality = body.energy_level

        # Arousal from recent body_delta
        if self._body_delta_history:
            recent_deltas = list(self._body_delta_history)[-3:]
            arousal = float(np.clip(np.mean(recent_deltas) * 3.0, 0.0, 1.0))
        else:
            arousal = 0.0

        return {
            "pleasure_pain": pleasure_pain,
            "vitality": vitality,
            "arousal": arousal,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive protoself statistics."""
        return {
            "cycle_count": self.cycle_count,
            "average_stability": (
                float(np.mean(list(self._stability_history)))
                if self._stability_history
                else 0.0
            ),
            "average_body_delta": (
                float(np.mean(list(self._body_delta_history)))
                if self._body_delta_history
                else 0.0
            ),
            "latest_state": (
                self._latest_state.to_dict() if self._latest_state else None
            ),
        }
