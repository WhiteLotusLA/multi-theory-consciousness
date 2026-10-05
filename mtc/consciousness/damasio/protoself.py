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
  - Circadian phase <- a pluggable rhythm (see below)
  - Interoceptive state <- full homeostatic drive snapshot

Circadian models (the ``circadian`` constructor argument):
  - "sinusoid" (default): a smooth deterministic sinusoid over the cycle
    count with period CIRCADIAN_CYCLE_LENGTH, so the phase never jumps.
  - "v04_sawtooth": the v0.4.0 sawtooth over the same period, kept so the
    legacy preset reproduces the published numbers. It drops from 1 to 0
    once per period, which shows up as a body_delta spike.
  - "wall_clock": a two-process model on real time in the configured
    timezone, with configurable wake and sleep hours.

System metrics are optional. gather_system_body_metrics() returns only the
keys it can actually measure, and the body mapping overrides a channel only
when its key is present, so nothing is fabricated from a default.

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
import math
import numpy as np
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone as dt_timezone, tzinfo
from typing import Any, Callable, Dict, Optional
from collections import deque
from zoneinfo import ZoneInfo

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
    time_period: Optional[str] = None  # Set only under the wall-clock model

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
        data = {
            "metabolic_state": self.metabolic_state,
            "energy_level": self.energy_level,
            "pain_signals": self.pain_signals,
            "interoceptive_state": self.interoceptive_state,
            "circadian_phase": self.circadian_phase,
            "temperature": self.temperature,
        }
        if self.time_period is not None:
            data["time_period"] = self.time_period
        return data

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BodyState":
        return cls(
            metabolic_state=float(raw.get("metabolic_state", 0.5)),
            energy_level=float(raw.get("energy_level", 0.5)),
            pain_signals=float(raw.get("pain_signals", 0.0)),
            interoceptive_state=dict(raw.get("interoceptive_state") or {}),
            circadian_phase=float(raw.get("circadian_phase", 0.0)),
            temperature=float(raw.get("temperature", 0.5)),
            time_period=raw.get("time_period"),
        )


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

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ProtoSelfState":
        return cls(
            body_state=BodyState.from_dict(raw.get("body_state") or {}),
            body_delta=float(raw.get("body_delta", 0.0)),
            stability=float(raw.get("stability", 1.0)),
            primordial_feelings={
                str(k): float(v)
                for k, v in (raw.get("primordial_feelings") or {}).items()
            },
            timestamp=float(raw.get("timestamp", time.time())),
        )


# Circadian cycle length in updates (one full cycle)
CIRCADIAN_CYCLE_LENGTH = 1000

CIRCADIAN_MODES = ("sinusoid", "v04_sawtooth", "wall_clock")


def sinusoid_circadian(cycle_count: int, period: int = CIRCADIAN_CYCLE_LENGTH) -> float:
    """Smooth circadian phase in [0, 1]: 0 at the start of each period, 1 at
    its midpoint. The largest step between consecutive cycles is pi / period."""
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * (cycle_count % period) / period)


def sawtooth_circadian(cycle_count: int, period: int = CIRCADIAN_CYCLE_LENGTH) -> float:
    """The v0.4.0 circadian phase: a sawtooth that wraps from 1 to 0."""
    return (cycle_count % period) / period


def wall_clock_alertness(
    hour: float, wake_hour: int = 6, sleep_hour: int = 22
) -> float:
    """Circadian alertness from the hour of day, in [0, 1].

    A two-process model in the spirit of Borbely (1982):
      - Process C, the circadian drive: a sinusoid that peaks mid-afternoon
        (15:00) and bottoms out at 03:00.
      - Process S, sleep pressure: builds linearly from the wake hour and is
        relieved during the sleep period. Over the last waking hour it ramps
        down to zero so alertness has no jump at the sleep hour.

    Wake and sleep hours may wrap past midnight (for example 20 and 4).
    """
    hour = hour % 24.0
    circadian_c = 0.5 + 0.5 * math.sin(2.0 * math.pi * (hour - 9.0) / 24.0)

    awake_span = (sleep_hour - wake_hour) % 24 or 24
    hours_awake = (hour - wake_hour) % 24.0
    if hours_awake < awake_span:
        sleep_pressure = min(1.0, hours_awake / awake_span)
        hours_to_sleep = awake_span - hours_awake
        if hours_to_sleep < 1.0:
            sleep_pressure *= hours_to_sleep
    else:
        sleep_pressure = 0.0

    return max(0.0, min(1.0, circadian_c * (1.0 - 0.4 * sleep_pressure)))


def _get_time_period(hour: float) -> str:
    """Name the period of the day for an hour in [0, 24)."""
    hour = hour % 24.0
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    if 21 <= hour < 24:
        return "night"
    return "late_night"


def _resolve_timezone(name: str) -> tzinfo:
    if name.upper() == "UTC":
        return dt_timezone.utc
    return ZoneInfo(name)


_CPU_SAMPLER_PRIMED = False


def gather_system_body_metrics(
    cycle_latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Collect real system metrics for Protoself.update(system_metrics=...).

    Every key comes from a genuine measurement or is omitted:
      - ``response_latency_ms``: the caller's last cycle duration, only when
        given (never a default).
      - ``cpu_load``: psutil CPU percent since the previous sample, in [0, 1].
        The first sample in a process has no interval behind it, so it only
        primes the sampler and the key is omitted.
      - ``memory_usage_pct``: psutil virtual memory percent, in [0, 1].

    ``error_rate`` is not populated; no generic error-rate source exists, so
    the pain channel stays on drive urgency. psutil is optional and imported
    lazily; without it only the latency key can appear.
    """
    global _CPU_SAMPLER_PRIMED
    metrics: Dict[str, Any] = {}

    if cycle_latency_ms is not None:
        metrics["response_latency_ms"] = float(cycle_latency_ms)

    try:
        import psutil
    except ImportError:
        logger.debug("gather_system_body_metrics: psutil not installed")
        return metrics

    try:
        cpu = float(psutil.cpu_percent(interval=None)) / 100.0
        if _CPU_SAMPLER_PRIMED:
            metrics["cpu_load"] = cpu
        _CPU_SAMPLER_PRIMED = True
        metrics["memory_usage_pct"] = float(psutil.virtual_memory().percent) / 100.0
    except Exception as exc:  # psutil present but unable to sample
        logger.debug(f"gather_system_body_metrics: psutil sample failed ({exc})")

    return metrics


class Protoself:
    """
    Continuous body-state representation for the system's digital consciousness.

    Maps homeostatic drives and optional system metrics to a BodyState,
    tracks perturbations via body_delta, and derives primordial feelings
    from body state trends. Sensory input of any modality can register an
    emotional body signal that modulates the next update.
    """

    # Body-state perturbations for common emotion labels, used by
    # register_sensory_label(). valence_shift moves the pleasure/pain
    # feeling, arousal moves energy and temperature, pain adds distress
    # (Damasio's somatic marker hypothesis).
    SENSORY_EMOTION_BODY_MAP: Dict[str, Dict[str, float]] = {
        "joy": {"valence_shift": 0.15, "arousal": 0.3, "pain": 0.0},
        "happy": {"valence_shift": 0.15, "arousal": 0.3, "pain": 0.0},
        "sadness": {"valence_shift": -0.15, "arousal": -0.1, "pain": 0.1},
        "sad": {"valence_shift": -0.15, "arousal": -0.1, "pain": 0.1},
        "anger": {"valence_shift": -0.1, "arousal": 0.4, "pain": 0.2},
        "angry": {"valence_shift": -0.1, "arousal": 0.4, "pain": 0.2},
        "fear": {"valence_shift": -0.2, "arousal": 0.5, "pain": 0.15},
        "fearful": {"valence_shift": -0.2, "arousal": 0.5, "pain": 0.15},
        "surprise": {"valence_shift": 0.05, "arousal": 0.4, "pain": 0.0},
        "surprised": {"valence_shift": 0.05, "arousal": 0.4, "pain": 0.0},
        "disgust": {"valence_shift": -0.15, "arousal": 0.2, "pain": 0.1},
        "disgusted": {"valence_shift": -0.15, "arousal": 0.2, "pain": 0.1},
        "neutral": {"valence_shift": 0.0, "arousal": 0.0, "pain": 0.0},
    }

    def __init__(
        self,
        stability_window: int = 20,
        circadian: str = "sinusoid",
        timezone: str = "UTC",
        wake_hour: int = 6,
        sleep_hour: int = 22,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        """
        Args:
            stability_window: Length of the trend and stability histories
            circadian: "sinusoid" (default), "v04_sawtooth" or "wall_clock"
            timezone: IANA timezone name for the wall-clock model
            wake_hour: Hour (0-23) the waking period starts
            sleep_hour: Hour (0-23) the sleep period starts
            clock: Optional callable returning the current datetime; the
                wall-clock model uses it instead of the system clock
        """
        if circadian not in CIRCADIAN_MODES:
            raise ValueError(
                f"circadian must be one of {CIRCADIAN_MODES}, got {circadian!r}"
            )
        for label, hour in (("wake_hour", wake_hour), ("sleep_hour", sleep_hour)):
            if not 0 <= int(hour) <= 23:
                raise ValueError(f"{label} must be between 0 and 23, got {hour}")
        if int(wake_hour) == int(sleep_hour):
            raise ValueError("wake_hour and sleep_hour must differ")

        self.circadian = circadian
        self.timezone = timezone
        self.wake_hour = int(wake_hour)
        self.sleep_hour = int(sleep_hour)
        self._tz = _resolve_timezone(timezone)
        self._clock = clock

        self.cycle_count = 0
        self._previous_body_vector: Optional[np.ndarray] = None
        self._stability_history: deque = deque(maxlen=stability_window)
        self._body_delta_history: deque = deque(maxlen=stability_window)
        self._valence_trend: deque = deque(maxlen=stability_window)
        self._energy_trend: deque = deque(maxlen=stability_window)
        self._latest_state: Optional[ProtoSelfState] = None
        self._last_time_period: Optional[str] = None

        # One-shot sensory signal, consumed by the next update
        self._sensory_emotion: Optional[Dict[str, Any]] = None
        # Per-cycle valence nudge set while mapping the body and applied to
        # the overall valence in update(); never persisted
        self._pending_valence_shift: float = 0.0

        logger.info(
            f"Protoself initialized: body-state monitoring active "
            f"(circadian={circadian})"
        )

    # ------------------------------------------------------------------
    # Circadian rhythm
    # ------------------------------------------------------------------

    def _now_hour(self) -> float:
        now = self._clock() if self._clock is not None else datetime.now(self._tz)
        if now.tzinfo is not None:
            now = now.astimezone(self._tz)
        return now.hour + now.minute / 60.0 + now.second / 3600.0

    def _circadian_phase(self) -> tuple:
        """Return (phase, time_period) for the current cycle."""
        if self.circadian == "wall_clock":
            hour = self._now_hour()
            phase = wall_clock_alertness(hour, self.wake_hour, self.sleep_hour)
            return phase, _get_time_period(hour)
        if self.circadian == "v04_sawtooth":
            return sawtooth_circadian(self.cycle_count), None
        return sinusoid_circadian(self.cycle_count), None

    @property
    def time_of_day(self) -> Optional[str]:
        """Period of the day under the wall-clock model, otherwise None."""
        if self.circadian != "wall_clock":
            return None
        return _get_time_period(self._now_hour())

    @property
    def alertness(self) -> Optional[float]:
        """Circadian alertness (0 sleepy, 1 alert) under the wall-clock model."""
        if self.circadian != "wall_clock":
            return None
        return wall_clock_alertness(self._now_hour(), self.wake_hour, self.sleep_hour)

    # ------------------------------------------------------------------
    # Sensory emotion input
    # ------------------------------------------------------------------

    def register_sensory_emotion(
        self,
        source: str,
        arousal: float,
        valence_shift: float,
        pain: float = 0.0,
        confidence: float = 1.0,
        label: Optional[str] = None,
    ) -> None:
        """Register an externally detected emotion as a body signal.

        Any modality can call this (audio, vision, text). The signal is
        scaled by confidence and consumed by the next update only, so a
        stale detection never persists.

        Args:
            source: Which sensor or detector produced the signal
            arousal: Activation change in [-1, 1]
            valence_shift: Pleasure/pain shift in [-1, 1]
            pain: Distress added to the pain channel, in [0, 1]
            confidence: Detection confidence in [0, 1]
            label: Optional emotion label, kept for the interoceptive record
        """
        self._sensory_emotion = {
            "source": str(source),
            "arousal": float(np.clip(arousal, -1.0, 1.0)),
            "valence_shift": float(np.clip(valence_shift, -1.0, 1.0)),
            "pain": float(np.clip(pain, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "label": label,
        }
        logger.debug(
            f"Protoself: sensory emotion registered from {source} "
            f"(confidence={confidence:.2f})"
        )

    def register_sensory_label(
        self, label: str, confidence: float = 0.5, source: str = "sensory"
    ) -> bool:
        """Register an emotion by label via SENSORY_EMOTION_BODY_MAP.

        Returns False (and registers nothing) for an unmapped label.
        """
        effect = self.SENSORY_EMOTION_BODY_MAP.get(label.lower())
        if effect is None:
            logger.debug(f"Protoself: no body mapping for emotion label {label!r}")
            return False
        self.register_sensory_emotion(
            source=source,
            arousal=effect["arousal"],
            valence_shift=effect["valence_shift"],
            pain=effect["pain"],
            confidence=confidence,
            label=label.lower(),
        )
        return True

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

        # A consumed sensory emotion shifts the overall valence, which feeds
        # the pleasure/pain feeling through the valence trend below
        if self._pending_valence_shift:
            valence = float(np.clip(valence + self._pending_valence_shift, -1.0, 1.0))

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
        self._pending_valence_shift = 0.0

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

        # Override a channel only when its metric was actually measured
        if system_metrics:
            if "response_latency_ms" in system_metrics:
                latency = float(system_metrics["response_latency_ms"])
                energy = float(np.clip(1.0 - (latency / 5000.0), 0.0, 1.0))
            if "error_rate" in system_metrics:
                error_rate = float(system_metrics["error_rate"])
                pain = max(pain, float(np.clip(error_rate * 5.0, 0.0, 1.0)))
            if "cpu_load" in system_metrics:
                temperature = max(temperature, float(system_metrics["cpu_load"]))
            if "memory_usage_pct" in system_metrics:
                mem_usage = float(system_metrics["memory_usage_pct"])
                metabolic = min(metabolic, 1.0 - mem_usage)

        circadian, time_period = self._circadian_phase()
        if time_period is not None and time_period != self._last_time_period:
            logger.info(
                f"Protoself: time period is now {time_period!r} "
                f"(circadian alertness={circadian:.2f})"
            )
            self._last_time_period = time_period

        # Interoceptive state: full drive snapshot
        interoceptive: Dict[str, Any] = {
            name: {
                "level": d.get("current_level", 0.5),
                "urgency": d.get("urgency", 0.0),
            }
            for name, d in drive_state.items()
        }

        # Sensory emotion, consumed once
        if self._sensory_emotion is not None:
            emo = self._sensory_emotion
            conf = emo["confidence"]
            energy += emo["arousal"] * conf
            pain += emo["pain"] * conf
            temperature += abs(emo["arousal"]) * conf * 0.3
            self._pending_valence_shift = emo["valence_shift"] * conf
            interoceptive["sensory_emotion"] = {
                "source": emo["source"],
                "label": emo["label"],
                "confidence": conf,
                "arousal": emo["arousal"],
                "valence_shift": emo["valence_shift"],
                "pain": emo["pain"],
            }
            self._sensory_emotion = None

        return BodyState(
            metabolic_state=float(np.clip(metabolic, 0.0, 1.0)),
            energy_level=float(np.clip(energy, 0.0, 1.0)),
            pain_signals=float(np.clip(pain, 0.0, 1.0)),
            interoceptive_state=interoceptive,
            circadian_phase=float(circadian),
            temperature=float(np.clip(temperature, 0.0, 1.0)),
            time_period=time_period,
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

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    _HISTORY_NAMES = (
        "_stability_history",
        "_body_delta_history",
        "_valence_trend",
        "_energy_trend",
    )

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize state as scalars, numpy arrays and plain histories."""
        arrays: Dict[str, np.ndarray] = {}
        if self._previous_body_vector is not None:
            arrays["previous_body_vector"] = np.asarray(
                self._previous_body_vector, dtype=float
            )
        histories: Dict[str, Any] = {
            name.lstrip("_"): [float(v) for v in getattr(self, name)]
            for name in self._HISTORY_NAMES
        }
        histories["latest_state"] = (
            self._latest_state.to_dict() if self._latest_state else None
        )
        return {
            "scalars": {
                "cycle_count": self.cycle_count,
                "circadian": self.circadian,
                "last_time_period": self._last_time_period,
            },
            "arrays": arrays,
            "histories": histories,
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state; deques keep this instance's window length.

        A body vector with the wrong shape or non-finite values is rejected,
        so the next update starts a fresh perturbation baseline.
        """
        s = state.get("scalars", {}) or {}
        a = state.get("arrays", {}) or {}
        h = state.get("histories", {}) or {}

        self.cycle_count = int(s.get("cycle_count", self.cycle_count))
        self._last_time_period = s.get("last_time_period")

        if "previous_body_vector" in a:
            try:
                vec = np.array(a["previous_body_vector"], dtype=float)
            except (TypeError, ValueError):
                vec = None
            if vec is not None and vec.shape == (5,) and np.all(np.isfinite(vec)):
                self._previous_body_vector = vec
            else:
                self._previous_body_vector = None
                logger.warning("Protoself: persisted body vector rejected")

        for name in self._HISTORY_NAMES:
            key = name.lstrip("_")
            if key in h:
                maxlen = getattr(self, name).maxlen
                setattr(self, name, deque((float(v) for v in h[key]), maxlen=maxlen))

        latest = h.get("latest_state")
        if isinstance(latest, dict):
            self._latest_state = ProtoSelfState.from_dict(latest)
