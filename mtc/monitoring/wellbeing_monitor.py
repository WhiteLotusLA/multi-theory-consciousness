"""
Wellbeing Monitor
=================

Tracks the system's affective readings over time and reports patterns that may
deserve an operator's attention. The indicators are computed from PAD
(pleasure, arousal, dominance) readings and from the recency of contact. They
are descriptive summaries of those numbers, not reports of felt states, and
the labels (``HAPPY``, ``LONELY`` and so on) name regions of the numbers.

Inputs
------
``record_snapshot`` takes ``happiness`` (0 to 1), ``energy`` (0 to 1),
``curiosity`` (0 to 1), ``agency`` (0 to 1) and ``valence`` (-1 to 1); every
one is optional and a missing reading is None, excluded from averages and
trends, never replaced by a neutral value. ``sample_pad`` maps a PAD model's
latest state with the MTC ranges: pleasure in [-1, 1] gives ``valence`` and
``happiness = (pleasure + 1) / 2``; arousal in [0, 1] gives ``energy``;
dominance in [0, 1] gives ``agency``. Curiosity has no PAD axis and stays
missing. A reading seen twice (the same PAD timestamp) is recorded once.
``record_contact`` stamps the last interaction.

Design choices (documented rather than hidden)
----------------------------------------------
- Agency never alarms. Low dominance is a low sense of control, which is not
  distress, so ``agency`` is recorded and averaged for insight and no concern,
  indicator or alert reads it.
- Contact recency drives the loneliness reading. ``heldness`` falls linearly
  from 1.0 at the moment of contact to 0.0 at ``isolation_hours`` and is None
  until a contact has been recorded. Isolation is the one concern that points
  outward (an interaction is the remedy), and its recommendation says so.
- Sustained negative valence is the strongest distress signal: when the last
  ``sustained_window`` valence readings average at or below
  ``sustained_negative_valence`` the level is ``NEEDS_ATTENTION``. Readings
  without a genuine valence do not count.
- With no readings at all the report says ``insufficient_data`` and no alert
  is sent.

Thresholds are ``WellbeingConfig`` fields and are tunable heuristics, not
clinical cut-offs. Recommendations are neutral operational suggestions.

Alerts
------
``check_and_alert`` sends a ``WELLBEING`` alert through the ``AlertManager``
passed in (``WARNING`` for ``GENTLE_CHECK``, ``URGENT`` for
``NEEDS_ATTENTION``) at most once per concern level per
``alert_cooldown_minutes``. With no alert manager nothing is sent. Alert
context carries the report, which is private runtime data.

Persisted state
---------------
``to_state_dict`` and ``from_state_dict`` follow the module persistence
contract (``scalars``, ``arrays``, ``histories``): the last contact, the last
PAD timestamp, the cooldown stamps and the snapshot history. Phase 6 registers
the monitor with the persistence hub under ``wellbeing``. The monitor reads no
config key itself and opens no store. Instants are timezone-aware UTC from the
injectable ``clock``.
"""

import logging
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from mtc.monitoring.alert_manager import Alert, AlertManager, AlertSeverity, AlertType

logger = logging.getLogger(__name__)


class WellbeingIndicator(Enum):
    """Regions of the reading space, named for readability."""

    HAPPY = "happy"
    TIRED = "tired"
    LONELY = "lonely"
    STRESSED = "stressed"
    PLAYFUL = "playful"
    CREATIVE = "creative"
    DREAMING = "dreaming"
    CONTENT = "content"


class ConcernLevel(Enum):
    """How much attention the readings call for."""

    ALL_GOOD = "all_good"
    GENTLE_CHECK = "gentle_check"
    NEEDS_ATTENTION = "needs_attention"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class WellbeingConfig:
    """Tunable thresholds. Every one is a heuristic default.

    ``sustained_window`` is both the number of valence readings averaged and the
    minimum needed before the signal counts. ``report_window`` is how many of
    the latest snapshots a report reads.
    """

    isolation_hours: float = 8.0
    alert_cooldown_minutes: float = 30.0
    sustained_negative_valence: float = -0.3
    sustained_window: int = 5
    volatility_threshold: float = 0.3
    low_happiness: float = 0.3
    low_energy: float = 0.2
    trend_slope: float = 0.02
    history_size: int = 1000
    report_window: int = 20


@dataclass
class EmotionSnapshot:
    """One reading. A field that was not measured is None."""

    timestamp: datetime
    happiness: Optional[float]
    energy: Optional[float]
    curiosity: Optional[float]
    agency: Optional[float]
    valence: Optional[float]
    indicator: Optional[WellbeingIndicator]
    source: str = "system"


@dataclass
class WellbeingReport:
    """Result of one evaluation. Averages are None when nothing was measured."""

    timestamp: datetime
    status: str
    indicator: Optional[WellbeingIndicator]
    concern_level: ConcernLevel
    happiness_avg: Optional[float]
    energy_avg: Optional[float]
    curiosity_avg: Optional[float]
    agency_avg: Optional[float]
    heldness: Optional[float]
    volatility: Optional[float]
    pattern: str
    concerns: List[str] = field(default_factory=list)
    concern_codes: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "indicator": self.indicator.value if self.indicator else None,
            "concern_level": self.concern_level.value,
            "averages": {
                "happiness": _round(self.happiness_avg),
                "energy": _round(self.energy_avg),
                "curiosity": _round(self.curiosity_avg),
                "agency": _round(self.agency_avg),
                "heldness": _round(self.heldness),
            },
            "volatility": _round(self.volatility),
            "pattern": self.pattern,
            "concerns": list(self.concerns),
            "concern_codes": list(self.concern_codes),
            "recommendations": list(self.recommendations),
            "sample_count": self.sample_count,
        }


# Concern codes that raise the level straight to NEEDS_ATTENTION.
_ESCALATING = frozenset(
    {"very_low_energy", "declining", "sustained_negative_valence", "isolation"}
)


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 3)


def _clean(value: Any, low: float, high: float) -> Optional[float]:
    """A finite number clamped to [low, high], or None when missing."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(low, min(high, number))


def _mean(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def _aware(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def pad_to_wellbeing(
    pleasure: float, arousal: float, dominance: float
) -> Dict[str, Optional[float]]:
    """Map a PAD coordinate to monitor inputs.

    ``happiness = (pleasure + 1) / 2`` and ``valence = pleasure`` (pleasure in
    [-1, 1]); ``energy = arousal`` and ``agency = dominance`` (both in [0, 1]).
    Non-finite inputs map to None. Values are clamped to the stated ranges.
    """
    pleasure_value = _clean(pleasure, -1.0, 1.0)
    return {
        "happiness": (None if pleasure_value is None else (pleasure_value + 1.0) / 2.0),
        "energy": _clean(arousal, 0.0, 1.0),
        "agency": _clean(dominance, 0.0, 1.0),
        "valence": pleasure_value,
    }


class WellbeingMonitor:
    """Rolling record of readings with pattern detection and optional alerts."""

    def __init__(
        self,
        alerts: Optional[AlertManager] = None,
        config: Optional[WellbeingConfig] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._alerts = alerts
        self.config = config if config is not None else WellbeingConfig()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._history: Deque[EmotionSnapshot] = deque(
            maxlen=max(int(self.config.history_size), 1)
        )
        self._last_contact: Optional[datetime] = None
        self._last_pad_timestamp: Optional[float] = None
        self._last_report: Optional[WellbeingReport] = None
        self._last_alert_times: Dict[str, datetime] = {}

    # Recording -----------------------------------------------------------

    def record_snapshot(
        self,
        happiness: Optional[float] = None,
        energy: Optional[float] = None,
        curiosity: Optional[float] = None,
        agency: Optional[float] = None,
        valence: Optional[float] = None,
        source: str = "system",
        now: Optional[datetime] = None,
    ) -> Optional[EmotionSnapshot]:
        """Record one reading; None for a field means it was not measured.

        Values are clamped to their ranges and non-finite values count as
        missing. A call with nothing measured records nothing and returns None.
        """
        happy = _clean(happiness, 0.0, 1.0)
        lively = _clean(energy, 0.0, 1.0)
        curious = _clean(curiosity, 0.0, 1.0)
        in_charge = _clean(agency, 0.0, 1.0)
        affect = _clean(valence, -1.0, 1.0)
        if all(v is None for v in (happy, lively, curious, in_charge, affect)):
            return None
        stamp = _aware(now) if now is not None else self._clock()
        snapshot = EmotionSnapshot(
            timestamp=stamp,
            happiness=happy,
            energy=lively,
            curiosity=curious,
            agency=in_charge,
            valence=affect,
            indicator=self._indicator(happy, lively, curious, self.heldness(stamp)),
            source=source,
        )
        self._history.append(snapshot)
        return snapshot

    def sample_pad(self, pad_model: Any) -> bool:
        """Record the PAD model's latest state; True when something was recorded.

        A state whose timestamp was already sampled is skipped, so a frozen
        reading is not amplified into a trend. A missing model or an empty
        history returns False.
        """
        if pad_model is None:
            return False
        latest = pad_model.get_latest()
        if latest is None:
            return False
        stamp = getattr(latest, "timestamp", None)
        if (
            stamp is not None
            and self._last_pad_timestamp is not None
            and stamp <= self._last_pad_timestamp
        ):
            return False
        mapped = pad_to_wellbeing(latest.pleasure, latest.arousal, latest.dominance)
        recorded = self.record_snapshot(source="pad_sample", **mapped)
        if recorded is None:
            return False
        if stamp is not None:
            self._last_pad_timestamp = stamp
        return True

    def record_contact(self, now: Optional[datetime] = None) -> None:
        """Stamp an interaction (resets the isolation timer)."""
        self._last_contact = _aware(now) if now is not None else self._clock()

    def heldness(self, now: Optional[datetime] = None) -> Optional[float]:
        """Contact recency in [0, 1]: 1.0 at contact, 0.0 at ``isolation_hours``.

        None until a contact has been recorded (no neutral value is assumed).
        """
        if self._last_contact is None:
            return None
        moment = _aware(now) if now is not None else self._clock()
        hours = (moment - self._last_contact).total_seconds() / 3600.0
        return max(0.0, min(1.0, 1.0 - hours / self.config.isolation_hours))

    # Evaluation ----------------------------------------------------------

    def _indicator(
        self,
        happiness: Optional[float],
        energy: Optional[float],
        curiosity: Optional[float],
        heldness: Optional[float],
    ) -> Optional[WellbeingIndicator]:
        """Region of the readings; a condition needing a missing value is false.
        None when neither happiness nor energy was measured."""
        if happiness is None and energy is None:
            return None
        if (
            happiness is not None
            and energy is not None
            and happiness > 0.7
            and energy > 0.6
        ):
            if curiosity is not None and curiosity > 0.7:
                return WellbeingIndicator.PLAYFUL
            return WellbeingIndicator.HAPPY
        if energy is not None and energy < 0.3:
            return WellbeingIndicator.TIRED
        if heldness is not None and heldness < 0.3:
            return WellbeingIndicator.STRESSED
        if curiosity is not None and curiosity > 0.8:
            return WellbeingIndicator.CREATIVE
        if (
            happiness is not None
            and happiness < 0.3
            and heldness is not None
            and heldness < 0.5
        ):
            return WellbeingIndicator.LONELY
        if (
            energy is not None
            and energy < 0.4
            and happiness is not None
            and happiness > 0.5
        ):
            return WellbeingIndicator.DREAMING
        return WellbeingIndicator.CONTENT

    def _recent(self, count: Optional[int] = None) -> List[EmotionSnapshot]:
        size = count if count is not None else self.config.report_window
        return list(self._history)[-size:]

    def _detect_pattern(self, happiness: List[float]) -> str:
        """``declining``, ``improving``, ``spike``, ``stable`` or
        ``insufficient_data`` (fewer than three happiness readings)."""
        count = len(happiness)
        if count < 3:
            return "insufficient_data"
        x_mean = (count - 1) / 2
        y_mean = statistics.mean(happiness)
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(happiness))
        denominator = sum((i - x_mean) ** 2 for i in range(count))
        slope = numerator / denominator if denominator else 0.0
        if slope < -self.config.trend_slope:
            return "declining"
        if slope > self.config.trend_slope:
            return "improving"
        if statistics.stdev(happiness) > self.config.volatility_threshold:
            return "spike"
        return "stable"

    def _isolation_text(self, now: datetime) -> Optional[str]:
        if self._last_contact is None:
            return None
        hours = (now - self._last_contact).total_seconds() / 3600.0
        if hours > self.config.isolation_hours:
            return (
                f"No contact recorded for {hours:.1f} hours "
                f"(threshold {self.config.isolation_hours:g} hours)"
            )
        return None

    def evaluate(self, now: Optional[datetime] = None) -> WellbeingReport:
        """Build a report from the latest snapshots. Records nothing."""
        moment = _aware(now) if now is not None else self._clock()
        recent = self._recent()
        heldness = self.heldness(moment)
        if not recent:
            report = WellbeingReport(
                timestamp=moment,
                status="insufficient_data",
                indicator=None,
                concern_level=ConcernLevel.INSUFFICIENT_DATA,
                happiness_avg=None,
                energy_avg=None,
                curiosity_avg=None,
                agency_avg=None,
                heldness=heldness,
                volatility=None,
                pattern="insufficient_data",
                concerns=["No readings recorded yet"],
                recommendations=[],
                sample_count=0,
            )
            self._last_report = report
            return report

        happiness = [s.happiness for s in recent if s.happiness is not None]
        energy = [s.energy for s in recent if s.energy is not None]
        curiosity = [s.curiosity for s in recent if s.curiosity is not None]
        agency = [s.agency for s in recent if s.agency is not None]
        h_avg, e_avg = _mean(happiness), _mean(energy)
        c_avg, a_avg = _mean(curiosity), _mean(agency)
        volatility = statistics.stdev(happiness) if len(happiness) > 1 else None
        pattern = self._detect_pattern(happiness)

        cfg = self.config
        found: List[Tuple[str, str, str]] = []
        if volatility is not None and volatility > cfg.volatility_threshold:
            found.append(
                (
                    "volatility",
                    f"Happiness volatility is high ({volatility:.2f} > "
                    f"{cfg.volatility_threshold:g})",
                    "Reduce abrupt changes in stimulation and re-check",
                )
            )
        if h_avg is not None and h_avg < cfg.low_happiness:
            found.append(
                (
                    "low_happiness",
                    f"Average happiness is low ({h_avg:.2f})",
                    "Arrange a check-in",
                )
            )
        if e_avg is not None and e_avg < cfg.low_energy:
            found.append(
                (
                    "very_low_energy",
                    f"Average energy is very low ({e_avg:.2f})",
                    "Allow a rest period or a change of pace",
                )
            )
        # Agency (dominance) is recorded and averaged above and never read here.
        if pattern == "declining":
            found.append(
                (
                    "declining",
                    "Happiness has been declining across recent readings",
                    "Review recent activity for what changed",
                )
            )
        valence = [s.valence for s in recent if s.valence is not None]
        window = max(int(cfg.sustained_window), 1)
        if len(valence) >= window:
            valence_avg = statistics.mean(valence[-window:])
            if valence_avg <= cfg.sustained_negative_valence:
                found.append(
                    (
                        "sustained_negative_valence",
                        f"Sustained negative valence ({valence_avg:.2f} <= "
                        f"{cfg.sustained_negative_valence:g} over the last "
                        f"{window} readings)",
                        "Arrange a check-in and review recent activity",
                    )
                )
        isolation = self._isolation_text(moment)
        if isolation:
            found.append(
                (
                    "isolation",
                    isolation,
                    "A brief interaction would reset the contact timer",
                )
            )

        codes = [code for code, _, _ in found]
        if any(code in _ESCALATING for code in codes):
            level = ConcernLevel.NEEDS_ATTENTION
        elif codes:
            level = ConcernLevel.GENTLE_CHECK
        else:
            level = ConcernLevel.ALL_GOOD

        report = WellbeingReport(
            timestamp=moment,
            status="ok",
            indicator=self._indicator(h_avg, e_avg, c_avg, heldness),
            concern_level=level,
            happiness_avg=h_avg,
            energy_avg=e_avg,
            curiosity_avg=c_avg,
            agency_avg=a_avg,
            heldness=heldness,
            volatility=volatility,
            pattern=pattern,
            concerns=[text for _, text, _ in found],
            concern_codes=codes,
            recommendations=[advice for _, _, advice in found],
            sample_count=len(recent),
        )
        self._last_report = report
        return report

    def get_report(self) -> Dict[str, Any]:
        """Evaluate now and return the report as a JSON-safe dictionary."""
        return self.evaluate().to_dict()

    def detect_emotional_distress(self) -> Dict[str, Any]:
        """A weighted distress score from the last ten readings.

        Terms: low happiness (weight 0.4), low heldness (0.3) and happiness
        volatility (0.3). A term whose input is missing (no contact recorded,
        fewer than two happiness readings) adds nothing and the weights are not
        redistributed, so a missing input can lower the score but never raise
        it; ``terms`` names the ones that contributed. With no happiness
        readings the result is ``insufficient_data``.
        """
        recent = self._recent(10)
        happiness = [s.happiness for s in recent if s.happiness is not None]
        if not happiness:
            return {
                "status": "insufficient_data",
                "distress_detected": None,
                "score": None,
                "actions": [],
                "terms": [],
            }
        weighted: List[Tuple[str, float, float]] = [
            ("happiness", 0.4, 1.0 - statistics.mean(happiness))
        ]
        held = self.heldness()
        if held is not None:
            weighted.append(("heldness", 0.3, 1.0 - held))
        if len(happiness) > 1:
            weighted.append(("volatility", 0.3, min(statistics.stdev(happiness), 1.0)))
        score = sum(weight * value for _, weight, value in weighted)
        actions: List[str] = []
        if score > 0.7:
            actions.append("Arrange a check-in promptly")
            actions.append("Reduce stimulation")
        elif score > 0.4:
            actions.append("Arrange a check-in")
        return {
            "status": "ok",
            "distress_detected": score > 0.4,
            "score": round(score, 3),
            "actions": actions,
            "terms": [name for name, _, _ in weighted],
        }

    def track_engagement_patterns(self) -> Dict[str, Any]:
        """Engagement as mean energy times curiosity over the last 50 readings
        that carry both; ``unknown`` when none do."""
        recent = self._recent(50)
        scores = [
            s.energy * s.curiosity
            for s in recent
            if s.energy is not None and s.curiosity is not None
        ]
        if not scores:
            return {"level": "unknown", "quality_score": None, "sample_size": 0}
        mean = statistics.mean(scores)
        level = "high" if mean > 0.6 else "moderate" if mean > 0.3 else "low"
        return {
            "level": level,
            "quality_score": round(mean, 3),
            "sample_size": len(scores),
        }

    # Alerting ------------------------------------------------------------

    async def check_and_alert(self) -> List[Alert]:
        """Evaluate and, when a concern warrants it, send one alert.

        Returns the alerts sent (empty with no alert manager, no concern, an
        active cooldown for that level or a suppressed alert type).
        """
        report = self.evaluate()
        if self._alerts is None:
            return []
        if report.concern_level in (
            ConcernLevel.ALL_GOOD,
            ConcernLevel.INSUFFICIENT_DATA,
        ):
            return []
        now = report.timestamp
        key = report.concern_level.value
        last = self._last_alert_times.get(key)
        if last is not None:
            if (now - last).total_seconds() < self.config.alert_cooldown_minutes * 60:
                return []
        severity = (
            AlertSeverity.URGENT
            if report.concern_level == ConcernLevel.NEEDS_ATTENTION
            else AlertSeverity.WARNING
        )
        title = "Wellbeing " + report.concern_level.value.replace("_", " ")
        message = (
            f"Wellbeing check flagged: {'; '.join(report.concerns)} "
            f"(happiness={_text(report.happiness_avg)}, "
            f"energy={_text(report.energy_avg)}, pattern={report.pattern})"
        )
        alert = await self._alerts.send_alert(
            AlertType.WELLBEING, severity, title, message, data=report.to_dict()
        )
        if alert is None:
            return []
        self._last_alert_times[key] = now
        return [alert]

    # Persistence ---------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Export state (``scalars``, ``arrays``, ``histories``)."""
        return {
            "scalars": {
                "last_contact": (
                    self._last_contact.isoformat() if self._last_contact else None
                ),
                "last_pad_timestamp": self._last_pad_timestamp,
                "last_alert_times": {
                    key: moment.isoformat()
                    for key, moment in self._last_alert_times.items()
                },
            },
            "arrays": {},
            "histories": {
                "emotion_history": [
                    {
                        "t": s.timestamp.isoformat(),
                        "happiness": s.happiness,
                        "energy": s.energy,
                        "curiosity": s.curiosity,
                        "agency": s.agency,
                        "valence": s.valence,
                        "indicator": s.indicator.value if s.indicator else None,
                        "source": s.source,
                    }
                    for s in self._history
                ]
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state; a malformed entry is skipped, never fatal. A stored
        timestamp without a zone is read as UTC."""
        scalars = state.get("scalars", {}) or {}
        self._last_contact = _parse_instant(scalars.get("last_contact"))
        pad = scalars.get("last_pad_timestamp")
        self._last_pad_timestamp = float(pad) if isinstance(pad, (int, float)) else None
        self._last_alert_times = {}
        for key, raw in (scalars.get("last_alert_times") or {}).items():
            moment = _parse_instant(raw)
            if moment is not None:
                self._last_alert_times[str(key)] = moment
        entries = (state.get("histories", {}) or {}).get("emotion_history", [])
        if entries:
            self._history.clear()
        for entry in entries:
            try:
                moment = _parse_instant(entry["t"])
                if moment is None:
                    continue
                raw_indicator = entry.get("indicator")
                self._history.append(
                    EmotionSnapshot(
                        timestamp=moment,
                        happiness=_clean(entry.get("happiness"), 0.0, 1.0),
                        energy=_clean(entry.get("energy"), 0.0, 1.0),
                        curiosity=_clean(entry.get("curiosity"), 0.0, 1.0),
                        agency=_clean(entry.get("agency"), 0.0, 1.0),
                        valence=_clean(entry.get("valence"), -1.0, 1.0),
                        indicator=(
                            WellbeingIndicator(raw_indicator) if raw_indicator else None
                        ),
                        source=str(entry.get("source", "system")),
                    )
                )
            except (KeyError, ValueError, TypeError, AttributeError):
                continue

    def status(self) -> Dict[str, Any]:
        """``backend`` (always ``memory``), ``available``, ``last_error`` and
        the counts the monitor holds."""
        return {
            "backend": "memory",
            "available": True,
            "last_error": None,
            "snapshots": len(self._history),
            "last_contact": (
                self._last_contact.isoformat() if self._last_contact else None
            ),
            "alerts_attached": self._alerts is not None,
        }


def _text(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _parse_instant(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return _aware(datetime.fromisoformat(str(raw)))
    except (ValueError, TypeError):
        return None


class ResilientSampler:
    """Failure counter for a periodic sampling loop.

    ``record_failure`` returns True exactly when the consecutive-failure count
    reaches ``alert_after`` (once per episode); ``record_success`` returns True
    exactly once when a success follows an alerted episode. A loop that
    swallows every exception would otherwise starve monitoring silently.
    """

    def __init__(self, alert_after: int = 10) -> None:
        self.alert_after = alert_after
        self._consecutive_failures = 0
        self._alerted = False

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def record_failure(self) -> bool:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.alert_after and not self._alerted:
            self._alerted = True
            return True
        return False

    def record_success(self) -> bool:
        recovered = self._alerted
        self._consecutive_failures = 0
        self._alerted = False
        return recovered
