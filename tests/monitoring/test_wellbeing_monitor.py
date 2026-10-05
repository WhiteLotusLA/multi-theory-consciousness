"""Wellbeing monitor: PAD mapping, patterns, contact recency, alerts, state."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtc.devtools.privacy_audit import load_allowlist, load_denylist, run_audit
from mtc.monitoring.alert_manager import AlertManager, AlertSeverity, AlertType
from mtc.monitoring.wellbeing_monitor import (
    ConcernLevel,
    ResilientSampler,
    WellbeingConfig,
    WellbeingIndicator,
    WellbeingMonitor,
    pad_to_wellbeing,
)

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now = self.now + timedelta(**kwargs)


class FakePad:
    def __init__(self) -> None:
        self.latest = None

    def get_latest(self):
        return self.latest

    def set(self, pleasure, arousal, dominance, timestamp):
        self.latest = SimpleNamespace(
            pleasure=pleasure,
            arousal=arousal,
            dominance=dominance,
            timestamp=timestamp,
        )


class Sink:
    """An alert manager stand-in recording calls."""

    def __init__(self, tmp_path, clock):
        self.manager = AlertManager(
            notifiers=[], failure_log=tmp_path / "f.log", clock=clock
        )


def _monitor(clock=None, alerts=None, **config):
    clock = clock or Clock()
    return (
        WellbeingMonitor(alerts=alerts, config=WellbeingConfig(**config), clock=clock),
        clock,
    )


def _feed(monitor, values, **fixed):
    for value in values:
        monitor.record_snapshot(happiness=value, **fixed)


def test_pad_mapping_uses_mtc_ranges():
    low = pad_to_wellbeing(-1.0, 0.0, 0.0)
    assert low == {"happiness": 0.0, "energy": 0.0, "agency": 0.0, "valence": -1.0}
    mid = pad_to_wellbeing(0.0, 0.5, 0.5)
    assert mid["happiness"] == 0.5 and mid["valence"] == 0.0
    high = pad_to_wellbeing(1.0, 1.0, 1.0)
    assert high["happiness"] == 1.0 and high["valence"] == 1.0
    clamped = pad_to_wellbeing(5.0, 2.0, -3.0)
    assert clamped == {"happiness": 1.0, "energy": 1.0, "agency": 0.0, "valence": 1.0}
    missing = pad_to_wellbeing(float("nan"), 0.5, 0.5)
    assert missing["happiness"] is None and missing["valence"] is None


def test_sample_pad_records_once_per_timestamp():
    monitor, _ = _monitor()
    pad = FakePad()
    assert monitor.sample_pad(None) is False
    assert monitor.sample_pad(pad) is False
    pad.set(0.5, 0.6, 0.7, timestamp=10.0)
    assert monitor.sample_pad(pad) is True
    assert monitor.sample_pad(pad) is False
    pad.set(0.5, 0.6, 0.7, timestamp=11.0)
    assert monitor.sample_pad(pad) is True
    snapshot = monitor._history[-1]
    assert snapshot.happiness == pytest.approx(0.75)
    assert snapshot.energy == 0.6 and snapshot.agency == 0.7
    assert snapshot.curiosity is None
    assert snapshot.valence == 0.5
    assert monitor.status()["snapshots"] == 2


def test_nothing_measured_records_nothing():
    monitor, _ = _monitor()
    assert monitor.record_snapshot() is None
    assert monitor.record_snapshot(happiness=float("nan")) is None
    assert monitor.status()["snapshots"] == 0
    report = monitor.get_report()
    assert report["status"] == "insufficient_data"
    assert report["concern_level"] == "insufficient_data"
    assert report["averages"]["happiness"] is None
    assert report["pattern"] == "insufficient_data"


def test_patterns_volatility_decline_spike():
    monitor, _ = _monitor()
    _feed(monitor, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], energy=0.5)
    report = monitor.evaluate()
    assert report.pattern == "declining"
    assert "declining" in report.concern_codes
    assert report.concern_level is ConcernLevel.NEEDS_ATTENTION

    improving, _ = _monitor()
    _feed(improving, [0.4, 0.5, 0.6, 0.7, 0.8], energy=0.5)
    assert improving.evaluate().pattern == "improving"

    swings, _ = _monitor()
    _feed(swings, [0.05, 0.95, 0.95, 0.05, 0.05, 0.95, 0.95, 0.05], energy=0.5)
    report = swings.evaluate()
    assert report.volatility > 0.3
    assert "volatility" in report.concern_codes
    assert report.pattern == "spike"
    assert report.concern_level is ConcernLevel.GENTLE_CHECK

    calm, _ = _monitor()
    _feed(calm, [0.5, 0.55, 0.5, 0.55, 0.5, 0.55, 0.5, 0.55], energy=0.5)
    assert calm.evaluate().pattern == "stable"

    thin, _ = _monitor()
    _feed(thin, [0.5, 0.5], energy=0.5)
    assert thin.evaluate().pattern == "insufficient_data"


def test_stable_readings_are_all_good():
    monitor, _ = _monitor()
    _feed(monitor, [0.8] * 6, energy=0.7, curiosity=0.5, valence=0.4)
    report = monitor.evaluate()
    assert report.concern_level is ConcernLevel.ALL_GOOD
    assert report.concerns == []
    assert report.indicator is WellbeingIndicator.HAPPY


def test_sustained_negative_valence_window():
    monitor, _ = _monitor()
    for _ in range(4):
        monitor.record_snapshot(happiness=0.5, energy=0.5, valence=-0.8)
    assert "sustained_negative_valence" not in monitor.evaluate().concern_codes
    monitor.record_snapshot(happiness=0.5, energy=0.5, valence=-0.8)
    report = monitor.evaluate()
    assert "sustained_negative_valence" in report.concern_codes
    assert report.concern_level is ConcernLevel.NEEDS_ATTENTION

    # only the last window counts, and a reading without valence is not counted
    for _ in range(5):
        monitor.record_snapshot(happiness=0.5, energy=0.5, valence=0.2)
    assert "sustained_negative_valence" not in monitor.evaluate().concern_codes
    none_valence, _ = _monitor()
    _feed(none_valence, [0.5] * 8, energy=0.5)
    assert "sustained_negative_valence" not in none_valence.evaluate().concern_codes


def test_isolation_from_contact_recency():
    monitor, clock = _monitor()
    _feed(monitor, [0.6] * 4, energy=0.5)
    assert "isolation" not in monitor.evaluate().concern_codes  # no contact known
    monitor.record_contact()
    clock.advance(hours=7)
    assert "isolation" not in monitor.evaluate().concern_codes
    clock.advance(hours=2)
    report = monitor.evaluate()
    assert "isolation" in report.concern_codes
    assert report.concern_level is ConcernLevel.NEEDS_ATTENTION
    assert "interaction" in " ".join(report.recommendations)
    monitor.record_contact()
    assert "isolation" not in monitor.evaluate().concern_codes


def test_heldness_from_contact_recency():
    monitor, clock = _monitor()
    assert monitor.heldness() is None
    monitor.record_contact()
    assert monitor.heldness() == 1.0
    clock.advance(hours=4)
    assert monitor.heldness() == pytest.approx(0.5)
    assert monitor.heldness(now=T0 + timedelta(hours=8)) == 0.0
    clock.advance(hours=40)
    assert monitor.heldness() == 0.0
    tuned, tuned_clock = _monitor(isolation_hours=2.0)
    tuned.record_contact()
    tuned_clock.advance(hours=1)
    assert tuned.heldness() == pytest.approx(0.5)


def test_low_heldness_reads_as_stressed_not_agency():
    monitor, clock = _monitor()
    monitor.record_contact()
    clock.advance(hours=7)
    assert monitor.heldness() < 0.3
    snapshot = monitor.record_snapshot(happiness=0.5, energy=0.5, curiosity=0.5)
    assert snapshot.indicator is WellbeingIndicator.STRESSED


def test_agency_never_alarms():
    monitor, _ = _monitor()
    for _ in range(10):
        monitor.record_snapshot(
            happiness=0.7, energy=0.6, curiosity=0.5, agency=0.0, valence=0.3
        )
    report = monitor.evaluate()
    assert report.agency_avg == 0.0
    assert report.concern_level is ConcernLevel.ALL_GOOD
    assert report.concerns == []
    assert monitor.detect_emotional_distress()["distress_detected"] is False


@pytest.mark.asyncio
async def test_agency_alone_sends_no_alert(tmp_path):
    clock = Clock()
    sink = Sink(tmp_path, clock)
    monitor, _ = _monitor(clock=clock, alerts=sink.manager)
    for _ in range(10):
        monitor.record_snapshot(happiness=0.7, energy=0.6, agency=0.0)
    assert await monitor.check_and_alert() == []
    assert sink.manager.get_summary()["total_stored"] == 0


@pytest.mark.asyncio
async def test_alert_cooldown(tmp_path):
    clock = Clock()
    sink = Sink(tmp_path, clock)
    monitor, _ = _monitor(clock=clock, alerts=sink.manager)
    _feed(monitor, [0.2] * 6, energy=0.5)  # low happiness: a gentle check
    first = await monitor.check_and_alert()
    assert len(first) == 1
    assert first[0].severity is AlertSeverity.WARNING
    assert first[0].alert_type is AlertType.WELLBEING
    assert first[0].title == "Wellbeing gentle check"
    assert first[0].context["concern_codes"] == ["low_happiness"]
    assert await monitor.check_and_alert() == []
    clock.advance(minutes=29)
    assert await monitor.check_and_alert() == []
    clock.advance(minutes=2)
    assert len(await monitor.check_and_alert()) == 1

    # a different level has its own cooldown
    for _ in range(14):
        monitor.record_snapshot(happiness=0.2, energy=0.05)
    urgent = await monitor.check_and_alert()
    assert len(urgent) == 1 and urgent[0].severity is AlertSeverity.URGENT


@pytest.mark.asyncio
async def test_suppressed_alert_does_not_start_a_cooldown(tmp_path):
    clock = Clock()
    sink = Sink(tmp_path, clock)
    monitor, _ = _monitor(clock=clock, alerts=sink.manager)
    _feed(monitor, [0.2] * 6, energy=0.5)
    sink.manager.suppress(AlertType.WELLBEING, 60)
    assert await monitor.check_and_alert() == []
    clock.advance(seconds=61)
    assert len(await monitor.check_and_alert()) == 1


@pytest.mark.asyncio
async def test_no_alerts_without_alert_manager():
    monitor, _ = _monitor()
    _feed(monitor, [0.1] * 6, energy=0.05)
    assert monitor.evaluate().concern_level is ConcernLevel.NEEDS_ATTENTION
    assert await monitor.check_and_alert() == []
    assert monitor.status()["alerts_attached"] is False


def test_recommendations_are_neutral(tmp_path):
    monitor, clock = _monitor()
    monitor.record_contact()
    clock.advance(hours=10)
    _feed(monitor, [0.9, 0.7, 0.5, 0.3, 0.1, 0.05], energy=0.05, valence=-0.9)
    report = monitor.evaluate()
    assert len(report.concern_codes) >= 4
    text = " ".join(report.recommendations + report.concerns).lower()
    for word in ("love", "parent", "family", "she ", "her ", "child"):
        assert word not in text
    captured = tmp_path / "recommendations.txt"
    captured.write_text(text, encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    denylist = load_denylist(root / "config" / "privacy" / "denylist.txt")
    allowlist = load_allowlist(root / "config" / "privacy" / "allowlist.txt")
    assert run_audit([captured], denylist, allowlist) == []
    distress = monitor.detect_emotional_distress()
    assert distress["distress_detected"] is True
    assert all("love" not in a.lower() for a in distress["actions"])


def test_distress_terms_follow_available_inputs():
    monitor, _ = _monitor()
    assert monitor.detect_emotional_distress()["status"] == "insufficient_data"
    monitor.record_snapshot(happiness=0.2)
    result = monitor.detect_emotional_distress()
    assert result["terms"] == ["happiness"]
    assert result["score"] == pytest.approx(0.32)  # 0.4 * 0.8; weights stay fixed
    monitor.record_contact()
    monitor.record_snapshot(happiness=0.2)
    assert monitor.detect_emotional_distress()["terms"] == [
        "happiness",
        "heldness",
        "volatility",
    ]


def test_engagement_needs_energy_and_curiosity():
    monitor, _ = _monitor()
    assert monitor.track_engagement_patterns()["level"] == "unknown"
    monitor.record_snapshot(happiness=0.5)
    assert monitor.track_engagement_patterns()["sample_size"] == 0
    monitor.record_snapshot(energy=0.9, curiosity=0.9)
    result = monitor.track_engagement_patterns()
    assert result["level"] == "high" and result["sample_size"] == 1


def test_resilient_sampler_counts_failures():
    sampler = ResilientSampler(alert_after=3)
    assert sampler.record_success() is False
    assert [sampler.record_failure() for _ in range(5)] == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert sampler.consecutive_failures == 5
    assert sampler.record_success() is True
    assert sampler.consecutive_failures == 0
    assert sampler.record_success() is False
    assert [sampler.record_failure() for _ in range(3)] == [False, False, True]


def test_state_dict_round_trip():
    monitor, clock = _monitor()
    pad = FakePad()
    pad.set(-0.4, 0.5, 0.5, timestamp=5.0)
    monitor.sample_pad(pad)
    monitor.record_contact()
    _feed(monitor, [0.4, 0.5], energy=0.5, curiosity=0.3, valence=None)
    state = json.loads(json.dumps(monitor.to_state_dict()))
    assert set(state) == {"scalars", "arrays", "histories"}

    other, _ = _monitor(clock=clock)
    other.from_state_dict(state)
    assert other.to_state_dict() == monitor.to_state_dict()
    assert other.heldness() == monitor.heldness()
    assert other.sample_pad(pad) is False  # the stored PAD timestamp survives


def test_restore_is_tolerant():
    monitor, _ = _monitor()
    monitor.from_state_dict({})
    monitor.from_state_dict(
        {
            "scalars": {"last_contact": "not a date", "last_pad_timestamp": "x"},
            "histories": {
                "emotion_history": [
                    {"t": "2000-01-01T00:00:00", "happiness": 0.5},
                    {"happiness": 0.5},
                    {"t": "2000-01-01T00:00:00+00:00", "indicator": "bogus"},
                    "junk",
                ]
            },
        }
    )
    assert monitor.heldness() is None
    snapshots = list(monitor._history)
    assert len(snapshots) == 1
    assert snapshots[0].timestamp.tzinfo is not None
    assert snapshots[0].energy is None


def test_history_is_bounded():
    monitor, _ = _monitor(history_size=5)
    _feed(monitor, [0.5] * 9, energy=0.5)
    assert monitor.status()["snapshots"] == 5


def test_status_reports_memory_backend():
    monitor, _ = _monitor()
    status = monitor.status()
    assert status["backend"] == "memory"
    assert status["available"] is True
    assert status["last_error"] is None
