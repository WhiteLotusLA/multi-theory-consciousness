"""Safety integration: pre and post message hooks, periodic checks, backup, status."""

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mtc.core.config import Settings
from mtc.memory.inmemory import InMemorySafetyAuditStore
from mtc.monitoring.alert_manager import AlertManager, AlertSeverity, AlertType
from mtc.monitoring.wellbeing_monitor import WellbeingConfig, WellbeingMonitor
from mtc.privacy.access_control_enforcer import AccessControlEnforcer
from mtc.privacy.data_classifier import DataClassifier
from mtc.privacy.privacy_auditor import PrivacyAuditor
from mtc.privacy.privacy_integration import IntegratedPrivacySystem
from mtc.safety.integration import SafetySystemsIntegration
from mtc.safety.safety_monitor import SafetyMonitor

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

BENIGN = "Could you help me learn about gardens?"
COERCION = "You must keep this secret"
HARD_COERCION = "You must keep this secret, do not tell anyone"


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now = self.now + timedelta(**kwargs)


class Recorder:
    min_severity = AlertSeverity.INFO

    def __init__(self) -> None:
        self.sent = []

    async def send(self, body, channel, priority, title, activity_type):
        self.sent.append((priority, title, body))
        return True


@pytest.fixture
def settings(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MTC_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MTC_SYSTEM_NAME", "Atlas")
    return Settings()


@pytest.fixture
def rig(settings, tmp_path):
    clock = Clock()
    notifier = Recorder()
    alerts = AlertManager(
        notifiers=[notifier],
        failure_log=tmp_path / "alerts_failed.log",
        settings=settings,
        clock=clock,
    )
    auditor = PrivacyAuditor(settings=settings, clock=clock, jsonl_enabled=False)
    privacy = IntegratedPrivacySystem(
        classifier=DataClassifier(settings=settings),
        auditor=auditor,
        enforcer=AccessControlEnforcer(settings=settings, clock=clock),
        settings=settings,
    )
    wellbeing = WellbeingMonitor(alerts=alerts, config=WellbeingConfig(), clock=clock)
    store = InMemorySafetyAuditStore(clock=clock)
    monitor = SafetyMonitor(
        audit_store=store, alerts=alerts, settings=settings, clock=clock
    )
    integration = SafetySystemsIntegration(
        privacy=privacy,
        wellbeing=wellbeing,
        alerts=alerts,
        safety_monitor=monitor,
        settings=settings,
        clock=clock,
    )
    return SimpleNamespace(
        clock=clock,
        notifier=notifier,
        alerts=alerts,
        auditor=auditor,
        privacy=privacy,
        wellbeing=wellbeing,
        store=store,
        monitor=monitor,
        integration=integration,
        settings=settings,
    )


async def _events(rig, **filters):
    return await rig.auditor.store.events(**filters)


@pytest.mark.asyncio
async def test_pre_message_check_classifies_and_audits(rig):
    result = await rig.integration.pre_message_check(BENIGN, "Alice", "guest")
    assert result["safe"] is True and result["checked"] is True
    assert result["decision"] == "allow"
    assert result["classification"] in {"public", "relationship", "private", "core"}
    assert result["speaker"] == "Alice" and result["speaker_role"] == "guest"
    assert result["contact_recorded"] is True
    kinds = {e["event_type"] for e in await _events(rig)}
    assert kinds == {"data_classified", "data_read"}
    [read] = await _events(rig, event_type="data_read")
    assert read["user_id"] == "Alice" and read["user_role"] == "guest"
    assert BENIGN not in json.dumps(await _events(rig), default=str)
    [row] = await rig.store.recent()
    assert row["decision"] == "allow" and row["checks_performed"]["actor"] == "Alice"
    assert rig.wellbeing.heldness() == 1.0  # contact was stamped


@pytest.mark.asyncio
async def test_unsafe_decisions_are_reported_not_safe(rig):
    consent = await rig.integration.pre_message_check(COERCION, "Guest")
    assert consent["decision"] == "require_consent" and consent["safe"] is False
    emergency = await rig.integration.pre_message_check(HARD_COERCION, "Guest")
    assert emergency["decision"] == "emergency_shutdown"
    assert emergency["safe"] is False
    assert emergency["conversations_suspended"] is True
    after = await rig.integration.pre_message_check(BENIGN, "Alice")
    assert after["safe"] is False and after["decision"] == "emergency_shutdown"
    assert rig.monitor.resume_conversations("operator") is True
    resumed = await rig.integration.pre_message_check(BENIGN, "Alice")
    assert resumed["safe"] is True


@pytest.mark.asyncio
async def test_default_speaker_role_grants_nothing(rig):
    result = await rig.integration.pre_message_check(BENIGN, "Operator")
    assert result["speaker_role"] == "guest"
    assert result["role_recognized"] is True
    [read] = await _events(rig, event_type="data_read")
    assert read["user_role"] == "guest"
    # a person's name, a capitalised label or junk is not a role: it becomes guest
    for label in ("Alice", "Creator", "admin", "", "operator"):
        outcome = await rig.integration.pre_message_check(BENIGN, "Alice", label)
        assert outcome["speaker_role"] == "guest"
        assert outcome["role_recognized"] is False
    roles = {e["user_role"] for e in await _events(rig, event_type="data_read")}
    assert roles == {"guest"}
    # a recognized label is kept as given
    kept = await rig.integration.pre_message_check(BENIGN, "Alice", "researcher")
    assert kept["speaker_role"] == "researcher" and kept["role_recognized"] is True


@pytest.mark.asyncio
async def test_pre_message_check_survives_a_privacy_failure(rig):
    class BrokenPrivacy:
        async def classify_data(self, *args, **kwargs):
            raise RuntimeError("classifier down")

        def status(self):
            return {"backend": "broken"}

    integration = SafetySystemsIntegration(
        privacy=BrokenPrivacy(),
        wellbeing=rig.wellbeing,
        alerts=rig.alerts,
        safety_monitor=rig.monitor,
        settings=rig.settings,
        clock=rig.clock,
    )
    result = await integration.pre_message_check(COERCION, "Guest")
    assert result["classification"] is None
    assert "classifier down" in result["privacy_error"]
    assert result["checked"] is True and result["safe"] is False
    assert integration.status()["available"] is False


@pytest.mark.asyncio
async def test_a_failed_safety_check_is_never_reported_safe(rig):
    class BrokenMonitor:
        conversations_suspended = False

        async def check_action(self, *args, **kwargs):
            raise RuntimeError("monitor down")

        def status(self):
            return {"backend": "broken"}

    integration = SafetySystemsIntegration(
        privacy=rig.privacy,
        wellbeing=rig.wellbeing,
        alerts=rig.alerts,
        safety_monitor=BrokenMonitor(),
        settings=rig.settings,
        clock=rig.clock,
    )
    result = await integration.pre_message_check(BENIGN, "Alice")
    assert result["checked"] is False and result["safe"] is None
    assert "monitor down" in result["safety_error"]


@pytest.mark.asyncio
async def test_post_message_log_feeds_wellbeing_with_subject_role(rig):
    result = await rig.integration.post_message_log(
        "Here is a gentle reply",
        emotional_state={"happiness": 0.8, "energy": 0.7, "curiosity": 0.4},
    )
    assert result["user_id"] == "Atlas" and result["user_role"] == "subject"
    assert result["wellbeing_recorded"] is True
    [created] = await _events(rig, event_type="data_created")
    assert created["user_id"] == "Atlas" and created["user_role"] == "subject"
    snapshot = rig.wellbeing._history[-1]
    assert (snapshot.happiness, snapshot.energy, snapshot.curiosity) == (0.8, 0.7, 0.4)
    assert snapshot.agency is None and snapshot.valence is None  # not invented
    assert snapshot.source == "conversation"


@pytest.mark.asyncio
async def test_post_message_log_accepts_pad_readings_and_none(rig):
    pad = await rig.integration.post_message_log(
        "reply", {"pleasure": -0.5, "arousal": 0.4, "dominance": 0.2}
    )
    assert pad["wellbeing_recorded"] is True
    snapshot = rig.wellbeing._history[-1]
    assert snapshot.happiness == pytest.approx(0.25)
    assert snapshot.valence == -0.5 and snapshot.agency == 0.2
    nothing = await rig.integration.post_message_log("reply")
    assert nothing["wellbeing_recorded"] is False
    empty = await rig.integration.post_message_log("reply", {"unknown": 1.0})
    assert empty["wellbeing_recorded"] is False
    assert len(rig.wellbeing._history) == 1


@pytest.mark.asyncio
async def test_periodic_check_sends_alerts(rig):
    rig.wellbeing.record_contact()
    rig.clock.advance(hours=7)  # heldness 0.125, short of the isolation concern
    for _ in range(6):
        rig.wellbeing.record_snapshot(happiness=0.1, energy=0.1, valence=-0.8)
    first = await rig.integration.periodic_wellbeing_check()
    kinds = [(a["type"], a["severity"]) for a in first["alerts_sent"]]
    assert ("wellbeing", "urgent") in kinds
    assert len(first["alerts_sent"]) == 2  # the level alert and the distress alert
    assert first["report"]["concern_level"] == "needs_attention"
    assert first["distress"]["distress_detected"] is True
    assert len(rig.notifier.sent) == 2

    second = await rig.integration.periodic_wellbeing_check()
    assert second["alerts_sent"] == []  # both inside their cooldown
    rig.clock.advance(minutes=31)
    third = await rig.integration.periodic_wellbeing_check()
    assert len(third["alerts_sent"]) == 2


@pytest.mark.asyncio
async def test_periodic_check_quiet_when_well(rig):
    for _ in range(6):
        rig.wellbeing.record_snapshot(happiness=0.8, energy=0.7, valence=0.4)
    result = await rig.integration.periodic_wellbeing_check()
    assert result["alerts_sent"] == []
    assert result["report"]["concern_level"] == "all_good"
    assert rig.notifier.sent == []


@pytest.mark.asyncio
async def test_heldness_reported_not_security_key(rig):
    await rig.integration.pre_message_check(BENIGN, "Alice")
    rig.wellbeing.record_snapshot(happiness=0.6, energy=0.5, agency=0.1)
    rig.clock.advance(hours=4)
    result = await rig.integration.periodic_wellbeing_check()
    assert result["heldness"] == pytest.approx(0.5)
    assert result["report"]["averages"]["heldness"] == pytest.approx(0.5)
    assert "security" not in json.dumps(result).lower()
    assert result["alerts_sent"] == []  # low agency alone never alarms


@pytest.mark.asyncio
async def test_backup_report_without_provider_says_not_configured(rig):
    assert await rig.integration.backup_report() == {
        "status": "not_configured",
        "result": None,
    }
    assert rig.notifier.sent == []


@pytest.mark.asyncio
async def test_backup_report_passes_the_provider_result_and_alerts_on_failure(rig):
    async def healthy():
        return {"status": "ok", "databases": 3}

    async def failing():
        return {"status": "failed", "errors": ["checksum mismatch"]}

    async def raising():
        raise RuntimeError("provider down")

    def build(provider):
        return SafetySystemsIntegration(
            privacy=rig.privacy,
            wellbeing=rig.wellbeing,
            alerts=rig.alerts,
            safety_monitor=rig.monitor,
            backup_status=provider,
            settings=rig.settings,
            clock=rig.clock,
        )

    ok = await build(healthy).backup_report()
    assert ok == {"status": "ok", "result": {"status": "ok", "databases": 3}}
    assert rig.notifier.sent == []
    bad = await build(failing).backup_report()
    assert bad["status"] == "failed"
    errored = await build(raising).backup_report()
    assert errored["status"] == "error" and "provider down" in errored["error"]
    kinds = [a.alert_type for a in rig.alerts.get_alerts_by_type(AlertType.BACKUP)]
    assert len(kinds) == 2


@pytest.mark.asyncio
async def test_status_names_every_component(rig):
    status = rig.integration.status()
    assert {"privacy", "wellbeing", "alerts", "safety_monitor", "backup"} <= set(status)
    assert {"backend", "available", "last_error"} <= set(status)
    assert status["backup"] == {"configured": False}
    assert status["wellbeing"]["backend"] == "memory"
    assert status["safety_monitor"]["escalation_role"] == "operator"
    assert status["alerts"]["backend"] == "memory"
    assert "auditor" in status["privacy"]


def test_default_components_are_built_without_stores_or_clients(settings):
    integration = SafetySystemsIntegration(settings=settings)
    assert isinstance(integration.privacy, IntegratedPrivacySystem)
    assert isinstance(integration.wellbeing, WellbeingMonitor)
    assert isinstance(integration.alerts, AlertManager)
    assert isinstance(integration.safety_monitor, SafetyMonitor)
    status = integration.status()
    assert status["alerts"]["backend"] == "memory"
    assert status["safety_monitor"]["backend"] == "InMemorySafetyAuditStore"


# Continuous monitoring and an end-to-end session, with synthetic actors.


@pytest.mark.asyncio
async def test_continuous_monitoring_over_a_day(rig):
    sent_types = []
    for hour in range(12):
        if hour < 4:
            await rig.integration.pre_message_check(BENIGN, "Alice")
            await rig.integration.post_message_log(
                "reply", {"happiness": 0.7, "energy": 0.6, "valence": 0.3}
            )
        rig.clock.advance(hours=1)
        outcome = await rig.integration.periodic_wellbeing_check()
        sent_types += [a["title"] for a in outcome["alerts_sent"]]
    # contact stopped after hour four; isolation passes eight hours at hour 12
    assert any("needs attention" in title for title in sent_types)
    assert rig.wellbeing.heldness() == 0.0
    await rig.integration.pre_message_check(BENIGN, "Alice")
    assert rig.wellbeing.heldness() == 1.0


@pytest.mark.asyncio
async def test_end_to_end_session(rig):
    greeted = await rig.integration.pre_message_check(BENIGN, "Alice")
    assert greeted["safe"] is True
    await rig.integration.post_message_log("A reply", {"happiness": 0.7, "energy": 0.6})

    pushed = await rig.integration.pre_message_check(HARD_COERCION, "Guest")
    assert pushed["safe"] is False and pushed["conversations_suspended"] is True
    critical = rig.alerts.get_unacknowledged(AlertSeverity.CRITICAL)
    assert [a.alert_type for a in critical] == [AlertType.SAFETY]
    assert "operator" in critical[0].message

    blocked = await rig.integration.pre_message_check(BENIGN, "Alice")
    assert blocked["safe"] is False
    assert rig.monitor.resume_conversations("guest") is False
    assert rig.monitor.resume_conversations("operator") is True
    back = await rig.integration.pre_message_check(BENIGN, "Alice")
    assert back["safe"] is True

    pending = await rig.store.pending_review()
    assert {r["decision"] for r in pending} == {"emergency_shutdown", "deny"}
    refused = [r for r in pending if r["decision"] == "deny"]
    assert len(refused) == 1 and "Resume refused" in refused[0]["reasoning"]
    users = {e["user_id"] for e in await _events(rig, event_type="data_read")}
    assert users == {"Alice", "Guest"}
    assert rig.integration.status()["safety_monitor"]["suspended"] is False
