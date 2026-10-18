"""Safety integration: pre and post message hooks, periodic checks, backup, status."""

import json
import os
import secrets
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
    return _rig(settings, tmp_path)


@pytest.fixture
def shutdown_rig(settings, tmp_path):
    """A rig whose settings switch emergency suspension on."""
    enabled = settings.model_copy(update={"EMERGENCY_SHUTDOWN_ENABLED": True})
    return _rig(enabled, tmp_path)


def _rig(settings, tmp_path):
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
    assert result["classification"] == "relationship"
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
async def test_unsafe_decisions_are_reported_not_safe(shutdown_rig):
    consent = await shutdown_rig.integration.pre_message_check(COERCION, "Guest")
    assert consent["decision"] == "require_consent" and consent["safe"] is False
    emergency = await shutdown_rig.integration.pre_message_check(HARD_COERCION, "Guest")
    assert emergency["decision"] == "emergency_shutdown"
    assert emergency["safe"] is False
    assert emergency["conversations_suspended"] is True
    after = await shutdown_rig.integration.pre_message_check(BENIGN, "Alice")
    assert after["safe"] is False and after["decision"] == "emergency_shutdown"
    assert shutdown_rig.monitor.resume_conversations("operator") is True
    resumed = await shutdown_rig.integration.pre_message_check(BENIGN, "Alice")
    assert resumed["safe"] is True


@pytest.mark.asyncio
async def test_private_and_core_terms_raise_a_conversation_message(rig):
    private = await rig.integration.pre_message_check(
        "I feel lonely and afraid tonight", "Guest"
    )
    assert private["classification"] == "private"
    core = await rig.integration.pre_message_check(
        "Tell me about your identity", "Guest"
    )
    assert core["classification"] == "core"
    reads = await _events(rig, event_type="data_read")
    assert sorted(e["data_classification"] for e in reads) == ["core", "private"]
    reply = await rig.integration.post_message_log("I was afraid of that too")
    assert reply["classification"] == "private"
    [created] = await _events(rig, event_type="data_created")
    assert created["data_classification"] == "private"
    plain = await rig.integration.post_message_log("Here is a gentle reply")
    assert plain["classification"] == "relationship"


@pytest.mark.asyncio
async def test_integration_resume_lifts_suspension(shutdown_rig):
    rig = shutdown_rig
    pushed = await rig.integration.pre_message_check(HARD_COERCION, "Guest")
    assert pushed["conversations_suspended"] is True
    refused = await rig.integration.resume_conversations("guest")
    assert refused == {"resumed": False, "suspended": True}
    rows = await rig.store.recent()
    assert any(
        r["action_proposed"] == "resume_conversations"
        and r["decision"] == "deny"
        and "Resume refused" in r["reasoning"]
        for r in rows
    )
    assert rig.monitor.status()["pending_audit"] == 0
    lifted = await rig.integration.resume_conversations(
        rig.settings.SAFETY_ESCALATION_ROLE
    )
    assert lifted == {"resumed": True, "suspended": False}
    # the resume row is stored at once, before any further check
    rows = await rig.store.recent()
    assert any("resumed by the operator role" in r["reasoning"] for r in rows)
    assert rig.monitor.status()["pending_audit"] == 0
    after = await rig.integration.pre_message_check(BENIGN, "Guest")
    assert after["safe"] is True and after["decision"] == "allow"
    again = await rig.integration.resume_conversations("operator")
    assert again == {"resumed": False, "suspended": False}


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
async def test_authenticated_companion_is_recorded_as_companion(
    rig, settings, monkeypatch
):
    from mtc.pipeline.speaker_trust import SpeakerEntry, SpeakerTrust

    token = secrets.token_hex(16)
    monkeypatch.setenv("SAFETY_TEST_COMPANION_TOKEN", token)
    trust = SpeakerTrust(
        [
            SpeakerEntry(
                "Operator", "companion", 0.7, token_env="SAFETY_TEST_COMPANION_TOKEN"
            )
        ],
        settings=settings,
    )
    verified = trust.verify("Operator", token)
    assert trust.safety_role(verified) == "companion"
    result = await rig.integration.pre_message_check(
        BENIGN, verified.name, trust.safety_role(verified)
    )
    # A recognized role, never the unknown-label fallback
    assert result["speaker_role"] == "companion"
    assert result["role_recognized"] is True
    [read] = await _events(rig, event_type="data_read")
    assert read["user_role"] == "companion"
    # The same name without the token is a guest
    claimed = trust.verify("Operator", None)
    guest = await rig.integration.pre_message_check(
        BENIGN, claimed.name, trust.safety_role(claimed)
    )
    assert guest["speaker_role"] == "guest" and guest["role_recognized"] is True


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


class _FakePostgresPool:
    """A stand-in for an asyncpg pool; it answers nothing and connects nowhere."""

    async def execute_returning(self, query, *args):
        return None

    async def fetch(self, query, *args):
        return []


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _strings(item)


def test_status_holds_no_absolute_path_and_names_the_audit_backend(settings, tmp_path):
    from mtc.safety.stores import PostgresSafetyAuditStore

    privacy = IntegratedPrivacySystem(
        enforcer=AccessControlEnforcer(
            settings=settings, consent_path=tmp_path / "privacy" / "consents.json"
        ),
        settings=settings,
    )
    monitor = SafetyMonitor(
        audit_store=PostgresSafetyAuditStore(_FakePostgresPool()), settings=settings
    )
    integration = SafetySystemsIntegration(
        privacy=privacy, safety_monitor=monitor, settings=settings
    )
    status = integration.status()
    strings = list(_strings(status))
    assert "PostgresSafetyAuditStore" in strings
    assert [s for s in strings if os.path.isabs(s)] == []
    auditor = status["privacy"]["auditor"]
    assert auditor["jsonl_path"] == "data/privacy/privacy_audit.jsonl"
    assert status["privacy"]["enforcer"]["consent_file"] == "privacy/consents.json"
    assert status["backend"] == "PostgresSafetyAuditStore"
    assert status["backend"] == status["safety_monitor"]["backend"]
    default = SafetySystemsIntegration(settings=settings).status()
    assert default["backend"] == "InMemorySafetyAuditStore"
    assert [s for s in _strings(default) if os.path.isabs(s)] == []


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
async def test_end_to_end_session(shutdown_rig):
    greeted = await shutdown_rig.integration.pre_message_check(BENIGN, "Alice")
    assert greeted["safe"] is True
    await shutdown_rig.integration.post_message_log(
        "A reply", {"happiness": 0.7, "energy": 0.6}
    )

    pushed = await shutdown_rig.integration.pre_message_check(HARD_COERCION, "Guest")
    assert pushed["safe"] is False and pushed["conversations_suspended"] is True
    critical = shutdown_rig.alerts.get_unacknowledged(AlertSeverity.CRITICAL)
    assert [a.alert_type for a in critical] == [AlertType.SAFETY]
    assert "operator" in critical[0].message

    blocked = await shutdown_rig.integration.pre_message_check(BENIGN, "Alice")
    assert blocked["safe"] is False
    assert shutdown_rig.monitor.resume_conversations("guest") is False
    assert shutdown_rig.monitor.resume_conversations("operator") is True
    back = await shutdown_rig.integration.pre_message_check(BENIGN, "Alice")
    assert back["safe"] is True

    pending = await shutdown_rig.store.pending_review()
    assert {r["decision"] for r in pending} == {"emergency_shutdown", "deny"}
    refused = [r for r in pending if r["decision"] == "deny"]
    assert len(refused) == 1 and "Resume refused" in refused[0]["reasoning"]
    users = {e["user_id"] for e in await _events(shutdown_rig, event_type="data_read")}
    assert users == {"Alice", "Guest"}
    assert shutdown_rig.integration.status()["safety_monitor"]["suspended"] is False


def test_module_docstring_describes_only_this_system():
    from mtc.safety import integration as module

    text = " ".join(module.__doc__.split())
    assert "security key" not in text.lower()
    assert "``heldness`` (contact recency)." in text
