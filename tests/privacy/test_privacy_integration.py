"""Privacy facade: classification and access checks are audited, roles are labels."""

from datetime import datetime, timedelta, timezone

import pytest

from mtc.core.config import Settings
from mtc.memory.inmemory import InMemoryPrivacyAuditStore
from mtc.privacy import privacy_integration
from mtc.privacy.access_control_enforcer import AccessControlEnforcer, UserRole
from mtc.privacy.data_classifier import DataClassification as DC
from mtc.privacy.privacy_auditor import EventType, PrivacyAuditor
from mtc.privacy.privacy_integration import IntegratedPrivacySystem, get_privacy_system

T0 = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)


def _settings(tmp_path, **overrides) -> Settings:
    values = {
        "PROJECT_ROOT": str(tmp_path),
        "DATA_DIRECTORY": "data",
        "SYSTEM_BIRTH_DATE": "2000-01-01",
        "DEVELOPMENTAL_STAGE_ENABLED": True,
        "COMPANION_TYPE": "research_subject",
        **overrides,
    }
    return Settings(_env_file=None, **values)


def _system(tmp_path, day: int = 1100, **overrides) -> IntegratedPrivacySystem:
    settings = _settings(tmp_path, **overrides)

    def clock():
        return T0 + timedelta(days=day)

    return IntegratedPrivacySystem(
        auditor=PrivacyAuditor(settings=settings, clock=clock),
        enforcer=AccessControlEnforcer(settings=settings, clock=clock),
        settings=settings,
    )


@pytest.mark.asyncio
async def test_classify_audits_the_event(tmp_path):
    system = _system(tmp_path)
    content = "my brother told me a secret about the garden"
    tier = await system.classify_data(content)
    assert tier is DC.PRIVATE
    [event] = system.auditor.events_for_user("system")
    assert event.event_type is EventType.DATA_CLASSIFIED
    assert event.data_classification is DC.PRIVATE
    assert event.success is True
    assert "garden" not in event.data_accessed  # a description, not the content
    assert "secret" not in event.data_accessed
    assert event.metadata["confidence"] == pytest.approx(0.6)
    typed = await system.classify_data("hello", content_type="conversation")
    assert typed is DC.RELATIONSHIP
    assert system.auditor.event_count() == 2
    plain = system.classify("hello")
    assert plain.classification is DC.PUBLIC
    assert system.auditor.event_count() == 2  # classify alone is not audited


@pytest.mark.asyncio
async def test_check_access_maps_role_strings(tmp_path):
    system = _system(tmp_path, day=1100)  # stage 3
    allowed, reason = await system.check_access("Operator", "creator", DC.RELATIONSHIP)
    assert allowed is True and "creator" in reason
    denied, why = await system.check_access("Guest", "guest", DC.PRIVATE)
    assert denied is False and "denied" in why
    for label in ("Creator", "Alice", "admin", "operator"):
        ok, unknown = await system.check_access("Alice", label, DC.PUBLIC)
        assert ok is False and "unknown role" in unknown
    events = list(system.auditor.events_for_user("Operator"))
    assert events[0].event_type is EventType.ACCESS_GRANTED
    assert events[0].metadata["decision"] == "allow_with_audit"
    assert events[0].metadata["stage"] == 3
    guest_events = system.auditor.events_for_user("Guest")
    assert guest_events[0].event_type is EventType.ACCESS_DENIED
    assert guest_events[0].success is False
    assert len(system.auditor.events_for_user("Alice")) == 4
    violations = system.detect_violations()
    assert len(violations) == 5
    assert {v["violation_type"] for v in violations} == {"unauthorized_access"}

    consent = await system.register_consent("researcher", DC.PRIVATE, "subject")
    assert consent["role"] == "researcher"
    granted, _ = await system.check_access("Guest", "researcher", DC.PRIVATE)
    assert granted is True
    emergency, _ = await system.check_access(
        "Operator",
        "creator",
        DC.CORE,
        emergency=True,
        emergency_reason="synthetic emergency drill",
    )
    assert emergency is True


@pytest.mark.asyncio
async def test_status_names_components(tmp_path):
    system = _system(tmp_path, day=100)
    await system.initialize()
    await system.classify_data("hello")
    status = system.status()
    assert status["initialized"] is True
    assert status["stage"] == 1
    for component in ("classifier", "auditor", "enforcer"):
        assert component in status
    assert status["auditor"]["backend"] == "memory"
    assert status["auditor"]["events_held"] == 1
    assert status["enforcer"]["enabled"] is True
    assert 0.0 <= status["health_score"] <= 100.0
    assert system.get_status() == status
    info = system.get_stage_info()
    assert info["stage"] == 1 and info["age_days"] == 100
    matrix = system.get_access_matrix()
    assert matrix["creator"]["relationship"] is True
    daily = system.get_report("daily", (T0 + timedelta(days=100)).date())
    assert daily["total_events"] == 1
    weekly = system.get_report("weekly", (T0 + timedelta(days=97)).date())
    assert weekly["report_type"] == "weekly"
    with pytest.raises(ValueError):
        system.get_report("monthly")
    event_id = await system.log_data_event(
        EventType.DATA_READ, "Alice", "guest", "a synthetic record", DC.PUBLIC
    )
    assert event_id


@pytest.mark.asyncio
async def test_singleton_builds_from_settings_without_store_by_default(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(privacy_integration, "_privacy_system", None)
    settings = _settings(tmp_path, DEVELOPMENTAL_STAGE_ENABLED=False)
    first = await get_privacy_system(settings)
    second = await get_privacy_system()
    assert first is second
    assert first.status()["initialized"] is True
    assert isinstance(first.auditor.store, InMemoryPrivacyAuditStore)
    assert first.enforcer.enabled is False
    assert first.auditor.jsonl_path == tmp_path / "data" / "privacy" / (
        "privacy_audit.jsonl"
    )
    assert not (tmp_path / "data").exists()  # nothing written until an event
    privacy_integration.reset_privacy_system()
    assert privacy_integration._privacy_system is None


def test_package_exports_renamed_classes():
    import mtc.privacy as package

    for name in (
        "DataClassifier",
        "DataClassification",
        "PrivacyAuditor",
        "PrivacyEvent",
        "EventType",
        "ViolationType",
        "AccessControlEnforcer",
        "AccessDecision",
        "AccessOutcome",
        "AccessPolicy",
        "AccessContext",
        "UserRole",
        "IntegratedPrivacySystem",
        "get_privacy_system",
    ):
        assert hasattr(package, name), name
    assert [r.value for r in package.UserRole][0] == "subject"
    assert not hasattr(package, "DevelopmentalStage")
    assert UserRole.SUBJECT.value == "subject"
