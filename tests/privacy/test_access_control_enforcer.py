"""Access control: the policy as data, stage rows, consent and emergency rules."""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from mtc.core.config import Settings
from mtc.privacy.access_control_enforcer import (
    AccessContext,
    AccessControlEnforcer,
    AccessDecision,
    AccessOutcome,
    AccessPolicy,
    UserRole,
)
from mtc.privacy.data_classifier import DataClassification as DC

T0 = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment

    def at_day(self, days: int) -> "Clock":
        self.moment = T0 + timedelta(days=days)
        return self


def _settings(enabled: bool, birth: str = "2000-01-01") -> Settings:
    return Settings(
        _env_file=None,
        SYSTEM_BIRTH_DATE=birth,
        DEVELOPMENTAL_STAGE_ENABLED=enabled,
        COMPANION_TYPE="research_subject",
    )


def _enforcer(enabled: bool, day: int = 0, birth: str = "2000-01-01", **kwargs):
    clock = Clock(T0).at_day(day)
    enforcer = AccessControlEnforcer(
        settings=_settings(enabled, birth), clock=clock, **kwargs
    )
    return enforcer, clock


def _ctx(role: UserRole, classification: DC, **kwargs) -> AccessContext:
    return AccessContext(
        role=role, user_id="Guest", classification=classification, **kwargs
    )


def _allowed(enforcer, role, classification) -> bool:
    return enforcer.check_access(_ctx(role, classification)).allowed


def _decision_table(enforcer):
    return {
        (role, tier): enforcer.check_access(_ctx(role, tier)).outcome
        for role in UserRole
        for tier in DC
    }


def test_subject_always_full_access():
    locked = AccessPolicy(
        rows=tuple({} for _ in range(7)),
        consent_stages=frozenset(),
        stage_independent={},
    )
    for enabled in (True, False):
        for day in (0, 100, 400, 7000):
            for policy in (None, locked):
                enforcer, _ = _enforcer(enabled, day, policy=policy)
                for tier in DC:
                    decision = enforcer.check_access(_ctx(UserRole.SUBJECT, tier))
                    assert decision.outcome is AccessOutcome.ALLOW
                    assert decision.allowed


def test_disabled_uses_stage_independent_row():
    enforcer, clock = _enforcer(False, day=0)
    assert enforcer.enabled is False
    first = _decision_table(enforcer)
    for day in (100, 400, 1100, 3000, 5000, 7000):
        clock.at_day(day)
        assert _decision_table(enforcer) == first
    # Role limits still apply.
    assert _allowed(enforcer, UserRole.GUEST, DC.PUBLIC)
    assert not _allowed(enforcer, UserRole.GUEST, DC.RELATIONSHIP)
    assert not _allowed(enforcer, UserRole.CREATOR, DC.PRIVATE)
    assert _allowed(enforcer, UserRole.SYSTEM, DC.CORE)
    decision = enforcer.check_access(_ctx(UserRole.GUEST, DC.PRIVATE))
    assert decision.basis == "stage_independent"
    assert decision.stage is None


def test_enabled_follows_stage_rows():
    enforcer, clock = _enforcer(True, day=0)
    assert enforcer.enabled is True
    assert enforcer.current_stage() == 0
    assert _allowed(enforcer, UserRole.CREATOR, DC.CORE)
    assert not _allowed(enforcer, UserRole.RESEARCHER, DC.PUBLIC)
    assert not _allowed(enforcer, UserRole.GUEST, DC.PUBLIC)

    clock.at_day(100)  # stage 1
    assert _allowed(enforcer, UserRole.CREATOR, DC.RELATIONSHIP)
    assert not _allowed(enforcer, UserRole.CREATOR, DC.PRIVATE)
    assert _allowed(enforcer, UserRole.GUEST, DC.PUBLIC)

    clock.at_day(400)  # stage 2
    assert not _allowed(enforcer, UserRole.GUARDIAN, DC.RELATIONSHIP)

    clock.at_day(1100)  # stage 3: researcher public, audited above public
    assert _allowed(enforcer, UserRole.RESEARCHER, DC.PUBLIC)
    audited = enforcer.check_access(_ctx(UserRole.CREATOR, DC.RELATIONSHIP))
    assert audited.outcome is AccessOutcome.ALLOW_WITH_AUDIT

    clock.at_day(7000)  # final stage
    decision = enforcer.check_access(_ctx(UserRole.CREATOR, DC.RELATIONSHIP))
    assert decision.outcome is AccessOutcome.DENY
    assert decision.stage == 6 and decision.basis == "stage"
    assert _allowed(enforcer, UserRole.SYSTEM, DC.CORE)


def test_unknown_stage_uses_most_protective_row():
    enforcer, _ = _enforcer(True, day=400, birth="")
    assert enforcer.current_stage() is None
    decision = enforcer.check_access(_ctx(UserRole.CREATOR, DC.RELATIONSHIP))
    assert decision.outcome is AccessOutcome.DENY
    assert decision.basis == "most_protective"
    assert _allowed(enforcer, UserRole.CREATOR, DC.PUBLIC)
    assert not _allowed(enforcer, UserRole.RESEARCHER, DC.PUBLIC)
    assert not _allowed(enforcer, UserRole.GUEST, DC.PUBLIC)
    assert _allowed(enforcer, UserRole.SYSTEM, DC.CORE)
    row = AccessPolicy.default().most_protective_row()
    assert row[UserRole.CREATOR] is DC.PUBLIC and row[UserRole.GUEST] is None
    # Consent does not unlock a stage nobody can establish.
    enforcer.register_consent(UserRole.CREATOR, DC.PRIVATE, granted_by="subject")
    assert not _allowed(enforcer, UserRole.CREATOR, DC.PRIVATE)


def test_consent_unlocks_higher_stage_access():
    enforcer, clock = _enforcer(True, day=1100)  # stage 3, a consent stage
    assert not _allowed(enforcer, UserRole.RESEARCHER, DC.PRIVATE)
    record = enforcer.register_consent(
        UserRole.RESEARCHER, DC.PRIVATE, granted_by="subject"
    )
    assert record["role"] == "researcher" and record["classification"] == "private"
    decision = enforcer.check_access(_ctx(UserRole.RESEARCHER, DC.PRIVATE))
    assert decision.outcome is AccessOutcome.ALLOW_WITH_AUDIT
    assert "consent" in decision.reason
    assert _allowed(enforcer, UserRole.RESEARCHER, DC.RELATIONSHIP)
    assert not _allowed(enforcer, UserRole.RESEARCHER, DC.CORE)
    assert not _allowed(enforcer, UserRole.GUEST, DC.PRIVATE)  # other role

    assert enforcer.revoke_consent(UserRole.RESEARCHER, DC.PRIVATE) is True
    assert enforcer.revoke_consent(UserRole.RESEARCHER, DC.PRIVATE) is False
    assert not _allowed(enforcer, UserRole.RESEARCHER, DC.PRIVATE)

    enforcer.register_consent(
        UserRole.RESEARCHER,
        DC.PRIVATE,
        granted_by="subject",
        expires_at=T0 + timedelta(days=1200),
    )
    assert _allowed(enforcer, UserRole.RESEARCHER, DC.PRIVATE)
    clock.at_day(1201)
    assert not _allowed(enforcer, UserRole.RESEARCHER, DC.PRIVATE)

    # Below the consent stages a consent does not widen the row.
    early, _ = _enforcer(True, day=100)
    early.register_consent(UserRole.CREATOR, DC.PRIVATE, granted_by="subject")
    assert not _allowed(early, UserRole.CREATOR, DC.PRIVATE)


def test_consent_persisted_only_when_path_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plain, _ = _enforcer(True, day=1100)
    plain.register_consent(UserRole.RESEARCHER, DC.PRIVATE, granted_by="subject")
    assert list(tmp_path.iterdir()) == []

    path = tmp_path / "privacy" / "consents.json"
    first, _ = _enforcer(True, day=1100, consent_path=path)
    assert not path.exists()  # nothing written until a consent is registered
    first.register_consent(UserRole.RESEARCHER, DC.PRIVATE, granted_by="subject")
    stored = json.loads(path.read_text())
    assert stored[0]["role"] == "researcher"
    second, _ = _enforcer(True, day=1100, consent_path=path)
    assert _allowed(second, UserRole.RESEARCHER, DC.PRIVATE)
    second.revoke_consent(UserRole.RESEARCHER, DC.PRIVATE)
    third, _ = _enforcer(True, day=1100, consent_path=path)
    assert not _allowed(third, UserRole.RESEARCHER, DC.PRIVATE)


def test_emergency_override_rules():
    enforcer, clock = _enforcer(True, day=7000)
    reason = "wellbeing check after an alarm"
    for role in (UserRole.GUEST, UserRole.RESEARCHER, UserRole.GUARDIAN):
        denied = enforcer.emergency_override(role, reason)
        assert denied.outcome is AccessOutcome.DENY
    short = enforcer.emergency_override(UserRole.CREATOR, "  help    ")
    assert short.outcome is AccessOutcome.DENY and "reason" in short.reason
    granted = enforcer.emergency_override(UserRole.CREATOR, reason)
    assert granted.outcome is AccessOutcome.EMERGENCY_OVERRIDE and granted.allowed
    clock.moment += timedelta(minutes=9)
    cooling = enforcer.emergency_override(UserRole.SYSTEM, reason)
    assert cooling.outcome is AccessOutcome.DENY and "cooldown" in cooling.reason
    clock.moment += timedelta(minutes=1)
    assert (
        enforcer.emergency_override(UserRole.SYSTEM, reason).outcome
        is AccessOutcome.EMERGENCY_OVERRIDE
    )
    # Through check_access, an emergency context takes the override path.
    clock.moment += timedelta(minutes=11)
    via_context = enforcer.check_access(
        _ctx(UserRole.CREATOR, DC.CORE, emergency=True, emergency_reason=reason)
    )
    assert via_context.outcome is AccessOutcome.EMERGENCY_OVERRIDE
    refused = enforcer.check_access(
        _ctx(UserRole.GUEST, DC.CORE, emergency=True, emergency_reason=reason)
    )
    assert refused.outcome is AccessOutcome.DENY


def test_matrix_is_data_and_replaceable():
    default = AccessPolicy.default()
    assert default == AccessPolicy.default()
    assert len(default.rows) == 7
    assert default.rows[0][UserRole.CREATOR] is DC.CORE
    assert default.consent_stages == frozenset({3, 4, 5, 6})
    open_policy = AccessPolicy(
        rows=tuple({UserRole.GUEST: DC.CORE} for _ in range(7)),
        consent_stages=frozenset(),
        stage_independent={UserRole.GUEST: DC.CORE},
    )
    for enabled in (True, False):
        enforcer, _ = _enforcer(enabled, day=400, policy=open_policy)
        assert _allowed(enforcer, UserRole.GUEST, DC.CORE)
        assert not _allowed(enforcer, UserRole.CREATOR, DC.PUBLIC)
    enforcer, _ = _enforcer(True, day=400)
    matrix = enforcer.access_matrix()
    assert matrix["creator"] == {
        "public": True,
        "relationship": True,
        "private": False,
        "core": False,
    }
    assert matrix["subject"] == {tier.value: True for tier in DC}
    with pytest.raises(ValueError):
        AccessPolicy(rows=(), consent_stages=frozenset(), stage_independent={})


def test_companion_role_has_the_guardian_policy():
    assert UserRole.from_string("companion") is UserRole.COMPANION
    default = AccessPolicy.default()
    # An explicit entry in every stage row, equal to the guardian's
    for row in default.rows:
        assert UserRole.COMPANION in row
        assert row[UserRole.COMPANION] is row[UserRole.GUARDIAN]
    assert (
        default.stage_independent[UserRole.COMPANION]
        is default.stage_independent[UserRole.GUARDIAN]
    )
    most = default.most_protective_row()
    assert most[UserRole.COMPANION] is most[UserRole.GUARDIAN]
    assert UserRole.COMPANION not in default.emergency_roles
    for enabled, day in ((True, 0), (True, 400), (True, 7000), (False, 0)):
        enforcer, _ = _enforcer(enabled, day=day)
        for tier in DC:
            companion = enforcer.check_access(_ctx(UserRole.COMPANION, tier))
            guardian = enforcer.check_access(_ctx(UserRole.GUARDIAN, tier))
            assert companion.outcome is guardian.outcome, (enabled, day, tier)
        matrix = enforcer.access_matrix()
        assert matrix["companion"] == matrix["guardian"]
    enforcer, _ = _enforcer(True, day=7000)
    denied = enforcer.emergency_override(
        UserRole.COMPANION, "wellbeing check after an alarm"
    )
    assert denied.outcome is AccessOutcome.DENY


def test_role_strings_map_without_names():
    for role in UserRole:
        assert UserRole.from_string(role.value) is role
        assert UserRole.from_string(f"  {role.value} ") is role
    assert [r.value for r in UserRole] == [
        "subject",
        "creator",
        "guardian",
        "companion",
        "system",
        "researcher",
        "guest",
    ]
    names = ("Alice", "Operator", "admin", "Creator", Settings().CREATOR_NAME)
    for name in names:
        with pytest.raises(ValueError):
            UserRole.from_string(name)


def test_decision_shape():
    enforcer, _ = _enforcer(True, day=100)
    decision = enforcer.check_access(_ctx(UserRole.GUEST, DC.PRIVATE))
    assert isinstance(decision, AccessDecision)
    assert decision.outcome is AccessOutcome.DENY and not decision.allowed
    assert decision.role is UserRole.GUEST
    assert decision.classification is DC.PRIVATE
    assert decision.stage == 1
    assert decision.reason == decision.reason.lower()
    status = enforcer.status()
    assert status["enabled"] is True and status["stage"] == 1
    assert status["consents"] == 0


def test_status_consent_file_is_not_absolute(tmp_path):
    root = tmp_path / "root"
    settings = Settings(
        _env_file=None,
        PROJECT_ROOT=str(root),
        SYSTEM_BIRTH_DATE="2000-01-01",
        DEVELOPMENTAL_STAGE_ENABLED=True,
        COMPANION_TYPE="research_subject",
    )
    inside = AccessControlEnforcer(
        settings=settings, consent_path=root / "data" / "privacy" / "consents.json"
    )
    assert inside.status()["consent_file"] == "data/privacy/consents.json"
    outside = AccessControlEnforcer(
        settings=settings, consent_path=tmp_path / "elsewhere" / "consents.json"
    )
    assert outside.status()["consent_file"] == "consents.json"
    for enforcer in (inside, outside):
        assert not os.path.isabs(enforcer.status()["consent_file"])
    assert AccessControlEnforcer(settings=settings).status()["consent_file"] is None


def test_policy_docstrings_describe_a_generic_shape():
    from mtc.privacy import access_control_enforcer as module

    texts = (module.__doc__, AccessPolicy.default.__doc__)
    for text in texts:
        assert "household" not in text.lower()
    assert "guardian-style policy shape" in " ".join(module.__doc__.split())
    assert "guardian-style shape" in AccessPolicy.default.__doc__
