"""
Integrated Privacy System
=========================

One facade over the data classifier, the privacy auditor and the access
control enforcer, for the modules that classify content and check access
(the safety integration first among them).

- ``classify_data`` classifies and audits a ``data_classified`` event whose
  description names the content type and length, never the content itself.
  ``classify`` classifies without auditing.
- ``check_access`` maps a role label through ``UserRole.from_string`` (an
  unknown label, a person's name included, is denied), decides through the
  enforcer and audits an ``access_granted`` or ``access_denied`` event with
  the decision, the reason, the stage and the basis.
- ``register_consent`` records a consent and audits a ``consent_granted``
  event; ``log_data_event`` audits any other event.
- ``detect_violations``, ``get_report``, ``get_stage_info``,
  ``get_access_matrix``, ``get_status`` and ``status`` read the components.

Components not passed are built from Settings: ``DataClassifier``,
``PrivacyAuditor`` (an in-memory store and the JSONL trail under
``DATA_DIRECTORY``) and ``AccessControlEnforcer``. ``get_privacy_system`` is
the process-wide instance, built on the first call with the settings given
then (later calls return it unchanged); building it opens no network client.
Audit events hold user ids and access records and are private data.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from mtc.core.config import Settings
from mtc.privacy.access_control_enforcer import (
    AccessContext,
    AccessControlEnforcer,
    UserRole,
)
from mtc.privacy.data_classifier import (
    ClassificationResult,
    DataClassification,
    DataClassifier,
    stage_info,
)
from mtc.privacy.privacy_auditor import EventType, PrivacyAuditor, PrivacyEvent

logger = logging.getLogger(__name__)

SYSTEM_USER_ID = "system"
REPORT_TYPES = ("daily", "weekly")


class IntegratedPrivacySystem:
    """Classification, access control and auditing behind one interface."""

    def __init__(
        self,
        classifier: Optional[DataClassifier] = None,
        auditor: Optional[PrivacyAuditor] = None,
        enforcer: Optional[AccessControlEnforcer] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings
        self.classifier = classifier or DataClassifier(settings=settings)
        self.auditor = auditor or PrivacyAuditor(settings=settings)
        self.enforcer = enforcer or AccessControlEnforcer(settings=settings)
        self._initialized = False

    async def initialize(self) -> None:
        """Mark the facade ready; it opens no connection."""
        self._initialized = True

    @property
    def current_stage(self) -> Optional[int]:
        """The access stage now, or None when it cannot be established."""
        return self.enforcer.current_stage()

    # ------------------------------------------------------------ classify

    def classify(
        self,
        content: str,
        content_type: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> ClassificationResult:
        """Classify without auditing."""
        return self.classifier.classify(content, content_type, tags)

    async def classify_data(
        self,
        content: str,
        content_type: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> DataClassification:
        """Classify and audit the classification; return the tier."""
        result = self.classifier.classify(content, content_type, tags)
        await self.auditor.log_event(
            PrivacyEvent(
                event_type=EventType.DATA_CLASSIFIED,
                user_role=UserRole.SYSTEM.value,
                user_id=SYSTEM_USER_ID,
                data_accessed=(
                    f"{content_type or 'content'} of {len(content or '')} characters"
                ),
                data_classification=result.classification,
                success=True,
                metadata={"confidence": result.confidence, "reason": result.reason},
            )
        )
        return result.classification

    # ------------------------------------------------------------ access

    async def check_access(
        self,
        user_id: str,
        user_role: str,
        data_classification: Union[DataClassification, str],
        purpose: str = "",
        emergency: bool = False,
        emergency_reason: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Decide and audit one access request; return ``(allowed, reason)``."""
        tier = DataClassification.parse(data_classification)
        try:
            role = UserRole.from_string(user_role)
        except ValueError as exc:
            reason = str(exc)
            await self._audit_access(
                user_id, str(user_role), tier, purpose, False, {"reason": reason}
            )
            return False, reason
        decision = self.enforcer.check_access(
            AccessContext(
                role=role,
                user_id=user_id,
                classification=tier,
                purpose=purpose,
                emergency=emergency,
                emergency_reason=emergency_reason,
            )
        )
        await self._audit_access(
            user_id,
            role.value,
            tier,
            purpose,
            decision.allowed,
            {
                "decision": decision.outcome.value,
                "reason": decision.reason,
                "stage": decision.stage,
                "basis": decision.basis,
                "emergency": emergency,
            },
        )
        return decision.allowed, decision.reason

    async def _audit_access(
        self,
        user_id: str,
        user_role: str,
        tier: DataClassification,
        purpose: str,
        allowed: bool,
        metadata: Dict[str, Any],
    ) -> None:
        await self.auditor.log_event(
            PrivacyEvent(
                event_type=(
                    EventType.ACCESS_GRANTED if allowed else EventType.ACCESS_DENIED
                ),
                user_role=user_role,
                user_id=user_id,
                data_accessed=f"[{tier.value}] {purpose}".strip(),
                data_classification=tier,
                success=allowed,
                metadata=metadata,
            )
        )

    async def register_consent(
        self,
        role: Union[UserRole, str],
        classification: Union[DataClassification, str],
        granted_by: str,
    ) -> Dict[str, Any]:
        """Register a consent and audit it; return the consent record."""
        target = role if isinstance(role, UserRole) else UserRole.from_string(role)
        tier = DataClassification.parse(classification)
        record = self.enforcer.register_consent(target, tier, granted_by)
        await self.auditor.log_event(
            PrivacyEvent(
                event_type=EventType.CONSENT_GRANTED,
                user_role=target.value,
                user_id=str(granted_by),
                data_accessed=f"consent for role {target.value} up to {tier.value}",
                data_classification=tier,
                success=True,
                subject_consent=True,
            )
        )
        return record

    async def log_data_event(
        self,
        event_type: EventType,
        user_id: str,
        user_role: str,
        data_description: str,
        classification: Union[DataClassification, str],
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Audit one event; return its id."""
        return await self.auditor.log_event(
            PrivacyEvent(
                event_type=event_type,
                user_role=user_role,
                user_id=user_id,
                data_accessed=data_description,
                data_classification=DataClassification.parse(classification),
                success=success,
                metadata=dict(metadata or {}),
            )
        )

    # ------------------------------------------------------------ reports

    def detect_violations(self) -> List[Dict[str, Any]]:
        """Run violation detection; return the new violations as dicts."""
        return [v.to_dict() for v in self.auditor.detect_violations()]

    def get_report(
        self, report_type: str = "daily", day: Optional[date] = None
    ) -> Dict[str, Any]:
        """A daily report of ``day`` or a weekly report starting on ``day``.

        Without ``day``: today (UTC), or the week ending today.
        """
        today = self.auditor.now().date()
        if report_type == "daily":
            return self.auditor.daily_report(day or today)
        if report_type == "weekly":
            return self.auditor.weekly_report(day or today - timedelta(days=6))
        raise ValueError(f"unknown report type {report_type!r}; use {REPORT_TYPES}")

    def get_stage_info(self) -> Dict[str, Any]:
        """The access stage and its arithmetic."""
        info = stage_info(
            self.enforcer.now(),
            settings=self.enforcer.settings,
            policy=self.enforcer.stage_policy,
        )
        info["enabled"] = self.enforcer.enabled
        return info

    def get_access_matrix(self) -> Dict[str, Dict[str, bool]]:
        """Role to tier to readable under the row in force."""
        return self.enforcer.access_matrix()

    def status(self) -> Dict[str, Any]:
        """Every component's report."""
        enforcer = self.enforcer.status()
        return {
            "initialized": self._initialized,
            "stage": enforcer["stage"],
            "classifier": {
                "relationship_keywords": len(self.classifier.relationship_keywords()),
                "content_types": len(self.classifier.content_type_map),
                "protective_confidence": self.classifier.protective_confidence,
            },
            "auditor": self.auditor.status(),
            "enforcer": enforcer,
            "health_score": self.auditor.health_score(),
        }

    def get_status(self) -> Dict[str, Any]:
        """Alias of ``status`` under the facade's reference name."""
        return self.status()


_privacy_system: Optional[IntegratedPrivacySystem] = None


async def get_privacy_system(
    settings: Optional[Settings] = None,
) -> IntegratedPrivacySystem:
    """The process-wide privacy system, built and initialized on first use."""
    global _privacy_system
    if _privacy_system is None:
        system = IntegratedPrivacySystem(settings=settings)
        await system.initialize()
        _privacy_system = system
    return _privacy_system


def reset_privacy_system() -> None:
    """Forget the process-wide instance (tests and reconfiguration)."""
    global _privacy_system
    _privacy_system = None
