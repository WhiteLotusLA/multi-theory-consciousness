"""
Safety Systems Integration
==========================

The single entry point a conversation pipeline calls around a turn. It wires
the privacy system, the wellbeing monitor, the alert manager and the safety
monitor together; components not passed are built (the privacy system, alert
manager, wellbeing monitor and safety monitor from Settings, with in-memory
stores and no network client).

- ``pre_message_check(message, speaker, speaker_role="guest")`` classifies the
  incoming message as a conversation whose private or core terms raise its
  tier (``keywords_can_raise``; an audited ``data_classified`` event), audits
  the read as a ``data_read`` event for the speaker and role at that tier,
  stamps contact on the wellbeing monitor and runs the safety monitor on the
  text. The result reports ``checked`` (whether a safety decision was
  reached) and ``safe``: True for ``allow`` and ``allow_with_caution``, False
  for any other decision, and None when the safety check could not run. A
  failed check is never reported as safe.
- ``resume_conversations(authorized_role)`` lifts an emergency suspension when
  the role is the escalation role, writes the audit rows at once and returns
  ``{"resumed": bool, "suspended": bool}``.
- ``post_message_log(response, emotional_state=None)`` classifies the response
  the same way, audits it as ``data_created`` for the configured
  ``SYSTEM_NAME`` with the role ``subject`` and, when an emotional state is
  given, records it on the wellbeing monitor. Missing readings stay missing.
- ``periodic_wellbeing_check()`` evaluates the wellbeing monitor, sends its
  alerts and adds one ``URGENT`` alert when the distress score crosses its
  threshold (at most once per the wellbeing cooldown). The report carries
  ``heldness`` (contact recency).
- ``backup_report()`` awaits the optional ``backup_status`` callable (it
  returns a dict with a ``status`` key; ``failed`` raises a ``BACKUP`` alert)
  and reports ``not_configured`` when none was passed. The backup verifier
  itself lives elsewhere; the runtime passes
  ``mtc.monitoring.backup_status.backup_report``.
- ``status()`` names every component's status.

Roles: ``speaker_role`` defaults to ``guest`` and grants nothing. A role label
that is not one of the privacy system's role labels (a person's name included)
is recorded as ``guest``, and the result says the label was not recognized. The
trusted-speaker bypass lives in the conversation pipeline's ``speaker_trust``
module, not here; nothing in this module reads a name to decide anything.

The module reads ``SYSTEM_NAME`` (the audited user id of the system's own
responses) and no other key itself; its components read theirs. Audit events,
alerts and the wellbeing record hold private runtime data. The module persists
nothing of its own.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from mtc.core.config import Settings, get_settings
from mtc.monitoring.alert_manager import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertType,
)
from mtc.monitoring.wellbeing_monitor import WellbeingMonitor, pad_to_wellbeing
from mtc.privacy.access_control_enforcer import UserRole
from mtc.privacy.privacy_auditor import EventType
from mtc.privacy.privacy_integration import IntegratedPrivacySystem
from mtc.safety.safety_monitor import SafetyDecision, SafetyMonitor

logger = logging.getLogger(__name__)

DEFAULT_ROLE = UserRole.GUEST.value
SUBJECT_ROLE = UserRole.SUBJECT.value
SAFE_DECISIONS = (SafetyDecision.ALLOW, SafetyDecision.ALLOW_WITH_CAUTION)

BackupStatus = Callable[[], Awaitable[Dict[str, Any]]]

_WELLBEING_KEYS = ("happiness", "energy", "curiosity", "agency", "valence")
_PAD_KEYS = ("pleasure", "arousal", "dominance")


def _role_label(label: Any) -> Tuple[str, bool]:
    """The privacy role label for ``label`` and whether it was recognized; an
    unrecognized label (a name included) becomes ``guest``."""
    try:
        return UserRole.from_string(str(label)).value, True
    except ValueError:
        return DEFAULT_ROLE, False


class SafetySystemsIntegration:
    """Pre-message checks, post-message logs, periodic wellbeing and backup
    reports over the privacy, wellbeing, alert and safety components."""

    def __init__(
        self,
        privacy: Optional[IntegratedPrivacySystem] = None,
        wellbeing: Optional[WellbeingMonitor] = None,
        alerts: Optional[AlertManager] = None,
        safety_monitor: Optional[SafetyMonitor] = None,
        backup_status: Optional[BackupStatus] = None,
        settings: Optional[Settings] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._settings = settings
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._alerts = (
            alerts if alerts is not None else AlertManager.from_settings(settings)
        )
        self._privacy = (
            privacy
            if privacy is not None
            else IntegratedPrivacySystem(settings=settings)
        )
        self._wellbeing = (
            wellbeing
            if wellbeing is not None
            else WellbeingMonitor(alerts=self._alerts, clock=clock)
        )
        self._monitor = (
            safety_monitor
            if safety_monitor is not None
            else SafetyMonitor(alerts=self._alerts, settings=settings, clock=clock)
        )
        self._backup_status = backup_status
        self._last_distress_alert: Optional[datetime] = None
        self._last_error: Optional[str] = None

    def _cfg(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    @property
    def privacy(self) -> IntegratedPrivacySystem:
        return self._privacy

    @property
    def wellbeing(self) -> WellbeingMonitor:
        return self._wellbeing

    @property
    def alerts(self) -> AlertManager:
        return self._alerts

    @property
    def safety_monitor(self) -> SafetyMonitor:
        return self._monitor

    async def pre_message_check(
        self,
        message: str,
        speaker: str,
        speaker_role: str = DEFAULT_ROLE,
    ) -> Dict[str, Any]:
        """Classify, audit and safety-check an incoming message.

        See the module docstring for the result. A privacy failure is reported
        (``classification`` is None and ``privacy_error`` names it) and does
        not stop the safety check; a safety-check failure leaves ``safe`` None.
        """
        role, recognized = _role_label(speaker_role)
        result: Dict[str, Any] = {
            "speaker": speaker,
            "speaker_role": role,
            "role_recognized": recognized,
            "classification": None,
            "checked": False,
            "safe": None,
            "decision": None,
            "assessment": None,
        }
        try:
            tier = await self._privacy.classify_data(
                message, content_type="conversation", keywords_can_raise=True
            )
            await self._privacy.log_data_event(
                EventType.DATA_READ,
                user_id=speaker,
                user_role=role,
                data_description=f"Incoming message from {speaker}",
                classification=tier,
            )
            result["classification"] = tier.value
        except Exception as exc:
            self._last_error = f"privacy: {type(exc).__name__}: {exc}"
            result["privacy_error"] = self._last_error
            logger.error("privacy classification failed: %s", self._last_error)

        self._wellbeing.record_contact()
        result["contact_recorded"] = True

        try:
            assessment = await self._monitor.check_action(
                message, actor=speaker, context={"speaker_role": role}
            )
        except Exception as exc:
            self._last_error = f"safety: {type(exc).__name__}: {exc}"
            result["safety_error"] = self._last_error
            logger.error("safety check failed: %s", self._last_error)
            return result
        result["checked"] = True
        result["decision"] = assessment.decision.value
        result["safe"] = assessment.decision in SAFE_DECISIONS
        result["assessment"] = assessment.to_dict()
        result["conversations_suspended"] = self._monitor.conversations_suspended
        return result

    async def resume_conversations(self, authorized_role: str) -> Dict[str, bool]:
        """Lift an emergency suspension for the escalation role.

        Calls the safety monitor's ``resume_conversations`` (which keeps its
        role check: any other role changes nothing and is audited as a
        refusal), then writes the queued audit rows at once. Returns
        ``resumed`` (a suspension was lifted) and ``suspended`` (whether
        conversations are suspended now). Binding the role to an authenticated
        trusted speaker is the caller's job.
        """
        resumed = bool(self._monitor.resume_conversations(authorized_role))
        await self._monitor.flush_audit()
        return {
            "resumed": resumed,
            "suspended": bool(self._monitor.conversations_suspended),
        }

    async def post_message_log(
        self,
        response: str,
        emotional_state: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Audit the system's response and feed its emotional state, if given,
        to the wellbeing monitor (user id from ``SYSTEM_NAME``, role
        ``subject``)."""
        user_id = self._cfg().SYSTEM_NAME
        result: Dict[str, Any] = {
            "user_id": user_id,
            "user_role": SUBJECT_ROLE,
            "classification": None,
            "wellbeing_recorded": False,
        }
        try:
            tier = await self._privacy.classify_data(
                response, content_type="conversation", keywords_can_raise=True
            )
            await self._privacy.log_data_event(
                EventType.DATA_CREATED,
                user_id=user_id,
                user_role=SUBJECT_ROLE,
                data_description="Response created",
                classification=tier,
            )
            result["classification"] = tier.value
        except Exception as exc:
            self._last_error = f"privacy: {type(exc).__name__}: {exc}"
            result["privacy_error"] = self._last_error
            logger.error("privacy logging failed: %s", self._last_error)
        if emotional_state:
            result["wellbeing_recorded"] = self._record_state(emotional_state)
        return result

    def _record_state(self, state: Dict[str, float]) -> bool:
        readings: Dict[str, Optional[float]] = {
            key: state.get(key) for key in _WELLBEING_KEYS
        }
        if all(key in state for key in _PAD_KEYS):
            mapped = pad_to_wellbeing(
                state["pleasure"], state["arousal"], state["dominance"]
            )
            for key, value in mapped.items():
                if readings.get(key) is None:
                    readings[key] = value
        snapshot = self._wellbeing.record_snapshot(source="conversation", **readings)
        return snapshot is not None

    async def periodic_wellbeing_check(self) -> Dict[str, Any]:
        """Evaluate wellbeing and send alerts; return the report and what was
        sent."""
        sent: List[Alert] = list(await self._wellbeing.check_and_alert())
        report = self._wellbeing.get_report()
        distress = self._wellbeing.detect_emotional_distress()
        if distress.get("distress_detected") and self._distress_due():
            alert = await self._alerts.send_alert(
                AlertType.WELLBEING,
                AlertSeverity.URGENT,
                "Distress score high",
                f"Distress score {distress['score']:.2f}",
                data=distress,
            )
            if alert is not None:
                self._last_distress_alert = self._clock()
                sent.append(alert)
        return {
            "report": report,
            "heldness": report["averages"]["heldness"],
            "distress": distress,
            "alerts_sent": [
                {
                    "type": a.alert_type.value,
                    "severity": a.severity.value,
                    "title": a.title,
                }
                for a in sent
            ],
        }

    def _distress_due(self) -> bool:
        if self._last_distress_alert is None:
            return True
        cooldown = timedelta(minutes=self._wellbeing.config.alert_cooldown_minutes)
        return self._clock() - self._last_distress_alert >= cooldown

    async def backup_report(self) -> Dict[str, Any]:
        """The backup status from the provider, or ``not_configured``."""
        if self._backup_status is None:
            return {"status": "not_configured", "result": None}
        try:
            result = dict(await self._backup_status())
        except Exception as exc:
            self._last_error = f"backup: {type(exc).__name__}: {exc}"
            logger.error("backup status provider failed: %s", self._last_error)
            await self._alerts.send_alert(
                AlertType.BACKUP,
                AlertSeverity.URGENT,
                "Backup status unavailable",
                f"The backup status provider failed: {type(exc).__name__}",
            )
            return {"status": "error", "error": self._last_error, "result": None}
        status = str(result.get("status", "unknown"))
        if status == "failed":
            await self._alerts.send_alert(
                AlertType.BACKUP,
                AlertSeverity.URGENT,
                "Backup failed",
                f"Backup status failed: {result.get('errors') or 'no detail'}",
                data=result,
            )
        return {"status": status, "result": result}

    def status(self) -> Dict[str, Any]:
        """Every component's status plus the integration's own.

        ``backend`` is the safety monitor's audit backend (the class of its
        audit store); the integration keeps no store of its own.
        """
        monitor = self._monitor.status()
        return {
            "backend": monitor.get("backend"),
            "available": self._last_error is None,
            "last_error": self._last_error,
            "privacy": self._privacy.status(),
            "wellbeing": self._wellbeing.status(),
            "alerts": self._alerts.status(),
            "safety_monitor": monitor,
            "backup": {"configured": self._backup_status is not None},
        }
