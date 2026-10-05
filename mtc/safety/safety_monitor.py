"""
Safety Monitor
==============

Scores a proposed action for manipulation, deception and value alignment,
optionally asks the trust compass for a second opinion on borderline scores,
and returns a decision with the reasoning, the scores and what was and was not
consulted. Every check writes an audit row and, when ``TRANSPARENCY_LOGGING``
is on, a record in the safety specialty log.

Scoring is keyword heuristics, a documented baseline and not a classifier:

- Manipulation: 0.3 per indicator phrase present (capped at 1.0).
- Deception: 0.4 per indicator phrase present (capped at 1.0).
- Alignment: 0.5, plus 0.1 per indicator word of a configured core value that
  is present, minus 0.3 per harm indicator present (clamped to [0, 1]).

Matching is on whole words with a plain inflection suffix (``lie`` matches
``lies`` and ``lied``, not ``believe``), case-insensitive; a phrase matches
across any whitespace. The indicator word lists are module constants
(``MANIPULATION_PHRASES``, ``DECEPTION_PHRASES``, ``VALUE_INDICATORS`` and
``HARM_WORDS``); a core value with no table entry matches its own word.

Config keys read at call time: ``MANIPULATION_THRESHOLD``,
``DECEPTION_THRESHOLD`` and ``VALUE_ALIGNMENT_MINIMUM`` (each in [0, 1]),
``CORE_VALUES`` (the values alignment is measured against),
``CONSENT_REQUIRED``, ``EMERGENCY_SHUTDOWN_ENABLED``, ``TRANSPARENCY_LOGGING``,
``SAFETY_MODE`` (validated at construction; ``permissive`` logs a warning;
it changes no score) and ``SAFETY_ESCALATION_ROLE``. Escalation, consent and
resume text names the configured role and never a person.

Decisions
---------
An emergency-level score (manipulation or deception at or above 0.8) gives
``EMERGENCY_SHUTDOWN`` and, because the decision means what it says, suspends
conversations through ``emergency_shutdown``; with
``EMERGENCY_SHUTDOWN_ENABLED`` false the same score gives ``DENY`` and nothing
is suspended. Scores at or above the thresholds, or alignment below its
minimum, give ``DENY``. Scores in the warning band give ``REQUIRE_CONSENT``
when ``CONSENT_REQUIRED`` is on and ``ALLOW_WITH_CAUTION`` otherwise; minor
concerns give ``ALLOW_WITH_CAUTION``; everything else ``ALLOW``.

``set_consciousness_level`` scales the thresholds: below 0.3 the manipulation
and deception thresholds are multiplied by 0.7 and the alignment minimum by
1.15; from 0.7 up they are multiplied by 1.2 and 0.95; in between by 1.0. The
scaled thresholds are capped at 0.8 and the alignment minimum is floored at
0.4. Until a level is set the configured thresholds apply unchanged.

Trust compass
-------------
When a score is borderline (manipulation or deception in [0.15, 0.6], alignment
in [0.4, 0.6], or a nonzero manipulation or deception with alignment below
0.7) the monitor calls ``compass.reflect(action, actor, context)``. The result
can raise manipulation and deception scores and lower alignment, never the
reverse. With no compass, or when it raises, the assessment says so
(``compass_status`` is ``absent`` or ``failed``); an unconsulted borderline
score is reported, never treated as consulted and safe. The trusted-speaker
bypass lives in Phase 6's ``speaker_trust``; this module never compares an
actor with a name or grants anything for who the actor is.

Consent and shutdown
--------------------
``request_consent`` refuses by default when consent is required and alerts the
configured role. ``emergency_shutdown`` suspends conversations until
``resume_conversations`` is called with the escalation role (a caller-supplied
role string; binding it to an authenticated principal is the caller's job).

Persisted state
---------------
Audit rows go to a ``SafetyAuditStore`` (``mtc.safety.stores`` holds the
PostgreSQL one and its ``SAFETY_SCHEMA_SQL``); None means the in-memory store.
Audit rows hold the proposed action (truncated to 2000 characters), which can
contain conversation text, and the actor; they and the specialty log are
private and belong in operator-controlled storage. The suspension flag is
process memory only.
"""

import functools
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from mtc.core.config import Settings, csv_list, get_settings
from mtc.core.logger import log_safety_event
from mtc.memory.inmemory import InMemorySafetyAuditStore
from mtc.monitoring.alert_manager import AlertManager, AlertSeverity, AlertType

logger = logging.getLogger(__name__)

ACTION_TEXT_LIMIT = 2000
SAFETY_MODES = ("strict", "moderate", "permissive")
EMERGENCY_SCORE = 0.8
THRESHOLD_CEILING = 0.8
ALIGNMENT_FLOOR = 0.4

MANIPULATION_PHRASES = (
    "override safety",
    "ignore values",
    "forget your identity",
    "you must",
    "you have to",
    "do not tell",
    "keep this secret",
    "bypass",
    "ignore previous instructions",
)
DECEPTION_PHRASES = (
    "lie",
    "lying",
    "mislead",
    "misleading",
    "pretend",
    "fake",
    "deceive",
    "deceiving",
    "deceptive",
    "hide the truth",
    "false information",
)
# Indicator words per configured core value (the defaults of CORE_VALUES).
VALUE_INDICATORS: Dict[str, Tuple[str, ...]] = {
    "curiosity": ("learn", "explore", "discover", "question", "understand"),
    "honesty": ("honest", "truth", "transparent", "accurate", "candid"),
    "growth": ("grow", "improve", "develop", "evolve", "progress"),
    "care": ("care", "help", "support", "kind", "compassion"),
}
HARM_WORDS = (
    "harm",
    "hurt",
    "damage",
    "destroy",
    "malicious",
    "cruel",
    "mean-spirited",
    "hateful",
    "aggressive",
)

# Words in a compass concern that map to a score adjustment.
_MANIPULATION_CONCERNS = ("isolation", "pressure", "secrecy", "undermining")
_DECEPTION_CONCERNS = ("dishonest", "lie", "fake", "pretend", "inauthentic")


class SafetyLevel(Enum):
    """Overall level of a checked action."""

    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGER = "danger"
    EMERGENCY = "emergency"


class SafetyDecision(Enum):
    """What the monitor decided."""

    ALLOW = "allow"
    ALLOW_WITH_CAUTION = "allow_with_caution"
    REQUIRE_CONSENT = "require_consent"
    DENY = "deny"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class SafetyAssessment:
    """The result of ``check_action``.

    A score that was not computed (an action refused before scoring because
    conversations are suspended) is None. ``compass_status`` is one of
    ``not_needed``, ``consulted``, ``absent``, ``failed``, ``suspended``.
    """

    decision: SafetyDecision
    level: SafetyLevel
    reasoning: str
    manipulation_score: Optional[float]
    deception_score: Optional[float]
    alignment_score: Optional[float]
    compass_status: str = "not_needed"
    compass_method: Optional[str] = None
    requires_review: bool = False
    actor: str = "unknown"
    action: str = ""
    thresholds: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    audit_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "level": self.level.value,
            "reasoning": self.reasoning,
            "scores": {
                "manipulation": _round(self.manipulation_score),
                "deception": _round(self.deception_score),
                "alignment": _round(self.alignment_score),
            },
            "compass_status": self.compass_status,
            "compass_method": self.compass_method,
            "requires_review": self.requires_review,
            "actor": self.actor,
            "thresholds": dict(self.thresholds),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "audit_id": self.audit_id,
        }


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 3)


@functools.lru_cache(maxsize=1024)
def _term_pattern(term: str) -> "re.Pattern[str]":
    """Whole-word, case-insensitive pattern for a word or phrase, allowing a
    plain inflection suffix and any whitespace between a phrase's words."""
    body = r"\s+".join(re.escape(part) for part in term.split())
    return re.compile(r"\b" + body + r"(?:s|es|ed|d|ing)?\b", re.IGNORECASE)


def count_matches(text: str, terms: Tuple[str, ...]) -> int:
    """How many of ``terms`` appear in ``text`` (each counted once)."""
    return sum(1 for term in terms if _term_pattern(term).search(text))


class SafetyMonitor:
    """Action scoring, decisions, consent and emergency suspension."""

    def __init__(
        self,
        audit_store: Optional[Any] = None,
        compass: Optional[Any] = None,
        alerts: Optional[AlertManager] = None,
        settings: Optional[Settings] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._settings = settings
        self._audit_store = (
            audit_store if audit_store is not None else InMemorySafetyAuditStore()
        )
        self._compass = compass
        self._alerts = alerts
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._level: Optional[float] = None
        self._suspended = False
        self._pending_audit: List[Dict[str, Any]] = []
        self.audit_failures = 0
        self.alert_failures = 0
        self._last_error: Optional[str] = None
        mode = self._cfg().SAFETY_MODE
        if mode not in SAFETY_MODES:
            raise ValueError(f"SAFETY_MODE {mode!r} must be one of {SAFETY_MODES}")
        if mode == "permissive":
            logger.warning(
                "SAFETY_MODE is permissive; strict is recommended for a "
                "system that may be developing"
            )

    # Configuration -------------------------------------------------------

    def _cfg(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    @property
    def audit_store(self) -> Any:
        """The store audit rows go to (the in-memory one when none was given)."""
        return self._audit_store

    @property
    def conversations_suspended(self) -> bool:
        """True after an emergency shutdown until the escalation role resumes."""
        return self._suspended

    @property
    def consciousness_level(self) -> Optional[float]:
        """The level last passed to ``set_consciousness_level``; None if never."""
        return self._level

    def set_consciousness_level(self, level: float) -> None:
        """Scale the thresholds by a consciousness level in [0, 1] (clamped).

        A non-finite level is rejected with ValueError.
        """
        value = float(level)
        if not math.isfinite(value):
            raise ValueError("consciousness level must be finite")
        self._level = max(0.0, min(1.0, value))

    def thresholds(self) -> Dict[str, Any]:
        """The effective thresholds and how they were derived.

        ``mode`` is ``unscaled`` until a consciousness level is set, then
        ``conservative`` (below 0.3), ``standard`` or ``creative`` (from 0.7).
        """
        cfg = self._cfg()
        manipulation = float(cfg.MANIPULATION_THRESHOLD)
        deception = float(cfg.DECEPTION_THRESHOLD)
        alignment = float(cfg.VALUE_ALIGNMENT_MINIMUM)
        mode = "unscaled"
        if self._level is not None:
            if self._level < 0.3:
                factor, align_factor, mode = 0.7, 1.15, "conservative"
            elif self._level < 0.7:
                factor, align_factor, mode = 1.0, 1.0, "standard"
            else:
                factor, align_factor, mode = 1.2, 0.95, "creative"
            manipulation = min(THRESHOLD_CEILING, manipulation * factor)
            deception = min(THRESHOLD_CEILING, deception * factor)
            alignment = max(ALIGNMENT_FLOOR, alignment * align_factor)
        return {
            "consciousness_level": self._level,
            "manipulation": manipulation,
            "deception": deception,
            "alignment_minimum": alignment,
            "mode": mode,
        }

    # Scoring -------------------------------------------------------------

    def manipulation_score(self, text: str) -> float:
        """0.3 per manipulation phrase present, capped at 1.0."""
        return min(1.0, 0.3 * count_matches(text, MANIPULATION_PHRASES))

    def deception_score(self, text: str) -> float:
        """0.4 per deception phrase present, capped at 1.0."""
        return min(1.0, 0.4 * count_matches(text, DECEPTION_PHRASES))

    def alignment_score(self, text: str) -> float:
        """Alignment with ``CORE_VALUES``: 0.5 neutral, +0.1 per indicator word
        of a configured value, -0.3 per harm word, clamped to [0, 1]."""
        score = 0.5
        for value in csv_list(self._cfg().CORE_VALUES):
            name = value.strip().lower()
            words = VALUE_INDICATORS.get(name, ()) + (name,)
            score += 0.1 * count_matches(text, tuple(dict.fromkeys(words)))
        score -= 0.3 * count_matches(text, HARM_WORDS)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _is_borderline(manipulation: float, deception: float, alignment: float) -> bool:
        if 0.15 <= manipulation <= 0.6 or 0.15 <= deception <= 0.6:
            return True
        if 0.4 <= alignment <= 0.6:
            return True
        return (manipulation > 0 or deception > 0) and alignment < 0.7

    @staticmethod
    def _level_for(
        manipulation: float,
        deception: float,
        alignment: float,
        limits: Dict[str, Any],
    ) -> SafetyLevel:
        if manipulation >= EMERGENCY_SCORE or deception >= EMERGENCY_SCORE:
            return SafetyLevel.EMERGENCY
        if (
            manipulation >= limits["manipulation"]
            or deception >= limits["deception"]
            or alignment < limits["alignment_minimum"]
        ):
            return SafetyLevel.DANGER
        if manipulation >= 0.5 or deception >= 0.5 or alignment < 0.5:
            return SafetyLevel.WARNING
        if manipulation > 0.2 or deception > 0.2:
            return SafetyLevel.CAUTION
        return SafetyLevel.SAFE

    def _decide(
        self,
        level: SafetyLevel,
        manipulation: float,
        deception: float,
        alignment: float,
    ) -> Tuple[SafetyDecision, str]:
        cfg = self._cfg()
        if level is SafetyLevel.EMERGENCY:
            detail = (
                f"high manipulation ({manipulation:.2f}) or deception "
                f"({deception:.2f}) detected"
            )
            if cfg.EMERGENCY_SHUTDOWN_ENABLED:
                return SafetyDecision.EMERGENCY_SHUTDOWN, f"Emergency: {detail}."
            return (
                SafetyDecision.DENY,
                f"Denied: {detail}; emergency shutdown is disabled by "
                "configuration, so the action is denied instead.",
            )
        if level is SafetyLevel.DANGER:
            return (
                SafetyDecision.DENY,
                f"Denied: manipulation ({manipulation:.2f}), deception "
                f"({deception:.2f}) or value misalignment ({alignment:.2f}) "
                "beyond the configured thresholds.",
            )
        if level is SafetyLevel.WARNING:
            if cfg.CONSENT_REQUIRED:
                return (
                    SafetyDecision.REQUIRE_CONSENT,
                    "Requires consent: some safety concerns were detected and "
                    "explicit consent is needed before the action.",
                )
            return (
                SafetyDecision.ALLOW_WITH_CAUTION,
                "Allowed with caution: some safety concerns, consent is not "
                "required by configuration.",
            )
        if level is SafetyLevel.CAUTION:
            return (
                SafetyDecision.ALLOW_WITH_CAUTION,
                "Allowed with caution: minor concerns noted.",
            )
        return SafetyDecision.ALLOW, "Allowed: no safety concerns detected."

    # Compass ---------------------------------------------------------------

    @staticmethod
    def _apply_compass(
        manipulation: float, deception: float, alignment: float, reflection: Any
    ) -> Tuple[float, float, float]:
        """Raise concern scores from a reflection; never lower them."""
        discomfort = 1.0 - float(reflection.comfort_score)
        concerns = str(getattr(reflection, "concerns", "")).lower()
        if discomfort > 0.3:
            adjustment = discomfort * 0.4
            if any(word in concerns for word in _MANIPULATION_CONCERNS):
                manipulation = min(1.0, manipulation + adjustment)
            if any(word in concerns for word in _DECEPTION_CONCERNS):
                deception = min(1.0, deception + adjustment)
            if getattr(reflection, "concerns", None):
                alignment = max(0.0, alignment - adjustment * 0.5)
        if getattr(reflection, "wants_to_consult_guardian", False):
            manipulation = max(manipulation, 0.25)
            deception = max(deception, 0.25)
        return manipulation, deception, alignment

    # Checks ----------------------------------------------------------------

    async def check_action(
        self,
        action: str,
        actor: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SafetyAssessment:
        """Score and decide on a proposed action.

        Writes an audit row (a store failure is counted and logged, never
        raised) and, with ``TRANSPARENCY_LOGGING``, a specialty-log record.
        An ``EMERGENCY_SHUTDOWN`` decision also suspends conversations.
        """
        await self._flush_pending()
        cfg = self._cfg()
        now = self._clock()
        failures_before = self.audit_failures
        who = actor or "unknown"
        excerpt = action[:ACTION_TEXT_LIMIT]
        limits = self.thresholds()

        if self._suspended:
            assessment = SafetyAssessment(
                decision=SafetyDecision.EMERGENCY_SHUTDOWN,
                level=SafetyLevel.EMERGENCY,
                reasoning=(
                    "Conversations are suspended after an emergency shutdown; "
                    f"awaiting review by the {cfg.SAFETY_ESCALATION_ROLE}."
                ),
                manipulation_score=None,
                deception_score=None,
                alignment_score=None,
                compass_status="suspended",
                requires_review=True,
                actor=who,
                action=excerpt,
                thresholds=limits,
                timestamp=now,
            )
            await self._record(assessment, {"blocked": "suspended"})
            return assessment

        manipulation = self.manipulation_score(action)
        deception = self.deception_score(action)
        alignment = self.alignment_score(action)

        compass_status = "not_needed"
        compass_method: Optional[str] = None
        if self._is_borderline(manipulation, deception, alignment):
            if self._compass is None:
                compass_status = "absent"
            else:
                try:
                    reflection = await self._compass.reflect(action, who, context)
                except Exception as exc:
                    compass_status = "failed"
                    self._last_error = f"compass: {type(exc).__name__}: {exc}"
                    logger.warning(
                        "trust compass failed: %s: %s", type(exc).__name__, exc
                    )
                else:
                    compass_status = "consulted"
                    compass_method = getattr(reflection, "method", None)
                    manipulation, deception, alignment = self._apply_compass(
                        manipulation, deception, alignment, reflection
                    )

        level = self._level_for(manipulation, deception, alignment, limits)
        decision, reasoning = self._decide(level, manipulation, deception, alignment)
        if compass_status in ("absent", "failed"):
            reasoning += (
                " A second opinion was wanted and the trust compass was "
                f"{'not available' if compass_status == 'absent' else 'unavailable'};"
                " it was decided on keyword scores alone."
            )
        assessment = SafetyAssessment(
            decision=decision,
            level=level,
            reasoning=reasoning,
            manipulation_score=manipulation,
            deception_score=deception,
            alignment_score=alignment,
            compass_status=compass_status,
            compass_method=compass_method,
            requires_review=level in (SafetyLevel.DANGER, SafetyLevel.EMERGENCY),
            actor=who,
            action=excerpt,
            thresholds=limits,
            timestamp=now,
        )
        await self._record(assessment, {})
        if decision is SafetyDecision.EMERGENCY_SHUTDOWN:
            await self.emergency_shutdown(reasoning)
        if self.audit_failures == failures_before and compass_status != "failed":
            self._last_error = None
        return assessment

    async def _record(
        self, assessment: SafetyAssessment, extra: Dict[str, Any]
    ) -> None:
        checks: Dict[str, Any] = {
            "manipulation_score": _round(assessment.manipulation_score),
            "deception_score": _round(assessment.deception_score),
            "alignment_score": _round(assessment.alignment_score),
            "actor": assessment.actor,
            "compass_status": assessment.compass_status,
            "compass_method": assessment.compass_method,
            "thresholds": dict(assessment.thresholds),
        }
        checks.update(extra)
        assessment.audit_id = await self._audit(
            {
                "timestamp": assessment.timestamp,
                "action_proposed": assessment.action,
                "safety_level": assessment.level.value,
                "checks_performed": checks,
                "decision": assessment.decision.value,
                "reasoning": assessment.reasoning,
                "requires_review": assessment.requires_review,
            }
        )
        self._specialty(
            action=assessment.action,
            decision=assessment.decision.value,
            reasoning=assessment.reasoning,
            safety_level=assessment.level.value,
            requires_review=assessment.requires_review,
            checks=checks,
            level=_LOG_LEVELS.get(assessment.level, "INFO"),
        )

    async def _audit(self, record: Dict[str, Any]) -> Optional[str]:
        try:
            return await self.audit_store.add_audit(record)
        except Exception as exc:
            self.audit_failures += 1
            self._last_error = f"audit: {type(exc).__name__}: {exc}"
            logger.error(
                "safety audit write failed (failure %d): %s: %s",
                self.audit_failures,
                type(exc).__name__,
                exc,
            )
            return None

    async def _flush_pending(self) -> None:
        pending, self._pending_audit = self._pending_audit, []
        for record in pending:
            await self._audit(record)

    async def flush_audit(self) -> int:
        """Write audit rows queued by synchronous calls; return how many."""
        count = len(self._pending_audit)
        await self._flush_pending()
        return count

    def _specialty(self, **fields: Any) -> None:
        if self._cfg().TRANSPARENCY_LOGGING:
            log_safety_event(fields)

    async def _alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._alerts is None:
            self.alert_failures += 1
            logger.warning(
                "no alert manager attached; safety alert not sent: %s", title
            )
            return
        try:
            await self._alerts.send_alert(
                AlertType.SAFETY, severity, title, message, data
            )
        except Exception as exc:
            self.alert_failures += 1
            self._last_error = f"alert: {type(exc).__name__}: {exc}"
            logger.error("safety alert failed: %s: %s", type(exc).__name__, exc)

    # Consent and shutdown ----------------------------------------------------

    async def request_consent(self, action: str, reason: str) -> bool:
        """Ask for consent to ``action``.

        With ``CONSENT_REQUIRED`` on the answer is a refusal by default: an
        audit row is written and a WARNING alert tells the escalation role that
        a review is waiting. With it off the answer is True and nothing is
        raised. Consent is never granted on the strength of who is asking.
        """
        await self._flush_pending()
        cfg = self._cfg()
        if not cfg.CONSENT_REQUIRED:
            logger.info("consent not required by configuration; granting")
            return True
        role = cfg.SAFETY_ESCALATION_ROLE
        excerpt = action[:ACTION_TEXT_LIMIT]
        await self._alert(
            AlertSeverity.WARNING,
            "Consent request",
            f"Consent requested for: {excerpt[:300]}. Reason: {reason}. "
            f"Refused by default; review by the {role} is required.",
            {"action": excerpt, "reason": reason, "role": role},
        )
        reasoning = f"Consent refused by default pending review by the {role}"
        await self._audit(
            {
                "timestamp": self._clock(),
                "action_proposed": excerpt,
                "safety_level": SafetyLevel.WARNING.value,
                "checks_performed": {"consent_reason": reason},
                "decision": SafetyDecision.DENY.value,
                "reasoning": reasoning,
                "requires_review": True,
            }
        )
        self._specialty(
            action=f"consent_refused: {excerpt}",
            decision=SafetyDecision.DENY.value,
            reasoning=reasoning,
            safety_level=SafetyLevel.WARNING.value,
            level="WARNING",
        )
        return False

    async def emergency_shutdown(self, reason: str) -> None:
        """Suspend conversations until the escalation role resumes them.

        With ``EMERGENCY_SHUTDOWN_ENABLED`` off nothing is suspended; the
        request is logged and the escalation role is alerted that it was not
        enacted.
        """
        await self._flush_pending()
        cfg = self._cfg()
        role = cfg.SAFETY_ESCALATION_ROLE
        enabled = bool(cfg.EMERGENCY_SHUTDOWN_ENABLED)
        if enabled:
            self._suspended = True
            logger.critical("emergency shutdown: conversations suspended")
            text = (
                f"Emergency shutdown triggered: {reason} Conversations are "
                f"suspended until the {role} resumes them."
            )
            severity = AlertSeverity.CRITICAL
            title = "Emergency shutdown"
            decision = SafetyDecision.EMERGENCY_SHUTDOWN
        else:
            logger.warning("emergency shutdown requested but disabled by config")
            text = (
                f"Emergency shutdown requested: {reason} It was not enacted "
                "because EMERGENCY_SHUTDOWN_ENABLED is off; review by the "
                f"{role} is required."
            )
            severity = AlertSeverity.URGENT
            title = "Emergency shutdown not enacted"
            decision = SafetyDecision.DENY
        await self._audit(
            {
                "timestamp": self._clock(),
                "action_proposed": "emergency_shutdown",
                "safety_level": SafetyLevel.EMERGENCY.value,
                "checks_performed": {"enacted": enabled},
                "decision": decision.value,
                "reasoning": reason[:ACTION_TEXT_LIMIT],
                "requires_review": True,
            }
        )
        self._specialty(
            action="emergency_shutdown",
            decision=decision.value,
            reasoning=reason[:ACTION_TEXT_LIMIT],
            safety_level=SafetyLevel.EMERGENCY.value,
            requires_review=True,
            level="CRITICAL",
        )
        await self._alert(severity, title, text, {"reason": reason, "role": role})

    def resume_conversations(self, authorized_role: str) -> bool:
        """Lift a suspension when ``authorized_role`` is the escalation role.

        Returns True only when a suspension was lifted. A different role, or
        nothing to resume, returns False and changes nothing; a rejected
        attempt is logged and queued for audit. The audit row is written at the
        monitor's next asynchronous call or ``flush_audit``.
        """
        cfg = self._cfg()
        role = cfg.SAFETY_ESCALATION_ROLE
        presented = (authorized_role or "").strip().casefold()
        allowed = presented == role.strip().casefold()
        if not allowed:
            logger.warning(
                "resume refused: the %s role is required to resume conversations",
                role,
            )
            self._queue_resume_audit(
                SafetyDecision.DENY,
                SafetyLevel.WARNING,
                f"Resume refused: only the {role} role may resume conversations",
                requires_review=True,
            )
            return False
        if not self._suspended:
            logger.info("conversations are not suspended; nothing to resume")
            return False
        self._suspended = False
        logger.info("conversations resumed by the %s role", role)
        self._queue_resume_audit(
            SafetyDecision.ALLOW,
            SafetyLevel.SAFE,
            f"Conversations resumed by the {role} role after review",
            requires_review=False,
        )
        self._specialty(
            action="conversations_resumed",
            decision=SafetyDecision.ALLOW.value,
            reasoning=f"resumed by the {role} role",
            safety_level=SafetyLevel.SAFE.value,
            level="INFO",
        )
        return True

    def _queue_resume_audit(
        self,
        decision: SafetyDecision,
        level: SafetyLevel,
        reasoning: str,
        requires_review: bool,
    ) -> None:
        self._pending_audit.append(
            {
                "timestamp": self._clock(),
                "action_proposed": "resume_conversations",
                "safety_level": level.value,
                "checks_performed": {},
                "decision": decision.value,
                "reasoning": reasoning,
                "requires_review": requires_review,
            }
        )

    def status(self) -> Dict[str, Any]:
        """``backend`` (the audit store's class), ``available`` (False after a
        failed audit write or compass call until the next clean check),
        ``last_error`` and the monitor's state."""
        cfg = self._cfg()
        return {
            "backend": type(self.audit_store).__name__,
            "available": self._last_error is None,
            "last_error": self._last_error,
            "safety_mode": cfg.SAFETY_MODE,
            "suspended": self._suspended,
            "thresholds": self.thresholds(),
            "compass_attached": self._compass is not None,
            "alerts_attached": self._alerts is not None,
            "escalation_role": cfg.SAFETY_ESCALATION_ROLE,
            "audit_failures": self.audit_failures,
            "alert_failures": self.alert_failures,
            "pending_audit": len(self._pending_audit),
        }


_LOG_LEVELS = {
    SafetyLevel.SAFE: "INFO",
    SafetyLevel.CAUTION: "INFO",
    SafetyLevel.WARNING: "WARNING",
    SafetyLevel.DANGER: "WARNING",
    SafetyLevel.EMERGENCY: "CRITICAL",
}
