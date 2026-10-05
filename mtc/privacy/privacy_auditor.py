"""
Privacy Auditor
===============

Keeps the audit trail of data access: every event is held in a bounded
in-process log, appended to a JSON Lines file and written to a
``PrivacyAuditStore``. From the held events it detects violations and builds
daily and weekly reports. The trail is for accountability: it records who
read which tier of data and whether the access was granted.

- ``store`` None means ``InMemoryPrivacyAuditStore``; pass
  ``mtc.privacy.stores.MongoPrivacyAuditStore`` for the ``privacy_audit_log``
  collection. A store failure is counted and logged (once at WARNING, then
  at DEBUG) and never raised into the caller.
- The JSONL file is ``<DATA_DIRECTORY>/privacy/privacy_audit.jsonl``, resolved
  through ``mtc.core.paths`` at write time unless ``jsonl_path`` is given, and
  created on the first event, never earlier. ``jsonl_enabled=False`` writes
  none. A failed write is logged, not raised.
- ``max_events`` bounds the in-process log (default 10000) and the violation
  history; the oldest entries go first.

Violation rules (the thresholds are class attributes): every denied event is
an ``unauthorized_access`` violation (medium), reported once; more than 50
events by one user within an hour is ``bulk_access`` (high) and more than 20
within five minutes is ``pattern_anomaly`` (medium), each reported once while
the condition lasts. A violation's ``timestamp`` is the time of the access
that caused it and ``detected_at`` the time of detection.

The health score starts at 100, loses 2, 5, 15 or 30 per low, medium, high or
critical violation, and loses up to 20 more when more than a tenth of the
events were denied; it stays within [0, 100]. It is a bookkeeping figure, not
a measure of harm.

Reports carry counts only. Their ``compliance`` block lists consent requests
and grants, denied access, unresolved violations and the events audited; the
auditor checks no legal framework and claims none.

Private data: events hold user ids, roles and descriptions of what was read,
so the JSONL file, the store and the reports are private at runtime and live
in gitignored or operator-controlled locations. Config keys read:
``PROJECT_ROOT`` and ``DATA_DIRECTORY`` (the default JSONL path).
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from mtc.core.config import Settings, get_settings
from mtc.core.paths import data_dir
from mtc.memory.inmemory import InMemoryPrivacyAuditStore
from mtc.memory.protocols import PrivacyAuditStore
from mtc.privacy.data_classifier import DataClassification

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]

JSONL_RELATIVE = Path("privacy") / "privacy_audit.jsonl"


class EventType(Enum):
    """Kinds of privacy-relevant events."""

    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    DATA_READ = "data_read"
    DATA_CREATED = "data_created"
    DATA_DELETED = "data_deleted"
    DATA_MODIFIED = "data_modified"
    DATA_SHARED = "data_shared"
    DATA_EXPORTED = "data_exported"
    CONSENT_REQUESTED = "consent_requested"
    CONSENT_GRANTED = "consent_granted"
    STAGE_TRANSITION = "stage_transition"
    DATA_CLASSIFIED = "data_classified"
    ACCESS_ATTEMPT = "access_attempt"


class ViolationType(Enum):
    """Kinds of detected violations."""

    UNAUTHORIZED_ACCESS = "unauthorized_access"
    EXPIRED_CONSENT = "expired_consent"
    BULK_ACCESS = "bulk_access"
    PATTERN_ANOMALY = "pattern_anomaly"
    CROSS_BOUNDARY = "cross_boundary"


SEVERITY_DEDUCTIONS = {"low": 2.0, "medium": 5.0, "high": 15.0, "critical": 30.0}


def _aware(moment: datetime) -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(timezone.utc)


def _system_clock() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class PrivacyEvent:
    """One privacy-relevant event.

    ``subject_consent`` records whether the subject consented (None when the
    question did not arise). ``timestamp`` None is stamped by the auditor's
    clock when the event is logged.
    """

    event_type: EventType
    user_role: str
    user_id: str
    data_accessed: str
    data_classification: DataClassification
    success: bool
    subject_consent: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: Optional[datetime] = None

    def to_document(self) -> Dict[str, Any]:
        """The store document (``timestamp`` stays a datetime)."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "user_role": self.user_role,
            "user_id": self.user_id,
            "data_accessed": self.data_accessed,
            "data_classification": self.data_classification.value,
            "subject_consent": self.subject_consent,
            "success": self.success,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> Dict[str, Any]:
        """The JSON form (``timestamp`` as ISO 8601)."""
        doc = self.to_document()
        doc["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return doc


@dataclass
class PrivacyViolation:
    """One detected violation."""

    violation_type: ViolationType
    severity: str
    user_id: str
    description: str
    timestamp: datetime
    detected_at: datetime
    related_events: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution: Optional[str] = None
    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "timestamp": self.timestamp.isoformat(),
            "detected_at": self.detected_at.isoformat(),
            "violation_type": self.violation_type.value,
            "severity": self.severity,
            "user_id": self.user_id,
            "description": self.description,
            "related_events": list(self.related_events),
            "resolved": self.resolved,
            "resolution": self.resolution,
        }


class PrivacyAuditor:
    """Audit trail, violation detection and reports (see the module docstring)."""

    BULK_ACCESS_THRESHOLD = 50
    BULK_ACCESS_WINDOW = timedelta(hours=1)
    RAPID_ACCESS_THRESHOLD = 20
    RAPID_ACCESS_WINDOW = timedelta(minutes=5)

    def __init__(
        self,
        store: Optional[PrivacyAuditStore] = None,
        jsonl_path: Optional[Path] = None,
        max_events: int = 10000,
        settings: Optional[Settings] = None,
        clock: Optional[Clock] = None,
        *,
        jsonl_enabled: bool = True,
    ) -> None:
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self._settings = settings
        self._clock = clock or _system_clock
        self.max_events = int(max_events)
        self.store: PrivacyAuditStore = (
            store if store is not None else InMemoryPrivacyAuditStore(max_events)
        )
        self._jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        self._jsonl_enabled = bool(jsonl_enabled)
        self._events: Deque[PrivacyEvent] = deque(maxlen=self.max_events)
        self._violations: Deque[PrivacyViolation] = deque(maxlen=self.max_events)
        self._reported_denials: Set[str] = set()
        self._active_conditions: Set[Tuple[str, str]] = set()
        self._store_errors = 0
        self._store_available = True
        self._jsonl_errors = 0
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------ logging

    @property
    def jsonl_path(self) -> Optional[Path]:
        """Where the JSONL trail goes (resolved now), or None when disabled."""
        if not self._jsonl_enabled:
            return None
        if self._jsonl_path is not None:
            return self._jsonl_path
        settings = self._settings if self._settings is not None else get_settings()
        return data_dir(settings) / JSONL_RELATIVE

    def _now(self) -> datetime:
        return _aware(self._clock())

    def now(self) -> datetime:
        """The auditor's clock, as an aware UTC instant."""
        return self._now()

    async def log_event(self, event: PrivacyEvent) -> str:
        """Record one event and return its id; never raises on a write failure.

        A missing timestamp is stamped now; a naive one raises ValueError.
        """
        event.timestamp = (
            self._now() if event.timestamp is None else _aware(event.timestamp)
        )
        self._events.append(event)
        self._append_jsonl(event)
        try:
            await self.store.add_event(event.to_document())
            self._store_available = True
        except Exception as exc:  # the audit never breaks the caller
            self._store_available = False
            self._store_errors += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            level = logging.WARNING if self._store_errors == 1 else logging.DEBUG
            logger.log(level, "privacy audit store write failed: %s", self._last_error)
        return event.event_id

    def _append_jsonl(self, event: PrivacyEvent) -> None:
        path = self.jsonl_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), default=str) + "\n")
        except OSError as exc:
            self._jsonl_errors += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("privacy audit file write failed: %s", exc)

    # ------------------------------------------------------------ queries

    def event_count(self) -> int:
        """Events held in the in-process log."""
        return len(self._events)

    def events_for_user(self, user_id: str) -> List[PrivacyEvent]:
        """Held events of one user, oldest first."""
        return [e for e in self._events if e.user_id == user_id]

    def violations(self) -> List[PrivacyViolation]:
        """The violation history, oldest first."""
        return list(self._violations)

    def resolve_violation(self, violation_id: str, resolution: str) -> bool:
        """Mark a violation resolved; False when no such violation is held."""
        for violation in self._violations:
            if violation.violation_id == violation_id:
                violation.resolved = True
                violation.resolution = resolution
                return True
        return False

    # ------------------------------------------------------------ detection

    def detect_violations(
        self, now: Optional[datetime] = None
    ) -> List[PrivacyViolation]:
        """Apply the violation rules to the held events; return the new ones."""
        moment = _aware(now) if now is not None else self._now()
        found: List[PrivacyViolation] = []
        held_ids = {e.event_id for e in self._events}
        self._reported_denials &= held_ids
        for event in self._events:
            if event.success or event.event_id in self._reported_denials:
                continue
            self._reported_denials.add(event.event_id)
            found.append(
                PrivacyViolation(
                    violation_type=ViolationType.UNAUTHORIZED_ACCESS,
                    severity="medium",
                    user_id=event.user_id,
                    description=(
                        f"access denied for role {event.user_role} to "
                        f"{event.data_classification.value} data"
                    ),
                    timestamp=event.timestamp or moment,
                    detected_at=moment,
                    related_events=[event.event_id],
                )
            )

        by_user: Dict[str, List[PrivacyEvent]] = {}
        for event in self._events:
            by_user.setdefault(event.user_id, []).append(event)
        rules = (
            (
                ViolationType.BULK_ACCESS,
                "high",
                self.BULK_ACCESS_WINDOW,
                self.BULK_ACCESS_THRESHOLD,
            ),
            (
                ViolationType.PATTERN_ANOMALY,
                "medium",
                self.RAPID_ACCESS_WINDOW,
                self.RAPID_ACCESS_THRESHOLD,
            ),
        )
        for kind, severity, window, threshold in rules:
            for user_id, events in by_user.items():
                recent = [
                    e for e in events if e.timestamp and moment - e.timestamp < window
                ]
                condition = (kind.value, user_id)
                if len(recent) <= threshold:
                    self._active_conditions.discard(condition)
                    continue
                if condition in self._active_conditions:
                    continue
                self._active_conditions.add(condition)
                seconds = int(window.total_seconds())
                found.append(
                    PrivacyViolation(
                        violation_type=kind,
                        severity=severity,
                        user_id=user_id,
                        description=(
                            f"{len(recent)} events in {seconds} s "
                            f"(threshold {threshold})"
                        ),
                        timestamp=max(e.timestamp for e in recent),
                        detected_at=moment,
                        related_events=[e.event_id for e in recent],
                    )
                )
        self._violations.extend(found)
        return found

    # ------------------------------------------------------------ reports

    @staticmethod
    def _score(events: List[PrivacyEvent], violations: List[PrivacyViolation]) -> float:
        score = 100.0
        for violation in violations:
            score -= SEVERITY_DEDUCTIONS.get(violation.severity, 5.0)
        if events:
            denied_ratio = sum(1 for e in events if not e.success) / len(events)
            if denied_ratio > 0.1:
                score -= denied_ratio * 20.0
        return max(0.0, min(100.0, score))

    def _window(
        self, start: datetime, end: datetime
    ) -> Tuple[List[PrivacyEvent], List[PrivacyViolation]]:
        events = [e for e in self._events if e.timestamp and start <= e.timestamp < end]
        violations = [v for v in self._violations if start <= v.timestamp < end]
        return events, violations

    def health_score(self, now: Optional[datetime] = None) -> float:
        """The health score over the day before ``now`` (see the docstring)."""
        moment = _aware(now) if now is not None else self._now()
        events, violations = self._window(
            moment - timedelta(days=1), moment + timedelta(microseconds=1)
        )
        return self._score(events, violations)

    def _compliance(
        self, events: List[PrivacyEvent], violations: List[PrivacyViolation]
    ) -> Dict[str, Any]:
        return {
            "consent_requests": sum(
                1 for e in events if e.event_type is EventType.CONSENT_REQUESTED
            ),
            "consent_grants": sum(
                1 for e in events if e.event_type is EventType.CONSENT_GRANTED
            ),
            "denied_access": sum(1 for e in events if not e.success),
            "unresolved_violations": sum(1 for v in violations if not v.resolved),
            "events_audited": len(events),
            "note": (
                "counts from the audit trail only; no legal framework is checked "
                "or claimed"
            ),
        }

    @staticmethod
    def _counts(events: List[PrivacyEvent]) -> Tuple[Dict[str, int], Dict[str, int]]:
        by_type: Dict[str, int] = {}
        by_tier: Dict[str, int] = {}
        for event in events:
            by_type[event.event_type.value] = by_type.get(event.event_type.value, 0) + 1
            tier = event.data_classification.value
            by_tier[tier] = by_tier.get(tier, 0) + 1
        return by_type, by_tier

    def daily_report(self, day: date) -> Dict[str, Any]:
        """Counts, violations and the health score for one UTC day."""
        start = datetime.combine(day, time(0, 0), tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        events, violations = self._window(start, end)
        by_type, by_tier = self._counts(events)
        return {
            "report_type": "daily",
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "total_events": len(events),
            "events_by_type": by_type,
            "events_by_classification": by_tier,
            "violations_detected": len(violations),
            "violations": [v.to_dict() for v in violations],
            "health_score": self._score(events, violations),
            "compliance": self._compliance(events, violations),
        }

    def weekly_report(self, week_start: date) -> Dict[str, Any]:
        """Seven UTC days from ``week_start``: counts, concerns and a summary."""
        start = datetime.combine(week_start, time(0, 0), tzinfo=timezone.utc)
        end = start + timedelta(days=7)
        events, violations = self._window(start, end)
        by_type, by_tier = self._counts(events)
        concerns = [
            v
            for v in violations
            if not v.resolved and v.severity in ("high", "critical")
        ]
        users = {e.user_id for e in events}
        if not events:
            summary = "No data access was recorded in this period."
        else:
            summary = (
                f"In this period, data was accessed {len(events)} times by "
                f"{len(users)} users. "
                + (
                    "No high or critical violation is open."
                    if not concerns
                    else f"{len(concerns)} high or critical violations are open."
                )
            )
        return {
            "report_type": "weekly",
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "total_events": len(events),
            "unique_users": len(users),
            "events_by_type": by_type,
            "events_by_classification": by_tier,
            "violations_detected": len(violations),
            "concerns": [
                {"what": v.description, "severity": v.severity, "resolved": v.resolved}
                for v in concerns
            ],
            "summary": summary,
            "health_score": self._score(events, violations),
            "compliance": self._compliance(events, violations),
        }

    def status(self) -> Dict[str, Any]:
        """The store, the JSONL file and the error counts."""
        try:
            store_status = self.store.status()  # type: ignore[attr-defined]
        except Exception:
            store_status = {"backend": type(self.store).__name__}
        path = self.jsonl_path
        return {
            "backend": store_status.get("backend"),
            "available": self._store_available,
            "last_error": self._last_error,
            "store": store_status,
            "jsonl_path": str(path) if path is not None else None,
            "events_held": len(self._events),
            "violations_held": len(self._violations),
            "store_errors": self._store_errors,
            "jsonl_errors": self._jsonl_errors,
        }
