"""
Alert Manager
=============

A small alerting hub. A component sends an alert (a type, a severity, a title,
a message and optional context); the manager keeps it in a bounded in-process
ring, writes it to an optional ``AlertStore`` and delivers it to every notifier
whose severity floor it meets.

Alert types: ``WELLBEING``, ``PRIVACY``, ``BACKUP``, ``SYSTEM``,
``CONSCIOUSNESS`` and ``SAFETY``. Severities, lowest first: ``INFO``,
``WARNING``, ``URGENT`` and ``CRITICAL``; severities compare by that order.

Notifiers
---------
A notifier is any object with ``async send(body, channel, priority, title,
activity_type) -> bool`` and, optionally, a ``min_severity`` attribute (the
manager skips the notifier for lower alerts; the default floor is ``INFO``).
The call shape lets an operator add a channel without editing the manager.
Only two notifiers ship: ``ConsoleNotifier`` (a log record) and
``FileNotifier`` (a JSON Lines file). No push, mail or chat channel exists in
this package. Alerts can carry private context (conversation excerpts, speaker
names, wellbeing readings); a notifier an operator adds sends that text where
the operator points it.

The manager calls each notifier with ``channel="auto"`` and a priority derived
from the severity (``INFO`` and ``WARNING`` are ``low``, ``URGENT`` is
``normal``, ``CRITICAL`` is ``urgent``). The title is the configured
``SYSTEM_NAME`` followed by the alert title.

Failure visibility
------------------
A channel that reports problems must not hide its own failures. A notifier that
raises, times out or returns False, and a store write that raises, are counted
(``notify_failures``, ``persist_failures``), logged at ERROR and appended to a
local failure log (``alerts_failed.log`` under ``LOG_DIRECTORY``, or the path
passed in). The failure log holds the alert title and message and is private.
A failing store never blocks delivery and a failing notifier never blocks the
store or the other notifiers.

Config keys read: ``SYSTEM_NAME`` (notifier titles), ``LOG_DIRECTORY`` (the
failure log and the default ``FileNotifier`` path) and ``ALERT_NOTIFIERS``
(``AlertManager.from_settings``: a comma list over ``console`` and ``file``).
Persisted state: the optional ``AlertStore`` and the files named above; the
ring is process memory only. The MongoDB store lives in
``mtc.monitoring.stores``. Nothing in the module reads a clock except through
the injectable ``clock``.
"""

import asyncio
import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from mtc.core.config import Settings, csv_list, get_settings
from mtc.core.paths import ensure_dir, log_dir

logger = logging.getLogger(__name__)

DEFAULT_RING_SIZE = 10000
NOTIFIER_TIMEOUT_S = 10.0
FAILURE_LOG_NAME = "alerts_failed.log"
FILE_NOTIFIER_NAME = "alerts.jsonl"
NOTIFIER_NAMES = ("console", "file")


class AlertType(Enum):
    """Categories of alerts."""

    WELLBEING = "wellbeing"
    PRIVACY = "privacy"
    BACKUP = "backup"
    SYSTEM = "system"
    CONSCIOUSNESS = "consciousness"
    SAFETY = "safety"


class AlertSeverity(Enum):
    """Alert severity, ordered ``INFO`` < ``WARNING`` < ``URGENT`` < ``CRITICAL``."""

    INFO = "info"
    WARNING = "warning"
    URGENT = "urgent"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Position in the severity order, lowest first."""
        return _SEVERITY_ORDER.index(self)

    def __ge__(self, other: Any) -> Any:
        if not isinstance(other, AlertSeverity):
            return NotImplemented
        return self.rank >= other.rank

    def __gt__(self, other: Any) -> Any:
        if not isinstance(other, AlertSeverity):
            return NotImplemented
        return self.rank > other.rank

    def __le__(self, other: Any) -> Any:
        if not isinstance(other, AlertSeverity):
            return NotImplemented
        return self.rank <= other.rank

    def __lt__(self, other: Any) -> Any:
        if not isinstance(other, AlertSeverity):
            return NotImplemented
        return self.rank < other.rank


_SEVERITY_ORDER = (
    AlertSeverity.INFO,
    AlertSeverity.WARNING,
    AlertSeverity.URGENT,
    AlertSeverity.CRITICAL,
)

_SEVERITY_TO_PRIORITY = {
    AlertSeverity.INFO: "low",
    AlertSeverity.WARNING: "low",
    AlertSeverity.URGENT: "normal",
    AlertSeverity.CRITICAL: "urgent",
}

_PRIORITY_TO_LOG_LEVEL = {
    "low": logging.INFO,
    "normal": logging.WARNING,
    "urgent": logging.ERROR,
}


def _json_safe(value: Any) -> Any:
    """A JSON-round-tripped copy; anything not serializable becomes its text."""
    return json.loads(json.dumps(value, default=str))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Alert:
    """One alert. ``timestamp`` is timezone-aware UTC."""

    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dictionary (timestamps as ISO 8601 strings)."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "context": _json_safe(self.context),
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
        }

    def to_record(self) -> Dict[str, Any]:
        """The document an ``AlertStore`` receives: the same keys as
        ``to_dict`` with timezone-aware datetimes for the two timestamps."""
        record = self.to_dict()
        record["timestamp"] = self.timestamp
        record["acknowledged_at"] = self.acknowledged_at
        return record


@runtime_checkable
class Notifier(Protocol):
    """A delivery channel. Returns True when the message was delivered."""

    async def send(
        self,
        body: str,
        channel: str,
        priority: str,
        title: str,
        activity_type: str,
    ) -> bool: ...


class ConsoleNotifier:
    """Writes one log record per alert (ERROR for ``urgent`` priority, WARNING
    for ``normal`` and INFO for ``low``). Receives alerts at or above
    ``min_severity``."""

    name = "console"

    def __init__(self, min_severity: AlertSeverity = AlertSeverity.WARNING) -> None:
        self.min_severity = min_severity

    async def send(
        self,
        body: str,
        channel: str,
        priority: str,
        title: str,
        activity_type: str,
    ) -> bool:
        level = _PRIORITY_TO_LOG_LEVEL.get(priority, logging.INFO)
        logger.log(level, "%s | %s", title, body)
        return True


class FileNotifier:
    """Appends one JSON Lines record per alert to a file.

    The default path is ``<LOG_DIRECTORY>/alerts.jsonl``, resolved at the first
    send; the directory is created then. The file holds alert messages, which
    can carry private context, and belongs in a gitignored or
    operator-controlled directory.
    """

    name = "file"

    def __init__(
        self,
        path: Optional[Path] = None,
        min_severity: AlertSeverity = AlertSeverity.INFO,
        *,
        settings: Optional[Settings] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.min_severity = min_severity
        self._path = Path(path) if path is not None else None
        self._settings = settings
        self._clock = clock or _utc_now

    @property
    def path(self) -> Path:
        """The file the notifier writes (the default is resolved on demand)."""
        if self._path is not None:
            return self._path
        return log_dir(self._settings) / FILE_NOTIFIER_NAME

    def _append(self, line: str) -> None:
        target = self.path
        ensure_dir(target.parent)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    async def send(
        self,
        body: str,
        channel: str,
        priority: str,
        title: str,
        activity_type: str,
    ) -> bool:
        record = {
            "timestamp": self._clock().isoformat(),
            "priority": priority,
            "channel": channel,
            "activity_type": activity_type,
            "title": title,
            "body": body,
        }
        await asyncio.to_thread(self._append, json.dumps(record))
        return True


def build_notifiers(
    names: Sequence[str], settings: Optional[Settings] = None
) -> List[Notifier]:
    """Build the shipped notifiers named in ``names`` (``console``, ``file``).

    An unknown name raises ValueError naming the allowed ones.
    """
    built: List[Notifier] = []
    for raw in names:
        name = raw.strip().lower()
        if name == "console":
            built.append(ConsoleNotifier())
        elif name == "file":
            built.append(FileNotifier(settings=settings))
        else:
            raise ValueError(
                f"unknown alert notifier {raw!r}; allowed: {', '.join(NOTIFIER_NAMES)}"
            )
    return built


class AlertManager:
    """Sends, stores, suppresses and acknowledges alerts.

    ``store`` is an optional ``AlertStore`` (``mtc.memory.protocols``); with
    none, alerts live in the bounded in-process ring only. ``notifiers`` is a
    sequence of notifiers; None means a single ``ConsoleNotifier`` and an empty
    sequence means no delivery. ``failure_log`` overrides the failure-log path.
    ``clock`` returns timezone-aware UTC instants.
    """

    def __init__(
        self,
        store: Optional[Any] = None,
        notifiers: Optional[Sequence[Notifier]] = None,
        failure_log: Optional[Path] = None,
        ring_size: int = DEFAULT_RING_SIZE,
        settings: Optional[Settings] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._store = store
        self._notifiers: List[Notifier] = (
            [ConsoleNotifier()] if notifiers is None else list(notifiers)
        )
        self._failure_log = Path(failure_log) if failure_log is not None else None
        self._settings = settings
        self._clock = clock or _utc_now
        self._ring: Deque[Alert] = deque(maxlen=max(int(ring_size), 1))
        self._suppressed: Dict[AlertType, datetime] = {}
        self.notify_failures = 0
        self.persist_failures = 0
        self.breadcrumb_failures = 0
        self.suppressed_count = 0
        self._last_error: Optional[str] = None

    @classmethod
    def from_settings(
        cls,
        settings: Optional[Settings] = None,
        store: Optional[Any] = None,
    ) -> "AlertManager":
        """Build a manager whose notifiers come from ``ALERT_NOTIFIERS``.

        No store is built here (pass one, for example a ``MongoAlertStore``);
        no network client is created.
        """
        cfg = settings if settings is not None else get_settings()
        return cls(
            store=store,
            notifiers=build_notifiers(csv_list(cfg.ALERT_NOTIFIERS), cfg),
            settings=cfg,
        )

    def _cfg(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    @property
    def failure_log_path(self) -> Path:
        """Where delivery and persistence failures leave a breadcrumb."""
        if self._failure_log is not None:
            return self._failure_log
        return log_dir(self._cfg()) / FAILURE_LOG_NAME

    @property
    def notifiers(self) -> List[Notifier]:
        """The configured notifiers (a copy of the list)."""
        return list(self._notifiers)

    async def send_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Alert]:
        """Record and deliver an alert.

        Returns the alert, or None when the type is suppressed. The ring takes
        the alert first; the store write and each notifier then run
        independently, so a failure in one never prevents the others.
        """
        now = self._clock()
        until = self._suppressed.get(alert_type)
        if until is not None:
            if now < until:
                self.suppressed_count += 1
                return None
            del self._suppressed[alert_type]

        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            context=_json_safe(data or {}),
            timestamp=now,
        )
        self._ring.append(alert)
        failures_before = self.notify_failures + self.persist_failures
        await self._persist(alert)
        await self._deliver(alert)
        if self.notify_failures + self.persist_failures == failures_before:
            self._last_error = None
        return alert

    async def _persist(self, alert: Alert) -> None:
        if self._store is None:
            return
        try:
            await self._store.add_alert(alert.to_record())
        except Exception as exc:
            self.persist_failures += 1
            self._last_error = f"store: {type(exc).__name__}: {exc}"
            logger.error(
                "alert store write failed (failure %d): %s: %s",
                self.persist_failures,
                type(exc).__name__,
                exc,
            )
            self._breadcrumb(alert, "store", exc)

    async def _deliver(self, alert: Alert) -> None:
        priority = _SEVERITY_TO_PRIORITY[alert.severity]
        title = f"{self._cfg().SYSTEM_NAME} alert: {alert.title}"
        body = (
            f"[{alert.severity.value.upper()}] "
            f"{alert.alert_type.value}: {alert.message}"
        )
        for notifier in self._notifiers:
            floor = getattr(notifier, "min_severity", AlertSeverity.INFO)
            if alert.severity < floor:
                continue
            try:
                delivered = await asyncio.wait_for(
                    notifier.send(
                        body=body,
                        channel="auto",
                        priority=priority,
                        title=title,
                        activity_type="alert",
                    ),
                    timeout=NOTIFIER_TIMEOUT_S,
                )
                if delivered is False:
                    raise RuntimeError("notifier reported the alert as undelivered")
            except Exception as exc:
                self.notify_failures += 1
                label = type(notifier).__name__
                self._last_error = f"notifier {label}: {type(exc).__name__}: {exc}"
                logger.error(
                    "alert delivery failed through %s (failure %d): %s: %s",
                    label,
                    self.notify_failures,
                    type(exc).__name__,
                    exc,
                )
                self._breadcrumb(alert, f"notifier {label}", exc)

    def _breadcrumb(self, alert: Alert, kind: str, error: BaseException) -> None:
        """Append a failure line locally; best effort by definition."""
        try:
            target = self.failure_log_path
            ensure_dir(target.parent)
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(
                    f"{self._clock().isoformat()} [{alert.severity.value}] "
                    f"{alert.title}: {alert.message} | {kind} failed: "
                    f"{type(error).__name__}: {error}\n"
                )
        except Exception as exc:
            self.breadcrumb_failures += 1
            logger.error("alert failure log could not be written: %s", exc)

    def suppress(self, alert_type: AlertType, seconds: float) -> None:
        """Drop alerts of ``alert_type`` for the next ``seconds`` seconds."""
        self._suppressed[alert_type] = self._clock() + timedelta(seconds=seconds)
        logger.info("suppressing %s alerts for %.0f seconds", alert_type.value, seconds)

    async def acknowledge(
        self, alert_id: str, acknowledged_by: Optional[str] = None
    ) -> bool:
        """Mark an alert acknowledged in the ring and in the store.

        True when the ring or the store knew the alert. A store failure is
        counted and logged; the ring result still stands.
        """
        found = False
        for alert in self._ring:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = self._clock()
                found = True
                break
        if self._store is not None:
            try:
                if await self._store.acknowledge(alert_id):
                    found = True
            except Exception as exc:
                self.persist_failures += 1
                self._last_error = f"store: {type(exc).__name__}: {exc}"
                logger.error(
                    "alert store acknowledge failed: %s: %s", type(exc).__name__, exc
                )
        return found

    def get_alert_history(self, hours: float = 24) -> List[Alert]:
        """Ring alerts from the last ``hours`` hours, newest first."""
        cutoff = self._clock() - timedelta(hours=hours)
        recent = [a for a in self._ring if a.timestamp > cutoff]
        return sorted(recent, key=lambda a: a.timestamp, reverse=True)

    def get_alerts_by_type(self, alert_type: AlertType) -> List[Alert]:
        """Ring alerts of one type, oldest first."""
        return [a for a in self._ring if a.alert_type == alert_type]

    def get_unacknowledged(
        self, min_severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Unacknowledged ring alerts, newest first, optionally above a floor."""
        pending = [a for a in self._ring if not a.acknowledged]
        if min_severity is not None:
            pending = [a for a in pending if a.severity >= min_severity]
        return sorted(pending, key=lambda a: a.timestamp, reverse=True)

    def get_summary(self) -> Dict[str, Any]:
        """Counts over the last 24 hours plus the failure counters."""
        cutoff = self._clock() - timedelta(hours=24)
        last_day = [a for a in self._ring if a.timestamp > cutoff]
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for alert in last_day:
            by_type[alert.alert_type.value] = by_type.get(alert.alert_type.value, 0) + 1
            by_severity[alert.severity.value] = (
                by_severity.get(alert.severity.value, 0) + 1
            )
        return {
            "total_24h": len(last_day),
            "by_type": by_type,
            "by_severity": by_severity,
            "unacknowledged": sum(1 for a in self._ring if not a.acknowledged),
            "total_stored": len(self._ring),
            "notify_failures": self.notify_failures,
            "persist_failures": self.persist_failures,
            "suppressed": self.suppressed_count,
        }

    def status(self) -> Dict[str, Any]:
        """``backend`` (``store`` or ``memory``), ``available`` (False while the
        latest alert had a failed store write or delivery), ``last_error`` and
        the counters."""
        return {
            "backend": "store" if self._store is not None else "memory",
            "available": self._last_error is None,
            "last_error": self._last_error,
            "notifiers": [type(n).__name__ for n in self._notifiers],
            "ring_size": len(self._ring),
            "notify_failures": self.notify_failures,
            "persist_failures": self.persist_failures,
            "suppressed": self.suppressed_count,
        }


_alert_manager: Optional[AlertManager] = None


def get_alert_manager(settings: Optional[Settings] = None) -> AlertManager:
    """The process-wide manager, built from Settings on first use."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager.from_settings(settings)
    return _alert_manager


def reset_alert_manager() -> None:
    """Forget the process-wide manager (tests and reconfiguration)."""
    global _alert_manager
    _alert_manager = None
