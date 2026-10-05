"""
Consciousness Snapshot Service
==============================

Stores periodic and event-driven snapshots of the 30-indicator assessment
for longitudinal study, through a ``SnapshotStore`` (``mtc.memory.protocols``;
None means ``InMemorySnapshotStore``), with a JSON backup of each report.

Each ``SnapshotRecord`` carries the report's ``status`` and ``coverage``. The
overall score of a report whose status is ``insufficient_data`` is stored as
null: such a snapshot contributes no measured value, so it never crosses a
threshold, never raises an event and is left out of trend averages (it is
still counted). Events compare a measured snapshot with the last measured
values: the overall score crossing ``crossing_threshold`` in either
direction, a theory score changing by more than ``theory_change_threshold``
and Phi shifting by more than ``phi_shift_threshold``, all from
``SnapshotConfig``.

``maybe_snapshot`` takes a snapshot when ``periodic_interval_s`` has passed
since the last one or when an event is detected (trigger ``periodic``,
``threshold_crossing`` or ``phi_change``). An optional ``developmental_hook``
returns a developmental assessment that lands in the record's metadata under
``developmental``; the service itself reads no private directory and no
message store.

Backups go to ``<backup_dir>/snapshot_<trigger>_<timestamp>.json``, by default
``<RESULTS_DIR>/consciousness_snapshots`` with ``RESULTS_DIR`` read from
settings at write time; the folder is created on the first write and a failed
write is logged, never raised. Timestamps are timezone-aware UTC from the
injectable clock. The service keeps only the last measured values in memory;
``initialize`` reloads them from the store.
"""

import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
)

from mtc.core.config import get_settings
from mtc.memory.inmemory import InMemorySnapshotStore
from mtc.memory.protocols import SnapshotEvent, SnapshotRecord, SnapshotStore

if TYPE_CHECKING:  # pragma: no cover
    from mtc.assessment.assessment import ConsciousnessReport

logger = logging.getLogger(__name__)

INSUFFICIENT_DATA = "insufficient_data"
BACKUP_SUBDIR = "consciousness_snapshots"

# Records fetched when a query spans a time range rather than a page.
_RANGE_LIMIT = 100000


@dataclass
class SnapshotConfig:
    """Snapshot cadence and event thresholds."""

    periodic_interval_s: float = 7200.0
    crossing_threshold: float = 0.5
    theory_change_threshold: float = 0.1
    phi_shift_threshold: float = 1.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(moment: datetime) -> datetime:
    """The instant in UTC; a naive datetime is read as UTC."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


class SnapshotService:
    """Takes, stores and queries consciousness snapshots.

    Args:
        store: The snapshot store; None means a new ``InMemorySnapshotStore``.
        assess: Async callable returning a ``ConsciousnessReport``; used when
            a snapshot is requested without a report.
        config: Cadence and thresholds.
        backup_dir: Backup folder; None derives it from ``RESULTS_DIR``.
        developmental_hook: Async callable returning a developmental
            assessment dict (or None) stored in each record's metadata.
        clock: Returns the current aware UTC time.
    """

    def __init__(
        self,
        store: Optional[SnapshotStore] = None,
        assess: Optional[Callable[[], Awaitable["ConsciousnessReport"]]] = None,
        config: Optional[SnapshotConfig] = None,
        backup_dir: Optional[Path] = None,
        developmental_hook: Optional[
            Callable[[], Awaitable[Optional[Dict[str, Any]]]]
        ] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.store: SnapshotStore = (
            store if store is not None else InMemorySnapshotStore()
        )
        self._assess = assess
        self.config = config or SnapshotConfig()
        self._backup_dir = Path(backup_dir) if backup_dir is not None else None
        self._developmental_hook = developmental_hook
        self._clock = clock or _utcnow
        self._last_snapshot_at: Optional[datetime] = None
        self._last_overall: Optional[float] = None
        self._last_theories: Dict[str, float] = {}
        self._last_phi: Optional[float] = None

    def _now(self) -> datetime:
        return _aware(self._clock())

    @property
    def backup_dir(self) -> Path:
        """The explicit backup folder, or the settings folder resolved now."""
        if self._backup_dir is not None:
            return self._backup_dir
        return Path(get_settings().RESULTS_DIR) / BACKUP_SUBDIR

    async def initialize(self) -> None:
        """Reload the last snapshot time and measured values from the store."""
        for record in await self.store.snapshots(limit=_RANGE_LIMIT):
            if self._last_snapshot_at is None:
                self._last_snapshot_at = _aware(record.timestamp)
            if record.status != INSUFFICIENT_DATA:
                self._remember(record)
                break

    # ======== RECORDS ========

    def _record_from_report(
        self, trigger: str, report: Any, timestamp: datetime
    ) -> SnapshotRecord:
        exported = report.to_dict() if hasattr(report, "to_dict") else {}
        status = str(getattr(report, "status", INSUFFICIENT_DATA))
        overall = None
        if status != INSUFFICIENT_DATA:
            overall = _finite(getattr(report, "overall_score", None))
        phi = _finite(getattr(getattr(report, "phi_measurement", None), "phi", None))
        theories = {
            str(name): score
            for name, value in (getattr(report, "theory_scores", None) or {}).items()
            if (score := _finite(value)) is not None
        }
        indicators = {
            name: {
                key: entry.get(key)
                for key in (
                    "theory",
                    "status",
                    "score",
                    "confidence",
                    "passes_threshold",
                )
            }
            for name, entry in (exported.get("indicator_results") or {}).items()
            if isinstance(entry, dict)
        }
        phi_measurement = getattr(report, "phi_measurement", None)
        metadata: Dict[str, Any] = {
            "session_id": getattr(report, "session_id", None),
            "total_indicators": getattr(report, "total_indicators", None),
            "insufficient_indicators": list(
                getattr(report, "insufficient_indicators", None) or []
            ),
            "confidence": _finite(getattr(report, "confidence", None)),
            "phi_method": getattr(phi_measurement, "method", None),
        }
        return SnapshotRecord(
            timestamp=timestamp,
            trigger=trigger,
            overall_score=overall,
            passing_count=int(getattr(report, "passing_count", 0) or 0),
            phi_value=phi,
            theory_scores=theories,
            indicator_results=indicators,
            processing_time_ms=_finite(getattr(report, "processing_time_ms", None)),
            status=status,
            coverage=_finite(getattr(report, "coverage", None)),
            metadata=metadata,
        )

    def _remember(self, record: SnapshotRecord) -> None:
        """Keep a measured snapshot's values as the comparison baseline."""
        if record.overall_score is not None:
            self._last_overall = record.overall_score
        self._last_theories.update(
            {k: v for k, v in record.theory_scores.items() if _finite(v) is not None}
        )
        if record.phi_value is not None:
            self._last_phi = record.phi_value

    def _detect_events(self, record: SnapshotRecord) -> List[Dict[str, Any]]:
        """Changes from the last measured values; none for an insufficient record."""
        if record.status == INSUFFICIENT_DATA:
            return []
        cfg = self.config
        events: List[Dict[str, Any]] = []
        previous, current = self._last_overall, record.overall_score
        if previous is not None and current is not None:
            threshold = cfg.crossing_threshold
            if previous < threshold <= current:
                events.append(
                    {
                        "event_type": "score_crossed_up",
                        "description": (
                            f"Overall score rose across {threshold} "
                            f"({previous:.3f} to {current:.3f})"
                        ),
                        "old_value": previous,
                        "new_value": current,
                    }
                )
            elif previous >= threshold > current:
                events.append(
                    {
                        "event_type": "score_crossed_down",
                        "description": (
                            f"Overall score fell below {threshold} "
                            f"({previous:.3f} to {current:.3f})"
                        ),
                        "old_value": previous,
                        "new_value": current,
                    }
                )
        for theory, score in record.theory_scores.items():
            before = self._last_theories.get(theory)
            if before is not None and abs(score - before) > cfg.theory_change_threshold:
                direction = "rose" if score > before else "fell"
                events.append(
                    {
                        "event_type": "theory_change",
                        "description": (
                            f"{theory} score {direction} ({before:.3f} to {score:.3f})"
                        ),
                        "old_value": before,
                        "new_value": score,
                    }
                )
        if self._last_phi is not None and record.phi_value is not None:
            if abs(record.phi_value - self._last_phi) > cfg.phi_shift_threshold:
                events.append(
                    {
                        "event_type": "phi_shift",
                        "description": (
                            f"Phi shifted ({self._last_phi:.2f} to "
                            f"{record.phi_value:.2f})"
                        ),
                        "old_value": self._last_phi,
                        "new_value": record.phi_value,
                    }
                )
        return events

    async def _developmental_block(self) -> Optional[Dict[str, Any]]:
        if self._developmental_hook is None:
            return None
        try:
            block = await self._developmental_hook()
        except Exception as e:
            logger.warning(f"Developmental hook failed: {e}")
            return None
        return dict(block) if isinstance(block, dict) else None

    async def _resolve_report(self, report: Optional[Any]) -> Any:
        if report is not None:
            return report
        if self._assess is None:
            raise ValueError("a snapshot needs a report or an assess callable")
        return await self._assess()

    # ======== TAKING SNAPSHOTS ========

    async def take_snapshot(
        self, trigger: str, report: Optional["ConsciousnessReport"] = None
    ) -> SnapshotRecord:
        """Store a snapshot of the report (assessed now when None) and its events.

        Raises ValueError without a report and without an assess callable;
        store errors propagate. The JSON backup never raises.
        """
        return await self._take(trigger, report, self._now())

    async def _take(
        self, trigger: str, report: Optional[Any], timestamp: datetime
    ) -> SnapshotRecord:
        report = await self._resolve_report(report)
        record = self._record_from_report(trigger, report, timestamp)
        developmental = await self._developmental_block()
        if developmental is not None:
            record.metadata["developmental"] = developmental
        events = self._detect_events(record)
        record.id = await self.store.add_snapshot(record)
        for event in events:
            await self.store.add_event(
                SnapshotEvent(
                    timestamp=timestamp,
                    event_type=event["event_type"],
                    description=event["description"],
                    snapshot_id=record.id,
                    old_value=event["old_value"],
                    new_value=event["new_value"],
                )
            )
        if record.status != INSUFFICIENT_DATA:
            self._remember(record)
        self._last_snapshot_at = timestamp
        self._write_backup(record, report)
        score = (
            "insufficient_data"
            if record.overall_score is None
            else f"{record.overall_score:.3f}"
        )
        logger.info(
            f"Snapshot {record.id} stored (trigger={trigger}, score={score}, "
            f"status={record.status}, events={len(events)})"
        )
        return record

    def _write_backup(self, record: SnapshotRecord, report: Any) -> Optional[Path]:
        """Write the full report as JSON next to the record's fields."""
        try:
            folder = self.backup_dir
            folder.mkdir(parents=True, exist_ok=True)
            label = re.sub(r"[^a-z0-9_]+", "_", record.trigger.lower()).strip("_")
            stamp = record.timestamp.strftime("%Y%m%dT%H%M%S%fZ")
            path = folder / f"snapshot_{label or 'snapshot'}_{stamp}.json"
            if path.exists():
                path = (
                    folder / f"snapshot_{label or 'snapshot'}_{stamp}_{record.id}.json"
                )
            exported = report.to_dict() if hasattr(report, "to_dict") else {}
            backup = {
                **exported,
                "snapshot_id": record.id,
                "timestamp": record.timestamp.isoformat(),
                "trigger": record.trigger,
                "overall_score": record.overall_score,
                "status": record.status,
                "coverage": record.coverage,
                "metadata": record.metadata,
            }
            path.write_text(json.dumps(backup, indent=2, default=str), encoding="utf-8")
            return path
        except Exception as e:
            logger.warning(f"Snapshot backup failed: {e}")
            return None

    async def maybe_snapshot(
        self,
        now: Optional[datetime] = None,
        *,
        report: Optional["ConsciousnessReport"] = None,
    ) -> Optional[SnapshotRecord]:
        """Snapshot when the interval has passed or an event is detected.

        Without ``report`` the assess callable runs; without either, or when
        the assessment or the store fails, nothing is stored and None is
        returned.
        """
        moment = _aware(now) if now is not None else self._now()
        try:
            if report is None:
                if self._assess is None:
                    return None
                report = await self._assess()
            probe = self._record_from_report("probe", report, moment)
            events = self._detect_events(probe)
            last = self._last_snapshot_at
            periodic = (
                last is None
                or (moment - last).total_seconds() >= self.config.periodic_interval_s
            )
            if events:
                first = events[0]["event_type"]
                trigger = "phi_change" if first == "phi_shift" else "threshold_crossing"
            elif periodic:
                trigger = "periodic"
            else:
                return None
            return await self._take(trigger, report, moment)
        except Exception as e:
            logger.warning(f"Snapshot check failed: {e}")
            return None

    # ======== QUERIES ========

    async def get_history(
        self, hours: float = 24, limit: int = 100
    ) -> List[SnapshotRecord]:
        """Snapshots from the last ``hours``, newest first."""
        since = self._now() - timedelta(hours=hours)
        return await self.store.snapshots(since=since, limit=limit)

    async def get_events(
        self, hours: float = 24, limit: int = 100, event_type: Optional[str] = None
    ) -> List[SnapshotEvent]:
        """Events from the last ``hours``, newest first, optionally of one type."""
        since = self._now() - timedelta(hours=hours)
        return await self.store.events(since=since, limit=limit, event_type=event_type)

    async def get_trends(self, hours: float = 168) -> Dict[str, Any]:
        """Per-UTC-day aggregates over the last ``hours``, oldest day first.

        Each day reports ``snapshot_count`` (every snapshot) and
        ``measured_count``; the score, Phi and theory averages use measured
        snapshots only and are None when the day has none.
        """
        records = await self.store.snapshots(
            since=self._now() - timedelta(hours=hours), limit=_RANGE_LIMIT
        )
        by_day: Dict[str, List[SnapshotRecord]] = defaultdict(list)
        for record in records:
            by_day[_aware(record.timestamp).date().isoformat()].append(record)
        days = []
        for day in sorted(by_day):
            items = by_day[day]
            measured = [r for r in items if r.status != INSUFFICIENT_DATA]
            scores = [r.overall_score for r in measured if r.overall_score is not None]
            phis = [r.phi_value for r in measured if r.phi_value is not None]
            theories: Dict[str, List[float]] = defaultdict(list)
            for record in measured:
                for name, value in record.theory_scores.items():
                    score = _finite(value)
                    if score is not None:
                        theories[name].append(score)
            days.append(
                {
                    "day": day,
                    "snapshot_count": len(items),
                    "measured_count": len(measured),
                    "avg_score": _mean(scores),
                    "min_score": min(scores) if scores else None,
                    "max_score": max(scores) if scores else None,
                    "avg_phi": _mean(phis),
                    "theory_scores": {k: _mean(v) for k, v in sorted(theories.items())},
                }
            )
        return {"hours": hours, "days": days}

    async def get_current(self) -> Dict[str, Any]:
        """The latest snapshot and its change over the last 24 hours."""
        return {
            "latest": await self.store.latest_snapshot(),
            "delta_24h": await self.get_delta(24),
        }

    async def get_delta(
        self, hours: float = 24
    ) -> Optional[Dict[str, Optional[float]]]:
        """Latest snapshot minus the newest one taken more than ``hours`` ago.

        None without both snapshots. Each value is None when either side
        lacks it (an insufficient overall, an unmeasured Phi).
        """
        latest = await self.store.latest_snapshot()
        if latest is None:
            return None
        older = await self.store.snapshot_before(self._now() - timedelta(hours=hours))
        if older is None or (older.id is not None and older.id == latest.id):
            return None

        def diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
            a, b = _finite(a), _finite(b)
            return a - b if a is not None and b is not None else None

        return {
            "overall_score": diff(latest.overall_score, older.overall_score),
            "phi_value": diff(latest.phi_value, older.phi_value),
            "passing_count": diff(latest.passing_count, older.passing_count),
            "coverage": diff(latest.coverage, older.coverage),
        }
