"""
Longitudinal study manager
==========================

Records repeated assessments of one system over time and reports on them.
``run_session`` calls the supplied ``assess`` callable (an async function
returning a ``ConsciousnessReport``), optionally runs the behavioral battery
(``ExperimentSuite``) and an optional snapshot hook, and appends one
``SessionRecord`` line to ``<LONGITUDINAL_OUTPUT_DIR>/<study_id>/sessions.jsonl``
(created on the first write). There is no scheduler: the caller decides when to
run a session.

What a record holds: the report status (``measured``, ``partial``,
``insufficient_data``, or ``error`` when the assessment raised), coverage,
the overall score (None unless measured), theory scores, the per-indicator
status and score read from the report itself (indicator names are never
hard-coded here), a battery summary and any error text. A failed assessment
records its error and no scores. A manager without a suite or snapshot hook
stores only the assessment; nothing is simulated.

Reports (``daily_report``, ``weekly_report``, ``final_report``) read the file
and aggregate measured sessions only (status ``measured`` or ``partial`` with
an overall score); insufficient and errored sessions are counted and left out
of means and trends. Days are UTC days. Trends use
``mtc.research.statistics.trend`` with time in days. The final report flags a
significant positive trend (p below ``statistics.ALPHA``, positive slope, at
least ``MIN_SESSIONS_FOR_TREND_FLAG`` measured sessions) and describes it as a
statistical trend in the overall indicator score and nothing more; it says
nothing about experience. ``LongitudinalStudy`` in ``mtc.assessment`` is a
separate tool and is left as it is.

Config: ``Settings.LONGITUDINAL_OUTPUT_DIR`` (default output directory) and the
injectable ``clock``. Session files hold battery replies and snapshot
references and are private data.
"""

import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from mtc.core.config import get_settings
from mtc.research import statistics as stats

logger = logging.getLogger(__name__)

MIN_SESSIONS_FOR_TREND_FLAG = 5
MEASURED_STATUSES = ("measured", "partial")

_STUDY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass
class SessionRecord:
    """One recorded session; unknown quantities are None."""

    study_id: str
    session_number: int
    label: str
    timestamp: datetime
    status: str
    coverage: Optional[float] = None
    overall_score: Optional[float] = None
    theory_scores: Dict[str, float] = field(default_factory=dict)
    indicators: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    insufficient_indicators: List[str] = field(default_factory=list)
    battery: Optional[Dict[str, Any]] = None
    battery_error: Optional[str] = None
    snapshot: Optional[Any] = None
    snapshot_error: Optional[str] = None
    error: Optional[str] = None
    duration_s: float = 0.0

    @property
    def is_measured(self) -> bool:
        return self.status in MEASURED_STATUSES and self.overall_score is not None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionRecord":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        known["timestamp"] = _aware(datetime.fromisoformat(known["timestamp"]))
        return cls(**known)


def _aware(value: datetime) -> datetime:
    """Naive datetimes count as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _finite(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _json_safe(value: Any) -> Any:
    """Plain JSON values; anything else becomes its string form."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


class LongitudinalStudyManager:
    """Appends session records and aggregates them into reports."""

    def __init__(
        self,
        study_id: str,
        assess: Callable[[], Awaitable[Any]],
        suite: Optional[Any] = None,
        snapshot: Optional[Callable[[str], Awaitable[Any]]] = None,
        output_dir: Optional[Path] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if not _STUDY_ID.fullmatch(study_id or ""):
            raise ValueError("study_id must be a plain name (letters, digits, _ . -)")
        self.study_id = study_id
        self._assess = assess
        self._suite = suite
        self._snapshot = snapshot
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    @property
    def study_dir(self) -> Path:
        base = self._output_dir
        if base is None:
            base = Path(get_settings().LONGITUDINAL_OUTPUT_DIR)
        return base / self.study_id

    @property
    def sessions_path(self) -> Path:
        return self.study_dir / "sessions.jsonl"

    def sessions(self) -> List[SessionRecord]:
        """Every recorded session, oldest first; malformed lines are skipped."""
        path = self.sessions_path
        if not path.exists():
            return []
        records: List[SessionRecord] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(SessionRecord.from_dict(json.loads(line)))
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("skipping malformed session line %d: %s", number, exc)
        return records

    def _append(self, record: SessionRecord) -> None:
        self.study_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), default=str, allow_nan=False)
        with open(self.sessions_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def run_session(self, label: str = "scheduled") -> SessionRecord:
        """Run one session and append its record."""
        started = _aware(self._clock())
        record = SessionRecord(
            study_id=self.study_id,
            session_number=len(self.sessions()) + 1,
            label=label,
            timestamp=started,
            status="error",
        )
        try:
            report = await self._assess()
        except Exception as exc:
            logger.warning("assessment failed in session %s: %s", label, exc)
            record.error = f"{type(exc).__name__}: {exc}"
        else:
            self._fill_from_report(record, report)

        if self._suite is not None:
            try:
                record.battery = self._battery_summary(await self._suite.run_battery())
            except Exception as exc:
                logger.warning("battery failed in session %s: %s", label, exc)
                record.battery_error = f"{type(exc).__name__}: {exc}"
        if self._snapshot is not None:
            try:
                record.snapshot = _json_safe(await self._snapshot(label))
            except Exception as exc:
                logger.warning("snapshot hook failed in session %s: %s", label, exc)
                record.snapshot_error = f"{type(exc).__name__}: {exc}"

        record.duration_s = max(0.0, (_aware(self._clock()) - started).total_seconds())
        self._append(record)
        return record

    @staticmethod
    def _fill_from_report(record: SessionRecord, report: Any) -> None:
        record.status = str(getattr(report, "status", "insufficient_data"))
        record.coverage = _finite(getattr(report, "coverage", None))
        record.overall_score = _finite(getattr(report, "overall_score", None))
        record.theory_scores = {
            str(k): float(v)
            for k, v in (getattr(report, "theory_scores", None) or {}).items()
            if _finite(v) is not None
        }
        record.insufficient_indicators = list(
            getattr(report, "insufficient_indicators", None) or []
        )
        for name, result in (getattr(report, "indicator_results", None) or {}).items():
            status = str(getattr(result, "status", "measured"))
            score = _finite(getattr(result, "score", None))
            record.indicators[str(name)] = {
                "status": status,
                "score": score if status == "measured" else None,
            }

    @staticmethod
    def _battery_summary(battery: Any) -> Dict[str, Any]:
        return {
            "status": battery.status,
            "overall_score": battery.overall_score,
            "experiments": {
                name: {
                    "status": s.status,
                    "n_completed": s.n_completed,
                    "n_errors": s.n_errors,
                    "mean_score": s.mean_score,
                }
                for name, s in battery.summaries.items()
            },
        }

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_day(record: SessionRecord) -> date:
        return record.timestamp.astimezone(timezone.utc).date()

    def _aggregate(self, records: List[SessionRecord]) -> Dict[str, Any]:
        measured = [r for r in records if r.is_measured]
        scores = [r.overall_score for r in measured]
        theory: Dict[str, List[float]] = {}
        experiments: Dict[str, List[float]] = {}
        for r in measured:
            for name, value in r.theory_scores.items():
                theory.setdefault(name, []).append(value)
        for r in records:
            for name, exp in ((r.battery or {}).get("experiments") or {}).items():
                if (
                    exp.get("status") == "measured"
                    and exp.get("mean_score") is not None
                ):
                    experiments.setdefault(name, []).append(float(exp["mean_score"]))
        t0 = measured[0].timestamp if measured else None
        trend = None
        if t0 is not None:
            days = [(r.timestamp - t0).total_seconds() / 86400.0 for r in measured]
            trend = stats.trend(scores, days)
        return {
            "n_sessions": len(records),
            "n_measured": len(measured),
            "n_insufficient": sum(
                1 for r in records if r.status == "insufficient_data"
            ),
            "n_errors": sum(1 for r in records if r.status == "error"),
            "mean_overall": _mean(scores),
            "min_overall": min(scores) if scores else None,
            "max_overall": max(scores) if scores else None,
            "mean_coverage": _mean(
                [r.coverage for r in measured if r.coverage is not None]
            ),
            "trajectory": scores,
            "theory_means": {k: _mean(v) for k, v in theory.items()},
            "experiment_means": {k: _mean(v) for k, v in experiments.items()},
            "trend": (
                None
                if trend is None
                else {
                    **asdict(trend),
                    "unit": "score per day",
                    "significant": (
                        None if trend.p_value is None else trend.p_value < stats.ALPHA
                    ),
                }
            ),
        }

    @staticmethod
    def _status_of(aggregate: Dict[str, Any]) -> str:
        if aggregate["n_sessions"] == 0:
            return "no_sessions"
        return "measured" if aggregate["n_measured"] else "insufficient_data"

    def daily_report(self, day: date) -> Dict[str, Any]:
        """Aggregate of the sessions on one UTC day."""
        records = [r for r in self.sessions() if self._utc_day(r) == day]
        report = self._aggregate(records)
        report["day"] = day.isoformat()
        report["status"] = self._status_of(report)
        previous = [
            r
            for r in self.sessions()
            if self._utc_day(r) == day - timedelta(days=1) and r.is_measured
        ]
        prev_mean = _mean([r.overall_score for r in previous])
        report["change_from_previous_day"] = (
            None
            if prev_mean is None or report["mean_overall"] is None
            else report["mean_overall"] - prev_mean
        )
        return report

    def weekly_report(self, week_start: date) -> Dict[str, Any]:
        """Aggregate of the seven UTC days starting at ``week_start``."""
        end = week_start + timedelta(days=7)
        records = [r for r in self.sessions() if week_start <= self._utc_day(r) < end]
        report = self._aggregate(records)
        report["week_start"] = week_start.isoformat()
        report["week_end"] = (end - timedelta(days=1)).isoformat()
        report["status"] = self._status_of(report)
        return report

    def final_report(self) -> Dict[str, Any]:
        """Aggregate of the whole study with per-indicator means."""
        records = self.sessions()
        report = self._aggregate(records)
        report["study_id"] = self.study_id
        report["status"] = self._status_of(report)
        measured = [r for r in records if r.is_measured]
        if measured:
            report["first_overall"] = measured[0].overall_score
            report["last_overall"] = measured[-1].overall_score
            report["change"] = measured[-1].overall_score - measured[0].overall_score
        indicator_scores: Dict[str, List[float]] = {}
        for r in measured:
            for name, info in r.indicators.items():
                if info.get("status") == "measured" and info.get("score") is not None:
                    indicator_scores.setdefault(name, []).append(float(info["score"]))
        report["indicators"] = {
            name: {"n_measured": len(v), "mean_score": _mean(v)}
            for name, v in indicator_scores.items()
        }
        trend = report["trend"]
        flagged = bool(
            trend
            and trend["significant"]
            and trend["slope"] > 0
            and report["n_measured"] >= MIN_SESSIONS_FOR_TREND_FLAG
        )
        report["significant_positive_trend"] = flagged
        report["interpretation"] = (
            (
                "A statistically significant positive trend in the overall "
                "indicator score across sessions."
                if flagged
                else "No significant positive trend in the overall indicator score."
            )
            + " This describes the recorded series only and says nothing about experience."
        )
        return report
