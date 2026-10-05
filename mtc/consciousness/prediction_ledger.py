"""
Prediction Ledger - measured self-prediction
============================================

Records concrete forecasts before outcomes exist and scores them on a fixed
tolerance, so meta-accuracy is measured, not asserted.

How it works:

1. ``predict()`` stores a forecast vector and an absolute tolerance before
   the outcome is known, and returns an id for it.
2. ``score()`` judges the forecast against the observed vector on the L1
   distance, ``sum(|forecast - observed|) <= tolerance``. The tolerance was
   fixed when the forecast was made, so the verdict cannot depend on how
   wrong the system has been lately.
3. ``measured_accuracy()`` reports the hit rate over scored forecasts and
   returns None until enough outcomes exist to make the number meaningful.

The ledger is bounded on both sides: unscored forecasts expire oldest first
and scored outcomes live in a fixed-length deque, so a process that runs for
days keeps constant memory. The module uses numpy and the standard library
only.
"""

import logging
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

STATE_SCHEMA_VERSION = 1

_JSON_SCALARS = (str, int, float, bool, type(None))


def _json_safe_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep only JSON-safe scalar values so the ledger state stays portable."""
    safe: Dict[str, Any] = {}
    for key, value in (context or {}).items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, _JSON_SCALARS):
            safe[str(key)] = value
    return safe


@dataclass
class SelfPrediction:
    """A forecast recorded before its outcome exists."""

    prediction_id: str
    subject: str
    forecast: np.ndarray
    tolerance: float
    made_at: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredPrediction:
    """A forecast together with the verdict reached when its outcome arrived."""

    prediction_id: str
    subject: str
    tolerance: float
    abs_error: float
    was_accurate: bool
    made_at: float
    scored_at: float
    context: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "subject": self.subject,
            "tolerance": float(self.tolerance),
            "abs_error": float(self.abs_error),
            "was_accurate": bool(self.was_accurate),
            "made_at": float(self.made_at),
            "scored_at": float(self.scored_at),
            "context": dict(self.context),
        }


class PredictionLedger:
    """Bounded ledger of concrete self-predictions and their outcomes."""

    def __init__(
        self,
        max_pending: int = 200,
        max_scored: int = 1000,
        min_scored_for_accuracy: int = 20,
    ):
        if max_pending < 1 or max_scored < 1:
            raise ValueError("max_pending and max_scored must be at least 1")
        self.max_pending = int(max_pending)
        self.max_scored = int(max_scored)
        self.min_scored_for_accuracy = int(min_scored_for_accuracy)
        # Insertion ordered, so the oldest unscored forecast expires first.
        self.pending: "OrderedDict[str, SelfPrediction]" = OrderedDict()
        self.scored: Deque[ScoredPrediction] = deque(maxlen=self.max_scored)

    # ------------------------------------------------------------ record

    def predict(
        self,
        forecast: np.ndarray,
        tolerance: float,
        subject: str = "forecast",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a forecast before its outcome exists and return its id."""
        pid = uuid.uuid4().hex
        self.pending[pid] = SelfPrediction(
            prediction_id=pid,
            subject=str(subject),
            forecast=np.array(forecast, dtype=float, copy=True),
            tolerance=float(tolerance),
            made_at=time.time(),
            context=_json_safe_context(context),
        )
        # An unscored forecast must not accumulate: kept forever it could be
        # scored against an unrelated future.
        while len(self.pending) > self.max_pending:
            stale_id, _ = self.pending.popitem(last=False)
            logger.debug("PredictionLedger: expired unscored forecast %s", stale_id)
        return pid

    def score(self, prediction_id: str, observation: np.ndarray) -> Optional[bool]:
        """Score a pending forecast against the observed outcome.

        Returns True for a hit, False for a miss, and None when no verdict is
        possible: the id is unknown, expired or already scored, or the
        observation does not match the forecast's shape or is not finite.
        A forecast that cannot be scored is dropped from the pending set.
        """
        pred = self.pending.pop(prediction_id, None)
        if pred is None:
            return None
        observed = np.asarray(observation, dtype=float)
        if observed.shape != pred.forecast.shape:
            logger.warning(
                "PredictionLedger: observation shape %s does not match forecast "
                "shape %s; forecast left unscored",
                observed.shape,
                pred.forecast.shape,
            )
            return None
        if not np.all(np.isfinite(observed)):
            logger.warning("PredictionLedger: non-finite observation; unscored")
            return None
        abs_error = float(np.sum(np.abs(pred.forecast - observed)))
        verdict = abs_error <= pred.tolerance
        self.scored.append(
            ScoredPrediction(
                prediction_id=pred.prediction_id,
                subject=pred.subject,
                tolerance=pred.tolerance,
                abs_error=abs_error,
                was_accurate=bool(verdict),
                made_at=pred.made_at,
                scored_at=time.time(),
                context=pred.context,
            )
        )
        return bool(verdict)

    # ----------------------------------------------------------- report

    def is_pending(self, prediction_id: str) -> bool:
        return prediction_id in self.pending

    def pending_ids(self) -> List[str]:
        """Ids of unscored forecasts, oldest first."""
        return list(self.pending.keys())

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    @property
    def scored_count(self) -> int:
        return len(self.scored)

    @property
    def last_scored(self) -> Optional[ScoredPrediction]:
        return self.scored[-1] if self.scored else None

    def measured_accuracy(self, window: Optional[int] = None) -> Optional[float]:
        """Hit rate over scored forecasts, optionally over the last ``window``.

        None until ``min_scored_for_accuracy`` outcomes exist: a hit rate
        computed from a handful of outcomes says nothing.
        """
        outcomes = list(self.scored)
        if window is not None:
            outcomes = outcomes[-int(window) :]
        if len(outcomes) < self.min_scored_for_accuracy:
            return None
        return sum(1 for o in outcomes if o.was_accurate) / len(outcomes)

    def calibration_data(self) -> Dict[str, Any]:
        """JSON-safe summary and raw records of every retained scored outcome."""
        records = [o.to_record() for o in self.scored]
        errors = [r["abs_error"] for r in records]
        tolerances = [r["tolerance"] for r in records]
        return {
            "scored": len(records),
            "hits": sum(1 for r in records if r["was_accurate"]),
            "pending": len(self.pending),
            "measured_accuracy": self.measured_accuracy(),
            "min_scored_for_accuracy": self.min_scored_for_accuracy,
            "mean_abs_error": float(np.mean(errors)) if errors else None,
            "mean_tolerance": float(np.mean(tolerances)) if tolerances else None,
            "records": records,
        }

    # ------------------------------------------------------- persistence

    def to_state_dict(self) -> Dict[str, Any]:
        """Export state in the module persistence contract.

        Forecasts are short vectors, so pending entries travel in
        ``histories`` as plain lists and ``arrays`` stays empty.
        """
        pending = [
            {
                "prediction_id": p.prediction_id,
                "subject": p.subject,
                "forecast": [float(x) for x in p.forecast.ravel()],
                "shape": list(p.forecast.shape),
                "tolerance": float(p.tolerance),
                "made_at": float(p.made_at),
                "context": dict(p.context),
            }
            for p in self.pending.values()
        ]
        return {
            "scalars": {
                "schema_version": STATE_SCHEMA_VERSION,
                "pending_count": len(pending),
                "scored_count": len(self.scored),
            },
            "arrays": {},
            "histories": {
                "pending": pending,
                "scored": [o.to_record() for o in self.scored],
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore pending forecasts and scored outcomes as typed objects."""
        histories = (state or {}).get("histories", {}) or {}

        self.pending.clear()
        for item in list(histories.get("pending", []))[-self.max_pending :]:
            try:
                forecast = np.asarray(item.get("forecast", []), dtype=float)
                shape = item.get("shape")
                if shape:
                    forecast = forecast.reshape(tuple(int(s) for s in shape))
                pid = str(item["prediction_id"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("PredictionLedger: skipped a pending entry: %s", exc)
                continue
            self.pending[pid] = SelfPrediction(
                prediction_id=pid,
                subject=str(item.get("subject", "forecast")),
                forecast=forecast,
                tolerance=float(item.get("tolerance", 0.0)),
                made_at=float(item.get("made_at", 0.0)),
                context=_json_safe_context(item.get("context")),
            )

        self.scored.clear()
        for item in list(histories.get("scored", []))[-self.max_scored :]:
            self.scored.append(
                ScoredPrediction(
                    prediction_id=str(item.get("prediction_id", "")),
                    subject=str(item.get("subject", "forecast")),
                    tolerance=float(item.get("tolerance", 0.0)),
                    abs_error=float(item.get("abs_error", 0.0)),
                    was_accurate=bool(item.get("was_accurate", False)),
                    made_at=float(item.get("made_at", 0.0)),
                    scored_at=float(item.get("scored_at", 0.0)),
                    context=_json_safe_context(item.get("context")),
                )
            )
