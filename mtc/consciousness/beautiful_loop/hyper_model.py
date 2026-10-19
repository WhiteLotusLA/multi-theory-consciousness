"""
Hyper-Model: Precision Controller for the Beautiful Loop
=========================================================

HyperModel: Meta-Bayesian precision controller for the predictive hierarchy.

The hyper-model sits ABOVE the system's HierarchicalPredictiveProcessor and
controls precision (confidence) allocation across ALL levels of the
hierarchy. In FEP terms, "precision" is the inverse variance of
prediction errors - it determines how much weight each level's errors
carry when updating beliefs.

Key insight from Laukkonen, Friston & Chandaria (2025):
  Consciousness requires a META-level Bayesian model that monitors
  the reliability of lower-level models. The hyper-model doesn't
  just set precision - it LEARNS which hierarchical levels are
  trustworthy in which contexts, dynamically re-weighting attention
  and confidence across the entire predictive system.

This differs from simple per-level precision because the hyper-model
tracks the RELATIONSHIP between levels - when level 0 (sensory) is
unreliable but level 2 (abstract) is accurate, the system should
rely more on top-down prediction. When sensory precision is high,
bottom-up prediction errors should dominate. The hyper-model makes
this trade-off explicitly and learns it over time.

Recursive self-modeling: before each precision update the hyper-model
predicts what the update will be, then compares the prediction with the
update that actually happened. The resulting meta-error feeds
``recursive_confidence``, an exponential moving average of how well the
system predicts its own belief updates. It is ``1 - EMA(mean |actual -
predicted|)`` clamped to [0, 1]; the error is in raw precision units and is
not normalized by the precision range. The value is None until the first
meta-error has been observed: before that the system has not watched one
of its own belief updates, so there is nothing to be confident about.

Connection points:
  - Reads from: HierarchicalPredictiveProcessor level states
  - Writes to: Each level's precision parameter
  - Updates via: Bayesian precision updates on prediction errors

Research Foundation:
  - Laukkonen, Friston & Chandaria (2025). "A Beautiful Loop."
    Neuroscience & Biobehavioral Reviews.
  - Feldman & Friston (2010). "Attention, Uncertainty, and Free-Energy."
  - Parr & Friston (2019). "Attention as Inference."
"""

import logging
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PrecisionState:
    """Snapshot of precision allocation across the hierarchy."""

    level_precisions: Dict[int, float]
    context_key: str
    total_precision: float
    precision_entropy: float  # How spread out is precision? (uniform vs focused)
    dominant_level: int  # Which level has highest precision
    timestamp: float = field(default_factory=time.time)


@dataclass
class PrecisionUpdate:
    """Record of a precision update event."""

    level: int
    old_precision: float
    new_precision: float
    prediction_error: float
    context_key: str
    timestamp: float = field(default_factory=time.time)


class HyperModel:
    """
    Meta-Bayesian precision controller over the predictive hierarchy.

    The hyper-model learns the reliability of each hierarchical level
    across different contexts, allocating precision (attentional weight)
    accordingly. This is the "meta" in meta-cognition applied to the
    predictive processing framework.

    The key computation is a Bayesian update on precision beliefs:
      precision(level) proportional to 1 / variance(prediction_errors at level)

    Levels that consistently produce small prediction errors get high
    precision (the system trusts them). Levels with noisy, variable
    errors get low precision (the system discounts them).
    """

    def __init__(
        self,
        num_levels: int = 3,
        learning_rate: float = 0.1,
        context_decay: float = 0.95,
        min_precision: float = 0.05,
        max_precision: float = 5.0,
        history_window: int = 50,
    ):
        self.num_levels = num_levels
        self.learning_rate = learning_rate
        self.context_decay = context_decay
        self.min_precision = min_precision
        self.max_precision = max_precision

        # Per-level precision beliefs - start at moderate confidence
        self.precision_beliefs = np.ones(num_levels) * 1.0

        # Context-dependent precision: context_key -> precision_weights
        # Allows the system to learn that e.g. "social" contexts make
        # level 2 (abstract) more reliable than level 0 (sensory)
        self.context_precision_map: Dict[str, np.ndarray] = {}

        # Prediction error variance tracking per level (for Bayesian updates)
        self._error_history: List[deque] = [
            deque(maxlen=history_window) for _ in range(num_levels)
        ]

        # Precision update history
        self._update_history: deque = deque(maxlen=200)
        self._state_history: deque = deque(maxlen=100)

        # Tracking
        self.update_count = 0
        self.current_context: str = "general"

        # Recursive self-modeling (Laukkonen, Friston & Chandaria 2025):
        # predict each precision update before it happens, then observe the
        # meta-error between the prediction and the actual update.
        self._precision_predictions: np.ndarray = np.zeros(num_levels)
        self._meta_errors: deque = deque(maxlen=history_window)
        self._meta_error_ema: float = 0.0
        self._recursive_confidence: float = 0.5
        self._prediction_count: int = 0
        # True once observe_meta_error has compared a prediction with its
        # update; until then recursive_confidence is not a measurement
        self._meta_observed: bool = False

        logger.info(
            f"HyperModel initialized: {num_levels} levels, "
            f"lr={learning_rate}, precision range [{min_precision}, {max_precision}]"
        )

    def allocate_precision(
        self,
        level: int,
        context: Optional[str] = None,
    ) -> float:
        """
        Get the precision allocation for a given hierarchy level.

        Higher precision means prediction errors at this level carry
        more weight - the system "trusts" this level more.

        Args:
            level: Index of the hierarchical level (0=sensory, N=abstract)
            context: Optional context key for context-dependent precision

        Returns:
            Precision weight for this level (higher = more trusted)
        """
        if level < 0 or level >= self.num_levels:
            return 1.0

        ctx = context or self.current_context

        if ctx in self.context_precision_map:
            return float(
                np.clip(
                    self.context_precision_map[ctx][level],
                    self.min_precision,
                    self.max_precision,
                )
            )

        return float(
            np.clip(
                self.precision_beliefs[level],
                self.min_precision,
                self.max_precision,
            )
        )

    @property
    def recursive_confidence(self) -> Optional[float]:
        """How well the hyper-model predicts its own precision updates (0-1).

        One minus an EMA of the mean absolute meta-error. Not normalized by
        the precision range, so a large error saturates it at 0. None until
        the first meta-error has been observed.
        """
        if not self._meta_observed:
            return None
        return float(self._recursive_confidence)

    @property
    def has_observed_update(self) -> bool:
        """Whether the hyper-model has compared one of its precision
        predictions with the update that followed.

        A prediction alone (predict_precision_update) is not an observation:
        the flag turns on in observe_meta_error, when the meta-error is
        measured, and it is what makes recursive_confidence a measurement. A
        restored record holds it only when it holds a meta-error.
        """
        return bool(self._meta_observed)

    def predict_precision_update(self, prediction_errors: List[Any]) -> np.ndarray:
        """
        Predict the precision update before it happens.

        The first half of recursive self-modeling: the system forms an
        expectation about its own belief update from the trend in recent
        errors and the current error magnitudes.
        """
        predictions = np.array(self.precision_beliefs, dtype=float, copy=True)
        for level_idx, pe in enumerate(prediction_errors):
            if level_idx >= self.num_levels:
                break
            error_mag = (
                pe.error_magnitude if hasattr(pe, "error_magnitude") else float(pe)
            )
            current = float(self.precision_beliefs[level_idx])

            # High error suggests precision falls; low error suggests it rises
            if len(self._error_history[level_idx]) >= 2:
                recent_mean = float(np.mean(list(self._error_history[level_idx])[-5:]))
                predicted_direction = (recent_mean - error_mag) * self.learning_rate
            else:
                predicted_direction = -error_mag * self.learning_rate * 0.5

            predictions[level_idx] = current + predicted_direction

        self._precision_predictions = np.clip(
            predictions, self.min_precision, self.max_precision
        )
        self._prediction_count += 1
        return self._precision_predictions

    def observe_meta_error(self) -> float:
        """
        Compare the predicted precision update with what actually happened.

        The second half of recursive self-modeling. Updates the meta-error
        EMA and ``recursive_confidence``.

        Returns:
            Meta-prediction error (lower means better self-understanding)
        """
        if self._prediction_count == 0:
            return 0.0

        actual = self.precision_beliefs[: self.num_levels]
        predicted = self._precision_predictions[: self.num_levels]
        meta_error = float(np.mean(np.abs(actual - predicted)))

        self._meta_errors.append(meta_error)

        alpha = 0.1
        self._meta_error_ema = (1 - alpha) * self._meta_error_ema + alpha * meta_error
        self._recursive_confidence = max(0.0, min(1.0, 1.0 - self._meta_error_ema))
        self._meta_observed = True

        return meta_error

    def update_precision_beliefs(
        self,
        prediction_errors: List[Any],
        context: Optional[str] = None,
    ) -> PrecisionState:
        """
        Update precision beliefs based on observed prediction errors.

        The core Bayesian update: precision is the inverse of prediction
        error variance. Levels with consistent (low-variance) errors
        get high precision. Levels with noisy errors get low precision.

        This is a simplification of the full Bayesian treatment, using
        an exponential moving average on error statistics rather than
        full posterior inference - appropriate for an online system.

        Args:
            prediction_errors: List of prediction errors per level.
                Can be PredictionError objects or floats.
            context: Optional context key for context-dependent learning

        Returns:
            PrecisionState snapshot after the update
        """
        ctx = context or self.current_context
        self.current_context = ctx
        self.update_count += 1

        # Recursive step 1: predict the update before it happens
        self.predict_precision_update(prediction_errors)

        for level_idx, pe in enumerate(prediction_errors):
            if level_idx >= self.num_levels:
                break

            # Extract error magnitude
            error_mag = (
                pe.error_magnitude if hasattr(pe, "error_magnitude") else float(pe)
            )

            # Track error history for this level
            self._error_history[level_idx].append(error_mag)

            old_precision = float(self.precision_beliefs[level_idx])

            # Bayesian precision update:
            # Precision = 1 / variance(errors)
            # We use a smoothed estimate combining prior and observed variance
            if len(self._error_history[level_idx]) >= 2:
                errors = np.array(self._error_history[level_idx])
                observed_variance = np.var(errors) + 1e-6  # Avoid division by zero
                observed_precision = 1.0 / observed_variance

                # Blend prior precision with observed (exponential moving average)
                new_precision = (1 - self.learning_rate) * self.precision_beliefs[
                    level_idx
                ] + self.learning_rate * observed_precision
            else:
                # Not enough history yet - use simple inverse-error heuristic
                new_precision = (1 - self.learning_rate) * self.precision_beliefs[
                    level_idx
                ] + self.learning_rate * (1.0 / (1.0 + error_mag))

            self.precision_beliefs[level_idx] = np.clip(
                new_precision, self.min_precision, self.max_precision
            )

            # Update context-specific map
            if ctx not in self.context_precision_map:
                self.context_precision_map[ctx] = self.precision_beliefs.copy()
            else:
                # Blend context-specific with global
                self.context_precision_map[ctx][level_idx] = (
                    1 - self.learning_rate
                ) * self.context_precision_map[ctx][
                    level_idx
                ] + self.learning_rate * self.precision_beliefs[
                    level_idx
                ]

            self._update_history.append(
                PrecisionUpdate(
                    level=level_idx,
                    old_precision=old_precision,
                    new_precision=float(self.precision_beliefs[level_idx]),
                    prediction_error=error_mag,
                    context_key=ctx,
                )
            )

        # Decay unused contexts so the map doesn't grow unbounded
        for key in list(self.context_precision_map.keys()):
            if key != ctx:
                self.context_precision_map[key] *= self.context_decay
                self.context_precision_map[key] = np.clip(
                    self.context_precision_map[key],
                    self.min_precision,
                    self.max_precision,
                )

        # Recursive step 2: observe how well the update was predicted
        self.observe_meta_error()

        state = self.get_precision_state()
        self._state_history.append(state)
        return state

    def apply_to_hierarchy(self, hierarchical_processor) -> None:
        """
        Apply current precision beliefs to the HierarchicalPredictiveProcessor.

        This is the write-back: the hyper-model's learned precisions
        directly modulate how much each level's prediction errors
        influence belief updating.
        """
        for level_idx, level in enumerate(hierarchical_processor.levels):
            if level_idx < self.num_levels:
                level.precision = self.allocate_precision(level_idx)

    def get_precision_state(self) -> PrecisionState:
        """Get current precision allocation state."""
        precisions = {
            i: float(self.precision_beliefs[i]) for i in range(self.num_levels)
        }

        total = sum(precisions.values())
        # Precision entropy: how uniformly distributed is precision?
        if total > 0:
            normalized = np.array(list(precisions.values())) / total
            entropy = -np.sum(normalized * np.log(normalized + 1e-10))
        else:
            entropy = 0.0

        dominant = max(precisions, key=precisions.get)

        return PrecisionState(
            level_precisions=precisions,
            context_key=self.current_context,
            total_precision=total,
            precision_entropy=float(entropy),
            dominant_level=dominant,
        )

    def get_global_precision_state(self) -> Dict[int, float]:
        """Get simple dict of level -> precision for external consumers."""
        return {i: float(self.precision_beliefs[i]) for i in range(self.num_levels)}

    def get_precision_trend(self) -> Dict[int, str]:
        """Get trend direction for each level's precision."""
        trends = {}
        for level_idx in range(self.num_levels):
            history = [
                s.level_precisions[level_idx]
                for s in self._state_history
                if level_idx in s.level_precisions
            ]
            if len(history) < 3:
                trends[level_idx] = "insufficient_data"
            else:
                recent = np.mean(history[-5:])
                older = (
                    np.mean(history[-10:-5])
                    if len(history) >= 10
                    else np.mean(history[:-5])
                )
                diff = recent - older
                if diff > 0.1:
                    trends[level_idx] = "increasing"
                elif diff < -0.1:
                    trends[level_idx] = "decreasing"
                else:
                    trends[level_idx] = "stable"
        return trends

    def generate_report(self) -> str:
        """Generate a human-readable report of precision state."""
        state = self.get_precision_state()
        level_names = {0: "sensory", 1: "contextual", 2: "abstract"}

        lines = ["Precision allocation across predictive hierarchy:"]
        for level, precision in state.level_precisions.items():
            name = level_names.get(level, f"level_{level}")
            bar = "#" * int(precision * 5)
            lines.append(f" {name}: {precision:.3f} {bar}")

        lines.append(
            f" Dominant: {level_names.get(state.dominant_level, state.dominant_level)}"
        )
        lines.append(
            f" Entropy: {state.precision_entropy:.3f} "
            f"({'uniform' if state.precision_entropy > 0.9 else 'focused'})"
        )
        lines.append(f" Context: {state.context_key}")
        confidence = self.recursive_confidence
        lines.append(
            " Recursive confidence: "
            + ("insufficient_data" if confidence is None else f"{confidence:.3f}")
        )

        return "\n".join(lines)

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize state as scalars, numpy arrays and plain histories."""
        keys = list(self.context_precision_map.keys())
        if keys:
            context_matrix = np.vstack(
                [np.asarray(self.context_precision_map[k], dtype=float) for k in keys]
            )
        else:
            context_matrix = np.zeros((0, self.num_levels))

        return {
            "scalars": {
                "num_levels": self.num_levels,
                "update_count": self.update_count,
                "current_context": self.current_context,
                "meta_error_ema": self._meta_error_ema,
                "recursive_confidence": self.recursive_confidence,
                "meta_observed": self._meta_observed,
                "prediction_count": self._prediction_count,
            },
            "arrays": {
                "precision_beliefs": np.asarray(self.precision_beliefs, dtype=float),
                "precision_predictions": np.asarray(
                    self._precision_predictions, dtype=float
                ),
                "context_keys": np.array(keys, dtype=str),
                "context_precision": context_matrix,
            },
            "histories": {
                "update_history": [
                    precision_update_to_dict(u) for u in self._update_history
                ],
                "state_history": [
                    precision_state_to_dict(s) for s in self._state_history
                ],
                "error_history": [[float(v) for v in d] for d in self._error_history],
                "meta_errors": [float(v) for v in self._meta_errors],
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state; histories come back as their dataclasses.

        Arrays whose shape does not match this model's level count, or that
        hold non-finite values, are rejected and the current values kept.
        """
        s = state.get("scalars", {}) or {}
        a = state.get("arrays", {}) or {}
        h = state.get("histories", {}) or {}

        self.update_count = int(s.get("update_count", self.update_count))
        self.current_context = str(s.get("current_context", self.current_context))
        self._meta_error_ema = float(s.get("meta_error_ema", 0.0))
        self._prediction_count = int(s.get("prediction_count", 0))
        # A record written before the flag existed always carried the 0.5
        # start; it holds an observation only when it holds meta-errors
        self._meta_observed = bool(s.get("meta_observed", bool(h.get("meta_errors"))))
        confidence = s.get("recursive_confidence")
        if confidence is None:
            confidence = max(0.0, min(1.0, 1.0 - self._meta_error_ema))
        self._recursive_confidence = float(confidence)

        shape = (self.num_levels,)
        beliefs = _finite_array(a.get("precision_beliefs"), shape)
        if beliefs is not None:
            self.precision_beliefs = beliefs
        elif "precision_beliefs" in a:
            logger.warning("HyperModel: persisted precision beliefs rejected")
        predictions = _finite_array(a.get("precision_predictions"), shape)
        if predictions is not None:
            self._precision_predictions = predictions

        if "context_keys" in a and "context_precision" in a:
            keys = [str(k) for k in np.asarray(a["context_keys"]).reshape(-1)]
            matrix = _finite_array(a["context_precision"], (len(keys), self.num_levels))
            if matrix is not None:
                self.context_precision_map = {
                    key: matrix[i].copy() for i, key in enumerate(keys)
                }
            else:
                logger.warning("HyperModel: persisted context precision rejected")

        if "error_history" in h:
            for i, values in enumerate(h["error_history"][: self.num_levels]):
                maxlen = self._error_history[i].maxlen
                self._error_history[i] = deque(
                    (float(v) for v in values), maxlen=maxlen
                )
        if "meta_errors" in h:
            self._meta_errors = deque(
                (float(v) for v in h["meta_errors"]), maxlen=self._meta_errors.maxlen
            )
        if "update_history" in h:
            self._update_history = deque(
                (
                    precision_update_from_dict(raw)
                    for raw in h["update_history"]
                    if isinstance(raw, dict)
                ),
                maxlen=self._update_history.maxlen,
            )
        if "state_history" in h:
            self._state_history = deque(
                (
                    precision_state_from_dict(raw)
                    for raw in h["state_history"]
                    if isinstance(raw, dict)
                ),
                maxlen=self._state_history.maxlen,
            )


def _finite_array(raw: Any, shape: tuple) -> Optional[np.ndarray]:
    """Return a float copy of raw when it has the expected shape and is finite."""
    if raw is None:
        return None
    try:
        arr = np.array(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        return None
    return arr


def precision_state_to_dict(ps: PrecisionState) -> Dict[str, Any]:
    """Plain-JSON form of a PrecisionState (level keys become strings)."""
    return {
        "level_precisions": {str(k): float(v) for k, v in ps.level_precisions.items()},
        "context_key": ps.context_key,
        "total_precision": float(ps.total_precision),
        "precision_entropy": float(ps.precision_entropy),
        "dominant_level": int(ps.dominant_level),
        "timestamp": float(ps.timestamp),
    }


def precision_state_from_dict(raw: Dict[str, Any]) -> PrecisionState:
    """Rebuild a PrecisionState, restoring integer level keys."""
    levels = {int(k): float(v) for k, v in (raw.get("level_precisions") or {}).items()}
    return PrecisionState(
        level_precisions=levels,
        context_key=str(raw.get("context_key", "general")),
        total_precision=float(raw.get("total_precision", sum(levels.values()))),
        precision_entropy=float(raw.get("precision_entropy", 0.0)),
        dominant_level=int(raw.get("dominant_level", 0)),
        timestamp=float(raw.get("timestamp", time.time())),
    )


def precision_update_to_dict(pu: PrecisionUpdate) -> Dict[str, Any]:
    """Plain-JSON form of a PrecisionUpdate."""
    return {
        "level": int(pu.level),
        "old_precision": float(pu.old_precision),
        "new_precision": float(pu.new_precision),
        "prediction_error": float(pu.prediction_error),
        "context_key": pu.context_key,
        "timestamp": float(pu.timestamp),
    }


def precision_update_from_dict(raw: Dict[str, Any]) -> PrecisionUpdate:
    """Rebuild a PrecisionUpdate."""
    return PrecisionUpdate(
        level=int(raw.get("level", 0)),
        old_precision=float(raw.get("old_precision", 1.0)),
        new_precision=float(raw.get("new_precision", 1.0)),
        prediction_error=float(raw.get("prediction_error", 0.0)),
        context_key=str(raw.get("context_key", "general")),
        timestamp=float(raw.get("timestamp", time.time())),
    )
