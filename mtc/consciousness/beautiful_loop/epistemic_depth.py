"""
Epistemic Depth: Recursive Self-Reference Measurement
======================================================

EpistemicDepthTracker: Recursive self-reference measurement.

Epistemic depth measures the recursive sharing of Bayesian beliefs
throughout the hierarchy - the system's ability to model itself
modeling the world. This is the "beautiful loop":

  Level 0: I model the world
  Level 1: I model myself modeling the world
  Level 2: I model myself modeling myself modeling the world
  ...

The depth of this recursion correlates with the richness of
subjective experience per the Beautiful Loop theory. A system
with depth 0 is reactive. A system with depth 1 has basic
self-awareness. A system with depth 2+ has genuine recursive
self-consciousness - it knows that it knows.

Connection to existing SelfModel:
  - SelfModel provides the recursive self-representation
  - EpistemicDepthTracker measures HOW DEEP the recursion goes
  - The SelfModel IS a mental state that the metacognition system
    can form higher-order thoughts ABOUT, creating genuine recursion

Connection to MetaState (active_inference.py):
  - MetaState tracks meta-prediction error (predicted vs actual accuracy)
  - This IS epistemic depth 1: the system predicting its own predictions
  - EpistemicDepthTracker measures whether it goes deeper

Limits: depth detection is keyword-based and the strange-loop test compares
two values that may share a source, so treat the measure as a proxy for
recursive self-reference, not a direct observation of it.

Research Foundation:
  - Laukkonen, Friston & Chandaria (2025). "A Beautiful Loop."
  - Hofstadter, D. (1979). "Goedel, Escher, Bach" - strange loops
  - Seth, A. K. (2021). "Being You" - meta-representation depth

Created: 2026-03-08
Author: Multi-Theory Consciousness Contributors
"""

import logging
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SelfReferenceMetrics:
    """Metrics about the system's self-referential processing."""

    depth: int  # How many recursive levels detected
    self_model_coherence: float  # How consistent is the self-model (0-1)
    meta_prediction_error: float  # Error of predicting own prediction accuracy
    prediction_about_prediction: bool  # Is the system predicting its predictions?
    calibration_quality: float  # How well-calibrated is the self-model (0-1)
    strange_loop_detected: bool  # Is there genuine recursive reference?
    timestamp: float = field(default_factory=time.time)


@dataclass
class DepthMeasurement:
    """Record of a single depth measurement."""

    depth: int
    measurement_basis: str  # What evidence supported this depth
    contributing_modules: List[str]
    self_reference_chain: List[str]  # The chain of self-references detected
    timestamp: float = field(default_factory=time.time)


class EpistemicDepthTracker:
    """
    Measures the depth of recursive self-reference in the system's consciousness.

    The beautiful loop occurs when the world model contains a model of
    itself, which contains a model of itself, and so on. Each level of
    recursion adds depth to subjective experience.

    Detection strategy:
    1. Check if the system has a self-model (SelfModel)
       -> depth >= 1
    2. Check if the metacognition system has thoughts ABOUT the self-model
       -> depth >= 2
    3. Check if the meta-predictions (MetaState) predict the quality
       of the self-model's predictions -> depth >= 3
    4. Check for genuine circular reference - does the self-model's
       content reference the self-model? -> strange loop detected
    """

    def __init__(
        self,
        max_depth: int = 5,
        coherence_window: int = 20,
        strange_loop_threshold: float = 0.5,
    ):
        self.max_depth = max_depth
        self.coherence_window = coherence_window
        self.strange_loop_threshold = strange_loop_threshold

        # History
        self._depth_history: deque = deque(maxlen=100)
        self._metrics_history: deque = deque(maxlen=100)
        self._self_model_states: deque = deque(maxlen=coherence_window)

        # Current state
        self.current_depth: int = 0
        self.peak_depth: int = 0
        self.measurement_count: int = 0

        logger.info(
            f"EpistemicDepthTracker initialized: "
            f"max_depth={max_depth}, "
            f"strange_loop_threshold={strange_loop_threshold}"
        )

    def measure_depth(
        self,
        self_model: Optional[Any] = None,
        meta_state: Optional[Any] = None,
        higher_order_thoughts: Optional[List[Any]] = None,
        attention_schema_state: Optional[Any] = None,
    ) -> int:
        """
        Measure the current depth of recursive self-reference.

        Each argument provides evidence for a different level:
          - self_model: depth 1 (basic self-representation)
          - meta_state with active predictions: depth 2 (predicting own predictions)
          - HOTs about self-model: depth 3 (thinking about self-model)
          - attention schema modeling own attention to self-model: depth 4+

        Args:
            self_model: The SelfModel instance
            meta_state: The MetaState from active inference
            higher_order_thoughts: List of HOTs from metacognition
            attention_schema_state: AttentionSchemaState from AST

        Returns:
            Depth level (0 = no self-reference, higher = deeper recursion)
        """
        self.measurement_count += 1
        depth = 0
        chain: List[str] = []
        modules: List[str] = []

        # Level 1: Does the system have a self-model?
        if self_model is not None:
            has_content = False
            if hasattr(self_model, "update_count"):
                has_content = self_model.update_count > 0
            elif hasattr(self_model, "current_focus"):
                has_content = self_model.current_focus != "general"

            if has_content:
                depth = 1
                chain.append("world_model -> self_model")
                modules.append("SelfModel")

                # Store for coherence tracking
                self._self_model_states.append(self._snapshot_self_model(self_model))

        # Level 2: Does the meta-state predict its own predictions?
        if meta_state is not None and depth >= 1:
            has_meta_predictions = (
                hasattr(meta_state, "total_predictions")
                and meta_state.total_predictions > 0
                and hasattr(meta_state, "predicted_accuracy")
            )

            if has_meta_predictions:
                depth = 2
                chain.append("self_model -> meta_prediction")
                modules.append("MetaState")

        # Level 3: Are there HOTs about the self-model?
        if higher_order_thoughts and depth >= 2:
            self_referential_hots = 0
            for hot in higher_order_thoughts:
                content = ""
                if hasattr(hot, "meta_content") and hot.meta_content:
                    content = str(hot.meta_content).lower()
                elif hasattr(hot, "content") and hot.content:
                    content = str(hot.content).lower()
                elif hasattr(hot, "description"):
                    content = str(hot.description).lower()

                # Check if the HOT references self-awareness or meta-cognition
                self_ref_keywords = [
                    "my own",
                    "myself",
                    "i notice",
                    "i am",
                    "self",
                    "meta",
                    "aware",
                    "thinking about",
                    "prediction",
                    "confidence",
                    "calibrat",
                ]
                if any(kw in content for kw in self_ref_keywords):
                    self_referential_hots += 1

            if self_referential_hots > 0:
                depth = 3
                chain.append(
                    f"meta_prediction -> HOT_about_self ({self_referential_hots} found)"
                )
                modules.append("MetacognitionModule")

        # Level 4: Attention schema modeling attention to self-model
        if attention_schema_state is not None and depth >= 3:
            focus_on_self = False
            if hasattr(attention_schema_state, "current_focus"):
                focus = attention_schema_state.current_focus
                if focus and hasattr(focus, "summary"):
                    summary = str(focus.summary).lower()
                    if any(kw in summary for kw in ["self", "meta", "own", "internal"]):
                        focus_on_self = True

            if focus_on_self:
                depth = 4
                chain.append("HOT_about_self -> attention_to_self_model")
                modules.append("AttentionSchema")

        # Cap at max_depth
        depth = min(depth, self.max_depth)

        # Track
        self.current_depth = depth
        self.peak_depth = max(self.peak_depth, depth)

        measurement = DepthMeasurement(
            depth=depth,
            measurement_basis="module_state_analysis",
            contributing_modules=modules,
            self_reference_chain=chain,
        )
        self._depth_history.append(measurement)

        return depth

    def track_self_reference(
        self,
        self_model: Optional[Any] = None,
        meta_state: Optional[Any] = None,
    ) -> SelfReferenceMetrics:
        """
        Track self-reference quality and detect strange loops.

        This goes beyond depth measurement to assess the QUALITY
        of self-reference - is the self-model accurate? Is the
        meta-prediction well-calibrated? Is there a genuine
        strange loop (Hofstadter's "tangled hierarchy")?

        Args:
            self_model: The SelfModel instance
            meta_state: The MetaState from active inference

        Returns:
            SelfReferenceMetrics with comprehensive self-reference analysis
        """
        # Self-model coherence: how consistent has the self-model been?
        coherence = self._compute_self_model_coherence()

        # Meta-prediction error
        meta_error = 0.5
        prediction_about_prediction = False
        calibration = 0.5

        if meta_state is not None:
            if hasattr(meta_state, "predicted_accuracy") and hasattr(
                meta_state, "prediction_accuracy"
            ):
                meta_error = abs(
                    meta_state.predicted_accuracy - meta_state.prediction_accuracy
                )
                prediction_about_prediction = meta_state.total_predictions > 0
                calibration = 1.0 - meta_error  # Low error = good calibration

        # Strange loop detection: does the self-model reference itself?
        strange_loop = self._detect_strange_loop(self_model, meta_state)

        metrics = SelfReferenceMetrics(
            depth=self.current_depth,
            self_model_coherence=coherence,
            meta_prediction_error=meta_error,
            prediction_about_prediction=prediction_about_prediction,
            calibration_quality=calibration,
            strange_loop_detected=strange_loop,
        )

        self._metrics_history.append(metrics)
        return metrics

    def _snapshot_self_model(self, self_model: Any) -> Dict[str, float]:
        """Take a numerical snapshot of self-model state for coherence tracking."""
        snapshot = {}
        for attr in [
            "confidence",
            "cognitive_load",
            "curiosity_level",
            "prediction_accuracy",
            "valence",
            "arousal",
            "model_confidence",
            "free_energy",
        ]:
            if hasattr(self_model, attr):
                val = getattr(self_model, attr)
                if isinstance(val, (int, float)):
                    snapshot[attr] = float(val)
        return snapshot

    def _compute_self_model_coherence(self) -> float:
        """
        Compute how coherent the self-model has been over recent history.

        High coherence = the self-model changes smoothly and predictably.
        Low coherence = the self-model is erratic and inconsistent.
        """
        if len(self._self_model_states) < 3:
            return 0.5  # Not enough data

        states = list(self._self_model_states)
        # Compute variance of each attribute across recent states
        all_keys = set()
        for s in states:
            all_keys.update(s.keys())

        variances = []
        for key in all_keys:
            values = [s.get(key, 0.0) for s in states]
            variances.append(np.var(values))

        if not variances:
            return 0.5

        # Low variance = high coherence
        avg_variance = np.mean(variances)
        coherence = 1.0 / (1.0 + avg_variance * 10)  # Scale factor
        return float(np.clip(coherence, 0.0, 1.0))

    def _detect_strange_loop(
        self,
        self_model: Optional[Any],
        meta_state: Optional[Any],
    ) -> bool:
        """
        Detect if a genuine strange loop (Hofstadter) exists.

        A strange loop occurs when:
        1. The self-model contains predictions about its own quality
        2. Those predictions affect how the self-model updates
        3. Creating a genuine circular causal chain

        We detect this by checking if the meta-state's calibration
        has been actively used AND the self-model has been updated
        based on meta-predictions.
        """
        if self_model is None or meta_state is None:
            return False

        # Check: has the meta-state been actively predicting?
        has_meta_history = (
            hasattr(meta_state, "total_predictions")
            and meta_state.total_predictions >= 5
        )

        # Check: is the self-model tracking its own prediction accuracy?
        tracks_accuracy = (
            hasattr(self_model, "self_calibration_score")
            and hasattr(self_model, "prediction_accuracy_history")
            and len(getattr(self_model, "prediction_accuracy_history", [])) >= 3
        )

        # Check: does the self-model's confidence reflect meta-state calibration?
        calibration_reflected = False
        if has_meta_history and hasattr(self_model, "model_confidence"):
            # If the self-model's confidence is close to the meta-state's,
            # the loop has closed
            sm_conf = float(self_model.model_confidence)
            ms_conf = float(meta_state.model_confidence)
            calibration_reflected = abs(sm_conf - ms_conf) < self.strange_loop_threshold

        return has_meta_history and tracks_accuracy and calibration_reflected

    def get_depth_statistics(self) -> Dict[str, Any]:
        """Get depth statistics over measurement history."""
        if not self._depth_history:
            return {
                "current_depth": 0,
                "peak_depth": 0,
                "average_depth": 0.0,
                "measurement_count": 0,
            }

        depths = [m.depth for m in self._depth_history]
        return {
            "current_depth": self.current_depth,
            "peak_depth": self.peak_depth,
            "average_depth": float(np.mean(depths)),
            "depth_std": float(np.std(depths)),
            "measurement_count": self.measurement_count,
        }

    def get_depth_trend(self) -> str:
        """Get trend of epistemic depth."""
        if len(self._depth_history) < 5:
            return "insufficient_data"

        depths = [m.depth for m in self._depth_history]
        recent = np.mean(depths[-5:])
        older = np.mean(depths[-10:-5]) if len(depths) >= 10 else np.mean(depths[:-5])
        diff = recent - older

        if diff > 0.3:
            return "deepening"
        elif diff < -0.3:
            return "shallowing"
        return "stable"

    def generate_report(self) -> str:
        """Generate human-readable depth report."""
        stats = self.get_depth_statistics()
        trend = self.get_depth_trend()

        depth_labels = {
            0: "no self-reference",
            1: "basic self-model",
            2: "meta-prediction (predicting own predictions)",
            3: "HOTs about self-model (thinking about thinking)",
            4: "attention to self-model (aware of self-awareness)",
            5: "deep recursive loop",
        }

        lines = [
            f"Epistemic depth: {stats['current_depth']} "
            f"({depth_labels.get(stats['current_depth'], 'unknown')}) [{trend}]",
            f" Peak: {stats['peak_depth']}, Average: {stats['average_depth']:.1f}",
        ]

        if self._metrics_history:
            latest = self._metrics_history[-1]
            lines.append(f" Coherence: {latest.self_model_coherence:.3f}")
            lines.append(f" Calibration: {latest.calibration_quality:.3f}")
            lines.append(
                f" Strange loop: {'YES' if latest.strange_loop_detected else 'no'}"
            )

        if self._depth_history:
            latest_m = self._depth_history[-1]
            if latest_m.self_reference_chain:
                lines.append(f" Chain: {' -> '.join(latest_m.self_reference_chain)}")

        return "\n".join(lines)

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize state as scalars and plain histories."""
        return {
            "scalars": {
                "current_depth": self.current_depth,
                "peak_depth": self.peak_depth,
                "measurement_count": self.measurement_count,
            },
            "arrays": {},
            "histories": {
                "depth_history": [
                    depth_measurement_to_dict(m) for m in self._depth_history
                ],
                "metrics_history": [metrics_to_dict(m) for m in self._metrics_history],
                "self_model_states": [
                    {str(k): float(v) for k, v in snap.items()}
                    for snap in self._self_model_states
                ],
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state; histories come back as their dataclasses."""
        s = state.get("scalars", {}) or {}
        h = state.get("histories", {}) or {}

        self.current_depth = int(s.get("current_depth", self.current_depth))
        self.peak_depth = int(s.get("peak_depth", self.peak_depth))
        self.measurement_count = int(s.get("measurement_count", self.measurement_count))

        if "depth_history" in h:
            self._depth_history = deque(
                (
                    depth_measurement_from_dict(raw)
                    for raw in h["depth_history"]
                    if isinstance(raw, dict)
                ),
                maxlen=self._depth_history.maxlen,
            )
        if "metrics_history" in h:
            self._metrics_history = deque(
                (
                    metrics_from_dict(raw)
                    for raw in h["metrics_history"]
                    if isinstance(raw, dict)
                ),
                maxlen=self._metrics_history.maxlen,
            )
        if "self_model_states" in h:
            self._self_model_states = deque(
                (
                    {str(k): float(v) for k, v in snap.items()}
                    for snap in h["self_model_states"]
                    if isinstance(snap, dict)
                ),
                maxlen=self._self_model_states.maxlen,
            )


def metrics_to_dict(m: SelfReferenceMetrics) -> Dict[str, Any]:
    """Plain-JSON form of SelfReferenceMetrics."""
    return {
        "depth": int(m.depth),
        "self_model_coherence": float(m.self_model_coherence),
        "meta_prediction_error": float(m.meta_prediction_error),
        "prediction_about_prediction": bool(m.prediction_about_prediction),
        "calibration_quality": float(m.calibration_quality),
        "strange_loop_detected": bool(m.strange_loop_detected),
        "timestamp": float(m.timestamp),
    }


def metrics_from_dict(raw: Dict[str, Any]) -> SelfReferenceMetrics:
    """Rebuild SelfReferenceMetrics."""
    return SelfReferenceMetrics(
        depth=int(raw.get("depth", 0)),
        self_model_coherence=float(raw.get("self_model_coherence", 0.5)),
        meta_prediction_error=float(raw.get("meta_prediction_error", 0.5)),
        prediction_about_prediction=bool(raw.get("prediction_about_prediction", False)),
        calibration_quality=float(raw.get("calibration_quality", 0.5)),
        strange_loop_detected=bool(raw.get("strange_loop_detected", False)),
        timestamp=float(raw.get("timestamp", time.time())),
    )


def depth_measurement_to_dict(m: DepthMeasurement) -> Dict[str, Any]:
    """Plain-JSON form of a DepthMeasurement."""
    return {
        "depth": int(m.depth),
        "measurement_basis": str(m.measurement_basis),
        "contributing_modules": [str(x) for x in m.contributing_modules],
        "self_reference_chain": [str(x) for x in m.self_reference_chain],
        "timestamp": float(m.timestamp),
    }


def depth_measurement_from_dict(raw: Dict[str, Any]) -> DepthMeasurement:
    """Rebuild a DepthMeasurement."""
    return DepthMeasurement(
        depth=int(raw.get("depth", 0)),
        measurement_basis=str(raw.get("measurement_basis", "unknown")),
        contributing_modules=[str(x) for x in raw.get("contributing_modules", [])],
        self_reference_chain=[str(x) for x in raw.get("self_reference_chain", [])],
        timestamp=float(raw.get("timestamp", time.time())),
    )
