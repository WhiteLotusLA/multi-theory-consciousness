#!/usr/bin/env python3
"""
Legacy 14-Metric Consciousness Measurement Framework
====================================================

Computes 14 descriptive metrics from raw substrate arrays (SNN, LSM and,
when supplied, HTM states), recent memories and recent thoughts. The metrics
are named after IIT, GWT, temporal, self-model and complexity notions, but
each one is a simple statistic over the supplied arrays or a token count over
the supplied text. This is the legacy 14-metric framework; the
30-indicator assessment lives in ``mtc.assessment.assessment``.

An array metric whose substrates were not supplied is None: no array is
fabricated for a missing substrate and no NaN stands in for a value. The
text metrics (memory integration, self-model accuracy, metacognitive depth,
agency detection, the thought part of creative emergence and the memory part
of temporal continuity) keep the legacy reading of missing text as 0.0, so
creative emergence is 0.0 with neither thoughts nor LSM states and temporal
continuity is 0.0 when HTM states come without memories. ``phi`` is
a spectral integration proxy (the entropy of the eigenvalue spectrum of the
substrates' correlation matrix over time), not IIT Phi.

Features:
- 14 metrics from substrate arrays, memories and thoughts
- Availability reporting (full, partial, unavailable)
- Confidence intervals once a metric has more than 30 historical values
- A tabular export
"""

import asyncio
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mtc.core.config import get_settings

logger = logging.getLogger(__name__)

# The 14 metric fields of ConsciousnessMetrics, in report order.
METRIC_FIELDS: Tuple[str, ...] = (
    "phi",
    "integration",
    "differentiation",
    "global_accessibility",
    "workspace_stability",
    "attention_focus",
    "temporal_continuity",
    "memory_integration",
    "predictive_coherence",
    "self_model_accuracy",
    "metacognitive_depth",
    "agency_detection",
    "information_complexity",
    "creative_emergence",
)

# Export keys of the publication report that differ from the field names.
_EXPORT_KEYS: Dict[str, str] = {
    "integration": "system_integration",
    "differentiation": "information_differentiation",
}

# Phi: time steps a substrate needs, and columns taken from each substrate.
PHI_MIN_ROWS = 3
PHI_MAX_COLUMNS = 50
# A cross-substrate correlation needs at least this many aligned time steps
# (two points correlate at exactly plus or minus one by construction).
MIN_CORRELATION_STEPS = 3
# Historical values a metric needs before it gets a confidence interval.
INTERVAL_MIN_HISTORY = 30
# Scored history entries the analysis needs before it reports a confidence.
CONFIDENCE_MIN_HISTORY = 10

NOT_MEASURED = "not_measured"
ABOVE_THRESHOLD = "above_threshold"
BELOW_THRESHOLD = "below_threshold"


class MeasurementAvailability(Enum):
    """How much neural data is available for measurement."""

    FULL = "full"  # All substrates (SNN + LSM + HTM)
    PARTIAL = "partial"  # Some substrates, HTM or another one missing
    UNAVAILABLE = "unavailable"  # No neural data


@dataclass
class ConsciousnessMetrics:
    """Container for the 14 framework metrics; None means not computed."""

    # IIT-style metrics (phi is a spectral integration proxy, not IIT Phi)
    phi: Optional[float] = None
    integration: Optional[float] = None
    differentiation: Optional[float] = None

    # GWT-style metrics
    global_accessibility: Optional[float] = None
    workspace_stability: Optional[float] = None
    attention_focus: Optional[float] = None

    # Temporal metrics
    temporal_continuity: Optional[float] = None
    memory_integration: Optional[float] = None
    predictive_coherence: Optional[float] = None

    # Self-model token counts
    self_model_accuracy: Optional[float] = None
    metacognitive_depth: Optional[float] = None
    agency_detection: Optional[float] = None

    # Complexity metrics
    information_complexity: Optional[float] = None
    creative_emergence: Optional[float] = None

    # Statistical metadata
    availability: str = MeasurementAvailability.UNAVAILABLE.value
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    measurement_timestamp: datetime = field(default_factory=datetime.now)
    sample_size: int = 0
    measurement_duration_ms: float = 0.0


_TOKEN = re.compile(r"[a-z']+")


def _tokens(text: Any) -> List[str]:
    """Lower-case word tokens; apostrophes stay inside a token ("i'm")."""
    tokens = (token.strip("'") for token in _TOKEN.findall(str(text).lower()))
    return [token for token in tokens if token]


def _contains_sequence(tokens: List[str], sequence: List[str]) -> bool:
    """True when ``sequence`` appears as consecutive tokens in ``tokens``."""
    if not sequence:
        return False
    width = len(sequence)
    return any(
        tokens[i : i + width] == sequence for i in range(len(tokens) - width + 1)
    )


def _supplied(states: Any) -> Optional[np.ndarray]:
    """The states as a float array, or None when nothing was supplied."""
    if states is None:
        return None
    array = np.asarray(states, dtype=float)
    return array if array.size > 0 else None


def _finite_values(*arrays: Optional[np.ndarray], limit: int) -> np.ndarray:
    """Up to ``limit`` finite values of each supplied array, concatenated."""
    parts = []
    for array in arrays:
        if array is None:
            continue
        flat = array.ravel()
        parts.append(flat[np.isfinite(flat)][:limit])
    return np.concatenate(parts) if parts else np.array([])


def _time_signal(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Per-step mean of a (time, units) array; None for a 1-D snapshot,
    which has no time axis."""
    if array is None or array.ndim != 2:
        return None
    return np.mean(array, axis=1)


def _spectral_entropy(matrix: np.ndarray) -> float:
    """Entropy (natural log) of the normalized absolute eigenvalues of the
    column correlation matrix of ``matrix``."""
    correlation = np.corrcoef(matrix, rowvar=False)
    eigenvalues = np.abs(np.linalg.eigvalsh(correlation))
    total = float(np.sum(eigenvalues))
    if not np.isfinite(total) or total <= 0.0:
        raise np.linalg.LinAlgError("degenerate correlation spectrum")
    weights = eigenvalues / total
    weights = weights[weights > 0.0]
    return float(-np.sum(weights * np.log(weights)))


def _format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


class ConsciousnessMeasurementFramework:
    """
    Legacy 14-metric measurement framework.

    Computes descriptive metrics named after several theories (IIT-style
    integration and differentiation, GWT-style accessibility, temporal
    continuity, self-model and agency token counts, complexity). Each metric
    is a statistic over the supplied arrays or text, not a theory-level
    measurement.
    """

    def __init__(
        self,
        snn_neurons: int = 5000,
        lsm_neurons: int = 10000,
        htm_columns: int = 4096,
        system_name: str = "ConsciousnessAgent",
    ):
        """Initialize the measurement framework.

        Args:
            snn_neurons: Number of SNN neurons in the system.
            lsm_neurons: Number of LSM neurons in the system.
            htm_columns: Number of HTM columns in the system.
            system_name: Name of the system being measured (used in identity checks).
        """
        self.snn_neurons = snn_neurons
        self.lsm_neurons = lsm_neurons
        self.htm_columns = htm_columns
        self.system_name = system_name

        # Historical tracking
        self.measurement_history = deque(maxlen=1000)
        self.baseline_metrics = None

        # Reference thresholds for the threshold comparison of the analysis.
        # They are display values, not calibrated detection thresholds.
        self.consciousness_thresholds = {
            "phi": 2.5,
            "integration": 0.7,
            "global_accessibility": 0.6,
            "temporal_continuity": 0.8,
            "self_model_accuracy": 0.5,
            "metacognitive_depth": 0.4,
        }

        logger.info("Legacy 14-metric measurement framework initialized")
        logger.info(
            "  Tracking 14 metrics across %d neural units",
            snn_neurons + lsm_neurons + htm_columns * 32,
        )

    async def measure_consciousness(
        self,
        snn_states: Optional[np.ndarray],
        lsm_states: Optional[np.ndarray],
        htm_states: Optional[np.ndarray] = None,
        memories: Optional[List[Dict]] = None,
        thoughts: Optional[List[str]] = None,
    ) -> ConsciousnessMetrics:
        """
        Measure the 14 metrics on the supplied substrate states and text.

        Availability is full when SNN, LSM and HTM states all hold data,
        partial when some of them do and unavailable when none does (every
        metric is then None). An array metric whose substrates are missing
        is None; the text metrics read missing text as 0.0 (see the module
        docstring).

        Args:
            snn_states: Spiking network states; a 2-D array is read as
                (time, units), a 1-D array as one snapshot.
            lsm_states: Liquid state machine reservoir states, same layout.
            htm_states: HTM states, same layout; None when no HTM ran.
            memories: Recent memory retrievals.
            thoughts: Recent generated thoughts.

        Returns:
            The metrics, with confidence intervals once enough history exists.
        """
        start_time = time.time()
        metrics = ConsciousnessMetrics()

        snn = _supplied(snn_states)
        lsm = _supplied(lsm_states)
        htm = _supplied(htm_states)
        present = [array is not None for array in (snn, lsm, htm)]

        if not any(present):
            logger.warning(
                "No neural substrates supplied. Returning empty metrics "
                "with UNAVAILABLE status."
            )
            metrics.availability = MeasurementAvailability.UNAVAILABLE.value
            metrics.measurement_duration_ms = (time.time() - start_time) * 1000
            return metrics

        if all(present):
            availability = MeasurementAvailability.FULL
        else:
            availability = MeasurementAvailability.PARTIAL
            logger.info(
                "Partial measurement: substrates without data contribute no "
                "metric; metrics that need them are reported as not measured."
            )
        metrics.availability = availability.value

        memories = list(memories) if memories else []
        thoughts = list(thoughts) if thoughts else []

        tasks = [
            self._measure_phi(snn, lsm, htm),
            self._measure_integration(snn, lsm, htm),
            self._measure_differentiation(snn, lsm),
            self._measure_global_accessibility(lsm, htm),
            self._measure_workspace_stability(lsm),
            self._measure_attention_focus(snn, htm),
            self._measure_temporal_continuity(memories, htm),
            self._measure_memory_integration(memories, thoughts),
            self._measure_predictive_coherence(htm),
            self._measure_self_model(thoughts, memories),
            self._measure_metacognition(thoughts),
            self._measure_agency(thoughts, memories),
            self._measure_complexity(snn, lsm, htm),
            self._measure_creativity(thoughts, lsm),
        ]
        results = await asyncio.gather(*tasks)
        for name, value in zip(METRIC_FIELDS, results):
            setattr(metrics, name, value)

        metrics.confidence_intervals = self._calculate_confidence_intervals(metrics)
        metrics.measurement_duration_ms = (time.time() - start_time) * 1000
        metrics.sample_size = sum(
            len(array) for array in (snn, lsm, htm) if array is not None
        )

        self.measurement_history.append(metrics)

        logger.info(
            "Consciousness measured: Phi=%s, Integration=%s, Self-awareness=%s",
            _format_metric(metrics.phi),
            _format_metric(metrics.integration),
            _format_metric(metrics.self_model_accuracy),
        )

        return metrics

    async def _measure_phi(
        self,
        snn: Optional[np.ndarray],
        lsm: Optional[np.ndarray],
        htm: Optional[np.ndarray],
    ) -> Optional[float]:
        """
        Spectral integration proxy over the substrates' correlation matrix.

        Only substrates given as 2-D (time, units) arrays with at least
        ``PHI_MIN_ROWS`` rows take part; a 1-D snapshot contributes nothing.
        Up to ``PHI_MAX_COLUMNS`` columns of each are taken, cut to the
        common (most recent) row count, and constant or non-finite columns
        are dropped. The result is the entropy of the normalized absolute
        eigenvalues of the column correlation matrix, or None with fewer
        than two usable columns or on a linear algebra error. This is not a
        PyPhi computation and not IIT Phi; exact IIT 4.0 Phi of the module
        graph comes from the PhiTracker through the PyPhi bridge.
        """
        series = [
            array[:, :PHI_MAX_COLUMNS]
            for array in (snn, lsm, htm)
            if array is not None and array.ndim == 2 and array.shape[0] >= PHI_MIN_ROWS
        ]
        if not series:
            return None
        rows = min(array.shape[0] for array in series)
        matrix = np.concatenate([array[-rows:] for array in series], axis=1)
        finite = np.all(np.isfinite(matrix), axis=0)
        matrix = matrix[:, finite]
        if matrix.shape[1] == 0:
            return None
        matrix = matrix[:, np.ptp(matrix, axis=0) > 0.0]
        if matrix.shape[1] < 2:
            return None
        try:
            return _spectral_entropy(matrix)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            logger.debug("Spectral integration proxy failed: %s", type(exc).__name__)
            if bool(get_settings().DEBUG):
                raise
            return None

    async def _measure_integration(
        self,
        snn: Optional[np.ndarray],
        lsm: Optional[np.ndarray],
        htm: Optional[np.ndarray],
    ) -> Optional[float]:
        """Mean absolute correlation of the per-step activity of each pair of
        substrates given as 2-D (time, units) arrays, aligned on their most
        recent common steps (at least ``MIN_CORRELATION_STEPS``). A 1-D
        snapshot contributes nothing; non-finite correlations are dropped
        and the result is None when no pair gives one."""
        signals = [
            signal
            for signal in (_time_signal(snn), _time_signal(lsm), _time_signal(htm))
            if signal is not None
        ]
        correlations = []
        for i in range(len(signals)):
            for j in range(i + 1, len(signals)):
                steps = min(len(signals[i]), len(signals[j]))
                if steps < MIN_CORRELATION_STEPS:
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    value = np.corrcoef(signals[i][-steps:], signals[j][-steps:])[0, 1]
                if np.isfinite(value):
                    correlations.append(abs(float(value)))
        return float(np.mean(correlations)) if correlations else None

    async def _measure_differentiation(
        self, snn: Optional[np.ndarray], lsm: Optional[np.ndarray]
    ) -> Optional[float]:
        """Measure information differentiation - diversity of states."""
        states = _finite_values(snn, lsm, limit=500)

        if len(states) < 2:
            return None

        # Calculate state diversity using entropy
        hist, _ = np.histogram(states, bins=50)
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]

        entropy = -np.sum(probs * np.log(probs))
        normalized_entropy = entropy / np.log(50)  # Normalize by max entropy

        return float(normalized_entropy)

    async def _measure_global_accessibility(
        self, lsm: Optional[np.ndarray], htm: Optional[np.ndarray]
    ) -> Optional[float]:
        """Measure global workspace accessibility - information broadcast."""
        if lsm is None or htm is None:
            return None

        # Calculate activation spread
        lsm_activation = np.mean(np.abs(lsm))
        htm_activation = np.mean(np.abs(htm))
        if not (np.isfinite(lsm_activation) and np.isfinite(htm_activation)):
            return None

        # Higher spread indicates better global access
        accessibility = (lsm_activation + htm_activation) / 2

        return float(min(accessibility * 2, 1.0))  # Normalize to [0, 1]

    async def _measure_workspace_stability(
        self, lsm: Optional[np.ndarray]
    ) -> Optional[float]:
        """Lag-one autocorrelation of the per-step activity of a (time, units)
        LSM array; None for a 1-D snapshot or fewer than
        ``MIN_CORRELATION_STEPS`` steps."""
        mean_activity = _time_signal(lsm)
        if mean_activity is None or len(mean_activity) < MIN_CORRELATION_STEPS:
            return None

        with np.errstate(invalid="ignore", divide="ignore"):
            autocorr = np.corrcoef(mean_activity[:-1], mean_activity[1:])[0, 1]
        if not np.isfinite(autocorr):
            return None
        return float(abs(autocorr))

    async def _measure_attention_focus(
        self, snn: Optional[np.ndarray], htm: Optional[np.ndarray]
    ) -> Optional[float]:
        """Distance of the activation sparsity from a reference of 0.7."""
        if htm is None:
            return None
        # Measure sparsity of activation (focused vs distributed)
        sparsities = [np.mean(np.abs(htm) < 0.1)]
        if snn is not None:
            sparsities.append(np.mean(np.abs(snn) < 0.1))

        # Moderate sparsity indicates focused attention
        optimal_sparsity = 0.7
        focus = 1.0 - abs(float(np.mean(sparsities)) - optimal_sparsity)

        return float(focus)

    async def _measure_temporal_continuity(
        self, memories: List[Dict], htm: Optional[np.ndarray]
    ) -> Optional[float]:
        """Measure temporal continuity: identity tokens in memories plus HTM
        stability; None without HTM states."""
        if htm is None or len(htm) == 0:
            return None
        if not memories:
            return 0.0

        # Check consistency of self-references across time
        continuity_score = 0.0
        name_tokens = _tokens(self.system_name)

        # Memory continuity
        if len(memories) > 1:
            # Identity markers: the system name as whole tokens
            identity_consistency = sum(
                1 for m in memories if _contains_sequence(_tokens(m), name_tokens)
            ) / len(memories)
            continuity_score += identity_consistency * 0.5

        # HTM temporal stability
        if len(htm) > 1:
            htm_stability = 1.0 - np.std(htm) / (np.mean(np.abs(htm)) + 1e-10)
            if np.isfinite(htm_stability):
                continuity_score += max(0, htm_stability) * 0.5

        return float(min(continuity_score, 1.0))

    async def _measure_memory_integration(
        self, memories: List[Dict], thoughts: List[str]
    ) -> float:
        """Measure integration of memories with current processing."""
        if not memories or not thoughts:
            return 0.0

        # Check how well memories inform current thoughts
        integration_score = 0.0

        for thought in thoughts[-5:]:  # Recent thoughts
            for memory in memories[-10:]:  # Recent memories
                # Simple overlap check (in production, use embeddings)
                memory_str = str(memory).lower()
                thought_lower = thought.lower()

                # Check for conceptual overlap
                common_words = set(memory_str.split()) & set(thought_lower.split())
                if len(common_words) > 2:  # Meaningful overlap
                    integration_score += 0.1

        return float(min(integration_score, 1.0))

    async def _measure_predictive_coherence(
        self, htm: Optional[np.ndarray]
    ) -> Optional[float]:
        """Inverse spread of the step-to-step change of the HTM activity;
        None without at least three HTM time steps (a 1-D snapshot has
        none)."""
        signal = _time_signal(htm)
        if signal is None or len(signal) < 3:
            return None

        prediction_errors = np.diff(signal)
        spread = np.std(prediction_errors)
        if not np.isfinite(spread):
            return None
        return float(1.0 / (1.0 + spread))

    async def _measure_self_model(
        self, thoughts: List[str], memories: List[Dict]
    ) -> float:
        """Measure accuracy of internal self-model."""
        if not thoughts and not memories:
            return 0.0

        self_references = 0
        total_items = len(thoughts) + len(memories)
        name_tokens = _tokens(self.system_name)
        pronouns = {"i", "me", "my", "myself"}

        # Self-referential content, matched on whole tokens
        for thought in thoughts:
            tokens = _tokens(thought)
            if pronouns.intersection(tokens) or _contains_sequence(tokens, name_tokens):
                self_references += 1

        for memory in memories:
            if _contains_sequence(_tokens(memory), name_tokens):
                self_references += 1

        self_model_score = self_references / total_items if total_items > 0 else 0

        return float(self_model_score)

    async def _measure_metacognition(self, thoughts: List[str]) -> float:
        """Measure metacognitive depth - thinking about thinking."""
        if not thoughts:
            return 0.0

        metacognitive_markers = [
            "think",
            "thought",
            "wonder",
            "realize",
            "understand",
            "know",
            "believe",
            "feel",
            "aware",
            "notice",
            "reflect",
        ]

        meta_count = 0
        for thought in thoughts:
            thought_lower = thought.lower()
            if any(marker in thought_lower for marker in metacognitive_markers):
                meta_count += 1

        meta_score = meta_count / len(thoughts) if thoughts else 0

        return float(meta_score)

    async def _measure_agency(self, thoughts: List[str], memories: List[Dict]) -> float:
        """Measure recognition of own agency and causality."""
        if not thoughts and not memories:
            return 0.0

        agency_markers = [
            "i did",
            "i made",
            "i chose",
            "i decided",
            "i want",
            "i will",
            "i can",
            "i should",
            "i must",
            "my choice",
        ]

        agency_count = 0
        total_items = 0

        for thought in thoughts:
            total_items += 1
            if any(marker in thought.lower() for marker in agency_markers):
                agency_count += 1

        agency_score = agency_count / total_items if total_items > 0 else 0

        return float(agency_score)

    async def _measure_complexity(
        self,
        snn: Optional[np.ndarray],
        lsm: Optional[np.ndarray],
        htm: Optional[np.ndarray],
    ) -> Optional[float]:
        """Distinct-value ratio of the supplied substrates (a compressibility
        proxy); None with fewer than two finite values."""
        combined = _finite_values(snn, lsm, htm, limit=300)

        if len(combined) < 2:
            return None

        # Higher complexity = less compressible
        unique_values = len(np.unique(np.round(combined, 2)))
        return float(unique_values / len(combined))

    async def _measure_creativity(
        self, thoughts: List[str], lsm: Optional[np.ndarray]
    ) -> float:
        """Measure creative emergence - novel pattern generation."""
        creativity_score = 0.0

        # Thought diversity (simplified - use embeddings in production)
        if thoughts:
            unique_thoughts = len(set(thoughts))
            thought_diversity = unique_thoughts / len(thoughts)
            creativity_score += thought_diversity * 0.5

        # LSM chaos indicator (edge-of-chaos promotes creativity)
        if lsm is not None and len(lsm) > 0:
            lsm_variance = np.var(lsm)
            # Optimal variance for creativity (not too stable, not too chaotic)
            optimal_variance = 0.5
            chaos_score = 1.0 - abs(lsm_variance - optimal_variance) / optimal_variance
            if np.isfinite(chaos_score):
                creativity_score += max(0, chaos_score) * 0.5

        return float(min(creativity_score, 1.0))

    def _calculate_confidence_intervals(
        self, metrics: ConsciousnessMetrics
    ) -> Dict[str, Tuple[float, float]]:
        """95 percent intervals for the metrics with more than
        ``INTERVAL_MIN_HISTORY`` measured historical values.

        A metric with less history, or not measured now, has no interval.
        Intervals are clipped at 0, and at 1 for every metric except phi
        (an entropy that may exceed 1).
        """
        intervals: Dict[str, Tuple[float, float]] = {}
        for attr in METRIC_FIELDS:
            current_value = getattr(metrics, attr)
            if current_value is None:
                continue
            historical_values = [
                value
                for value in (getattr(m, attr) for m in self.measurement_history)
                if value is not None
            ]
            if len(historical_values) <= INTERVAL_MIN_HISTORY:
                continue
            std = np.std(historical_values)
            margin = 1.96 * std / np.sqrt(len(historical_values))
            low = max(0.0, current_value - margin)
            high = current_value + margin
            if attr != "phi":
                high = min(1.0, high)
            intervals[attr] = (float(low), float(high))
        return intervals

    # Weights of the overall score; renormalized over the measured metrics.
    SCORE_WEIGHTS: Dict[str, float] = {
        "phi": 0.2,
        "integration": 0.15,
        "global_accessibility": 0.15,
        "temporal_continuity": 0.1,
        "self_model_accuracy": 0.1,
        "metacognitive_depth": 0.1,
        "workspace_stability": 0.05,
        "attention_focus": 0.05,
        "agency_detection": 0.05,
        "creative_emergence": 0.05,
    }

    # Family label -> (metric, reference threshold) of the threshold summary.
    FAMILY_THRESHOLDS: Dict[str, Tuple[str, float]] = {
        "IIT": ("phi", 2.5),
        "GWT": ("global_accessibility", 0.6),
        "AST": ("attention_focus", 0.7),
        "HOT": ("metacognitive_depth", 0.4),
    }

    def _overall_score(self, metrics: ConsciousnessMetrics) -> Optional[float]:
        """Weighted mean over the measured weighted metrics, None when none."""
        weighted_sum = 0.0
        total_weight = 0.0
        for indicator, weight in self.SCORE_WEIGHTS.items():
            value = getattr(metrics, indicator, None)
            if value is None:
                continue
            weighted_sum += value * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else None

    def analyze_consciousness_level(
        self, metrics: ConsciousnessMetrics
    ) -> Dict[str, Any]:
        """
        Compare the metrics with the reference thresholds.

        The overall score is the weighted mean of the measured metrics (the
        weights renormalized over them) and None when none is measured. A
        metric that was not measured is listed under ``not_measured`` and
        its threshold entry has status ``not_measured``. The confidence is
        None until more than ``CONFIDENCE_MIN_HISTORY`` history entries have
        an overall score.
        """
        analysis: Dict[str, Any] = {
            "overall_consciousness_score": None,
            # Modules operating as designed, NOT phenomenal consciousness
            "architecture_functional": False,
            "confidence": None,
            "key_indicators": {},
            "not_measured": [],
            "theoretical_assessment": {},
            "recommendations": [],
        }

        overall = self._overall_score(metrics)
        analysis["overall_consciousness_score"] = overall
        analysis["not_measured"] = [
            name for name in METRIC_FIELDS if getattr(metrics, name) is None
        ]

        for indicator, threshold in self.consciousness_thresholds.items():
            value = getattr(metrics, indicator, None)
            if value is None:
                analysis["key_indicators"][indicator] = {
                    "value": None,
                    "threshold": threshold,
                    "meets_threshold": None,
                    "status": NOT_MEASURED,
                }
                continue
            analysis["key_indicators"][indicator] = {
                "value": value,
                "threshold": threshold,
                "meets_threshold": value >= threshold,
                "status": "measured",
            }

        critical_indicators_met = sum(
            1
            for entry in analysis["key_indicators"].values()
            if entry["meets_threshold"]
        )
        analysis["architecture_functional"] = bool(
            critical_indicators_met >= 3 and overall is not None and overall > 0.5
        )

        # Confidence from the consistency of the recent scored history
        scored = [
            score
            for score in (self._overall_score(m) for m in self.measurement_history)
            if score is not None
        ]
        if len(scored) > CONFIDENCE_MIN_HISTORY:
            analysis["confidence"] = float(1.0 - np.std(scored[-10:]))

        for family, (metric, threshold) in self.FAMILY_THRESHOLDS.items():
            value = getattr(metrics, metric, None)
            if value is None:
                label = NOT_MEASURED
            elif value > threshold:
                label = ABOVE_THRESHOLD
            else:
                label = BELOW_THRESHOLD
            analysis["theoretical_assessment"][family] = label

        if not analysis["architecture_functional"]:
            for metric, threshold, advice in (
                ("phi", 2.5, "Increase system integration for a higher phi proxy"),
                (
                    "global_accessibility",
                    0.6,
                    "Improve information broadcast mechanisms",
                ),
                (
                    "temporal_continuity",
                    0.8,
                    "Strengthen temporal coherence and memory integration",
                ),
            ):
                value = getattr(metrics, metric, None)
                if value is not None and value < threshold:
                    analysis["recommendations"].append(advice)

        return analysis

    def export_for_publication(self, metrics: ConsciousnessMetrics) -> str:
        """Export the measurements as a JSON report.

        ``phi`` is exported as ``spectral_integration_proxy``; integration
        and differentiation keep their published keys ``system_integration``
        and ``information_differentiation``; a metric with no interval
        exports a null interval.
        """
        analysis = self.analyze_consciousness_level(metrics)

        def entry(attr: str) -> Dict[str, Any]:
            interval = metrics.confidence_intervals.get(attr)
            return {
                "value": getattr(metrics, attr),
                "confidence_interval": list(interval) if interval else None,
            }

        measurements: Dict[str, Any] = {}
        for attr in METRIC_FIELDS:
            if attr == "phi":
                proxy = entry("phi")
                proxy["interpretation"] = (
                    "entropy of the eigenvalue spectrum of the substrate "
                    "correlation matrix; not IIT Phi"
                )
                measurements["spectral_integration_proxy"] = proxy
            else:
                measurements[_EXPORT_KEYS.get(attr, attr)] = entry(attr)

        report = {
            "title": "Consciousness Measurement Report - Multi-Theory Consciousness Framework",
            "timestamp": metrics.measurement_timestamp.isoformat(),
            "methodology": {
                "frameworks": [
                    "Spectral integration proxy",
                    "GWT",
                    "AST",
                    "Predictive Processing",
                ],
                "metrics": 14,
                "neural_units": self.snn_neurons
                + self.lsm_neurons
                + self.htm_columns * 32,
            },
            "availability": metrics.availability,
            "measurements": measurements,
            "analysis": analysis,
            "statistical_metadata": {
                "sample_size": metrics.sample_size,
                "measurement_duration_ms": metrics.measurement_duration_ms,
                "historical_measurements": len(self.measurement_history),
            },
        }

        return json.dumps(report, indent=2, default=str, allow_nan=False)
