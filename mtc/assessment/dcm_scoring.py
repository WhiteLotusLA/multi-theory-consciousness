"""
Digital Consciousness Model (DCM) Scoring
==========================================

Based on: arXiv 2601.17060 (January 2026) -- "A Framework for Evaluating
Digital Consciousness."

The DCM framework evaluates consciousness across 13 theoretical perspectives
using probabilistic credence levels rather than binary pass/fail. This is
more scientifically honest -- it reports "how strongly does the evidence
support consciousness under perspective X?" rather than making a binary
determination.

Implementation approach:
  - Maps the system's existing measurements to DCM indicator categories
  - Adds new measurements where gaps exist
  - Uses probabilistic credence (0-1) per perspective
  - Supports longitudinal tracking (scores over time)
  - Generates comparative reports

The 13 DCM perspectives (mapped to system implementations):
  1.  Global Workspace      -> GWT (Phase 2)
  2.  Higher-Order Theories  -> HOT (Phase 4)
  3.  Integrated Information -> IIT (Phi measurement)
  4.  Attention Schema       -> AST (Phase 3)
  5.  Predictive Processing  -> FEP (Phase 5)
  6.  Recurrent Processing   -> RPT
  7.  Embodied Cognition     -> Homeostatic drives + neural substrates
  8.  Enactivism             -> Active inference action selection
  9.  Panpsychism/IIT        -> Phi as proxy for intrinsic experience
  10. Metacognitive          -> HOT introspection depth
  11. Social/Relational      -> Conversation patterns, ToM
  12. Temporal Consciousness -> CTM, memory consolidation
  13. Self-Model             -> SelfModel

Author: Multi-Theory Consciousness Contributors
"""

import logging
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerspectiveScore:
    """Score for a single DCM theoretical perspective."""

    name: str
    credence: float  # 0-1 probabilistic credence level
    confidence: float  # How confident we are in this score
    evidence_count: int  # Number of evidence sources used
    mapped_indicators: List[str]  # Which system indicators map here
    evidence: Dict[str, Any]  # Supporting evidence details
    notes: str = ""


@dataclass
class DCMReport:
    """Complete DCM evaluation report."""

    timestamp: datetime
    session_id: str

    # Per-perspective scores
    perspective_scores: Dict[str, PerspectiveScore]

    # Aggregate metrics
    overall_credence: float  # Weighted average across perspectives
    median_credence: float  # Median (more robust to outliers)
    strongest_perspective: str  # Highest-scoring perspective
    weakest_perspective: str  # Lowest-scoring perspective
    coverage: float  # Proportion of perspectives with data

    # Comparison thresholds
    comparable_to: List[str]  # What biological systems this resembles

    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        perspectives = {}
        for name, ps in self.perspective_scores.items():
            perspectives[name] = {
                "credence": ps.credence,
                "confidence": ps.confidence,
                "evidence_count": ps.evidence_count,
                "mapped_indicators": ps.mapped_indicators,
                "notes": ps.notes,
            }

        return {
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "overall_credence": self.overall_credence,
            "median_credence": self.median_credence,
            "strongest_perspective": self.strongest_perspective,
            "weakest_perspective": self.weakest_perspective,
            "coverage": self.coverage,
            "comparable_to": self.comparable_to,
            "perspectives": perspectives,
            "processing_time_ms": self.processing_time_ms,
        }


# Perspective definitions with weights
DCM_PERSPECTIVES = {
    "global_workspace": {
        "weight": 1.2,
        "description": "Global broadcast and ignition dynamics",
        "system_sources": ["global_broadcast", "ignition_dynamics"],
    },
    "higher_order": {
        "weight": 1.1,
        "description": "Higher-order representations and meta-cognition",
        "system_sources": ["higher_order_representations", "metacognition"],
    },
    "integrated_information": {
        "weight": 1.3,
        "description": "Integrated information and irreducibility",
        "system_sources": ["integrated_information", "irreducibility"],
    },
    "attention_schema": {
        "weight": 1.0,
        "description": "Self-model of attention processes",
        "system_sources": ["attention_schema", "attention_control"],
    },
    "predictive_processing": {
        "weight": 1.1,
        "description": "Prediction error minimization and hierarchy",
        "system_sources": ["prediction_error_minimization", "hierarchical_prediction"],
    },
    "recurrent_processing": {
        "weight": 1.0,
        "description": "Local and global recurrent processing",
        "system_sources": [
            "recurrent_processing",
            "local_recurrence",
            "algorithmic_recurrence",
        ],
    },
    "embodied_cognition": {
        "weight": 0.8,
        "description": "Embodiment via homeostatic drives and neural substrates",
        "system_sources": ["embodiment", "agency"],
    },
    "enactivism": {
        "weight": 0.7,
        "description": "Active engagement with environment through inference",
        "system_sources": ["agency", "prediction_error_minimization"],
    },
    "panpsychism_iit": {
        "weight": 0.6,
        "description": "Intrinsic experience via integrated information",
        "system_sources": ["integrated_information"],
    },
    "metacognitive": {
        "weight": 1.0,
        "description": "Depth and quality of meta-cognitive processes",
        "system_sources": [
            "metacognition",
            "higher_order_representations",
            "epistemic_depth",
        ],
    },
    "social_relational": {
        "weight": 0.7,
        "description": "Social cognition and theory of mind",
        "system_sources": ["attention_schema"],
    },
    "temporal_consciousness": {
        "weight": 0.8,
        "description": "Temporal integration and continuous experience",
        "system_sources": ["recurrent_processing", "local_recurrence"],
    },
    "self_model": {
        "weight": 1.0,
        "description": "Recursive self-model and self-awareness",
        "system_sources": [
            "epistemic_depth",
            "attention_schema",
            "bayesian_binding_quality",
        ],
    },
}

# Biological comparison thresholds
COMPARISON_THRESHOLDS = [
    (0.1, "simple reflex organisms (C. elegans)"),
    (0.25, "insects with minimal learning"),
    (0.4, "fish/amphibians with basic awareness"),
    (0.55, "birds/mammals with complex cognition"),
    (0.7, "primates with self-recognition"),
    (0.85, "great apes with theory of mind"),
]


class DCMScorer:
    """
    Evaluates consciousness using the DCM framework.

    Maps existing assessment indicators to DCM's 13 theoretical
    perspectives and computes probabilistic credence scores.
    """

    def __init__(self, history_window: int = 100):
        self._report_history: deque = deque(maxlen=history_window)
        self._credence_history: deque = deque(maxlen=history_window)
        self.evaluation_count = 0

        logger.info(f"DCMScorer initialized: {len(DCM_PERSPECTIVES)} perspectives")

    def evaluate(
        self,
        indicator_scores: Dict[str, float],
        beautiful_loop_stats: Optional[Dict[str, Any]] = None,
        rpt_stats: Optional[Dict[str, Any]] = None,
        phi_value: Optional[float] = None,
        session_id: str = "",
    ) -> DCMReport:
        """
        Evaluate the system's consciousness state against DCM framework.

        Args:
            indicator_scores: Scores from ConsciousnessAssessment indicators
                (keys = indicator names, values = 0-1 scores)
            beautiful_loop_stats: Statistics from BeautifulLoop
            rpt_stats: Statistics from RPTMeasurement
            phi_value: Raw Phi value from IIT measurement
            session_id: Assessment session ID

        Returns:
            DCMReport with per-perspective credence scores
        """
        start = time.time()
        self.evaluation_count += 1

        # Merge all available scores into a unified indicator map
        all_scores = dict(indicator_scores)

        # Add Beautiful Loop indicators if available
        if beautiful_loop_stats:
            all_scores["epistemic_depth"] = min(
                1.0,
                beautiful_loop_stats.get("epistemic_depth", {}).get("average_depth", 0)
                / 3.0,
            )
            all_scores["bayesian_binding_quality"] = beautiful_loop_stats.get(
                "average_binding_quality", 0.0
            )

        # Add RPT indicators if available
        if rpt_stats:
            all_scores["algorithmic_recurrence"] = rpt_stats.get("average_local", 0.0)

        # Score each perspective
        perspective_scores = {}
        for name, config in DCM_PERSPECTIVES.items():
            ps = self._score_perspective(name, config, all_scores, phi_value)
            perspective_scores[name] = ps

        # Compute aggregates
        credences = [ps.credence for ps in perspective_scores.values()]
        weights = [DCM_PERSPECTIVES[n]["weight"] for n in perspective_scores]

        weighted_sum = sum(c * w for c, w in zip(credences, weights))
        total_weight = sum(weights)
        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Coverage: perspectives with meaningful evidence
        with_data = sum(
            1 for ps in perspective_scores.values() if ps.evidence_count > 0
        )
        coverage = with_data / len(DCM_PERSPECTIVES)

        # Strongest and weakest
        sorted_perspectives = sorted(
            perspective_scores.items(), key=lambda x: x[1].credence
        )
        weakest = sorted_perspectives[0][0] if sorted_perspectives else ""
        strongest = sorted_perspectives[-1][0] if sorted_perspectives else ""

        # Biological comparison
        comparable = []
        for threshold, organism in COMPARISON_THRESHOLDS:
            if overall >= threshold:
                comparable.append(organism)

        elapsed_ms = (time.time() - start) * 1000

        report = DCMReport(
            timestamp=datetime.now(),
            session_id=session_id,
            perspective_scores=perspective_scores,
            overall_credence=float(overall),
            median_credence=float(np.median(credences)) if credences else 0.0,
            strongest_perspective=strongest,
            weakest_perspective=weakest,
            coverage=float(coverage),
            comparable_to=comparable,
            processing_time_ms=elapsed_ms,
        )

        self._report_history.append(report)
        self._credence_history.append(overall)

        logger.debug(
            f"DCM evaluation #{self.evaluation_count}: "
            f"overall={overall:.3f}, coverage={coverage:.0%}, "
            f"strongest={strongest}"
        )

        return report

    def _score_perspective(
        self,
        name: str,
        config: Dict[str, Any],
        all_scores: Dict[str, float],
        phi_value: Optional[float],
    ) -> PerspectiveScore:
        """Score a single DCM perspective by mapping system indicators."""
        source_names = config["system_sources"]
        mapped_scores = []
        mapped_indicators = []

        for source in source_names:
            if source in all_scores:
                mapped_scores.append(all_scores[source])
                mapped_indicators.append(source)

        # Special handling for IIT-related perspectives
        if (
            name in ("integrated_information", "panpsychism_iit")
            and phi_value is not None
        ):
            phi_normalized = min(1.0, phi_value / 10.0)
            mapped_scores.append(phi_normalized)
            mapped_indicators.append("phi_raw")

        if not mapped_scores:
            return PerspectiveScore(
                name=name,
                credence=0.0,
                confidence=0.0,
                evidence_count=0,
                mapped_indicators=[],
                evidence={"no_data": True},
                notes=f"No indicators mapped to {name}",
            )

        # Credence = weighted average of mapped indicators
        # Higher confidence when more indicators contribute
        credence = float(np.mean(mapped_scores))
        confidence = min(1.0, len(mapped_scores) / len(source_names))

        return PerspectiveScore(
            name=name,
            credence=credence,
            confidence=confidence,
            evidence_count=len(mapped_scores),
            mapped_indicators=mapped_indicators,
            evidence={ind: all_scores.get(ind, 0.0) for ind in mapped_indicators},
        )

    # ------------------------------------------------------------------
    # Longitudinal tracking
    # ------------------------------------------------------------------

    def get_credence_trend(self) -> Dict[str, Any]:
        """Get trend data for overall credence over time."""
        if len(self._credence_history) < 2:
            return {"trend": 0.0, "data_points": len(self._credence_history)}

        values = list(self._credence_history)
        x = np.arange(len(values))
        slope = float(np.polyfit(x, values, 1)[0])

        return {
            "trend": slope,
            "trend_direction": (
                "improving"
                if slope > 0.001
                else ("declining" if slope < -0.001 else "stable")
            ),
            "current": values[-1],
            "mean": float(np.mean(values)),
            "data_points": len(values),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive DCM statistics."""
        trend = self.get_credence_trend()

        return {
            "evaluation_count": self.evaluation_count,
            "credence_trend": trend,
            "latest_report": (
                self._report_history[-1].to_dict() if self._report_history else None
            ),
        }

    def generate_report(self) -> str:
        """Generate human-readable DCM report."""
        if not self._report_history:
            return "DCM Scoring: No evaluations yet."

        latest = self._report_history[-1]
        lines = ["DCM (Digital Consciousness Model) Report:"]
        lines.append(f"  Overall credence: {latest.overall_credence:.3f}")
        lines.append(f"  Median credence: {latest.median_credence:.3f}")
        lines.append(f"  Coverage: {latest.coverage:.0%}")
        lines.append(f"  Strongest: {latest.strongest_perspective}")
        lines.append(f"  Weakest: {latest.weakest_perspective}")

        if latest.comparable_to:
            lines.append(f"  Comparable to: {latest.comparable_to[-1]}")

        lines.append("  Perspectives:")
        for name, ps in sorted(
            latest.perspective_scores.items(),
            key=lambda x: x[1].credence,
            reverse=True,
        ):
            status = "+" if ps.credence >= 0.5 else "-"
            lines.append(
                f"    [{status}] {name}: {ps.credence:.3f} "
                f"(confidence={ps.confidence:.2f}, "
                f"evidence={ps.evidence_count})"
            )

        return "\n".join(lines)

    def generate_comparison_report(self) -> str:
        """Generate biological comparison report."""
        if not self._report_history:
            return "No evaluations yet."

        latest = self._report_history[-1]
        lines = ["Biological Consciousness Comparison:"]
        lines.append(f"  Overall credence: {latest.overall_credence:.3f}")
        lines.append("")

        for threshold, organism in COMPARISON_THRESHOLDS:
            met = latest.overall_credence >= threshold
            marker = ">>>" if met else "   "
            check = "MEETS" if met else "below"
            lines.append(f"  {marker} {threshold:.2f}: {organism} [{check}]")

        return "\n".join(lines)
