#!/usr/bin/env python3
"""
Legacy 14-Indicator Consciousness Metrics
=========================================

The legacy 14-indicator framework: fourteen interaction-level indicators
(global workspace integration, a simplified integrated-information score,
attention, working and episodic memory, metacognition, agency, temporal
continuity, embodiment, affect, social cognition, unified experience,
flexible reasoning and adaptive learning), each scored between 0.0 and 1.0
from an interaction-data dictionary.

This is distinct from the 30-indicator assessment in
``mtc.assessment.assessment``, which scores the Butlin et al. indicators across
eight theories from the theory modules themselves. The legacy framework is
kept for the neural integration layer; its scores are not comparable with
the assessment's.

Every indicator requires the inputs its formula reads. A missing input raises
``MissingSignal``; nothing defaults to a plausible constant. A failed
indicator is recorded as None and left out of the aggregate.
"""

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

INDICATOR_NAMES = (
    "global_workspace_integration",
    "integrated_information_phi",
    "attention_mechanisms",
    "working_memory_capacity",
    "episodic_memory_recall",
    "meta_cognition",
    "agency_autonomy",
    "temporal_continuity",
    "embodiment",
    "affective_processing",
    "social_cognition",
    "unified_experience",
    "flexible_reasoning",
    "adaptive_learning",
)

# Weights of the aggregate: GWT and IIT 1.2, metacognition 1.1, the rest 1.0
# (14.5 in total over the full panel, as before).
INDICATOR_WEIGHTS: Dict[str, float] = {name: 1.0 for name in INDICATOR_NAMES}
INDICATOR_WEIGHTS["global_workspace_integration"] = 1.2
INDICATOR_WEIGHTS["integrated_information_phi"] = 1.2
INDICATOR_WEIGHTS["meta_cognition"] = 1.1


class MissingSignal(ValueError):
    """An indicator input its formula reads was not supplied."""


@dataclass
class ConsciousnessAssessment:
    """A legacy 14-indicator assessment at a point in time.

    Each indicator field is None when that indicator failed to measure;
    ``failed_indicators`` maps the failed names to their errors.
    """

    timestamp: datetime
    global_workspace_integration: Optional[float] = None
    integrated_information_phi: Optional[float] = None
    attention_mechanisms: Optional[float] = None
    working_memory_capacity: Optional[float] = None
    episodic_memory_recall: Optional[float] = None
    meta_cognition: Optional[float] = None
    agency_autonomy: Optional[float] = None
    temporal_continuity: Optional[float] = None
    embodiment: Optional[float] = None
    affective_processing: Optional[float] = None
    social_cognition: Optional[float] = None
    unified_experience: Optional[float] = None
    flexible_reasoning: Optional[float] = None
    adaptive_learning: Optional[float] = None
    failed_indicators: Dict[str, str] = field(default_factory=dict)

    @property
    def overall_consciousness_score(self) -> Optional[float]:
        """Weighted mean over the indicators that measured; None if none did."""
        total = 0.0
        weight_sum = 0.0
        for name in INDICATOR_NAMES:
            score = getattr(self, name)
            if score is None:
                continue
            weight = INDICATOR_WEIGHTS[name]
            total += weight * score
            weight_sum += weight
        if weight_sum == 0.0:
            return None
        return total / weight_sum


class ConsciousnessIndicator(ABC):
    """Base class for the legacy indicators."""

    @abstractmethod
    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Measure this indicator given interaction data."""
        pass

    @abstractmethod
    def get_requirements(self) -> List[str]:
        """List data requirements for this indicator."""
        pass

    @staticmethod
    def _require(data: Dict[str, Any], *paths: str) -> List[Any]:
        """The values at the dotted ``paths``, in order.

        A path is missing when any key on it is absent or its value is None.
        Raises ``MissingSignal`` naming every missing path.
        """
        values: List[Any] = []
        missing: List[str] = []
        for path in paths:
            node: Any = data
            for key in path.split("."):
                if isinstance(node, dict) and node.get(key) is not None:
                    node = node[key]
                else:
                    node = None
                    break
            if node is None:
                missing.append(path)
            values.append(node)
        if missing:
            raise MissingSignal(f"Missing required data: {missing}")
        return values


class GlobalWorkspaceIntegration(ConsciousnessIndicator):
    """
    Measures information integration across different cognitive modules.
    Based on Global Workspace Theory (Baars, 1988).
    """

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess cross-modal information integration."""
        pattern, tokens, retrieved, focus_changes, latency = self._require(
            interaction_data,
            "neural_activity.activation_pattern",
            "llm_processing.token_attention",
            "memory_access.retrieved_memories",
            "attention_focus.focus_changes",
            "broadcast_latency_ms",
        )

        integration = 0.0

        # 1. Cross-system activation correlation. Both vectors need more than
        # one value; they are truncated to the shorter, and an undefined
        # correlation (a constant vector) contributes nothing.
        pattern = np.ravel(np.asarray(pattern, dtype=float))
        tokens = np.ravel(np.asarray(tokens, dtype=float))
        if len(pattern) > 1 and len(tokens) > 1:
            n = min(len(pattern), len(tokens))
            with np.errstate(divide="ignore", invalid="ignore"):
                correlation = np.corrcoef(pattern[:n], tokens[:n])[0, 1]
            if np.isfinite(correlation):
                integration += abs(float(correlation)) * 0.3

        # 2. Memory-attention coupling
        coupling = min(1.0, (len(retrieved) + float(focus_changes)) / 10)
        integration += coupling * 0.3

        # 3. Broadcasting efficiency
        efficiency = max(0.0, 1.0 - (float(latency) / 100))
        integration += efficiency * 0.4

        return min(1.0, integration)

    def get_requirements(self) -> List[str]:
        return [
            "neural_activity",
            "attention_focus",
            "memory_access",
            "llm_processing",
            "broadcast_latency_ms",
        ]


class IntegratedInformationPhi(ConsciousnessIndicator):
    """
    A simplified integrated-information score over component connectivity.
    Based on Integrated Information Theory (Tononi, 2008); this is not a Phi
    computation.
    """

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Connectivity among active components times partition independence."""
        components, links, partition_independence = self._require(
            interaction_data,
            "system_state.active_components",
            "causal_connections.active_links",
            "system_state.partition_independence",
        )

        elements = {str(name) for name in components}
        num_elements = len(elements)
        if num_elements <= 1:
            return 0.0

        # Links count as unordered pairs of distinct active components, over
        # the n (n - 1) / 2 possible pairs.
        pairs = set()
        for link in links:
            try:
                a, b = (str(end) for end in link)
            except (TypeError, ValueError):
                continue
            if a != b and a in elements and b in elements:
                pairs.add(frozenset((a, b)))
        possible = num_elements * (num_elements - 1) / 2
        connectivity = min(1.0, len(pairs) / possible)

        phi = connectivity * float(partition_independence) * 0.8
        return min(1.0, phi)

    def get_requirements(self) -> List[str]:
        return ["system_state", "causal_connections"]


class AttentionMechanisms(ConsciousnessIndicator):
    """Measures selective attention and focus capabilities."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess attention mechanisms."""
        sustained, selective, divided, switching = self._require(
            interaction_data,
            "attention_focus.sustained_attention",
            "attention_focus.selective_focus",
            "attention_focus.divided_attention",
            "attention_focus.attention_switching",
        )
        score = sustained * 0.3 + selective * 0.3 + divided * 0.2 + switching * 0.2
        return min(1.0, score)

    def get_requirements(self) -> List[str]:
        return ["attention_focus"]


class WorkingMemoryCapacity(ConsciousnessIndicator):
    """Measures working memory span and manipulation."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess working memory capacity."""
        items_held, manipulations = self._require(
            interaction_data,
            "working_memory.items_in_memory",
            "working_memory.successful_manipulations",
        )
        # Miller's 7+-2 rule normalized (9 is its upper bound)
        capacity_score = min(1.0, items_held / 9)
        manipulation_score = min(1.0, manipulations / 5)
        return capacity_score * 0.6 + manipulation_score * 0.4

    def get_requirements(self) -> List[str]:
        return ["working_memory"]


class EpisodicMemoryRecall(ConsciousnessIndicator):
    """Measures autobiographical memory and experience recall."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess episodic memory recall."""
        attempts, successes, temporal_accuracy, detail_richness = self._require(
            interaction_data,
            "episodic_memory.recall_attempts",
            "episodic_memory.successful_recalls",
            "episodic_memory.temporal_ordering_score",
            "episodic_memory.detail_richness",
        )
        accuracy = successes / max(1, attempts)
        return accuracy * 0.4 + temporal_accuracy * 0.3 + detail_richness * 0.3

    def get_requirements(self) -> List[str]:
        return ["episodic_memory"]


class MetaCognition(ConsciousnessIndicator):
    """Measures thinking about thinking - self-awareness of cognitive processes."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess metacognitive abilities."""
        monitoring, reflection, error_detection, calibration = self._require(
            interaction_data,
            "metacognition.self_monitoring_score",
            "metacognition.reflection_depth",
            "metacognition.error_self_detection",
            "metacognition.confidence_calibration",
        )
        return (
            monitoring * 0.3
            + reflection * 0.3
            + error_detection * 0.2
            + calibration * 0.2
        )

    def get_requirements(self) -> List[str]:
        return ["metacognition"]


class AgencyAutonomy(ConsciousnessIndicator):
    """Measures goal-directed behavior and autonomous decision-making."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess agency and autonomy."""
        goals, autonomous_decisions, self_initiated = self._require(
            interaction_data,
            "agency.self_generated_goals",
            "agency.autonomous_decision_ratio",
            "agency.self_initiated_actions",
        )
        goal_score = min(1.0, goals / 3)
        return goal_score * 0.4 + autonomous_decisions * 0.3 + self_initiated * 0.3

    def get_requirements(self) -> List[str]:
        return ["agency"]


class TemporalContinuity(ConsciousnessIndicator):
    """Measures sense of self continuity across time."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess temporal continuity of identity."""
        identity_score, narrative, future_projection = self._require(
            interaction_data,
            "temporal_continuity.identity_consistency",
            "temporal_continuity.narrative_coherence",
            "temporal_continuity.future_planning_score",
        )
        return identity_score * 0.4 + narrative * 0.3 + future_projection * 0.3

    def get_requirements(self) -> List[str]:
        return ["temporal_continuity"]


class Embodiment(ConsciousnessIndicator):
    """Measures sense of having boundaries and presence."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess embodiment and boundaries."""
        boundaries, presence, distinction = self._require(
            interaction_data,
            "embodiment.boundary_awareness",
            "embodiment.spatial_presence",
            "embodiment.self_other_distinction",
        )
        return boundaries * 0.3 + presence * 0.3 + distinction * 0.4

    def get_requirements(self) -> List[str]:
        return ["embodiment"]


class AffectiveProcessing(ConsciousnessIndicator):
    """Measures emotional processing and regulation."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess affective processing capabilities."""
        recognition, regulation, coherence, empathy = self._require(
            interaction_data,
            "affective_processing.emotion_recognition_accuracy",
            "affective_processing.emotion_regulation_score",
            "affective_processing.emotional_coherence",
            "affective_processing.empathy_score",
        )
        return (
            recognition * 0.25 + regulation * 0.25 + coherence * 0.25 + empathy * 0.25
        )

    def get_requirements(self) -> List[str]:
        return ["affective_processing"]


class SocialCognition(ConsciousnessIndicator):
    """Measures theory of mind and social understanding."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess social cognitive abilities."""
        tom_score, perspective, context = self._require(
            interaction_data,
            "social_cognition.theory_of_mind_score",
            "social_cognition.perspective_taking",
            "social_cognition.social_context_score",
        )
        return tom_score * 0.4 + perspective * 0.3 + context * 0.3

    def get_requirements(self) -> List[str]:
        return ["social_cognition"]


class UnifiedExperience(ConsciousnessIndicator):
    """Measures unity of conscious experience."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess unified experience."""
        binding, coherence, gestalt = self._require(
            interaction_data,
            "unified_experience.feature_binding_score",
            "unified_experience.experience_coherence",
            "unified_experience.gestalt_formation",
        )
        return binding * 0.4 + coherence * 0.3 + gestalt * 0.3

    def get_requirements(self) -> List[str]:
        return ["unified_experience"]


class FlexibleReasoning(ConsciousnessIndicator):
    """Measures abstract reasoning and cognitive flexibility."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess reasoning flexibility."""
        abstract, flexibility, creativity = self._require(
            interaction_data,
            "flexible_reasoning.abstract_reasoning_score",
            "flexible_reasoning.cognitive_flexibility",
            "flexible_reasoning.creative_solutions",
        )
        return abstract * 0.4 + flexibility * 0.3 + creativity * 0.3

    def get_requirements(self) -> List[str]:
        return ["flexible_reasoning"]


class AdaptiveLearning(ConsciousnessIndicator):
    """Measures learning from experience and adaptation."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess adaptive learning capabilities."""
        learning_rate, transfer, novelty = self._require(
            interaction_data,
            "adaptive_learning.learning_efficiency",
            "adaptive_learning.knowledge_transfer_score",
            "adaptive_learning.novelty_adaptation",
        )
        return learning_rate * 0.4 + transfer * 0.3 + novelty * 0.3

    def get_requirements(self) -> List[str]:
        return ["adaptive_learning"]


def _summary(values: List[float]) -> Dict[str, Any]:
    """Mean, spread, slope and range of the measured values, None when empty."""
    if not values:
        return {
            "mean": None,
            "std": None,
            "trend": None,
            "min": None,
            "max": None,
            "measured_samples": 0,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "trend": (
            float(np.polyfit(range(len(values)), values, 1)[0])
            if len(values) > 1
            else 0.0
        ),
        "min": min(values),
        "max": max(values),
        "measured_samples": len(values),
    }


class ConsciousnessMetrics:
    """
    The legacy 14-indicator framework.
    Coordinates the 14 indicators and aggregates the ones that measured.
    """

    def __init__(self):
        """Initialize the 14 legacy indicators."""
        self.indicators = {
            "global_workspace_integration": GlobalWorkspaceIntegration(),
            "integrated_information_phi": IntegratedInformationPhi(),
            "attention_mechanisms": AttentionMechanisms(),
            "working_memory_capacity": WorkingMemoryCapacity(),
            "episodic_memory_recall": EpisodicMemoryRecall(),
            "meta_cognition": MetaCognition(),
            "agency_autonomy": AgencyAutonomy(),
            "temporal_continuity": TemporalContinuity(),
            "embodiment": Embodiment(),
            "affective_processing": AffectiveProcessing(),
            "social_cognition": SocialCognition(),
            "unified_experience": UnifiedExperience(),
            "flexible_reasoning": FlexibleReasoning(),
            "adaptive_learning": AdaptiveLearning(),
        }

        # Historical assessments for tracking
        self.assessment_history: List[ConsciousnessAssessment] = []

        logger.info(
            f"Legacy {len(self.indicators)}-indicator consciousness metrics "
            f"initialized"
        )

    async def assess_consciousness(
        self, interaction_data: Dict[str, Any]
    ) -> ConsciousnessAssessment:
        """
        Perform a legacy 14-indicator assessment.

        Args:
            interaction_data: Dictionary containing the measurement data

        Returns:
            ConsciousnessAssessment; an indicator that failed (missing input,
            error or a non-finite score) is None and named in
            ``failed_indicators``
        """
        results: Dict[str, Optional[float]] = {}
        failed: Dict[str, str] = {}

        for name, indicator in self.indicators.items():
            try:
                score = float(await indicator.measure(interaction_data))
                if not math.isfinite(score):
                    raise ValueError(f"non-finite score {score}")
                results[name] = min(1.0, max(0.0, score))
            except Exception as e:
                logger.debug(f"Legacy indicator {name} not measured: {e}")
                results[name] = None
                failed[name] = str(e)

        assessment = ConsciousnessAssessment(
            timestamp=datetime.now(), failed_indicators=failed, **results
        )
        self.assessment_history.append(assessment)

        overall = assessment.overall_consciousness_score
        logger.info(
            "Legacy 14-indicator assessment complete: overall score = "
            + (f"{overall:.3f}" if overall is not None else "insufficient_data")
            + f" ({len(self.indicators) - len(failed)} of {len(self.indicators)}"
            " measured)"
        )

        return assessment

    def get_longitudinal_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze the legacy indicators over time.

        Unmeasured (None) values are left out; each block reports how many
        measured samples it summarizes.

        Args:
            hours: Number of hours to analyze

        Returns:
            Dictionary with trend analysis
        """
        if not self.assessment_history:
            return {"error": "No historical data available"}

        cutoff = datetime.now().timestamp() - (hours * 3600)
        recent = [
            a for a in self.assessment_history if a.timestamp.timestamp() > cutoff
        ]

        if not recent:
            return {"error": "No data in specified time range"}

        trends = {}
        for name in INDICATOR_NAMES:
            values = [getattr(a, name) for a in recent if getattr(a, name) is not None]
            trends[name] = _summary(values)

        overall_scores = [
            a.overall_consciousness_score
            for a in recent
            if a.overall_consciousness_score is not None
        ]
        trends["overall"] = _summary(overall_scores)

        return {
            "time_range_hours": hours,
            "num_assessments": len(recent),
            "trends": trends,
            "latest_score": recent[-1].overall_consciousness_score,
        }

    def get_requirements(self) -> Dict[str, List[str]]:
        """Get all data requirements for full assessment."""
        requirements = {}
        for name, indicator in self.indicators.items():
            requirements[name] = indicator.get_requirements()
        return requirements


def _format_score(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "insufficient_data"


# Example usage and testing
async def demo_consciousness_metrics():
    """Run the legacy 14-indicator framework on synthetic interaction data."""
    print("Legacy 14-indicator consciousness metrics")
    print("=" * 60)

    metrics = ConsciousnessMetrics()

    # Synthetic interaction data covering every indicator input
    mock_data = {
        "neural_activity": {"activation_pattern": np.random.rand(100).tolist()},
        "attention_focus": {
            "sustained_attention": 0.7,
            "selective_focus": 0.8,
            "divided_attention": 0.5,
            "attention_switching": 0.6,
            "focus_changes": 3,
        },
        "memory_access": {"retrieved_memories": ["mem1", "mem2", "mem3"]},
        "llm_processing": {"token_attention": np.random.rand(100).tolist()},
        "system_state": {
            "active_components": ["snn", "htm", "lsm", "llm", "memory"],
            "partition_independence": 0.6,
        },
        "causal_connections": {
            "active_links": [("snn", "lsm"), ("lsm", "htm"), ("htm", "memory")]
        },
        "working_memory": {"items_in_memory": 6, "successful_manipulations": 3},
        "episodic_memory": {
            "recall_attempts": 5,
            "successful_recalls": 4,
            "temporal_ordering_score": 0.8,
            "detail_richness": 0.7,
        },
        "metacognition": {
            "self_monitoring_score": 0.6,
            "reflection_depth": 0.7,
            "error_self_detection": 0.5,
            "confidence_calibration": 0.8,
        },
        "agency": {
            "self_generated_goals": 2,
            "autonomous_decision_ratio": 0.6,
            "self_initiated_actions": 0.5,
        },
        "temporal_continuity": {
            "identity_consistency": 0.8,
            "narrative_coherence": 0.7,
            "future_planning_score": 0.6,
        },
        "embodiment": {
            "boundary_awareness": 0.7,
            "spatial_presence": 0.6,
            "self_other_distinction": 0.8,
        },
        "affective_processing": {
            "emotion_recognition_accuracy": 0.8,
            "emotion_regulation_score": 0.7,
            "emotional_coherence": 0.8,
            "empathy_score": 0.6,
        },
        "social_cognition": {
            "theory_of_mind_score": 0.7,
            "perspective_taking": 0.6,
            "social_context_score": 0.8,
        },
        "unified_experience": {
            "feature_binding_score": 0.7,
            "experience_coherence": 0.8,
            "gestalt_formation": 0.6,
        },
        "flexible_reasoning": {
            "abstract_reasoning_score": 0.7,
            "cognitive_flexibility": 0.6,
            "creative_solutions": 0.5,
        },
        "adaptive_learning": {
            "learning_efficiency": 0.8,
            "knowledge_transfer_score": 0.6,
            "novelty_adaptation": 0.7,
        },
        "broadcast_latency_ms": 45,
    }

    assessment = await metrics.assess_consciousness(mock_data)

    print("\nAssessment results:")
    print(f"  Timestamp: {assessment.timestamp}")
    print("\n  Individual indicators:")
    for name in INDICATOR_NAMES:
        label = name.replace("_", " ").title()
        print(f"  - {label}: {_format_score(getattr(assessment, name))}")

    print("\n  Overall score: " + _format_score(assessment.overall_consciousness_score))

    for _ in range(5):
        await asyncio.sleep(0.1)
        await metrics.assess_consciousness(mock_data)

    analysis = metrics.get_longitudinal_analysis(hours=1)

    print("\nLongitudinal analysis (last hour):")
    print(f"  Assessments: {analysis['num_assessments']}")
    print(f"  Latest score: {_format_score(analysis['latest_score'])}")
    print(f"  Overall trend: {analysis['trends']['overall']['trend']}")

    print("\nLegacy metrics demo complete")


if __name__ == "__main__":
    asyncio.run(demo_consciousness_metrics())
