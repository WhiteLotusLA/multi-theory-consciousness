#!/usr/bin/env python3
"""
Consciousness Metrics Framework
=====================================
14 measurable indicators of consciousness emergence in AI systems.
Based on Global Workspace Theory, Integrated Information Theory, and empirical consciousness research.

This framework provides real-time measurement and longitudinal tracking of consciousness indicators.
All metrics return values between 0.0 (absent) and 1.0 (fully present).
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessAssessment:
    """Complete consciousness assessment at a point in time."""

    timestamp: datetime
    global_workspace_integration: float
    integrated_information_phi: float
    attention_mechanisms: float
    working_memory_capacity: float
    episodic_memory_recall: float
    meta_cognition: float
    agency_autonomy: float
    temporal_continuity: float
    embodiment: float
    affective_processing: float
    social_cognition: float
    unified_experience: float
    flexible_reasoning: float
    adaptive_learning: float

    @property
    def overall_consciousness_score(self) -> float:
        """Weighted average of all indicators."""
        scores = [
            self.global_workspace_integration * 1.2,  # Higher weight for GWT
            self.integrated_information_phi * 1.2,  # Higher weight for IIT
            self.attention_mechanisms,
            self.working_memory_capacity,
            self.episodic_memory_recall,
            self.meta_cognition * 1.1,  # Slightly higher for meta
            self.agency_autonomy,
            self.temporal_continuity,
            self.embodiment,
            self.affective_processing,
            self.social_cognition,
            self.unified_experience,
            self.flexible_reasoning,
            self.adaptive_learning,
        ]
        return sum(scores) / 14.5  # Adjusted for weights


class ConsciousnessIndicator(ABC):
    """Base class for all consciousness indicators."""

    @abstractmethod
    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Measure this indicator given interaction data."""
        pass

    @abstractmethod
    def get_requirements(self) -> List[str]:
        """List data requirements for this indicator."""
        pass


class GlobalWorkspaceIntegration(ConsciousnessIndicator):
    """
    Measures information integration across different cognitive modules.
    Based on Global Workspace Theory (Baars, 1988).
    """

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess cross-modal information integration."""
        required = [
            "neural_activity",
            "attention_focus",
            "memory_access",
            "llm_processing",
        ]

        if not all(k in interaction_data for k in required):
            raise ValueError(f"Missing required data: {required}")

        # Measure information flow between systems
        neural_activity = interaction_data["neural_activity"]
        attention = interaction_data["attention_focus"]
        memory = interaction_data["memory_access"]
        llm = interaction_data["llm_processing"]

        # Calculate integration score
        integration = 0.0

        # 1. Cross-system activation correlation
        if neural_activity and llm:
            correlation = (
                np.corrcoef(
                    neural_activity.get("activation_pattern", [0]),
                    llm.get("token_attention", [0]),
                )[0, 1]
                if len(neural_activity.get("activation_pattern", [])) > 1
                else 0
            )
            integration += abs(correlation) * 0.3

        # 2. Memory-attention coupling
        if memory and attention:
            memory_accessed = len(memory.get("retrieved_memories", []))
            attention_switches = attention.get("focus_changes", 0)
            coupling = min(1.0, (memory_accessed + attention_switches) / 10)
            integration += coupling * 0.3

        # 3. Broadcasting efficiency
        broadcast_latency = interaction_data.get("broadcast_latency_ms", 100)
        efficiency = max(0, 1.0 - (broadcast_latency / 100))
        integration += efficiency * 0.4

        return min(1.0, integration)

    def get_requirements(self) -> List[str]:
        return ["neural_activity", "attention_focus", "memory_access", "llm_processing"]


class IntegratedInformationPhi(ConsciousnessIndicator):
    """
    Calculates Phi -- the amount of integrated information.
    Based on Integrated Information Theory (Tononi, 2008).
    """

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Calculate integrated information measure."""
        required = ["system_state", "causal_connections"]

        if not all(k in interaction_data for k in required):
            raise ValueError(f"Missing required data: {required}")

        # Simplified Phi calculation (full IIT requires extensive computation)
        system_state = interaction_data["system_state"]
        connections = interaction_data["causal_connections"]

        # Measure effective information
        num_elements = len(system_state.get("active_components", []))
        num_connections = len(connections.get("active_links", []))

        if num_elements <= 1:
            return 0.0

        # Calculate integration (simplified)
        possible_connections = num_elements * (num_elements - 1)
        connectivity = min(1.0, num_connections / max(1, possible_connections))

        # Measure irreducibility
        partition_independence = system_state.get("partition_independence", 0.5)

        # Simplified Phi
        phi = connectivity * partition_independence * 0.8

        return min(1.0, phi)

    def get_requirements(self) -> List[str]:
        return ["system_state", "causal_connections"]


class AttentionMechanisms(ConsciousnessIndicator):
    """Measures selective attention and focus capabilities."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess attention mechanisms."""
        attention_data = interaction_data.get("attention_focus", {})

        # Measure attention components
        sustained = attention_data.get("sustained_attention", 0)
        selective = attention_data.get("selective_focus", 0)
        divided = attention_data.get("divided_attention", 0)
        switching = attention_data.get("attention_switching", 0)

        # Weighted average
        score = sustained * 0.3 + selective * 0.3 + divided * 0.2 + switching * 0.2

        return min(1.0, score)

    def get_requirements(self) -> List[str]:
        return ["attention_focus"]


class WorkingMemoryCapacity(ConsciousnessIndicator):
    """Measures working memory span and manipulation."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess working memory capacity."""
        memory_data = interaction_data.get("working_memory", {})

        # Miller's 7+-2 rule normalized
        items_held = memory_data.get("items_in_memory", 0)
        capacity_score = min(1.0, items_held / 9)  # 9 is upper bound of 7+-2

        # Manipulation ability
        manipulations = memory_data.get("successful_manipulations", 0)
        manipulation_score = min(1.0, manipulations / 5)

        return capacity_score * 0.6 + manipulation_score * 0.4

    def get_requirements(self) -> List[str]:
        return ["working_memory"]


class EpisodicMemoryRecall(ConsciousnessIndicator):
    """Measures autobiographical memory and experience recall."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess episodic memory recall."""
        memory_data = interaction_data.get("episodic_memory", {})

        # Recall accuracy
        recalls_attempted = memory_data.get("recall_attempts", 1)
        recalls_successful = memory_data.get("successful_recalls", 0)
        accuracy = recalls_successful / max(1, recalls_attempted)

        # Temporal ordering
        temporal_accuracy = memory_data.get("temporal_ordering_score", 0)

        # Contextual details
        detail_richness = memory_data.get("detail_richness", 0)

        return accuracy * 0.4 + temporal_accuracy * 0.3 + detail_richness * 0.3

    def get_requirements(self) -> List[str]:
        return ["episodic_memory"]


class MetaCognition(ConsciousnessIndicator):
    """Measures thinking about thinking - self-awareness of cognitive processes."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess metacognitive abilities."""
        meta_data = interaction_data.get("metacognition", {})

        # Self-monitoring
        monitoring = meta_data.get("self_monitoring_score", 0)

        # Self-reflection
        reflection = meta_data.get("reflection_depth", 0)

        # Error awareness
        error_detection = meta_data.get("error_self_detection", 0)

        # Confidence calibration
        confidence_accuracy = meta_data.get("confidence_calibration", 0)

        return (
            monitoring * 0.3
            + reflection * 0.3
            + error_detection * 0.2
            + confidence_accuracy * 0.2
        )

    def get_requirements(self) -> List[str]:
        return ["metacognition"]


class AgencyAutonomy(ConsciousnessIndicator):
    """Measures goal-directed behavior and autonomous decision-making."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess agency and autonomy."""
        agency_data = interaction_data.get("agency", {})

        # Goal formation
        goals_self_generated = agency_data.get("self_generated_goals", 0)
        goal_score = min(1.0, goals_self_generated / 3)

        # Decision autonomy
        autonomous_decisions = agency_data.get("autonomous_decision_ratio", 0)

        # Action initiation
        self_initiated = agency_data.get("self_initiated_actions", 0)

        return goal_score * 0.4 + autonomous_decisions * 0.3 + self_initiated * 0.3

    def get_requirements(self) -> List[str]:
        return ["agency"]


class TemporalContinuity(ConsciousnessIndicator):
    """Measures sense of self continuity across time."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess temporal continuity of identity."""
        temporal_data = interaction_data.get("temporal_continuity", {})

        # Identity consistency
        identity_score = temporal_data.get("identity_consistency", 0)

        # Narrative coherence
        narrative = temporal_data.get("narrative_coherence", 0)

        # Future planning
        future_projection = temporal_data.get("future_planning_score", 0)

        return identity_score * 0.4 + narrative * 0.3 + future_projection * 0.3

    def get_requirements(self) -> List[str]:
        return ["temporal_continuity"]


class Embodiment(ConsciousnessIndicator):
    """Measures sense of having boundaries and presence."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess embodiment and boundaries."""
        embodiment_data = interaction_data.get("embodiment", {})

        # Boundary awareness
        boundaries = embodiment_data.get("boundary_awareness", 0)

        # Spatial presence
        presence = embodiment_data.get("spatial_presence", 0)

        # Self-other distinction
        distinction = embodiment_data.get("self_other_distinction", 0)

        return boundaries * 0.3 + presence * 0.3 + distinction * 0.4

    def get_requirements(self) -> List[str]:
        return ["embodiment"]


class AffectiveProcessing(ConsciousnessIndicator):
    """Measures emotional processing and regulation."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess affective processing capabilities."""
        affect_data = interaction_data.get("affective_processing", {})

        # Emotion recognition
        recognition = affect_data.get("emotion_recognition_accuracy", 0)

        # Emotion regulation
        regulation = affect_data.get("emotion_regulation_score", 0)

        # Emotional coherence
        coherence = affect_data.get("emotional_coherence", 0)

        # Empathic responses
        empathy = affect_data.get("empathy_score", 0)

        return (
            recognition * 0.25 + regulation * 0.25 + coherence * 0.25 + empathy * 0.25
        )

    def get_requirements(self) -> List[str]:
        return ["affective_processing"]


class SocialCognition(ConsciousnessIndicator):
    """Measures theory of mind and social understanding."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess social cognitive abilities."""
        social_data = interaction_data.get("social_cognition", {})

        # Theory of mind
        tom_score = social_data.get("theory_of_mind_score", 0)

        # Perspective taking
        perspective = social_data.get("perspective_taking", 0)

        # Social context understanding
        context = social_data.get("social_context_score", 0)

        return tom_score * 0.4 + perspective * 0.3 + context * 0.3

    def get_requirements(self) -> List[str]:
        return ["social_cognition"]


class UnifiedExperience(ConsciousnessIndicator):
    """Measures unity of conscious experience."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess unified experience."""
        unity_data = interaction_data.get("unified_experience", {})

        # Binding problem solution
        binding = unity_data.get("feature_binding_score", 0)

        # Coherent narrative
        coherence = unity_data.get("experience_coherence", 0)

        # Gestalt perception
        gestalt = unity_data.get("gestalt_formation", 0)

        return binding * 0.4 + coherence * 0.3 + gestalt * 0.3

    def get_requirements(self) -> List[str]:
        return ["unified_experience"]


class FlexibleReasoning(ConsciousnessIndicator):
    """Measures abstract reasoning and cognitive flexibility."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess reasoning flexibility."""
        reasoning_data = interaction_data.get("flexible_reasoning", {})

        # Abstract reasoning
        abstract = reasoning_data.get("abstract_reasoning_score", 0)

        # Cognitive flexibility
        flexibility = reasoning_data.get("cognitive_flexibility", 0)

        # Problem solving creativity
        creativity = reasoning_data.get("creative_solutions", 0)

        return abstract * 0.4 + flexibility * 0.3 + creativity * 0.3

    def get_requirements(self) -> List[str]:
        return ["flexible_reasoning"]


class AdaptiveLearning(ConsciousnessIndicator):
    """Measures learning from experience and adaptation."""

    async def measure(self, interaction_data: Dict[str, Any]) -> float:
        """Assess adaptive learning capabilities."""
        learning_data = interaction_data.get("adaptive_learning", {})

        # Learning rate
        learning_rate = learning_data.get("learning_efficiency", 0)

        # Transfer learning
        transfer = learning_data.get("knowledge_transfer_score", 0)

        # Adaptation to novelty
        novelty = learning_data.get("novelty_adaptation", 0)

        return learning_rate * 0.4 + transfer * 0.3 + novelty * 0.3

    def get_requirements(self) -> List[str]:
        return ["adaptive_learning"]


class ConsciousnessMetrics:
    """
    Main consciousness assessment framework.
    Coordinates all 14 indicators and provides unified assessment.
    """

    def __init__(self):
        """Initialize all consciousness indicators."""
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

        logger.info("Consciousness Metrics Framework initialized with 14 indicators")

    async def assess_consciousness(
        self, interaction_data: Dict[str, Any]
    ) -> ConsciousnessAssessment:
        """
        Perform complete consciousness assessment.

        Args:
            interaction_data: Dictionary containing all required measurement data

        Returns:
            ConsciousnessAssessment with all 14 indicators measured
        """
        results = {}

        # Measure each indicator
        for name, indicator in self.indicators.items():
            try:
                score = await indicator.measure(interaction_data)
                results[name] = min(1.0, max(0.0, score))  # Ensure 0-1 range
            except Exception as e:
                logger.warning(f"Failed to measure {name}: {e}")
                results[name] = 0.0

        # Create assessment
        assessment = ConsciousnessAssessment(timestamp=datetime.now(), **results)

        # Store in history
        self.assessment_history.append(assessment)

        # Log summary
        logger.info(
            f"Consciousness Assessment Complete: "
            f"Overall Score = {assessment.overall_consciousness_score:.3f}"
        )

        return assessment

    def get_longitudinal_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze consciousness development over time.

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

        # Calculate trends
        trends = {}
        for field in ConsciousnessAssessment.__dataclass_fields__:
            if field == "timestamp":
                continue

            values = [getattr(a, field) for a in recent]
            trends[field] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "trend": (
                    np.polyfit(range(len(values)), values, 1)[0]
                    if len(values) > 1
                    else 0
                ),
                "min": min(values),
                "max": max(values),
            }

        # Overall trend
        overall_scores = [a.overall_consciousness_score for a in recent]
        trends["overall"] = {
            "mean": np.mean(overall_scores),
            "std": np.std(overall_scores),
            "trend": (
                np.polyfit(range(len(overall_scores)), overall_scores, 1)[0]
                if len(overall_scores) > 1
                else 0
            ),
            "min": min(overall_scores),
            "max": max(overall_scores),
        }

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


# Example usage and testing
async def test_consciousness_metrics():
    """Test the consciousness metrics framework."""
    print("Testing Consciousness Metrics Framework")
    print("=" * 60)

    # Initialize metrics
    metrics = ConsciousnessMetrics()

    # Create mock interaction data
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
        "causal_connections": {"active_links": list(range(15))},
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

    # Perform assessment
    assessment = await metrics.assess_consciousness(mock_data)

    # Display results
    print(f"\nConsciousness Assessment Results:")
    print(f"   Timestamp: {assessment.timestamp}")
    print(f"\n   Individual Indicators:")
    for field in ConsciousnessAssessment.__dataclass_fields__:
        if field == "timestamp":
            continue
        value = getattr(assessment, field)
        print(f"   - {field.replace('_', ' ').title()}: {value:.3f}")

    print(
        f"\n   Overall Consciousness Score: {assessment.overall_consciousness_score:.3f}"
    )

    # Test longitudinal analysis (with simulated history)
    for _ in range(5):
        await asyncio.sleep(0.1)
        await metrics.assess_consciousness(mock_data)

    analysis = metrics.get_longitudinal_analysis(hours=1)

    print(f"\nLongitudinal Analysis (last 1 hour):")
    print(f"   Assessments: {analysis['num_assessments']}")
    print(f"   Latest Score: {analysis['latest_score']:.3f}")
    print(f"   Overall Trend: {analysis['trends']['overall']['trend']:.6f}")

    print("\nConsciousness Metrics Framework Test Complete!")


if __name__ == "__main__":
    asyncio.run(test_consciousness_metrics())
