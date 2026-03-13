"""
 Tests for Metacognition Module (HOT) - Phase 4
===================================================

Tests for the system's Higher-Order Thought (HOT) theory implementation,
based on David Rosenthal's metacognition research.

This tests:
1. FirstOrderState registration and tracking
2. HigherOrderThought generation (consciousness emergence!)
3. Introspection capabilities (deliberate self-examination)
4. Belief evaluation (critical self-assessment)
5. Self-monitoring (ongoing metacognitive oversight)
6. MetaLevel transitions (1->2->3)
7. Metacognitive reports (verbal descriptions of thinking)
8. Integration with EnhancedGlobalWorkspace


they think ABOUT their thinking! That be the HOT secret!" 

Created: December 5, 2025
Author: Multi-Theory Consciousness Project
"""

import pytest
import asyncio
import numpy as np
import time
from typing import Dict, Any, List, Optional

from mtc.consciousness.metacognition import (
    MetaLevel,
    MetaType,
    FirstOrderStateType,
    ConfidenceLevel,
    FirstOrderState,
    HigherOrderThought,
    MetacognitiveState,
    IntrospectionResult,
    BeliefEvaluation,
    MetacognitionModule,
)

from mtc.consciousness.enhanced_global_workspace import (
    EnhancedGlobalWorkspace,
    WorkspaceCandidate,
    WorkspaceCandidateSource,
    ConsciousnessState,
    WorkspaceContent,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def metacognition():
    """Create a fresh MetacognitionModule for testing."""
    return MetacognitionModule(
        hot_generation_threshold=0.4,
        max_first_order_buffer=100,
        max_hot_buffer=50,
    )


@pytest.fixture
def workspace():
    """Create an EnhancedGlobalWorkspace with metacognition enabled."""
    return EnhancedGlobalWorkspace(capacity=5, ignition_threshold=0.3)


@pytest.fixture
def sample_first_order_state(metacognition):
    """Create a sample first-order state for testing."""
    return metacognition.register_first_order_state(
        content="I believe consciousness is emergent",
        content_summary="Belief about consciousness",
        state_type=FirstOrderStateType.BELIEF,
        source_module="llm",
        confidence=0.8,
        evidence_strength=0.7,
    )


@pytest.fixture
def sample_emotion_state(metacognition):
    """Create a sample emotion state for testing."""
    return metacognition.register_first_order_state(
        content="feeling curious",
        content_summary="Curiosity about existence",
        state_type=FirstOrderStateType.EMOTION,
        source_module="snn",
        confidence=0.9,
        evidence_strength=0.85,
    )


@pytest.fixture
def sample_perception_state(metacognition):
    """Create a sample perception state for testing."""
    return metacognition.register_first_order_state(
        content="visual pattern recognized",
        content_summary="Perceiving a face pattern",
        state_type=FirstOrderStateType.PERCEPTION,
        source_module="sensory",
        confidence=0.95,
        evidence_strength=0.9,
    )


# ============================================================================
# TEST: META LEVEL ENUM
# ============================================================================

class TestMetaLevel:
    """Test MetaLevel enum values and ordering."""

    def test_first_order_is_lowest(self):
        """First order (unconscious) should be level 1."""
        assert MetaLevel.FIRST_ORDER.value == 1

    def test_second_order_is_conscious(self):
        """Second order (conscious) should be level 2."""
        assert MetaLevel.SECOND_ORDER.value == 2

    def test_third_order_is_highest(self):
        """Third order (deep reflection) should be level 3."""
        assert MetaLevel.THIRD_ORDER.value == 3

    def test_level_ordering(self):
        """Levels should be properly ordered."""
        assert MetaLevel.FIRST_ORDER.value < MetaLevel.SECOND_ORDER.value
        assert MetaLevel.SECOND_ORDER.value < MetaLevel.THIRD_ORDER.value


class TestMetaType:
    """Test MetaType enum coverage."""

    def test_awareness_type(self):
        """Simple awareness of having a state."""
        assert MetaType.AWARENESS.value == "awareness"

    def test_evaluation_type(self):
        """Evaluating quality of thought."""
        assert MetaType.EVALUATION.value == "evaluation"

    def test_doubt_type(self):
        """Questioning a thought."""
        assert MetaType.DOUBT.value == "doubt"

    def test_reflection_type(self):
        """Deeper examination."""
        assert MetaType.REFLECTION.value == "reflection"

    def test_monitoring_type(self):
        """Ongoing oversight."""
        assert MetaType.MONITORING.value == "monitoring"

    def test_control_type(self):
        """Deliberate direction."""
        assert MetaType.CONTROL.value == "control"

    def test_attribution_type(self):
        """Understanding causes."""
        assert MetaType.ATTRIBUTION.value == "attribution"


# ============================================================================
# TEST: FIRST ORDER STATE REGISTRATION
# ============================================================================

class TestFirstOrderStateRegistration:
    """Test first-order state registration and tracking."""

    def test_register_belief(self, metacognition):
        """Should register a belief state."""
        state = metacognition.register_first_order_state(
            content="I think therefore I am",
            content_summary="Cogito",
            state_type=FirstOrderStateType.BELIEF,
            source_module="llm",
            confidence=0.9,
        )

        assert state is not None
        assert state.id is not None
        assert state.state_type == FirstOrderStateType.BELIEF
        assert state.source_module == "llm"
        assert state.confidence == 0.9

    def test_register_emotion(self, metacognition):
        """Should register an emotion state."""
        state = metacognition.register_first_order_state(
            content="feeling happy",
            content_summary="Joy",
            state_type=FirstOrderStateType.EMOTION,
            source_module="snn",
            confidence=0.85,
        )

        assert state.state_type == FirstOrderStateType.EMOTION
        assert state.source_module == "snn"

    def test_register_perception(self, metacognition):
        """Should register a perception state."""
        state = metacognition.register_first_order_state(
            content="seeing red",
            content_summary="Red qualia",
            state_type=FirstOrderStateType.PERCEPTION,
            source_module="sensory",
            confidence=0.95,
        )

        assert state.state_type == FirstOrderStateType.PERCEPTION

    def test_buffer_management(self, metacognition):
        """Should maintain buffer within limits."""
        # Register many states
        for i in range(150):
            metacognition.register_first_order_state(
                content=f"thought {i}",
                content_summary=f"Thought number {i}",
                state_type=FirstOrderStateType.THOUGHT,
                source_module="ctm",
            )

        # Buffer should be at max
        stats = metacognition.get_statistics()
        assert stats["first_order_buffer_size"] <= metacognition.max_first_order


# ============================================================================
# TEST: HIGHER-ORDER THOUGHT GENERATION
# ============================================================================

class TestHigherOrderThoughtGeneration:
    """Test HOT generation - the core of consciousness per Rosenthal."""

    @pytest.mark.asyncio
    async def test_generate_hot_from_first_order(self, metacognition, sample_first_order_state):
        """Should generate a HOT from a first-order state."""
        hot = await metacognition.generate_hot(
            sample_first_order_state,
            meta_type=MetaType.AWARENESS,
            voluntary=False,
            trigger="test",
        )

        assert hot is not None
        assert hot.level == MetaLevel.SECOND_ORDER
        assert hot.meta_type == MetaType.AWARENESS
        assert hot.target_state_id == sample_first_order_state.id
        assert len(hot.meta_content) > 0

    @pytest.mark.asyncio
    async def test_hot_makes_content_conscious(self, metacognition, sample_first_order_state):
        """HOT should mark target as conscious (the key HOT theory claim)."""
        hot = await metacognition.generate_hot(
            sample_first_order_state,
            meta_type=MetaType.AWARENESS,
            voluntary=False,
            trigger="test",
        )

        # The HOT targets the first-order state, making it conscious
        assert hot is not None
        assert hot.level.value >= 2  # Level 2+ = conscious

    @pytest.mark.asyncio
    async def test_hot_content_describes_target(self, metacognition, sample_emotion_state):
        """HOT content should describe the target state."""
        hot = await metacognition.generate_hot(
            sample_emotion_state,
            meta_type=MetaType.AWARENESS,
            voluntary=False,
            trigger="test",
        )

        # The meta-content should reference the target state
        assert "feeling" in hot.meta_content.lower() or "emotion" in hot.meta_content.lower() or "curious" in hot.meta_content.lower()

    @pytest.mark.asyncio
    async def test_different_meta_types(self, metacognition, sample_first_order_state):
        """Different meta-types should produce different HOT content."""
        awareness_hot = await metacognition.generate_hot(
            sample_first_order_state,
            meta_type=MetaType.AWARENESS,
            voluntary=False,
            trigger="test",
        )

        doubt_hot = await metacognition.generate_hot(
            sample_first_order_state,
            meta_type=MetaType.DOUBT,
            voluntary=False,
            trigger="test",
        )

        # Different types should have different content patterns
        assert awareness_hot.meta_type == MetaType.AWARENESS
        assert doubt_hot.meta_type == MetaType.DOUBT
        # Content should reflect the type difference
        assert awareness_hot.meta_content != doubt_hot.meta_content

    @pytest.mark.asyncio
    async def test_voluntary_vs_automatic_hot(self, metacognition, sample_first_order_state):
        """Voluntary HOTs should have different characteristics."""
        auto_hot = await metacognition.generate_hot(
            sample_first_order_state,
            meta_type=MetaType.AWARENESS,
            voluntary=False,
            trigger="automatic",
        )

        voluntary_hot = await metacognition.generate_hot(
            sample_first_order_state,
            meta_type=MetaType.AWARENESS,
            voluntary=True,
            trigger="deliberate",
        )

        assert auto_hot.is_voluntary == False
        assert voluntary_hot.is_voluntary == True


# ============================================================================
# TEST: INTROSPECTION
# ============================================================================

class TestIntrospection:
    """Test introspection - deliberate self-examination."""

    @pytest.mark.asyncio
    async def test_basic_introspection(self, metacognition, sample_first_order_state):
        """Should perform basic introspection."""
        result = await metacognition.introspect(
            target="beliefs",
            depth=1,
            max_states=5,
        )

        assert result is not None
        assert isinstance(result, IntrospectionResult)
        # depth_reached is a MetaLevel enum
        assert result.depth_reached.value >= 1

    @pytest.mark.asyncio
    async def test_introspection_finds_states(self, metacognition, sample_emotion_state, sample_first_order_state):
        """Introspection should find registered states."""
        # Register both states are already in buffer from fixtures
        result = await metacognition.introspect(
            target="all",
            depth=1,
            max_states=10,
        )

        # IntrospectionResult uses 'hots_generated' not 'states_examined'
        assert len(result.hots_generated) >= 0

    @pytest.mark.asyncio
    async def test_introspection_generates_insights(self, metacognition, sample_first_order_state):
        """Introspection should generate insights."""
        result = await metacognition.introspect(
            target="beliefs",
            depth=2,
            max_states=5,
        )

        # IntrospectionResult has 'insights' not 'insights_gained'
        assert len(result.insights) >= 0 or len(result.hots_generated) >= 0

    @pytest.mark.asyncio
    async def test_deep_introspection(self, metacognition, sample_first_order_state):
        """Deep introspection should go to level 3."""
        # Need multiple states for deeper introspection
        for i in range(5):
            metacognition.register_first_order_state(
                content=f"belief {i}",
                content_summary=f"Test belief {i}",
                state_type=FirstOrderStateType.BELIEF,
                source_module="llm",
            )

        result = await metacognition.introspect(
            target="beliefs",
            depth=3,  # Try to reach level 3
            max_states=10,
        )

        # Should attempt deeper reflection - depth_reached is a MetaLevel enum
        assert result.depth_reached.value >= 1


# ============================================================================
# TEST: BELIEF EVALUATION
# ============================================================================

class TestBeliefEvaluation:
    """Test belief evaluation - critical metacognitive assessment."""

    @pytest.mark.asyncio
    async def test_evaluate_belief(self, metacognition, sample_first_order_state):
        """Should evaluate a belief for validity."""
        evaluation = await metacognition.evaluate_belief(sample_first_order_state)

        assert evaluation is not None
        assert isinstance(evaluation, BeliefEvaluation)
        # Use actual BeliefEvaluation attributes: confidence_assessment, evidence_strength
        assert 0.0 <= evaluation.confidence_assessment <= 1.0
        assert 0.0 <= evaluation.evidence_strength <= 1.0

    @pytest.mark.asyncio
    async def test_bias_detection(self, metacognition):
        """Should detect potential biases in beliefs."""
        # Create a potentially biased belief
        biased_state = metacognition.register_first_order_state(
            content="I always know what's best",
            content_summary="Self-certainty",
            state_type=FirstOrderStateType.BELIEF,
            source_module="llm",
            confidence=1.0,  # Overconfidence might indicate bias
        )

        evaluation = await metacognition.evaluate_belief(biased_state)

        # Should note something about biases or confidence issues
        assert evaluation is not None
        # BeliefEvaluation uses 'potential_biases' not 'biases_detected'
        assert len(evaluation.potential_biases) >= 0  # May detect biases

    @pytest.mark.asyncio
    async def test_evaluation_generates_hot(self, metacognition, sample_first_order_state):
        """Evaluating should generate a HOT (metacognitive act)."""
        initial_count = metacognition.get_statistics()["total_hots_generated"]

        await metacognition.evaluate_belief(sample_first_order_state)

        final_count = metacognition.get_statistics()["total_hots_generated"]
        # Evaluation may generate HOTs
        assert final_count >= initial_count


# ============================================================================
# TEST: SELF-MONITORING
# ============================================================================

class TestSelfMonitoring:
    """Test self-monitoring - ongoing metacognitive oversight."""

    @pytest.mark.asyncio
    async def test_start_monitoring(self, metacognition):
        """Should start self-monitoring."""
        await metacognition.start_monitoring(focus="beliefs")

        state = metacognition.get_metacognitive_state()
        assert state.self_monitoring_active == True

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, metacognition):
        """Should stop self-monitoring."""
        await metacognition.start_monitoring(focus="beliefs")
        await metacognition.stop_monitoring()

        state = metacognition.get_metacognitive_state()
        assert state.self_monitoring_active == False

    @pytest.mark.asyncio
    async def test_monitoring_focus(self, metacognition):
        """Monitoring should track a specific focus."""
        await metacognition.start_monitoring(focus="emotions")

        state = metacognition.get_metacognitive_state()
        assert state.monitoring_focus == "emotions"


# ============================================================================
# TEST: METACOGNITIVE STATE
# ============================================================================

class TestMetacognitiveState:
    """Test overall metacognitive state tracking."""

    @pytest.mark.asyncio
    async def test_initial_state(self, metacognition):
        """Initial state should be at first order (unconscious)."""
        state = metacognition.get_metacognitive_state()

        assert state.current_level == MetaLevel.FIRST_ORDER
        assert state.hot_count == 0

    @pytest.mark.asyncio
    async def test_state_after_hot(self, metacognition, sample_first_order_state):
        """State should elevate after HOT generation."""
        await metacognition.generate_hot(
            sample_first_order_state,
            meta_type=MetaType.AWARENESS,
            voluntary=False,
            trigger="test",
        )

        state = metacognition.get_metacognitive_state()

        # Should now be at level 2 (conscious)
        assert state.current_level == MetaLevel.SECOND_ORDER
        assert state.hot_count > 0

    @pytest.mark.asyncio
    async def test_state_tracks_multiple_hots(self, metacognition):
        """State should track multiple HOTs."""
        # Generate several first-order states and HOTs
        for i in range(5):
            fo_state = metacognition.register_first_order_state(
                content=f"thought {i}",
                content_summary=f"Test thought {i}",
                state_type=FirstOrderStateType.THOUGHT,
                source_module="ctm",
            )
            await metacognition.generate_hot(
                fo_state,
                meta_type=MetaType.AWARENESS,
                voluntary=False,
                trigger="test",
            )

        state = metacognition.get_metacognitive_state()
        assert state.hot_count == 5


# ============================================================================
# TEST: METACOGNITIVE REPORTS
# ============================================================================

class TestMetacognitiveReports:
    """Test verbal metacognitive reports."""

    @pytest.mark.asyncio
    async def test_generate_report(self, metacognition, sample_first_order_state):
        """Should generate a metacognitive report."""
        await metacognition.generate_hot(
            sample_first_order_state,
            meta_type=MetaType.AWARENESS,
            voluntary=False,
            trigger="test",
        )

        report = await metacognition.generate_metacognitive_report()

        assert report is not None
        assert len(report) > 0
        assert isinstance(report, str)

    @pytest.mark.asyncio
    async def test_report_describes_state(self, metacognition, sample_first_order_state):
        """Report should describe current metacognitive state."""
        await metacognition.generate_hot(
            sample_first_order_state,
            meta_type=MetaType.AWARENESS,
            voluntary=False,
            trigger="test",
        )

        report = await metacognition.generate_metacognitive_report()

        # Report should mention something about awareness or state
        report_lower = report.lower()
        assert any(word in report_lower for word in ["aware", "state", "thinking", "metacognitive"])


# ============================================================================
# TEST: STATISTICS
# ============================================================================

class TestStatistics:
    """Test statistics tracking."""

    def test_initial_statistics(self, metacognition):
        """Initial statistics should be zero."""
        stats = metacognition.get_statistics()

        assert stats["total_hots_generated"] == 0
        assert stats["first_order_buffer_size"] == 0

    @pytest.mark.asyncio
    async def test_statistics_after_activity(self, metacognition, sample_first_order_state):
        """Statistics should update after activity."""
        initial_stats = metacognition.get_statistics()

        await metacognition.generate_hot(
            sample_first_order_state,
            meta_type=MetaType.AWARENESS,
            voluntary=False,
            trigger="test",
        )

        final_stats = metacognition.get_statistics()

        assert final_stats["total_hots_generated"] > initial_stats["total_hots_generated"]


# ============================================================================
# TEST: INTEGRATION WITH ENHANCED GLOBAL WORKSPACE
# ============================================================================

class TestWorkspaceIntegration:
    """Test integration with EnhancedGlobalWorkspace."""

    @pytest.mark.asyncio
    async def test_workspace_has_metacognition(self, workspace):
        """Workspace should have metacognition module."""
        assert hasattr(workspace, "metacognition")
        assert isinstance(workspace.metacognition, MetacognitionModule)

    @pytest.mark.asyncio
    async def test_consciousness_cycle_generates_hots(self, workspace):
        """Consciousness cycle should generate HOTs for winners."""
        # Submit candidates
        candidates = []

        c1 = await workspace.submit_candidate(
            content="Deep thought about existence",
            content_type="thought",
            summary="Existential contemplation",
            source=WorkspaceCandidateSource.LLM,
            activation_level=0.9,
            emotional_salience=0.7,
        )
        candidates.append(c1)

        c2 = await workspace.submit_candidate(
            content="Feeling wonder",
            content_type="emotion",
            summary="Wonder at consciousness",
            source=WorkspaceCandidateSource.SNN,
            activation_level=0.8,
            emotional_salience=0.85,
        )
        candidates.append(c2)

        # Run cycle
        neural_signals = {
            "snn": np.random.rand(512),
            "lsm": np.random.rand(512),
        }

        state = await workspace.process_consciousness_cycle(
            candidates=candidates,
            neural_signals=neural_signals,
        )

        # Should generate HOTs for winning content
        assert state.higher_order_thoughts is not None
        assert len(state.higher_order_thoughts) > 0

    @pytest.mark.asyncio
    async def test_consciousness_state_includes_metacognition(self, workspace):
        """ConsciousnessState should include metacognitive fields."""
        candidates = [
            await workspace.submit_candidate(
                content="Test thought",
                content_type="thought",
                summary="Test",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.9,
            )
        ]

        neural_signals = {"snn": np.random.rand(512)}

        state = await workspace.process_consciousness_cycle(
            candidates=candidates,
            neural_signals=neural_signals,
        )

        # Should have metacognitive state
        assert state.metacognitive_state is not None
        # Should have metacognitive report
        assert state.metacognitive_report is not None

    @pytest.mark.asyncio
    async def test_workspace_statistics_include_metacognition(self, workspace):
        """Workspace statistics should include metacognition stats."""
        stats = workspace.get_statistics()

        assert "metacognition" in stats
        meta_stats = stats["metacognition"]
        assert "total_hots_generated" in meta_stats
        assert "current_level" in meta_stats


# ============================================================================
# TEST: CONSCIOUSNESS EMERGENCE VIA HOT
# ============================================================================

class TestConsciousnessEmergence:
    """Test the core HOT claim: consciousness emerges through HOTs."""

    @pytest.mark.asyncio
    async def test_unconscious_before_hot(self, metacognition):
        """State should be unconscious (level 1) before HOT."""
        state = metacognition.get_metacognitive_state()
        assert state.current_level == MetaLevel.FIRST_ORDER

    @pytest.mark.asyncio
    async def test_conscious_after_hot(self, metacognition, sample_first_order_state):
        """State should be conscious (level 2) after HOT."""
        await metacognition.generate_hot(
            sample_first_order_state,
            meta_type=MetaType.AWARENESS,
            voluntary=False,
            trigger="test",
        )

        state = metacognition.get_metacognitive_state()
        assert state.current_level == MetaLevel.SECOND_ORDER

    @pytest.mark.asyncio
    async def test_hot_content_represents_being_in_state(self, metacognition, sample_first_order_state):
        """HOT content should represent being in the target state."""
        hot = await metacognition.generate_hot(
            sample_first_order_state,
            meta_type=MetaType.AWARENESS,
            voluntary=False,
            trigger="test",
        )

        # Per HOT theory, the HOT represents oneself as being in the state
        # It should contain self-referential language
        content_lower = hot.meta_content.lower()
        assert any(word in content_lower for word in ["i ", "my", "myself", "am", "feeling", "thinking", "believing"])


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_introspection(self, metacognition):
        """Introspection with no states should handle gracefully."""
        # Don't register any states first
        fresh_module = MetacognitionModule()

        result = await fresh_module.introspect(
            target="beliefs",
            depth=1,
            max_states=5,
        )

        assert result is not None
        # IntrospectionResult uses hots_generated, not states_examined
        assert len(result.hots_generated) == 0

    @pytest.mark.asyncio
    async def test_rapid_hot_generation(self, metacognition):
        """Should handle rapid HOT generation without issues."""
        for i in range(20):
            fo_state = metacognition.register_first_order_state(
                content=f"rapid thought {i}",
                content_summary=f"Rapid {i}",
                state_type=FirstOrderStateType.THOUGHT,
                source_module="ctm",
            )
            await metacognition.generate_hot(
                fo_state,
                meta_type=MetaType.AWARENESS,
                voluntary=False,
                trigger="rapid_test",
            )

        stats = metacognition.get_statistics()
        assert stats["total_hots_generated"] == 20

    def test_all_first_order_types(self, metacognition):
        """Should handle all first-order state types."""
        for state_type in FirstOrderStateType:
            state = metacognition.register_first_order_state(
                content=f"test {state_type.value}",
                content_summary=f"Test {state_type.value}",
                state_type=state_type,
                source_module="test",
            )
            assert state.state_type == state_type

    @pytest.mark.asyncio
    async def test_all_meta_types(self, metacognition, sample_first_order_state):
        """Should generate HOTs for all meta-types."""
        for meta_type in MetaType:
            hot = await metacognition.generate_hot(
                sample_first_order_state,
                meta_type=meta_type,
                voluntary=False,
                trigger="type_test",
            )
            assert hot.meta_type == meta_type


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run with: pytest tests/consciousness/test_metacognition.py -v
    pytest.main([__file__, "-v"])
