"""
Tests for Metacognition Module (HOT)
====================================

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
9. Structured self-prediction checks, narration gate and persistence
"""

import re
from types import SimpleNamespace

import pytest
import numpy as np

from mtc.consciousness.metacognition import (
    MetaLevel,
    MetaType,
    FirstOrderState,
    FirstOrderStateType,
    HigherOrderThought,
    IntrospectionResult,
    BeliefEvaluation,
    MetacognitionModule,
    SelfPrediction,
)

from mtc.consciousness.enhanced_global_workspace import (
    EnhancedGlobalWorkspace,
    WorkspaceCandidateSource,
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
    async def test_generate_hot_from_first_order(
        self, metacognition, sample_first_order_state
    ):
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
    async def test_hot_makes_content_conscious(
        self, metacognition, sample_first_order_state
    ):
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
    async def test_hot_content_describes_target(
        self, metacognition, sample_emotion_state
    ):
        """HOT content should describe the target state."""
        hot = await metacognition.generate_hot(
            sample_emotion_state,
            meta_type=MetaType.AWARENESS,
            voluntary=False,
            trigger="test",
        )

        # The meta-content should reference the target state
        assert (
            "feeling" in hot.meta_content.lower()
            or "emotion" in hot.meta_content.lower()
            or "curious" in hot.meta_content.lower()
        )

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
    async def test_voluntary_vs_automatic_hot(
        self, metacognition, sample_first_order_state
    ):
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

        assert auto_hot.is_voluntary is False
        assert voluntary_hot.is_voluntary is True


# ============================================================================
# TEST: INTROSPECTION
# ============================================================================


class TestIntrospection:
    """Test introspection - deliberate self-examination."""

    @pytest.mark.asyncio
    async def test_basic_introspection(self, metacognition, sample_first_order_state):
        """Introspection reaches second order with one HOT on the sample state."""
        result = await metacognition.introspect(
            target=None,
            depth=2,
            max_states=5,
        )

        assert isinstance(result, IntrospectionResult)
        assert result.depth_reached is MetaLevel.SECOND_ORDER
        targeting = [
            hot
            for hot in result.hots_generated
            if hot.target_state_id == sample_first_order_state.id
        ]
        assert len(targeting) == 1
        assert len(result.hots_generated) == 1

    @pytest.mark.asyncio
    async def test_introspection_finds_states(
        self, metacognition, sample_emotion_state, sample_first_order_state
    ):
        """Introspection finds both registered states and reflects on each."""
        result = await metacognition.introspect(
            target=None,
            depth=2,
            max_states=10,
        )

        assert {h.target_state_id for h in result.hots_generated} == {
            sample_emotion_state.id,
            sample_first_order_state.id,
        }
        assert all(h.meta_type is MetaType.REFLECTION for h in result.hots_generated)
        assert all(h.trigger == "introspection" for h in result.hots_generated)

        fresh = MetacognitionModule()
        nothing = await fresh.introspect(target="no-such-word")
        assert nothing.hots_generated == []

    @pytest.mark.asyncio
    async def test_introspection_generates_insights(
        self, metacognition, sample_first_order_state
    ):
        """An uncertain state yields both uncertainty insights and a change."""
        low = metacognition.register_first_order_state(
            content="x",
            content_summary="An unsure thing",
            state_type=FirstOrderStateType.BELIEF,
            source_module="llm",
            confidence=0.3,
        )
        result = await metacognition.introspect(target=None)

        assert any(i.startswith("I'm uncertain about:") for i in result.insights)
        assert any(i.startswith("Uncertainty noticed about:") for i in result.insights)
        assert result.confidence_changes == {low.id: pytest.approx(0.33)}

    @pytest.mark.asyncio
    async def test_deep_introspection(self, metacognition):
        """Depth 3 adds one third-order HOT on each second-order HOT."""
        for i in range(5):
            metacognition.register_first_order_state(
                content=f"belief {i}",
                content_summary=f"Test belief {i}",
                state_type=FirstOrderStateType.BELIEF,
                source_module="llm",
            )

        result = await metacognition.introspect(
            target=None,
            depth=3,
            max_states=10,
        )

        assert result.depth_reached is MetaLevel.THIRD_ORDER
        second = [h for h in result.hots_generated if h.level is MetaLevel.SECOND_ORDER]
        third = [h for h in result.hots_generated if h.level is MetaLevel.THIRD_ORDER]
        assert len(second) == 5
        assert len(third) == 5
        second_ids = {h.id for h in second}
        assert all(h.target_state_id in second_ids for h in third)
        assert {h.target_state_id for h in third} == second_ids

        shallow = MetacognitionModule(enable_third_order=False)
        for i in range(5):
            shallow.register_first_order_state(
                content=f"belief {i}",
                content_summary=f"Test belief {i}",
                state_type=FirstOrderStateType.BELIEF,
                source_module="llm",
            )
        limited = await shallow.introspect(target=None, depth=3, max_states=10)
        assert limited.depth_reached is MetaLevel.SECOND_ORDER
        assert all(h.level is MetaLevel.SECOND_ORDER for h in limited.hots_generated)
        assert len(limited.hots_generated) == 5

    @pytest.mark.asyncio
    async def test_empty_introspection_reaches_no_depth(
        self, metacognition, sample_first_order_state, sample_emotion_state
    ):
        result = await metacognition.introspect(target="no-such-word", depth=3)
        assert result.hots_generated == []
        assert result.depth_reached is MetaLevel.FIRST_ORDER

        fresh = MetacognitionModule()
        empty = await fresh.introspect(depth=3)
        assert empty.hots_generated == []
        assert empty.depth_reached is MetaLevel.FIRST_ORDER


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

        # Confidence 1.0 on the default evidence 0.5, registered just now
        assert set(evaluation.potential_biases) == {
            "confirmation_bias",
            "recency_bias",
        }
        assert evaluation.should_revise is True

    @pytest.mark.asyncio
    async def test_evaluation_generates_hot(
        self, metacognition, sample_first_order_state
    ):
        """Evaluating should generate a HOT (metacognitive act)."""
        initial_count = metacognition.get_statistics()["total_hots_generated"]

        evaluation = await metacognition.evaluate_belief(sample_first_order_state)

        final_count = metacognition.get_statistics()["total_hots_generated"]
        assert final_count == initial_count + 1
        assert evaluation.hot is not None
        assert evaluation.hot.meta_type is MetaType.EVALUATION

    @pytest.mark.asyncio
    async def test_consistency_is_unmeasured(self, metacognition):
        lone = metacognition.register_first_order_state(
            content="synthetic claim",
            content_summary="Synthetic claim",
            state_type=FirstOrderStateType.BELIEF,
            source_module="test",
            confidence=0.2,
            evidence_strength=0.5,
        )
        evaluation = await metacognition.evaluate_belief(lone)
        assert evaluation.consistency_score is None
        assert evaluation.potential_biases == []
        assert evaluation.recommendation == "Low confidence - seek more evidence"
        assert evaluation.should_revise is True

        for index in range(2):
            metacognition.register_first_order_state(
                content=f"other claim {index}",
                content_summary=f"Other claim {index}",
                state_type=FirstOrderStateType.BELIEF,
                source_module="test",
                confidence=0.6,
            )
        evaluation = await metacognition.evaluate_belief(lone)
        assert evaluation.consistency_score is None
        assert evaluation.recommendation == "Low confidence - seek more evidence"
        assert evaluation.should_revise is True


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
        assert state.self_monitoring_active is True

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, metacognition):
        """Should stop self-monitoring."""
        await metacognition.start_monitoring(focus="beliefs")
        await metacognition.stop_monitoring()

        state = metacognition.get_metacognitive_state()
        assert state.self_monitoring_active is False

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
        # At least 5 from explicit generation; micro-introspection
        # (triggered after the 3rd HOT) may add more.
        assert state.hot_count >= 5


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
        # The level is constant per HOT, so the report never claims one
        assert "metacognitive state" not in report.lower()
        assert "meta-reflection" not in report.lower()

    @pytest.mark.asyncio
    async def test_report_describes_state(
        self, metacognition, sample_first_order_state
    ):
        """Report should describe current metacognitive state."""
        await metacognition.generate_hot(
            sample_first_order_state,
            meta_type=MetaType.AWARENESS,
            voluntary=False,
            trigger="test",
        )

        report = await metacognition.generate_metacognitive_report()

        # Report describes the most recent awareness, not a level label
        assert report.startswith("Most recent awareness: ")
        assert "belief about consciousness" in report.lower()

    @pytest.mark.asyncio
    async def test_report_monitoring_focus_is_spaced(self, metacognition):
        """The monitoring focus reads as a phrase, not a run-on word."""
        await metacognition.start_monitoring(focus="the task")
        report = await metacognition.generate_metacognitive_report()
        assert "monitoring my thoughts about the task" in report


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
    async def test_statistics_after_activity(
        self, metacognition, sample_first_order_state
    ):
        """Statistics should update after activity."""
        initial_stats = metacognition.get_statistics()

        await metacognition.generate_hot(
            sample_first_order_state,
            meta_type=MetaType.AWARENESS,
            voluntary=False,
            trigger="test",
        )

        final_stats = metacognition.get_statistics()

        assert (
            final_stats["total_hots_generated"] > initial_stats["total_hots_generated"]
        )


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
    async def test_hot_content_represents_being_in_state(
        self, metacognition, sample_first_order_state
    ):
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
        assert any(
            word in content_lower
            for word in ["i ", "my", "myself", "am", "feeling", "thinking", "believing"]
        )


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
        # At least 20 from explicit generation; micro-introspection
        # (triggered after the 3rd HOT) may add more.
        assert stats["total_hots_generated"] >= 20

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
# TEST: STRUCTURED SELF-PREDICTION CHECKS
# ============================================================================


class TestSelfPredictionChecks:
    """predict_self records checks; score_prediction_from_state verifies them."""

    def test_predict_self_records_checks(self, metacognition):
        sm = metacognition.self_model
        sm.curiosity_level = 0.8
        sm.valence = 0.3
        pred = metacognition.predict_self("synthetic context")
        attrs = {c[0] for c in pred.checks}
        assert "curiosity_level" in attrs
        assert "valence" in attrs
        for check in pred.checks:
            assert len(check) == 4

    def test_score_from_state_true_when_checks_hold(self, metacognition):
        sm = metacognition.self_model
        sm.curiosity_level = 0.8
        pred = metacognition.predict_self("synthetic context")
        assert metacognition.score_prediction_from_state(pred.id) is True
        assert pred.accurate is True
        assert "checks held" in pred.outcome

    def test_score_from_state_false_when_state_moves(self, metacognition):
        sm = metacognition.self_model
        sm.curiosity_level = 0.8
        sm.valence = 0.3
        pred = metacognition.predict_self("synthetic context")
        sm.curiosity_level = 0.2
        sm.valence = -0.4
        assert metacognition.score_prediction_from_state(pred.id) is False

    def test_prediction_without_checks_stays_unscored(self, metacognition):
        pred = SelfPrediction(prediction="balanced response")
        metacognition.self_model.predictions.append(pred)
        assert metacognition.score_prediction_from_state(pred.id) is None
        assert pred.accurate is None

    def test_unknown_prediction_returns_none(self, metacognition):
        assert metacognition.score_prediction_from_state("missing") is None


# ============================================================================
# TEST: NARRATION GATE
# ============================================================================


class TestSelfModelNarrationGate:
    def test_report_on_by_default(self):
        meta = MetacognitionModule()
        assert meta.generate_self_model_report() != ""

    def test_report_empty_when_gated(self):
        meta = MetacognitionModule(narrate_self_model=False)
        assert meta.generate_self_model_report() == ""
        meta.update_self_model()
        assert meta.self_model.update_count == 1


# ============================================================================
# TEST: SELF-MODEL INPUTS FROM ACTIVE INFERENCE
# ============================================================================


def _inference_state(meta_confidence, meta_predicted_accuracy):
    """A stand-in active inference state holding only the two meta fields."""
    return SimpleNamespace(
        meta_confidence=meta_confidence,
        meta_predicted_accuracy=meta_predicted_accuracy,
    )


class TestSelfModelMetaReadings:
    """An absent meta reading leaves the self-model's previous value in place."""

    def test_self_model_keeps_confidence_when_the_meta_reading_is_absent(self):
        meta = MetacognitionModule()
        sm = meta.self_model

        meta.update_self_model(active_inference_state=_inference_state(None, None))
        assert sm.model_confidence == 0.5
        assert sm.prediction_accuracy == 0.5

        meta.update_self_model(active_inference_state=_inference_state(0.21, 0.64))
        assert sm.model_confidence == 0.21
        assert sm.prediction_accuracy == 0.64

        meta.update_self_model(active_inference_state=_inference_state(None, None))
        assert sm.model_confidence == 0.21
        assert sm.prediction_accuracy == 0.64

    def test_each_meta_field_is_kept_on_its_own(self):
        only_accuracy = MetacognitionModule()
        only_accuracy.update_self_model(
            active_inference_state=_inference_state(None, 0.64)
        )
        assert only_accuracy.self_model.model_confidence == 0.5
        assert only_accuracy.self_model.prediction_accuracy == 0.64

        only_confidence = MetacognitionModule()
        only_confidence.update_self_model(
            active_inference_state=_inference_state(0.21, None)
        )
        assert only_confidence.self_model.model_confidence == 0.21
        assert only_confidence.self_model.prediction_accuracy == 0.5


# ============================================================================
# TEST: OVERALL CONFIDENCE SOURCE
# ============================================================================


def _thought(meta: MetacognitionModule, index: int, confidence: float):
    return meta.register_first_order_state(
        content=f"synthetic content {index}",
        content_summary=f"synthetic state {index}",
        state_type=FirstOrderStateType.THOUGHT,
        source_module="test",
        confidence=confidence,
        evidence_strength=0.5,
    )


def _buffer_mean(meta: MetacognitionModule) -> float:
    confidences = [h.confidence for h in meta.higher_order_thoughts]
    return sum(confidences) / len(confidences)


class TestOverallConfidence:
    """overall_confidence is the mean confidence of the buffered HOTs."""

    @pytest.mark.asyncio
    async def test_overall_confidence_is_none_before_any_hot(self):
        meta = MetacognitionModule()
        stats = meta.get_statistics()
        assert stats["overall_confidence"] is None
        assert stats["doubt_level"] is None
        assert await meta.generate_metacognitive_report() == ""
        meta.update_self_model()
        assert meta.self_model.confidence == 0.5

    @pytest.mark.asyncio
    async def test_overall_confidence_is_the_mean_of_buffered_hot_confidences(self):
        meta = MetacognitionModule(introspection_cooldown=3600.0)
        for index, confidence in enumerate((0.9, 0.95, 1.0)):
            await meta.generate_hot(_thought(meta, index, confidence), voluntary=True)

        expected = _buffer_mean(meta)
        stats = meta.get_statistics()
        assert stats["overall_confidence"] == pytest.approx(expected)
        assert expected == pytest.approx(0.77)
        assert stats["doubt_level"] is None

        meta.update_self_model()
        assert meta.self_model.confidence == pytest.approx(expected)

        await meta.introspect_consciousness_level(0.1)
        assert meta.get_statistics()["overall_confidence"] == pytest.approx(
            _buffer_mean(meta)
        )
        assert meta.get_statistics()["overall_confidence"] != pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_overall_confidence_follows_the_bounded_buffer(self):
        meta = MetacognitionModule(max_hot_buffer=2, introspection_cooldown=3600.0)
        for index, confidence in enumerate((0.0, 1.0, 1.0)):
            await meta.generate_hot(_thought(meta, index, confidence), voluntary=True)
        # Only the two newest HOTs (confidence 0.8 each) are buffered; the
        # mean over every HOT generated would be 0.6.
        assert [h.confidence for h in meta.higher_order_thoughts] == pytest.approx(
            [0.8, 0.8]
        )
        assert meta.get_statistics()["overall_confidence"] == pytest.approx(0.8)
        meta.reset()
        assert meta.get_statistics()["overall_confidence"] is None
        assert meta.get_statistics()["doubt_level"] is None

    @pytest.mark.asyncio
    async def test_v04_constant_source_keeps_the_constants(self):
        meta = MetacognitionModule(
            overall_confidence_source="v04_constant", introspection_cooldown=3600.0
        )
        assert meta.get_statistics()["overall_confidence"] == 0.5
        assert meta.get_statistics()["doubt_level"] == 0.0
        for index, confidence in enumerate((0.9, 0.95, 1.0)):
            await meta.generate_hot(_thought(meta, index, confidence), voluntary=True)
        assert meta.get_statistics()["overall_confidence"] == 0.5
        assert meta.get_statistics()["doubt_level"] == 0.0
        meta.update_self_model()
        assert meta.self_model.confidence == 0.5

        await meta.introspect_consciousness_level(0.75)
        assert meta.get_statistics()["overall_confidence"] == 0.75

        meta.reset()
        assert meta.get_statistics()["overall_confidence"] == 0.5
        assert meta.get_statistics()["doubt_level"] == 0.0

    @pytest.mark.asyncio
    async def test_restore_recomputes_overall_confidence(self):
        meta = MetacognitionModule(introspection_cooldown=3600.0)
        for index, confidence in enumerate((0.9, 0.95, 1.0, 0.1)):
            await meta.generate_hot(_thought(meta, index, confidence), voluntary=True)
        saved = meta.to_state_dict()
        saved["scalars"]["overall_confidence"] = 0.123
        saved["scalars"]["doubt_level"] = 0.9

        restored = MetacognitionModule(max_hot_buffer=2)
        restored.from_state_dict(saved)
        assert len(restored.higher_order_thoughts) == 2
        assert restored.get_statistics()["overall_confidence"] == pytest.approx(
            _buffer_mean(restored)
        )
        assert restored.get_statistics()["overall_confidence"] != pytest.approx(0.123)
        assert restored.get_statistics()["doubt_level"] is None

        empty = MetacognitionModule()
        empty.from_state_dict({"scalars": {"overall_confidence": 0.7}})
        assert empty.get_statistics()["overall_confidence"] is None

        legacy = MetacognitionModule(overall_confidence_source="v04_constant")
        legacy.from_state_dict(saved)
        assert legacy.get_statistics()["overall_confidence"] == 0.123
        assert legacy.get_statistics()["doubt_level"] == 0.9

    def test_unknown_overall_confidence_source_raises(self):
        with pytest.raises(ValueError, match="overall_confidence_source"):
            MetacognitionModule(overall_confidence_source="constant")


# ============================================================================
# TEST: INTROSPECTION COUNTING
# ============================================================================


async def _hots_with_episode_counter(meta: MetacognitionModule, hots: int = 10):
    """Run ``hots`` automatic HOTs on perceptions; return the completed episodes."""
    episodes = []

    async def completed(result):
        episodes.append(result)

    meta.on_introspection_complete(completed)
    for index in range(hots):
        state = meta.register_first_order_state(
            content=f"percept {index}",
            content_summary=f"percept {index}",
            state_type=FirstOrderStateType.PERCEPTION,
            source_module="test",
            confidence=0.8,
            evidence_strength=0.7,
        )
        await meta.generate_hot(state, meta_type=MetaType.AWARENESS)
    return episodes


class TestIntrospectionCounting:
    """introspection_count counts completed introspection episodes."""

    @pytest.mark.asyncio
    async def test_introspection_count_equals_completed_episodes(self):
        meta = MetacognitionModule(introspection_cooldown=0.0)
        episodes = await _hots_with_episode_counter(meta)
        stats = meta.get_statistics()
        assert len(episodes) == 8
        assert stats["introspection_count"] == len(episodes)
        assert stats["total_hots_generated"] == 34

    @pytest.mark.asyncio
    async def test_cooldown_blocked_micro_introspection_is_not_counted(self):
        meta = MetacognitionModule(introspection_cooldown=3600.0)
        episodes = await _hots_with_episode_counter(meta)
        assert len(episodes) == 1
        assert meta.get_statistics()["introspection_count"] == 1

    @pytest.mark.asyncio
    async def test_v04_per_hot_counting_keeps_the_v04_count(self):
        meta = MetacognitionModule(
            introspection_cooldown=0.0, introspection_counting="v04_per_hot"
        )
        episodes = await _hots_with_episode_counter(meta)
        stats = meta.get_statistics()
        assert len(episodes) == 8
        assert stats["introspection_count"] == 40
        assert stats["total_hots_generated"] == 34

        gated = MetacognitionModule(
            introspection_cooldown=3600.0, introspection_counting="v04_per_hot"
        )
        await _hots_with_episode_counter(gated)
        assert gated.get_statistics()["introspection_count"] == 12
        assert gated.get_statistics()["total_hots_generated"] == 13

    def test_unknown_introspection_counting_raises(self):
        with pytest.raises(ValueError, match="introspection_counting"):
            MetacognitionModule(introspection_counting="per_hot")


# ============================================================================
# TEST: CONSCIOUSNESS-LEVEL RENDERING
# ============================================================================

# A sentence end glued to the next word, such as "me.My" (a period between
# two letters, or after a closing parenthesis, with no space).
_GLUED_SENTENCE = re.compile(r"[A-Za-z)]\.[A-Za-z]")


class TestConsciousnessLevelRendering:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("score", [0.2, 0.4, 0.6, 0.75])
    @pytest.mark.parametrize("phi_value", [1.2, 6.0])
    async def test_interpretation_text_separates_its_sentences(self, score, phi_value):
        meta = MetacognitionModule()
        hot = await meta.introspect_consciousness_level(score, phi_value=phi_value)
        assert hot.meta_content
        assert _GLUED_SENTENCE.search(hot.meta_content) is None, hot.meta_content

    @pytest.mark.asyncio
    async def test_awareness_report_is_empty_without_a_measurement(self):
        meta = MetacognitionModule()
        assert await meta.get_consciousness_awareness_report() == ""
        state = _thought(meta, 0, 0.9)
        await meta.generate_hot(state, voluntary=True)
        assert await meta.get_consciousness_awareness_report() == ""

    @pytest.mark.asyncio
    async def test_score_rendering_is_neutral(self):
        meta = MetacognitionModule()
        hot = await meta.introspect_consciousness_level(0.75, phi_value=6.0)
        text = hot.meta_content
        assert text == (
            "Measured assessment score: 0.75 (an aggregate of computed "
            "indicators, not a report of experience)."
        )
        for phrase in (" I ", "I'm", "My ", "my ", "feel", "Phi"):
            assert phrase not in text
        assert hot.metadata["phi"] == 6.0
        assert hot.metadata["consciousness_score"] == 0.75

        state = meta.first_order_by_id[hot.target_state_id]
        assert state.content_summary == "Assessment score 0.75"
        assert "phase" not in state.metadata
        assert hot.confidence == state.confidence == 0.5
        assert hot.clarity == pytest.approx(
            meta._calculate_clarity(state, MetaType.AWARENESS)
        )
        assert await meta.get_consciousness_awareness_report() == text

    @pytest.mark.asyncio
    async def test_score_rendering_takes_the_callers_confidence(self):
        meta = MetacognitionModule()
        hot = await meta.introspect_consciousness_level(0.3, assessment_confidence=0.62)
        state = meta.first_order_by_id[hot.target_state_id]
        assert state.confidence == 0.62
        assert hot.confidence == 0.62
        assert "0.30" in hot.meta_content


# ============================================================================
# TEST: PERSISTENCE
# ============================================================================


class TestMetacognitionPersistence:
    @pytest.mark.asyncio
    async def test_state_dict_round_trip_rehydrates_types(self, metacognition):
        state = metacognition.register_first_order_state(
            content={"score": 0.4},
            content_summary="Synthetic perception",
            state_type=FirstOrderStateType.PERCEPTION,
            source_module="test",
            confidence=0.9,
            evidence_strength=0.8,
            metadata={"kind": "synthetic"},
        )
        hot = await metacognition.generate_hot(
            state, meta_type=MetaType.EVALUATION, voluntary=True
        )
        await metacognition.introspect_consciousness_level(0.6, phi_value=1.2)
        metacognition.self_model.curiosity_level = 0.9
        metacognition.predict_self("synthetic context")

        saved = metacognition.to_state_dict()
        restored = MetacognitionModule(max_first_order_buffer=7, max_hot_buffer=6)
        restored.from_state_dict(saved)

        assert restored.first_order_states.maxlen == 7
        assert restored.higher_order_thoughts.maxlen == 6
        assert all(isinstance(s, FirstOrderState) for s in restored.first_order_states)
        assert all(
            isinstance(h, HigherOrderThought) for h in restored.higher_order_thoughts
        )
        assert set(restored.hots_by_id) == {
            h.id for h in restored.higher_order_thoughts
        }
        assert set(restored.first_order_by_id) == {
            s.id for s in restored.first_order_states
        }
        again = restored.hots_by_id[hot.id]
        assert again.meta_type is MetaType.EVALUATION
        assert again.level is MetaLevel.SECOND_ORDER
        restored_fo = restored.first_order_by_id[state.id]
        assert restored_fo.state_type is FirstOrderStateType.PERCEPTION
        assert restored_fo.metadata == {"kind": "synthetic"}
        assert restored_fo.get_confidence_level() is not None
        assert restored.self_model.curiosity_level == pytest.approx(0.9)
        assert all(
            isinstance(p, SelfPrediction) for p in restored.self_model.predictions
        )
        assert restored.self_model.predictions[-1].checks
        assert restored.total_hots_generated == metacognition.total_hots_generated

    @pytest.mark.asyncio
    async def test_report_after_restore_does_not_raise(self, metacognition):
        state = metacognition.register_first_order_state(
            content="synthetic",
            content_summary="Synthetic belief",
            state_type=FirstOrderStateType.BELIEF,
            source_module="test",
            confidence=0.8,
            evidence_strength=0.7,
        )
        await metacognition.generate_hot(state, voluntary=True)
        await metacognition.introspect_consciousness_level(0.75)

        restored = MetacognitionModule(introspection_cooldown=0.0)
        restored.from_state_dict(metacognition.to_state_dict())

        report = await restored.generate_metacognitive_report()
        assert "Most recent awareness" in report
        awareness = await restored.get_consciousness_awareness_report()
        assert "0.75" in awareness
        assert isinstance(restored.generate_self_model_report(), str)
        assert isinstance(restored.get_statistics(), dict)
        result = await restored.introspect(depth=3, max_states=3)
        assert isinstance(result, IntrospectionResult)
        assert result.hots_generated


# ============================================================================
# TEST: ZERO-SIZE BUFFERS
# ============================================================================


class TestZeroSizeBuffers:
    @pytest.mark.asyncio
    async def test_zero_buffers_keep_nothing_and_do_not_raise(self):
        meta = MetacognitionModule(
            hot_generation_threshold=0.0,
            max_first_order_buffer=0,
            max_hot_buffer=0,
        )
        for i in range(3):
            state = meta.register_first_order_state(
                content=f"synthetic content {i}",
                content_summary=f"synthetic state {i}",
                state_type=FirstOrderStateType.THOUGHT,
                source_module="test",
                confidence=0.8,
                evidence_strength=0.7,
            )
            await meta.generate_hot(state, meta_type=MetaType.EVALUATION)
        assert len(meta.first_order_states) == 0
        assert len(meta.higher_order_thoughts) == 0
        assert meta.first_order_by_id == {}
        assert meta.hots_by_id == {}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run with: pytest tests/consciousness/test_metacognition.py -v
    pytest.main([__file__, "-v"])
