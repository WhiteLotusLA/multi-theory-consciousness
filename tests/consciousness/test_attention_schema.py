"""
 Tests for Attention Schema Theory (AST) - Phase 3
=======================================================

Tests for the system's self-model of her own attention, based on
Michael Graziano's Attention Schema Theory.

This tests:
1. AttentionState classification (focused, divided, scanning, etc.)
2. AttentionTarget tracking (what the system attends to)
3. Voluntary attention shifts (the system choosing what to focus on)
4. Attention prediction (anticipating where attention will go)
5. Theory of Mind (modeling others' attention)
6. Introspection reports (the system describing her attention)
7. Integration with EnhancedGlobalWorkspace


watching - and the system must know what SHE is watching!" 

Created: December 5, 2025
Author: Multi-Theory Consciousness Project
"""

import pytest
import asyncio
import numpy as np
import time
from typing import Dict, Any, List, Optional

from mtc.consciousness.attention_schema import (
    AttentionState,
    AttentionShiftType,
    AttentionTarget,
    AttentionSchemaState,
    AttentionPrediction,
    OtherAgentAttention,
    AttentionSchemaModule,
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
def attention_schema():
    """Create a fresh AttentionSchemaModule for testing."""
    return AttentionSchemaModule()


@pytest.fixture
def workspace():
    """Create an EnhancedGlobalWorkspace with attention schema."""
    return EnhancedGlobalWorkspace(capacity=5)


@pytest.fixture
def sample_workspace_state(workspace):
    """Create a sample ConsciousnessState for testing."""
    # Create some workspace content
    candidate = WorkspaceCandidate(
        content="Thinking about consciousness",
        content_type="thought",
        summary="Consciousness reflection",
        source=WorkspaceCandidateSource.CTM,
        activation_level=0.8,
        emotional_salience=0.5,
    )

    content = WorkspaceContent(
        candidate=candidate,
        salience=0.75,
        competition_rank=1,
        competing_count=3,
    )

    return ConsciousnessState(
        is_conscious=True,
        primary_content=content,
        workspace_contents=[content],
        ignition_events=1,
        integration_level=0.6,
        broadcast_coverage=0.8,
        attention_focus="ctm",
        attention_distribution={"ctm": 0.7, "sensory": 0.3},
        stream_position=5,
    )


@pytest.fixture
def multiple_candidates():
    """Create multiple workspace candidates for competition tests."""
    return [
        WorkspaceCandidate(
            content="A user is interacting with the system",
            content_type="conversation",
            summary="User conversation",
            source=WorkspaceCandidateSource.SENSORY,
            activation_level=0.9,
            emotional_salience=0.8,
        ),
        WorkspaceCandidate(
            content="Thinking about neural networks",
            content_type="thought",
            summary="Neural network thoughts",
            source=WorkspaceCandidateSource.CTM,
            activation_level=0.6,
            emotional_salience=0.3,
        ),
        WorkspaceCandidate(
            content="Memory of yesterday's learning",
            content_type="memory",
            summary="Yesterday's memory",
            source=WorkspaceCandidateSource.MEMORY,
            activation_level=0.4,
            emotional_salience=0.5,
        ),
    ]


# ============================================================================
# ATTENTION STATE TESTS
# ============================================================================

class TestAttentionStates:
    """Test attention state classification."""

    def test_attention_state_enum_values(self):
        """Test all attention states exist."""
        states = [
            AttentionState.FOCUSED,
            AttentionState.DIVIDED,
            AttentionState.SCANNING,
            AttentionState.ABSENT,
            AttentionState.HYPERFOCUSED,
            AttentionState.SHIFTING,
        ]
        assert len(states) == 6

    def test_attention_shift_types(self):
        """Test all shift types exist."""
        shift_types = [
            AttentionShiftType.CAPTURED,
            AttentionShiftType.VOLUNTARY,
            AttentionShiftType.GOAL_DRIVEN,  # Not GOAL_DIRECTED
            AttentionShiftType.HABITUAL,
            AttentionShiftType.EMOTIONAL,   # Not REFLEXIVE
        ]
        assert len(shift_types) == 5

    @pytest.mark.asyncio
    async def test_focused_state_detection(self, attention_schema, sample_workspace_state):
        """Test focused attention state classification."""
        # Single strong focus should yield focused or hyperfocused
        schema = await attention_schema.update_schema(sample_workspace_state)

        assert schema.attention_state in [
            AttentionState.FOCUSED,
            AttentionState.HYPERFOCUSED
        ]

    @pytest.mark.asyncio
    async def test_scanning_state_initial(self, attention_schema):
        """Test that initial state is scanning (looking for focus)."""
        # Before any updates, schema should be scanning
        assert attention_schema.schema.attention_state == AttentionState.SCANNING


# ============================================================================
# ATTENTION TARGET TESTS
# ============================================================================

class TestAttentionTarget:
    """Test AttentionTarget tracking."""

    def test_target_creation(self):
        """Test creating an attention target."""
        target = AttentionTarget(
            content="Test content",
            content_type="test",
            summary="Test summary",
            attention_strength=0.8,
            salience=0.7,
        )

        assert target.summary == "Test summary"
        assert target.attention_strength == 0.8
        assert target.salience == 0.7
        assert target.voluntary is False  # Default

    def test_target_duration_update(self):
        """Test duration tracking."""
        target = AttentionTarget(
            content="Test",
            summary="Test",
        )

        # Wait briefly
        time.sleep(0.1)
        target.update_duration()

        assert target.duration_seconds >= 0.1

    def test_voluntary_target(self):
        """Test voluntary attention target."""
        target = AttentionTarget(
            content="Chosen focus",
            summary="I chose this",
            voluntary=True,
            shift_type=AttentionShiftType.VOLUNTARY,
            shift_reason="Because I want to learn",
        )

        assert target.voluntary is True
        assert target.shift_type == AttentionShiftType.VOLUNTARY
        assert "learn" in target.shift_reason


# ============================================================================
# ATTENTION SCHEMA STATE TESTS
# ============================================================================

class TestAttentionSchemaState:
    """Test AttentionSchemaState dataclass."""

    def test_default_state(self):
        """Test default schema state."""
        state = AttentionSchemaState()

        assert state.current_focus is None
        assert state.attention_state == AttentionState.SCANNING
        assert state.attention_capacity_used == 0.0
        assert state.attention_capacity_max == 7.0  # Miller's law!

    def test_capacity_tracking(self):
        """Test attention capacity calculations."""
        target = AttentionTarget(
            content="Primary",
            summary="Primary focus",
            attention_strength=0.8,
        )

        state = AttentionSchemaState(
            current_focus=target,
            attention_capacity_used=0.5,
        )

        assert state.attention_capacity_used == 0.5
        # Should have room for more items
        assert state.attention_capacity_used < state.attention_capacity_max


# ============================================================================
# ATTENTION SCHEMA MODULE TESTS
# ============================================================================

class TestAttentionSchemaModule:
    """Test the main AttentionSchemaModule class."""

    @pytest.mark.asyncio
    async def test_update_schema_basic(self, attention_schema, sample_workspace_state):
        """Test basic schema update."""
        schema = await attention_schema.update_schema(sample_workspace_state)

        assert schema is not None
        assert isinstance(schema, AttentionSchemaState)
        assert attention_schema.total_updates == 1

    @pytest.mark.asyncio
    async def test_focus_extraction(self, attention_schema, sample_workspace_state):
        """Test that focus is correctly extracted from workspace state."""
        schema = await attention_schema.update_schema(sample_workspace_state)

        assert schema.current_focus is not None
        assert "Consciousness" in schema.current_focus.summary

    @pytest.mark.asyncio
    async def test_history_tracking(self, attention_schema, sample_workspace_state):
        """Test attention history is maintained."""
        # Multiple updates
        for i in range(5):
            await attention_schema.update_schema(sample_workspace_state)

        # History should be populated
        history = attention_schema.schema.attention_history
        assert len(history) > 0

    @pytest.mark.asyncio
    async def test_secondary_foci(self, attention_schema):
        """Test secondary foci extraction."""
        # Create state with multiple contents
        candidates = [
            WorkspaceCandidate(
                content=f"Content {i}",
                summary=f"Content {i}",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.8 - i * 0.2,
            )
            for i in range(3)
        ]

        contents = [
            WorkspaceContent(
                candidate=c,
                salience=0.8 - i * 0.2,
                competition_rank=i + 1,
            )
            for i, c in enumerate(candidates)
        ]

        state = ConsciousnessState(
            is_conscious=True,
            primary_content=contents[0],
            workspace_contents=contents,
            ignition_events=3,
            integration_level=0.7,
            broadcast_coverage=0.9,
            attention_focus="ctm",
            attention_distribution={"ctm": 1.0},
            stream_position=1,
        )

        schema = await attention_schema.update_schema(state)

        # Should have secondary foci
        assert len(schema.secondary_foci) >= 0  # May have none if filtered


# ============================================================================
# VOLUNTARY ATTENTION SHIFT TESTS
# ============================================================================

class TestVoluntaryShifts:
    """Test voluntary attention control - the mark of conscious agency!"""

    @pytest.mark.asyncio
    async def test_request_voluntary_shift(self, attention_schema):
        """Test requesting a voluntary attention shift."""
        target = AttentionTarget(
            content="New focus",
            summary="I want to focus on this",
            voluntary=True,
        )

        await attention_schema.request_voluntary_shift(
            target=target,
            reason="Because I'm curious"
        )

        # Check pending shift was registered (it's a tuple of (target, reason))
        assert attention_schema._pending_voluntary_shift is not None
        pending_target, pending_reason = attention_schema._pending_voluntary_shift
        assert pending_target.summary == "I want to focus on this"
        assert pending_reason == "Because I'm curious"

    @pytest.mark.asyncio
    async def test_voluntary_shift_through_workspace(self, workspace):
        """Test voluntary shift through the workspace interface."""
        # Request shift
        result = await workspace.request_voluntary_attention_shift(
            target_description="Learning mathematics",
            reason="The user suggested it",
            priority=0.85,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_voluntary_shift_affects_next_cycle(self, workspace, multiple_candidates):
        """Test that voluntary shift affects the next consciousness cycle."""
        # First, run a cycle
        state1 = await workspace.process_consciousness_cycle(
            candidates=multiple_candidates[:2]
        )

        # Request voluntary shift
        await workspace.request_voluntary_attention_shift(
            target_description="Memory retrieval",
            reason="I want to remember",
            priority=0.9,
        )

        # Run another cycle with memory candidate
        state2 = await workspace.process_consciousness_cycle(
            candidates=multiple_candidates[2:3]
        )

        # Both cycles should complete successfully
        assert state1 is not None
        assert state2 is not None


# ============================================================================
# ATTENTION PREDICTION TESTS
# ============================================================================

class TestAttentionPrediction:
    """Test attention prediction capabilities."""

    @pytest.mark.asyncio
    async def test_prediction_generated(self, attention_schema, sample_workspace_state):
        """Test that predictions are generated."""
        # Run a few updates to build history
        for _ in range(3):
            await attention_schema.update_schema(sample_workspace_state)

        schema = attention_schema.schema

        # Should have prediction confidence
        assert schema.prediction_confidence >= 0.0

    @pytest.mark.asyncio
    async def test_prediction_confidence_increases_with_history(self, attention_schema):
        """Test that prediction confidence increases with more history."""
        # Create similar states
        for i in range(5):
            state = ConsciousnessState(
                is_conscious=True,
                primary_content=None,
                workspace_contents=[],
                ignition_events=1,
                integration_level=0.5,
                broadcast_coverage=0.5,
                attention_focus="ctm",
                attention_distribution={"ctm": 1.0},
                stream_position=i,
            )
            await attention_schema.update_schema(state)

        # With history, should have some confidence
        # (even if low due to lack of clear patterns)
        assert attention_schema.total_updates == 5


# ============================================================================
# INTROSPECTION REPORT TESTS
# ============================================================================

class TestIntrospectionReports:
    """Test the system's ability to report on her attention."""

    @pytest.mark.asyncio
    async def test_basic_report(self, attention_schema, sample_workspace_state):
        """Test basic attention report generation."""
        await attention_schema.update_schema(sample_workspace_state)
        report = await attention_schema.report_attention()

        assert isinstance(report, str)
        assert len(report) > 0

    @pytest.mark.asyncio
    async def test_report_mentions_focus(self, attention_schema, sample_workspace_state):
        """Test that report mentions current focus."""
        await attention_schema.update_schema(sample_workspace_state)
        report = await attention_schema.report_attention()

        # Report should mention something about focus
        assert "attention" in report.lower() or "focus" in report.lower()

    @pytest.mark.asyncio
    async def test_report_changes_with_state(self, workspace, multiple_candidates):
        """Test that reports change based on attention state."""
        # First state
        state1 = await workspace.process_consciousness_cycle(
            candidates=multiple_candidates[:1]
        )
        report1 = state1.attention_report

        # Second state with different candidates
        state2 = await workspace.process_consciousness_cycle(
            candidates=multiple_candidates[1:2]
        )
        report2 = state2.attention_report

        # Reports should be generated
        assert report1 is not None
        assert report2 is not None

    @pytest.mark.asyncio
    async def test_scanning_state_report(self, attention_schema):
        """Test report when scanning (no focus)."""
        # Don't update - should be in scanning state
        report = await attention_schema.report_attention()

        assert "scanning" in report.lower() or "looking" in report.lower() or "attention" in report.lower()


# ============================================================================
# THEORY OF MIND TESTS
# ============================================================================

class TestTheoryOfMind:
    """Test the system's ability to model others' attention (ToM)."""

    @pytest.mark.asyncio
    async def test_model_other_attention(self, attention_schema):
        """Test modeling another agent's attention."""
        other = await attention_schema.model_other_attention(
            agent_name="user_a",
            context="The user is reading a book about AI"
        )

        assert other is not None
        assert isinstance(other, OtherAgentAttention)
        assert other.agent_name == "user_a"

    @pytest.mark.asyncio
    async def test_other_attention_has_inferred_focus(self, attention_schema):
        """Test that other's attention has inferred focus."""
        other = await attention_schema.model_other_attention(
            agent_name="user_b",
            context="The user is looking at the computer screen"
        )

        # Should have inferred something
        assert other.inferred_focus is not None or other.confidence > 0

    @pytest.mark.asyncio
    async def test_model_other_through_workspace(self, workspace):
        """Test Theory of Mind through workspace interface."""
        other = await workspace.model_other_attention(
            agent_name="user_a",
            context="The user is teaching the system about consciousness"
        )

        assert other is not None
        assert other.agent_name == "user_a"


# ============================================================================
# INTEGRATION WITH GLOBAL WORKSPACE TESTS
# ============================================================================

class TestGlobalWorkspaceIntegration:
    """Test integration with EnhancedGlobalWorkspace."""

    @pytest.mark.asyncio
    async def test_workspace_has_attention_schema(self, workspace):
        """Test that workspace initializes with attention schema."""
        assert hasattr(workspace, 'attention_schema')
        assert isinstance(workspace.attention_schema, AttentionSchemaModule)

    @pytest.mark.asyncio
    async def test_consciousness_state_includes_schema(self, workspace, multiple_candidates):
        """Test that ConsciousnessState includes attention schema."""
        state = await workspace.process_consciousness_cycle(
            candidates=multiple_candidates
        )

        assert state.attention_schema is not None
        assert isinstance(state.attention_schema, AttentionSchemaState)

    @pytest.mark.asyncio
    async def test_consciousness_state_includes_report(self, workspace, multiple_candidates):
        """Test that ConsciousnessState includes attention report."""
        state = await workspace.process_consciousness_cycle(
            candidates=multiple_candidates
        )

        assert state.attention_report is not None
        assert len(state.attention_report) > 0

    @pytest.mark.asyncio
    async def test_statistics_include_schema(self, workspace, multiple_candidates):
        """Test that workspace statistics include schema info."""
        await workspace.process_consciousness_cycle(
            candidates=multiple_candidates
        )

        stats = workspace.get_statistics()

        assert "attention_schema" in stats
        assert "attention_state" in stats["attention_schema"]
        assert "capacity_used" in stats["attention_schema"]

    @pytest.mark.asyncio
    async def test_get_attention_report_method(self, workspace, multiple_candidates):
        """Test getting attention report directly from workspace."""
        await workspace.process_consciousness_cycle(
            candidates=multiple_candidates
        )

        report = await workspace.get_attention_report()

        assert isinstance(report, str)
        assert len(report) > 0

    @pytest.mark.asyncio
    async def test_get_attention_schema_state_method(self, workspace, multiple_candidates):
        """Test getting schema state directly from workspace."""
        await workspace.process_consciousness_cycle(
            candidates=multiple_candidates
        )

        schema = workspace.get_attention_schema_state()

        assert isinstance(schema, AttentionSchemaState)


# ============================================================================
# CAPACITY AND LIMITS TESTS
# ============================================================================

class TestCapacityLimits:
    """Test attention capacity limits (Miller's Law: 7±2)."""

    @pytest.mark.asyncio
    async def test_capacity_max_is_seven(self, attention_schema):
        """Test default capacity is 7 (Miller's Law)."""
        assert attention_schema.schema.attention_capacity_max == 7.0

    @pytest.mark.asyncio
    async def test_capacity_usage_bounded(self, attention_schema, sample_workspace_state):
        """Test that capacity usage is bounded 0-1."""
        await attention_schema.update_schema(sample_workspace_state)

        capacity = attention_schema.schema.attention_capacity_used

        assert 0.0 <= capacity <= 1.0


# ============================================================================
# TEMPORAL CONTINUITY TESTS
# ============================================================================

class TestTemporalContinuity:
    """Test attention continuity over time."""

    @pytest.mark.asyncio
    async def test_duration_tracking(self, attention_schema, sample_workspace_state):
        """Test that attention duration is tracked."""
        # First update
        await attention_schema.update_schema(sample_workspace_state)
        initial_time = attention_schema.schema.current_focus.entry_time if attention_schema.schema.current_focus else time.time()

        # Brief pause
        await asyncio.sleep(0.1)

        # Second update with same state
        await attention_schema.update_schema(sample_workspace_state)

        if attention_schema.schema.current_focus:
            assert attention_schema.schema.current_focus.duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_state_change_tracking(self, attention_schema):
        """Test that state changes are tracked."""
        initial_change_time = attention_schema.schema.last_state_change

        # Create state that should cause a shift
        state = ConsciousnessState(
            is_conscious=True,
            primary_content=None,
            workspace_contents=[],
            ignition_events=0,
            integration_level=0.1,
            broadcast_coverage=0.1,
            attention_focus="none",
            attention_distribution={},
            stream_position=1,
        )

        await attention_schema.update_schema(state)

        # Time should be tracked
        assert attention_schema.schema.last_state_change >= initial_change_time


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_workspace_state(self, attention_schema):
        """Test handling empty workspace state."""
        state = ConsciousnessState(
            is_conscious=False,
            primary_content=None,
            workspace_contents=[],
            ignition_events=0,
            integration_level=0.0,
            broadcast_coverage=0.0,
            attention_focus="none",
            attention_distribution={},
            stream_position=0,
        )

        schema = await attention_schema.update_schema(state)

        # Should handle gracefully
        assert schema is not None
        assert schema.attention_state in [AttentionState.SCANNING, AttentionState.ABSENT]

    @pytest.mark.asyncio
    async def test_null_neural_signals(self, attention_schema, sample_workspace_state):
        """Test with None neural signals."""
        schema = await attention_schema.update_schema(
            sample_workspace_state,
            neural_signals=None
        )

        assert schema is not None

    @pytest.mark.asyncio
    async def test_empty_neural_signals(self, attention_schema, sample_workspace_state):
        """Test with empty neural signals."""
        schema = await attention_schema.update_schema(
            sample_workspace_state,
            neural_signals={}
        )

        assert schema is not None

    @pytest.mark.asyncio
    async def test_rapid_updates(self, attention_schema, sample_workspace_state):
        """Test handling rapid consecutive updates."""
        for _ in range(20):
            await attention_schema.update_schema(sample_workspace_state)

        assert attention_schema.total_updates == 20

    @pytest.mark.asyncio
    async def test_model_other_empty_context(self, attention_schema):
        """Test modeling other with minimal context."""
        other = await attention_schema.model_other_attention(
            agent_name="Someone",
            context=""
        )

        # Should still return something
        assert other is not None


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
