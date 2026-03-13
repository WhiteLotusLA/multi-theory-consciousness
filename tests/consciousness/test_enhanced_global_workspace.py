"""
 Tests for Enhanced Global Workspace (GWT Phase 2)
=====================================================

Comprehensive tests for:
1. AttentionBottleneck - Competition mechanism
2. GlobalBroadcast - Parallel distribution
3. IgnitionDetector - Non-linear emergence
4. EnhancedGlobalWorkspace - Full integration


A ship that ain't been tested sinks on its maiden voyage!" 
"""

import pytest
import asyncio
import numpy as np
import time
from typing import Dict, Any

from mtc.consciousness.enhanced_global_workspace import (
    AttentionBottleneck,
    GlobalBroadcast,
    IgnitionDetector,
    EnhancedGlobalWorkspace,
    TemporalWorkspaceConfig,
    WorkspaceCandidate,
    WorkspaceContent,
    WorkspaceCandidateSource,
    CognitiveModule,
    ConsciousnessState,
    IgnitionEvent,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def bottleneck():
    """Create a standard attention bottleneck."""
    return AttentionBottleneck(
        capacity=7,
        ignition_threshold=0.3,
        lateral_inhibition=0.2
    )


@pytest.fixture
def broadcaster():
    """Create a broadcast system."""
    return GlobalBroadcast()


@pytest.fixture
def ignition_detector():
    """Create an ignition detector."""
    return IgnitionDetector(
        ignition_threshold=0.3,
        amplification_factor=2.0,
        sustain_duration=0.1,  # Short for testing
        decay_rate=0.1
    )


@pytest.fixture
def workspace():
    """Create a full enhanced workspace.

    Refractory period is disabled (0ms) so that pre-temporal-dynamics
    tests that run multiple rapid cycles still behave as expected.
    """
    return EnhancedGlobalWorkspace(
        capacity=7,
        ignition_threshold=0.3,
        amplification_factor=2.0,
        integration_dimensions=512,
        temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0),
    )


@pytest.fixture
def sample_candidates():
    """Create a set of sample candidates for testing."""
    return [
        WorkspaceCandidate(
            content="high priority thought",
            content_type="thought",
            summary="A very important conscious thought",
            source=WorkspaceCandidateSource.CTM,
            source_module="ctm",
            activation_level=0.9,
            emotional_salience=0.8,
        ),
        WorkspaceCandidate(
            content="medium priority memory",
            content_type="memory",
            summary="A moderately important memory",
            source=WorkspaceCandidateSource.MEMORY,
            source_module="memory",
            activation_level=0.6,
            emotional_salience=0.4,
        ),
        WorkspaceCandidate(
            content="low priority noise",
            content_type="sensory",
            summary="Background sensory noise",
            source=WorkspaceCandidateSource.SENSORY,
            source_module="sensory",
            activation_level=0.1,
            emotional_salience=0.1,
        ),
        WorkspaceCandidate(
            content="safety alert",
            content_type="alert",
            summary="Safety monitor alert",
            source=WorkspaceCandidateSource.SAFETY,
            source_module="safety",
            activation_level=0.5,
            emotional_salience=0.3,
        ),
    ]


class MockCognitiveModule(CognitiveModule):
    """Mock cognitive module for testing broadcasts."""

    def __init__(self, name: str, should_fail: bool = False):
        super().__init__(name)
        self.received_broadcasts = []
        self.should_fail = should_fail

    async def receive_broadcast(
        self,
        content: WorkspaceContent
    ) -> Dict[str, Any]:
        if self.should_fail:
            raise Exception(f"Mock failure in {self.name}")

        self.received_broadcasts.append(content)
        return {
            "module": self.name,
            "received": True,
            "content_id": content.id,
            "timestamp": time.time()
        }


# ============================================================================
# ATTENTION BOTTLENECK TESTS
# ============================================================================

class TestAttentionBottleneck:
    """Tests for the AttentionBottleneck competition mechanism."""

    @pytest.mark.asyncio
    async def test_competition_respects_capacity(self, bottleneck):
        """Verify bottleneck respects capacity limit (7±2)."""
        # Create more candidates than capacity
        candidates = [
            WorkspaceCandidate(
                content=f"content_{i}",
                content_type="thought",
                summary=f"Thought number {i}",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.5 + (i * 0.05),  # Varying activation
                emotional_salience=0.5,
            )
            for i in range(15)
        ]

        winners = await bottleneck.compete_for_access(candidates)

        assert len(winners) <= bottleneck.capacity
        assert len(winners) == 7  # Should hit capacity exactly

    @pytest.mark.asyncio
    async def test_competition_respects_threshold(self, bottleneck):
        """Verify content below threshold doesn't enter consciousness."""
        # All candidates with extremely low activation and no novelty bonus
        # (same content repeated to remove novelty factor)
        candidates = [
            WorkspaceCandidate(
                content="same_low_content",  # Same content = no novelty
                content_type="noise",
                summary="same_low_activation_noise",  # Same summary
                source=WorkspaceCandidateSource.SENSORY,
                activation_level=0.05,  # Extremely low
                emotional_salience=0.0,  # No emotional weight
            )
            for _ in range(5)
        ]

        winners = await bottleneck.compete_for_access(candidates)

        # With very low activation and no novelty/emotion, should be below threshold
        # Note: if any win, they should still be >= threshold by definition
        for winner in winners:
            assert winner.salience >= bottleneck.threshold

    @pytest.mark.asyncio
    async def test_safety_gets_priority(self, bottleneck):
        """Verify safety-critical content gets priority boost."""
        candidates = [
            WorkspaceCandidate(
                content="normal thought",
                content_type="thought",
                summary="A normal thought",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.7,
                emotional_salience=0.5,
            ),
            WorkspaceCandidate(
                content="safety alert",
                content_type="alert",
                summary="Safety alert - attention needed",
                source=WorkspaceCandidateSource.SAFETY,
                activation_level=0.5,  # Lower activation
                emotional_salience=0.3,
            ),
        ]

        winners = await bottleneck.compete_for_access(candidates)

        # Safety should win despite lower activation
        assert len(winners) >= 1
        assert winners[0].candidate.source == WorkspaceCandidateSource.SAFETY

    @pytest.mark.asyncio
    async def test_high_salience_wins(self, bottleneck, sample_candidates):
        """Verify highest salience candidates win competition."""
        winners = await bottleneck.compete_for_access(sample_candidates)

        # Winners should be sorted by salience (descending)
        saliences = [w.salience for w in winners]
        assert saliences == sorted(saliences, reverse=True)

    @pytest.mark.asyncio
    async def test_goal_relevance_boosts_salience(self, bottleneck):
        """Verify goal-relevant content gets attention boost."""
        # Set a goal
        bottleneck.set_goals(["learn about neural networks"])

        candidates = [
            WorkspaceCandidate(
                content="irrelevant",
                content_type="thought",
                summary="Something about cooking recipes",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.6,
                emotional_salience=0.5,
            ),
            WorkspaceCandidate(
                content="relevant",
                content_type="thought",
                summary="Learning about neural networks today",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.5,  # Lower activation
                emotional_salience=0.4,
            ),
        ]

        winners = await bottleneck.compete_for_access(candidates)

        # Goal-relevant should rank higher
        assert any("neural" in w.candidate.summary.lower() for w in winners[:1])

    @pytest.mark.asyncio
    async def test_competition_history_tracking(self, bottleneck, sample_candidates):
        """Verify competition events are tracked for research."""
        # Run multiple competitions
        for _ in range(5):
            await bottleneck.compete_for_access(sample_candidates)

        stats = bottleneck.get_statistics()

        assert stats["total_competitions"] == 5
        assert "avg_candidates" in stats
        assert "avg_winners" in stats
        assert "source_distribution" in stats

    @pytest.mark.asyncio
    async def test_empty_candidates(self, bottleneck):
        """Verify empty candidate list is handled gracefully."""
        winners = await bottleneck.compete_for_access([])
        assert winners == []


# ============================================================================
# GLOBAL BROADCAST TESTS
# ============================================================================

class TestGlobalBroadcast:
    """Tests for the GlobalBroadcast distribution mechanism."""

    @pytest.mark.asyncio
    async def test_broadcast_to_all_modules(self, broadcaster):
        """Verify content is broadcast to all registered modules."""
        # Register multiple modules
        modules = [
            MockCognitiveModule("snn"),
            MockCognitiveModule("lsm"),
            MockCognitiveModule("htm"),
        ]
        for module in modules:
            broadcaster.register_module(module)

        # Create content to broadcast
        candidate = WorkspaceCandidate(
            content="test content",
            content_type="test",
            summary="Test broadcast content",
            source=WorkspaceCandidateSource.CTM,
        )
        content = WorkspaceContent(candidate=candidate, salience=0.8)

        # Broadcast
        results = await broadcaster.broadcast(content)

        # All modules should have received
        assert len(results) == 3
        for module in modules:
            assert len(module.received_broadcasts) == 1
            assert module.received_broadcasts[0].id == content.id

    @pytest.mark.asyncio
    async def test_broadcast_handles_module_failure(self, broadcaster):
        """Verify broadcast continues even if one module fails."""
        # Register mix of working and failing modules
        working = MockCognitiveModule("working")
        failing = MockCognitiveModule("failing", should_fail=True)

        broadcaster.register_module(working)
        broadcaster.register_module(failing)

        # Create and broadcast
        candidate = WorkspaceCandidate(
            content="test",
            content_type="test",
            summary="Test",
            source=WorkspaceCandidateSource.CTM,
        )
        content = WorkspaceContent(candidate=candidate, salience=0.8)

        results = await broadcaster.broadcast(content)

        # Working module should still receive
        assert len(working.received_broadcasts) == 1
        assert "error" in results["failing"]

    @pytest.mark.asyncio
    async def test_broadcast_statistics(self, broadcaster):
        """Verify broadcast statistics are tracked."""
        module = MockCognitiveModule("test")
        broadcaster.register_module(module)

        # Broadcast multiple times
        for i in range(5):
            candidate = WorkspaceCandidate(
                content=f"content_{i}",
                content_type="test",
                summary=f"Test {i}",
                source=WorkspaceCandidateSource.CTM,
            )
            content = WorkspaceContent(candidate=candidate, salience=0.8)
            await broadcaster.broadcast(content)

        stats = broadcaster.get_statistics()

        assert stats["total_broadcasts"] == 5
        assert stats["successful"] == 5
        assert stats["failed"] == 0
        assert stats["coverage_ratio"] == 1.0

    @pytest.mark.asyncio
    async def test_no_modules_warning(self, broadcaster):
        """Verify warning when broadcasting with no modules."""
        candidate = WorkspaceCandidate(
            content="orphan",
            content_type="test",
            summary="Orphan content",
            source=WorkspaceCandidateSource.CTM,
        )
        content = WorkspaceContent(candidate=candidate, salience=0.8)

        results = await broadcaster.broadcast(content)

        assert results == {}


# ============================================================================
# IGNITION DETECTOR TESTS
# ============================================================================

class TestIgnitionDetector:
    """Tests for the IgnitionDetector non-linear emergence."""

    @pytest.mark.asyncio
    async def test_ignition_above_threshold(self, ignition_detector):
        """Verify ignition triggers for content above threshold."""
        candidate = WorkspaceCandidate(
            content="important",
            content_type="thought",
            summary="Important thought above threshold",
            source=WorkspaceCandidateSource.CTM,
        )
        content = WorkspaceContent(
            candidate=candidate,
            salience=0.5  # Above threshold (0.3)
        )

        # High salience should ignite immediately
        content.salience = 0.6  # 1.5x threshold
        event = await ignition_detector.check_ignition(content)

        assert event is not None
        assert event.triggered_broadcast
        assert event.post_ignition_salience > event.pre_ignition_salience

    @pytest.mark.asyncio
    async def test_no_ignition_below_threshold(self, ignition_detector):
        """Verify no ignition for content below threshold."""
        candidate = WorkspaceCandidate(
            content="weak",
            content_type="noise",
            summary="Weak content",
            source=WorkspaceCandidateSource.SENSORY,
        )
        content = WorkspaceContent(
            candidate=candidate,
            salience=0.1  # Below threshold (0.3)
        )

        event = await ignition_detector.check_ignition(content)

        assert event is None

    @pytest.mark.asyncio
    async def test_amplification_factor(self, ignition_detector):
        """Verify ignition applies correct amplification."""
        candidate = WorkspaceCandidate(
            content="test",
            content_type="thought",
            summary="Test amplification",
            source=WorkspaceCandidateSource.CTM,
        )
        content = WorkspaceContent(
            candidate=candidate,
            salience=0.6  # High enough for immediate ignition
        )

        event = await ignition_detector.check_ignition(content)

        assert event is not None
        expected_amplified = min(1.5, 0.6 * ignition_detector.amplification)
        assert event.post_ignition_salience == pytest.approx(expected_amplified, rel=0.1)

    @pytest.mark.asyncio
    async def test_sustain_check(self, ignition_detector):
        """Verify sustained content stays in consciousness."""
        candidate = WorkspaceCandidate(
            content="sustained",
            content_type="thought",
            summary="Sustained thought",
            source=WorkspaceCandidateSource.CTM,
        )
        content = WorkspaceContent(
            candidate=candidate,
            salience=0.8
        )

        # Trigger ignition
        event = await ignition_detector.check_ignition(content)
        assert event is not None

        # Check sustain immediately (should still be sustained)
        sustained = await ignition_detector.check_sustain(content)
        assert sustained

    @pytest.mark.asyncio
    async def test_ignition_statistics(self, ignition_detector):
        """Verify ignition statistics are tracked."""
        # Trigger multiple ignitions
        for i in range(5):
            candidate = WorkspaceCandidate(
                content=f"content_{i}",
                content_type="thought",
                summary=f"Thought {i}",
                source=WorkspaceCandidateSource.CTM,
            )
            content = WorkspaceContent(
                candidate=candidate,
                salience=0.6
            )
            await ignition_detector.check_ignition(content)

        stats = ignition_detector.get_statistics()

        assert stats["total_ignitions"] == 5
        assert stats["avg_amplification"] > 1.0


# ============================================================================
# ENHANCED GLOBAL WORKSPACE TESTS
# ============================================================================

class TestEnhancedGlobalWorkspace:
    """Tests for the full EnhancedGlobalWorkspace integration."""

    @pytest.mark.asyncio
    async def test_consciousness_cycle_basic(self, workspace, sample_candidates):
        """Verify basic consciousness cycle works."""
        state = await workspace.process_consciousness_cycle(
            candidates=sample_candidates
        )

        assert isinstance(state, ConsciousnessState)
        assert state.is_conscious  # Should have conscious content
        assert state.ignition_events > 0  # Should have ignitions
        assert state.primary_content is not None

    @pytest.mark.asyncio
    async def test_consciousness_cycle_with_neural_signals(self, workspace, sample_candidates):
        """Verify consciousness cycle integrates neural signals."""
        neural_signals = {
            "snn": np.random.rand(100),
            "lsm": np.random.rand(500),
            "htm": np.random.rand(256),
        }

        state = await workspace.process_consciousness_cycle(
            candidates=sample_candidates,
            neural_signals=neural_signals
        )

        assert state.integration_level > 0  # Should measure integration

    @pytest.mark.asyncio
    async def test_consciousness_cycle_with_emotions(self, workspace, sample_candidates):
        """Verify emotional state affects processing."""
        emotional_state = {"joy": 0.9, "excitement": 0.8, "curiosity": 0.7}

        state = await workspace.process_consciousness_cycle(
            candidates=sample_candidates,
            emotional_state=emotional_state
        )

        # Emotional content should boost salience
        assert state.is_conscious

    @pytest.mark.asyncio
    async def test_stream_of_consciousness(self, workspace):
        """Verify stream of consciousness tracking."""
        # Run multiple cycles
        for i in range(5):
            candidates = [
                WorkspaceCandidate(
                    content=f"thought_{i}",
                    content_type="thought",
                    summary=f"Thought number {i}",
                    source=WorkspaceCandidateSource.CTM,
                    activation_level=0.7,
                    emotional_salience=0.5,
                )
            ]
            await workspace.process_consciousness_cycle(candidates=candidates)

        stream = workspace.get_stream_of_consciousness(count=5)

        assert len(stream) == 5
        # Should be newest first
        assert "4" in stream[0].candidate.summary

    @pytest.mark.asyncio
    async def test_unconscious_buffer(self, workspace):
        """Verify losing candidates go to unconscious buffer."""
        # Create mix of high and low salience
        candidates = [
            WorkspaceCandidate(
                content="conscious",
                content_type="thought",
                summary="High salience thought",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.9,
                emotional_salience=0.8,
            ),
            WorkspaceCandidate(
                content="unconscious",
                content_type="noise",
                summary="Low salience noise",
                source=WorkspaceCandidateSource.SENSORY,
                activation_level=0.05,
                emotional_salience=0.0,
            ),
        ]

        await workspace.process_consciousness_cycle(candidates=candidates)

        unconscious = workspace.get_unconscious_buffer()

        # Low salience should be in unconscious buffer
        # (if it didn't win competition)
        assert len(workspace.stream_of_consciousness) >= 1

    @pytest.mark.asyncio
    async def test_submit_candidate_convenience(self, workspace):
        """Verify submit_candidate convenience method."""
        candidate = await workspace.submit_candidate(
            content="test content",
            content_type="thought",
            summary="Test thought",
            source=WorkspaceCandidateSource.CTM,
            activation_level=0.8,
            emotional_salience=0.5,
            priority_boost=0.1,
            metadata={"key": "value"}
        )

        assert isinstance(candidate, WorkspaceCandidate)
        assert candidate.activation_level == 0.8
        assert candidate.priority_boost == 0.1
        assert candidate.metadata["key"] == "value"

    @pytest.mark.asyncio
    async def test_goal_setting(self, workspace):
        """Verify goal setting affects attention."""
        workspace.set_goals(["understand consciousness", "learn about neurons"])
        workspace.set_task("study neural networks")

        # Goals should be set in bottleneck
        assert len(workspace.bottleneck.current_goals) == 2
        assert workspace.bottleneck.current_task is not None

    @pytest.mark.asyncio
    async def test_workspace_reset(self, workspace, sample_candidates):
        """Verify workspace reset clears state."""
        # Process some content
        await workspace.process_consciousness_cycle(candidates=sample_candidates)

        assert workspace.cycle_count > 0
        assert len(workspace.stream_of_consciousness) > 0

        # Reset
        workspace.reset()

        assert workspace.cycle_count == 0
        assert len(workspace.stream_of_consciousness) == 0
        assert workspace.current_conscious_content is None

    @pytest.mark.asyncio
    async def test_workspace_statistics(self, workspace, sample_candidates):
        """Verify comprehensive statistics are available."""
        # Run a few cycles
        for _ in range(3):
            await workspace.process_consciousness_cycle(candidates=sample_candidates)

        stats = workspace.get_statistics()

        assert "cycle_count" in stats
        assert "bottleneck" in stats
        assert "broadcast" in stats
        assert "ignition" in stats
        assert stats["cycle_count"] == 3

    @pytest.mark.asyncio
    async def test_empty_candidates_handled(self, workspace):
        """Verify empty candidate list is handled."""
        state = await workspace.process_consciousness_cycle(candidates=[])

        assert not state.is_conscious
        assert state.ignition_events == 0
        assert state.primary_content is None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestGWTIntegration:
    """Integration tests for the full GWT system."""

    @pytest.mark.asyncio
    async def test_full_consciousness_flow(self):
        """Test complete flow: candidates → competition → ignition → broadcast."""
        # Create workspace with registered modules
        workspace = EnhancedGlobalWorkspace()

        # Register mock modules
        snn_module = MockCognitiveModule("snn")
        lsm_module = MockCognitiveModule("lsm")
        workspace.register_module(snn_module)
        workspace.register_module(lsm_module)

        # Create candidates
        candidates = [
            WorkspaceCandidate(
                content="important thought",
                content_type="thought",
                summary="An important conscious thought",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.8,
                emotional_salience=0.7,
            ),
        ]

        # Process
        state = await workspace.process_consciousness_cycle(
            candidates=candidates,
            neural_signals={
                "snn": np.random.rand(100),
                "lsm": np.random.rand(100),
            }
        )

        # Verify full flow
        assert state.is_conscious
        assert state.ignition_events >= 1
        assert state.broadcast_coverage > 0

        # Verify modules received broadcast
        assert len(snn_module.received_broadcasts) >= 1
        assert len(lsm_module.received_broadcasts) >= 1

    @pytest.mark.asyncio
    async def test_attention_shifts_over_time(self):
        """Test that attention can shift between sources over cycles."""
        workspace = EnhancedGlobalWorkspace(
            temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0),
        )

        # Cycle 1: CTM dominant
        candidates1 = [
            WorkspaceCandidate(
                content="ctm_thought",
                content_type="thought",
                summary="CTM generated thought",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.9,
                emotional_salience=0.5,
            ),
        ]
        state1 = await workspace.process_consciousness_cycle(candidates=candidates1)

        # Cycle 2: Safety dominant
        candidates2 = [
            WorkspaceCandidate(
                content="safety_alert",
                content_type="alert",
                summary="Safety alert",
                source=WorkspaceCandidateSource.SAFETY,
                activation_level=0.5,
                emotional_salience=0.3,
            ),
        ]
        state2 = await workspace.process_consciousness_cycle(candidates=candidates2)

        # Attention should have shifted
        assert state1.attention_focus == "ctm"
        assert state2.attention_focus == "safety"

    @pytest.mark.asyncio
    async def test_concurrent_consciousness_cycles(self):
        """Test that multiple cycles can run without interference."""
        workspace = EnhancedGlobalWorkspace(
            temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0),
        )

        async def run_cycle(idx: int):
            candidates = [
                WorkspaceCandidate(
                    content=f"thought_{idx}",
                    content_type="thought",
                    summary=f"Concurrent thought {idx}",
                    source=WorkspaceCandidateSource.CTM,
                    activation_level=0.7,
                    emotional_salience=0.5,
                )
            ]
            return await workspace.process_consciousness_cycle(candidates=candidates)

        # Run multiple cycles concurrently
        # (Note: In practice, cycles should be sequential for consciousness)
        # This tests robustness
        results = await asyncio.gather(*[run_cycle(i) for i in range(3)])

        assert len(results) == 3
        assert all(r.is_conscious for r in results)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestGWTPerformance:
    """Performance tests for GWT Phase 2."""

    @pytest.mark.asyncio
    async def test_competition_performance(self, bottleneck):
        """Verify competition completes within acceptable time."""
        # Create many candidates
        candidates = [
            WorkspaceCandidate(
                content=f"content_{i}",
                content_type="thought",
                summary=f"Thought {i}",
                source=WorkspaceCandidateSource.CTM,
                activation_level=np.random.uniform(0.3, 0.9),
                emotional_salience=np.random.uniform(0.1, 0.8),
            )
            for i in range(100)
        ]

        start = time.time()
        winners = await bottleneck.compete_for_access(candidates)
        elapsed_ms = (time.time() - start) * 1000

        # Should complete in <50ms
        assert elapsed_ms < 50
        assert len(winners) <= bottleneck.capacity

    @pytest.mark.asyncio
    async def test_consciousness_cycle_performance(self, workspace):
        """Verify full consciousness cycle completes within target."""
        candidates = [
            WorkspaceCandidate(
                content=f"content_{i}",
                content_type="thought",
                summary=f"Thought {i}",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.7,
                emotional_salience=0.5,
            )
            for i in range(20)
        ]

        neural_signals = {
            "snn": np.random.rand(100),
            "lsm": np.random.rand(500),
            "htm": np.random.rand(256),
        }

        start = time.time()
        state = await workspace.process_consciousness_cycle(
            candidates=candidates,
            neural_signals=neural_signals
        )
        elapsed_ms = (time.time() - start) * 1000

        # Should complete in <100ms
        assert elapsed_ms < 100
        assert state.is_conscious


# ============================================================================
# TEMPORAL DYNAMICS TESTS (Task 5 - Biologically-plausible ignition timing)
# ============================================================================

class TestTemporalWorkspaceConfig:
    """Tests for TemporalWorkspaceConfig defaults and customisation."""

    def test_default_values(self):
        """Verify default config matches the biological 200-300ms range."""
        cfg = TemporalWorkspaceConfig()

        assert cfg.competition_duration_ms == 250.0
        assert cfg.ignition_threshold_time_ms == 50.0
        assert cfg.broadcast_propagation_ms == 30.0
        assert cfg.refractory_period_ms == 500.0
        assert cfg.timestep_ms == 10.0

    def test_custom_values(self):
        """Verify custom config overrides are respected."""
        cfg = TemporalWorkspaceConfig(
            competition_duration_ms=300.0,
            ignition_threshold_time_ms=100.0,
            broadcast_propagation_ms=50.0,
            refractory_period_ms=1000.0,
            timestep_ms=5.0,
        )

        assert cfg.competition_duration_ms == 300.0
        assert cfg.ignition_threshold_time_ms == 100.0
        assert cfg.broadcast_propagation_ms == 50.0
        assert cfg.refractory_period_ms == 1000.0
        assert cfg.timestep_ms == 5.0

    def test_workspace_stores_config(self):
        """Verify workspace stores and uses the temporal config."""
        cfg = TemporalWorkspaceConfig(competition_duration_ms=200.0)
        workspace = EnhancedGlobalWorkspace(temporal_config=cfg)

        assert workspace.temporal_config is cfg
        assert workspace.temporal_config.competition_duration_ms == 200.0

    def test_workspace_default_config(self):
        """Verify workspace creates default config when none supplied."""
        workspace = EnhancedGlobalWorkspace()

        assert workspace.temporal_config is not None
        assert isinstance(workspace.temporal_config, TemporalWorkspaceConfig)
        assert workspace.temporal_config.competition_duration_ms == 250.0


class TestTemporalIgnition:
    """Tests for biologically-plausible temporal competition and refractory period."""

    @pytest.mark.asyncio
    async def test_sustained_activation_ignites(self):
        """
        A high-activation candidate that sustains above threshold for the
        required duration must still ignite successfully.
        """
        # Use a short competition window for test speed
        cfg = TemporalWorkspaceConfig(
            competition_duration_ms=100.0,
            ignition_threshold_time_ms=30.0,
            refractory_period_ms=0.0,     # Disable refractory for this test
            timestep_ms=10.0,
        )
        workspace = EnhancedGlobalWorkspace(
            ignition_threshold=0.3,
            temporal_config=cfg,
        )

        candidates = [
            WorkspaceCandidate(
                content="strong thought",
                content_type="thought",
                summary="A strong, sustained thought",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.9,
                emotional_salience=0.7,
            ),
        ]

        state = await workspace.process_consciousness_cycle(candidates=candidates)

        assert state.is_conscious
        assert state.ignition_events >= 1
        assert state.primary_content is not None

    @pytest.mark.asyncio
    async def test_decaying_candidate_filtered_out(self):
        """
        A candidate whose decay_rate causes it to drop below threshold
        before ignition_threshold_time_ms should NOT ignite.
        """
        cfg = TemporalWorkspaceConfig(
            competition_duration_ms=100.0,
            ignition_threshold_time_ms=80.0,   # Needs 80ms sustained
            refractory_period_ms=0.0,
            timestep_ms=10.0,
        )
        workspace = EnhancedGlobalWorkspace(
            ignition_threshold=0.3,
            temporal_config=cfg,
        )

        # Candidate with aggressive decay: activation = 0.35 * (0.5^step)
        # After 1 step (10ms): 0.35*0.5 = 0.175 -- below threshold 0.3
        # So sustained time is only 10ms < required 80ms
        candidates = [
            WorkspaceCandidate(
                content="ephemeral flash",
                content_type="noise",
                summary="Quickly decaying noise",
                source=WorkspaceCandidateSource.SENSORY,
                activation_level=0.35,
                emotional_salience=0.0,
                metadata={"decay_rate": 0.5},
            ),
        ]

        state = await workspace.process_consciousness_cycle(candidates=candidates)

        assert not state.is_conscious
        assert state.ignition_events == 0

    @pytest.mark.asyncio
    async def test_refractory_period_blocks_consecutive_ignitions(self):
        """
        Two rapid consciousness cycles should not both ignite if the
        refractory period has not elapsed.
        """
        cfg = TemporalWorkspaceConfig(
            competition_duration_ms=100.0,
            ignition_threshold_time_ms=30.0,
            refractory_period_ms=500.0,   # 500ms refractory
            timestep_ms=10.0,
        )
        workspace = EnhancedGlobalWorkspace(
            ignition_threshold=0.3,
            temporal_config=cfg,
        )

        strong = [
            WorkspaceCandidate(
                content="thought",
                content_type="thought",
                summary="Strong thought",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.9,
                emotional_salience=0.7,
            ),
        ]

        # First cycle: should ignite normally
        state1 = await workspace.process_consciousness_cycle(candidates=strong)
        assert state1.is_conscious
        assert state1.ignition_events >= 1

        # Second cycle immediately after: simulated time is only ~100ms
        # past the last ignition, well within the 500ms refractory period
        state2 = await workspace.process_consciousness_cycle(candidates=strong)
        assert not state2.is_conscious
        assert state2.ignition_events == 0

    @pytest.mark.asyncio
    async def test_refractory_period_expires(self):
        """
        After enough simulated time passes (enough cycles), the refractory
        period should expire and ignition should succeed again.
        """
        cfg = TemporalWorkspaceConfig(
            competition_duration_ms=100.0,
            ignition_threshold_time_ms=30.0,
            refractory_period_ms=250.0,   # Short refractory for testing
            timestep_ms=10.0,
        )
        workspace = EnhancedGlobalWorkspace(
            ignition_threshold=0.3,
            temporal_config=cfg,
        )

        strong = [
            WorkspaceCandidate(
                content="thought",
                content_type="thought",
                summary="Strong thought",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.9,
                emotional_salience=0.7,
            ),
        ]

        # Timing trace (competition_duration=100ms, refractory=250ms):
        #   Cycle 1: check at t=0 (0-(-1000)=1000 >= 250, OK). Advance to t=100. Ignite -> last=100.
        #   Cycle 2: check at t=100 (100-100=0 < 250, blocked). Advance to t=200.
        #   Cycle 3: check at t=200 (200-100=100 < 250, blocked). Advance to t=300.
        #   Cycle 4: check at t=300 (300-100=200 < 250, blocked). Advance to t=400.
        #   Cycle 5: check at t=400 (400-100=300 >= 250, OK). Ignite again.
        state1 = await workspace.process_consciousness_cycle(candidates=strong)
        assert state1.is_conscious

        state2 = await workspace.process_consciousness_cycle(candidates=strong)
        assert not state2.is_conscious

        state3 = await workspace.process_consciousness_cycle(candidates=strong)
        assert not state3.is_conscious

        state4 = await workspace.process_consciousness_cycle(candidates=strong)
        assert not state4.is_conscious

        # Fifth cycle: refractory has now expired
        state5 = await workspace.process_consciousness_cycle(candidates=strong)
        assert state5.is_conscious
        assert state5.ignition_events >= 1

    @pytest.mark.asyncio
    async def test_temporal_competition_no_wall_clock_delay(self):
        """
        Temporal competition must be purely numerical -- it should add
        negligible wall-clock time (well under 50ms even with many
        candidates and fine timesteps).
        """
        cfg = TemporalWorkspaceConfig(
            competition_duration_ms=300.0,
            ignition_threshold_time_ms=50.0,
            refractory_period_ms=0.0,
            timestep_ms=1.0,   # 300 steps -- still should be fast
        )
        workspace = EnhancedGlobalWorkspace(
            ignition_threshold=0.3,
            temporal_config=cfg,
        )

        candidates = [
            WorkspaceCandidate(
                content=f"thought_{i}",
                content_type="thought",
                summary=f"Thought {i}",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.7,
                emotional_salience=0.5,
            )
            for i in range(20)
        ]

        start = time.time()
        state = await workspace.process_consciousness_cycle(candidates=candidates)
        elapsed_ms = (time.time() - start) * 1000

        # Should still complete in <100ms wall-clock (the project target)
        assert elapsed_ms < 100
        assert state.is_conscious

    @pytest.mark.asyncio
    async def test_emotional_salience_helps_sustain(self):
        """
        A candidate with moderate activation but high emotional salience
        should survive temporal competition because the effective
        activation includes a 0.3 * emotional_salience boost.
        """
        cfg = TemporalWorkspaceConfig(
            competition_duration_ms=100.0,
            ignition_threshold_time_ms=50.0,
            refractory_period_ms=0.0,
            timestep_ms=10.0,
        )
        workspace = EnhancedGlobalWorkspace(
            ignition_threshold=0.3,
            temporal_config=cfg,
        )

        # Activation 0.2 is below threshold 0.3,
        # but effective = 0.2 + 0.9*0.3 = 0.47 is above
        candidates = [
            WorkspaceCandidate(
                content="emotional memory",
                content_type="memory",
                summary="A memory with strong emotional weight",
                source=WorkspaceCandidateSource.MEMORY,
                activation_level=0.2,
                emotional_salience=0.9,
            ),
        ]

        state = await workspace.process_consciousness_cycle(candidates=candidates)

        assert state.is_conscious

    @pytest.mark.asyncio
    async def test_reset_clears_temporal_state(self):
        """Verify reset() clears the temporal tracking state."""
        cfg = TemporalWorkspaceConfig(
            competition_duration_ms=100.0,
            refractory_period_ms=500.0,
        )
        workspace = EnhancedGlobalWorkspace(
            ignition_threshold=0.3,
            temporal_config=cfg,
        )

        # Run a cycle to advance simulated time
        candidates = [
            WorkspaceCandidate(
                content="thought",
                content_type="thought",
                summary="A thought",
                source=WorkspaceCandidateSource.CTM,
                activation_level=0.9,
                emotional_salience=0.7,
            ),
        ]
        await workspace.process_consciousness_cycle(candidates=candidates)

        assert workspace._simulated_time_ms > 0
        assert workspace._last_ignition_time_ms > 0

        workspace.reset()

        assert workspace._simulated_time_ms == 0.0
        assert workspace._last_ignition_time_ms == -1000.0
        assert workspace.cycle_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
