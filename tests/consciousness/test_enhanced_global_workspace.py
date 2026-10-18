"""
Tests for the Enhanced Global Workspace (GWT)
=============================================

1. AttentionBottleneck - Competition mechanism
2. GlobalBroadcast - Parallel distribution and coverage
3. IgnitionDetector - Non-linear emergence
4. EnhancedGlobalWorkspace - Full integration
"""

import pytest
import asyncio
import dataclasses
import logging
import numpy as np
import time
import warnings
from typing import Dict, Any

from mtc.consciousness.active_inference import ActiveInferenceConfig
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
)

# A small active inference geometry: the cycle tests below do not measure it
FAST_FEP = dict(num_hidden_states=4, num_actions=2, planning_horizon=2)


def _fast_workspace(**kwargs) -> EnhancedGlobalWorkspace:
    """A workspace that runs cycles quickly and samples no machine metrics."""
    kwargs.setdefault("active_inference_config", ActiveInferenceConfig(**FAST_FEP))
    kwargs.setdefault("body_metrics_enabled", False)
    return EnhancedGlobalWorkspace(**kwargs)


def _strong(summary: str = "a strong thought", **overrides) -> WorkspaceCandidate:
    """A candidate that wins the bottleneck and ignites on its first occurrence."""
    fields = dict(
        content=summary,
        content_type="thought",
        summary=summary,
        source=WorkspaceCandidateSource.CTM,
        source_module="ctm",
        activation_level=0.9,
        emotional_salience=0.7,
    )
    fields.update(overrides)
    return WorkspaceCandidate(**fields)


class _FakeClock:
    """A clock the test advances by hand."""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def bottleneck():
    """Create a standard attention bottleneck."""
    return AttentionBottleneck(
        capacity=7, ignition_threshold=0.3, lateral_inhibition=0.2
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
        decay_rate=0.1,
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

    async def receive_broadcast(self, content: WorkspaceContent) -> Dict[str, Any]:
        if self.should_fail:
            raise Exception(f"Mock failure in {self.name}")

        self.received_broadcasts.append(content)
        return {
            "module": self.name,
            "received": True,
            "content_id": content.id,
            "timestamp": time.time(),
        }


# ============================================================================
# ATTENTION BOTTLENECK TESTS
# ============================================================================


class TestAttentionBottleneck:
    """Tests for the AttentionBottleneck competition mechanism."""

    @pytest.mark.asyncio
    async def test_competition_respects_capacity(self, bottleneck):
        """Verify bottleneck respects capacity limit (7 plus or minus 2)."""
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

    @staticmethod
    def _weak_candidate() -> WorkspaceCandidate:
        """Very low activation, no emotional weight, the same content each time."""
        return WorkspaceCandidate(
            content="same_low_content",
            content_type="noise",
            summary="same_low_activation_noise",
            source=WorkspaceCandidateSource.SENSORY,
            activation_level=0.05,
            emotional_salience=0.0,
        )

    @pytest.mark.asyncio
    async def test_competition_respects_threshold(self, bottleneck):
        """Already-seen weak content scores under the threshold and no copy
        of it wins: 0.4 * (0.6 * 0.05 + 0.4 * 0.2) + 0.15 + 0.1 is 0.294."""
        # One copy first, so the content's digest is in the novelty window
        await bottleneck.compete_for_access([self._weak_candidate()])

        candidates = [self._weak_candidate() for _ in range(5)]
        winners = await bottleneck.compete_for_access(candidates)

        assert winners == []
        assert [c.novelty_score for c in candidates] == [0.2] * 5

    @pytest.mark.asyncio
    async def test_first_occurrence_of_weak_content_wins_on_novelty_and_neutral_relevance(
        self, bottleneck
    ):
        """The bare bottleneck admits a first occurrence of weak content:
        novelty 0.9 and the neutral relevance of 0.5 (which adds 0.15 to
        every salience when the workspace has no goals or task; see the
        relevance sentence of docs/HONEST_LIMITATIONS.md) put it at about
        0.406 against the 0.3 threshold. A recalibration of the thresholds
        flips this test deliberately."""
        winners = await bottleneck.compete_for_access([self._weak_candidate()])

        assert len(winners) == 1
        assert winners[0].salience == pytest.approx(0.406, abs=1e-3)

    @pytest.mark.asyncio
    async def test_unset_goals_do_not_overwrite_producer_relevance(self, bottleneck):
        """With no goals or task set, the competition uses its neutral
        constants but leaves the producer's relevance on the candidate."""
        candidate = WorkspaceCandidate(
            content="a question",
            content_type="conversation",
            summary="a question from the operator",
            source=WorkspaceCandidateSource.LLM,
            activation_level=0.9,
            emotional_salience=0.2,
            goal_relevance=0.9,
            task_relevance=0.2,
        )
        winners = await bottleneck.compete_for_access([candidate])

        assert candidate.goal_relevance == 0.9
        assert candidate.task_relevance == 0.2
        # The salience still reads the neutral 0.5 for both terms
        expected = 0.4 * (0.6 * 0.9 + 0.4 * 0.9) + 0.3 * 0.5 + 0.2 * 0.2 + 0.1
        assert winners[0].salience == pytest.approx(expected, abs=1e-3)

    @pytest.mark.asyncio
    async def test_set_goals_record_measured_relevance(self, bottleneck):
        """With goals set, the keyword match is measured and recorded."""
        bottleneck.set_goals(["alpha beta"])
        matching = WorkspaceCandidate(
            content="m",
            content_type="thought",
            summary="a note about alpha",
            source=WorkspaceCandidateSource.CTM,
            activation_level=0.8,
        )
        unrelated = WorkspaceCandidate(
            content="u",
            content_type="thought",
            summary="a note about gamma",
            source=WorkspaceCandidateSource.CTM,
            activation_level=0.8,
        )
        await bottleneck.compete_for_access([matching, unrelated])

        assert matching.goal_relevance == pytest.approx(0.5)
        assert unrelated.goal_relevance == 0.0

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
    async def test_no_modules_counts_no_broadcast(self, broadcaster, caplog):
        """With no registered module a broadcast delivers nothing, counts
        nothing and says so once at DEBUG, not once per ignition at WARNING."""
        candidate = WorkspaceCandidate(
            content="orphan",
            content_type="test",
            summary="Orphan content",
            source=WorkspaceCandidateSource.CTM,
        )
        logger_name = "mtc.consciousness.enhanced_global_workspace"
        with caplog.at_level(logging.DEBUG, logger=logger_name):
            for _ in range(3):
                content = WorkspaceContent(candidate=candidate, salience=0.8)
                results = await broadcaster.broadcast(content)
                assert results == {}

        assert broadcaster.total_broadcasts == 0
        assert broadcaster.coverage_ratio is None
        records = [
            r
            for r in caplog.records
            if r.name == logger_name and "registered" in r.getMessage()
        ]
        assert [r.levelno for r in records] == [logging.DEBUG]

    def test_coverage_ratio_unmeasured_without_receivers(self, broadcaster):
        """Nothing could have received, so the coverage is undefined (None);
        with a receiver and no broadcast yet it is a measured 0.0."""
        assert broadcaster.coverage_ratio is None
        assert broadcaster.get_statistics()["coverage_ratio"] is None
        assert broadcaster.get_statistics()["receivers"] == 0

        broadcaster.register_module(MockCognitiveModule("listener"))
        assert broadcaster.coverage_ratio == 0.0
        assert broadcaster.get_statistics()["receivers"] == 1

    def test_legacy_coverage_ratio_branch(self):
        """legacy_coverage_ratio keeps the v0.4.0 reading of 1.0."""
        legacy = GlobalBroadcast(legacy_coverage_ratio=True)
        assert legacy.coverage_ratio == 1.0

    @pytest.mark.asyncio
    async def test_coverage_ratio_counts_failures(self, broadcaster):
        """With broadcasts, coverage is the success share in both modes."""
        broadcaster.register_module(MockCognitiveModule("working"))
        broadcaster.register_module(MockCognitiveModule("failing", should_fail=True))
        candidate = WorkspaceCandidate(content="x", summary="x")
        await broadcaster.broadcast(WorkspaceContent(candidate=candidate))
        assert broadcaster.coverage_ratio == pytest.approx(0.5)


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
            candidate=candidate, salience=0.5  # Above threshold (0.3)
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
            candidate=candidate, salience=0.1  # Below threshold (0.3)
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
            candidate=candidate, salience=0.6  # High enough for immediate ignition
        )

        event = await ignition_detector.check_ignition(content)

        assert event is not None
        expected_amplified = min(1.5, 0.6 * ignition_detector.amplification)
        assert event.post_ignition_salience == pytest.approx(
            expected_amplified, rel=0.1
        )

    @pytest.mark.asyncio
    async def test_sustain_check(self, ignition_detector):
        """Verify sustained content stays in consciousness."""
        candidate = WorkspaceCandidate(
            content="sustained",
            content_type="thought",
            summary="Sustained thought",
            source=WorkspaceCandidateSource.CTM,
        )
        content = WorkspaceContent(candidate=candidate, salience=0.8)

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
            content = WorkspaceContent(candidate=candidate, salience=0.6)
            await ignition_detector.check_ignition(content)

        stats = ignition_detector.get_statistics()

        assert stats["total_ignitions"] == 5
        assert stats["avg_amplification"] > 1.0

    @pytest.mark.asyncio
    async def test_sustain_rate_is_none_until_a_sustain_check_runs(
        self, ignition_detector
    ):
        """No sustain check has run, so no rate is reported; a check that
        finds the content sustained gives 1.0 and one that finds it decayed
        gives 0.0, over the checked events only."""
        content = WorkspaceContent(
            candidate=WorkspaceCandidate(content="s", summary="sustained"),
            salience=0.8,
        )
        event = await ignition_detector.check_ignition(content)
        assert event is not None
        assert event.sustain_checked is False

        stats = ignition_detector.get_statistics()
        assert stats["total_ignitions"] == 1
        assert stats["sustain_rate"] is None
        assert stats["sustain_checked_count"] == 0
        assert stats["sustained_count"] == 0

        assert await ignition_detector.check_sustain(content) is True
        stats = ignition_detector.get_statistics()
        assert event.sustain_checked is True
        assert stats["sustain_checked_count"] == 1
        assert stats["sustained_count"] == 1
        assert stats["sustain_rate"] == 1.0

        event.ignition_time -= 60.0
        assert await ignition_detector.check_sustain(content) is False
        stats = ignition_detector.get_statistics()
        assert stats["sustain_checked_count"] == 1
        assert stats["sustained_count"] == 0
        assert stats["sustain_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_legacy_sustain_rate_keeps_the_v04_constant(self):
        """legacy_sustain_rate reports 1.0 straight after an ignition,
        although nothing checked it (v0.4.0), with the v0.4.0 keys alone."""
        detector = IgnitionDetector(legacy_sustain_rate=True)
        content = WorkspaceContent(
            candidate=WorkspaceCandidate(content="s", summary="sustained"),
            salience=0.8,
        )
        assert await detector.check_ignition(content) is not None

        stats = detector.get_statistics()
        assert stats["sustain_rate"] == 1.0
        assert stats["sustained_count"] == 1
        assert "sustain_checked_count" not in stats

        # The workspace forwards the switch to its detector
        legacy = _fast_workspace(legacy_sustain_rate=True)
        assert legacy.ignition_detector.legacy_sustain_rate is True
        assert _fast_workspace().ignition_detector.legacy_sustain_rate is False


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
    async def test_consciousness_cycle_with_neural_signals(
        self, workspace, sample_candidates
    ):
        """Verify consciousness cycle integrates neural signals."""
        neural_signals = {
            "snn": np.random.rand(100),
            "lsm": np.random.rand(500),
            "htm": np.random.rand(256),
        }

        state = await workspace.process_consciousness_cycle(
            candidates=sample_candidates, neural_signals=neural_signals
        )

        assert state.integration_level > 0  # Should measure integration

    @pytest.mark.asyncio
    async def test_consciousness_cycle_with_emotions(
        self, workspace, sample_candidates
    ):
        """Verify emotional state affects processing."""
        emotional_state = {"joy": 0.9, "excitement": 0.8, "curiosity": 0.7}

        state = await workspace.process_consciousness_cycle(
            candidates=sample_candidates, emotional_state=emotional_state
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

        state = await workspace.process_consciousness_cycle(candidates=candidates)

        # The noise lost (the gate filtered it) and sits in the buffer; the
        # thought alone was admitted and ignited
        assert [c.summary for c in workspace.get_unconscious_buffer()] == [
            "Low salience noise"
        ]
        assert [w.candidate.summary for w in state.workspace_contents] == [
            "High salience thought"
        ]
        assert state.is_conscious
        assert [c.candidate.summary for c in workspace.stream_of_consciousness] == [
            "High salience thought"
        ]

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
            metadata={"key": "value"},
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

    @pytest.mark.asyncio
    async def test_cycle_without_receivers_reports_unmeasured_coverage(self, workspace):
        """A default workspace has no receiver, so the coverage is None; with
        a receiver registered and nothing broadcast it is a measured 0.0."""
        state = await workspace.process_consciousness_cycle(candidates=[])
        assert workspace.broadcaster.total_broadcasts == 0
        assert state.broadcast_coverage is None

        workspace.register_module(MockCognitiveModule("listener"))
        state = await workspace.process_consciousness_cycle(candidates=[])
        assert state.broadcast_coverage == 0.0

    @pytest.mark.asyncio
    async def test_default_workspace_registers_no_stub_receivers(self):
        """The workspace hands its winners to its own modules by direct
        call, which cannot fail and is not counted as coverage: nothing is
        registered and the coverage stays unmeasured through ignitions."""
        workspace = _fast_workspace(
            temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0)
        )
        assert workspace.broadcaster.registered_modules == {}

        ignitions = 0
        for i in range(3):
            state = await workspace.process_consciousness_cycle([_strong(f"t {i}")])
            ignitions += state.ignition_events
            assert state.broadcast_coverage is None
        assert ignitions == 3

        stats = workspace.get_statistics()["broadcast"]
        assert stats["coverage_ratio"] is None
        assert stats["receivers"] == 0
        assert stats["total_broadcasts"] == 0

    @pytest.mark.asyncio
    async def test_failing_receiver_lowers_coverage(self):
        """Coverage is the delivery share over the receivers a caller
        registered: one of two failing gives 0.5."""
        workspace = _fast_workspace(
            temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0)
        )
        workspace.register_module(MockCognitiveModule("good"))
        workspace.register_module(MockCognitiveModule("bad", should_fail=True))

        state = await workspace.process_consciousness_cycle([_strong()])

        assert state.ignition_events == 1
        assert state.broadcast_coverage == pytest.approx(0.5)
        stats = workspace.get_statistics()["broadcast"]
        assert stats["receivers"] == 2
        assert stats["coverage_ratio"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_legacy_broadcast_receivers_keep_v04_coverage(self):
        """legacy_broadcast_receivers registers the three acknowledgement-only
        receivers, which makes the coverage 1.0 by construction (v0.4.0)."""
        workspace = _fast_workspace(
            legacy_broadcast_receivers=True,
            temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0),
        )
        assert sorted(workspace.broadcaster.registered_modules) == [
            "active_inference",
            "attention_schema",
            "metacognition",
        ]

        state = await workspace.process_consciousness_cycle([_strong()])

        assert state.ignition_events == 1
        assert state.broadcast_coverage == 1.0
        stats = workspace.get_statistics()["broadcast"]
        assert stats["receivers"] == 3
        assert stats["total_broadcasts"] == 1

    @pytest.mark.asyncio
    async def test_cycle_without_broadcast_legacy_coverage(self):
        """legacy_coverage_ratio=True keeps the v0.4.0 reading of 1.0."""
        workspace = EnhancedGlobalWorkspace(legacy_coverage_ratio=True)
        state = await workspace.process_consciousness_cycle(candidates=[])
        assert workspace.broadcaster.total_broadcasts == 0
        assert state.broadcast_coverage == 1.0

    def test_default_ignition_threshold_is_point_three(self):
        workspace = EnhancedGlobalWorkspace()
        assert workspace.bottleneck.threshold == 0.3
        assert workspace.ignition_detector.threshold == 0.3

    @pytest.mark.asyncio
    async def test_weak_candidate_does_not_enter_the_workspace(self):
        """At workspace level the first occurrence of weak content is
        refused: the temporal competition needs activation above the
        threshold, whatever novelty and the neutral relevance gave it."""
        workspace = _fast_workspace(
            competition_mode="flat",
            thalamic_gate_enabled=False,
            temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0),
        )
        weak = WorkspaceCandidate(
            content="same_low_content",
            content_type="noise",
            summary="same_low_activation_noise",
            source=WorkspaceCandidateSource.SENSORY,
            activation_level=0.05,
            emotional_salience=0.0,
        )
        state = await workspace.process_consciousness_cycle([weak])

        assert state.workspace_contents == []
        assert state.ignition_events == 0
        assert not state.is_conscious

    @pytest.mark.asyncio
    async def test_authored_template_winner_gets_no_hot(self):
        """A winner marked as an authored template competes, ignites and
        appears in the workspace, but is registered as no first-order state
        and targeted by no higher-order thought; the same candidate without
        the mark gets one."""
        flagged = _strong("an authored line", metadata={"authored_template": True})
        workspace = _fast_workspace(
            temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0)
        )
        state = await workspace.process_consciousness_cycle([flagged])

        assert state.is_conscious
        assert state.ignition_events == 1
        assert [w.candidate for w in state.workspace_contents] == [flagged]
        assert state.higher_order_thoughts == []
        assert len(workspace.metacognition.first_order_states) == 0

        plain = _strong("an authored line")
        other = _fast_workspace(
            temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0)
        )
        state = await other.process_consciousness_cycle([plain])

        assert state.ignition_events == 1
        assert len(state.higher_order_thoughts) == 1
        assert len(other.metacognition.first_order_states) == 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestGWTIntegration:
    """Integration tests for the full GWT system."""

    @pytest.mark.asyncio
    async def test_full_consciousness_flow(self):
        """Test complete flow: candidates, competition, ignition, broadcast."""
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
            },
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
    """Performance tests for the workspace."""

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

    @pytest.mark.slow
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
            candidates=candidates, neural_signals=neural_signals
        )
        elapsed_ms = (time.time() - start) * 1000

        # wall-clock bound; machine and load dependent, so this is a smoke bound not a benchmark
        assert elapsed_ms < 1000
        assert state.is_conscious


# ============================================================================
# TEMPORAL DYNAMICS TESTS (biologically plausible ignition timing)
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
            refractory_period_ms=0.0,  # Disable refractory for this test
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
            ignition_threshold_time_ms=80.0,  # Needs 80ms sustained
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
            refractory_period_ms=500.0,  # 500ms refractory
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
            refractory_period_ms=250.0,  # Short refractory for testing
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

    @pytest.mark.slow
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
            timestep_ms=1.0,  # 300 steps -- still should be fast
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

        # wall-clock bound; machine and load dependent, so this is a smoke bound not a benchmark
        assert elapsed_ms < 1000
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


# ============================================================================
# UNMEASURED STATE FIELDS (binding, protoself, feeling of knowing, identity)
# ============================================================================


class TestUnmeasuredStateFields:
    """The state carries None where a module measured nothing, and the
    workspace's own readers (the cycle log line) accept it."""

    @pytest.mark.asyncio
    async def test_cycle_log_line_formats_unmeasured_fields(self, monkeypatch, caplog):
        """A damasio result and a loop moment that measured nothing leave the
        state fields None, and the cycle completes with 'unmeasured' in its
        log line instead of a formatting error."""
        workspace = _fast_workspace(
            temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0)
        )
        damasio_process = workspace.damasio.process
        loop_process = workspace.beautiful_loop.process_conscious_moment

        async def unmeasured_damasio(*args, **kwargs):
            result = await damasio_process(*args, **kwargs)
            result["protoself_stability"] = None
            result["feeling_of_knowing"] = None
            result["autobiographic_continuity"] = None
            return result

        async def unmeasured_loop(*args, **kwargs):
            # A copy: the loop's own history keeps the moment it recorded
            moment = await loop_process(*args, **kwargs)
            return dataclasses.replace(moment, binding_quality=None)

        monkeypatch.setattr(workspace.damasio, "process", unmeasured_damasio)
        monkeypatch.setattr(
            workspace.beautiful_loop, "process_conscious_moment", unmeasured_loop
        )

        with caplog.at_level(
            logging.INFO, logger="mtc.consciousness.enhanced_global_workspace"
        ):
            state = await workspace.process_consciousness_cycle([_strong()])

        assert state.is_conscious
        assert state.binding_quality is None
        assert state.protoself_stability is None
        assert state.feeling_of_knowing is None
        assert state.autobiographic_continuity is None
        cycle_lines = [
            r.getMessage() for r in caplog.records if "complete" in r.getMessage()
        ]
        assert len(cycle_lines) == 1
        assert "bind=unmeasured" in cycle_lines[0]
        assert "fok=unmeasured" in cycle_lines[0]

    @pytest.mark.asyncio
    async def test_default_cycle_leaves_binding_unmeasured(self):
        """Salience-only winners carry no vector of their own, so the loop
        binds nothing and the state's binding quality is None."""
        workspace = _fast_workspace(
            temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0)
        )
        state = await workspace.process_consciousness_cycle(
            [_strong("first thought"), _strong("second thought")]
        )
        assert len(state.workspace_contents) == 2
        assert state.binding_quality is None

    @pytest.mark.asyncio
    async def test_first_cycle_leaves_protoself_stability_unmeasured(self):
        """The first protoself update has no previous body state, so the
        first cycle reports no stability and no feeling of knowing."""
        workspace = _fast_workspace()
        first = await workspace.process_consciousness_cycle([_strong()])
        assert first.protoself_stability is None
        assert first.feeling_of_knowing is None
        second = await workspace.process_consciousness_cycle([_strong("another")])
        assert second.protoself_stability is not None


# ============================================================================
# REFRACTORY CLOCK (per_cycle by default, elapsed on an injected clock)
# ============================================================================


async def _cadence(workspace, clock, cycles: int, gap_s: float):
    """Admitted and ignition counts per cycle, the clock advanced between
    cycles by ``gap_s`` seconds."""
    admitted, ignitions = [], []
    for i in range(cycles):
        state = await workspace.process_consciousness_cycle([_strong(f"thought {i}")])
        admitted.append(len(state.workspace_contents))
        ignitions.append(state.ignition_events)
        clock.advance(gap_s)
    return admitted, ignitions


class TestRefractoryClock:
    """The refractory period runs on the simulated clock; ``elapsed`` also
    credits the real time between cycles."""

    @pytest.mark.asyncio
    async def test_default_clock_is_per_cycle(self):
        """The library default is the simulated per-cycle clock: sixty real
        seconds between cycles do not clear the refractory period."""
        assert TemporalWorkspaceConfig().refractory_clock == "per_cycle"
        clock = _FakeClock()
        workspace = _fast_workspace(clock=clock)
        assert workspace.temporal_config.refractory_clock == "per_cycle"
        assert workspace.refractory_clock == "per_cycle"
        admitted, ignitions = await _cadence(workspace, clock, 4, 60.0)
        assert admitted == [1, 0, 0, 1]
        assert ignitions == [1, 0, 0, 1]

    @pytest.mark.asyncio
    async def test_per_cycle_clock_keeps_the_simulated_cadence(self):
        clock = _FakeClock()
        workspace = _fast_workspace(
            temporal_config=TemporalWorkspaceConfig(refractory_clock="per_cycle"),
            clock=clock,
        )
        admitted, ignitions = await _cadence(workspace, clock, 4, 60.0)
        assert admitted == [1, 0, 0, 1]
        assert ignitions == [1, 0, 0, 1]
        # The simulated clock advanced by the competition duration alone
        assert workspace._simulated_time_ms == pytest.approx(4 * 250.0)

    @pytest.mark.asyncio
    async def test_refractory_clears_when_cycles_are_far_apart(self):
        clock = _FakeClock()
        workspace = _fast_workspace(
            temporal_config=TemporalWorkspaceConfig(refractory_clock="elapsed"),
            clock=clock,
        )
        assert workspace.refractory_clock == "elapsed"
        admitted, ignitions = await _cadence(workspace, clock, 3, 60.0)
        assert admitted == [1, 1, 1]
        assert ignitions == [1, 1, 1]
        # Two gaps of sixty seconds joined the simulated clock
        assert workspace._simulated_time_ms == pytest.approx(3 * 250.0 + 2 * 60000.0)

    @pytest.mark.asyncio
    async def test_rapid_cycles_stay_refractory_under_elapsed_clock(self):
        clock = _FakeClock()
        workspace = _fast_workspace(
            temporal_config=TemporalWorkspaceConfig(refractory_clock="elapsed")
        )
        workspace.set_clock(clock)
        assert workspace.clock is clock
        admitted, ignitions = await _cadence(workspace, clock, 4, 0.001)
        assert admitted == [1, 0, 0, 1]
        assert ignitions == [1, 0, 0, 1]

    @pytest.mark.asyncio
    async def test_elapsed_clock_records_and_resets_the_cycle_end(self):
        clock = _FakeClock(now=500.0)
        workspace = _fast_workspace(
            temporal_config=TemporalWorkspaceConfig(refractory_clock="elapsed"),
            clock=clock,
        )
        assert workspace._last_cycle_end_s is None
        await workspace.process_consciousness_cycle([_strong()])
        assert workspace._last_cycle_end_s == 500.0
        workspace.reset()
        assert workspace._last_cycle_end_s is None

    def test_unknown_refractory_clock_rejected(self):
        with pytest.raises(ValueError):
            TemporalWorkspaceConfig(refractory_clock="wall")


# ============================================================================
# DROPPED WINNERS (refractory period, temporal competition)
# ============================================================================


class TestDroppedWinners:
    """A bottleneck winner that the refractory period or the temporal
    competition drops is recorded in the unconscious buffer and counted."""

    @pytest.mark.asyncio
    async def test_refractory_dropped_winner_goes_to_the_unconscious_buffer(self):
        workspace = _fast_workspace(
            temporal_config=TemporalWorkspaceConfig(refractory_clock="per_cycle")
        )
        first = await workspace.process_consciousness_cycle([_strong("first")])
        assert first.ignition_events == 1
        assert workspace.get_unconscious_buffer() == []

        second = await workspace.process_consciousness_cycle([_strong("second")])
        assert second.workspace_contents == []
        assert second.ignition_events == 0
        assert [c.summary for c in workspace.get_unconscious_buffer()] == ["second"]
        stats = workspace.get_statistics()
        assert stats["refractory_dropped"] == 1
        assert stats["temporal_dropped"] == 0

    @pytest.mark.asyncio
    async def test_temporal_competition_loser_is_buffered(self):
        workspace = _fast_workspace(
            competition_mode="flat",
            thalamic_gate_enabled=False,
            temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0),
        )
        weak = WorkspaceCandidate(
            content="faint",
            content_type="noise",
            summary="a faint first occurrence",
            source=WorkspaceCandidateSource.SENSORY,
            activation_level=0.05,
            emotional_salience=0.0,
        )
        state = await workspace.process_consciousness_cycle([weak])
        # It won the bottleneck on novelty and the neutral relevance ...
        assert workspace.bottleneck.last_competition.winners_count == 1
        # ... and the temporal competition dropped it
        assert state.workspace_contents == []
        assert state.ignition_events == 0
        assert [c.summary for c in workspace.get_unconscious_buffer()] == [
            "a faint first occurrence"
        ]
        stats = workspace.get_statistics()
        assert stats["temporal_dropped"] == 1
        assert stats["refractory_dropped"] == 0


# ============================================================================
# PENDING IGNITIONS (bounded) AND THE HOT TRIGGER OF AN ADMITTED WINNER
# ============================================================================


def _repeated_message() -> WorkspaceCandidate:
    """The same moderate message every cycle: above the threshold on its
    first occurrence, between the threshold and 1.5 times it afterwards."""
    return WorkspaceCandidate(
        content="the same message",
        content_type="conversation",
        summary="the same message",
        source=WorkspaceCandidateSource.LLM,
        activation_level=0.35,
        emotional_salience=0.0,
    )


async def _repeated_run(cycles: int = 40, step_s: float = 2.0):
    """``cycles`` cycles of the repeated message on a workspace whose
    ignition detector reads a clock stepped ``step_s`` past each cycle."""
    workspace = _fast_workspace(
        thalamic_gate_enabled=False,
        temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0),
    )
    clock = _FakeClock(now=1_000_000.0)
    workspace.ignition_detector.clock = clock
    states = []
    for _ in range(cycles):
        states.append(
            await workspace.process_consciousness_cycle([_repeated_message()])
        )
        clock.advance(step_s)
    return workspace, states


class TestPendingIgnitions:
    @pytest.mark.asyncio
    async def test_pending_ignitions_stay_bounded(self):
        workspace, states = await _repeated_run()

        # The first occurrence ignites; the repeats are admitted and park in
        # the pending path, whose stale entries are dropped
        assert [s.ignition_events for s in states] == [1] + [0] * 39
        assert [len(s.workspace_contents) for s in states] == [1] * 40
        assert len(workspace.ignition_detector.pending_ignitions) <= 1
        assert workspace.get_statistics()["ignition"]["pending_count"] <= 1

    @pytest.mark.asyncio
    async def test_hot_trigger_names_admission_without_ignition(self):
        workspace, states = await _repeated_run()

        triggers = [[h.trigger for h in s.higher_order_thoughts] for s in states]
        assert triggers[0] == ["workspace_ignition"]
        assert triggers[1:] == [["workspace_admission"]] * 39
        assert sum(s.ignition_events for s in states) == 1


# ============================================================================
# INTEGRATION LEVEL (None without two usable signals)
# ============================================================================


class TestIntegrationLevel:
    """integration_level is a measurement over at least two usable signals
    and None otherwise; the internal consumers read 0.0 for None."""

    @pytest.mark.asyncio
    async def test_integration_level_is_none_when_unmeasurable(self, monkeypatch):
        """No signals, one signal and two constant signals give None, with
        no numpy warning; the homeostasis activity and channel 0 of the
        compact observation read 0.0 for the unmeasured level."""
        workspace = _fast_workspace(
            temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0)
        )
        activities = []
        update = workspace.active_inference.update_homeostasis

        async def spy_update(activity):
            activities.append(dict(activity))
            return await update(activity)

        monkeypatch.setattr(
            workspace.active_inference, "update_homeostasis", spy_update
        )

        cases = {
            "none": None,
            "one": {"snn": np.random.rand(20)},
            "constant": {"snn": np.ones(20), "lsm": np.ones(20)},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for name, signals in cases.items():
                state = await workspace.process_consciousness_cycle(
                    [_strong(f"thought {name}")], neural_signals=signals
                )
                assert state.integration_level is None, name
        assert [a["intensity"] for a in activities] == [0.0, 0.0, 0.0]

        # Channel 0 carries no mass for an unmeasured level (one winner, so
        # the vector is not empty and the uniform fallback does not apply)
        winner = WorkspaceContent(candidate=_strong("observed"), salience=0.6)
        observation = workspace._compact_observation([winner], [], None, None)
        assert observation[0] == 0.0
        assert observation.sum() == pytest.approx(1.0)

        # Two varying signals are a measurement, as before
        measured = await workspace._measure_integration(
            {"snn": np.arange(20.0), "lsm": np.arange(20.0) ** 2}
        )
        assert measured is not None and measured > 0.0

    @pytest.mark.asyncio
    async def test_short_or_degenerate_signals_are_unmeasured(self):
        workspace = _fast_workspace()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # A common length under 2
            assert (
                await workspace._measure_integration(
                    {"snn": np.array([1.0]), "lsm": np.arange(5.0)}
                )
                is None
            )
            # One zero-variance vector beside a varying one: one usable signal
            assert (
                await workspace._measure_integration(
                    {"snn": np.zeros(10), "lsm": np.arange(10.0)}
                )
                is None
            )

    @pytest.mark.asyncio
    async def test_constant_signal_is_dropped_not_fatal(self):
        """A zero-variance vector beside two varying ones is left out of the
        statistic; the level is the mean over the usable pairs, the number
        the two varying signals give on their own, with no numpy warning."""
        workspace = _fast_workspace()
        varying = {"snn": np.arange(20.0), "lsm": np.arange(20.0) ** 2}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            two = await workspace._measure_integration(varying)
            three = await workspace._measure_integration(
                {**varying, "reservoir": np.ones(20)}
            )
        assert two is not None and two > 0.0
        assert three is not None
        assert three == pytest.approx(two)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
