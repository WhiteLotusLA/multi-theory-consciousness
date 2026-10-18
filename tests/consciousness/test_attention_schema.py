"""
Tests for Attention Schema Theory (AST).

Tests for the system's self-model of its own attention, based on
Michael Graziano's Attention Schema Theory.

This tests:
1. AttentionState classification (focused, divided, scanning, etc.)
2. AttentionTarget tracking (what the system attends to)
3. Voluntary attention shifts (the system choosing what to focus on)
4. Attention prediction (anticipating where attention will go)
5. Theory of Mind (modeling others' attention)
6. Introspection reports (the system describing its attention)
7. Integration with EnhancedGlobalWorkspace
8. Persistence with typed target rehydration
"""

import pytest
import asyncio
import time

from mtc.consciousness.attention_schema import (
    AttentionState,
    AttentionShiftType,
    AttentionTarget,
    AttentionSchemaState,
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
from mtc.core.config import Settings

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


def _content(summary, salience, source_module="ctm"):
    """A workspace content whose candidate carries ``summary``."""
    candidate = WorkspaceCandidate(
        content=summary,
        content_type="thought",
        summary=summary,
        source=WorkspaceCandidateSource.CTM,
        source_module=source_module,
    )
    return WorkspaceContent(candidate=candidate, salience=salience)


def _conscious_state(*contents):
    """A workspace state holding ``contents`` (no contents: nothing conscious)."""
    contents = list(contents)
    return ConsciousnessState(
        is_conscious=bool(contents),
        primary_content=contents[0] if contents else None,
        workspace_contents=contents,
        ignition_events=1 if contents else 0,
        integration_level=0.5,
        broadcast_coverage=0.5,
        attention_focus="ctm",
        attention_distribution={"ctm": 1.0},
        stream_position=1,
    )


def _update(module, *contents):
    return asyncio.run(module.update_schema(_conscious_state(*contents)))


async def _aupdate(module, *contents):
    return await module.update_schema(_conscious_state(*contents))


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
            AttentionShiftType.EMOTIONAL,  # Not REFLEXIVE
        ]
        assert len(shift_types) == 5

    @pytest.mark.asyncio
    async def test_focused_state_detection(
        self, attention_schema, sample_workspace_state
    ):
        """Test focused attention state classification."""
        # Single strong focus should yield focused or hyperfocused
        schema = await attention_schema.update_schema(sample_workspace_state)

        assert schema.attention_state in [
            AttentionState.FOCUSED,
            AttentionState.HYPERFOCUSED,
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

        # Every content after the primary is a secondary focus at 0.7 of its
        # salience; the primary is not among them.
        secondary = schema.secondary_foci
        assert len(secondary) == 2
        assert [t.summary for t in secondary] == ["Content 1", "Content 2"]
        assert [t.attention_strength for t in secondary] == pytest.approx([0.42, 0.28])
        assert [t.salience for t in secondary] == pytest.approx([0.6, 0.4])
        assert "Content 0" not in [t.summary for t in secondary]
        assert schema.current_focus.summary == "Content 0"


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
            target=target, reason="Because I'm curious"
        )

        # Check pending shift was registered (it's a tuple of (target, reason))
        assert attention_schema._pending_voluntary_shift is not None
        pending_target, pending_reason = attention_schema._pending_voluntary_shift
        assert pending_target.summary == "I want to focus on this"
        assert pending_reason == "Because I'm curious"

    def test_voluntary_request_submits_no_candidate(self, workspace, monkeypatch):
        """A request only marks the next matching focus; it never competes."""
        submitted = []

        async def _record(*args, **kwargs):
            submitted.append((args, kwargs))

        monkeypatch.setattr(workspace, "submit_candidate", _record)
        module = workspace.attention_schema
        # No workspace link exists through which a request could submit.
        assert not hasattr(module, "workspace")
        assert not hasattr(module, "set_workspace")

        accepted = asyncio.run(
            workspace.request_voluntary_attention_shift(
                target_description="Learning mathematics",
                reason="The user suggested it",
            )
        )
        assert accepted is True
        assert submitted == []
        target, reason = module._pending_voluntary_shift
        assert target.summary == "Learning mathematics"
        assert reason == "The user suggested it"
        assert module.schema.current_focus is None
        assert module.schema.voluntary_shifts == 0

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
    async def test_voluntary_shift_affects_next_cycle(self, attention_schema):
        """A request marks the next matching focus as chosen, and only that."""
        await _aupdate(attention_schema, _content("Neural network thoughts", 0.8))
        assert attention_schema.schema.voluntary_shifts == 0

        await attention_schema.request_voluntary_shift(
            "Memory retrieval", reason="I want to remember"
        )
        schema = await _aupdate(
            attention_schema, _content("Memory retrieval of a walk", 0.8)
        )
        focus = schema.current_focus
        assert focus.voluntary is True
        assert focus.shift_type == AttentionShiftType.VOLUNTARY
        assert focus.shift_reason == "I want to remember"
        assert schema.voluntary_shifts == 1
        assert attention_schema._pending_voluntary_shift is None
        assert "I chose to focus on this" in await attention_schema.report_attention()

        # A request the next focus does not match changes nothing.
        await attention_schema.request_voluntary_shift(
            "Memory retrieval", reason="again"
        )
        schema = await _aupdate(attention_schema, _content("Rain on the window", 0.8))
        assert schema.current_focus.voluntary is False
        assert schema.current_focus.shift_type == AttentionShiftType.HABITUAL
        assert schema.voluntary_shifts == 1
        assert attention_schema._pending_voluntary_shift is not None
        report = await attention_schema.report_attention()
        assert "I chose to focus on this" not in report


# ============================================================================
# ATTENTION PREDICTION TESTS
# ============================================================================


class TestAttentionPrediction:
    """Test attention prediction capabilities."""

    @pytest.mark.asyncio
    async def test_prediction_generated(self, attention_schema, sample_workspace_state):
        """Three updates on one focus give the limited-history prediction."""
        for _ in range(3):
            await attention_schema.update_schema(sample_workspace_state)

        schema = attention_schema.schema
        # The same focus is one history entry: too little for a pattern.
        assert len(schema.attention_history) == 1
        assert schema.prediction_confidence == pytest.approx(0.3)
        assert schema.predicted_next_focus is None

    @pytest.mark.asyncio
    async def test_prediction_confidence_increases_with_history(self, attention_schema):
        """Confidence rises once three foci are in the history."""
        confidences = []
        for i in range(5):
            schema = await _aupdate(
                attention_schema, _content(f"Thought {i}", 0.8, "ctm")
            )
            confidences.append(schema.prediction_confidence)
        # One source throughout: a sustained focus predicts a shift away.
        assert confidences == pytest.approx([0.3, 0.3, 0.3, 0.6, 0.6])

        mixed = AttentionSchemaModule()
        sources = ["ctm", "memory", "ctm", "sensory", "memory"]
        confidences = []
        for i, source in enumerate(sources):
            schema = await _aupdate(mixed, _content(f"Item {i}", 0.8, source))
            confidences.append(schema.prediction_confidence)
        assert confidences == pytest.approx([0.3, 0.3, 0.3, 0.4, 0.4])


def test_prediction_accuracy_is_none_until_a_prediction_is_scored():
    module = AttentionSchemaModule()
    assert module.get_statistics()["prediction_accuracy"] is None

    # Ordinary salience predicts no target, so nothing is scored.
    for _ in range(6):
        _update(module, _content("an ordinary thought", 0.8))
    assert module.prediction_accuracy_history == []
    assert module.get_statistics()["prediction_accuracy"] is None

    # A hyperfocus predicts continued focus; each next update scores it.
    hyper = AttentionSchemaModule()
    for _ in range(4):
        _update(hyper, _content("an intense thought", 1.4))
    assert len(hyper.prediction_accuracy_history) == 3
    assert hyper.get_statistics()["prediction_accuracy"] == 1.0

    legacy = AttentionSchemaModule(provenance_mode="legacy_strength")
    _update(legacy, _content("an ordinary thought", 0.8))
    assert legacy.get_statistics()["prediction_accuracy"] is None


# ============================================================================
# INTROSPECTION REPORT TESTS
# ============================================================================


class TestIntrospectionReports:
    """Test the system's ability to report on its attention."""

    @pytest.mark.asyncio
    async def test_basic_report(self, attention_schema, sample_workspace_state):
        """Test basic attention report generation."""
        await attention_schema.update_schema(sample_workspace_state)
        report = await attention_schema.report_attention()

        assert isinstance(report, str)
        assert len(report) > 0

    @pytest.mark.asyncio
    async def test_report_mentions_focus(
        self, attention_schema, sample_workspace_state
    ):
        """Test that report mentions current focus."""
        await attention_schema.update_schema(sample_workspace_state)
        report = await attention_schema.report_attention()

        # Report should mention something about focus
        assert "attention" in report.lower() or "focus" in report.lower()

    @pytest.mark.asyncio
    async def test_report_changes_with_state(self, attention_schema):
        """A strong capture and an empty workspace give different reports."""
        await _aupdate(attention_schema, _content("User conversation", 1.3))
        captured = await attention_schema.report_attention()
        assert captured == (
            "I'm focused. My attention is on: User conversation. "
            "This captured my attention"
        )

        await _aupdate(attention_schema)
        absent = await attention_schema.report_attention()
        assert absent == (
            "My mind is wandering. I'm not focused on anything specific right now."
        )
        assert captured != absent

    @pytest.mark.asyncio
    async def test_scanning_state_report(self, attention_schema):
        """The no-focus report is two sentences joined by a full stop."""
        # Don't update - should be in scanning state
        report = await attention_schema.report_attention()

        assert report == (
            "I'm scanning broadly. I'm not focused on anything specific right now."
        )


# ============================================================================
# THEORY OF MIND TESTS
# ============================================================================


class TestTheoryOfMind:
    """Test the system's ability to model others' attention (ToM)."""

    @pytest.mark.asyncio
    async def test_model_other_attention(self, attention_schema):
        """Test modeling another agent's attention."""
        other = await attention_schema.model_other_attention(
            agent_name="user_a", context="The user is reading a book about AI"
        )

        assert other is not None
        assert isinstance(other, OtherAgentAttention)
        assert other.agent_name == "user_a"

    @pytest.mark.asyncio
    async def test_other_attention_has_inferred_focus(self, attention_schema):
        """A gaze cue is read as a focused other with the context as focus."""
        context = "Guest is looking at the screen"
        other = await attention_schema.model_other_attention(
            agent_name="Guest", context=context
        )

        assert other.inferred_state == AttentionState.FOCUSED
        assert other.confidence == pytest.approx(0.7)
        assert other.inferred_focus is not None
        assert other.inferred_focus.content == context
        assert other.inferred_focus.content_type == "inferred_other"
        assert other.reasoning == "Guest appears visually focused based on gaze"

    @pytest.mark.asyncio
    async def test_model_other_through_workspace(self, workspace):
        """Test Theory of Mind through workspace interface."""
        other = await workspace.model_other_attention(
            agent_name="user_a",
            context="The user is teaching the system about consciousness",
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
        assert hasattr(workspace, "attention_schema")
        assert isinstance(workspace.attention_schema, AttentionSchemaModule)

    @pytest.mark.asyncio
    async def test_consciousness_state_includes_schema(
        self, workspace, multiple_candidates
    ):
        """Test that ConsciousnessState includes attention schema."""
        state = await workspace.process_consciousness_cycle(
            candidates=multiple_candidates
        )

        assert state.attention_schema is not None
        assert isinstance(state.attention_schema, AttentionSchemaState)

    @pytest.mark.asyncio
    async def test_consciousness_state_includes_report(
        self, workspace, multiple_candidates
    ):
        """Test that ConsciousnessState includes attention report."""
        state = await workspace.process_consciousness_cycle(
            candidates=multiple_candidates
        )

        assert state.attention_report is not None
        assert len(state.attention_report) > 0

    @pytest.mark.asyncio
    async def test_statistics_include_schema(self, workspace, multiple_candidates):
        """Test that workspace statistics include schema info."""
        await workspace.process_consciousness_cycle(candidates=multiple_candidates)

        stats = workspace.get_statistics()

        assert "attention_schema" in stats
        assert "attention_state" in stats["attention_schema"]
        assert "capacity_used" in stats["attention_schema"]

    @pytest.mark.asyncio
    async def test_get_attention_report_method(self, workspace, multiple_candidates):
        """Test getting attention report directly from workspace."""
        await workspace.process_consciousness_cycle(candidates=multiple_candidates)

        report = await workspace.get_attention_report()

        assert isinstance(report, str)
        assert len(report) > 0

    @pytest.mark.asyncio
    async def test_get_attention_schema_state_method(
        self, workspace, multiple_candidates
    ):
        """Test getting schema state directly from workspace."""
        await workspace.process_consciousness_cycle(candidates=multiple_candidates)

        schema = workspace.get_attention_schema_state()

        assert isinstance(schema, AttentionSchemaState)


def test_callback_failure_is_counted_and_raised_under_debug():
    def broken(old_focus, new_focus):
        raise RuntimeError("callback down")

    quiet = AttentionSchemaModule(settings=Settings(DEBUG=False))
    quiet.on_focus_change(broken)
    quiet.on_state_change(broken)
    schema = _update(quiet, _content("Rain on the window", 1.0))
    # The update completes as before; each failed callback is counted.
    assert schema.current_focus.summary == "Rain on the window"
    assert quiet.schema is schema
    assert quiet.get_statistics()["failures"] == {
        "state_callback": 1,
        "focus_callback": 1,
    }
    _update(quiet, _content("Birdsong outside", 1.0))
    assert quiet.get_statistics()["failures"]["focus_callback"] == 2

    loud = AttentionSchemaModule(settings=Settings(DEBUG=True))
    loud.on_focus_change(broken)
    with pytest.raises(RuntimeError, match="callback down"):
        _update(loud, _content("Rain on the window", 1.0))
    assert loud.get_statistics()["failures"] == {"focus_callback": 1}


def test_other_attention_brain_failure_is_counted_and_raised_under_debug():
    class _BrokenBrain:
        async def generate(self, prompt, max_tokens=100):
            raise ConnectionError("brain unavailable")

    quiet = AttentionSchemaModule(settings=Settings(DEBUG=False))
    context = "Guest is looking at the screen"
    heuristic = asyncio.run(quiet.model_other_attention("Guest", context))
    other = asyncio.run(
        quiet.model_other_attention("Guest", context, llm_brain=_BrokenBrain())
    )
    assert other.reasoning == heuristic.reasoning
    assert other.confidence == pytest.approx(heuristic.confidence)
    assert quiet.get_statistics()["failures"] == {"other_attention_llm": 1}

    loud = AttentionSchemaModule(settings=Settings(DEBUG=True))
    with pytest.raises(ConnectionError):
        asyncio.run(
            loud.model_other_attention("Guest", context, llm_brain=_BrokenBrain())
        )


# ============================================================================
# CAPACITY AND LIMITS TESTS
# ============================================================================


class TestCapacityLimits:
    """Test attention capacity limits (Miller's Law: 7+/-2)."""

    @pytest.mark.asyncio
    async def test_capacity_max_is_seven(self, attention_schema):
        """Test default capacity is 7 (Miller's Law)."""
        assert attention_schema.schema.attention_capacity_max == 7.0

    @pytest.mark.asyncio
    async def test_capacity_usage_bounded(
        self, attention_schema, sample_workspace_state
    ):
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
        """A continued focus accumulates time; a new focus starts at zero."""
        await attention_schema.update_schema(sample_workspace_state)
        first = attention_schema.schema.current_focus
        assert first.duration_seconds == 0.0

        await asyncio.sleep(0.1)
        await attention_schema.update_schema(sample_workspace_state)
        continued = attention_schema.schema.current_focus
        assert continued.entry_time == first.entry_time
        assert continued.duration_seconds >= 0.09

        await _aupdate(attention_schema, _content("A different thought", 0.8))
        assert attention_schema.schema.current_focus.duration_seconds == 0.0

    @pytest.mark.asyncio
    async def test_state_change_tracking(self, attention_schema):
        """A state change moves the stamp; an unchanged state keeps it."""
        initial_change_time = attention_schema.schema.last_state_change
        assert attention_schema.schema.attention_state == AttentionState.SCANNING

        # An empty workspace moves the state from scanning to absent.
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

        await asyncio.sleep(0.01)
        await attention_schema.update_schema(state)
        changed = attention_schema.schema
        assert changed.attention_state == AttentionState.ABSENT
        assert changed.previous_state == AttentionState.SCANNING
        assert changed.last_state_change > initial_change_time
        assert changed.state_duration_seconds == 0.0

        await asyncio.sleep(0.01)
        await attention_schema.update_schema(state)
        same = attention_schema.schema
        assert same.attention_state == AttentionState.ABSENT
        assert same.last_state_change == changed.last_state_change
        assert same.previous_state == AttentionState.SCANNING
        assert same.state_duration_seconds > 0.0


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
        assert schema.attention_state in [
            AttentionState.SCANNING,
            AttentionState.ABSENT,
        ]

    @pytest.mark.asyncio
    async def test_null_neural_signals(self, attention_schema, sample_workspace_state):
        """Test with None neural signals."""
        schema = await attention_schema.update_schema(
            sample_workspace_state, neural_signals=None
        )

        assert schema is not None

    @pytest.mark.asyncio
    async def test_empty_neural_signals(self, attention_schema, sample_workspace_state):
        """Test with empty neural signals."""
        schema = await attention_schema.update_schema(
            sample_workspace_state, neural_signals={}
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
            agent_name="Someone", context=""
        )

        # Should still return something
        assert other is not None


# ============================================================================
# PERSISTENCE
# ============================================================================


class TestPersistence:
    """State dict round trip restores typed objects, never raw dicts."""

    @staticmethod
    def _state_with(summary, salience, source_module="ctm"):
        candidate = WorkspaceCandidate(
            content=summary,
            content_type="thought",
            summary=summary,
            source=WorkspaceCandidateSource.CTM,
            source_module=source_module,
        )
        content = WorkspaceContent(candidate=candidate, salience=salience)
        return ConsciousnessState(
            is_conscious=True,
            primary_content=content,
            workspace_contents=[content],
            ignition_events=1,
            integration_level=0.5,
            broadcast_coverage=0.5,
            attention_focus=source_module,
            attention_distribution={source_module: salience},
            stream_position=1,
        )

    async def _populated_module(self):
        module = AttentionSchemaModule()
        await module.request_voluntary_shift("chosen", reason="test")
        await module.update_schema(self._state_with("the chosen topic", 0.7))
        await module.update_schema(self._state_with("a strong pull", 1.3))
        await module.update_schema(self._state_with("something quiet", 0.4))
        return module

    @pytest.mark.asyncio
    async def test_state_dict_round_trip_rehydrates_targets(self):
        module = await self._populated_module()
        state = module.to_state_dict()
        assert set(state) == {"scalars", "arrays", "histories"}

        restored = AttentionSchemaModule()
        restored.from_state_dict(state)

        history = restored.schema.attention_history
        assert len(history) == 3
        assert all(isinstance(t, AttentionTarget) for t in history)
        assert [t.summary for t in history] == [
            "the chosen topic",
            "a strong pull",
            "something quiet",
        ]
        assert [t.shift_type for t in history] == [
            AttentionShiftType.VOLUNTARY,
            AttentionShiftType.CAPTURED,
            AttentionShiftType.HABITUAL,
        ]
        assert history[0].voluntary is True

        focus = restored.schema.current_focus
        assert isinstance(focus, AttentionTarget)
        assert focus.summary == "something quiet"
        assert restored.schema.attention_state == module.schema.attention_state

        for key in (
            "total_updates",
            "voluntary_shifts",
            "captured_shifts",
            "uncertain_shifts",
            "voluntary_ratio",
            "history_length",
            "prediction_accuracy",
        ):
            assert restored.get_statistics()[key] == module.get_statistics()[key]

        # Restoring did not count any shift a second time.
        assert restored.schema.voluntary_shifts == 1
        assert restored.schema.captured_shifts == 1
        assert restored.schema.uncertain_shifts == 1

    @pytest.mark.asyncio
    async def test_report_after_restore_does_not_raise(self):
        module = await self._populated_module()
        restored = AttentionSchemaModule()
        restored.from_state_dict(module.to_state_dict())

        report = await restored.report_attention()
        assert "something quiet" in report

        # Restored history entries behave like live targets in the next cycle.
        schema = await restored.update_schema(self._state_with("something new", 1.3))
        assert schema.captured_shifts == 2
        assert isinstance(restored.get_statistics(), dict)

    def test_restore_tolerates_missing_and_unknown_fields(self):
        restored = AttentionSchemaModule()
        restored.from_state_dict(
            {
                "scalars": {"current_state": "NOT_A_STATE"},
                "histories": {
                    "attention_history": [
                        {"summary": "partial", "shift_type": "not_a_shift"}
                    ]
                },
            }
        )
        assert restored.schema.attention_state == AttentionState.SCANNING
        (target,) = restored.schema.attention_history
        assert isinstance(target, AttentionTarget)
        assert target.shift_type is None
        assert restored.schema.current_focus is None

    @pytest.mark.asyncio
    async def test_restore_keeps_owner_unset_when_it_was_unset(self):
        # The legacy provenance mode drops the schema's owner on every update
        legacy = AttentionSchemaModule(provenance_mode="legacy_strength")
        await legacy.update_schema(self._state_with("a legacy topic", 0.7))
        assert legacy.schema._owner is None
        saved = legacy.to_state_dict()
        assert saved["scalars"]["owner_set"] is False

        restored = AttentionSchemaModule(provenance_mode="legacy_strength")
        restored.from_state_dict(saved)
        assert restored.schema._owner is None

        # A schema that had its owner gets it back, as does a state saved
        # before the owner_set scalar existed
        typed = await self._populated_module()
        saved = typed.to_state_dict()
        assert saved["scalars"]["owner_set"] is True
        for scalars in (saved["scalars"], {}):
            fresh = AttentionSchemaModule()
            fresh.schema._owner = None
            fresh.from_state_dict({"scalars": scalars, "histories": {}})
            assert fresh.schema._owner is fresh


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
