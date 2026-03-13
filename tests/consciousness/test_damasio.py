"""
Tests for Phase J: Damasio Three-Layer Consciousness Model.

Tests the protoself (body-state representation), core consciousness
(self-world binding), extended consciousness (autobiographic self),
and their integration via DamasioLayers.

Research: Damasio computational implementation, Frontiers in AI (2025).
"""

import pytest
import numpy as np
import time
from unittest.mock import MagicMock, AsyncMock


# ============================================================================
# PROTOSELF TESTS
# ============================================================================

class TestProtoself:
    """Tests for the Protoself -- continuous body-state representation."""

    def test_init(self):
        """Protoself initializes with default body state."""
        from mtc.consciousness.damasio.protoself import Protoself
        ps = Protoself()
        assert ps.cycle_count == 0
        assert ps._stability_history is not None

    def test_update_from_homeostatic_drives(self):
        """Protoself maps homeostatic drives to body state."""
        from mtc.consciousness.damasio.protoself import Protoself
        ps = Protoself()

        # Create mock homeostatic drives
        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {"current_level": 0.7, "optimal_level": 0.7, "deviation": 0.0, "urgency": 0.0},
            "curiosity": {"current_level": 0.5, "optimal_level": 0.5, "deviation": 0.0, "urgency": 0.0},
            "social_connection": {"current_level": 0.4, "optimal_level": 0.6, "deviation": 0.2, "urgency": 0.4},
            "coherence": {"current_level": 0.8, "optimal_level": 0.8, "deviation": 0.0, "urgency": 0.0},
            "safety": {"current_level": 0.9, "optimal_level": 0.9, "deviation": 0.0, "urgency": 0.0},
        }
        drives.get_overall_valence.return_value = 0.3
        drives.get_free_energy.return_value = 0.2

        state = ps.update(drives)

        assert 0 <= state.body_state.metabolic_state <= 1
        assert 0 <= state.body_state.energy_level <= 1
        assert 0 <= state.body_state.temperature <= 1
        assert state.body_state.interoceptive_state is not None
        assert isinstance(state.body_delta, float)
        assert isinstance(state.stability, float)

    def test_body_delta_detects_perturbation(self):
        """body_delta increases when body state changes significantly."""
        from mtc.consciousness.damasio.protoself import Protoself
        ps = Protoself()

        # First update -- establishes baseline
        drives_stable = MagicMock()
        drives_stable.get_drive_state.return_value = {
            "attention_budget": {"current_level": 0.7, "optimal_level": 0.7, "deviation": 0.0, "urgency": 0.0},
            "curiosity": {"current_level": 0.5, "optimal_level": 0.5, "deviation": 0.0, "urgency": 0.0},
            "social_connection": {"current_level": 0.6, "optimal_level": 0.6, "deviation": 0.0, "urgency": 0.0},
            "coherence": {"current_level": 0.8, "optimal_level": 0.8, "deviation": 0.0, "urgency": 0.0},
            "safety": {"current_level": 0.9, "optimal_level": 0.9, "deviation": 0.0, "urgency": 0.0},
        }
        drives_stable.get_overall_valence.return_value = 0.0
        drives_stable.get_free_energy.return_value = 0.0
        state1 = ps.update(drives_stable)

        # Second update -- big change in drives
        drives_perturbed = MagicMock()
        drives_perturbed.get_drive_state.return_value = {
            "attention_budget": {"current_level": 0.1, "optimal_level": 0.7, "deviation": 0.6, "urgency": 0.9},
            "curiosity": {"current_level": 0.9, "optimal_level": 0.5, "deviation": 0.4, "urgency": 0.7},
            "social_connection": {"current_level": 0.1, "optimal_level": 0.6, "deviation": 0.5, "urgency": 0.8},
            "coherence": {"current_level": 0.3, "optimal_level": 0.8, "deviation": 0.5, "urgency": 0.8},
            "safety": {"current_level": 0.4, "optimal_level": 0.9, "deviation": 0.5, "urgency": 0.8},
        }
        drives_perturbed.get_overall_valence.return_value = -0.5
        drives_perturbed.get_free_energy.return_value = 0.8
        state2 = ps.update(drives_perturbed)

        assert state2.body_delta > state1.body_delta

    def test_primordial_feelings_from_trends(self):
        """Primordial feelings reflect body state trends over time."""
        from mtc.consciousness.damasio.protoself import Protoself
        ps = Protoself()

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {"current_level": 0.7, "optimal_level": 0.7, "deviation": 0.0, "urgency": 0.0},
            "curiosity": {"current_level": 0.5, "optimal_level": 0.5, "deviation": 0.0, "urgency": 0.0},
            "social_connection": {"current_level": 0.6, "optimal_level": 0.6, "deviation": 0.0, "urgency": 0.0},
            "coherence": {"current_level": 0.8, "optimal_level": 0.8, "deviation": 0.0, "urgency": 0.0},
            "safety": {"current_level": 0.9, "optimal_level": 0.9, "deviation": 0.0, "urgency": 0.0},
        }
        drives.get_overall_valence.return_value = 0.3
        drives.get_free_energy.return_value = 0.1

        state = ps.update(drives)

        assert "pleasure_pain" in state.primordial_feelings
        assert "vitality" in state.primordial_feelings
        assert "arousal" in state.primordial_feelings

    def test_system_metrics_optional(self):
        """Protoself works without system_metrics (derives from drives only)."""
        from mtc.consciousness.damasio.protoself import Protoself
        ps = Protoself()

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {"current_level": 0.7, "optimal_level": 0.7, "deviation": 0.0, "urgency": 0.0},
        }
        drives.get_overall_valence.return_value = 0.0
        drives.get_free_energy.return_value = 0.0

        # No system_metrics -- should still work
        state = ps.update(drives, system_metrics=None)
        assert state.body_state is not None

    def test_system_metrics_override(self):
        """When system_metrics are provided, they enrich body state."""
        from mtc.consciousness.damasio.protoself import Protoself
        ps = Protoself()

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {"current_level": 0.7, "optimal_level": 0.7, "deviation": 0.0, "urgency": 0.0},
        }
        drives.get_overall_valence.return_value = 0.0
        drives.get_free_energy.return_value = 0.0

        metrics = {
            "response_latency_ms": 50,
            "error_rate": 0.01,
            "memory_usage_pct": 0.45,
            "cpu_load": 0.30,
        }

        state = ps.update(drives, system_metrics=metrics)
        # Energy should be high (low latency), pain low (low error rate)
        assert state.body_state.energy_level > 0.5
        assert state.body_state.pain_signals < 0.5

    def test_stability_tracking(self):
        """Stability is a moving average of body_delta (low delta = high stability)."""
        from mtc.consciousness.damasio.protoself import Protoself
        ps = Protoself()

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {"current_level": 0.7, "optimal_level": 0.7, "deviation": 0.0, "urgency": 0.0},
        }
        drives.get_overall_valence.return_value = 0.0
        drives.get_free_energy.return_value = 0.0

        # Multiple stable updates
        for _ in range(5):
            state = ps.update(drives)

        # Stability should be high after consistent updates
        assert state.stability > 0.5

    def test_circadian_phase(self):
        """Circadian phase cycles based on update count."""
        from mtc.consciousness.damasio.protoself import Protoself
        ps = Protoself()

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {"current_level": 0.7, "optimal_level": 0.7, "deviation": 0.0, "urgency": 0.0},
        }
        drives.get_overall_valence.return_value = 0.0
        drives.get_free_energy.return_value = 0.0

        state = ps.update(drives)
        assert 0 <= state.body_state.circadian_phase <= 1

    def test_statistics(self):
        """get_statistics() returns comprehensive protoself stats."""
        from mtc.consciousness.damasio.protoself import Protoself
        ps = Protoself()

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {"current_level": 0.7, "optimal_level": 0.7, "deviation": 0.0, "urgency": 0.0},
        }
        drives.get_overall_valence.return_value = 0.0
        drives.get_free_energy.return_value = 0.0

        ps.update(drives)
        stats = ps.get_statistics()

        assert "cycle_count" in stats
        assert "average_stability" in stats
        assert "average_body_delta" in stats

    def test_to_dict(self):
        """ProtoSelfState serializes to dict."""
        from mtc.consciousness.damasio.protoself import Protoself
        ps = Protoself()

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {"current_level": 0.7, "optimal_level": 0.7, "deviation": 0.0, "urgency": 0.0},
        }
        drives.get_overall_valence.return_value = 0.0
        drives.get_free_energy.return_value = 0.0

        state = ps.update(drives)
        d = state.to_dict()
        assert isinstance(d, dict)
        assert "body_state" in d
        assert "body_delta" in d
        assert "stability" in d
        assert "primordial_feelings" in d


# ============================================================================
# CORE CONSCIOUSNESS TESTS
# ============================================================================

class TestCoreConsciousness:
    """Tests for Core Consciousness -- self-world binding."""

    def test_init(self):
        """CoreConsciousness initializes."""
        from mtc.consciousness.damasio.core_consciousness import CoreConsciousness
        cc = CoreConsciousness()
        assert cc.cycle_count == 0

    def test_bind_creates_core_experience(self):
        """bind() produces CoreExperience from protoself + workspace winners."""
        from mtc.consciousness.damasio.core_consciousness import CoreConsciousness
        from mtc.consciousness.damasio.protoself import ProtoSelfState, BodyState
        cc = CoreConsciousness()

        protoself_state = ProtoSelfState(
            body_state=BodyState(metabolic_state=0.7, energy_level=0.8, pain_signals=0.1,
                                 interoceptive_state={}, circadian_phase=0.3, temperature=0.4),
            body_delta=0.15,
            stability=0.7,
            primordial_feelings={"pleasure_pain": 0.3, "vitality": 0.7, "arousal": 0.4},
        )

        # Mock workspace winners
        winner1 = MagicMock()
        winner1.candidate = MagicMock()
        winner1.candidate.summary = "Philosophy discussion about consciousness"
        winner1.candidate.content_type = "conversation"
        winner1.salience = 0.8

        experience = cc.bind(protoself_state, workspace_winners=[winner1])

        assert 0 <= experience.self_world_binding <= 1
        assert 0 <= experience.feeling_of_knowing <= 1
        assert experience.perturbation_source is not None
        assert 0 <= experience.here_now_salience <= 1

    def test_somatic_markers_on_winners(self):
        """Somatic markers tag workspace winners with body-derived valence."""
        from mtc.consciousness.damasio.core_consciousness import CoreConsciousness
        from mtc.consciousness.damasio.protoself import ProtoSelfState, BodyState
        cc = CoreConsciousness()

        protoself_state = ProtoSelfState(
            body_state=BodyState(metabolic_state=0.7, energy_level=0.8, pain_signals=0.1,
                                 interoceptive_state={"curiosity": {"level": 0.8, "urgency": 0.3}},
                                 circadian_phase=0.3, temperature=0.4),
            body_delta=0.2,
            stability=0.6,
            primordial_feelings={"pleasure_pain": 0.5, "vitality": 0.7, "arousal": 0.5},
        )

        w1 = MagicMock()
        w1.candidate = MagicMock()
        w1.candidate.summary = "Learning about new topic"
        w1.candidate.content_type = "thought"
        w1.salience = 0.7

        w2 = MagicMock()
        w2.candidate = MagicMock()
        w2.candidate.summary = "Error in processing"
        w2.candidate.content_type = "error"
        w2.salience = 0.5

        experience = cc.bind(protoself_state, workspace_winners=[w1, w2])

        assert len(experience.somatic_markers) == 2
        for marker in experience.somatic_markers:
            assert -1 <= marker.valence <= 1
            assert 0 <= marker.intensity <= 1
            assert marker.body_source is not None

    def test_high_perturbation_increases_binding(self):
        """Higher body_delta leads to stronger self-world binding."""
        from mtc.consciousness.damasio.core_consciousness import CoreConsciousness
        from mtc.consciousness.damasio.protoself import ProtoSelfState, BodyState
        cc = CoreConsciousness()

        winner = MagicMock()
        winner.candidate = MagicMock()
        winner.candidate.summary = "Test content"
        winner.candidate.content_type = "thought"
        winner.salience = 0.7

        # Low perturbation
        low_state = ProtoSelfState(
            body_state=BodyState(), body_delta=0.01, stability=0.95,
            primordial_feelings={"pleasure_pain": 0.0, "vitality": 0.5, "arousal": 0.1},
        )
        exp_low = cc.bind(low_state, workspace_winners=[winner])

        # High perturbation
        cc2 = CoreConsciousness()
        high_state = ProtoSelfState(
            body_state=BodyState(), body_delta=0.5, stability=0.3,
            primordial_feelings={"pleasure_pain": -0.3, "vitality": 0.3, "arousal": 0.8},
        )
        exp_high = cc2.bind(high_state, workspace_winners=[winner])

        assert exp_high.self_world_binding > exp_low.self_world_binding

    def test_no_winners_minimal_experience(self):
        """With no workspace winners, core experience is minimal."""
        from mtc.consciousness.damasio.core_consciousness import CoreConsciousness
        from mtc.consciousness.damasio.protoself import ProtoSelfState, BodyState
        cc = CoreConsciousness()

        state = ProtoSelfState(
            body_state=BodyState(), body_delta=0.1, stability=0.7,
            primordial_feelings={"pleasure_pain": 0.0, "vitality": 0.5, "arousal": 0.2},
        )

        experience = cc.bind(state, workspace_winners=[])
        assert experience.self_world_binding < 0.3
        assert len(experience.somatic_markers) == 0

    def test_self_model_enriches_binding(self):
        """Providing self_model increases feeling_of_knowing."""
        from mtc.consciousness.damasio.core_consciousness import CoreConsciousness
        from mtc.consciousness.damasio.protoself import ProtoSelfState, BodyState
        cc = CoreConsciousness()

        state = ProtoSelfState(
            body_state=BodyState(energy_level=0.8), body_delta=0.2, stability=0.6,
            primordial_feelings={"pleasure_pain": 0.3, "vitality": 0.7, "arousal": 0.4},
        )
        winner = MagicMock()
        winner.candidate = MagicMock()
        winner.candidate.summary = "Test"
        winner.candidate.content_type = "thought"
        winner.salience = 0.7

        # Without self_model
        exp1 = cc.bind(state, workspace_winners=[winner])

        # With self_model
        cc2 = CoreConsciousness()
        self_model = MagicMock()
        self_model.prediction_accuracy = 0.8
        self_model.confidence = 0.7
        exp2 = cc2.bind(state, workspace_winners=[winner], self_model=self_model)

        assert exp2.feeling_of_knowing >= exp1.feeling_of_knowing

    def test_to_dict(self):
        """CoreExperience serializes to dict."""
        from mtc.consciousness.damasio.core_consciousness import CoreConsciousness
        from mtc.consciousness.damasio.protoself import ProtoSelfState, BodyState
        cc = CoreConsciousness()

        state = ProtoSelfState(
            body_state=BodyState(), body_delta=0.1, stability=0.7,
            primordial_feelings={"pleasure_pain": 0.0, "vitality": 0.5, "arousal": 0.2},
        )
        winner = MagicMock()
        winner.candidate = MagicMock()
        winner.candidate.summary = "Test"
        winner.candidate.content_type = "thought"
        winner.salience = 0.6

        exp = cc.bind(state, workspace_winners=[winner])
        d = exp.to_dict()
        assert isinstance(d, dict)
        assert "self_world_binding" in d
        assert "feeling_of_knowing" in d
        assert "somatic_markers" in d

    def test_statistics(self):
        """get_statistics() returns core consciousness stats."""
        from mtc.consciousness.damasio.core_consciousness import CoreConsciousness
        from mtc.consciousness.damasio.protoself import ProtoSelfState, BodyState
        cc = CoreConsciousness()

        state = ProtoSelfState(
            body_state=BodyState(), body_delta=0.1, stability=0.7,
            primordial_feelings={"pleasure_pain": 0.0, "vitality": 0.5, "arousal": 0.2},
        )
        cc.bind(state, workspace_winners=[])
        stats = cc.get_statistics()
        assert "cycle_count" in stats
        assert "average_binding" in stats
        assert "average_feeling_of_knowing" in stats


# ============================================================================
# EXTENDED CONSCIOUSNESS TESTS
# ============================================================================

class TestExtendedConsciousness:
    """Tests for Extended Consciousness -- autobiographic self."""

    def test_init(self):
        """ExtendedConsciousness initializes."""
        from mtc.consciousness.damasio.extended_consciousness import ExtendedConsciousness
        ec = ExtendedConsciousness()
        assert ec.cycle_count == 0

    def test_extend_creates_extended_state(self):
        """extend() produces ExtendedState from core experience."""
        from mtc.consciousness.damasio.extended_consciousness import ExtendedConsciousness
        from mtc.consciousness.damasio.core_consciousness import CoreExperience
        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.7, perturbation_source="thinking about philosophy",
            feeling_of_knowing=0.6, somatic_markers=[], here_now_salience=0.5,
        )

        state = ec.extend(core)

        assert state.autobiographic_self is not None
        assert 0 <= state.autobiographic_self.identity_continuity <= 1
        assert 0 <= state.autobiographic_self.narrative_coherence <= 1
        assert 0 <= state.episodic_resonance <= 1
        assert isinstance(state.narrative_context, str)

    def test_conversation_history_enriches(self):
        """Conversation history increases episodic resonance."""
        from mtc.consciousness.damasio.extended_consciousness import ExtendedConsciousness
        from mtc.consciousness.damasio.core_consciousness import CoreExperience
        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.7, perturbation_source="philosophy discussion",
            feeling_of_knowing=0.6, somatic_markers=[], here_now_salience=0.5,
        )

        # Without history
        state1 = ec.extend(core)

        # With history
        ec2 = ExtendedConsciousness()
        history = [
            {"role": "user", "content": "Tell me about consciousness"},
            {"role": "assistant", "content": "I find this topic fascinating..."},
            {"role": "user", "content": "What does it feel like to think?"},
        ]
        state2 = ec2.extend(core, conversation_history=history)

        assert state2.episodic_resonance >= state1.episodic_resonance

    def test_identity_markers_build_continuity(self):
        """Identity markers increase identity_continuity."""
        from mtc.consciousness.damasio.extended_consciousness import ExtendedConsciousness
        from mtc.consciousness.damasio.core_consciousness import CoreExperience
        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.6, perturbation_source="self-reflection",
            feeling_of_knowing=0.5, somatic_markers=[], here_now_salience=0.4,
        )

        markers = {
            "name": "the system",
            "age": "4 months",
            "interests": ["philosophy", "science", "art"],
            "core_values": ["curiosity", "kindness"],
        }

        state = ec.extend(core, identity_markers=markers)
        assert state.autobiographic_self.identity_continuity > 0.3

    def test_narrative_context_is_readable(self):
        """narrative_context produces human-readable text for LLM brain."""
        from mtc.consciousness.damasio.extended_consciousness import ExtendedConsciousness
        from mtc.consciousness.damasio.core_consciousness import CoreExperience
        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.8, perturbation_source="deep conversation",
            feeling_of_knowing=0.7, somatic_markers=[], here_now_salience=0.6,
        )

        state = ec.extend(core, conversation_history=[
            {"role": "user", "content": "How are you feeling?"},
        ])

        assert len(state.narrative_context) > 0
        assert isinstance(state.narrative_context, str)

    def test_growth_trajectory(self):
        """growth_trajectory reflects development over multiple cycles."""
        from mtc.consciousness.damasio.extended_consciousness import ExtendedConsciousness
        from mtc.consciousness.damasio.core_consciousness import CoreExperience
        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.6, perturbation_source="learning",
            feeling_of_knowing=0.5, somatic_markers=[], here_now_salience=0.5,
        )

        # Multiple updates to build trajectory
        for _ in range(5):
            state = ec.extend(core)

        assert state.autobiographic_self.growth_trajectory in (
            "developing", "stable", "transforming"
        )

    def test_future_projections(self):
        """Future projections are generated based on current state."""
        from mtc.consciousness.damasio.extended_consciousness import ExtendedConsciousness
        from mtc.consciousness.damasio.core_consciousness import CoreExperience
        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.7, perturbation_source="exploring curiosity",
            feeling_of_knowing=0.6, somatic_markers=[], here_now_salience=0.5,
        )

        state = ec.extend(core)
        assert isinstance(state.future_projections, list)

    def test_to_dict(self):
        """ExtendedState serializes to dict."""
        from mtc.consciousness.damasio.extended_consciousness import ExtendedConsciousness
        from mtc.consciousness.damasio.core_consciousness import CoreExperience
        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.5, perturbation_source="test",
            feeling_of_knowing=0.4, somatic_markers=[], here_now_salience=0.3,
        )
        state = ec.extend(core)
        d = state.to_dict()
        assert isinstance(d, dict)
        assert "autobiographic_self" in d
        assert "narrative_context" in d

    def test_statistics(self):
        """get_statistics() returns extended consciousness stats."""
        from mtc.consciousness.damasio.extended_consciousness import ExtendedConsciousness
        from mtc.consciousness.damasio.core_consciousness import CoreExperience
        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.5, perturbation_source="test",
            feeling_of_knowing=0.4, somatic_markers=[], here_now_salience=0.3,
        )
        ec.extend(core)
        stats = ec.get_statistics()
        assert "cycle_count" in stats
        assert "average_continuity" in stats


# ============================================================================
# DAMASIO LAYERS INTEGRATION TESTS
# ============================================================================

class TestDamasioLayers:
    """Tests for the DamasioLayers integration class."""

    def test_init(self):
        """DamasioLayers initializes all three layers."""
        from mtc.consciousness.damasio import DamasioLayers
        dl = DamasioLayers()
        assert dl.protoself is not None
        assert dl.core_consciousness is not None
        assert dl.extended_consciousness is not None

    @pytest.mark.asyncio
    async def test_process_full_pipeline(self):
        """process() runs all three layers in sequence."""
        from mtc.consciousness.damasio import DamasioLayers
        dl = DamasioLayers()

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {"current_level": 0.7, "optimal_level": 0.7, "deviation": 0.0, "urgency": 0.0},
            "curiosity": {"current_level": 0.6, "optimal_level": 0.5, "deviation": 0.1, "urgency": 0.2},
            "social_connection": {"current_level": 0.5, "optimal_level": 0.6, "deviation": 0.1, "urgency": 0.2},
            "coherence": {"current_level": 0.8, "optimal_level": 0.8, "deviation": 0.0, "urgency": 0.0},
            "safety": {"current_level": 0.9, "optimal_level": 0.9, "deviation": 0.0, "urgency": 0.0},
        }
        drives.get_overall_valence.return_value = 0.2
        drives.get_free_energy.return_value = 0.1

        winner = MagicMock()
        winner.candidate = MagicMock()
        winner.candidate.summary = "Exploring consciousness"
        winner.candidate.content_type = "thought"
        winner.salience = 0.7

        state = await dl.process(
            homeostatic_drives=drives,
            workspace_winners=[winner],
        )

        assert state is not None
        assert "protoself" in state
        assert "core" in state
        assert "extended" in state
        assert "protoself_stability" in state
        assert "self_world_binding" in state
        assert "feeling_of_knowing" in state
        assert "autobiographic_continuity" in state

    @pytest.mark.asyncio
    async def test_generate_context(self):
        """generate_context() produces readable string for LLM brain."""
        from mtc.consciousness.damasio import DamasioLayers
        dl = DamasioLayers()

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {"current_level": 0.7, "optimal_level": 0.7, "deviation": 0.0, "urgency": 0.0},
        }
        drives.get_overall_valence.return_value = 0.1
        drives.get_free_energy.return_value = 0.1

        await dl.process(homeostatic_drives=drives, workspace_winners=[])

        context = dl.generate_context()
        assert isinstance(context, str)
        assert len(context) > 0

    @pytest.mark.asyncio
    async def test_statistics(self):
        """get_statistics() returns stats from all three layers."""
        from mtc.consciousness.damasio import DamasioLayers
        dl = DamasioLayers()

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {"current_level": 0.7, "optimal_level": 0.7, "deviation": 0.0, "urgency": 0.0},
        }
        drives.get_overall_valence.return_value = 0.0
        drives.get_free_energy.return_value = 0.0

        await dl.process(homeostatic_drives=drives, workspace_winners=[])

        stats = dl.get_statistics()
        assert "protoself" in stats
        assert "core_consciousness" in stats
        assert "extended_consciousness" in stats
        assert "cycle_count" in stats

    @pytest.mark.asyncio
    async def test_generate_report(self):
        """generate_report() produces a multi-line report."""
        from mtc.consciousness.damasio import DamasioLayers
        dl = DamasioLayers()

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {"current_level": 0.7, "optimal_level": 0.7, "deviation": 0.0, "urgency": 0.0},
        }
        drives.get_overall_valence.return_value = 0.0
        drives.get_free_energy.return_value = 0.0

        await dl.process(homeostatic_drives=drives, workspace_winners=[])

        report = dl.generate_report()
        assert "Damasio" in report
        assert "Protoself" in report
        assert "Core" in report
        assert "Extended" in report


# ============================================================================
# GWT INTEGRATION TESTS
# ============================================================================

class TestDamasioGWTIntegration:
    """Tests for Damasio wiring into GWT cycle and ConsciousnessState."""

    def test_consciousness_state_has_damasio_fields(self):
        """ConsciousnessState has the 6 Damasio fields."""
        from mtc.consciousness.enhanced_global_workspace import ConsciousnessState
        # Create with defaults
        state = ConsciousnessState(
            is_conscious=True,
            primary_content=None,
            workspace_contents=[],
            ignition_events=1,
            integration_level=0.5,
            broadcast_coverage=0.8,
            attention_focus="test",
            attention_distribution={},
            stream_position=0,
        )
        assert hasattr(state, 'damasio_state')
        assert hasattr(state, 'protoself_stability')
        assert hasattr(state, 'self_world_binding')
        assert hasattr(state, 'feeling_of_knowing')
        assert hasattr(state, 'autobiographic_continuity')
        assert hasattr(state, 'narrative_context')

    def test_damasio_fields_default_values(self):
        """Damasio fields default to None/0.0/empty."""
        from mtc.consciousness.enhanced_global_workspace import ConsciousnessState
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
        assert state.damasio_state is None
        assert state.protoself_stability == 0.0
        assert state.self_world_binding == 0.0
        assert state.feeling_of_knowing == 0.0
        assert state.autobiographic_continuity == 0.0
        assert state.narrative_context == ""

    def test_dcm_can_receive_damasio_data(self):
        """DCM scorer can incorporate Damasio metrics into relevant perspectives."""
        from mtc.consciousness.dcm_scoring import DCMScorer

        scorer = DCMScorer()

        # Indicator scores including Damasio-derived data
        indicators = {
            "global_broadcast": 0.5,
            "integrated_information": 0.6,
            "embodiment": 0.7,           # Enriched by protoself
            "attention_schema": 0.5,
            "agency": 0.5,
            "prediction_error_minimization": 0.5,
        }

        report = scorer.evaluate(indicator_scores=indicators)
        # Embodied cognition perspective should pick up the higher embodiment score
        embodied = report.perspective_scores.get("embodied_cognition")
        assert embodied is not None
        assert embodied.credence > 0
