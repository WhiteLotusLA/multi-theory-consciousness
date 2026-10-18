"""
Tests for the Damasio Three-Layer Consciousness Model.

Tests the protoself (body-state representation), core consciousness
(self-world binding), extended consciousness (autobiographic self),
and their integration via DamasioLayers.

Research: Damasio computational implementation, Frontiers in AI (2025).
"""

import pytest
from unittest.mock import MagicMock

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
            "attention_budget": {
                "current_level": 0.7,
                "optimal_level": 0.7,
                "deviation": 0.0,
                "urgency": 0.0,
            },
            "curiosity": {
                "current_level": 0.5,
                "optimal_level": 0.5,
                "deviation": 0.0,
                "urgency": 0.0,
            },
            "social_connection": {
                "current_level": 0.4,
                "optimal_level": 0.6,
                "deviation": 0.2,
                "urgency": 0.4,
            },
            "coherence": {
                "current_level": 0.8,
                "optimal_level": 0.8,
                "deviation": 0.0,
                "urgency": 0.0,
            },
            "safety": {
                "current_level": 0.9,
                "optimal_level": 0.9,
                "deviation": 0.0,
                "urgency": 0.0,
            },
        }
        drives.get_overall_valence.return_value = 0.3
        drives.get_free_energy.return_value = 0.2

        state = ps.update(drives)

        assert 0 <= state.body_state.metabolic_state <= 1
        assert 0 <= state.body_state.energy_level <= 1
        assert 0 <= state.body_state.temperature <= 1
        assert state.body_state.interoceptive_state is not None
        # The first update has no previous body state to measure a delta
        # against, so the perturbation and the stability are unmeasured
        assert state.body_delta is None
        assert state.stability is None
        second = ps.update(drives)
        assert isinstance(second.body_delta, float)
        assert isinstance(second.stability, float)

    @staticmethod
    def _attention_drives(level: float = 0.7, valence: float = 0.0):
        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {
                "current_level": level,
                "optimal_level": 0.7,
                "deviation": abs(level - 0.7),
                "urgency": 0.0,
            },
        }
        drives.get_overall_valence.return_value = valence
        drives.get_free_energy.return_value = 0.1
        return drives

    def test_first_update_reports_no_delta(self):
        """The first update has no previous body state. Under the default
        first_delta ("unmeasured") body_delta, stability and arousal are None
        and nothing enters the delta history; under "v04_zero" they read
        0.0, 1.0 and 0.0 as in v0.4.0 and the 0.0 enters the history."""
        from mtc.consciousness.damasio.protoself import Protoself

        drives = self._attention_drives()

        fresh = Protoself()
        state = fresh.update(drives)
        assert state.body_delta is None
        assert state.stability is None
        assert state.primordial_feelings["arousal"] is None
        assert list(fresh._body_delta_history) == []
        assert list(fresh._stability_history) == []
        assert state.to_dict()["body_delta"] is None
        assert state.to_dict()["stability"] is None
        stats = fresh.get_statistics()
        assert stats["average_body_delta"] is None
        assert stats["average_stability"] is None

        legacy = Protoself(first_delta="v04_zero")
        old = legacy.update(drives)
        assert old.body_delta == 0.0
        assert old.stability == 1.0
        assert old.primordial_feelings["arousal"] == 0.0
        assert list(legacy._body_delta_history) == [0.0]
        assert list(legacy._stability_history) == [1.0]

        with pytest.raises(ValueError, match="first_delta"):
            Protoself(first_delta="other")

        # The second update is a measurement in both modes
        for ps in (fresh, legacy):
            second = ps.update(drives)
            assert isinstance(second.body_delta, float)
            assert isinstance(second.stability, float)
            assert isinstance(second.primordial_feelings["arousal"], float)

    def test_body_delta_detects_perturbation(self):
        """A second update after a large change in the drives has a larger
        body_delta, a lower stability and a higher arousal than a second
        update after no change: two measured deltas, neither of them the
        unmeasured first update."""
        from mtc.consciousness.damasio.protoself import Protoself

        drives_stable = MagicMock()
        drives_stable.get_drive_state.return_value = {
            "attention_budget": {
                "current_level": 0.7,
                "optimal_level": 0.7,
                "deviation": 0.0,
                "urgency": 0.0,
            },
            "curiosity": {
                "current_level": 0.5,
                "optimal_level": 0.5,
                "deviation": 0.0,
                "urgency": 0.0,
            },
            "social_connection": {
                "current_level": 0.6,
                "optimal_level": 0.6,
                "deviation": 0.0,
                "urgency": 0.0,
            },
            "coherence": {
                "current_level": 0.8,
                "optimal_level": 0.8,
                "deviation": 0.0,
                "urgency": 0.0,
            },
            "safety": {
                "current_level": 0.9,
                "optimal_level": 0.9,
                "deviation": 0.0,
                "urgency": 0.0,
            },
        }
        drives_stable.get_overall_valence.return_value = 0.0
        drives_stable.get_free_energy.return_value = 0.0

        # Second update -- big change in drives
        drives_perturbed = MagicMock()
        drives_perturbed.get_drive_state.return_value = {
            "attention_budget": {
                "current_level": 0.1,
                "optimal_level": 0.7,
                "deviation": 0.6,
                "urgency": 0.9,
            },
            "curiosity": {
                "current_level": 0.9,
                "optimal_level": 0.5,
                "deviation": 0.4,
                "urgency": 0.7,
            },
            "social_connection": {
                "current_level": 0.1,
                "optimal_level": 0.6,
                "deviation": 0.5,
                "urgency": 0.8,
            },
            "coherence": {
                "current_level": 0.3,
                "optimal_level": 0.8,
                "deviation": 0.5,
                "urgency": 0.8,
            },
            "safety": {
                "current_level": 0.4,
                "optimal_level": 0.9,
                "deviation": 0.5,
                "urgency": 0.8,
            },
        }
        drives_perturbed.get_overall_valence.return_value = -0.5
        drives_perturbed.get_free_energy.return_value = 0.8

        steady = Protoself()
        assert steady.update(drives_stable).body_delta is None  # no baseline yet
        calm = steady.update(drives_stable)

        shaken = Protoself()
        shaken.update(drives_stable)
        perturbed = shaken.update(drives_perturbed)

        assert isinstance(calm.body_delta, float)
        assert isinstance(perturbed.body_delta, float)
        assert perturbed.body_delta > calm.body_delta
        assert perturbed.stability < calm.stability
        assert (
            perturbed.primordial_feelings["arousal"]
            > calm.primordial_feelings["arousal"]
        )

    def test_primordial_feelings_from_trends(self):
        """Primordial feelings reflect body state trends over time."""
        from mtc.consciousness.damasio.protoself import Protoself

        ps = Protoself()

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {
                "current_level": 0.7,
                "optimal_level": 0.7,
                "deviation": 0.0,
                "urgency": 0.0,
            },
            "curiosity": {
                "current_level": 0.5,
                "optimal_level": 0.5,
                "deviation": 0.0,
                "urgency": 0.0,
            },
            "social_connection": {
                "current_level": 0.6,
                "optimal_level": 0.6,
                "deviation": 0.0,
                "urgency": 0.0,
            },
            "coherence": {
                "current_level": 0.8,
                "optimal_level": 0.8,
                "deviation": 0.0,
                "urgency": 0.0,
            },
            "safety": {
                "current_level": 0.9,
                "optimal_level": 0.9,
                "deviation": 0.0,
                "urgency": 0.0,
            },
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
            "attention_budget": {
                "current_level": 0.7,
                "optimal_level": 0.7,
                "deviation": 0.0,
                "urgency": 0.0,
            },
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
            "attention_budget": {
                "current_level": 0.7,
                "optimal_level": 0.7,
                "deviation": 0.0,
                "urgency": 0.0,
            },
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
            "attention_budget": {
                "current_level": 0.7,
                "optimal_level": 0.7,
                "deviation": 0.0,
                "urgency": 0.0,
            },
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
            "attention_budget": {
                "current_level": 0.7,
                "optimal_level": 0.7,
                "deviation": 0.0,
                "urgency": 0.0,
            },
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
            "attention_budget": {
                "current_level": 0.7,
                "optimal_level": 0.7,
                "deviation": 0.0,
                "urgency": 0.0,
            },
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
            "attention_budget": {
                "current_level": 0.7,
                "optimal_level": 0.7,
                "deviation": 0.0,
                "urgency": 0.0,
            },
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
            body_state=BodyState(
                metabolic_state=0.7,
                energy_level=0.8,
                pain_signals=0.1,
                interoceptive_state={},
                circadian_phase=0.3,
                temperature=0.4,
            ),
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
            body_state=BodyState(
                metabolic_state=0.7,
                energy_level=0.8,
                pain_signals=0.1,
                interoceptive_state={"curiosity": {"level": 0.8, "urgency": 0.3}},
                circadian_phase=0.3,
                temperature=0.4,
            ),
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
            body_state=BodyState(),
            body_delta=0.01,
            stability=0.95,
            primordial_feelings={"pleasure_pain": 0.0, "vitality": 0.5, "arousal": 0.1},
        )
        exp_low = cc.bind(low_state, workspace_winners=[winner])

        # High perturbation
        cc2 = CoreConsciousness()
        high_state = ProtoSelfState(
            body_state=BodyState(),
            body_delta=0.5,
            stability=0.3,
            primordial_feelings={
                "pleasure_pain": -0.3,
                "vitality": 0.3,
                "arousal": 0.8,
            },
        )
        exp_high = cc2.bind(high_state, workspace_winners=[winner])

        assert exp_high.self_world_binding > exp_low.self_world_binding

    def test_no_winners_minimal_experience(self):
        """With no workspace winners, core experience is minimal."""
        from mtc.consciousness.damasio.core_consciousness import CoreConsciousness
        from mtc.consciousness.damasio.protoself import ProtoSelfState, BodyState

        cc = CoreConsciousness()

        state = ProtoSelfState(
            body_state=BodyState(),
            body_delta=0.1,
            stability=0.7,
            primordial_feelings={"pleasure_pain": 0.0, "vitality": 0.5, "arousal": 0.2},
        )

        experience = cc.bind(state, workspace_winners=[])
        assert experience.self_world_binding < 0.3
        assert len(experience.somatic_markers) == 0

    def test_unmeasured_arousal_leaves_intensity_and_here_now_unmeasured(self):
        """A protoself state whose arousal is unmeasured (None, the update
        with no previous body state) yields somatic markers with a measured
        valence and no intensity, and no here-now salience, with or without
        winners; the dicts carry the None. A measured arousal of 0.0 (the
        legacy first update) yields the numbers 0.0 and 0.6 x salience."""
        from mtc.consciousness.damasio.core_consciousness import (
            CoreConsciousness,
            CoreExperience,
        )
        from mtc.consciousness.damasio.protoself import ProtoSelfState, BodyState

        def state(arousal):
            return ProtoSelfState(
                body_state=BodyState(),
                body_delta=None if arousal is None else 0.0,
                stability=None if arousal is None else 1.0,
                primordial_feelings={
                    "pleasure_pain": 0.3,
                    "vitality": 0.5,
                    "arousal": arousal,
                },
            )

        w1 = MagicMock()
        w1.candidate = MagicMock()
        w1.candidate.summary = "A synthetic thought"
        w1.candidate.content_type = "thought"
        w1.salience = 0.8
        w2 = MagicMock()
        w2.candidate = MagicMock()
        w2.candidate.summary = "A synthetic warning"
        w2.candidate.content_type = "warning"
        w2.salience = 0.4

        cc = CoreConsciousness()
        unmeasured = cc.bind(state(None), workspace_winners=[w1, w2])
        assert len(unmeasured.somatic_markers) == 2
        for marker in unmeasured.somatic_markers:
            assert marker.intensity is None
            assert isinstance(marker.valence, float)
            assert marker.to_dict()["intensity"] is None
        assert isinstance(unmeasured.somatic_valence, float)
        assert unmeasured.here_now_salience is None
        raw = unmeasured.to_dict()
        assert raw["here_now_salience"] is None
        again = CoreExperience.from_dict(raw)
        assert again.here_now_salience is None
        assert again.somatic_markers[1].intensity is None
        assert again.somatic_valence == pytest.approx(unmeasured.somatic_valence)

        quiet = cc.bind(state(None), workspace_winners=[])
        assert quiet.here_now_salience is None
        assert quiet.somatic_markers == []

        legacy = cc.bind(state(0.0), workspace_winners=[w1, w2])
        assert [m.intensity for m in legacy.somatic_markers] == [0.0, 0.0]
        assert legacy.here_now_salience == pytest.approx(0.6 * 0.6)
        assert cc.bind(state(0.0), workspace_winners=[]).here_now_salience == 0.0

        measured = cc.bind(state(0.5), workspace_winners=[w1, w2])
        assert [m.intensity for m in measured.somatic_markers] == pytest.approx(
            [0.4, 0.2]
        )
        assert measured.here_now_salience == pytest.approx(0.5 * 0.4 + 0.6 * 0.6)

    def test_self_model_enriches_binding(self):
        """Providing self_model increases feeling_of_knowing."""
        from mtc.consciousness.damasio.core_consciousness import CoreConsciousness
        from mtc.consciousness.damasio.protoself import ProtoSelfState, BodyState

        cc = CoreConsciousness()

        state = ProtoSelfState(
            body_state=BodyState(energy_level=0.8),
            body_delta=0.2,
            stability=0.6,
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
            body_state=BodyState(),
            body_delta=0.1,
            stability=0.7,
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
            body_state=BodyState(),
            body_delta=0.1,
            stability=0.7,
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
        from mtc.consciousness.damasio.extended_consciousness import (
            ExtendedConsciousness,
        )

        ec = ExtendedConsciousness()
        assert ec.cycle_count == 0

    def test_extend_creates_extended_state(self):
        """extend() produces ExtendedState from core experience."""
        from mtc.consciousness.damasio.extended_consciousness import (
            ExtendedConsciousness,
        )
        from mtc.consciousness.damasio.core_consciousness import CoreExperience

        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.7,
            perturbation_source="thinking about philosophy",
            feeling_of_knowing=0.6,
            somatic_markers=[],
            here_now_salience=0.5,
        )

        state = ec.extend(core)

        # A first cycle with no history and no markers measures none of the
        # autobiographic quantities
        assert state.autobiographic_self is not None
        assert state.autobiographic_self.identity_continuity is None
        assert state.autobiographic_self.narrative_coherence is None
        assert state.episodic_resonance is None
        assert isinstance(state.narrative_context, str)

        # With a history and identity markers, a second cycle measures all
        # three within their ranges
        history = [
            {"role": "user", "content": "a synthetic question about patterns"},
            {"role": "assistant", "content": "a synthetic answer about patterns"},
        ]
        markers = {"interests": ["patterns"]}
        ec.extend(core, conversation_history=history, identity_markers=markers)
        later = ec.extend(core, conversation_history=history, identity_markers=markers)
        assert 0 <= later.autobiographic_self.identity_continuity <= 1
        assert 0 <= later.autobiographic_self.narrative_coherence <= 1
        assert 0 <= later.autobiographic_self.temporal_horizon <= 1
        assert 0 <= later.episodic_resonance <= 1

    def test_conversation_history_enriches(self):
        """Episodic resonance is unmeasured without a history and a positive
        measurement with one."""
        from mtc.consciousness.damasio.extended_consciousness import (
            ExtendedConsciousness,
        )
        from mtc.consciousness.damasio.core_consciousness import CoreExperience

        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.7,
            perturbation_source="philosophy discussion",
            feeling_of_knowing=0.6,
            somatic_markers=[],
            here_now_salience=0.5,
        )

        # Without history
        state1 = ec.extend(core)
        assert state1.episodic_resonance is None
        assert state1.relevant_episodes == []

        # With history
        ec2 = ExtendedConsciousness()
        history = [
            {"role": "user", "content": "Tell me about consciousness"},
            {"role": "assistant", "content": "I find this topic fascinating..."},
            {"role": "user", "content": "What does it feel like to think?"},
        ]
        state2 = ec2.extend(core, conversation_history=history)

        assert isinstance(state2.episodic_resonance, float)
        assert state2.episodic_resonance > 0.0
        assert state2.episodic_resonance == pytest.approx(
            min(1.0, len(history) / 10.0) * 0.6 + 0.7 * 0.4
        )

    def test_extended_without_history_reports_none(self):
        """With no conversation history and no identity markers the extended
        layer has no measured coherence, horizon, resonance or continuity on
        its first cycles; once the binding trend holds more than three points
        continuity is measured from that trend alone."""
        from mtc.consciousness.damasio.extended_consciousness import (
            AutobiographicSelf,
            ExtendedConsciousness,
            ExtendedState,
        )
        from mtc.consciousness.damasio.core_consciousness import CoreExperience

        ec = ExtendedConsciousness()
        core = CoreExperience(
            self_world_binding=0.6,
            perturbation_source="internal body state change",
            feeling_of_knowing=0.5,
            somatic_markers=[],
            here_now_salience=0.2,
        )

        states = [ec.extend(core) for _ in range(3)]
        for state in states:
            auto = state.autobiographic_self
            assert auto.narrative_coherence is None
            assert auto.temporal_horizon is None
            assert auto.identity_continuity is None
            assert state.episodic_resonance is None
            assert "identity" not in state.narrative_context
            assert state.to_dict()["autobiographic_self"]["identity_continuity"] is None

        fourth = ec.extend(core)
        auto = fourth.autobiographic_self
        assert auto.identity_continuity == pytest.approx(0.3)  # a flat trend
        assert auto.narrative_coherence is None
        assert auto.temporal_horizon is None
        assert fourth.episodic_resonance is None

        stats = ec.get_statistics()
        assert stats["average_coherence"] is None
        assert stats["average_continuity"] == pytest.approx(0.3)

        # Round trip keeps the unmeasured values unmeasured
        restored = ExtendedState.from_dict(states[0].to_dict())
        assert restored.episodic_resonance is None
        assert restored.autobiographic_self.identity_continuity is None
        assert AutobiographicSelf.from_dict({}).narrative_coherence is None
        assert AutobiographicSelf.from_dict({}).temporal_horizon is None

    def test_identity_markers_alone_measure_continuity_and_horizon(self):
        """Identity markers measure continuity and horizon on the first
        cycle, while coherence needs an earlier cycle's themes."""
        from mtc.consciousness.damasio.extended_consciousness import (
            ExtendedConsciousness,
        )
        from mtc.consciousness.damasio.core_consciousness import CoreExperience

        ec = ExtendedConsciousness()
        core = CoreExperience(
            self_world_binding=0.6,
            perturbation_source="self-reflection",
            feeling_of_knowing=0.5,
            somatic_markers=[],
            here_now_salience=0.4,
        )
        markers = {"interests": ["patterns", "data"], "core_values": ["care"]}

        first = ec.extend(core, identity_markers=markers).autobiographic_self
        assert first.identity_continuity == pytest.approx(0.2)
        assert first.temporal_horizon == pytest.approx(0.2)
        assert first.narrative_coherence is None

        second = ec.extend(core, identity_markers=markers).autobiographic_self
        assert second.narrative_coherence == pytest.approx(1.0)

    def test_identity_markers_build_continuity(self):
        """Identity markers increase identity_continuity."""
        from mtc.consciousness.damasio.extended_consciousness import (
            ExtendedConsciousness,
        )
        from mtc.consciousness.damasio.core_consciousness import CoreExperience

        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.6,
            perturbation_source="self-reflection",
            feeling_of_knowing=0.5,
            somatic_markers=[],
            here_now_salience=0.4,
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
        from mtc.consciousness.damasio.extended_consciousness import (
            ExtendedConsciousness,
        )
        from mtc.consciousness.damasio.core_consciousness import CoreExperience

        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.8,
            perturbation_source="deep conversation",
            feeling_of_knowing=0.7,
            somatic_markers=[],
            here_now_salience=0.6,
        )

        state = ec.extend(
            core,
            conversation_history=[
                {"role": "user", "content": "How are you feeling?"},
            ],
        )

        assert len(state.narrative_context) > 0
        assert isinstance(state.narrative_context, str)

    def test_growth_trajectory(self):
        """growth_trajectory reflects development over multiple cycles."""
        from mtc.consciousness.damasio.extended_consciousness import (
            ExtendedConsciousness,
        )
        from mtc.consciousness.damasio.core_consciousness import CoreExperience

        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.6,
            perturbation_source="learning",
            feeling_of_knowing=0.5,
            somatic_markers=[],
            here_now_salience=0.5,
        )

        # Multiple updates to build trajectory
        for _ in range(5):
            state = ec.extend(core)

        assert state.autobiographic_self.growth_trajectory in (
            "developing",
            "stable",
            "transforming",
        )

    def test_future_projections(self):
        """Future projections are generated based on current state."""
        from mtc.consciousness.damasio.extended_consciousness import (
            ExtendedConsciousness,
        )
        from mtc.consciousness.damasio.core_consciousness import CoreExperience

        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.7,
            perturbation_source="exploring curiosity",
            feeling_of_knowing=0.6,
            somatic_markers=[],
            here_now_salience=0.5,
        )

        state = ec.extend(core)
        assert isinstance(state.future_projections, list)

    def test_to_dict(self):
        """ExtendedState serializes to dict."""
        from mtc.consciousness.damasio.extended_consciousness import (
            ExtendedConsciousness,
        )
        from mtc.consciousness.damasio.core_consciousness import CoreExperience

        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.5,
            perturbation_source="test",
            feeling_of_knowing=0.4,
            somatic_markers=[],
            here_now_salience=0.3,
        )
        state = ec.extend(core)
        d = state.to_dict()
        assert isinstance(d, dict)
        assert "autobiographic_self" in d
        assert "narrative_context" in d

    def test_statistics(self):
        """get_statistics() returns extended consciousness stats."""
        from mtc.consciousness.damasio.extended_consciousness import (
            ExtendedConsciousness,
        )
        from mtc.consciousness.damasio.core_consciousness import CoreExperience

        ec = ExtendedConsciousness()

        core = CoreExperience(
            self_world_binding=0.5,
            perturbation_source="test",
            feeling_of_knowing=0.4,
            somatic_markers=[],
            here_now_salience=0.3,
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
            "attention_budget": {
                "current_level": 0.7,
                "optimal_level": 0.7,
                "deviation": 0.0,
                "urgency": 0.0,
            },
            "curiosity": {
                "current_level": 0.6,
                "optimal_level": 0.5,
                "deviation": 0.1,
                "urgency": 0.2,
            },
            "social_connection": {
                "current_level": 0.5,
                "optimal_level": 0.6,
                "deviation": 0.1,
                "urgency": 0.2,
            },
            "coherence": {
                "current_level": 0.8,
                "optimal_level": 0.8,
                "deviation": 0.0,
                "urgency": 0.0,
            },
            "safety": {
                "current_level": 0.9,
                "optimal_level": 0.9,
                "deviation": 0.0,
                "urgency": 0.0,
            },
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
    async def test_no_markers_means_no_somatic_valence(self):
        """A cycle with no workspace winners has no somatic marker, so the
        core layer reports no somatic valence rather than a measured 0.0."""
        from mtc.consciousness.damasio import DamasioLayers

        dl = DamasioLayers()
        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {
                "current_level": 0.6,
                "optimal_level": 0.7,
                "deviation": 0.1,
                "urgency": 0.2,
            },
        }
        drives.get_overall_valence.return_value = 0.1
        drives.get_free_energy.return_value = 0.2

        result = await dl.process(homeostatic_drives=drives, workspace_winners=[])
        assert result["core"]["somatic_markers"] == []
        assert result["core"]["somatic_valence"] is None

    @pytest.mark.asyncio
    async def test_first_layers_cycle_reports_no_stability(self):
        """On the first cycle the protoself has no previous body state, so
        the aggregate reports protoself_stability and feeling_of_knowing as
        None and the context and report say so; the legacy first_delta
        records the v0.4.0 constants."""
        from mtc.consciousness.damasio import DamasioLayers

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {
                "current_level": 0.6,
                "optimal_level": 0.7,
                "deviation": 0.1,
                "urgency": 0.2,
            },
        }
        drives.get_overall_valence.return_value = 0.1
        drives.get_free_energy.return_value = 0.2

        dl = DamasioLayers()
        first = await dl.process(homeostatic_drives=drives, workspace_winners=[])
        assert first["protoself_stability"] is None
        assert first["feeling_of_knowing"] is None
        assert first["protoself"]["body_delta"] is None
        assert first["protoself"]["stability"] is None
        assert first["protoself"]["primordial_feelings"]["arousal"] is None
        assert first["core"]["feeling_of_knowing"] is None
        assert isinstance(first["self_world_binding"], float)
        context = dl.generate_context()
        assert "stability=insufficient_data" in context
        assert "knowing=insufficient_data" in context
        report = dl.generate_report()
        assert "Stability: insufficient_data" in report
        assert "Body delta: insufficient_data" in report
        assert "Feeling of knowing: insufficient_data" in report

        second = await dl.process(homeostatic_drives=drives, workspace_winners=[])
        assert isinstance(second["protoself_stability"], float)
        assert isinstance(second["feeling_of_knowing"], float)

        legacy = DamasioLayers(first_delta="v04_zero")
        old = await legacy.process(homeostatic_drives=drives, workspace_winners=[])
        assert old["protoself_stability"] == 1.0
        assert old["protoself"]["body_delta"] == 0.0
        assert old["protoself"]["primordial_feelings"]["arousal"] == 0.0
        assert old["feeling_of_knowing"] == pytest.approx(0.1)  # 1.0 x 0.1
        assert legacy.protoself.first_delta == "v04_zero"

    @pytest.mark.asyncio
    async def test_first_layers_cycle_with_winners_has_no_marker_intensity(self):
        """On the first cycle the arousal is unmeasured, so the somatic
        markers carry a valence but no intensity and the here-now salience
        is None; the context still names the dominant marker (by valence)
        and the report says insufficient_data. The second cycle measures
        both; the legacy first_delta records the v0.4.0 numbers."""
        from mtc.consciousness.damasio import DamasioLayers

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {
                "current_level": 0.6,
                "optimal_level": 0.7,
                "deviation": 0.1,
                "urgency": 0.2,
            },
        }
        drives.get_overall_valence.return_value = 0.3
        drives.get_free_energy.return_value = 0.2

        def winners():
            w1 = MagicMock()
            w1.candidate = MagicMock()
            w1.candidate.summary = "A synthetic discovery"
            w1.candidate.content_type = "discovery"
            w1.salience = 0.9
            w2 = MagicMock()
            w2.candidate = MagicMock()
            w2.candidate.summary = "A synthetic error"
            w2.candidate.content_type = "error"
            w2.salience = 0.5
            return [w1, w2]

        dl = DamasioLayers()
        first = await dl.process(homeostatic_drives=drives, workspace_winners=winners())
        markers = first["core"]["somatic_markers"]
        assert len(markers) == 2
        assert [m["intensity"] for m in markers] == [None, None]
        assert all(isinstance(m["valence"], float) for m in markers)
        assert isinstance(first["core"]["somatic_valence"], float)
        assert first["core"]["here_now_salience"] is None
        context = dl.generate_context()
        assert "[Somatic: approach (A synthetic discovery)]" in context
        report = dl.generate_report()
        assert "Here-now salience: insufficient_data" in report
        assert "Somatic markers: 2" in report

        second = await dl.process(
            homeostatic_drives=drives, workspace_winners=winners()
        )
        assert all(
            isinstance(m["intensity"], float) for m in second["core"]["somatic_markers"]
        )
        assert isinstance(second["core"]["here_now_salience"], float)
        assert "Here-now salience: insufficient_data" not in dl.generate_report()

        legacy = DamasioLayers(first_delta="v04_zero")
        old = await legacy.process(
            homeostatic_drives=drives, workspace_winners=winners()
        )
        assert [m["intensity"] for m in old["core"]["somatic_markers"]] == [0.0, 0.0]
        assert old["core"]["here_now_salience"] == pytest.approx(0.6 * 0.7)

    @pytest.mark.asyncio
    async def test_first_cycle_markers_feed_the_emotion_extractor(self):
        """The first default cycle's somatic markers are consumable by the
        post-turn emotion step: the extractor's raw read neither raises nor
        drops the measured valence, and it reports the somatic intensity as
        unmeasured (None) because no first-cycle marker carries one. The
        legacy first delta feeds it the v0.4.0 zero."""
        from types import SimpleNamespace

        from mtc.consciousness.damasio import DamasioLayers
        from mtc.consciousness.emotion_extractor import EmotionExtractor

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {
                "current_level": 0.6,
                "optimal_level": 0.7,
                "deviation": 0.1,
                "urgency": 0.2,
            },
        }
        drives.get_overall_valence.return_value = 0.3
        drives.get_free_energy.return_value = 0.2

        def winners():
            out = []
            for summary, kind, salience in (
                ("A synthetic note", "thought", 0.8),
                ("A synthetic warning", "warning", 0.4),
            ):
                w = MagicMock()
                w.candidate = MagicMock()
                w.candidate.summary = summary
                w.candidate.content_type = kind
                w.salience = salience
                out.append(w)
            return out

        async def first_cycle(first_delta):
            dl = DamasioLayers(first_delta=first_delta)
            result = await dl.process(
                homeostatic_drives=drives, workspace_winners=winners()
            )
            assert len(result["core"]["somatic_markers"]) == 2
            return SimpleNamespace(
                damasio_state=result,
                protoself_stability=result["protoself_stability"],
                epistemic_depth=None,
            )

        state = await first_cycle("unmeasured")
        raw = EmotionExtractor()._extract_raw(state)
        assert isinstance(raw["somatic_valence_avg"], float)
        assert raw["somatic_intensity_avg"] is None
        emotion = await EmotionExtractor().extract(state, session_id="first")
        assert isinstance(emotion.label, str) and emotion.label

        legacy = await first_cycle("v04_zero")
        assert EmotionExtractor()._extract_raw(legacy)["somatic_intensity_avg"] == 0.0

    @pytest.mark.asyncio
    async def test_layers_without_history_report_no_continuity(self):
        """The aggregate's autobiographic_continuity is None while the
        extended layer has nothing to measure it from, and the report says
        so instead of printing a number."""
        from mtc.consciousness.damasio import DamasioLayers

        dl = DamasioLayers()
        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {
                "current_level": 0.6,
                "optimal_level": 0.7,
                "deviation": 0.1,
                "urgency": 0.2,
            },
        }
        drives.get_overall_valence.return_value = 0.1
        drives.get_free_energy.return_value = 0.2

        result = await dl.process(homeostatic_drives=drives, workspace_winners=[])
        assert result["autobiographic_continuity"] is None
        extended = result["extended"]
        assert extended["episodic_resonance"] is None
        assert extended["autobiographic_self"]["narrative_coherence"] is None
        assert extended["autobiographic_self"]["temporal_horizon"] is None

        report = dl.generate_report()
        assert "Identity continuity: insufficient_data" in report
        assert "Narrative coherence: insufficient_data" in report
        assert "Episodic resonance: insufficient_data" in report
        assert isinstance(dl.generate_context(), str)

        winner = MagicMock()
        winner.candidate = MagicMock()
        winner.candidate.summary = "synthetic discovery"
        winner.candidate.content_type = "discovery"
        winner.salience = 0.8
        marked = await dl.process(homeostatic_drives=drives, workspace_winners=[winner])
        assert len(marked["core"]["somatic_markers"]) == 1
        assert isinstance(marked["core"]["somatic_valence"], float)
        assert marked["core"]["somatic_valence"] == pytest.approx(
            marked["core"]["somatic_markers"][0]["valence"]
        )

    @pytest.mark.asyncio
    async def test_generate_context(self):
        """generate_context() produces readable string for LLM brain."""
        from mtc.consciousness.damasio import DamasioLayers

        dl = DamasioLayers()

        drives = MagicMock()
        drives.get_drive_state.return_value = {
            "attention_budget": {
                "current_level": 0.7,
                "optimal_level": 0.7,
                "deviation": 0.0,
                "urgency": 0.0,
            },
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
            "attention_budget": {
                "current_level": 0.7,
                "optimal_level": 0.7,
                "deviation": 0.0,
                "urgency": 0.0,
            },
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
            "attention_budget": {
                "current_level": 0.7,
                "optimal_level": 0.7,
                "deviation": 0.0,
                "urgency": 0.0,
            },
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
        assert hasattr(state, "damasio_state")
        assert hasattr(state, "protoself_stability")
        assert hasattr(state, "self_world_binding")
        assert hasattr(state, "feeling_of_knowing")
        assert hasattr(state, "autobiographic_continuity")
        assert hasattr(state, "narrative_context")

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

    # The former DCM scorer test was removed with mtc.consciousness.dcm_scoring
    # (v0.2.0). DCM coverage lives in tests/assessment/test_dcm_benchmark.py.


# ============================================================================
# NARRATION GATES
# ============================================================================


def _gate_drives():
    drives = MagicMock()
    drives.get_drive_state.return_value = {
        "attention_budget": {
            "current_level": 0.8,
            "optimal_level": 0.7,
            "deviation": 0.1,
            "urgency": 0.1,
        },
    }
    drives.get_overall_valence.return_value = 0.3
    drives.get_free_energy.return_value = 0.1
    return drives


def _gate_winner():
    winner = MagicMock()
    winner.candidate = MagicMock()
    winner.candidate.summary = "synthetic discovery"
    winner.candidate.content_type = "discovery"
    winner.salience = 0.8
    return winner


class TestDamasioNarrationGates:
    """The three narration kwargs drop context segments, not computation."""

    async def _run(self, **kwargs):
        from mtc.consciousness.damasio import DamasioLayers

        dl = DamasioLayers(**kwargs)
        result = None
        for _ in range(3):
            result = await dl.process(
                homeostatic_drives=_gate_drives(),
                workspace_winners=[_gate_winner()],
                conversation_history=[
                    {"role": "user", "content": "a synthetic message for themes"}
                ],
            )
        return dl, result

    @pytest.mark.asyncio
    async def test_all_segments_on_by_default(self):
        dl, _ = await self._run()
        context = dl.generate_context()
        assert "[Body:" in context
        assert "[Present moment:" in context
        assert "[Somatic:" in context
        assert "[Narrative:" in context

    @pytest.mark.asyncio
    async def test_each_gate_drops_only_its_segment(self):
        body_off, _ = await self._run(narrate_body_drives=False)
        assert "[Body:" not in body_off.generate_context()
        assert "[Somatic:" in body_off.generate_context()

        somatic_off, _ = await self._run(narrate_somatic=False)
        assert "[Somatic:" not in somatic_off.generate_context()
        assert "[Narrative:" in somatic_off.generate_context()

        narrative_off, _ = await self._run(narrate_narrative_identity=False)
        assert "[Narrative:" not in narrative_off.generate_context()
        assert "[Body:" in narrative_off.generate_context()

    @pytest.mark.asyncio
    async def test_all_gates_off_leave_only_present_moment(self):
        dl, _ = await self._run(
            narrate_body_drives=False,
            narrate_somatic=False,
            narrate_narrative_identity=False,
        )
        context = dl.generate_context()
        assert context.startswith("[Present moment:")
        assert "[Body:" not in context
        assert "[Somatic:" not in context
        assert "[Narrative:" not in context

    @pytest.mark.asyncio
    async def test_empty_context_before_any_cycle_when_gated(self):
        from mtc.consciousness.damasio import DamasioLayers

        dl = DamasioLayers(
            narrate_body_drives=False,
            narrate_somatic=False,
            narrate_narrative_identity=False,
        )
        assert dl.generate_context() == ""

    @pytest.mark.asyncio
    async def test_gates_leave_computation_unchanged(self):
        _, on = await self._run()
        _, off = await self._run(
            narrate_body_drives=False,
            narrate_somatic=False,
            narrate_narrative_identity=False,
        )
        for key in (
            "protoself_stability",
            "self_world_binding",
            "feeling_of_knowing",
            "autobiographic_continuity",
            "narrative_context",
        ):
            if isinstance(on[key], float):
                assert off[key] == pytest.approx(on[key])
            else:
                assert off[key] == on[key]

    @pytest.mark.asyncio
    async def test_agent_name_in_narrative(self):
        _, result = await self._run()
        assert result["narrative_context"].startswith("The system is")
        assert " she " not in f" {result['narrative_context']} "

        _, named = await self._run(agent_name="the agent")
        assert named["narrative_context"].startswith("The agent is")
