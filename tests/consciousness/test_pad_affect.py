"""Tests for PAD Affect Model."""

import numpy as np
import pytest
from mtc.consciousness.pad_affect import (
    PADAffectModel,
    PADSources,
    PADState,
    get_affect_label,
)


class TestPADComputation:
    """Test PAD dimension computation."""

    def test_neutral_sources_produce_neutral_pad(self):
        """Default sources should produce near-neutral affect."""
        model = PADAffectModel()
        sources = PADSources()  # all defaults
        state = model.compute(sources)
        assert isinstance(state, PADState)
        # Default homeostatic_satisfaction=0.0, snn_valence=0.0, somatic_valence=0.0
        assert -0.1 <= state.pleasure <= 0.1
        assert 0.0 <= state.arousal <= 1.0
        assert 0.0 <= state.dominance <= 1.0

    def test_maximum_pleasure(self):
        """All pleasure sources maxed should give P close to 1.0."""
        model = PADAffectModel()
        sources = PADSources(
            somatic_valence=1.0,
            homeostatic_satisfaction=1.0,
            snn_valence=1.0,
        )
        state = model.compute(sources)
        assert state.pleasure > 0.9

    def test_minimum_pleasure(self):
        """All pleasure sources minimized should give P close to -1.0."""
        model = PADAffectModel()
        sources = PADSources(
            somatic_valence=-1.0,
            homeostatic_satisfaction=-1.0,
            snn_valence=-1.0,
        )
        state = model.compute(sources)
        assert state.pleasure < -0.9

    def test_arousal_clamped(self):
        """Arousal should be clamped to [0, 1]."""
        model = PADAffectModel()
        sources = PADSources(
            protoself_arousal=1.0,
            snn_spike_rate=1.0,
            ignition_rate=1.0,
            prediction_error=1.0,
        )
        state = model.compute(sources)
        assert state.arousal <= 1.0
        assert state.arousal >= 0.0

    def test_dominance_clamped(self):
        """Dominance should be clamped to [0, 1]."""
        model = PADAffectModel()
        sources = PADSources(
            agency_score=1.0,
            voluntary_attention_ratio=1.0,
            epistemic_depth_normalized=1.0,
        )
        state = model.compute(sources)
        assert state.dominance <= 1.0
        assert state.dominance >= 0.0

    def test_history_accumulates(self):
        """Each compute() call adds to history."""
        model = PADAffectModel()
        assert len(model._history) == 0
        model.compute(PADSources())
        assert len(model._history) == 1
        model.compute(PADSources(somatic_valence=0.5))
        assert len(model._history) == 2


class TestAffectLabels:
    """Test affect label mapping."""

    def test_positive_high_arousal_high_dominance(self):
        label = get_affect_label(0.5, 0.8, 0.8)
        assert "exhilarated" in label.lower() or "confident" in label.lower()

    def test_negative_low_arousal_low_dominance(self):
        label = get_affect_label(-0.5, 0.2, 0.2)
        assert "sad" in label.lower() or "helpless" in label.lower()

    def test_positive_low_arousal_high_dominance(self):
        label = get_affect_label(0.5, 0.2, 0.8)
        assert "relaxed" in label.lower() or "content" in label.lower()

    def test_neutral_pleasure_gets_label(self):
        """Even near-zero pleasure should produce a label, not crash."""
        label = get_affect_label(0.01, 0.5, 0.5)
        assert isinstance(label, str) and len(label) > 0


class TestAffectCoherence:
    """Test affect coherence scoring."""

    def test_coherence_needs_history(self):
        """Score should be 0.3 (warming up) with insufficient history."""
        model = PADAffectModel()
        model.compute(PADSources())
        score = model.compute_coherence()
        assert abs(score - 0.3) < 0.01

    def test_coherence_with_varied_history(self):
        """Varied affect history should produce higher variance score."""
        model = PADAffectModel()
        for i in range(20):
            sources = PADSources(
                somatic_valence=np.sin(i * 0.5),
                protoself_arousal=0.3 + 0.3 * np.cos(i * 0.3),
            )
            model.compute(sources)
        score = model.compute_coherence()
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # should be above warming-up baseline

    def test_coherence_flat_affect_scores_low(self):
        """Constant flat affect should score lower than varied."""
        model = PADAffectModel()
        for _ in range(20):
            model.compute(PADSources())  # same neutral every time
        score = model.compute_coherence()
        assert score < 0.7  # flat affect = low variance component


class TestPersistence:
    """Test state persistence."""

    def test_round_trip(self):
        model = PADAffectModel()
        for i in range(5):
            model.compute(PADSources(somatic_valence=i * 0.1))
        state = model.to_state_dict()
        model2 = PADAffectModel()
        model2.from_state_dict(state)
        assert len(model2._history) == len(model._history)
        assert model2._computation_count == model._computation_count
