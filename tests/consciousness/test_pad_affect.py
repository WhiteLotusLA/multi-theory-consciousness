"""Tests for PAD Affect Model."""

import numpy as np
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
    """Affect coherence under the v0.4.0 variance method."""

    def test_coherence_needs_history(self):
        """Score should be 0.3 (warming up) with insufficient history."""
        model = PADAffectModel(coherence_method="v04_variance")
        model.compute(PADSources())
        score = model.compute_coherence()
        assert abs(score - 0.3) < 0.01

    def test_coherence_with_varied_history(self):
        """Varied affect history should produce higher variance score."""
        model = PADAffectModel(coherence_method="v04_variance")
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
        model = PADAffectModel(coherence_method="v04_variance")
        for _ in range(20):
            model.compute(PADSources())  # same neutral every time
        score = model.compute_coherence()
        assert score < 0.7  # flat affect = low variance component


def _uniform_sources(rng: np.random.Generator) -> PADSources:
    """Every PADSources field drawn over its valid range, as the runner does."""
    return PADSources(
        somatic_valence=rng.uniform(-1.0, 1.0),
        homeostatic_satisfaction=rng.uniform(-1.0, 1.0),
        snn_valence=rng.uniform(-1.0, 1.0),
        protoself_arousal=rng.uniform(0.0, 1.0),
        snn_spike_rate=rng.uniform(0.0, 1.0),
        ignition_rate=rng.uniform(0.0, 1.0),
        prediction_error=rng.uniform(0.0, 1.0),
        agency_score=rng.uniform(0.0, 1.0),
        voluntary_attention_ratio=rng.uniform(0.0, 1.0),
        epistemic_depth_normalized=rng.uniform(0.0, 1.0),
    )


def _sinusoid_sources(step: int, rng: np.random.Generator) -> PADSources:
    """Sources on slow 40-cycle sinusoids with a little noise."""
    phase = 2.0 * np.pi * step / 40.0

    def wave(offset: float, centre: float, amplitude: float) -> float:
        noise = rng.normal(0.0, 0.02)
        return centre + amplitude * np.sin(phase + offset) + noise

    return PADSources(
        somatic_valence=wave(0.0, 0.0, 0.8),
        homeostatic_satisfaction=wave(0.3, 0.0, 0.8),
        snn_valence=wave(0.6, 0.0, 0.8),
        protoself_arousal=wave(0.0, 0.5, 0.4),
        snn_spike_rate=wave(0.4, 0.5, 0.4),
        ignition_rate=wave(0.8, 0.5, 0.4),
        prediction_error=wave(1.2, 0.5, 0.4),
        agency_score=wave(0.0, 0.5, 0.4),
        voluntary_attention_ratio=wave(0.5, 0.5, 0.4),
        epistemic_depth_normalized=wave(1.0, 0.5, 0.4),
    )


class TestTemporalStructureCoherence:
    """The default coherence scores persistence over time, not variance."""

    def test_random_pad_histories_score_below_half(self):
        for seed in range(50):
            rng = np.random.default_rng(seed)
            model = PADAffectModel()
            for _ in range(30):
                model.compute(_uniform_sources(rng))
            score = model.compute_coherence()
            assert score is not None
            assert score < 0.5, f"seed {seed}: noise scored {score:.3f}"

    def test_structured_pad_history_scores_above_half(self):
        rng = np.random.default_rng(7)
        model = PADAffectModel()
        for step in range(30):
            model.compute(_sinusoid_sources(step, rng))
        score = model.compute_coherence()
        assert score is not None
        assert score > 0.5

    def test_flat_pad_history_scores_zero(self):
        model = PADAffectModel()
        for _ in range(25):
            model.compute(PADSources(somatic_valence=0.2, agency_score=0.6))
        assert model.compute_coherence() == 0.0

    def test_alternating_pad_history_scores_low(self):
        model = PADAffectModel()
        high = PADSources(
            somatic_valence=0.8,
            protoself_arousal=0.9,
            snn_spike_rate=0.9,
            agency_score=0.9,
            voluntary_attention_ratio=0.9,
        )
        low = PADSources(
            somatic_valence=-0.8,
            protoself_arousal=0.1,
            snn_spike_rate=0.1,
            agency_score=0.1,
            voluntary_attention_ratio=0.1,
        )
        for step in range(30):
            model.compute(high if step % 2 == 0 else low)
        score = model.compute_coherence()
        assert score is not None
        assert score < 0.2

    def test_coherence_none_until_min_history(self):
        model = PADAffectModel(coherence_min_history=10)
        rng = np.random.default_rng(1)
        for step in range(9):
            model.compute(_sinusoid_sources(step, rng))
            assert model.compute_coherence() is None
        model.compute(_sinusoid_sources(9, rng))
        assert model.compute_coherence() is not None

    def test_v04_variance_method_reproduces_old_values(self):
        model = PADAffectModel(coherence_method="v04_variance")
        values = [
            (0.1, 0.2, 0.3),
            (0.4, 0.6, 0.5),
            (-0.2, 0.3, 0.7),
            (0.0, 0.5, 0.2),
        ]
        for p, a, d in values:
            model._history.append(PADState(pleasure=p, arousal=a, dominance=d))
        assert model.compute_coherence() == 0.3  # below five states
        model._history.append(PADState(pleasure=0.3, arousal=0.4, dominance=0.6))
        recent = list(model._history)
        stds = [
            np.std([s.pleasure for s in recent]),
            np.std([s.arousal for s in recent]),
            np.std([s.dominance for s in recent]),
        ]
        variance_score = min(1.0, float(np.mean(stds)) * 5.0)
        dim_active = sum(1 for s in stds if s > 0.01) / 3.0
        expected = 0.5 * variance_score + 0.5 * dim_active
        assert model.compute_coherence() == expected

    def test_invalid_coherence_method_names_the_field(self):
        try:
            PADAffectModel(coherence_method="bogus")
        except ValueError as exc:
            assert "coherence_method" in str(exc)
        else:
            raise AssertionError("expected ValueError")

    def test_history_length_property(self):
        model = PADAffectModel()
        model.compute(PADSources())
        model.compute(PADSources())
        assert model.history_length == 2


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

    def test_restore_keeps_window_and_types(self):
        model = PADAffectModel(history_window=4)
        for i in range(9):
            model.compute(PADSources(somatic_valence=i * 0.1))
        restored = PADAffectModel(history_window=4)
        restored.from_state_dict(model.to_state_dict())
        assert restored._history.maxlen == 4
        assert all(isinstance(s, PADState) for s in restored._history)
        assert restored.generate_context() == model.generate_context()


def test_pad_context_gate():
    """narrate_affect_line=False silences the line, not the computation."""
    on = PADAffectModel()
    off = PADAffectModel(narrate_affect_line=False)
    sources = PADSources(somatic_valence=0.6, protoself_arousal=0.8)
    state_on = on.compute(sources)
    state_off = off.compute(sources)

    assert on.generate_context().startswith("Current affect:")
    assert off.generate_context() == ""
    assert state_off.pleasure == state_on.pleasure
    assert state_off.label == state_on.label
    assert off.get_latest() is state_off
