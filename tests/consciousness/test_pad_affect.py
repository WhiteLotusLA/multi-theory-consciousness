"""Tests for PAD Affect Model."""

import dataclasses

import numpy as np
import pytest

from mtc.consciousness.pad_affect import (
    AROUSAL_WEIGHTS,
    DOMINANCE_WEIGHTS,
    NEUTRAL_PLEASURE_BAND,
    PLEASURE_WEIGHTS,
    PADAffectModel,
    PADSources,
    PADState,
    get_affect_label,
    measured_dimensions,
)

ALL_INPUTS = tuple(field.name for field in dataclasses.fields(PADSources))


def _old_formula(sources):
    """The arithmetic of compute() before measured inputs existed."""
    pleasure = float(
        np.clip(
            0.4 * sources.somatic_valence
            + 0.3 * sources.homeostatic_satisfaction
            + 0.3 * sources.snn_valence,
            -1.0,
            1.0,
        )
    )
    arousal = float(
        np.clip(
            0.3 * sources.protoself_arousal
            + 0.2 * sources.snn_spike_rate
            + 0.25 * sources.ignition_rate
            + 0.25 * sources.prediction_error,
            0.0,
            1.0,
        )
    )
    dominance = float(
        np.clip(
            0.4 * sources.agency_score
            + 0.3 * sources.voluntary_attention_ratio
            + 0.3 * sources.epistemic_depth_normalized,
            0.0,
            1.0,
        )
    )
    return pleasure, arousal, dominance


class TestPADComputation:
    """Test PAD dimension computation."""

    def test_default_sources_give_zero_pleasure_and_the_neutral_label(self):
        """Default sources give P 0.0, A 0.09, D 0.35 and the Neutral label."""
        model = PADAffectModel()
        state = model.compute(PADSources())
        assert isinstance(state, PADState)
        assert state.pleasure == 0.0
        assert state.arousal == pytest.approx(0.09)
        assert state.dominance == pytest.approx(0.35)
        assert state.label == "Neutral"

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
        """Out-of-range arousal inputs are clipped to [0, 1]."""
        model = PADAffectModel()
        names = tuple(AROUSAL_WEIGHTS)
        high = model.compute(PADSources(**{name: 3.0 for name in names}))
        low = model.compute(PADSources(**{name: -3.0 for name in names}))
        assert high.arousal == 1.0
        assert low.arousal == 0.0

    def test_dominance_clamped(self):
        """Out-of-range dominance inputs are clipped to [0, 1]."""
        model = PADAffectModel()
        names = tuple(DOMINANCE_WEIGHTS)
        high = model.compute(PADSources(**{name: 3.0 for name in names}))
        low = model.compute(PADSources(**{name: -3.0 for name in names}))
        assert high.dominance == 1.0
        assert low.dominance == 0.0

    def test_pleasure_clamped(self):
        """Out-of-range pleasure inputs are clipped to [-1, 1]."""
        model = PADAffectModel()
        names = tuple(PLEASURE_WEIGHTS)
        high = model.compute(PADSources(**{name: 3.0 for name in names}))
        low = model.compute(PADSources(**{name: -3.0 for name in names}))
        assert high.pleasure == 1.0
        assert low.pleasure == -1.0

    def test_clip_applies_to_measured_inputs_too(self):
        """The renormalised path clips as the unlisted path does."""
        model = PADAffectModel()
        high = model.compute(
            PADSources(**{name: 3.0 for name in ALL_INPUTS}), measured=ALL_INPUTS
        )
        low = model.compute(
            PADSources(**{name: -3.0 for name in ALL_INPUTS}), measured=ALL_INPUTS
        )
        assert (high.pleasure, high.arousal, high.dominance) == (1.0, 1.0, 1.0)
        assert (low.pleasure, low.arousal, low.dominance) == (-1.0, 0.0, 0.0)

    def test_weight_tables_cover_every_source_once(self):
        """The three tables name every PADSources field exactly once."""
        tables = (PLEASURE_WEIGHTS, AROUSAL_WEIGHTS, DOMINANCE_WEIGHTS)
        names = [name for table in tables for name in table]
        assert sorted(names) == sorted(ALL_INPUTS)
        assert len(names) == len(set(names))
        for table in tables:
            assert sum(table.values()) == pytest.approx(1.0)

    def test_unlisted_call_keeps_v05_arithmetic(self):
        """compute(sources) reproduces the pre-change formula bit for bit."""
        rng = np.random.default_rng(11)
        model = PADAffectModel()
        for _ in range(200):
            sources = _uniform_sources(rng)
            state = model.compute(sources)
            assert (state.pleasure, state.arousal, state.dominance) == _old_formula(
                sources
            )
            assert state.label == get_affect_label(
                state.pleasure, state.arousal, state.dominance
            )
            assert state.measured_inputs is None
        assert model.history_length == 100
        assert model._computation_count == 200

    def test_fully_measured_call_matches_the_unlisted_arithmetic(self):
        """With every input measured the renormalised mean is the old formula."""
        rng = np.random.default_rng(12)
        model = PADAffectModel()
        for _ in range(100):
            sources = _uniform_sources(rng)
            state = model.compute(sources, measured=ALL_INPUTS)
            assert (state.pleasure, state.arousal, state.dominance) == _old_formula(
                sources
            )

    def test_measured_sources_are_renormalised(self):
        """Each dimension is the weighted mean over its measured inputs only."""
        model = PADAffectModel()
        sources = PADSources(
            somatic_valence=0.5,
            homeostatic_satisfaction=0.5,
            snn_valence=-0.9,
            protoself_arousal=0.6,
            snn_spike_rate=0.9,
            ignition_rate=0.4,
            prediction_error=0.2,
            agency_score=0.7,
            voluntary_attention_ratio=0.5,
            epistemic_depth_normalized=0.2,
        )
        measured = tuple(
            name for name in ALL_INPUTS if name not in ("snn_valence", "snn_spike_rate")
        )
        state = model.compute(sources, measured=measured)
        assert state is not None
        # snn_valence is absent: pleasure is 0.5, not 0.35 (or 0.08 with -0.9).
        assert state.pleasure == pytest.approx(0.5)
        expected_arousal = (0.3 * 0.6 + 0.25 * 0.4 + 0.25 * 0.2) / 0.8
        assert state.arousal == pytest.approx(expected_arousal)
        assert state.arousal == pytest.approx(0.4125)
        assert state.dominance == pytest.approx(0.4 * 0.7 + 0.3 * 0.5 + 0.3 * 0.2)
        assert state.label == get_affect_label(
            state.pleasure, state.arousal, state.dominance
        )
        unlisted = PADAffectModel().compute(sources)
        assert unlisted.pleasure == pytest.approx(0.08)

    def test_dimension_without_measured_source_yields_no_state(self):
        """A dimension with no measured input gives None and records nothing."""
        model = PADAffectModel()
        assert model.compute(PADSources(), measured=("ignition_rate",)) is None
        assert model.compute(PADSources(), measured=()) is None
        no_dominance = ("somatic_valence", "protoself_arousal")
        assert model.compute(PADSources(), measured=no_dominance) is None
        assert model.history_length == 0
        assert model.get_latest() is None
        assert model.to_state_dict()["scalars"]["computation_count"] == 0

    def test_compute_records_measured_inputs(self):
        """The measured list is stored, round-trips and is validated."""
        model = PADAffectModel()
        measured = ("somatic_valence", "ignition_rate", "agency_score")
        state = model.compute(PADSources(somatic_valence=0.4), measured=measured)
        assert state.measured_inputs == measured
        assert isinstance(state.measured_inputs, tuple)
        listed = model.compute(PADSources(), measured=list(measured))
        assert listed.measured_inputs == measured
        model.compute(PADSources())

        restored = PADAffectModel()
        restored.from_state_dict(model.to_state_dict())
        history = list(restored._history)
        assert [s.measured_inputs for s in history] == [measured, measured, None]
        assert [s.label for s in history] == [s.label for s in model._history]

        older = model.to_state_dict()
        for entry in older["histories"]["pad_history"]:
            entry.pop("measured", None)
        legacy = PADAffectModel()
        legacy.from_state_dict(older)
        assert all(s.measured_inputs is None for s in legacy._history)

        before = model.history_length
        with pytest.raises(ValueError, match="no_such_input"):
            model.compute(PADSources(), measured=("somatic_valence", "no_such_input"))
        assert model.history_length == before

    def test_history_accumulates(self):
        """Each compute() call adds to history."""
        model = PADAffectModel()
        assert len(model._history) == 0
        model.compute(PADSources())
        assert len(model._history) == 1
        model.compute(PADSources(somatic_valence=0.5))
        assert len(model._history) == 2


class TestMeasuredDimensions:
    """measured_dimensions names the dimensions a state has measured inputs for."""

    def test_empty_for_none_and_for_unknown_provenance(self):
        assert measured_dimensions(None) == frozenset()
        unknown = PADAffectModel().compute(PADSources())
        assert unknown.measured_inputs is None
        assert measured_dimensions(unknown) == frozenset()
        assert measured_dimensions(PADState(0.3, 0.6, 0.5)) == frozenset()

    def test_names_the_dimensions_with_a_measured_input(self):
        state = PADAffectModel().compute(
            PADSources(),
            measured=("homeostatic_satisfaction", "ignition_rate", "agency_score"),
        )
        assert measured_dimensions(state) == frozenset(
            {"pleasure", "arousal", "dominance"}
        )
        partial = PADState(
            pleasure=0.2,
            arousal=0.4,
            dominance=0.5,
            measured_inputs=("snn_valence", "voluntary_attention_ratio"),
        )
        assert measured_dimensions(partial) == frozenset({"pleasure", "dominance"})
        assert isinstance(measured_dimensions(partial), frozenset)


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

    def test_zero_and_near_zero_pleasure_is_neutral(self):
        """Pleasure inside the band reads Neutral whatever A and D are."""
        assert NEUTRAL_PLEASURE_BAND == 0.1
        assert get_affect_label(0.0, 0.5, 0.5) == "Neutral"
        assert get_affect_label(-0.05, 0.1, 0.3) == "Neutral"
        assert get_affect_label(0.05, 0.9, 0.9) == "Neutral"
        assert get_affect_label(-0.099, 0.2, 0.2) == "Neutral"
        assert get_affect_label(-0.5, 0.2, 0.2) in ("Sad", "Helpless")
        assert get_affect_label(0.15, 0.2, 0.2) in ("Calm", "Peaceful")
        assert get_affect_label(-0.15, 0.2, 0.2) in ("Sad", "Helpless")


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
