"""Emotion dynamics: inertia, mood, acute spikes, blends, momentum."""

import json

import pytest

from mtc.consciousness.emotion_dynamics import (
    DynamicEmotionState,
    EmotionDynamics,
    EmotionDynamicsConfig,
)
from mtc.consciousness.pad_affect import PADState

T0 = 946684800.0


def pad(p=0.0, a=0.5, d=0.5):
    return PADState(pleasure=p, arousal=a, dominance=d)


def test_inertia_smooths_jumps():
    dyn = EmotionDynamics()
    dyn.update(pad(0.0, 0.5), now=T0)
    state = dyn.update(pad(1.0, 0.5), now=T0 + 1)
    # inertia 0.6: the smoothed value moves 40 percent of the way
    assert state.pleasure == pytest.approx(0.4)
    state = dyn.update(pad(1.0, 0.5), now=T0 + 2)
    assert state.pleasure == pytest.approx(0.6 * 0.4 + 0.4 * 1.0)


def test_first_update_starts_at_the_observation_without_deltas():
    state = EmotionDynamics().update(pad(0.8, 0.9, 0.2), "joyful", now=T0)
    assert state.pleasure == pytest.approx(0.8)
    assert state.mood_valence == pytest.approx(0.8)
    assert state.valence_momentum is None
    assert state.emotional_volatility is None
    assert state.acute_emotion is None
    assert state.dominant == "joyful"


def test_mood_moves_slowly():
    dyn = EmotionDynamics()
    dyn.update(pad(0.0, 0.5), now=T0)
    state = None
    for i in range(3):
        state = dyn.update(pad(0.9, 0.5), now=T0 + i + 1)
    assert state.mood_valence < 0.15
    assert state.pleasure > state.mood_valence
    assert state.mood == "alert" or state.mood == "neutral"


def test_mood_label_follows_configured_words():
    cfg = EmotionDynamicsConfig(
        mood_words={
            "positive_high": "P+",
            "positive_low": "P-",
            "negative_high": "N+",
            "negative_low": "N-",
            "neutral_high": "Z+",
            "neutral_low": "Z-",
        }
    )
    dyn = EmotionDynamics(cfg)
    assert dyn.update(pad(0.9, 0.9), now=T0).mood == "P+"
    dyn = EmotionDynamics(cfg)
    assert dyn.update(pad(-0.9, 0.1), now=T0).mood == "N-"


def test_acute_spike_decays_thirty_percent_per_cycle():
    dyn = EmotionDynamics()
    dyn.update(pad(0.0, 0.5), "calm", now=T0)
    spike = dyn.update(pad(0.6, 0.5), "excited", now=T0 + 1)
    assert spike.acute_emotion == "excited"
    assert spike.acute_intensity == pytest.approx(1.0)
    after1 = dyn.update(pad(0.6, 0.5), "excited", now=T0 + 2)
    assert after1.acute_intensity == pytest.approx(0.7)
    after2 = dyn.update(pad(0.6, 0.5), "excited", now=T0 + 3)
    assert after2.acute_intensity == pytest.approx(0.49)
    assert after2.acute_decay_remaining == pytest.approx(0.49)


def test_acute_override_and_expiry():
    dyn = EmotionDynamics()
    dyn.update(pad(0.0, 0.5), "calm", now=T0)
    dyn.update(pad(0.5, 0.5), "excited", now=T0 + 1)
    state = dyn.update(pad(0.5, 0.5), "calm", now=T0 + 2)
    assert state.dominant == "excited"  # strong acute overrides
    for i in range(12):
        state = dyn.update(pad(0.5, 0.5), "calm", now=T0 + 3 + i)
    assert state.acute_emotion is None
    assert state.dominant == "calm"


def test_blend_detected_for_configured_pair():
    dyn = EmotionDynamics()
    state = dyn.update(pad(), "joyful", "sad", confidence=0.5, now=T0)
    assert state.blend_name == "bittersweet"
    assert state.blend_ratio == pytest.approx(0.5)
    # Order does not matter, and high confidence suppresses the blend.
    dyn = EmotionDynamics()
    assert dyn.update(pad(), "sad", "joyful", confidence=0.5, now=T0).blend_name
    dyn = EmotionDynamics()
    assert (
        dyn.update(pad(), "joyful", "sad", confidence=0.95, now=T0).blend_name is None
    )


def test_blend_without_confidence_has_no_ratio():
    state = EmotionDynamics().update(pad(), "joyful", "sad", now=T0)
    assert state.blend_name == "bittersweet"
    assert state.blend_ratio is None


def test_unknown_labels_never_blend():
    dyn = EmotionDynamics()
    state = dyn.update(pad(), "zorp", "blip", confidence=0.3, now=T0)
    assert state.blend_name is None
    cfg = EmotionDynamicsConfig(blend_pairs={frozenset({"zorp", "blip"}): "mix"})
    state = EmotionDynamics(cfg).update(pad(), "zorp", "blip", confidence=0.3, now=T0)
    assert state.blend_name == "mix"


def test_missing_label_means_no_label_logic():
    dyn = EmotionDynamics()
    dyn.update(pad(0.0), now=T0)
    state = dyn.update(pad(0.9), now=T0 + 1)
    assert state.dominant is None
    assert state.acute_emotion is None
    assert state.blend_name is None


def test_accepts_mtc_pad_ranges():
    dyn = EmotionDynamics()
    state = dyn.update(pad(-1.0, 0.0, 0.0), now=T0)
    assert state.pleasure == -1.0 and state.arousal == 0.0
    state = dyn.update(pad(1.0, 1.0, 1.0), now=T0 + 1)
    assert -1.0 <= state.pleasure <= 1.0
    assert 0.0 <= state.arousal <= 1.0 and 0.0 <= state.dominance <= 1.0
    # Out-of-range input is clamped at the boundary.
    clamped = EmotionDynamics().update(pad(3.0, 2.0, -1.0), now=T0)
    assert (clamped.pleasure, clamped.arousal, clamped.dominance) == (1.0, 1.0, 0.0)


def test_non_finite_pad_is_rejected():
    with pytest.raises(ValueError):
        EmotionDynamics().update(pad(float("nan")), now=T0)


def test_momentum_and_volatility_after_second_cycle():
    dyn = EmotionDynamics()
    dyn.update(pad(0.0, 0.5), now=T0)
    state = dyn.update(pad(1.0, 0.5), now=T0 + 1)
    assert state.valence_momentum == pytest.approx(0.4)
    assert state.arousal_momentum == pytest.approx(0.0)
    assert state.emotional_volatility == pytest.approx(1.0)


def test_payload_has_no_subtitle_unless_configured():
    dyn = EmotionDynamics()
    state = dyn.update(pad(), "calm", now=T0)
    assert isinstance(state, DynamicEmotionState)
    assert "subtitle" not in dyn.to_payload()
    dyn = EmotionDynamics(EmotionDynamicsConfig(subtitles={"calm": "settled"}))
    dyn.update(pad(), "calm", now=T0)
    assert dyn.to_payload()["subtitle"] == "settled"
    dyn.update(pad(), "excited", now=T0 + 1)
    assert "subtitle" not in dyn.to_payload()


def test_payload_before_any_update_is_empty():
    assert EmotionDynamics().to_payload() == {}


def test_state_dict_round_trip():
    dyn = EmotionDynamics()
    dyn.update(pad(0.0, 0.5), "calm", now=T0)
    dyn.update(pad(0.7, 0.8), "excited", now=T0 + 1)
    state = json.loads(json.dumps(dyn.to_state_dict()))
    fresh = EmotionDynamics()
    fresh.from_state_dict(state)
    a = dyn.update(pad(0.2, 0.4), "excited", now=T0 + 2)
    b = fresh.update(pad(0.2, 0.4), "excited", now=T0 + 2)
    assert a == b
